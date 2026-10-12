"""Core logic for PV adversarial detection (cyber-physical security).
Simulated single-diode PV system (P = V x I) -> fault injection -> LSTM autoencoder detector
-> adversarial attacks (FGSM, PGD, physics-projected PGD) -> physics-consistency check -> hybrid detector.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

L = 24                                   # window length (time steps)
FEATS = ["G (W/m2)", "T cell (C)", "V (V)", "I (A)", "P (W)"]
KINDS = ["shading", "drift", "inverter"]
FAULT_LABEL = {"shading": "Partial shading", "drift": "Sensor drift (voltage sensor)", "inverter": "Inverter fault"}

# ---------------------------------------------------------------- single-diode PV model (Rs = 0)
ISC, VOC, NS, NID = 8.2, 37.0, 60, 1.3
def a_of(T):  return NID * NS * 0.02585 * (T + 273.15) / 298.15
def i0_of(T): return ISC / (np.exp(VOC / a_of(25.0)) - 1) * np.exp(0.065 * (T - 25))
def iph(G, T): return ISC * (G / 1000) * (1 + 0.0005 * (T - 25))
def i_curve(V, G, T):
    return np.maximum(iph(G, T) - i0_of(T) * (np.exp(np.minimum(V / a_of(T), 50)) - 1), 0)
def mpp(G, T):
    Vg = np.linspace(0, VOC, 300)[None, :]
    I = i_curve(Vg, G[:, None], T[:, None]); P = Vg * I
    k = P.argmax(1)
    return Vg[0][k], I[np.arange(len(k)), k]

def pv_from_weather(G, T, rng):
    """Irradiance + cell temperature -> V, I, P with P = V x I (plus small sensor noise)."""
    V, I = np.zeros(len(G)), np.zeros(len(G))
    for s in range(0, len(G), 4000):
        V[s:s+4000], I[s:s+4000] = mpp(G[s:s+4000], T[s:s+4000])
    V = V * (1 + .005 * rng.standard_normal(len(G))); I = I * (1 + .005 * rng.standard_normal(len(G)))
    P = V * I * (1 + .003 * rng.standard_normal(len(G)))
    return np.stack([G, T, V, I, P], 1).astype(np.float32)

# ---------------------------------------------------------------- data sources
def synthetic_windows(n_days, rng):
    """Normal windows from synthetic sunny/cloudy days (works with no dataset). Returns (windows, day_id)."""
    W, gid = [], []
    h = np.arange(32) / 32 * 8 + 8
    sun = np.sin(np.pi * (h - 6) / 12)
    for d in range(n_days):
        c = np.clip(rng.uniform(.5, 1) + .08 * rng.standard_normal(32), .2, 1.1)
        G = 1000 * sun * np.convolve(np.pad(c, 1, mode="edge"), [.25, .5, .25], mode="valid")
        T = 20 + 8 * sun + 0.03 * G + .3 * rng.standard_normal(32)
        X = pv_from_weather(G, T, rng)
        for i in range(0, 32 - L + 1, 2): W.append(X[i:i+L]); gid.append(d)
    return np.stack(W), np.array(gid)

def csv_windows(path, rng, stride=6):
    """Normal windows built from the REAL irradiance/temperature of the user's dataset. Returns (windows, site_id)."""
    d = pd.read_csv(path, usecols=["instantaneous_global_radiation", "clearsky_Tamb", "radiation_mean", "radiation_std",
                                   "tprt_mean", "tprt_std", "label", "site_num", "date", "source_file"])
    d = d[d.label == 0]
    G = np.clip(d.instantaneous_global_radiation * d.radiation_std + d.radiation_mean, 0, None).to_numpy()
    T = (d.clearsky_Tamb * d.tprt_std + d.tprt_mean).to_numpy() + 0.03 * G
    X = pv_from_weather(G, T, rng)
    keys = d.groupby(["source_file", "site_num", "date"], sort=False).ngroup().to_numpy()
    site = d.site_num.to_numpy(); W, gid = [], []
    cut = np.flatnonzero(np.diff(keys)) + 1
    for xs, st_ in zip(np.split(X, cut), np.split(site, cut)):
        for i in range(0, len(xs) - L + 1, stride):
            if xs[i:i+L, 0].mean() >= 150: W.append(xs[i:i+L]); gid.append(st_[0])
    return np.stack(W), np.array(gid)

# ---------------------------------------------------------------- faults
def inject_window(w, kind, rng):
    """Inject one fault into a window. Returns (faulty window, boolean mask of faulty steps)."""
    X = w.copy(); s = int(rng.integers(3, 8)); e = min(L, s + int(rng.integers(10, 15)))
    if kind == "shading":      # current + voltage drop, power stays consistent
        X[s:e, 3] *= rng.uniform(.4, .7); X[s:e, 2] *= rng.uniform(.9, .97); X[s:e, 4] = X[s:e, 2] * X[s:e, 3]
    elif kind == "drift":      # voltage sensor drifts upward, power reading not updated
        X[s:e, 2] *= 1 + np.linspace(0, rng.uniform(.12, .25), e - s)
    else:                      # inverter fault: current collapses
        X[s:e, 3] *= rng.uniform(0, .2); X[s:e, 4] = X[s:e, 2] * X[s:e, 3]
    mask = np.zeros(L, bool); mask[s:e] = True
    return X, mask

# ---------------------------------------------------------------- scaling and physics rules
def make_scaler(Xfit):
    rows = Xfit.reshape(-1, 5); span = rows.max(0) - rows.min(0)
    return rows.min(0) - .05 * span, rows.max(0) + .05 * span
def to_norm(x, lo, hi): return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)
def to_raw(xn, lo, hi): return xn * (hi - lo) + lo

TOL = 0.15          # allowed distance from the single-diode I-V curve
P_LIMIT = 0.02      # allowed |P - V*I| relative error
def project(xn, lo, hi):
    """Physics projection: keep I near the I-V curve for (V, G, T), then enforce P = V x I."""
    r = to_raw(xn, lo, hi)
    G, T, V = np.maximum(r[..., 0], 0), r[..., 1], np.clip(r[..., 2], 0, VOC)
    Ic = i_curve(V, G, T)
    I = np.clip(r[..., 3], Ic * (1 - TOL), Ic * (1 + TOL))
    return to_norm(np.stack([G, T, V, I, V * I], -1), lo, hi)

def residuals(xn, lo, hi):
    """Per-time-step physics violations: power residual |P-VI|/P and I-V-curve residual."""
    r = to_raw(xn, lo, hi); G, T, V, I, P = [r[..., k] for k in range(5)]
    Ic = i_curve(V, G, T)
    return np.abs(P - V * I) / (np.abs(P) + 1), np.abs(I - Ic) / (Ic + .1)

def physical_ok(xn, lo, hi, slack=0.05):
    rp, ri = residuals(xn, lo, hi)
    return ((rp < P_LIMIT) & (ri < TOL + slack)).all(1)

# ---------------------------------------------------------------- model
class AE(nn.Module):
    def __init__(s, f=5, h=32):
        super().__init__()
        s.enc = nn.LSTM(f, h, batch_first=True); s.dec = nn.LSTM(h, h, batch_first=True); s.out = nn.Linear(h, f)
    def forward(s, x):
        _, (h, _) = s.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        return s.out(s.dec(z)[0])

def run_attack(model, xn, lo, hi, eps, kind):
    steps = 1 if kind == "FGSM" else 20
    alpha = eps if steps == 1 else eps / 5
    x0 = torch.as_tensor(np.asarray(xn, dtype=np.float32)); x = x0.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        loss = ((model(x) - x) ** 2).mean((1, 2)).sum()
        g, = torch.autograd.grad(loss, x)
        x = x.detach() - alpha * g.sign()                         # make reconstruction error SMALL
        x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0, 1)
        if kind == "PGD+physics": x = torch.as_tensor(project(x.numpy(), lo, hi))
    return x.detach().numpy()

class Detector:
    NAMES = ["LSTM autoencoder (AI)", "Physics check", "Hybrid (AI + physics)"]
    def __init__(self, model, lo, hi, thr, n_train=0):
        self.model, self.lo, self.hi, self.thr, self.n_train = model, np.asarray(lo), np.asarray(hi), float(thr), n_train
    def to_norm(self, x): return to_norm(x, self.lo, self.hi)
    def to_raw(self, xn): return to_raw(xn, self.lo, self.hi)
    def error(self, xn):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(xn, dtype=np.float32)); return ((self.model(x) - x) ** 2).mean((1, 2)).numpy()
    def ai_flag(self, xn):   return self.error(xn) > self.thr
    def phys_flag(self, xn): return ~physical_ok(xn, self.lo, self.hi)
    def flags(self, xn):
        a, p = self.ai_flag(xn), self.phys_flag(xn)
        return {self.NAMES[0]: a, self.NAMES[1]: p, self.NAMES[2]: a | p}
    def attack(self, xn, eps, kind): return run_attack(self.model, xn, self.lo, self.hi, eps, kind)

def get_detector(source="synthetic", train_csv=None, test_csv=None, epochs=20, cache_dir="pv_models"):
    """Train (first time) or load the LSTM autoencoder. Returns (Detector, pool of normal raw windows)."""
    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    if source == "synthetic":
        Xall, gid = synthetic_windows(300, rng); pool, _ = synthetic_windows(60, np.random.default_rng(1)); key = "synthetic"
    else:
        Xall, gid = csv_windows(train_csv, rng); pool, _ = csv_windows(test_csv, np.random.default_rng(1))
        key = f"dataset_{os.path.getsize(train_csv)}_{os.path.getsize(test_csv)}"
    path = os.path.join(cache_dir, key + ".pt")
    groups = np.unique(gid); val_g = groups[-max(1, int(round(.15 * len(groups)))):]
    isval = np.isin(gid, val_g); Xfit, Xval = Xall[~isval], Xall[isval]
    model = AE()
    if os.path.exists(path):
        ck = torch.load(path, weights_only=True)
        model.load_state_dict(ck["state"]); model.eval()
        return Detector(model, ck["lo"], ck["hi"], ck["thr"], ck["n_train"]), pool
    lo, hi = make_scaler(Xfit)
    Xn = torch.as_tensor(to_norm(Xfit, lo, hi)); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for b in torch.randperm(len(Xn)).split(64):
            loss = ((model(Xn[b]) - Xn[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    det = Detector(model, lo, hi, 1.0, len(Xfit))
    det.thr = float(np.percentile(det.error(to_norm(Xval, lo, hi)), 99))
    torch.save({"state": model.state_dict(), "lo": [float(v) for v in lo], "hi": [float(v) for v in hi],
                "thr": det.thr, "n_train": int(len(Xfit))}, path)
    return det, pool

# ---------------------------------------------------------------- batch evaluation
def f1_from(rate, fpr, npos, nneg):
    tp = rate * npos; fn = npos - tp; fp = fpr * nneg
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

def evaluate_batch(det, pool, fault, attack, eps, n=100, seed=0):
    """Many random scenarios: detection rate, false alarms and F1 before/after the attack."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(pool), size=2 * n)
    faulty = np.stack([inject_window(pool[i], fault or KINDS[int(rng.integers(3))], rng)[0] for i in idx[:n]])
    normal = pool[idx[n:]]
    fn_, nn_ = det.to_norm(faulty), det.to_norm(normal)
    att = det.attack(fn_, eps, attack) if attack else fn_
    fc, fa, fz = det.flags(fn_), det.flags(att), det.flags(nn_)
    rows = []
    for name in det.NAMES:
        rc, ra, fpr = fc[name].mean(), fa[name].mean(), fz[name].mean()
        rows.append({"Detector": name, "Detection (no attack)": rc, "Detection (under attack)": ra, "False alarms": fpr,
                     "F1 (no attack)": f1_from(rc, fpr, n, n), "F1 (under attack)": f1_from(ra, fpr, n, n)})
    extra = {"physically_valid": float(physical_ok(att, det.lo, det.hi).mean()),
             "avg_change": float(np.abs(att - fn_).max((1, 2)).mean())}
    return pd.DataFrame(rows).round(2), extra
