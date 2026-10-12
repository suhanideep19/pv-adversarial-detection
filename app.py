import numpy as np, pandas as pd, streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pvcore as pc

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

st.set_page_config(page_title="PV Adversarial Detection", layout="wide")
st.title("Physics-Aware vs Statistical Anomaly Detection in Solar PV Systems")
st.caption("Can an AI fault detector be fooled while the sensor data stays physically realistic?")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Data source")
source = st.sidebar.radio("Weather data for the PV simulator", ["Synthetic (works anywhere)", "My dataset (CSV)"])
train_csv = test_csv = None
if source.startswith("My"):
    train_csv = st.sidebar.text_input("Train CSV", "data/pv_train_merged.csv")
    test_csv = st.sidebar.text_input("Test CSV", "data/pv_test_merged.csv")

@st.cache_resource(show_spinner=False)
def load_detector(src, tr, te):
    return pc.get_detector(src, tr, te)

try:
    with st.spinner("Preparing the detector. The first run trains the LSTM autoencoder (about 1-2 min), later runs are instant."):
        det, pool = load_detector("synthetic" if source.startswith("Syn") else "dataset", train_csv, test_csv)
except Exception as ex:
    st.error(f"Could not prepare the detector: {ex}")
    st.stop()

st.sidebar.header("Scenario")
FAULTS = {"No fault (healthy)": "none", "Partial shading": "shading", "Sensor drift (V sensor)": "drift", "Inverter fault": "inverter"}
ATTACKS = {"No attack": None, "FGSM (conventional)": "FGSM", "PGD (conventional)": "PGD", "Physics-constrained PGD": "PGD+physics"}
fault_label = st.sidebar.selectbox("Inject fault", list(FAULTS))
attack_label = st.sidebar.selectbox("Attacker hides the fault with", list(ATTACKS))
eps = st.sidebar.slider("Attack strength (epsilon)", 0.01, 0.30, 0.10, 0.01)
if "seed" not in st.session_state: st.session_state.seed = 0
if st.sidebar.button("New random scenario"): st.session_state.seed += 1
fault, atk = FAULTS[fault_label], ATTACKS[attack_label]
st.sidebar.markdown("---")
st.sidebar.caption(f"Detector trained on {det.n_train} normal windows. Alarm threshold (reconstruction error) = {det.thr:.4f}")

tab1, tab2 = st.tabs(["Simulation", "Batch evaluation"])

# ------------------------------------------------------------------ simulation
with tab1:
    rng = np.random.default_rng(st.session_state.seed)
    healthy = pool[int(rng.integers(len(pool)))]
    mask = np.zeros(pc.L, bool); faulty = healthy.copy()
    if fault != "none": faulty, mask = pc.inject_window(healthy, fault, rng)
    versions = {"Healthy window": healthy}
    if fault != "none": versions["Faulty window"] = faulty
    if fault != "none" and atk:
        adv = det.attack(det.to_norm(faulty[None]), eps, atk)
        versions["Faulty + attack"] = det.to_raw(adv)[0]
    elif atk and fault == "none":
        st.info("Attacks hide a fault, so pick a fault in the sidebar to see an attack.")

    rows, norms = [], {}
    for name, w in versions.items():
        xn = det.to_norm(w[None]); norms[name] = xn
        e = float(det.error(xn)[0]); f = {k: bool(v[0]) for k, v in det.flags(xn).items()}
        v = lambda b: "Alert" if b else "Normal"
        rows.append({"Window": name, "AI error": round(e, 4), "AI detector": v(f[det.NAMES[0]]),
                     "Physics check": v(f[det.NAMES[1]]), "Hybrid": v(f[det.NAMES[2]])})
    st.subheader("Verdicts")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if fault != "none" and atk:
        r = rows[-1]
        fooled = lambda s: "fooled" if s == "Normal" else "still detected"
        st.markdown(f"Attack result: AI detector {fooled(r['AI detector'])}, physics check {fooled(r['Physics check'])}, hybrid {fooled(r['Hybrid'])}.")
        ok = pc.physical_ok(norms["Faulty + attack"], det.lo, det.hi)[0]
        st.markdown(f"The attacked data is {'physically consistent (P = V×I and on the I–V curve)' if ok else 'physically impossible'}.")

    st.subheader("Sensor signals")
    colors = {"Healthy window": "tab:green", "Faulty window": "black", "Faulty + attack": "tab:red"}
    fig, axs = plt.subplots(1, 5, figsize=(17, 3))
    for k, a in enumerate(axs):
        for name, w in versions.items():
            a.plot(w[:, k], color=colors[name], lw=2 if name == "Faulty window" else 1.4, label=name)
        if mask.any(): a.axvspan(np.flatnonzero(mask)[0], np.flatnonzero(mask)[-1], color="orange", alpha=.12)
        a.set_title(pc.FEATS[k], fontsize=10)
    axs[0].legend(fontsize=7); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("AI reconstruction error")
        fig, a = plt.subplots(figsize=(5, 3))
        errs = [float(det.error(norms[n])[0]) for n in versions]
        a.bar(list(versions), errs, color=[colors[n] for n in versions]); a.axhline(det.thr, color="k", ls="--", label="alarm threshold")
        a.set_ylabel("error"); a.legend(fontsize=8); plt.xticks(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Physics violation per time step")
        fig, a = plt.subplots(figsize=(5, 3))
        for n in versions:
            rp, ri = pc.residuals(norms[n], det.lo, det.hi)
            a.plot(np.maximum(rp[0] / pc.P_LIMIT, ri[0] / (pc.TOL + .05)), color=colors[n], label=n)
        a.axhline(1, color="k", ls="--", label="physics limit"); a.set_ylabel("violation (1 = limit)"); a.set_yscale("symlog", linthresh=1)
        a.legend(fontsize=7); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ batch evaluation
with tab2:
    st.write("Runs many random scenarios with the fault and attack chosen in the sidebar (no fault selected = a random mix of all three faults).")
    n = st.slider("Number of faulty and of normal windows", 30, 300, 100, 10)
    if st.button("Run batch evaluation"):
        with st.spinner("Running scenarios..."):
            df, extra = pc.evaluate_batch(det, pool, None if fault == "none" else fault, atk, eps, n=n, seed=st.session_state.seed)
        st.dataframe(df, hide_index=True)
        if atk:
            st.markdown(f"{attack_label} at ε = {eps:.2f}: {extra['physically_valid']*100:.0f}% of the attacked windows are physically valid; "
                        f"average largest change per window = {extra['avg_change']:.2f} (scaled units).")
        fig, a = plt.subplots(figsize=(7, 3.2)); x = np.arange(len(df)); w = .38
        a.bar(x - w/2, df["Detection (no attack)"], w, label="no attack"); a.bar(x + w/2, df["Detection (under attack)"], w, label="under attack")
        a.set_xticks(x); a.set_xticklabels(df["Detector"], fontsize=8); a.set_ylim(0, 1.2); a.set_ylabel("fault detection rate"); a.legend(ncol=2, loc="upper right", frameon=False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
