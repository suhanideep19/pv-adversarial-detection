# PV Adversarial Detection

[![CI](https://github.com/shreyaupadhyay868/pv-adversarial-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/shreyaupadhyay868/pv-adversarial-detection/actions/workflows/ci.yml)

Physics-aware vs statistical anomaly detection for solar PV systems.

An LSTM autoencoder trained on normal PV sensor data is a good fault detector until someone attacks it. This project builds a simulated PV system, injects faults, and measures how well three detectors hold up when an attacker perturbs the sensor readings to hide the fault: the autoencoder alone, a physics-consistency check, and a hybrid of the two. An interactive Streamlit app lets you run any scenario and see the signals, the verdicts and the physics violations.

![Simulation view of the Streamlit app](docs/app_simulation.png)

## Key findings

On the synthetic data, with attack strength epsilon = 0.10:

- A conventional gradient attack (FGSM or PGD) lowers the autoencoder's detection rate from 0.88 to 0.72 and 0.63, but it breaks P = V x I and the I-V curve, so the physics check still catches at least 0.98 of the attacked windows.
- A physics-constrained attack, which projects every step back onto the single-diode model, lowers the autoencoder's detection rate to 0.30, and all of its output passes the physics check. Physics check detection drops to 0.00.
- The hybrid detector inherits the autoencoder's weakness against that attack: it detects 0.30, the same as the autoencoder alone. Adding a physics check helps against naive attacks, not against one that knows the physics.
- Fault type matters. Without any attack, the autoencoder detects voltage sensor drift in only 0.65 of cases, while shading and inverter faults are detected in all of them. Under the physics-constrained attack, drift and inverter faults almost vanish (0.04 and 0.00), while shading stays at 0.83.

## Results

Mixed faults (shading, sensor drift, inverter), attack strength epsilon = 0.10, mean and standard deviation over 5 scenario seeds of 200 faulty and 200 normal windows each. False alarm rate: 0.03 for the autoencoder and the hybrid, 0.00 for the physics check.

| Attack | LSTM autoencoder | Physics check | Hybrid | Attacked windows that are physically valid |
|---|---|---|---|---|
| None | 0.88 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | not applicable |
| FGSM | 0.72 ± 0.03 | 0.98 ± 0.00 | 1.00 ± 0.00 | 2% |
| PGD | 0.63 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0% |
| Physics-constrained PGD | 0.30 ± 0.02 | 0.00 ± 0.00 | 0.30 ± 0.02 | 100% |

Values are fault detection rates (the share of faulty windows that raise an alarm).

![Detection rate by attack](docs/detection_by_attack.png)

Autoencoder detection rate by fault type, same setting:

| Fault | No attack | FGSM | PGD | Physics-constrained PGD |
|---|---|---|---|---|
| Partial shading | 1.00 | 0.90 | 0.88 | 0.83 |
| Sensor drift (voltage) | 0.65 | 0.18 | 0.04 | 0.04 |
| Inverter fault | 1.00 | 1.00 | 0.91 | 0.00 |

The physics check detects 0.95 or more of every fault under every attack except the physics-constrained one, where it detects none.

Effect of attack strength (mean over the same 5 seeds):

![Detection rate against attack strength](docs/epsilon_sweep.png)

Hybrid detection coincides with the physics check in the left panel and with the autoencoder in the right panel, because the hybrid raises an alarm if either one does.

All tables and figures come from `scripts/evaluate.py` and `scripts/make_figures.py`. The raw numbers are in `results/`. The detector is trained once, so the standard deviations reflect how scenarios are sampled (windows and fault parameters), not training variance.

## Method

- **Simulator.** A single-diode PV model driven by irradiance and cell temperature. Voltage, current and power are generated so that P = V x I holds at every step, with small sensor noise. Windows are 24 time steps of five signals: G, T, V, I, P.
- **Faults.** Partial shading (current and voltage drop), voltage sensor drift (V drifts upward while P is not updated) and inverter fault (current collapses).
- **AI detector.** An LSTM autoencoder trained on normal windows only. A window raises an alarm when its reconstruction error exceeds the 99th percentile of the error on held-out normal windows.
- **Attacks.** The attacker has white-box access to the autoencoder and perturbs the faulty window, within epsilon in min-max scaled units, to minimise its reconstruction error. FGSM takes one gradient step. PGD takes 20 steps of size epsilon / 5. Physics-constrained PGD projects the window onto the physics model after every step: the current is kept within a tolerance of the single-diode I-V curve and P is set to V x I.
- **Physics check.** Flags a window if the relative error |P - V x I| exceeds 2%, or if the current leaves a band around the single-diode I-V curve.
- **Hybrid detector.** Raises an alarm if the autoencoder or the physics check does.

## Getting started

```
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains the autoencoder, which takes about a minute or two, and saves it to `pv_models/`. Later launches load it directly. The sidebar lets you choose the fault, the attack and epsilon. The Simulation tab shows one scenario in detail. The Batch evaluation tab runs many scenarios and reports detection rate, false alarms and F1.

Run the tests:

```
pip install pytest
pytest
```

Reproduce the results:

```
python scripts/evaluate.py
python scripts/make_figures.py
```

## Using your own data

Choose "My dataset (CSV)" in the sidebar and point it at a train and a test CSV. The files need these columns: `instantaneous_global_radiation`, `clearsky_Tamb`, `radiation_mean`, `radiation_std`, `tprt_mean`, `tprt_std`, `label`, `site_num`, `date`, `source_file`. Windows with `label == 0` are used as normal data, and the simulator is driven by the real irradiance and temperature in the files. By default the app looks in `data/`. The notebooks read from `../data` or from the folder in the `PV_DATA_DIR` environment variable.

## Repository layout

```
app.py                  Streamlit app
pvcore.py               simulator, faults, autoencoder, attacks, physics checks, evaluation
scripts/evaluate.py     reproduces the results tables (writes results/)
scripts/make_figures.py draws the figures in docs/
notebooks/              experiments
tests/                  unit tests
results/                CSV output of the evaluation
docs/                   figures and screenshots
data/                   place for optional CSV datasets
pv_models/              trained detector weights
```

The notebooks:

- `pv_simulated_env.ipynb`: base study on the simulated environment
- `pv_advanced.ipynb`: multi-seed runs, baselines, hybrid detector, adversarial training
- `pv_real_data.ipynb`: experiments on a real labelled PV dataset

## Limitations

- Voltage, current and power are simulated and the faults are injected, so the numbers above describe this simulator, not a real plant.
- The attacker is white-box and controls all five signals. An attacker without the model, or without control of every sensor, would be weaker.
- The physics tolerances and the alarm threshold are chosen by hand and are not tuned per site.
- The detector is trained on one synthetic data set, with a single training run.

## Future work

Adversarial training and adaptive attacks against the hybrid detector, per-site thresholds, measured V and I data from a real array, and tighter physics checks that also use the temperature dependence.
