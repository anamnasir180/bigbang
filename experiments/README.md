# Fall-cause re-experiment pipeline

Implements the experiments in [`../docs/RESEARCH_PLAN_IEEE_SENSORS.md`](../docs/RESEARCH_PLAN_IEEE_SENSORS.md)
on the **existing** 200 trials. No new data collection is needed.

## 1. Prepare the data

Put the raw recordings under `data/` and write `data/manifest.csv`:

```csv
file,subject_id,label,direction,session
raw/S1_T001.txt,S1,collapse,forward,2024-05-02
raw/S1_T002.txt,S1,push,backward,2024-05-02
...
```

- `label`: `collapse` (simulated loss-of-consciousness) or `push`. `collapse` is the positive class.
- `file` can be:
  - a raw **GetSensorData** log (`.txt`). ACCE and GYRO lines are parsed automatically. Check the
    column order in `fallcause/io.py::read_getsensordata` against your app version.
  - a CSV with 3 numeric columns (x, y, z; g or m/s², detected automatically).
  - a legacy 1-column CSV of 121 magnitude values. Only F0/F1 features are meaningful here, and
    segmentation then works on the short window.
- Fill `direction` and `session` if you can recover them. They enable leave-one-direction-out and
  confound tests.

## 2. Run

```bash
pip install -r requirements.txt
python -m fallcause.run --manifest data/manifest.csv --out results/          # everything
python -m fallcause.run --manifest data/manifest.csv --exp protocol sensor    # selected experiments
python -m fallcause.run --synthetic --fast --out /tmp/smoke                  # pipeline smoke test only
```

| `--exp` | Plan section | Output |
|---|---|---|
| `protocol` | E1/E3/E4: feature sets F0–F3 × models under nested CV, LOSO, leave-one-direction-out, confound tests | `E1_protocol_summary.csv`, `E1_group_folds.csv` |
| `permutation` | E1: label-permutation p-value | `E1_permutation.json` |
| `learning_curve` | E1: performance vs. training size | `E1_learning_curve.csv` |
| `segmentation` | E2: window length / impact-jitter sensitivity | `E2_segmentation.csv` |
| `sensor` | E5: sampling rate, range, resolution, noise, axes | `E5_sensor_config.csv` |

`--best-features` / `--best-model` pick the configuration used by the permutation,
learning-curve, segmentation and sensor experiments. Choose them from **nested-CV** results,
never from LOSO results.

## Feature sets

| Set | Content |
|---|---|
| F0 | 121 raw magnitude samples around impact (the old paper's representation) |
| F1 | Phase features: pre-fall, descent, impact, post-impact (durations, free-fall depth, peak, impulse, jerk, settling time, dominant frequency) |
| F2 | F1 + tri-axial statistics and pre-to-post gravity-vector tilt |
| F3 | F2 + gyroscope features (only if GYRO lines exist in the raw logs) |

## Rules this pipeline enforces
- Hyperparameters are tuned **only inside training folds** (nested CV).
- Segmentation is automatic and identical for every trial (no visual inspection).
- Any augmentation or oversampling you add must go inside the training fold.
- Synthetic data (`--synthetic`) only exists to test the code. **Never report it.**
