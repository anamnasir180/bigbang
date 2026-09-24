"""Run the re-experiments.

    python -m fallcause.run --manifest data/manifest.csv --out results/ --exp all
    python -m fallcause.run --synthetic --out results_synthetic/ --fast   # pipeline smoke test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import degrade, evaluate, segment
from .features import FEATURE_SETS
from .io import load_manifest


def build_xy(trials, feature_set="F1", t_pre=1.5, t_post=1.0, cfg=None, jitter_s=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, ff = [], 0
    for tr in trials:
        if cfg is not None:
            tr = degrade.apply(tr, cfg, rng)
        acc, gyro, info = segment.window(tr, t_pre, t_post, jitter_s, rng)
        ff += info["freefall_found"]
        rows.append(FEATURE_SETS[feature_set](acc, gyro, tr.fs, info))
    X = pd.DataFrame(rows).fillna(0.0)
    y = np.array([t.label == evaluate.POSITIVE for t in trials], dtype=int)
    meta = pd.DataFrame({"subject": [t.subject_id for t in trials],
                         "direction": [t.direction for t in trials]})
    return X, y, meta, ff / len(trials)


def summarize(df, **tags):
    return {**tags, "bal_acc_mean": df["bal_acc"].mean(), "bal_acc_sd": df["bal_acc"].std(),
            "mcc_mean": df["mcc"].mean(), "auroc_mean": df["auroc"].mean(),
            "sens_mean": df["sensitivity"].mean(), "spec_mean": df["specificity"].mean()}


def exp_protocol(trials, out, a):
    """E1 + E3 + E4: every feature set x model under nested CV, LOSO, and LODO."""
    rows, loso_rows = [], []
    for fs_name in a.feature_sets:
        X, y, meta, ff_rate = build_xy(trials, fs_name)
        X = X.to_numpy()
        for name, (model, grid) in evaluate.models(a.seed).items():
            if name not in a.models:
                continue
            cv = evaluate.nested_cv(X, y, model, grid, repeats=a.repeats, seed=a.seed)
            rows.append(summarize(cv, features=fs_name, model=name, protocol="nested_cv"))
            for grp in ("subject", "direction"):
                if meta[grp].nunique() < 2:
                    continue
                lg = evaluate.leave_group_out(X, y, meta[grp].to_numpy(), model, grid, a.seed)
                if lg.empty:
                    continue
                r = summarize(lg, features=fs_name, model=name, protocol=f"leave_one_{grp}_out")
                r["pooled_bal_acc"] = lg.attrs.get("pooled_bal_acc")
                r["ci95_lo"], r["ci95_hi"] = lg.attrs.get("ci95", (np.nan, np.nan))
                rows.append(r)
                lg.assign(features=fs_name, model=name, protocol=grp).pipe(loso_rows.append)
            print(f"  {fs_name} {name} done")
        conf = {g: evaluate.confound_test(X, meta[g].to_numpy(), a.seed)
                for g in ("subject", "direction")}
        rows.append({"features": fs_name, "protocol": "confound",
                     "subject_predictable": conf["subject"],
                     "direction_predictable": conf["direction"], "freefall_found_rate": ff_rate})
    pd.DataFrame(rows).to_csv(out / "E1_protocol_summary.csv", index=False)
    if loso_rows:
        pd.concat(loso_rows).to_csv(out / "E1_group_folds.csv", index=False)


def exp_permutation(trials, out, a):
    X, y, _, _ = build_xy(trials, a.best_features)
    model, grid = evaluate.models(a.seed)[a.best_model]
    obs, p, null = evaluate.permutation_test(X.to_numpy(), y, model, grid, a.n_perm, a.seed)
    json.dump({"observed_bal_acc": obs, "p_value": p, "null_mean": float(null.mean()),
               "n_perm": a.n_perm}, open(out / "E1_permutation.json", "w"), indent=2)


def exp_learning_curve(trials, out, a):
    X, y, _, _ = build_xy(trials, a.best_features)
    model, grid = evaluate.models(a.seed)[a.best_model]
    lc = evaluate.learning_curve(X.to_numpy(), y, model, grid, repeats=max(1, a.repeats // 2), seed=a.seed)
    lc.groupby("fraction").agg(["mean", "std"]).to_csv(out / "E1_learning_curve.csv")


def exp_segmentation(trials, out, a):
    """E2: sensitivity to window length and impact-alignment jitter."""
    model, grid = evaluate.models(a.seed)[a.best_model]
    rows = []
    pres, posts, jits = ((1.0,), (1.0,), (0.0, 0.2)) if a.fast else ((0.5, 1.0, 1.5, 2.0), (0.5, 1.0, 2.0), (0.0, 0.1, 0.2))
    for t_pre in pres:
        for t_post in posts:
            for jit in jits:
                X, y, _, _ = build_xy(trials, a.best_features, t_pre, t_post, jitter_s=jit, seed=a.seed)
                cv = evaluate.nested_cv(X.to_numpy(), y, model, grid, repeats=a.repeats, seed=a.seed)
                rows.append(summarize(cv, t_pre=t_pre, t_post=t_post, jitter_s=jit))
    pd.DataFrame(rows).to_csv(out / "E2_segmentation.csv", index=False)


def exp_sensor(trials, out, a):
    """E5: simulated sensor specifications (rate, range, resolution, noise, axes)."""
    C = degrade.SensorConfig
    configs = ([C()] + [C(fs=f) for f in (25, 12.5, 10, 5)]
               + [C(range_g=r) for r in (2, 4, 8)]
               + [C(range_g=8, bits=b) for b in (8, 10, 12)]
               + [C(noise_ug_rthz=n) for n in (100, 300, 1000)] + [C(axes="mag")])
    if a.fast:
        configs = [C(), C(fs=10), C(range_g=2, bits=8), C(axes="mag")]
    model, grid = evaluate.models(a.seed)[a.best_model]
    rows = []
    for cfg in configs:
        X, y, meta, _ = build_xy(trials, a.best_features, cfg=cfg, seed=a.seed)
        cv = evaluate.nested_cv(X.to_numpy(), y, model, grid, repeats=a.repeats, seed=a.seed)
        r = summarize(cv, config=cfg.name(), protocol="nested_cv")
        if meta["subject"].nunique() > 1:
            lg = evaluate.leave_group_out(X.to_numpy(), y, meta["subject"].to_numpy(), model, grid, a.seed)
            r["loso_pooled_bal_acc"] = lg.attrs.get("pooled_bal_acc")
        rows.append(r)
        print(f"  sensor {cfg.name()} done")
    pd.DataFrame(rows).to_csv(out / "E5_sensor_config.csv", index=False)


EXPERIMENTS = {"protocol": exp_protocol, "permutation": exp_permutation,
               "learning_curve": exp_learning_curve, "segmentation": exp_segmentation,
               "sensor": exp_sensor}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest")
    p.add_argument("--synthetic", action="store_true", help="pipeline test only; never report")
    p.add_argument("--out", default="results")
    p.add_argument("--exp", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    p.add_argument("--feature-sets", nargs="+", default=["F0", "F1", "F2", "F3"])
    p.add_argument("--models", nargs="+", default=list(evaluate.models()))
    p.add_argument("--best-features", default="F2")
    p.add_argument("--best-model", default="RF")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="tiny settings for smoke tests")
    a = p.parse_args(argv)
    if a.fast:
        a.repeats, a.n_perm, a.models, a.best_model = 1, 5, ["LogReg"], "LogReg"
    if a.synthetic:
        from .synthetic import make_dataset
        trials = make_dataset(25 if a.fast else 50, a.seed)
    elif a.manifest:
        trials = load_manifest(a.manifest)
    else:
        p.error("give --manifest or --synthetic")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(trials)} trials, labels: {pd.Series([t.label for t in trials]).value_counts().to_dict()}")
    for name in (EXPERIMENTS if "all" in a.exp else a.exp):
        print(f"[{name}]")
        EXPERIMENTS[name](trials, out, a)
    print(f"results written to {out}/")


if __name__ == "__main__":
    main()
