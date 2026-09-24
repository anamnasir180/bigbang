"""Evaluation protocols (experiment E1): nested CV, LOSO, leave-one-direction-out,
permutation test, confound tests, bootstrap CIs. Hyperparameters are only ever
tuned inside the training folds."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

POSITIVE = "collapse"  # positive class, stated explicitly in the paper


def models(seed=0):
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    return {
        "kNN": (pipe(KNeighborsClassifier()),
                {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]}),
        "SVM": (pipe(SVC(probability=True, random_state=seed)),
                {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}),
        "LogReg": (pipe(LogisticRegression(max_iter=2000)), {"clf__C": [0.01, 0.1, 1, 10]}),
        "RF": (pipe(RandomForestClassifier(300, random_state=seed)),
               {"clf__max_depth": [3, 6, None]}),
        "GB": (pipe(GradientBoostingClassifier(random_state=seed)),
               {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3]}),
        "MLP": (pipe(MLPClassifier(max_iter=2000, random_state=seed)),
                {"clf__hidden_layer_sizes": [(32,), (100,)], "clf__alpha": [1e-3, 1e-1]}),
    }


def tuned(model, grid, y_train, seed=0):
    k = max(2, min(5, int(np.bincount(y_train).min())))
    return GridSearchCV(clone(model), grid, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                        scoring="balanced_accuracy", n_jobs=-1)


def metrics(y, pred, proba):
    out = {"bal_acc": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
           "mcc": matthews_corrcoef(y, pred), "sensitivity": recall_score(y, pred, pos_label=1),
           "specificity": recall_score(y, pred, pos_label=0), "n": len(y)}
    out["auroc"] = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    return out


def bootstrap_ci(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [balanced_accuracy_score(y[i], pred[i])
             for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
             if len(np.unique(y[i])) == 2]
    return np.percentile(stats, [2.5, 97.5])


def _run_splits(X, y, splits, model, grid, seed):
    """Fit on every split; return per-fold metrics and pooled out-of-fold predictions."""
    rows, pooled = [], []
    for fold, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            continue
        est = tuned(model, grid, y[tr], seed).fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"fold": fold, **metrics(y[te], pred, proba), "best": str(est.best_params_)})
        pooled.append((te, pred, proba))
    return rows, pooled


def nested_cv(X, y, model, grid, repeats=10, folds=5, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    rows, _ = _run_splits(X, y, cv.split(X, y), model, grid, seed)
    return pd.DataFrame(rows)


def leave_group_out(X, y, groups, model, grid, seed=0):
    """LOSO (groups=subject) or leave-one-direction-out (groups=direction)."""
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    rows, pooled = _run_splits(X, y, splits, model, grid, seed)
    df = pd.DataFrame(rows)
    df["held_out"] = [groups[splits[r["fold"]][1][0]] for r in rows]
    if pooled:
        idx = np.concatenate([p[0] for p in pooled])
        pred = np.concatenate([p[1] for p in pooled])
        lo, hi = bootstrap_ci(y[idx], pred, seed=seed)
        df.attrs["pooled_bal_acc"] = balanced_accuracy_score(y[idx], pred)
        df.attrs["ci95"] = (lo, hi)
    return df


def permutation_test(X, y, model, grid, n_perm=200, seed=0):
    """Is within-subject CV performance above chance? Tuning is inside each fit."""
    rng = np.random.default_rng(seed)

    def score(labels):
        return nested_cv(X, labels, model, grid, repeats=1, seed=seed)["bal_acc"].mean()

    observed = score(y)
    null = np.array([score(rng.permutation(y)) for _ in range(n_perm)])
    return observed, (1 + (null >= observed).sum()) / (1 + n_perm), null


def confound_test(X, target, seed=0):
    """Can the features predict a nuisance variable (subject, direction)? High = shortcut risk."""
    t = pd.factorize(target)[0]
    if len(np.unique(t)) < 2:
        return np.nan
    model, grid = models(seed)["RF"]
    cv = StratifiedKFold(min(5, np.bincount(t).min()), shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, t):
        est = tuned(model, grid, t[tr], seed).fit(X[tr], t[tr])
        scores.append(balanced_accuracy_score(t[te], est.predict(X[te])))
    return float(np.mean(scores))


def learning_curve(X, y, model, grid, fractions=(0.2, 0.4, 0.6, 0.8, 1.0), repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y):
            for f in fractions:
                sub = rng.choice(tr, max(10, int(f * len(tr))), replace=False)
                if np.bincount(y[sub], minlength=2).min() < 2:
                    continue
                est = tuned(model, grid, y[sub], seed).fit(X[sub], y[sub])
                rows.append({"fraction": f, "n_train": len(sub),
                             "bal_acc": balanced_accuracy_score(y[te], est.predict(X[te]))})
    return pd.DataFrame(rows)
