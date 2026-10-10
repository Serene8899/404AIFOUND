import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import core

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
SEED = 42


def load_data():
    real = DATA / "ai4i2020.csv"
    if real.exists():
        return pd.read_csv(real), "AI4I 2020 (UCI) — real dataset"
    syn = DATA / "ai4i2020_synthetic.csv"
    if not syn.exists():
        from make_data import generate
        generate().to_csv(syn, index=False)
    return pd.read_csv(syn), "AI4I 2020 replica (generated from the published rules)"


def lgbm(**kw):
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED, verbose=-1)
    params.update(kw)
    return LGBMClassifier(**params)


def metrics(y, p, thr):
    yhat = p >= thr
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "false_alarms_per_1000": float(((yhat) & (y == 0)).sum() / len(y) * 1000),
    }


corrupt = core.corrupt


def main():
    df, source = load_data()
    print(f"Data: {source} — {len(df):,} rows, failure rate {df[core.LABEL].mean():.2%}")

    X = core.build_features(df)
    y = df[core.LABEL].values

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=SEED)

    # 1) Detection model -------------------------------------------------------
    # Train on clean data PLUS a deliberately messy copy (sensor noise and
    # dropped readings) so the model stays reliable on real plant data.
    aug_rng = np.random.default_rng(SEED + 1)
    messy = corrupt(df.iloc[tr], 0.05, 0.15, aug_rng)
    X_tr = pd.concat([X.iloc[tr], core.build_features(messy)])
    y_tr = np.concatenate([y[tr], y[tr]])
    det = lgbm()
    det.fit(X_tr, y_tr)
    p_va = det.predict_proba(X.iloc[va])[:, 1]
    p_te = det.predict_proba(X.iloc[te])[:, 1]

    # 2) Cost-optimal alert threshold (tuned on validation, NOT on test) -------
    thr, curve = core.best_threshold(y[va], p_va, core.DEFAULT_COSTS)
    print(f"Cost-optimal threshold: {thr:.2f}")

    # 3) Categorisation models (one per failure mode) -------------------------
    modes = {}
    mode_metrics = {}
    trva = np.concatenate([tr, va])
    for m in core.MODES:
        ym = df[m].values
        clf = lgbm(n_estimators=300)
        clf.fit(X.iloc[trva], ym[trva])
        modes[m] = clf
        pm = clf.predict_proba(X.iloc[te])[:, 1]
        mode_metrics[m] = {
            "support_test": int(ym[te].sum()),
            "pr_auc": float(average_precision_score(ym[te], pm)) if ym[te].sum() else None,
        }

    # Categorisation accuracy on true failures in test (excluding random RNF-only)
    te_df = df.iloc[te]
    fail_te = te_df[(te_df[core.LABEL] == 1) & (te_df[core.MODES].sum(axis=1) > 0)]
    mp = pd.DataFrame({m: modes[m].predict_proba(core.build_features(fail_te))[:, 1]
                       for m in core.MODES}, index=fail_te.index)
    pred_mode = mp.idxmax(axis=1)
    cat_acc = float(np.mean([fail_te.loc[i, pred_mode[i]] == 1 for i in fail_te.index]))
    print(f"Failure-type accuracy on real failures: {cat_acc:.1%}")

    # 4) Test metrics ---------------------------------------------------------
    m_test = metrics(y[te], p_te, thr)
    print("Test:", {k: round(v, 3) for k, v in m_test.items()})

    # 5) Robustness to noisy / missing sensor data ----------------------------
    rng = np.random.default_rng(SEED)
    robust = []
    for noise, miss, label in [(0, 0, "Clean"), (0.05, 0, "5% noise"), (0.10, 0, "10% noise"),
                               (0.20, 0, "20% noise"), (0, 0.10, "10% missing"),
                               (0.10, 0.10, "10% noise + 10% missing")]:
        d = corrupt(df.iloc[te], noise, miss, rng)
        pc = det.predict_proba(core.build_features(d))[:, 1]
        r = metrics(y[te], pc, thr)
        r["scenario"] = label
        robust.append(r)
    robust = pd.DataFrame(robust)
    print(robust[["scenario", "pr_auc", "precision", "recall", "false_alarms_per_1000"]].round(3))

    # 6) Policy backtest ------------------------------------------------------
    test_tbl = pd.DataFrame({
        "y": y[te], "p": p_te, "wear_min": X.iloc[te]["wear_min"].values,
        "TWF": te_df["TWF"].values, "OSF": te_df["OSF"].values,
    })
    bt = core.backtest(test_tbl, core.DEFAULT_COSTS, thr)
    print(bt[["policy", "interventions", "failures_caught", "false_alarms", "total_cost"]])

    # 7) Save artefacts -------------------------------------------------------
    healthy = X.iloc[tr][y[tr] == 0]
    bundle = {
        "detector": det,
        "modes": modes,
        "threshold": thr,
        "normal": healthy.mean().to_dict(),
        "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in core.SENSORS},
        "source": source,
    }
    joblib.dump(bundle, OUT / "bundle.joblib")

    # held-out data so the dashboard can re-run backtests live with new costs
    val_tbl = pd.DataFrame({"y": y[va], "p": p_va})
    val_tbl.to_csv(OUT / "val_predictions.csv", index=False)
    test_raw = te_df.reset_index(drop=True).copy()
    test_raw["p"] = p_te
    test_raw.to_csv(OUT / "test_set.csv", index=False)
    robust.to_csv(OUT / "robustness.csv", index=False)

    imp = pd.Series(det.booster_.feature_importance("gain"), index=core.FEATURES)
    json.dump({
        "source": source,
        "n_rows": int(len(df)),
        "failure_rate": float(y.mean()),
        "split": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "threshold": thr,
        "test_metrics": m_test,
        "mode_metrics": mode_metrics,
        "failure_type_accuracy": cat_acc,
        "feature_importance": (imp / imp.sum()).sort_values(ascending=False).to_dict(),
    }, open(OUT / "metrics.json", "w"), indent=2)
    print(f"Saved models and reports to {OUT}/")


if __name__ == "__main__":
    main()
