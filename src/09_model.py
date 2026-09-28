"""Step 09: model "does this cell look like cells that already have a Safia branch?".

Models: logistic regression (baseline) and LightGBM, each with and without the competitor flag.
Validation (main): 20 repeats of spatial GroupKFold over H3 res-6 blocks, blocks assigned to folds
at random each time, with a 2-ring buffer of training cells removed around every test fold.
One plain split and a random KFold are reported only for comparison.
Outputs: metrics JSON, out-of-fold scores for step 10, SHAP and curve figures, final model.
The model learns where branches ARE, not how well they sell (see README, Limitations).
"""
import json
import sys
import warnings

import h3
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import DATA_PROCESSED, OUTPUTS_FIGURES

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
N_FOLDS = 5
TARGET = "has_safia"
LEAKAGE = ["safia_count", "dist_safia_m"]
COMPETITOR = "has_competitor_1km"
SELECT_TOL = 0.01
BUFFER_K = 2  # H3 rings (~2 km) of training cells removed around each test fold
REPEATS = 20  # random assignments of blocks to folds; metrics and scores are averaged
SIMPLICITY_ORDER = ["logreg__no_competitor", "logreg__with_competitor", "lgbm__no_competitor", "lgbm__with_competitor"]
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    base = [c for c in df.columns if c.startswith("n_")] + ["district_density", "dist_centre_km", "nonres_share"]
    assert not set(base) & set(LEAKAGE)
    return {"no_competitor": base, "with_competitor": base + [COMPETITOR]}


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
    return lgb.LGBMClassifier(**LGB_PARAMS)


def cross_validate(df, feats, kind, splitter, groups=None, buffer_k=0):
    """Out-of-fold scores. With buffer_k > 0, training cells within buffer_k rings of any test cell
    are dropped: features use the cell and its 6 neighbours (k=1), so cells up to 2 rings apart share
    neighbours and would leak information across the fold border."""
    X, y, cells = df[feats].values, df[TARGET].values, df["h3"].values
    oof = np.zeros(len(df))
    folds = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        n_train = len(tr)
        if buffer_k:
            near_test = set().union(*(h3.grid_disk(c, buffer_k) for c in cells[te]))
            tr = tr[~np.isin(cells[tr], list(near_test))]
        model = make_model(kind).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            folds.append({"fold": k, "n": len(te), "pos": int(y[te].sum()), "train": len(tr),
                          "train_dropped": n_train - len(tr),
                          "roc_auc": roc_auc_score(y[te], oof[te]), "pr_auc": average_precision_score(y[te], oof[te])})
    return oof, pd.DataFrame(folds)


def main() -> int:
    df = pd.read_parquet(DATA_PROCESSED / "hex_features.parquet")
    y = df[TARGET].values
    prevalence = y.mean()
    print(f"Cells: {len(df)} | positives: {y.sum()} ({prevalence:.1%}) -> PR-AUC of a random model ~ {prevalence:.3f}")
    print(f"Spatial CV: GroupKFold({N_FOLDS}) over {df['cv_block'].nunique()} H3 res-6 blocks\n")

    spatial = GroupKFold(n_splits=N_FOLDS)
    random_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, oof_scores, fold_tables = [], {}, {}
    blocks = df["cv_block"].unique()
    for fs_name, feats in feature_sets(df).items():
        for kind in ["logreg", "lgbm"]:
            name = f"{kind}__{fs_name}"
            # Repeated buffered spatial CV: blocks are assigned to folds at random in each repeat.
            # One split is not enough: in step 09 a single lucky split made LightGBM look +0.046 better.
            oofs, prs, rocs = [], [], []
            for rep in range(REPEATS):
                order = np.random.default_rng(SEED + rep).permutation(len(blocks))
                groups = df["cv_block"].map(dict(zip(blocks, order)))
                oof_k, folds_k = cross_validate(df, feats, kind, spatial, groups, buffer_k=BUFFER_K)
                oofs.append(oof_k)
                prs.append(average_precision_score(y, oof_k))
                rocs.append(roc_auc_score(y, oof_k))
                if rep == 0:
                    folds = folds_k
            oof = np.mean(oofs, axis=0)  # score per cell = mean over repeats (more stable ranking)
            oof_nb, _ = cross_validate(df, feats, kind, spatial, df["cv_block"])
            oof_r, _ = cross_validate(df, feats, kind, random_cv)
            oof_scores[name] = oof
            fold_tables[name] = folds
            rows.append({"model": name, "n_features": len(feats),
                         "spatial_roc_auc": np.mean(rocs), "spatial_pr_auc": np.mean(prs),
                         "repeat_roc_auc_std": np.std(rocs), "repeat_pr_auc_std": np.std(prs),
                         "fold_roc_auc_std": folds["roc_auc"].std(), "fold_pr_auc_std": folds["pr_auc"].std(),
                         "nobuffer_roc_auc": roc_auc_score(y, oof_nb), "nobuffer_pr_auc": average_precision_score(y, oof_nb),
                         "random_roc_auc": roc_auc_score(y, oof_r), "random_pr_auc": average_precision_score(y, oof_r)})
    res = pd.DataFrame(rows).set_index("model")
    print(f"Pooled out-of-fold metrics. spatial = mean of {REPEATS} repeats of GroupKFold({N_FOLDS}) over H3 res-6 "
          f"blocks with a {BUFFER_K}-ring buffer (the honest estimate); nobuffer (one split) and random for comparison:")
    print(res.round(3).to_string())

    # Selection rule: among models within SELECT_TOL of the best spatial PR-AUC, take the simplest
    # (logistic regression before LightGBM, no competitor flag before with: the flag uses incomplete data)
    top_score = res["spatial_pr_auc"].max()
    eligible = [m for m in SIMPLICITY_ORDER if res.loc[m, "spatial_pr_auc"] >= top_score - SELECT_TOL]
    best = eligible[0]
    print(f"\nBest spatial PR-AUC: {res['spatial_pr_auc'].idxmax()} ({top_score:.3f}); "
          f"within {SELECT_TOL}: {eligible}")
    kind, fs_name = best.split("__")
    feats = feature_sets(df)[fs_name]
    lr_same = f"logreg__{fs_name}"
    comp_gain = (res.loc[f"{kind}__with_competitor", "spatial_pr_auc"]
                 - res.loc[f"{kind}__no_competitor", "spatial_pr_auc"])
    print(f"Chosen (simplest within tolerance): {best}")
    print(f"LightGBM vs logistic regression, same features: "
          f"{res.loc['lgbm__' + fs_name, 'spatial_pr_auc'] - res.loc[lr_same, 'spatial_pr_auc']:+.3f} PR-AUC")
    print(f"Adding the competitor flag ({kind}): {comp_gain:+.3f} PR-AUC")
    print(f"\nPer-fold spatial metrics for {best}:")
    print(fold_tables[best].round(3).to_string(index=False))

    # Final model on all cells (for SHAP and step 10 explanations)
    final = make_model(kind).fit(df[feats].values, y)
    model_files = {"lgbm": DATA_PROCESSED / "model_lgbm.txt", "logreg": DATA_PROCESSED / "model_logreg.json"}
    for other, path in model_files.items():  # never leave a stale model of a kind that was not chosen
        if other != kind:
            path.unlink(missing_ok=True)
    if kind == "logreg":
        scaler, clf = final[1], final[-1]
        model_files["logreg"].write_text(json.dumps({
            "features": feats, "transform": "log1p then standardise",
            "mean": dict(zip(feats, scaler.mean_.round(6).tolist())), "scale": dict(zip(feats, scaler.scale_.round(6).tolist())),
            "coef": dict(zip(feats, clf.coef_[0].round(6).tolist())), "intercept": round(float(clf.intercept_[0]), 6)},
            indent=2), encoding="utf-8")
    if kind == "lgbm":
        final.booster_.save_model(str(model_files["lgbm"]))
        explainer = shap.TreeExplainer(final)
        sv = explainer.shap_values(df[feats])
        sv = sv[1] if isinstance(sv, list) else sv
    else:
        explainer = shap.LinearExplainer(final[-1], final[:-1].transform(df[feats].values))
        sv = explainer.shap_values(final[:-1].transform(df[feats].values))
    pd.DataFrame(sv, columns=feats).assign(h3=df["h3"].values).to_parquet(DATA_PROCESSED / "shap_values.parquet", index=False)
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values()
    print("\nMean |SHAP| (top 10):")
    print(imp.tail(10)[::-1].round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 7))
    top = imp.tail(15)
    ax.barh(top.index, top.values, color="#c0392b")
    ax.set_xlabel("Mean |SHAP value| (impact on log-odds)")
    ax.set_title(f"What makes a cell look like a Safia location\n({best}, all cells)")
    fig.savefig(OUTPUTS_FIGURES / "09_shap_importance.png", dpi=150, bbox_inches="tight")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in res.index:
        fpr, tpr, _ = roc_curve(y, oof_scores[name])
        prec, rec, _ = precision_recall_curve(y, oof_scores[name])
        style = "-" if name == best else ":"
        a1.plot(fpr, tpr, style, label=f"{name} ({res.loc[name, 'spatial_roc_auc']:.2f})")
        a2.plot(rec, prec, style, label=f"{name} ({res.loc[name, 'spatial_pr_auc']:.2f})")
    a1.plot([0, 1], [0, 1], color="#999", linewidth=0.8)
    a2.axhline(prevalence, color="#999", linewidth=0.8)
    a1.set(title="ROC, spatial out-of-fold", xlabel="False positive rate", ylabel="True positive rate")
    a2.set(title="Precision-recall, spatial out-of-fold", xlabel="Recall", ylabel="Precision")
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    fig.savefig(OUTPUTS_FIGURES / "09_cv_curves.png", dpi=150, bbox_inches="tight")

    oof = pd.DataFrame({"h3": df["h3"], TARGET: y}).assign(**{f"oof_{k}": v for k, v in oof_scores.items()})
    oof["score"] = oof_scores[best]
    oof.to_parquet(DATA_PROCESSED / "oof_scores.parquet", index=False)
    metrics = {"n_cells": len(df), "positives": int(y.sum()), "prevalence": prevalence,
               "cv": f"{REPEATS} x GroupKFold({N_FOLDS}) over H3 res-6 blocks, {BUFFER_K}-ring buffer",
               "chosen_model": best, "features": feats,
               "results": res.round(4).reset_index().to_dict(orient="records"),
               "chosen_folds": fold_tables[best].round(4).to_dict(orient="records"),
               "competitor_flag_gain_pr_auc": round(comp_gain, 4)}
    (DATA_PROCESSED / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved model_metrics.json, oof_scores.parquet, shap_values.parquet, figures 09_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
