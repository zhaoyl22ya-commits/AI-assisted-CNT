# -*- coding: utf-8 -*-
"""
Composite label robustness audit + optional reviewer-requested baselines.

Save to: <project_root>/models/evaluation/composite_robustness.py
Run:     python composite_robustness.py
Compare: python composite_robustness.py --compare-models

Uses existing L1+L2 data, frozen paper-grouped q50 out-of-fold predictions,
and the L12 split manifest to:
  1) count which of the five CNT quality components each record reports;
  2) evaluate frozen q50 predictions separately by reported-component pattern;
  3) on complete five-component TEST records, simulate realistic reporting
     patterns and compare scores and within-fold rankings to the full score.

IMPORTANT: Normalization is fitted ONLY on the saved OUTER TRAIN subset of
             each fold. The original test y_true MUST be reproduced before
             any conclusions are saved. Optional --compare-models adds complete-case, label-imputation and
             masked multitask baselines without changing frozen models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

TARGET = "y_composite_final_score"
SPLIT = "paper_grouped_cv"
COMPONENTS = ("gq", "igid", "ori", "purity", "bundle")
WEIGHTS = {"gq": 0.30, "igid": 0.20, "ori": 0.20,
           "purity": 0.15, "bundle": 0.15}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    needed = {"sample_id", "paper_id", "data_level", "y_growth_quality_score",
              "y_orientation_score", "y_purity_score", "y_bundle_density_per_um"}
    missing = sorted(needed - set(data.columns))
    if missing:
        raise ValueError(f"数据库缺少列：{missing}")
    data["data_level"] = data["data_level"].astype(str).str.upper()
    data = data.loc[data["data_level"].isin(("L1", "L2"))].copy()
    if data["sample_id"].isna().any() or data["sample_id"].duplicated().any():
        raise ValueError("L1/L2 数据中的 sample_id 为空或重复。")
    if data["paper_id"].isna().any():
        raise ValueError("数据库中存在空 paper_id。")
    data["sample_id"] = data["sample_id"].astype(str)
    data["paper_id"] = data["paper_id"].astype(str)

    igid_pref = (numeric(data["y_ig_id_ratio"])
                 if "y_ig_id_ratio" in data else pd.Series(np.nan, index=data.index))
    igid_old = (numeric(data["y_IG_ID"])
                if "y_IG_ID" in data else pd.Series(np.nan, index=data.index))
    bundle = numeric(data["y_bundle_density_per_um"])
    data["_gq"] = numeric(data["y_growth_quality_score"])
    data["_igid"] = igid_pref.combine_first(igid_old)
    data["_ori"] = numeric(data["y_orientation_score"])
    data["_purity"] = numeric(data["y_purity_score"])
    data["_bundle"] = np.log1p(bundle.where(bundle >= 0))

    observed = data[[f"_{k}" for k in COMPONENTS]].notna()
    observed.columns = list(COMPONENTS)
    data["reported_count"] = observed.sum(axis=1)
    data["reported_pattern"] = observed.apply(
        lambda row: "+".join(k for k in COMPONENTS if row[k]) or "none", axis=1
    )
    return data.set_index("sample_id", drop=False)


def load_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {"sample_id", "paper_id", "fold", "split_type", "y_true", "q50_raw"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"L12_predictions 缺少列：{missing}")
    if "target" in df.columns:
        df = df.loc[df["target"].astype(str) == TARGET]
    df = df.loc[df["split_type"].astype(str) == SPLIT].copy()
    if df.empty:
        raise ValueError("L12_predictions 没有 paper_grouped_cv 的 Composite 记录。")
    df["sample_id"] = df["sample_id"].astype(str)
    df["paper_id"] = df["paper_id"].astype(str)
    df["fold"] = numeric(df["fold"]).astype(int)
    df["y_true"] = numeric(df["y_true"])
    df["q50_raw"] = numeric(df["q50_raw"])
    if df[["y_true", "q50_raw"]].isna().any().any():
        raise ValueError("OOF 有缺失的真实标签或 q50 预测。")
    if df["sample_id"].duplicated().any():
        raise ValueError("paper-grouped OOF 中 sample_id 重复出现。")
    return df


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {"sample_id", "fold", "split_type", "role"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"L12_split_manifest 缺少列：{missing}")
    if "target" in df.columns:
        df = df.loc[df["target"].astype(str) == TARGET]
    df = df.loc[df["split_type"].astype(str) == SPLIT].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["role"] = df["role"].astype(str).str.lower().str.strip()
    df["fold"] = numeric(df["fold"]).astype(int)
    if df.empty or not {"train", "test"}.issubset(set(df["role"])):
        raise ValueError("Split manifest 中没有 paper_grouped_cv 的 train/test。")
    return df[["sample_id", "fold", "role"]].drop_duplicates()


def train_normalizer(train: pd.DataFrame) -> dict:
    result = {}
    for key in ("igid", "bundle"):
        finite = numeric(train[f"_{key}"]).dropna()
        lo = float(finite.min()) if len(finite) else np.nan
        hi = float(finite.max()) if len(finite) else np.nan
        result[key] = (lo, hi)
    return result


def minmax(values: pd.Series, bounds: tuple) -> pd.Series:
    lo, hi = bounds
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-15:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return ((numeric(values) - lo) / (hi - lo)).clip(0, 1)


def normalized_components(records: pd.DataFrame, normalizer: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "gq": (numeric(records["_gq"]) / 10).clip(0, 1),
        "igid": minmax(records["_igid"], normalizer["igid"]),
        "ori": numeric(records["_ori"]).clip(0, 1),
        "purity": numeric(records["_purity"]).clip(0, 1),
        "bundle": minmax(records["_bundle"], normalizer["bundle"]),
    }, index=records.index)


def composite(parts: pd.DataFrame, use: tuple[str, ...]) -> pd.Series:
    subset = parts[list(use)]
    w = pd.Series({k: WEIGHTS[k] for k in use}, dtype=float)
    den = subset.notna().mul(w, axis=1).sum(axis=1)
    num = subset.mul(w, axis=1).sum(axis=1)
    return (num / den.where(den > 0)).where(subset.notna().sum(axis=1) >= 2)


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 2:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics(group: pd.DataFrame) -> dict:
    a, b = group["y_true"].to_numpy(float), group["q50_raw"].to_numpy(float)
    return {
        "n_samples": len(group),
        "n_papers": group["paper_id"].nunique(),
        "r2": (float(r2_score(a, b)) if len(a) > 1 and np.ptp(a) > 1e-14 else np.nan),
        "spearman": spearman(a, b),
        "mae": float(np.mean(np.abs(a - b))) if len(a) else np.nan,
        "mean_signed_error": float(np.mean(b - a)) if len(a) else np.nan,
    }


def paper_bootstrap(group: pd.DataFrame, repeats: int, rng) -> dict:
    blank = {k: np.nan for k in ("spearman_ci_low", "spearman_ci_high",
                                       "r2_ci_low", "r2_ci_high")}
    if repeats < 1 or len(group) < 20 or group["paper_id"].nunique() < 5:
        return blank
    bags = [(sub["y_true"].to_numpy(float), sub["q50_raw"].to_numpy(float))
            for _, sub in group.groupby("paper_id", sort=False)]
    rhos, r2s = [], []
    for _ in range(repeats):
        indices = rng.integers(len(bags), size=len(bags))
        actual = np.concatenate([bags[i][0] for i in indices])
        predicted = np.concatenate([bags[i][1] for i in indices])
        rho = spearman(actual, predicted)
        if np.isfinite(rho):
            rhos.append(rho)
        if len(actual) > 1 and np.ptp(actual) > 1e-14:
            mean = actual.mean()
            denominator = np.sum((actual - mean) ** 2)
            r2s.append(1.0 - np.sum((actual - predicted) ** 2) / denominator)
    if len(rhos) >= 20:
        blank["spearman_ci_low"], blank["spearman_ci_high"] = np.quantile(rhos, [.025, .975])
    if len(r2s) >= 20:
        blank["r2_ci_low"], blank["r2_ci_high"] = np.quantile(r2s, [.025, .975])
    return blank


def pattern_distribution(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, sub in (("L1+L2", data),
                       ("L1", data.loc[data["data_level"] == "L1"]),
                       ("L2", data.loc[data["data_level"] == "L2"])):
        eligible = sub.loc[sub["reported_count"] >= 2]
        for pattern, group in eligible.groupby("reported_pattern"):
            out.append({"scope": label, "pattern": pattern,
                        "component_count": int(group["reported_count"].iloc[0]),
                        "n_samples": len(group), "n_papers": group["paper_id"].nunique(),
                        "share_eligible": len(group) / len(eligible) if len(eligible) else np.nan})
    return pd.DataFrame(out).sort_values(["scope", "n_samples"], ascending=[True, False])


def top_overlap(x: np.ndarray, y: np.ndarray, fraction: float) -> float:
    k = max(1, int(math.ceil(fraction * len(x))))
    first = set(np.argsort(-x, kind="mergesort")[:k])
    second = set(np.argsort(-y, kind="mergesort")[:k])
    return len(first & second) / len(first | second)


def plot_robustness(out: Path) -> None:
    """Save publication-ready reporting-pattern and masking diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"gq": "GQ", "igid": "IG", "ori": "Ori",
              "purity": "Pur", "bundle": "Bun"}

    def short(pattern):
        return "+".join(labels.get(part, part) for part in pattern.split("+"))

    dist = pd.read_csv(out / "composite_pattern_distribution.csv")
    detail = pd.read_csv(out / "composite_pattern_oof_detail.csv")
    masks = pd.read_csv(out / "composite_robustness_summary.csv")
    metric_path = out / "composite_pattern_oof_metrics.csv"
    if metric_path.is_file():
        oof_metrics = pd.read_csv(metric_path)
    else:
        # Plots-only works even if only the detailed OOF file was retained.
        oof_metrics = pd.DataFrame([
            {"scope": "L1+L2", "pattern": pat, **metrics(group)}
            for pat, group in detail.groupby("scored_pattern")
        ])

    # Panel A and B use the same observed-pattern order.
    distribution = (dist.loc[dist["scope"] == "L1+L2"]
                    .sort_values("n_samples", ascending=False).head(8))
    patterns = distribution["pattern"].tolist()
    y = np.arange(len(patterns))
    ordered = (oof_metrics.loc[oof_metrics["scope"] == "L1+L2"]
               .set_index("pattern").reindex(patterns))
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 11), constrained_layout=True)
    ax[0, 0].barh(y, distribution["n_samples"].to_numpy())
    ax[0, 0].set_yticks(y, [short(p) for p in patterns])
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlim(0, float(distribution["n_samples"].max()) * 1.20)
    ax[0, 0].set_xlabel("Number of samples")
    ax[0, 0].set_title("a  Most common reporting patterns", loc="left")
    for pos, size in zip(y, distribution["n_samples"]):
        ax[0, 0].text(size + 7, pos, str(int(size)), va="center", fontsize=9)

    ax[0, 1].axvline(0, color="0.45", lw=1, ls="--")
    for pos, (_, row) in zip(y, ordered.iterrows()):
        rho = row.get("spearman", np.nan)
        if not np.isfinite(rho):
            continue
        lo, hi = row.get("spearman_ci_low", np.nan), row.get("spearman_ci_high", np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and lo <= rho <= hi:
            ax[0, 1].errorbar(rho, pos, xerr=[[rho - lo], [hi - rho]],
                              fmt="o", capsize=3, markersize=6)
        else:
            ax[0, 1].plot(rho, pos, "o", markersize=6)
    ax[0, 1].set_yticks(y, [short(p) for p in patterns])
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlim(-1, 1)
    ax[0, 1].set_xlabel("OOF Spearman correlation (frozen q50)")
    ax[0, 1].set_title("b  Ranking performance by reporting pattern", loc="left")

    if not masks.empty:
        mask_view = masks.assign(_count=masks["retained_components"].str.count(r"\+") + 1)
        mask_view = mask_view.sort_values(["_count", "retained_components"])
        yp = np.arange(len(mask_view))
        names = [short(p) for p in mask_view["retained_components"]]
        ax[1, 0].barh(yp, mask_view["median_rank_spearman"].to_numpy())
        ax[1, 0].set_yticks(yp, names)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_xlim(-1, 1)
        ax[1, 0].axvline(0, color="0.5", lw=0.8)
        ax[1, 0].set_xlabel("Median foldwise rank correlation with full label")
        ax[1, 0].set_title("c  Label ranking after component masking", loc="left")
        ax[1, 1].barh(yp, mask_view["median_top10pct_jaccard"].to_numpy())
        ax[1, 1].set_yticks(yp, names)
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_xlim(0, 1.05)
        ax[1, 1].set_xlabel("Median foldwise Top-10% Jaccard overlap")
        ax[1, 1].set_title("d  Top-10% selection after component masking", loc="left")
    fig.suptitle("Composite: reporting-pattern and label-mask sensitivity", fontsize=16)
    fig.supxlabel("GQ = growth quality; IG = IG/ID; Ori = orientation; Pur = purity; Bun = bundle density", fontsize=10)
    fig.savefig(out / "figS_composite_reporting_robustness.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "figS_composite_reporting_robustness.svg", bbox_inches="tight")
    plt.close(fig)

    # Case-level illustration from the already-saved masking results.
    case_path = out / "composite_masking_sample_detail.csv"
    selected = ("gq+igid", "gq+ori+purity+bundle")
    if case_path.is_file():
        cases = pd.read_csv(case_path)
    else:
        # For archived outputs that kept normalized OOF components but not case-level masks.
        cols = {"gq": "gq_01", "igid": "igid_norm01", "ori": "ori_01",
                "purity": "purity_01", "bundle": "bundle_norm01"}
        cases = pd.DataFrame()
        complete = detail.loc[detail["scored_count"] == 5].copy()
        if len(complete) and set(cols.values()).issubset(complete.columns):
            parts = pd.DataFrame({k: numeric(complete[v]) for k, v in cols.items()},
                                 index=complete.index)
            cases = pd.concat([
                pd.DataFrame({"retained_components": pat,
                              "score_full": numeric(complete["y_true"]).to_numpy(float),
                              "score_masked": composite(parts, tuple(pat.split("+"))).to_numpy(float)})
                for pat in selected
            ], ignore_index=True)
    if not cases.empty and all(p in set(cases["retained_components"]) for p in selected):
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True,
                                 sharex=True, sharey=True)
        for i, (axis, pat) in enumerate(zip(axes, selected)):
            pair = cases.loc[cases["retained_components"] == pat]
            x = numeric(pair["score_full"]).to_numpy(float)
            val = numeric(pair["score_masked"]).to_numpy(float)
            axis.scatter(x, val, alpha=0.45, s=22)
            axis.plot([0, 1], [0, 1], ls="--", color="0.4", lw=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Original full five-component score")
            axis.set_title(f"{chr(97+i)}  Retain {short(pat)}", loc="left")
            axis.text(0.03, 0.97, f"n = {len(x)}\nMean |change| = {np.mean(np.abs(val - x)):.3f}",
                      transform=axis.transAxes, va="top", fontsize=10)
        axes[0].set_ylabel("Score after retaining selected components")
        fig2.suptitle("Same samples, different reported component sets", fontsize=14)
        fig2.savefig(out / "figS_composite_masking_examples.png", dpi=240, bbox_inches="tight")
        fig2.savefig(out / "figS_composite_masking_examples.svg", bbox_inches="tight")
        plt.close(fig2)
    print(f"[PLOT] Saved figures in: {out}")



# -----------------------------------------------------------------------------
# Optional reviewer-requested model comparisons (paper-grouped outer folds).
# These models are evaluation-only and NEVER overwrite frozen L12 predictions.
# -----------------------------------------------------------------------------

def comparison_feature_columns(root: Path, data: pd.DataFrame) -> list[str]:
    schema_path = root / "data/11_flattened/feature_schema.json"
    physical = ([f"N{i}" for i in range(1, 19)]
                + [f"C{i}" for i in range(1, 13)]
                + [f"F{i}" for i in range(1, 19)]
                + [f"P{i}" for i in range(1, 13)])
    expected = physical + [f"miss_{c}" for c in physical]
    if schema_path.is_file():
        columns = json.loads(schema_path.read_text(encoding="utf-8")).get("model_feature_columns")
        if not isinstance(columns, list) or len(columns) != len(set(columns)):
            raise ValueError(f"无效的特征顺序文件：{schema_path}")
        if set(columns) != set(expected):
            raise ValueError("feature_schema.json 与当前 120D N/C/F/P + missingness 定义不一致。")
    else:
        columns = expected
        print("[INFO] 未找到 feature_schema.json，按 N/C/F/P + 对应 miss_ 列的固定顺序读取120D。")
    absent = [col for col in columns if col not in data.columns]
    if absent:
        raise ValueError(f"120D 模型特征缺少：{absent[:12]}")
    return columns


def train_q50_baseline(X: pd.DataFrame, y: np.ndarray, papers: np.ndarray,
                       rounds: int, patience: int, seed: int):
    """Quantile LightGBM: inner split by paper; final fit on full outer train."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit

    if len(X) < 25 or len(set(papers)) < 5:
        raise ValueError("对照训练折完整样本/独立论文不足，无法进行可信的 q50 评估。")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    it, iv = next(splitter.split(X, groups=papers))
    if len(it) < 15 or len(iv) < 5:
        raise ValueError("对照模型的内层论文验证样本太少。")
    params = {"objective": "quantile", "alpha": 0.5,
              "metric": "quantile", "learning_rate": 0.03,
              "num_leaves": 63, "min_data_in_leaf": 20,
              "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "verbosity": -1, "num_threads": 2,
              "seed": seed, "feature_fraction_seed": seed,
              "bagging_seed": seed, "data_random_seed": seed}
    sample = lgb.Dataset(X.iloc[it], label=y[it])
    val = lgb.Dataset(X.iloc[iv], label=y[iv], reference=sample)
    interim = lgb.train(params, sample, num_boost_round=rounds,
                        valid_sets=[val],
                        callbacks=[lgb.early_stopping(patience, verbose=False),
                                   lgb.log_evaluation(0)])
    best = int(interim.best_iteration or rounds)
    model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=best)
    return model, best


def fit_masked_multitask(X_train: pd.DataFrame, targets: pd.DataFrame,
                         X_test: pd.DataFrame, rank: int,
                         reg: float, maxiter: int, seed: int) -> tuple[np.ndarray, dict]:
    """Genuine joint five-output low-rank model with observed-target masked loss.

    Shared representation H = X_scaled @ U, five heads Yhat = H @ V + b.
    Only observed labels contribute to the objective. No label imputation.
    Normalization of input features is fitted on the OUTER TRAIN only.
    """
    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler

    train = X_train.to_numpy(dtype=np.float64, na_value=np.nan)
    test = X_test.to_numpy(dtype=np.float64, na_value=np.nan)
    medians = pd.DataFrame(np.where(np.isfinite(train), train, np.nan)).median().fillna(0).to_numpy(dtype=float)
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    scaler = StandardScaler()
    z = np.clip(scaler.fit_transform(train), -8, 8)
    z_test = np.clip(scaler.transform(test), -8, 8)

    yt = targets[list(COMPONENTS)].to_numpy(dtype=float)
    observed = np.isfinite(yt)
    if not observed.any(axis=0).all():
        raise ValueError("某个质量目标在当前训练折中完全缺失，无法进行五目标对照。")
    y_filled = np.where(observed, yt, 0.0)
    d, n_task = z.shape[1], yt.shape[1]
    rank = max(1, min(rank, d, n_task * 4))
    rg = np.random.default_rng(seed)
    u0 = rg.normal(0, 0.025, size=(d, rank))
    v0 = rg.normal(0, 0.07, size=(rank, n_task))
    bias0 = np.sum(y_filled, axis=0) / observed.sum(axis=0)
    x0 = np.concatenate([u0.ravel(), v0.ravel(), bias0])
    count = observed.sum()

    def fun(theta):
        pos = d * rank
        U = theta[:pos].reshape(d, rank)
        V = theta[pos:pos + rank * n_task].reshape(rank, n_task)
        b = theta[pos + rank * n_task:]
        H = z @ U
        pred = H @ V + b
        residual = np.where(observed, pred - y_filled, 0.0)
        loss = np.square(residual).sum() / count + 0.5 * reg * (np.square(U).sum() + np.square(V).sum())
        grad_pred = (2.0 / count) * residual
        grad_U = z.T @ (grad_pred @ V.T) + reg * U
        grad_V = H.T @ grad_pred + reg * V
        grad_b = grad_pred.sum(axis=0)
        return float(loss), np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_b])

    solution = minimize(fun, x0, jac=True, method="L-BFGS-B",
                        options={"maxiter": maxiter, "ftol": 1e-9})
    t = solution.x
    U = t[:d * rank].reshape(d, rank)
    V = t[d * rank:d * rank + rank * n_task].reshape(rank, n_task)
    b = t[d * rank + rank * n_task:]
    predicted_tasks = np.clip((z_test @ U) @ V + b, 0.0, 1.0)
    weights = np.asarray([WEIGHTS[k] for k in COMPONENTS])
    result = predicted_tasks @ weights / weights.sum()
    return result, {"mt_iterations": int(solution.nit),
                    "mt_success": bool(solution.success),
                    "mt_final_loss": float(solution.fun), "mt_rank": rank}


def comparison_metrics(rows: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    options = {"Frozen q50": "frozen_q50",
               "Complete-case q50": "complete_case_q50",
               "Imputed-label q50": "imputation_q50",
               "Masked multitask": "multitask_q50"}
    results = []
    rng = np.random.default_rng(seed)
    for scope, sub in [("pooled", rows)] + [(f"fold_{int(f)}", g) for f, g in rows.groupby("fold")]:
        for name, field in options.items():
            base = sub[["sample_id", "paper_id", "y_true_full", field]].copy()
            base = base.rename(columns={"y_true_full": "y_true", field: "q50_raw"})
            entry = {"scope": scope, "model": name, **metrics(base)}
            if scope == "pooled":
                entry.update(paper_bootstrap(base, repeats, rng))
            results.append(entry)
    return pd.DataFrame(results)


def comparison_topk(rows: pd.DataFrame) -> pd.DataFrame:
    methods = {"Complete-case q50": "complete_case_q50",
               "Imputed-label q50": "imputation_q50",
               "Masked multitask": "multitask_q50"}
    records = []
    for scope, group in [("pooled", rows)] + [(f"fold_{int(f)}", s) for f, s in rows.groupby("fold")]:
        ref = group["frozen_q50"].to_numpy(dtype=float)
        truth = group["y_true_full"].to_numpy(dtype=float)
        for fraction in (0.1, 0.2):
            for name, col in methods.items():
                pred = group[col].to_numpy(dtype=float)
                records.append({"scope": scope, "method": name, "top_fraction": fraction,
                                "n_test": len(group),
                                "jaccard_vs_frozen": top_overlap(ref, pred, fraction),
                                "jaccard_vs_true": top_overlap(truth, pred, fraction)})
            records.append({"scope": scope, "method": "Frozen q50", "top_fraction": fraction,
                            "n_test": len(group), "jaccard_vs_frozen": 1.0,
                            "jaccard_vs_true": top_overlap(truth, ref, fraction)})
    return pd.DataFrame(records)


def plot_model_comparison(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = pd.read_csv(out / "composite_model_comparison_metrics.csv")
    tops = pd.read_csv(out / "composite_model_comparison_topk.csv")
    pool = result.loc[result["scope"] == "pooled"].copy()
    names = ["Frozen q50", "Complete-case q50", "Imputed-label q50", "Masked multitask"]
    pool = pool.set_index("model").reindex(names)
    ptop = tops.loc[tops["scope"] == "pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    xx = np.arange(len(names))
    label = ["Frozen\nq50", "Complete\ncase", "Imputed\nlabel", "Masked\nmultitask"]

    axes[0].axhline(0, ls="--", lw=0.8, color="0.55")
    for i, name in enumerate(names):
        row = pool.loc[name]
        rho = float(row["spearman"])
        lo, hi = float(row["spearman_ci_low"]), float(row["spearman_ci_high"])
        if np.isfinite(lo) and np.isfinite(hi):
            axes[0].errorbar(i, rho, yerr=[[max(0.,rho-lo)],[max(0.,hi-rho)]],
                             fmt="o", capsize=4)
        else:
            axes[0].plot(i, rho, "o")
    axes[0].set_xticks(xx, label)
    axes[0].set_ylabel("Paper-grouped OOF Spearman")
    axes[0].set_title("a  Same-sample ranking comparison", loc="left")

    axes[1].axhline(0, ls="--", lw=0.8, color="0.55")
    axes[1].bar(xx, pool["r2"].to_numpy(dtype=float))
    axes[1].set_xticks(xx, label)
    axes[1].set_ylabel("R² against full-component labels")
    axes[1].set_title("b  Absolute score prediction", loc="left")

    barwidth = 0.33
    for j, fraction in enumerate((0.1, 0.2)):
        values = []
        for name in names:
            row = ptop.loc[(ptop["method"] == name) & (np.isclose(ptop["top_fraction"], fraction))]
            values.append(float(row["jaccard_vs_frozen"].iloc[0]) if not row.empty else np.nan)
        axes[2].bar(xx + (j-.5)*barwidth, values, barwidth,
                    label=f"Top {int(fraction*100)}%")
    axes[2].set_xticks(xx, label)
    axes[2].set_ylim(0, 1.04)
    axes[2].set_ylabel("Jaccard vs frozen q50 selection")
    axes[2].set_title("c  Selection-list agreement", loc="left")
    axes[2].legend(frameon=False)
    fig.suptitle("Composite alternatives on the identical complete-case held-out set", fontsize=13)
    fig.savefig(out / "figS_composite_model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figS_composite_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def run_model_comparisons(root: Path, data: pd.DataFrame, oof: pd.DataFrame,
                          manifest: pd.DataFrame, out: Path, args) -> None:
    """Evaluation-only; identical complete-case TEST rows for all four methods."""
    features = comparison_feature_columns(root, data)
    X_all = data[features].apply(pd.to_numeric, errors="coerce")
    preds, audit = [], []
    for fold, original_test in oof.groupby("fold", sort=True):
        roles = manifest.loc[manifest["fold"] == fold]
        train_ids = set(roles.loc[roles["role"] == "train", "sample_id"])
        test_ids = set(roles.loc[roles["role"] == "test", "sample_id"])
        if train_ids & test_ids or test_ids != set(original_test["sample_id"]):
            raise ValueError(f"fold={fold}: 模型对照发现外层训练/测试划分不一致。")
        if (set(data.loc[list(train_ids), "paper_id"]) &
                set(data.loc[list(test_ids), "paper_id"])):
            raise ValueError(f"fold={fold}: 模型对照发现论文泄漏。")
        # Keep the exact existing >=2 population. All normalizers fit on outer TRAIN.
        train = data.loc[sorted(train_ids)]
        train = train.loc[train["reported_count"] >= 2]
        test = data.loc[original_test["sample_id"].tolist()]
        normalizer = train_normalizer(train)
        ytrain = normalized_components(train, normalizer)
        ytest = normalized_components(test, normalizer)
        full_test = ytest.notna().all(axis=1)
        if int(full_test.sum()) < 10:
            raise ValueError(f"fold={fold}: 完整五分量的测试样本过少。")
        test = test.loc[full_test]
        y_test = composite(ytest.loc[full_test], COMPONENTS)
        stored = original_test.set_index("sample_id").loc[test.index, "y_true"].to_numpy(float)
        if not np.allclose(stored, y_test.to_numpy(float), atol=args.tolerance, rtol=0):
            raise ValueError(f"fold={fold}: 完整五分量的测试标签与原始 OOF 不相符。")

        X_train = X_all.loc[train.index]
        X_test = X_all.loc[test.index]
        complete_train = ytrain.notna().all(axis=1)
        cc_train = train.loc[complete_train]
        if len(cc_train) < 40:
            raise ValueError(f"fold={fold}: 完整五分量训练样本不足40条。")
        y_cc = composite(ytrain.loc[complete_train], COMPONENTS).to_numpy(float)
        cc, cc_best = train_q50_baseline(X_train.loc[cc_train.index], y_cc,
                                         cc_train["paper_id"].to_numpy(),
                                         args.max_boost_rounds, args.early_stopping_rounds,
                                         args.seed + int(fold))
        predicted_cc = cc.predict(X_test)

        fill = ytrain.median(skipna=True).reindex(COMPONENTS)
        if fill.isna().any():
            raise ValueError(f"fold={fold}: 某些质量分量完全没有训练标签，无法进行中位数插补。")
        y_imputed = ytrain.fillna(fill)
        wt = np.asarray([WEIGHTS[k] for k in COMPONENTS])
        y_imp_score = y_imputed.to_numpy(float) @ wt / wt.sum()
        imp, imp_best = train_q50_baseline(X_train, y_imp_score,
                                           train["paper_id"].to_numpy(),
                                           args.max_boost_rounds, args.early_stopping_rounds,
                                           args.seed + int(fold))
        predicted_imp = imp.predict(X_test)
        predicted_mt, mtlog = fit_masked_multitask(X_train, ytrain, X_test,
                                                    args.multitask_rank, args.multitask_reg,
                                                    args.multitask_maxiter, args.seed + int(fold))
        frozen = original_test.set_index("sample_id").loc[test.index, "q50_raw"].to_numpy(float)
        part = pd.DataFrame({"sample_id": test.index.astype(str),
                             "paper_id": test["paper_id"].to_numpy(),
                             "fold": int(fold), "y_true_full": y_test.to_numpy(float),
                             "frozen_q50": frozen, "complete_case_q50": predicted_cc,
                             "imputation_q50": predicted_imp, "multitask_q50": predicted_mt})
        if part[["y_true_full", "frozen_q50", "complete_case_q50", "imputation_q50",
                 "multitask_q50"]].isna().any().any():
            raise ValueError(f"fold={fold}: 存在非数值对照模型输出。")
        preds.append(part)
        audit.append({"fold": int(fold), "n_outer_train": len(train),
                      "n_outer_train_complete": int(complete_train.sum()),
                      "n_outer_test": len(original_test),
                      "n_outer_test_complete": len(test),
                      "n_papers_complete_test": test["paper_id"].nunique(),
                      "complete_case_best_round": cc_best,
                      "imputed_best_round": imp_best,
                      "n_imputed_component_cells": int(ytrain.isna().to_numpy().sum()),
                      "imputed_component_medians": json.dumps(fill.to_dict()), **mtlog})
        print(f"[COMPARE] fold={fold}: train {len(train)} / full train {len(cc_train)}, "
              f"full test {len(test)}, q50 rounds {cc_best}/{imp_best}")

    joined = pd.concat(preds, ignore_index=True)
    if joined["sample_id"].duplicated().any():
        raise ValueError("完整测试样本出现重复：违反OOF条件。")
    full_expected = set(data.loc[data["reported_count"] == 5].index) & set(oof["sample_id"])
    if set(joined["sample_id"]) != full_expected:
        raise ValueError("完整五分量的测试样本没有全部进入模型对照。")
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "composite_model_comparison_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(out / "composite_model_comparison_fold_audit.csv", index=False, encoding="utf-8-sig")
    comparison_metrics(joined, args.bootstrap, args.seed).to_csv(
        out / "composite_model_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_topk(joined).to_csv(out / "composite_model_comparison_topk.csv", index=False, encoding="utf-8-sig")
    meta = {"status": "exploratory_baseline_comparison", "main_models_overwritten": False,
            "split": SPLIT, "evaluation": "identical five-component outer TEST samples",
            "n_test_complete": len(joined), "model_input": len(features),
            "methods": {"frozen_q50": "existing official OOF",
                        "complete_case_q50": "quantile LightGBM trained on 5-component training rows",
                        "imputation_q50": "training-fold median fill of missing normalized target components, quantile LightGBM",
                        "multitask": "joint low-rank 5-output model, shared latent basis, observed-target masked squared error"},
            "not_claimed": ["matching model capacities", "causal proof of reporting comparability",
                            "prospective experimental validation"],
            "seed": args.seed, "multitask_rank": args.multitask_rank,
            "multitask_regularization": args.multitask_reg}
    (out / "composite_model_comparison_run_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_model_comparison(out)
    print(f"[OK] Three reviewer-requested baseline comparisons completed: {len(joined)} held-out full cases")
    print(f"[PLOT] {out / 'figS_composite_model_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Frozen Composite reporting-pattern robustness audit")
    parser.add_argument("--compare-models", action="store_true", default=True,
                        help="Train only the 3 independent reviewer baselines and draw their SI figure")
    parser.add_argument("--max-boost-rounds", type=int, default=750)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--multitask-rank", type=int, default=8)
    parser.add_argument("--multitask-reg", type=float, default=0.01)
    parser.add_argument("--multitask-maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--min-mask-frequency", type=int, default=15)
    parser.add_argument("--max-masks", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--plots-only", action="store_true", help="Only redraw figures from existing CSV results")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output folder override")
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_path = root / "data/11_flattened/ml_dataset_3p0_full60D.csv"
    pred_path = root / "outputs/models/l12_quantile/L12_quantile_predictions.csv"
    split_path = root / "outputs/models/l12_quantile/L12_split_manifest.csv"
    out = root / "outputs/external_validation/composite_robustness"
    if args.output_dir is not None:
        out = args.output_dir.resolve()
    if args.plots_only:
        plot_robustness(out)
        if (out / "composite_model_comparison_metrics.csv").is_file() and (
                out / "composite_model_comparison_topk.csv").is_file():
            plot_model_comparison(out)
        return
    for path in (data_path, pred_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要输入：{path}")

    data = read_dataset(data_path)
    oof = load_oof(pred_path)
    manifest = load_manifest(split_path)
    absent = set(oof["sample_id"]) - set(data.index)
    if absent:
        raise ValueError(f"OOF 有 {len(absent)} 条 sample_id 在 L1/L2 数据中找不到。")
    dist = pattern_distribution(data)
    candidate_masks = dist.loc[
        (dist["scope"] == "L1+L2") &
        (dist["component_count"].between(2, 4)) &
        (dist["n_samples"] >= args.min_mask_frequency), "pattern"
    ].head(args.max_masks).tolist()

    rng = np.random.default_rng(20261014)
    audit_rows, fold_frames, mask_rows, mask_sample_rows = [], [], [], []
    for fold, test_prediction in oof.groupby("fold", sort=True):
        sub_manifest = manifest.loc[manifest["fold"] == fold]
        train_ids = set(sub_manifest.loc[sub_manifest["role"] == "train", "sample_id"])
        test_ids = set(sub_manifest.loc[sub_manifest["role"] == "test", "sample_id"])
        predicted_ids = set(test_prediction["sample_id"])
        if train_ids & test_ids:
            raise ValueError(f"fold={fold}: train/test 的 sample_id 重叠。")
        if test_ids != predicted_ids:
            raise ValueError(f"fold={fold}: split manifest 和 OOF 测试 sample_id 不一致。")
        if (train_ids | test_ids) - set(data.index):
            raise ValueError(f"fold={fold}: manifest 的样本在数据集中找不到。")
        if set(data.loc[list(train_ids), "paper_id"]) & set(data.loc[list(test_ids), "paper_id"]):
            raise ValueError(f"fold={fold}: 训练/测试存在论文重叠。")

        normalizer = train_normalizer(data.loc[list(train_ids)])
        test_data = data.loc[test_prediction["sample_id"].tolist()].copy()
        parts = normalized_components(test_data, normalizer)
        recon = composite(parts, COMPONENTS)
        true_stored = test_prediction["y_true"].to_numpy(float)
        delta = np.abs(recon.to_numpy(float) - true_stored)
        bad = ~np.isfinite(delta) | (delta > args.tolerance)
        audit_rows.append({"fold": fold, "n_train": len(train_ids),
                           "n_test": len(test_ids), "n_papers_train": data.loc[list(train_ids), "paper_id"].nunique(),
                           "n_papers_test": test_data["paper_id"].nunique(),
                           "n_label_mismatch": int(bad.sum()),
                           "max_absolute_label_difference": float(np.nanmax(delta)) if np.isfinite(delta).any() else np.nan,
                           "igid_train_min": normalizer["igid"][0],
                           "igid_train_max": normalizer["igid"][1],
                           "bundle_log_train_min": normalizer["bundle"][0],
                           "bundle_log_train_max": normalizer["bundle"][1]})
        if bad.any():
            raise RuntimeError(
                f"fold={fold}: {bad.sum()} 条真实 Composite 无法按训练折归一化复现；"
                f"最大差异 {np.nanmax(delta):.6g}。已停止，不能在不同标签定义上做稳健性结论。"
                "请核对当前数据库、L12 模型及 split manifest 是否来自同一次训练。"
            )
        actual_pattern = parts.notna().apply(
            lambda row: "+".join(k for k in COMPONENTS if row[k]) or "none", axis=1
        )
        full = test_prediction.copy().reset_index(drop=True)
        full["reported_pattern"] = test_data["reported_pattern"].to_numpy()
        full["scored_pattern"] = actual_pattern.to_numpy()
        full["reported_count"] = test_data["reported_count"].to_numpy()
        full["scored_count"] = parts.notna().sum(axis=1).to_numpy()
        full["data_level"] = test_data["data_level"].to_numpy()
        fold_frames.append(full)

        complete = parts.loc[parts.notna().all(axis=1)]
        original = composite(complete, COMPONENTS).to_numpy(float)
        if len(complete) < 10:
            continue
        for pattern in candidate_masks:
            included = tuple(pattern.split("+"))
            masked = composite(complete, included).to_numpy(float)
            if not np.isfinite(masked).all():
                continue
            mask_rows.append({
                "fold": fold, "retained_components": pattern,
                "n_complete_samples": len(original),
                "n_papers": data.loc[complete.index, "paper_id"].nunique(),
                "score_mae_vs_full": float(np.mean(np.abs(masked - original))),
                "score_median_abs_change": float(np.median(np.abs(masked - original))),
                "score_mean_signed_change": float(np.mean(masked - original)),
                "score_max_abs_change": float(np.max(np.abs(masked - original))),
                "spearman_rank_vs_full": spearman(original, masked),
                "top10pct_jaccard": top_overlap(original, masked, .10),
                "top20pct_jaccard": top_overlap(original, masked, .20),
            })
            mask_sample_rows.extend({
                "sample_id": sample_id, "fold": fold, "retained_components": pattern,
                "score_full": float(orig), "score_masked": float(mask),
                "score_delta": float(mask - orig)
            } for sample_id, orig, mask in zip(complete.index, original, masked))

    all_oof = pd.concat(fold_frames, ignore_index=True)
    eval_rows = []
    for scope, part in (("L1+L2", all_oof),
                        ("L1", all_oof.loc[all_oof["data_level"] == "L1"]),
                        ("L2", all_oof.loc[all_oof["data_level"] == "L2"])):
        for pattern, group in part.groupby("scored_pattern"):
            result = {"scope": scope, "pattern": pattern, **metrics(group)}
            # Interpret tiny groups descriptively; withhold unstable metric estimates.
            if len(group) < 10 or group["paper_id"].nunique() < 3:
                result["r2"] = np.nan
                result["spearman"] = np.nan
                result["mae"] = np.nan
                result["mean_signed_error"] = np.nan
            result.update(paper_bootstrap(group, args.bootstrap, rng))
            eval_rows.append(result)

    masking = pd.DataFrame(mask_rows)
    if not masking.empty:
        masking_summary = masking.groupby("retained_components", as_index=False).agg(
            n_folds=("fold", "nunique"),
            n_complete_test_samples_across_folds=("n_complete_samples", "sum"),
            median_score_mae=("score_mae_vs_full", "median"),
            median_rank_spearman=("spearman_rank_vs_full", "median"),
            median_top10pct_jaccard=("top10pct_jaccard", "median"),
            median_top20pct_jaccard=("top20pct_jaccard", "median"),
        )
    else:
        masking_summary = pd.DataFrame(columns=[
            "retained_components", "n_folds", "n_complete_test_samples_across_folds",
            "median_score_mae", "median_rank_spearman",
            "median_top10pct_jaccard", "median_top20pct_jaccard"])

    out.mkdir(parents=True, exist_ok=True)
    dist.to_csv(out / "composite_pattern_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eval_rows).sort_values(["scope", "n_samples"], ascending=[True, False]).to_csv(
        out / "composite_pattern_oof_metrics.csv", index=False, encoding="utf-8-sig")
    masking.to_csv(out / "composite_masking_rank_sensitivity.csv", index=False, encoding="utf-8-sig")
    masking_summary.to_csv(out / "composite_robustness_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mask_sample_rows).to_csv(out / "composite_masking_sample_detail.csv", index=False, encoding="utf-8-sig")
    all_oof.to_csv(out / "composite_pattern_oof_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(out / "composite_label_reproduction_audit.csv", index=False, encoding="utf-8-sig")
    meta = {"source_dataset": str(data_path), "source_oof": str(pred_path),
            "source_split_manifest": str(split_path), "model_retrained": False,
            "imputation_performed": False, "multitask_comparison_performed": False,
            "validation_protocol": SPLIT, "n_l1_l2": len(data),
            "n_eligible_ge2": int((data["reported_count"] >= 2).sum()),
            "n_full5_l1_l2": int((data["reported_count"] == 5).sum()),
            "n_oof": len(all_oof), "masks_tested": candidate_masks,
            "label_reproduction_tolerance": args.tolerance,
            "warning": "Masking is a label-only sensitivity test on complete cases; "
                       "it is not retrained-model performance, not imputation, "
                       "and does not prove missingness is random."}
    plot_robustness(out)
    if args.compare_models:
        run_model_comparisons(root, data, oof, manifest, out, args)
        meta["imputation_performed"] = True
        meta["multitask_comparison_performed"] = True
        meta["baseline_models_trained"] = True
    else:
        meta["baseline_models_trained"] = False
    (out / "composite_robustness_run_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] Composite robustness audit finished (official models unchanged; "
          + ("3 evaluation baselines trained)" if args.compare_models else "no model fitting)"))
    print(f"Samples L1+L2: {len(data)}, >=2: {meta['n_eligible_ge2']}, all 5: {meta['n_full5_l1_l2']}")
    print(f"Paper-grouped OOF samples: {len(all_oof)}; reporting masks: {len(candidate_masks)}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
