# -*- coding: utf-8 -*-
"""
train_l1_composite.py

L1 综合评分模型：
    0.30 × growth_quality
  + 0.20 × IG/ID
  + 0.20 × orientation
  + 0.15 × purity
  + 0.15 × bundle_density

完成：
1. random_holdout
2. paper_grouped_cv
3. temporal_holdout
4. 外层测试集不参与归一化或 early stopping
5. 使用全部有效 L1 数据训练最终模型

输入：
    data/11_flattened/ml_dataset_3p0_full60D.csv
    data/11_flattened/feature_schema.json

输出：
    outputs/models/l1_composite/
"""

import json
import math
import os
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


# =============================================================================
# 1. 路径
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "11_flattened")
CSV_PATH = os.path.join(DATA_DIR, "ml_dataset_3p0_full60D.csv")
SCHEMA_PATH = os.path.join(DATA_DIR, "feature_schema.json")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "models", "l1_composite")

FINAL_MODEL_DIR = os.path.join(OUT_DIR, "final_models")
VALIDATION_MODEL_DIR = os.path.join(OUT_DIR, "validation_models")

METRICS_FOLD_CSV = os.path.join(OUT_DIR, "L1_composite_metrics_fold.csv")
METRICS_SUMMARY_CSV = os.path.join(OUT_DIR, "L1_composite_metrics_summary.csv")
PREDICTIONS_CSV = os.path.join(OUT_DIR, "L1_composite_predictions.csv")
SPLIT_MANIFEST_CSV = os.path.join(OUT_DIR, "composite_split_manifest.csv")
NORMALIZATION_CSV = os.path.join(OUT_DIR, "composite_normalization_manifest.csv")
COMPONENT_COVERAGE_CSV = os.path.join(OUT_DIR, "composite_component_coverage.csv")
FINAL_NORMALIZATION_JSON = os.path.join(OUT_DIR, "final_composite_normalization.json")
FINAL_MODEL_MANIFEST_CSV = os.path.join(OUT_DIR, "final_model_manifest.csv")
TRAINING_SUMMARY_JSON = os.path.join(OUT_DIR, "training_summary_composite.json")


# =============================================================================
# 2. 训练设置
# =============================================================================

ONLY_TRAIN_LEVEL = "L1"
SEED = 42

RANDOM_TEST_SIZE = 0.20
INNER_VALID_SIZE = 0.15
GROUP_CV_SPLITS = 5

MIN_TOTAL_SAMPLES = 50
MIN_TOTAL_PAPERS = 10
MIN_OUTER_TRAIN_SAMPLES = 40
MIN_OUTER_TEST_SAMPLES = 10

MIN_TEMPORAL_TRAIN_SAMPLES = 50
MIN_TEMPORAL_TEST_SAMPLES = 20
MIN_TEMPORAL_TRAIN_PAPERS = 5
MIN_TEMPORAL_TEST_PAPERS = 3

MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
DEFAULT_FINAL_ITERATIONS = 500
SAVE_VALIDATION_MODELS = True


# =============================================================================
# 3. 综合评分
# =============================================================================

TARGET_COMPOSITE = "y_composite_final_score"

WEIGHTS: Dict[str, float] = {
    "gq": 0.30,
    "igid": 0.20,
    "ori": 0.20,
    "purity": 0.15,
    "bundle": 0.15,
}

MISSING_POLICY = "renorm"       # "renorm" 或 "strict"
MIN_COMPONENT_COUNT = 2        # 后续可根据覆盖统计改为 2 或 3

COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

COL_IGID_PREF = "y_ig_id_ratio"
COL_IGID_LEGACY = "y_IG_ID"
COL_IGID_UNIFIED = "y_ig_id_ratio_unified"

RAW_GQ = "_raw_gq"
RAW_IGID = "_raw_igid"
RAW_ORI = "_raw_ori"
RAW_PURITY = "_raw_purity"
RAW_BUNDLE = "_raw_bundle"
RAW_BUNDLE_LOG = "_raw_bundle_log1p"
RAW_COMPONENT_COUNT = "_raw_component_count"

COMPONENT_COLUMNS = {
    "gq": "gq_01",
    "igid": "igid_norm01",
    "ori": "ori_01",
    "purity": "purity_01",
    "bundle": "bundle_norm01",
}


# =============================================================================
# 4. LightGBM
# =============================================================================

LGB_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": ["l2", "l1"],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
}


# =============================================================================
# 5. 工具函数
# =============================================================================

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_feature_cols() -> List[str]:
    base = (
        [f"N{i}" for i in range(1, 19)]
        + [f"C{i}" for i in range(1, 13)]
        + [f"F{i}" for i in range(1, 19)]
        + [f"P{i}" for i in range(1, 13)]
    )
    return base + [f"miss_{column}" for column in base]


def load_feature_cols() -> Tuple[List[str], str]:
    if os.path.exists(SCHEMA_PATH):
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
            columns = schema.get("model_feature_columns")
            if isinstance(columns, list) and columns and all(isinstance(c, str) for c in columns):
                return columns, "feature_schema.json"
        except Exception as error:
            print(f"[WARN] feature_schema.json 读取失败，使用代码内定义：{error}")
    return build_feature_cols(), "code_fallback"


def ensure_igid_unified(df: pd.DataFrame) -> pd.DataFrame:
    preferred = (
        to_num(df[COL_IGID_PREF])
        if COL_IGID_PREF in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    legacy = (
        to_num(df[COL_IGID_LEGACY])
        if COL_IGID_LEGACY in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    out = df.copy()
    out[COL_IGID_UNIFIED] = preferred.where(preferred.notna(), legacy)
    return out


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    true_rank = pd.Series(y_true).rank(method="average")
    pred_rank = pd.Series(y_pred).rank(method="average")
    value = true_rank.corr(pred_rank, method="pearson")
    return float(value) if pd.notna(value) else float("nan")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": safe_r2(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
    }


def minmax_parameters(series: pd.Series) -> Dict[str, Any]:
    values = to_num(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return {
            "minimum": None,
            "maximum": None,
            "usable": False,
            "finite_count": 0,
            "reason": "no_finite_values",
        }

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))

    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        return {
            "minimum": minimum,
            "maximum": maximum,
            "usable": False,
            "finite_count": int(len(finite)),
            "reason": "constant_values",
        }

    return {
        "minimum": minimum,
        "maximum": maximum,
        "usable": True,
        "finite_count": int(len(finite)),
        "reason": "ok",
    }


def apply_minmax(
    series: pd.Series,
    minimum: Optional[float],
    maximum: Optional[float],
    usable: bool,
) -> pd.Series:
    if (
        not usable
        or minimum is None
        or maximum is None
        or math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15)
    ):
        return pd.Series(np.nan, index=series.index, dtype=float)

    normalized = (to_num(series) - minimum) / (maximum - minimum)
    return normalized.clip(lower=0.0, upper=1.0)


# =============================================================================
# 6. 基础数据
# =============================================================================

def prepare_base_data(df_all: pd.DataFrame, x_cols: Sequence[str]) -> Dict[str, Any]:
    work = ensure_igid_unified(df_all).copy()

    for column in [COL_GQ, COL_ORI, COL_PURITY, COL_BUNDLE, COL_IGID_UNIFIED]:
        if column not in work.columns:
            work[column] = np.nan

    work[RAW_GQ] = to_num(work[COL_GQ])
    work[RAW_IGID] = to_num(work[COL_IGID_UNIFIED])
    work[RAW_ORI] = to_num(work[COL_ORI])
    work[RAW_PURITY] = to_num(work[COL_PURITY])
    work[RAW_BUNDLE] = to_num(work[COL_BUNDLE])

    bundle = work[RAW_BUNDLE].to_numpy(dtype=float)
    work[RAW_BUNDLE_LOG] = np.where(
        np.isfinite(bundle) & (bundle >= 0.0),
        np.log1p(bundle),
        np.nan,
    )

    availability = pd.DataFrame({
        "gq": work[RAW_GQ].notna(),
        "igid": work[RAW_IGID].notna(),
        "ori": work[RAW_ORI].notna(),
        "purity": work[RAW_PURITY].notna(),
        "bundle": work[RAW_BUNDLE_LOG].notna(),
    })

    work[RAW_COMPONENT_COUNT] = availability.sum(axis=1).astype(int)
    work = work[work[RAW_COMPONENT_COUNT] >= MIN_COMPONENT_COUNT].copy()
    work = work.reset_index(drop=True)

    defaults = {
        "sample_id": "",
        "paper_id": "",
        "paper_year_int": np.nan,
        "growth_mode": "",
        "data_level": "",
        "source_file": "",
        "synthesis_id": "",
    }
    for column, default in defaults.items():
        if column not in work.columns:
            work[column] = default

    work["sample_id"] = work["sample_id"].fillna("").astype(str)
    work["paper_id"] = work["paper_id"].fillna("").astype(str)
    work["paper_year_int"] = pd.to_numeric(work["paper_year_int"], errors="coerce")

    X = work[list(x_cols)].copy()
    for column in x_cols:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.reset_index(drop=True)

    return {
        "work": work,
        "X": X,
        "samples": int(len(work)),
        "papers": int(work["paper_id"].nunique()),
        "dated_samples": int(work["paper_year_int"].notna().sum()),
    }


# =============================================================================
# 7. 综合分数归一化
# =============================================================================

def fit_composite_normalizer(
    work: pd.DataFrame,
    train_indices: np.ndarray,
) -> Dict[str, Any]:
    train = work.iloc[np.asarray(train_indices, dtype=int)]
    igid = minmax_parameters(train[RAW_IGID])
    bundle = minmax_parameters(train[RAW_BUNDLE_LOG])

    return {
        "igid_min": igid["minimum"],
        "igid_max": igid["maximum"],
        "igid_usable": igid["usable"],
        "igid_finite_count": igid["finite_count"],
        "igid_reason": igid["reason"],
        "bundle_log_min": bundle["minimum"],
        "bundle_log_max": bundle["maximum"],
        "bundle_usable": bundle["usable"],
        "bundle_finite_count": bundle["finite_count"],
        "bundle_reason": bundle["reason"],
        "fit_sample_count": int(len(train)),
        "fit_paper_count": int(train["paper_id"].nunique()),
    }


def transform_composite(
    work: pd.DataFrame,
    normalizer: Dict[str, Any],
) -> pd.DataFrame:
    out = pd.DataFrame(index=work.index)

    out["gq_01"] = (to_num(work[RAW_GQ]) / 10.0).clip(0.0, 1.0)

    out["igid_norm01"] = apply_minmax(
        work[RAW_IGID],
        normalizer["igid_min"],
        normalizer["igid_max"],
        bool(normalizer["igid_usable"]),
    )

    out["ori_01"] = to_num(work[RAW_ORI]).clip(0.0, 1.0)
    out["purity_01"] = to_num(work[RAW_PURITY]).clip(0.0, 1.0)

    out["bundle_norm01"] = apply_minmax(
        work[RAW_BUNDLE_LOG],
        normalizer["bundle_log_min"],
        normalizer["bundle_log_max"],
        bool(normalizer["bundle_usable"]),
    )

    components = pd.DataFrame(
        {name: out[column] for name, column in COMPONENT_COLUMNS.items()},
        index=work.index,
    )
    weights = pd.Series(WEIGHTS, dtype=float)

    if not np.isfinite(weights.values).all() or weights.sum() <= 0:
        raise ValueError("综合评分权重必须为有限数值，且权重和必须大于0。")

    component_count = components.notna().sum(axis=1).astype(int)

    if MISSING_POLICY == "strict":
        available = components.notna().all(axis=1)
        composite = pd.Series(np.nan, index=work.index, dtype=float)
        composite.loc[available] = components.loc[available].mul(weights, axis=1).sum(axis=1)
        weight_sum = pd.Series(np.nan, index=work.index, dtype=float)
        weight_sum.loc[available] = float(weights.sum())

    elif MISSING_POLICY == "renorm":
        available = components.notna()
        weight_sum = available.mul(weights, axis=1).sum(axis=1)
        numerator = components.mul(weights, axis=1).sum(axis=1, skipna=True)
        composite = pd.Series(
            np.where(weight_sum > 0, numerator / weight_sum, np.nan),
            index=work.index,
            dtype=float,
        )

    else:
        raise ValueError("MISSING_POLICY 必须为 'renorm' 或 'strict'。")

    composite = composite.where(component_count >= MIN_COMPONENT_COUNT, np.nan)

    out[TARGET_COMPOSITE] = composite
    out["composite_component_count"] = component_count
    out["composite_weight_sum_used"] = weight_sum.where(composite.notna(), np.nan)

    return out


# =============================================================================
# 8. 内部验证和训练
# =============================================================================

def make_inner_validation_split(
    outer_train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, str]:
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    groups = (
        groups_all.iloc[outer_train_indices]
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )

    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=INNER_VALID_SIZE,
            random_state=SEED,
        )
        rel_fit, rel_valid = next(
            splitter.split(
                np.zeros(len(outer_train_indices)),
                groups=groups.to_numpy(),
            )
        )
        return (
            outer_train_indices[rel_fit],
            outer_train_indices[rel_valid],
            "group_shuffle",
        )

    rel_indices = np.arange(len(outer_train_indices))
    rel_fit, rel_valid = train_test_split(
        rel_indices,
        test_size=INNER_VALID_SIZE,
        random_state=SEED,
    )
    return (
        outer_train_indices[rel_fit],
        outer_train_indices[rel_valid],
        "random_fallback",
    )


def fit_lightgbm_with_inner_valid(
    X: pd.DataFrame,
    y: np.ndarray,
    outer_train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[lgb.Booster, int, str]:
    fit_idx, valid_idx, method = make_inner_validation_split(
        outer_train_indices,
        groups_all,
    )

    train_set = lgb.Dataset(
        X.iloc[fit_idx],
        label=y[fit_idx],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X.iloc[valid_idx],
        label=y[valid_idx],
        reference=train_set,
        free_raw_data=False,
    )

    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[train_set, valid_set],
        valid_names=["inner_fit", "inner_valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = int(
        booster.best_iteration
        if booster.best_iteration and booster.best_iteration > 0
        else MAX_BOOST_ROUNDS
    )
    return booster, best_iteration, method


def save_validation_model(
    booster: lgb.Booster,
    split_type: str,
    fold: int,
) -> Optional[str]:
    if not SAVE_VALIDATION_MODELS:
        return None

    directory = os.path.join(VALIDATION_MODEL_DIR, split_type)
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(
        directory,
        f"L1_composite_final_score_fold{fold}.txt",
    )
    booster.save_model(path)
    return path


# =============================================================================
# 9. 外层划分
# =============================================================================

def prepare_outer_split(
    base: Dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Dict[str, Any]:
    work = base["work"]
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    normalizer = fit_composite_normalizer(work, train_indices)
    transformed = transform_composite(work, normalizer)

    y = pd.to_numeric(
        transformed[TARGET_COMPOSITE],
        errors="coerce",
    ).to_numpy(dtype=float)

    train_indices = train_indices[np.isfinite(y[train_indices])]
    test_indices = test_indices[np.isfinite(y[test_indices])]

    return {
        "normalizer": normalizer,
        "transformed": transformed,
        "target_values": y,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


def outer_split_is_usable(prepared: Dict[str, Any]) -> bool:
    return (
        len(prepared["train_indices"]) >= MIN_OUTER_TRAIN_SAMPLES
        and len(prepared["test_indices"]) >= MIN_OUTER_TEST_SAMPLES
    )


# =============================================================================
# 10. 输出行
# =============================================================================

def prediction_rows(
    split_type: str,
    fold: int,
    test_indices: np.ndarray,
    work: pd.DataFrame,
    transformed: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cutoff: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for position, index in enumerate(np.asarray(test_indices, dtype=int)):
        item = work.iloc[index]
        components = transformed.iloc[index]

        true_value = float(y_true[position])
        pred_value = float(y_pred[position])
        residual = pred_value - true_value

        rows.append({
            "sample_id": item["sample_id"],
            "paper_id": item["paper_id"],
            "paper_year_int": item["paper_year_int"],
            "growth_mode": item["growth_mode"],
            "data_level": item["data_level"],
            "source_file": item["source_file"],
            "synthesis_id": item["synthesis_id"],
            "target": TARGET_COMPOSITE,
            "split_type": split_type,
            "fold": fold,
            "temporal_cutoff_year": cutoff,
            "y_true": true_value,
            "y_pred": pred_value,
            "residual": residual,
            "absolute_error": abs(residual),
            "component_count": int(components["composite_component_count"]),
            "weight_sum_used": float(components["composite_weight_sum_used"]),
            "gq_01": components["gq_01"],
            "igid_norm01": components["igid_norm01"],
            "ori_01": components["ori_01"],
            "purity_01": components["purity_01"],
            "bundle_norm01": components["bundle_norm01"],
        })

    return rows


def split_rows(
    split_type: str,
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    work: pd.DataFrame,
    cutoff: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for role, indices in (("train", train_indices), ("test", test_indices)):
        for index in np.asarray(indices, dtype=int):
            item = work.iloc[index]
            rows.append({
                "sample_id": item["sample_id"],
                "paper_id": item["paper_id"],
                "paper_year_int": item["paper_year_int"],
                "growth_mode": item["growth_mode"],
                "data_level": item["data_level"],
                "target": TARGET_COMPOSITE,
                "split_type": split_type,
                "fold": fold,
                "role": role,
                "temporal_cutoff_year": cutoff,
            })

    return rows


def metric_row(
    split_type: str,
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    work: pd.DataFrame,
    metrics: Dict[str, float],
    best_iteration: int,
    inner_method: str,
    cutoff: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "target": TARGET_COMPOSITE,
        "split_type": split_type,
        "fold": fold,
        "status": "ok",
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "n_train_papers": int(work.iloc[train_indices]["paper_id"].nunique()),
        "n_test_papers": int(work.iloc[test_indices]["paper_id"].nunique()),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "spearman": metrics["spearman"],
        "best_iteration": int(best_iteration),
        "inner_split_method": inner_method,
        "temporal_cutoff_year": cutoff,
    }


def normalization_row(
    split_type: str,
    fold: int,
    normalizer: Dict[str, Any],
    n_train: int,
    n_test: int,
    cutoff: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "split_type": split_type,
        "fold": fold,
        "temporal_cutoff_year": cutoff,
        "n_train": int(n_train),
        "n_test": int(n_test),
        **normalizer,
    }


# =============================================================================
# 11. 单个外层split
# =============================================================================

def run_one_outer_split(
    base: Dict[str, Any],
    split_type: str,
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    cutoff: Optional[int] = None,
) -> Dict[str, Any]:
    work = base["work"]
    X = base["X"]

    prepared = prepare_outer_split(base, train_indices, test_indices)

    if not outer_split_is_usable(prepared):
        return {
            "status": "skipped_too_few_valid_composite_targets",
            "metrics": [],
            "predictions": [],
            "splits": [],
            "normalizations": [],
            "best_iterations": [],
        }

    train_idx = prepared["train_indices"]
    test_idx = prepared["test_indices"]
    y = prepared["target_values"]

    booster, best_iter, inner_method = fit_lightgbm_with_inner_valid(
        X,
        y,
        train_idx,
        work["paper_id"],
    )

    y_pred = booster.predict(X.iloc[test_idx], num_iteration=best_iter)
    y_true = y[test_idx]
    metrics = evaluate_predictions(y_true, y_pred)

    row = metric_row(
        split_type,
        fold,
        train_idx,
        test_idx,
        work,
        metrics,
        best_iter,
        inner_method,
        cutoff,
    )
    row["model_path"] = save_validation_model(
        booster,
        split_type,
        fold,
    )

    return {
        "status": "ok",
        "metrics": [row],
        "predictions": prediction_rows(
            split_type,
            fold,
            test_idx,
            work,
            prepared["transformed"],
            y_true,
            y_pred,
            cutoff,
        ),
        "splits": split_rows(
            split_type,
            fold,
            train_idx,
            test_idx,
            work,
            cutoff,
        ),
        "normalizations": [
            normalization_row(
                split_type,
                fold,
                prepared["normalizer"],
                len(train_idx),
                len(test_idx),
                cutoff,
            )
        ],
        "best_iterations": [best_iter],
    }


# =============================================================================
# 12. 三类验证
# =============================================================================

def run_random_validation(base: Dict[str, Any]) -> Dict[str, Any]:
    all_indices = np.arange(len(base["work"]))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=RANDOM_TEST_SIZE,
        random_state=SEED,
    )
    return run_one_outer_split(
        base,
        "random_holdout",
        1,
        train_idx,
        test_idx,
    )


def run_grouped_validation(base: Dict[str, Any]) -> Dict[str, Any]:
    work = base["work"]
    X = base["X"]
    groups = work["paper_id"].fillna("").astype(str)

    if groups.nunique() < GROUP_CV_SPLITS:
        return {
            "status": "skipped_too_few_papers",
            "metrics": [],
            "predictions": [],
            "splits": [],
            "normalizations": [],
            "best_iterations": [],
        }

    splitter = GroupKFold(n_splits=GROUP_CV_SPLITS)

    metrics: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    splits: List[Dict[str, Any]] = []
    normalizations: List[Dict[str, Any]] = []
    iterations: List[int] = []
    statuses: List[str] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, groups=groups),
        start=1,
    ):
        result = run_one_outer_split(
            base,
            "paper_grouped_cv",
            fold,
            train_idx,
            test_idx,
        )
        statuses.append(result["status"])
        metrics.extend(result.get("metrics", []))
        predictions.extend(result.get("predictions", []))
        splits.extend(result.get("splits", []))
        normalizations.extend(result.get("normalizations", []))
        iterations.extend(result.get("best_iterations", []))

    return {
        "status": "ok" if all(status == "ok" for status in statuses) else "partial",
        "fold_statuses": statuses,
        "metrics": metrics,
        "predictions": predictions,
        "splits": splits,
        "normalizations": normalizations,
        "best_iterations": iterations,
    }


def temporal_candidates(work: pd.DataFrame) -> List[int]:
    dated = work[work["paper_year_int"].notna()]
    years = sorted(int(year) for year in dated["paper_year_int"].unique())
    candidates: List[Tuple[float, int]] = []

    for cutoff in years[:-1]:
        train = dated[dated["paper_year_int"] <= cutoff]
        test = dated[dated["paper_year_int"] > cutoff]

        if len(train) < MIN_TEMPORAL_TRAIN_SAMPLES:
            continue
        if len(test) < MIN_TEMPORAL_TEST_SAMPLES:
            continue
        if train["paper_id"].nunique() < MIN_TEMPORAL_TRAIN_PAPERS:
            continue
        if test["paper_id"].nunique() < MIN_TEMPORAL_TEST_PAPERS:
            continue

        test_fraction = len(test) / (len(train) + len(test))
        candidates.append((abs(test_fraction - RANDOM_TEST_SIZE), cutoff))

    candidates.sort(key=lambda item: (item[0], -item[1]))
    return [int(cutoff) for _, cutoff in candidates]


def run_temporal_validation(base: Dict[str, Any]) -> Dict[str, Any]:
    work = base["work"]
    candidates = temporal_candidates(work)

    if not candidates:
        return {
            "status": "skipped_too_few_dated_samples",
            "metrics": [],
            "predictions": [],
            "splits": [],
            "normalizations": [],
            "best_iterations": [],
            "cutoff": None,
        }

    dated = work["paper_year_int"].notna()

    for cutoff in candidates:
        train_idx = np.where(
            (dated & (work["paper_year_int"] <= cutoff)).to_numpy()
        )[0]
        test_idx = np.where(
            (dated & (work["paper_year_int"] > cutoff)).to_numpy()
        )[0]

        result = run_one_outer_split(
            base,
            "temporal_holdout",
            1,
            train_idx,
            test_idx,
            cutoff,
        )

        if result["status"] == "ok":
            result["cutoff"] = cutoff
            result["metrics"][0]["temporal_train_year_max"] = cutoff
            result["metrics"][0]["temporal_test_year_min"] = int(
                work.iloc[test_idx]["paper_year_int"].min()
            )
            return result

    return {
        "status": "skipped_too_few_valid_composite_targets",
        "metrics": [],
        "predictions": [],
        "splits": [],
        "normalizations": [],
        "best_iterations": [],
        "cutoff": None,
    }


# =============================================================================
# 13. 最终模型
# =============================================================================

def choose_final_iterations(
    grouped_iterations: Sequence[int],
    random_iterations: Sequence[int],
    temporal_iterations: Sequence[int],
) -> Tuple[int, str]:
    grouped = [
        int(value)
        for value in grouped_iterations
        if value is not None and value > 0
    ]
    if grouped:
        return max(1, int(round(median(grouped)))), "median_paper_grouped_cv"

    fallback = [
        int(value)
        for value in list(random_iterations) + list(temporal_iterations)
        if value is not None and value > 0
    ]
    if fallback:
        return max(1, int(round(median(fallback)))), "median_available_validation"

    return DEFAULT_FINAL_ITERATIONS, "default"


def train_final_model(
    base: Dict[str, Any],
    x_cols: Sequence[str],
    final_iterations: int,
    iteration_source: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    work = base["work"]
    X = base["X"]
    all_indices = np.arange(len(work))

    normalizer = fit_composite_normalizer(work, all_indices)
    transformed = transform_composite(work, normalizer)

    y = pd.to_numeric(
        transformed[TARGET_COMPOSITE],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_idx = np.where(np.isfinite(y))[0]

    if len(valid_idx) < MIN_TOTAL_SAMPLES:
        raise RuntimeError(
            f"最终综合评分有效样本过少：{len(valid_idx)} < {MIN_TOTAL_SAMPLES}"
        )

    train_set = lgb.Dataset(
        X.iloc[valid_idx],
        label=y[valid_idx],
        free_raw_data=False,
    )

    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=train_set,
        num_boost_round=final_iterations,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(
        FINAL_MODEL_DIR,
        "L1_composite_final_score.txt",
    )
    booster.save_model(model_path)

    model_manifest = {
        "target": TARGET_COMPOSITE,
        "status": "ok",
        "samples": int(len(valid_idx)),
        "papers": int(work.iloc[valid_idx]["paper_id"].nunique()),
        "feature_count": int(len(x_cols)),
        "final_iterations": int(final_iterations),
        "iteration_source": iteration_source,
        "missing_policy": MISSING_POLICY,
        "minimum_component_count": MIN_COMPONENT_COUNT,
        "model_path": model_path,
    }

    final_normalization = {
        "target": TARGET_COMPOSITE,
        "normalization_scope": "all_valid_L1_rows",
        "normalizer": normalizer,
        "weights": WEIGHTS,
        "weight_sum": float(sum(WEIGHTS.values())),
        "missing_policy": MISSING_POLICY,
        "minimum_component_count": MIN_COMPONENT_COUNT,
        "component_columns": {
            "growth_quality": COL_GQ,
            "igid_preferred": COL_IGID_PREF,
            "igid_legacy": COL_IGID_LEGACY,
            "igid_unified": COL_IGID_UNIFIED,
            "orientation": COL_ORI,
            "purity": COL_PURITY,
            "bundle_density": COL_BUNDLE,
        },
        "transformations": {
            "growth_quality": "clip(y_growth_quality_score / 10, 0, 1)",
            "igid": "training-scope minmax then clip to [0,1]",
            "orientation": "clip(y_orientation_score, 0, 1)",
            "purity": "clip(y_purity_score, 0, 1)",
            "bundle_density": "log1p(y_bundle_density_per_um for y>=0), training-scope minmax, clip to [0,1]",
        },
        "final_training_samples": int(len(valid_idx)),
        "final_training_papers": int(work.iloc[valid_idx]["paper_id"].nunique()),
    }

    return model_manifest, final_normalization, transformed


# =============================================================================
# 14. 汇总
# =============================================================================

def summarize_metrics(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for split_type, group in metrics_df.groupby("split_type", dropna=False):
        pred_group = predictions_df[predictions_df["split_type"] == split_type]

        pooled = (
            evaluate_predictions(
                pred_group["y_true"].to_numpy(dtype=float),
                pred_group["y_pred"].to_numpy(dtype=float),
            )
            if len(pred_group) > 0
            else {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "spearman": np.nan}
        )

        rows.append({
            "target": TARGET_COMPOSITE,
            "split_type": split_type,
            "fold_count": int(len(group)),
            "total_test_predictions": int(len(pred_group)),
            "mae_mean": group["mae"].mean(),
            "mae_std": group["mae"].std(ddof=1) if len(group) > 1 else np.nan,
            "rmse_mean": group["rmse"].mean(),
            "rmse_std": group["rmse"].std(ddof=1) if len(group) > 1 else np.nan,
            "r2_mean": group["r2"].mean(),
            "r2_std": group["r2"].std(ddof=1) if len(group) > 1 else np.nan,
            "spearman_mean": group["spearman"].mean(),
            "spearman_std": group["spearman"].std(ddof=1) if len(group) > 1 else np.nan,
            "pooled_mae": pooled["mae"],
            "pooled_rmse": pooled["rmse"],
            "pooled_r2": pooled["r2"],
            "pooled_spearman": pooled["spearman"],
            "best_iteration_median": int(
                round(median(group["best_iteration"].astype(int).tolist()))
            ),
            "temporal_cutoff_year": (
                int(group["temporal_cutoff_year"].dropna().iloc[0])
                if (
                    "temporal_cutoff_year" in group.columns
                    and not group["temporal_cutoff_year"].dropna().empty
                )
                else ""
            ),
        })

    return pd.DataFrame(rows)


def build_component_coverage(
    work: pd.DataFrame,
    final_transformed: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(work)

    for stage, counts in (
        ("raw_components_before_normalization", work[RAW_COMPONENT_COUNT]),
        ("final_components_after_normalization", final_transformed["composite_component_count"]),
    ):
        for component_count, count in counts.value_counts(dropna=False).sort_index().items():
            rows.append({
                "stage": stage,
                "component_count": component_count,
                "sample_count": int(count),
                "fraction": float(count) / total if total > 0 else np.nan,
            })

    valid = final_transformed[TARGET_COMPOSITE].notna()

    rows.extend([
        {
            "stage": "final_composite_validity",
            "component_count": "valid",
            "sample_count": int(valid.sum()),
            "fraction": float(valid.mean()) if total > 0 else np.nan,
        },
        {
            "stage": "final_composite_validity",
            "component_count": "invalid",
            "sample_count": int((~valid).sum()),
            "fraction": float((~valid).mean()) if total > 0 else np.nan,
        },
    ])

    return pd.DataFrame(rows)


# =============================================================================
# 15. 主程序
# =============================================================================

def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 不存在：{CSV_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(VALIDATION_MODEL_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH, low_memory=False)

    if "data_level" not in df.columns:
        raise RuntimeError("CSV 缺少 data_level，无法筛选 L1。")

    df = df[df["data_level"].astype(str) == ONLY_TRAIN_LEVEL].copy()

    if df.empty:
        raise RuntimeError("筛选 data_level == L1 后没有数据。")

    x_cols, feature_source = load_feature_cols()

    missing_x = [column for column in x_cols if column not in df.columns]
    if missing_x:
        raise RuntimeError(
            f"缺少模型特征列 ({len(missing_x)})，示例：{missing_x[:10]}"
        )

    required_metadata = ["paper_id", "sample_id", "paper_year_int"]
    missing_metadata = [
        column for column in required_metadata if column not in df.columns
    ]
    if missing_metadata:
        raise RuntimeError(
            f"当前 CSV 不是新版 flatten 输出，缺少：{missing_metadata}"
        )

    base = prepare_base_data(df, x_cols)

    if base["samples"] < MIN_TOTAL_SAMPLES:
        raise RuntimeError(
            f"综合评分可用样本过少：{base['samples']} < {MIN_TOTAL_SAMPLES}"
        )

    if base["papers"] < MIN_TOTAL_PAPERS:
        raise RuntimeError(
            f"综合评分可用论文过少：{base['papers']} < {MIN_TOTAL_PAPERS}"
        )

    print("=" * 88)
    print("[L1 COMPOSITE VALIDATION START]")
    print(f"CSV                : {CSV_PATH}")
    print(f"L1 rows            : {len(df)}")
    print(f"Composite rows     : {base['samples']}")
    print(f"Composite papers   : {base['papers']}")
    print(f"Dated rows         : {base['dated_samples']}")
    print(f"Feature source     : {feature_source}")
    print(f"Feature count      : {len(x_cols)}")
    print(f"Missing policy     : {MISSING_POLICY}")
    print(f"Minimum components : {MIN_COMPONENT_COUNT}")
    print(f"Output             : {OUT_DIR}")
    print("=" * 88)

    random_result = run_random_validation(base)
    grouped_result = run_grouped_validation(base)
    temporal_result = run_temporal_validation(base)

    final_iterations, iteration_source = choose_final_iterations(
        grouped_result.get("best_iterations", []),
        random_result.get("best_iterations", []),
        temporal_result.get("best_iterations", []),
    )

    final_model, final_normalization, final_transformed = train_final_model(
        base,
        x_cols,
        final_iterations,
        iteration_source,
    )

    all_metrics: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    all_splits: List[Dict[str, Any]] = []
    all_normalizations: List[Dict[str, Any]] = []

    for result in (random_result, grouped_result, temporal_result):
        all_metrics.extend(result.get("metrics", []))
        all_predictions.extend(result.get("predictions", []))
        all_splits.extend(result.get("splits", []))
        all_normalizations.extend(result.get("normalizations", []))

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    splits_df = pd.DataFrame(all_splits)
    normalizations_df = pd.DataFrame(all_normalizations)
    summary_df = summarize_metrics(metrics_df, predictions_df)
    coverage_df = build_component_coverage(base["work"], final_transformed)

    metrics_df.to_csv(METRICS_FOLD_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(METRICS_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    splits_df.to_csv(SPLIT_MANIFEST_CSV, index=False, encoding="utf-8-sig")
    normalizations_df.to_csv(NORMALIZATION_CSV, index=False, encoding="utf-8-sig")
    coverage_df.to_csv(COMPONENT_COVERAGE_CSV, index=False, encoding="utf-8-sig")

    pd.DataFrame([final_model]).to_csv(
        FINAL_MODEL_MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with open(FINAL_NORMALIZATION_JSON, "w", encoding="utf-8") as f:
        json.dump(final_normalization, f, ensure_ascii=False, indent=2)

    training_summary = {
        "csv_path": CSV_PATH,
        "output_dir": OUT_DIR,
        "data_level": ONLY_TRAIN_LEVEL,
        "feature_source": feature_source,
        "feature_count": len(x_cols),
        "base_samples": base["samples"],
        "base_papers": base["papers"],
        "dated_samples": base["dated_samples"],
        "target": TARGET_COMPOSITE,
        "weights": WEIGHTS,
        "missing_policy": MISSING_POLICY,
        "minimum_component_count": MIN_COMPONENT_COUNT,
        "random_seed": SEED,
        "random_holdout_status": random_result["status"],
        "paper_grouped_cv_status": grouped_result["status"],
        "paper_grouped_fold_statuses": grouped_result.get("fold_statuses", []),
        "temporal_holdout_status": temporal_result["status"],
        "temporal_cutoff_year": temporal_result.get("cutoff"),
        "final_iterations": final_iterations,
        "iteration_source": iteration_source,
        "final_model_path": final_model["model_path"],
        "final_normalization_path": FINAL_NORMALIZATION_JSON,
    }

    with open(TRAINING_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("[L1 COMPOSITE VALIDATION FINISHED]")
    print(f"Random validation   : {random_result['status']}")
    print(f"Grouped validation  : {grouped_result['status']}")
    print(f"Temporal validation : {temporal_result['status']}")
    print(f"Final iterations    : {final_iterations}")
    print(f"Fold metrics        : {METRICS_FOLD_CSV}")
    print(f"Summary metrics     : {METRICS_SUMMARY_CSV}")
    print(f"Predictions         : {PREDICTIONS_CSV}")
    print(f"Split manifest      : {SPLIT_MANIFEST_CSV}")
    print(f"Normalization log   : {NORMALIZATION_CSV}")
    print(f"Component coverage  : {COMPONENT_COVERAGE_CSV}")
    print(f"Final normalization : {FINAL_NORMALIZATION_JSON}")
    print(f"Final model         : {final_model['model_path']}")
    print(f"Training summary    : {TRAINING_SUMMARY_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
