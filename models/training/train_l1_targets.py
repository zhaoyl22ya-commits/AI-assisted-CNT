# -*- coding: utf-8 -*-
"""
train_l1_targets.py

输入：
    data/11_flattened/ml_dataset_3p0_full60D.csv
    data/11_flattened/feature_schema.json

完成：
1. random_holdout：原有 80/20 随机划分基线；
2. paper_grouped_cv：按 paper_id 的 5 折交叉验证；
3. temporal_holdout：早期年份训练、后期年份测试；
4. early stopping 仅使用外层训练集内部验证集；
5. 验证结束后，用全部有效 L1 数据训练最终模型。

输出到：
    outputs/models/l1_targets/
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

# 当前脚本：
# project_root/models/training/train_l1_targets.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：
# E:\pycharm_files\20260721-CNTcodes
PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..")
)

# 11_flatten_dataset.py 的正式输出
DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "11_flattened",
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "ml_dataset_3p0_full60D.csv",
)

SCHEMA_PATH = os.path.join(
    DATA_DIR,
    "feature_schema.json",
)

# L1 单目标模型正式输出
OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "models",
    "l1_targets",
)

FINAL_MODEL_DIR = os.path.join(OUT_DIR, "final_models")
VALIDATION_MODEL_DIR = os.path.join(OUT_DIR, "validation_models")

METRICS_FOLD_CSV = os.path.join(OUT_DIR, "L1_metrics_fold.csv")
METRICS_SUMMARY_CSV = os.path.join(OUT_DIR, "L1_metrics_summary.csv")
PREDICTIONS_CSV = os.path.join(OUT_DIR, "L1_predictions.csv")
SPLIT_MANIFEST_CSV = os.path.join(OUT_DIR, "split_manifest.csv")
TARGET_COUNTS_CSV = os.path.join(OUT_DIR, "target_sample_counts.csv")
FINAL_MODEL_MANIFEST_CSV = os.path.join(OUT_DIR, "final_model_manifest.csv")
TRAINING_SUMMARY_JSON = os.path.join(OUT_DIR, "training_summary.json")


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

MIN_TEMPORAL_TRAIN_SAMPLES = 50
MIN_TEMPORAL_TEST_SAMPLES = 20
MIN_TEMPORAL_TRAIN_PAPERS = 5
MIN_TEMPORAL_TEST_PAPERS = 3

MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
DEFAULT_FINAL_ITERATIONS = 500
SAVE_VALIDATION_MODELS = True


# =============================================================================
# 3. 目标与模型参数
# =============================================================================

MAIN_TARGET = "y_growth_quality_score"

NUMERIC_TARGETS = [
    "y_diameter_nm",
    "y_length_mm",
    "y_bundle_density_per_um",
    "y_cnt_density_per_um",
    "y_ig_id_ratio",
    "y_orientation_score",
    "y_purity_score",
]

LOG_TARGETS = {
    "y_length_mm",
    "y_bundle_density_per_um",
    "y_cnt_density_per_um",
}

LEGACY_IGID_COL = "y_IG_ID"
PREFERRED_IGID_COL = "y_ig_id_ratio"
UNIFIED_IGID_COL = "y_ig_id_ratio_unified"

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
# 4. 工具函数
# =============================================================================

def safe_name(text: str) -> str:
    return str(text).replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


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
        except Exception as e:
            print(f"[WARN] feature_schema.json 读取失败，回退到代码定义：{e}")
    return build_feature_cols(), "code_fallback"


def ensure_igid_fallback(df: pd.DataFrame) -> pd.DataFrame:
    preferred = (
        pd.to_numeric(df[PREFERRED_IGID_COL], errors="coerce")
        if PREFERRED_IGID_COL in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    legacy = (
        pd.to_numeric(df[LEGACY_IGID_COL], errors="coerce")
        if LEGACY_IGID_COL in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df = df.copy()
    df[UNIFIED_IGID_COL] = preferred.where(preferred.notna(), legacy)
    return df


def build_target_list(df: pd.DataFrame) -> List[str]:
    targets = [MAIN_TARGET]
    for target in NUMERIC_TARGETS:
        targets.append(UNIFIED_IGID_COL if target == PREFERRED_IGID_COL else target)
    targets = [target for target in targets if target in df.columns]
    if MAIN_TARGET not in targets:
        raise RuntimeError(f"主目标不存在：{MAIN_TARGET}")
    return targets


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


def inverse_target_scale(values: np.ndarray, use_log: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.expm1(values) if use_log else values


def save_validation_model(
    booster: lgb.Booster,
    target: str,
    split_type: str,
    fold: int,
) -> Optional[str]:
    if not SAVE_VALIDATION_MODELS:
        return None

    directory = os.path.join(
        VALIDATION_MODEL_DIR,
        split_type,
    )
    os.makedirs(
        directory,
        exist_ok=True,
    )

    path = os.path.join(
        directory,
        f"L1_{safe_name(target)}_fold{fold}.txt",
    )

    booster.save_model(path)
    return path


# =============================================================================
# 5. 单目标数据准备
# =============================================================================

def prepare_target_data(
    df_all: pd.DataFrame,
    x_cols: Sequence[str],
    target_col: str,
) -> Dict[str, Any]:

    y_raw = pd.to_numeric(
        df_all[target_col],
        errors="coerce",
    )

    valid = y_raw.notna()

    work = df_all.loc[valid].copy()
    y_original = y_raw.loc[valid].astype(float)

    use_log = target_col in LOG_TARGETS

    if use_log:
        positive = y_original > 0

        work = work.loc[positive].copy()
        y_original = y_original.loc[positive]

        y_model = np.log1p(
            y_original.to_numpy(dtype=float)
        )

    else:
        y_model = y_original.to_numpy(dtype=float)

    work = work.reset_index(drop=True)

    y_original = (
        y_original
        .reset_index(drop=True)
        .to_numpy(dtype=float)
    )

    X = work[list(x_cols)].copy()

    for column in x_cols:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.reset_index(drop=True)

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

    work["sample_id"] = (
        work["sample_id"]
        .fillna("")
        .astype(str)
    )

    work["paper_id"] = (
        work["paper_id"]
        .fillna("")
        .astype(str)
    )

    work["paper_year_int"] = pd.to_numeric(
        work["paper_year_int"],
        errors="coerce",
    )

    return {
        "target": target_col,
        "use_log": use_log,
        "work": work,
        "X": X,
        "y_model": np.asarray(
            y_model,
            dtype=float,
        ),
        "y_original": y_original,
        "samples": int(len(work)),
        "papers": int(
            work["paper_id"].nunique()
        ),
        "dated_samples": int(
            work["paper_year_int"]
            .notna()
            .sum()
        ),
    }


# =============================================================================
# 6. 内部验证与 LightGBM
# =============================================================================

def make_inner_validation_split(
    outer_train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, str]:

    outer_train_indices = np.asarray(
        outer_train_indices,
        dtype=int,
    )

    groups = (
        groups_all
        .iloc[outer_train_indices]
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
                np.zeros(
                    len(outer_train_indices)
                ),
                groups=groups.to_numpy(),
            )
        )

        return (
            outer_train_indices[rel_fit],
            outer_train_indices[rel_valid],
            "group_shuffle",
        )

    rel_indices = np.arange(
        len(outer_train_indices)
    )

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
    y_model: np.ndarray,
    outer_train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[lgb.Booster, int, str]:

    fit_indices, valid_indices, inner_method = (
        make_inner_validation_split(
            outer_train_indices,
            groups_all,
        )
    )

    train_set = lgb.Dataset(
        X.iloc[fit_indices],
        label=y_model[fit_indices],
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X.iloc[valid_indices],
        label=y_model[valid_indices],
        reference=train_set,
        free_raw_data=False,
    )

    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[
            train_set,
            valid_set,
        ],
        valid_names=[
            "inner_fit",
            "inner_valid",
        ],
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0,
            ),
        ],
    )

    best_iteration = int(
        booster.best_iteration
        if (
            booster.best_iteration
            and booster.best_iteration > 0
        )
        else MAX_BOOST_ROUNDS
    )

    return (
        booster,
        best_iteration,
        inner_method,
    )


# =============================================================================
# 7. 结果行构造
# =============================================================================

def prediction_rows(
    target: str,
    split_type: str,
    fold: int,
    test_indices: np.ndarray,
    work: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cutoff: Optional[int] = None,
) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    for position, index in enumerate(
        np.asarray(
            test_indices,
            dtype=int,
        )
    ):

        true_value = float(
            y_true[position]
        )

        pred_value = float(
            y_pred[position]
        )

        residual = (
            pred_value
            - true_value
        )

        item = work.iloc[index]

        rows.append({
            "sample_id": item["sample_id"],
            "paper_id": item["paper_id"],
            "paper_year_int": item["paper_year_int"],
            "growth_mode": item["growth_mode"],
            "data_level": item["data_level"],
            "source_file": item["source_file"],
            "synthesis_id": item["synthesis_id"],
            "target": target,
            "split_type": split_type,
            "fold": fold,
            "temporal_cutoff_year": cutoff,
            "y_true": true_value,
            "y_pred": pred_value,
            "residual": residual,
            "absolute_error": abs(residual),
        })

    return rows


def split_rows(
    target: str,
    split_type: str,
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    work: pd.DataFrame,
    cutoff: Optional[int] = None,
) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    for role, indices in (
        ("train", train_indices),
        ("test", test_indices),
    ):

        for index in np.asarray(
            indices,
            dtype=int,
        ):

            item = work.iloc[index]

            rows.append({
                "sample_id": item["sample_id"],
                "paper_id": item["paper_id"],
                "paper_year_int": item["paper_year_int"],
                "growth_mode": item["growth_mode"],
                "data_level": item["data_level"],
                "target": target,
                "split_type": split_type,
                "fold": fold,
                "role": role,
                "temporal_cutoff_year": cutoff,
            })

    return rows


def metric_row(
    target: str,
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
        "target": target,
        "split_type": split_type,
        "fold": fold,
        "status": "ok",
        "n_train": int(
            len(train_indices)
        ),
        "n_test": int(
            len(test_indices)
        ),
        "n_train_papers": int(
            work
            .iloc[train_indices]["paper_id"]
            .nunique()
        ),
        "n_test_papers": int(
            work
            .iloc[test_indices]["paper_id"]
            .nunique()
        ),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "spearman": metrics["spearman"],
        "best_iteration": int(
            best_iteration
        ),
        "inner_split_method": inner_method,
        "temporal_cutoff_year": cutoff,
    }


# =============================================================================
# 8. 三类验证
# =============================================================================

def run_random_validation(
    prepared: Dict[str, Any],
) -> Dict[str, Any]:

    work = prepared["work"]
    X = prepared["X"]

    y_model = prepared["y_model"]
    use_log = prepared["use_log"]

    all_indices = np.arange(
        len(work)
    )

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=RANDOM_TEST_SIZE,
        random_state=SEED,
    )

    (
        booster,
        best_iter,
        inner_method,
    ) = fit_lightgbm_with_inner_valid(
        X,
        y_model,
        train_idx,
        work["paper_id"],
    )

    pred_model = booster.predict(
        X.iloc[test_idx],
        num_iteration=best_iter,
    )

    y_true = inverse_target_scale(
        y_model[test_idx],
        use_log,
    )

    y_pred = inverse_target_scale(
        pred_model,
        use_log,
    )

    metrics = evaluate_predictions(
        y_true,
        y_pred,
    )

    row = metric_row(
        prepared["target"],
        "random_holdout",
        1,
        train_idx,
        test_idx,
        work,
        metrics,
        best_iter,
        inner_method,
    )

    row["model_path"] = (
        save_validation_model(
            booster,
            prepared["target"],
            "random_holdout",
            1,
        )
    )

    return {
        "status": "ok",
        "metrics": [row],
        "predictions": prediction_rows(
            prepared["target"],
            "random_holdout",
            1,
            test_idx,
            work,
            y_true,
            y_pred,
        ),
        "splits": split_rows(
            prepared["target"],
            "random_holdout",
            1,
            train_idx,
            test_idx,
            work,
        ),
        "best_iterations": [
            best_iter
        ],
    }


def run_grouped_validation(
    prepared: Dict[str, Any],
) -> Dict[str, Any]:

    work = prepared["work"]
    X = prepared["X"]

    y_model = prepared["y_model"]
    use_log = prepared["use_log"]

    groups = (
        work["paper_id"]
        .fillna("")
        .astype(str)
    )

    if (
        groups.nunique()
        < GROUP_CV_SPLITS
    ):
        return {
            "status": "skipped_too_few_papers",
            "metrics": [],
            "predictions": [],
            "splits": [],
            "best_iterations": [],
        }

    metrics_all: List[
        Dict[str, Any]
    ] = []

    predictions_all: List[
        Dict[str, Any]
    ] = []

    splits_all: List[
        Dict[str, Any]
    ] = []

    iterations: List[int] = []

    splitter = GroupKFold(
        n_splits=GROUP_CV_SPLITS
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            X,
            y_model,
            groups=groups,
        ),
        start=1,
    ):

        (
            booster,
            best_iter,
            inner_method,
        ) = fit_lightgbm_with_inner_valid(
            X,
            y_model,
            train_idx,
            groups,
        )

        pred_model = booster.predict(
            X.iloc[test_idx],
            num_iteration=best_iter,
        )

        y_true = inverse_target_scale(
            y_model[test_idx],
            use_log,
        )

        y_pred = inverse_target_scale(
            pred_model,
            use_log,
        )

        metrics = evaluate_predictions(
            y_true,
            y_pred,
        )

        row = metric_row(
            prepared["target"],
            "paper_grouped_cv",
            fold,
            train_idx,
            test_idx,
            work,
            metrics,
            best_iter,
            inner_method,
        )

        row["model_path"] = (
            save_validation_model(
                booster,
                prepared["target"],
                "paper_grouped_cv",
                fold,
            )
        )

        metrics_all.append(
            row
        )

        predictions_all.extend(
            prediction_rows(
                prepared["target"],
                "paper_grouped_cv",
                fold,
                test_idx,
                work,
                y_true,
                y_pred,
            )
        )

        splits_all.extend(
            split_rows(
                prepared["target"],
                "paper_grouped_cv",
                fold,
                train_idx,
                test_idx,
                work,
            )
        )

        iterations.append(
            best_iter
        )

    return {
        "status": "ok",
        "metrics": metrics_all,
        "predictions": predictions_all,
        "splits": splits_all,
        "best_iterations": iterations,
    }


def choose_temporal_cutoff(
    work: pd.DataFrame,
) -> Optional[int]:

    dated = work[
        work["paper_year_int"]
        .notna()
    ].copy()

    if len(dated) < (
        MIN_TEMPORAL_TRAIN_SAMPLES
        + MIN_TEMPORAL_TEST_SAMPLES
    ):
        return None

    years = sorted(
        int(year)
        for year in (
            dated["paper_year_int"]
            .unique()
        )
    )

    candidates: List[
        Tuple[float, int]
    ] = []

    for cutoff in years[:-1]:

        train = dated[
            dated["paper_year_int"]
            <= cutoff
        ]

        test = dated[
            dated["paper_year_int"]
            > cutoff
        ]

        if (
            len(train)
            < MIN_TEMPORAL_TRAIN_SAMPLES
        ):
            continue

        if (
            len(test)
            < MIN_TEMPORAL_TEST_SAMPLES
        ):
            continue

        if (
            train["paper_id"]
            .nunique()
            < MIN_TEMPORAL_TRAIN_PAPERS
        ):
            continue

        if (
            test["paper_id"]
            .nunique()
            < MIN_TEMPORAL_TEST_PAPERS
        ):
            continue

        test_fraction = (
            len(test)
            / (
                len(train)
                + len(test)
            )
        )

        candidates.append(
            (
                abs(
                    test_fraction
                    - RANDOM_TEST_SIZE
                ),
                cutoff,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1],
        )
    )

    return int(
        candidates[0][1]
    )


def run_temporal_validation(
    prepared: Dict[str, Any],
) -> Dict[str, Any]:

    work = prepared["work"]
    X = prepared["X"]

    y_model = prepared["y_model"]
    use_log = prepared["use_log"]

    cutoff = choose_temporal_cutoff(
        work
    )

    if cutoff is None:
        return {
            "status": "skipped_too_few_dated_samples",
            "metrics": [],
            "predictions": [],
            "splits": [],
            "best_iterations": [],
            "cutoff": None,
        }

    dated = (
        work["paper_year_int"]
        .notna()
    )

    train_idx = np.where(
        (
            dated
            & (
                work["paper_year_int"]
                <= cutoff
            )
        ).to_numpy()
    )[0]

    test_idx = np.where(
        (
            dated
            & (
                work["paper_year_int"]
                > cutoff
            )
        ).to_numpy()
    )[0]

    (
        booster,
        best_iter,
        inner_method,
    ) = fit_lightgbm_with_inner_valid(
        X,
        y_model,
        train_idx,
        work["paper_id"],
    )

    pred_model = booster.predict(
        X.iloc[test_idx],
        num_iteration=best_iter,
    )

    y_true = inverse_target_scale(
        y_model[test_idx],
        use_log,
    )

    y_pred = inverse_target_scale(
        pred_model,
        use_log,
    )

    metrics = evaluate_predictions(
        y_true,
        y_pred,
    )

    row = metric_row(
        prepared["target"],
        "temporal_holdout",
        1,
        train_idx,
        test_idx,
        work,
        metrics,
        best_iter,
        inner_method,
        cutoff,
    )

    row["model_path"] = (
        save_validation_model(
            booster,
            prepared["target"],
            "temporal_holdout",
            1,
        )
    )

    row[
        "temporal_train_year_max"
    ] = cutoff

    row[
        "temporal_test_year_min"
    ] = int(
        work
        .iloc[test_idx][
            "paper_year_int"
        ]
        .min()
    )

    return {
        "status": "ok",
        "metrics": [row],
        "predictions": prediction_rows(
            prepared["target"],
            "temporal_holdout",
            1,
            test_idx,
            work,
            y_true,
            y_pred,
            cutoff,
        ),
        "splits": split_rows(
            prepared["target"],
            "temporal_holdout",
            1,
            train_idx,
            test_idx,
            work,
            cutoff,
        ),
        "best_iterations": [
            best_iter
        ],
        "cutoff": cutoff,
    }


# =============================================================================
# 9. 最终全量模型
# =============================================================================

def choose_final_iterations(
    grouped_iterations: Sequence[int],
    random_iterations: Sequence[int],
    temporal_iterations: Sequence[int],
) -> Tuple[int, str]:

    grouped = [
        int(value)
        for value in grouped_iterations
        if (
            value is not None
            and value > 0
        )
    ]

    if grouped:
        return (
            max(
                1,
                int(
                    round(
                        median(grouped)
                    )
                ),
            ),
            "median_paper_grouped_cv",
        )

    fallback = [
        int(value)
        for value in (
            list(
                random_iterations
            )
            + list(
                temporal_iterations
            )
        )
        if (
            value is not None
            and value > 0
        )
    ]

    if fallback:
        return (
            max(
                1,
                int(
                    round(
                        median(fallback)
                    )
                ),
            ),
            "median_available_validation",
        )

    return (
        DEFAULT_FINAL_ITERATIONS,
        "default",
    )


def train_final_model(
    prepared: Dict[str, Any],
    x_cols: Sequence[str],
    final_iterations: int,
    iteration_source: str,
) -> Dict[str, Any]:

    train_set = lgb.Dataset(
        prepared["X"],
        label=prepared["y_model"],
        free_raw_data=False,
    )

    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=train_set,
        num_boost_round=final_iterations,
        callbacks=[
            lgb.log_evaluation(
                period=0,
            )
        ],
    )

    os.makedirs(
        FINAL_MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        FINAL_MODEL_DIR,
        (
            f"L1_"
            f"{safe_name(prepared['target'])}"
            f".txt"
        ),
    )

    booster.save_model(
        model_path
    )

    return {
        "target": prepared["target"],
        "status": "ok",
        "samples": prepared["samples"],
        "papers": prepared["papers"],
        "feature_count": len(
            x_cols
        ),
        "log_target": prepared["use_log"],
        "final_iterations": final_iterations,
        "iteration_source": iteration_source,
        "model_path": model_path,
    }


# =============================================================================
# 10. 指标汇总
# =============================================================================

def summarize_metrics(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[
        Dict[str, Any]
    ] = []

    if metrics_df.empty:
        return pd.DataFrame()

    for (
        target,
        split_type,
    ), group in (
        metrics_df.groupby(
            [
                "target",
                "split_type",
            ],
            dropna=False,
        )
    ):

        pred_group = predictions_df[
            (
                predictions_df[
                    "target"
                ]
                == target
            )
            & (
                predictions_df[
                    "split_type"
                ]
                == split_type
            )
        ]

        pooled = (
            evaluate_predictions(
                pred_group[
                    "y_true"
                ].to_numpy(
                    dtype=float
                ),
                pred_group[
                    "y_pred"
                ].to_numpy(
                    dtype=float
                ),
            )
            if len(pred_group) > 0
            else {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
            }
        )

        rows.append({
            "target": target,
            "split_type": split_type,
            "fold_count": len(group),
            "total_test_predictions": len(
                pred_group
            ),
            "mae_mean": group[
                "mae"
            ].mean(),
            "mae_std": (
                group["mae"].std(
                    ddof=1
                )
                if len(group) > 1
                else np.nan
            ),
            "rmse_mean": group[
                "rmse"
            ].mean(),
            "rmse_std": (
                group["rmse"].std(
                    ddof=1
                )
                if len(group) > 1
                else np.nan
            ),
            "r2_mean": group[
                "r2"
            ].mean(),
            "r2_std": (
                group["r2"].std(
                    ddof=1
                )
                if len(group) > 1
                else np.nan
            ),
            "spearman_mean": group[
                "spearman"
            ].mean(),
            "spearman_std": (
                group[
                    "spearman"
                ].std(
                    ddof=1
                )
                if len(group) > 1
                else np.nan
            ),
            "pooled_mae": pooled[
                "mae"
            ],
            "pooled_rmse": pooled[
                "rmse"
            ],
            "pooled_r2": pooled[
                "r2"
            ],
            "pooled_spearman": pooled[
                "spearman"
            ],
            "best_iteration_median": int(
                round(
                    median(
                        group[
                            "best_iteration"
                        ]
                        .astype(int)
                        .tolist()
                    )
                )
            ),
            "temporal_cutoff_year": (
                int(
                    group[
                        "temporal_cutoff_year"
                    ]
                    .dropna()
                    .iloc[0]
                )
                if (
                    "temporal_cutoff_year"
                    in group.columns
                    and not group[
                        "temporal_cutoff_year"
                    ]
                    .dropna()
                    .empty
                )
                else ""
            ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 11. 主程序
# =============================================================================

def main() -> None:

    if not os.path.exists(
        CSV_PATH
    ):
        raise FileNotFoundError(
            f"CSV 不存在：{CSV_PATH}"
        )

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        FINAL_MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        VALIDATION_MODEL_DIR,
        exist_ok=True,
    )

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    if (
        "data_level"
        not in df.columns
    ):
        raise RuntimeError(
            "CSV 缺少 data_level，无法筛选 L1。"
        )

    df = df[
        df["data_level"]
        .astype(str)
        == ONLY_TRAIN_LEVEL
    ].copy()

    if df.empty:
        raise RuntimeError(
            "筛选 data_level == L1 后没有数据。"
        )

    x_cols, feature_source = (
        load_feature_cols()
    )

    missing_x = [
        column
        for column in x_cols
        if column not in df.columns
    ]

    if missing_x:
        raise RuntimeError(
            f"缺少模型特征列 "
            f"({len(missing_x)})，"
            f"示例：{missing_x[:10]}"
        )

    required_metadata = [
        "paper_id",
        "sample_id",
        "paper_year_int",
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in df.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            "当前 CSV 不是新版 flatten 输出，"
            f"缺少：{missing_metadata}"
        )

    df = ensure_igid_fallback(
        df
    )

    targets = build_target_list(
        df
    )

    print("=" * 88)
    print("[L1 VALIDATION START]")
    print(
        f"CSV            : "
        f"{CSV_PATH}"
    )
    print(
        f"L1 rows        : "
        f"{len(df)}"
    )
    print(
        f"Feature source : "
        f"{feature_source}"
    )
    print(
        f"Feature count  : "
        f"{len(x_cols)}"
    )
    print(
        f"Targets        : "
        f"{targets}"
    )
    print(
        f"Output         : "
        f"{OUT_DIR}"
    )
    print("=" * 88)

    all_metrics: List[
        Dict[str, Any]
    ] = []

    all_predictions: List[
        Dict[str, Any]
    ] = []

    all_splits: List[
        Dict[str, Any]
    ] = []

    target_counts: List[
        Dict[str, Any]
    ] = []

    final_models: List[
        Dict[str, Any]
    ] = []

    summary: Dict[
        str,
        Any,
    ] = {
        "csv_path": CSV_PATH,
        "output_dir": OUT_DIR,
        "data_level": ONLY_TRAIN_LEVEL,
        "feature_source": feature_source,
        "feature_count": len(
            x_cols
        ),
        "random_seed": SEED,
        "targets": {},
    }

    for target in targets:

        print(
            f"\n========== "
            f"TARGET: {target} "
            f"=========="
        )

        prepared = prepare_target_data(
            df,
            x_cols,
            target,
        )

        target_counts.append({
            "target": target,
            "samples": prepared[
                "samples"
            ],
            "papers": prepared[
                "papers"
            ],
            "dated_samples": prepared[
                "dated_samples"
            ],
            "log_target": prepared[
                "use_log"
            ],
        })

        target_status = {
            "samples": prepared[
                "samples"
            ],
            "papers": prepared[
                "papers"
            ],
            "dated_samples": prepared[
                "dated_samples"
            ],
            "log_target": prepared[
                "use_log"
            ],
        }

        if (
            prepared["samples"]
            < MIN_TOTAL_SAMPLES
        ):

            target_status[
                "status"
            ] = (
                "skipped_too_few_samples"
            )

            summary[
                "targets"
            ][target] = target_status

            print(
                f"[SKIP] "
                f"samples="
                f"{prepared['samples']} "
                f"< "
                f"{MIN_TOTAL_SAMPLES}"
            )

            continue

        if (
            prepared["papers"]
            < MIN_TOTAL_PAPERS
        ):

            target_status[
                "status"
            ] = (
                "skipped_too_few_papers"
            )

            summary[
                "targets"
            ][target] = target_status

            print(
                f"[SKIP] "
                f"papers="
                f"{prepared['papers']} "
                f"< "
                f"{MIN_TOTAL_PAPERS}"
            )

            continue

        print(
            f"[DATA] "
            f"samples="
            f"{prepared['samples']}, "
            f"papers="
            f"{prepared['papers']}, "
            f"dated="
            f"{prepared['dated_samples']}"
        )

        random_result = (
            run_random_validation(
                prepared
            )
        )

        grouped_result = (
            run_grouped_validation(
                prepared
            )
        )

        temporal_result = (
            run_temporal_validation(
                prepared
            )
        )

        for result in (
            random_result,
            grouped_result,
            temporal_result,
        ):

            all_metrics.extend(
                result.get(
                    "metrics",
                    [],
                )
            )

            all_predictions.extend(
                result.get(
                    "predictions",
                    [],
                )
            )

            all_splits.extend(
                result.get(
                    "splits",
                    [],
                )
            )

        (
            final_iterations,
            iteration_source,
        ) = choose_final_iterations(
            grouped_result.get(
                "best_iterations",
                [],
            ),
            random_result.get(
                "best_iterations",
                [],
            ),
            temporal_result.get(
                "best_iterations",
                [],
            ),
        )

        final_model = train_final_model(
            prepared,
            x_cols,
            final_iterations,
            iteration_source,
        )

        final_models.append(
            final_model
        )

        target_status.update({
            "status": "ok",
            "random_holdout": (
                random_result[
                    "status"
                ]
            ),
            "paper_grouped_cv": (
                grouped_result[
                    "status"
                ]
            ),
            "temporal_holdout": (
                temporal_result[
                    "status"
                ]
            ),
            "temporal_cutoff_year": (
                temporal_result.get(
                    "cutoff"
                )
            ),
            "final_iterations": (
                final_iterations
            ),
            "iteration_source": (
                iteration_source
            ),
            "final_model_path": (
                final_model[
                    "model_path"
                ]
            ),
        })

        summary[
            "targets"
        ][target] = target_status

        print(
            f"[DONE] "
            f"random="
            f"{random_result['status']}, "
            f"grouped="
            f"{grouped_result['status']}, "
            f"temporal="
            f"{temporal_result['status']}, "
            f"final_iter="
            f"{final_iterations}"
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    splits_df = pd.DataFrame(
        all_splits
    )

    target_counts_df = pd.DataFrame(
        target_counts
    )

    final_models_df = pd.DataFrame(
        final_models
    )

    summary_df = summarize_metrics(
        metrics_df,
        predictions_df,
    )

    metrics_df.to_csv(
        METRICS_FOLD_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        METRICS_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    splits_df.to_csv(
        SPLIT_MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    target_counts_df.to_csv(
        TARGET_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    final_models_df.to_csv(
        FINAL_MODEL_MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        TRAINING_SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "[L1 VALIDATION FINISHED]"
    )

    print(
        f"Fold metrics        : "
        f"{METRICS_FOLD_CSV}"
    )

    print(
        f"Summary metrics     : "
        f"{METRICS_SUMMARY_CSV}"
    )

    print(
        f"Predictions         : "
        f"{PREDICTIONS_CSV}"
    )

    print(
        f"Split manifest      : "
        f"{SPLIT_MANIFEST_CSV}"
    )

    print(
        f"Target counts       : "
        f"{TARGET_COUNTS_CSV}"
    )

    print(
        f"Final model manifest: "
        f"{FINAL_MODEL_MANIFEST_CSV}"
    )

    print(
        f"Final models        : "
        f"{FINAL_MODEL_DIR}"
    )

    print(
        f"Training summary    : "
        f"{TRAINING_SUMMARY_JSON}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()