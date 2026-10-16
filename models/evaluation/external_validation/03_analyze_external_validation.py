# -*- coding: utf-8 -*-
"""
03_analyze_external_validation.py

对 02_validate_external_models.py 生成的独立外部验证结果进行统一统计分析。

本代码不训练模型，也不重新计算外部数据的特征或归一化参数。
它只读取已经生成的 external validation predictions，并完成：

1. L1 单目标、L1 Composite、L12 q50 的统一回归/排序指标；
2. 以 paper_id 为重采样单位的 cluster bootstrap 95% CI；
3. L12 q10/q50/q90 的分位数指标及 paper-level bootstrap CI；
4. A-E growth mode 分层分析；
5. L1/L2 分层的 L12 quantile 分析；
6. parity plots 与 quantile calibration plot；
7. 论文表格、source data 和 analysis manifest。

默认输入：
outputs/evaluation/external_validation/

默认输出：
outputs/evaluation/external_validation/
├── figures/
├── source_data/
├── tables/
├── manifests/
├── external_regression_metrics.csv
├── external_regression_bootstrap_ci.csv
├── external_regression_metrics_with_ci.csv
├── external_quantile_metrics.csv
├── external_quantile_bootstrap_ci.csv
├── external_quantile_metrics_with_ci.csv
├── external_metrics_by_growth_mode.csv
├── external_quantile_metrics_by_growth_mode.csv
└── external_analysis_summary.json

运行：
python models/evaluation/external_validation/03_analyze_external_validation.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. 固定定义
# =============================================================================

DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_RANDOM_SEED = 20261003
DEFAULT_DPI = 600
CI_LEVEL = 0.95

MIN_BOOTSTRAP_SAMPLES = 10
MIN_BOOTSTRAP_PAPERS = 3

MIN_MODE_SAMPLES_FOR_R2 = 10
MIN_MODE_PAPERS_FOR_R2 = 3

REGRESSION_METRICS = [
    "mae",
    "rmse",
    "r2",
    "spearman",
]

QUANTILE_METRICS = [
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "empirical_coverage_q10",
    "empirical_coverage_q50",
    "empirical_coverage_q90",
    "calibration_error_q10",
    "calibration_error_q50",
    "calibration_error_q90",
    "q50_mae",
    "q50_rmse",
    "q50_r2",
    "q50_spearman",
    "picp_80_ordered",
    "picp_80_calibration_error",
    "mpiw_80_ordered",
    "winkler_80_ordered",
    "crossing_q10_q50_rate",
    "crossing_q50_q90_rate",
    "any_crossing_rate",
]

TARGET_ORDER = [
    "y_growth_quality_score",
    "y_diameter_nm",
    "y_length_mm",
    "y_bundle_density_per_um",
    "y_cnt_density_per_um",
    "y_ig_id_ratio_unified",
    "y_orientation_score",
    "y_purity_score",
    "y_composite_final_score",
]

TARGET_LABELS = {
    "y_growth_quality_score": "Growth quality",
    "y_diameter_nm": "Diameter (nm)",
    "y_length_mm": "Length (mm)",
    "y_bundle_density_per_um": "Bundle density (μm⁻¹)",
    "y_cnt_density_per_um": "CNT density (μm⁻¹)",
    "y_ig_id_ratio_unified": "IG/ID ratio",
    "y_orientation_score": "Orientation score",
    "y_purity_score": "Purity score",
    "y_composite_final_score": "Composite score",
}

SOURCE_ORDER = [
    "L1_multi_target",
    "L1_composite",
    "L12_q50",
]

SOURCE_LABELS = {
    "L1_multi_target": "L1 multi-target",
    "L1_composite": "L1 Composite",
    "L12_q50": "L1+L2 q50",
}

GROWTH_MODE_ORDER = ["A", "B", "C", "D", "E", "Unknown"]


# =============================================================================
# 2. 路径
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "outputs" / "evaluation" / "external_validation"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR


def resolve_project_path(
    value: Optional[str],
    default: Path,
) -> Path:
    if value is None or not str(value).strip():
        return default.resolve()

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} 不存在：{path}"
        )


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"{label} 缺少必要列：{missing}"
        )


# =============================================================================
# 3. 基础统计
# =============================================================================

def safe_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) < 2:
        return float("nan")

    if np.allclose(
        y_true,
        y_true[0],
    ):
        return float("nan")

    return float(
        r2_score(
            y_true,
            y_pred,
        )
    )


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) < 2:
        return float("nan")

    true_rank = (
        pd.Series(y_true)
        .rank(method="average")
        .to_numpy(dtype=float)
    )
    pred_rank = (
        pd.Series(y_pred)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    if (
        np.std(true_rank) == 0
        or np.std(pred_rank) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            true_rank,
            pred_rank,
        )[0, 1]
    )


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true_array)
        & np.isfinite(y_pred_array)
    )

    y_true_array = y_true_array[finite]
    y_pred_array = y_pred_array[finite]

    if len(y_true_array) == 0:
        return {
            "n_samples": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }

    return {
        "n_samples": int(
            len(y_true_array)
        ),
        "mae": float(
            mean_absolute_error(
                y_true_array,
                y_pred_array,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true_array,
                    y_pred_array,
                )
            )
        ),
        "r2": safe_r2(
            y_true_array,
            y_pred_array,
        ),
        "spearman": safe_spearman(
            y_true_array,
            y_pred_array,
        ),
    }


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    error = (
        np.asarray(y_true, dtype=float)
        - np.asarray(y_pred, dtype=float)
    )
    loss = np.maximum(
        alpha * error,
        (alpha - 1.0) * error,
    )
    return float(
        np.mean(loss)
    )


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    lower = np.asarray(
        lower,
        dtype=float,
    )
    upper = np.asarray(
        upper,
        dtype=float,
    )

    width = upper - lower
    score = width.copy()

    below = y_true < lower
    above = y_true > upper

    score[below] += (
        2.0 / alpha
        * (lower[below] - y_true[below])
    )
    score[above] += (
        2.0 / alpha
        * (y_true[above] - upper[above])
    )

    return score


def evaluate_quantiles(
    y_true: Sequence[float],
    q10_raw: Sequence[float],
    q50_raw: Sequence[float],
    q90_raw: Sequence[float],
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    q10_raw = np.asarray(
        q10_raw,
        dtype=float,
    )
    q50_raw = np.asarray(
        q50_raw,
        dtype=float,
    )
    q90_raw = np.asarray(
        q90_raw,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(q10_raw)
        & np.isfinite(q50_raw)
        & np.isfinite(q90_raw)
    )

    y_true = y_true[finite]
    q10_raw = q10_raw[finite]
    q50_raw = q50_raw[finite]
    q90_raw = q90_raw[finite]

    if len(y_true) == 0:
        result: Dict[str, float] = {
            "n_samples": 0,
        }
        for metric in QUANTILE_METRICS:
            result[metric] = np.nan
        return result

    raw_matrix = np.column_stack(
        [
            q10_raw,
            q50_raw,
            q90_raw,
        ]
    )
    ordered = np.sort(
        raw_matrix,
        axis=1,
    )

    q10_ordered = ordered[:, 0]
    q90_ordered = ordered[:, 2]

    coverage_q10 = float(
        np.mean(
            y_true <= q10_raw
        )
    )
    coverage_q50 = float(
        np.mean(
            y_true <= q50_raw
        )
    )
    coverage_q90 = float(
        np.mean(
            y_true <= q90_raw
        )
    )

    crossing_q10_q50 = (
        q10_raw > q50_raw
    )
    crossing_q50_q90 = (
        q50_raw > q90_raw
    )
    any_crossing = (
        crossing_q10_q50
        | crossing_q50_q90
    )

    covered = (
        (y_true >= q10_ordered)
        & (y_true <= q90_ordered)
    )
    width = (
        q90_ordered
        - q10_ordered
    )
    winkler = winkler_score(
        y_true,
        q10_ordered,
        q90_ordered,
    )

    return {
        "n_samples": int(
            len(y_true)
        ),
        "pinball_q10": pinball_loss(
            y_true,
            q10_raw,
            0.1,
        ),
        "pinball_q50": pinball_loss(
            y_true,
            q50_raw,
            0.5,
        ),
        "pinball_q90": pinball_loss(
            y_true,
            q90_raw,
            0.9,
        ),
        "empirical_coverage_q10": coverage_q10,
        "empirical_coverage_q50": coverage_q50,
        "empirical_coverage_q90": coverage_q90,
        "calibration_error_q10": abs(
            coverage_q10 - 0.1
        ),
        "calibration_error_q50": abs(
            coverage_q50 - 0.5
        ),
        "calibration_error_q90": abs(
            coverage_q90 - 0.9
        ),
        "q50_mae": float(
            mean_absolute_error(
                y_true,
                q50_raw,
            )
        ),
        "q50_rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    q50_raw,
                )
            )
        ),
        "q50_r2": safe_r2(
            y_true,
            q50_raw,
        ),
        "q50_spearman": safe_spearman(
            y_true,
            q50_raw,
        ),
        "picp_80_ordered": float(
            np.mean(covered)
        ),
        "picp_80_calibration_error": abs(
            float(np.mean(covered))
            - 0.8
        ),
        "mpiw_80_ordered": float(
            np.mean(width)
        ),
        "winkler_80_ordered": float(
            np.mean(winkler)
        ),
        "crossing_q10_q50_rate": float(
            np.mean(
                crossing_q10_q50
            )
        ),
        "crossing_q50_q90_rate": float(
            np.mean(
                crossing_q50_q90
            )
        ),
        "any_crossing_rate": float(
            np.mean(
                any_crossing
            )
        ),
    }


def percentile_interval(
    values: Sequence[float],
    confidence_level: float = CI_LEVEL,
) -> Tuple[float, float, int]:
    array = np.asarray(
        values,
        dtype=float,
    )
    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return (
            np.nan,
            np.nan,
            0,
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower = float(
        np.quantile(
            array,
            alpha / 2.0,
        )
    )
    upper = float(
        np.quantile(
            array,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
        int(len(array)),
    )


# =============================================================================
# 4. 输入标准化
# =============================================================================

def normalize_growth_mode(
    value: Any,
) -> str:
    if pd.isna(value):
        return "Unknown"

    raw = str(value).strip()
    if not raw:
        return "Unknown"

    upper = raw.upper()

    direct_patterns = [
        r"^([A-E])$",
        r"^MODE[\s_\-:]*([A-E])$",
        r"^([A-E])[\s_\-:]",
        r"[\s_\-:]([A-E])$",
    ]

    for pattern in direct_patterns:
        match = re.search(
            pattern,
            upper,
        )
        if match:
            return match.group(1)

    standalone = re.search(
        r"\b([A-E])\b",
        upper,
    )
    if standalone:
        return standalone.group(1)

    return "Unknown"


def standardize_metadata(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    defaults: Dict[str, Any] = {
        "sample_id": "",
        "paper_id": "",
        "paper_year_int": np.nan,
        "growth_mode": "Unknown",
        "data_level": "",
        "source_file": "",
        "synthesis_id": "",
    }

    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    out["sample_id"] = (
        out["sample_id"]
        .fillna("")
        .astype(str)
    )
    out["paper_id"] = (
        out["paper_id"]
        .fillna("")
        .astype(str)
    )
    out["paper_year_int"] = pd.to_numeric(
        out["paper_year_int"],
        errors="coerce",
    )
    out["growth_mode_raw"] = (
        out["growth_mode"]
        .fillna("Unknown")
        .astype(str)
    )
    out["growth_mode_normalized"] = (
        out["growth_mode"]
        .map(normalize_growth_mode)
    )
    out["data_level"] = (
        out["data_level"]
        .fillna("")
        .astype(str)
    )

    return out


def load_and_standardize_inputs(
    input_dir: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    l1_target_path = (
        input_dir
        / "l1_target_predictions.csv"
    )
    l1_composite_path = (
        input_dir
        / "l1_composite_predictions.csv"
    )
    l12_path = (
        input_dir
        / "l12_quantile_predictions.csv"
    )
    summary_path = (
        input_dir
        / "validation_summary.json"
    )

    require_file(
        l1_target_path,
        "L1 target predictions",
    )
    require_file(
        l1_composite_path,
        "L1 composite predictions",
    )
    require_file(
        l12_path,
        "L12 quantile predictions",
    )
    require_file(
        summary_path,
        "validation summary",
    )

    l1_target = pd.read_csv(
        l1_target_path,
        low_memory=False,
    )
    l1_composite = pd.read_csv(
        l1_composite_path,
        low_memory=False,
    )
    l12 = pd.read_csv(
        l12_path,
        low_memory=False,
    )

    require_columns(
        l1_target,
        [
            "sample_id",
            "paper_id",
            "target",
            "y_true",
            "y_pred",
        ],
        "l1_target_predictions.csv",
    )
    require_columns(
        l1_composite,
        [
            "sample_id",
            "paper_id",
            "target",
            "y_true",
            "y_pred",
        ],
        "l1_composite_predictions.csv",
    )
    require_columns(
        l12,
        [
            "sample_id",
            "paper_id",
            "target",
            "y_true",
            "q10_raw",
            "q50_raw",
            "q90_raw",
        ],
        "l12_quantile_predictions.csv",
    )

    l1_target = standardize_metadata(
        l1_target
    )
    l1_composite = standardize_metadata(
        l1_composite
    )
    l12 = standardize_metadata(
        l12
    )

    for frame in (
        l1_target,
        l1_composite,
    ):
        frame["target"] = (
            frame["target"]
            .astype(str)
        )
        frame["y_true"] = pd.to_numeric(
            frame["y_true"],
            errors="coerce",
        )
        frame["y_pred"] = pd.to_numeric(
            frame["y_pred"],
            errors="coerce",
        )

    l12["target"] = (
        l12["target"]
        .astype(str)
    )

    for column in (
        "y_true",
        "q10_raw",
        "q50_raw",
        "q90_raw",
    ):
        l12[column] = pd.to_numeric(
            l12[column],
            errors="coerce",
        )

    with summary_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        validation_summary = json.load(
            file
        )

    if not isinstance(
        validation_summary,
        dict,
    ):
        raise RuntimeError(
            "validation_summary.json 根对象不是 dict。"
        )

    return (
        l1_target,
        l1_composite,
        l12,
        validation_summary,
    )


# =============================================================================
# 5. 输入审计
# =============================================================================

def audit_external_validation(
    l1_target: pd.DataFrame,
    l1_composite: pd.DataFrame,
    l12: pd.DataFrame,
    validation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {
        "status": "ok",
        "checks": [],
    }

    if int(
        validation_summary.get(
            "training_overlap_papers",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "validation_summary 显示 external 与 training 存在论文重叠。"
        )

    if validation_summary.get(
        "retraining_performed"
    ) is not False:
        raise RuntimeError(
            "validation_summary 未确认 retraining_performed=False。"
        )

    if validation_summary.get(
        "normalization_refit_on_external_data"
    ) is not False:
        raise RuntimeError(
            "validation_summary 未确认 normalization_refit_on_external_data=False。"
        )

    if validation_summary.get(
        "external_feature_schema_exact_match"
    ) is not True:
        raise RuntimeError(
            "validation_summary 未确认 external feature schema 与 frozen schema 完全一致。"
        )

    audit["checks"].append({
        "name": "frozen_validation_conditions",
        "status": "passed",
        "training_overlap_papers": 0,
        "retraining_performed": False,
        "normalization_refit_on_external_data": False,
        "external_feature_schema_exact_match": True,
    })

    for label, frame, key_columns in (
        (
            "l1_target",
            l1_target,
            [
                "target",
                "sample_id",
            ],
        ),
        (
            "l1_composite",
            l1_composite,
            [
                "target",
                "sample_id",
            ],
        ),
        (
            "l12_quantile",
            l12,
            [
                "target",
                "sample_id",
            ],
        ),
    ):
        blank_paper = (
            frame["paper_id"]
            .astype(str)
            .str.len()
            .eq(0)
        )

        if blank_paper.any():
            raise RuntimeError(
                f"{label} 存在 "
                f"{int(blank_paper.sum())} 条空 paper_id。"
            )

        duplicate = frame.duplicated(
            subset=key_columns,
            keep=False,
        )

        if duplicate.any():
            examples = (
                frame.loc[
                    duplicate,
                    key_columns,
                ]
                .head(10)
                .to_dict("records")
            )
            raise RuntimeError(
                f"{label} 存在重复预测键：{examples}"
            )

        audit["checks"].append({
            "name": f"{label}_prediction_uniqueness",
            "status": "passed",
            "rows": int(len(frame)),
            "papers": int(
                frame["paper_id"]
                .nunique()
            ),
        })

    invalid_l12_level = (
        ~l12["data_level"]
        .isin(
            [
                "L1",
                "L2",
            ]
        )
    )

    if invalid_l12_level.any():
        raise RuntimeError(
            "L12 quantile predictions 中存在非 L1/L2 样本。"
        )

    audit["checks"].append({
        "name": "l12_levels",
        "status": "passed",
        "levels": sorted(
            l12["data_level"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    })

    return audit


# =============================================================================
# 6. 统一 regression prediction table
# =============================================================================

def build_regression_predictions(
    l1_target: pd.DataFrame,
    l1_composite: pd.DataFrame,
    l12: pd.DataFrame,
) -> pd.DataFrame:
    ordinary = l1_target.copy()
    ordinary["source_model"] = (
        "L1_multi_target"
    )

    composite = l1_composite.copy()
    composite["source_model"] = (
        "L1_composite"
    )

    q50 = l12.copy()
    q50["source_model"] = (
        "L12_q50"
    )
    q50["y_pred"] = q50["q50_raw"]

    keep_columns = [
        "source_model",
        "sample_id",
        "paper_id",
        "paper_year_int",
        "growth_mode_raw",
        "growth_mode_normalized",
        "data_level",
        "source_file",
        "synthesis_id",
        "target",
        "y_true",
        "y_pred",
    ]

    for frame in (
        ordinary,
        composite,
        q50,
    ):
        for column in keep_columns:
            if column not in frame.columns:
                frame[column] = np.nan

    combined = pd.concat(
        [
            ordinary[keep_columns],
            composite[keep_columns],
            q50[keep_columns],
        ],
        ignore_index=True,
        sort=False,
    )

    combined["y_true"] = pd.to_numeric(
        combined["y_true"],
        errors="coerce",
    )
    combined["y_pred"] = pd.to_numeric(
        combined["y_pred"],
        errors="coerce",
    )

    finite = (
        np.isfinite(
            combined["y_true"]
        )
        & np.isfinite(
            combined["y_pred"]
        )
    )

    combined = (
        combined.loc[finite]
        .copy()
        .reset_index(drop=True)
    )

    combined["residual"] = (
        combined["y_pred"]
        - combined["y_true"]
    )
    combined["absolute_error"] = np.abs(
        combined["residual"]
    )

    target_rank = {
        target: index
        for index, target in enumerate(
            TARGET_ORDER
        )
    }
    source_rank = {
        source: index
        for index, source in enumerate(
            SOURCE_ORDER
        )
    }

    combined["_target_order"] = (
        combined["target"]
        .map(
            lambda value: target_rank.get(
                value,
                len(target_rank),
            )
        )
    )
    combined["_source_order"] = (
        combined["source_model"]
        .map(
            lambda value: source_rank.get(
                value,
                len(source_rank),
            )
        )
    )

    combined = (
        combined.sort_values(
            [
                "_source_order",
                "_target_order",
                "paper_id",
                "sample_id",
            ]
        )
        .drop(
            columns=[
                "_source_order",
                "_target_order",
            ]
        )
        .reset_index(drop=True)
    )

    return combined


# =============================================================================
# 7. Regression metrics
# =============================================================================

def regression_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
        ],
        dropna=False,
        sort=False,
    )

    for (
        source_model,
        target,
    ), frame in grouped:
        metrics = evaluate_regression(
            frame["y_true"]
            .to_numpy(dtype=float),
            frame["y_pred"]
            .to_numpy(dtype=float),
        )

        years = (
            frame["paper_year_int"]
            .dropna()
        )

        rows.append({
            "source_model": source_model,
            "source_label": SOURCE_LABELS.get(
                source_model,
                source_model,
            ),
            "target": target,
            "target_label": TARGET_LABELS.get(
                target,
                target,
            ),
            "n_samples": metrics[
                "n_samples"
            ],
            "n_papers": int(
                frame["paper_id"]
                .nunique()
            ),
            "year_min": (
                int(years.min())
                if not years.empty
                else np.nan
            ),
            "year_max": (
                int(years.max())
                if not years.empty
                else np.nan
            ),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "spearman": metrics[
                "spearman"
            ],
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 8. Paper-cluster bootstrap for regression
# =============================================================================

def prepare_regression_clusters(
    frame: pd.DataFrame,
) -> Tuple[
    List[str],
    Dict[
        str,
        Tuple[
            np.ndarray,
            np.ndarray,
        ],
    ],
]:
    clusters: Dict[
        str,
        Tuple[
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    for (
        paper_id,
        paper_frame,
    ) in frame.groupby(
        "paper_id",
        sort=False,
    ):
        clusters[
            str(paper_id)
        ] = (
            paper_frame[
                "y_true"
            ].to_numpy(dtype=float),
            paper_frame[
                "y_pred"
            ].to_numpy(dtype=float),
        )

    paper_ids = list(
        clusters.keys()
    )

    return (
        paper_ids,
        clusters,
    )


def bootstrap_regression_frame(
    frame: pd.DataFrame,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    n_samples = int(
        len(frame)
    )
    n_papers = int(
        frame["paper_id"]
        .nunique()
    )

    result: Dict[str, Any] = {
        "bootstrap_status": "ok",
        "bootstrap_requested": int(
            repeats
        ),
        "n_samples": n_samples,
        "n_papers": n_papers,
    }

    if (
        n_samples
        < MIN_BOOTSTRAP_SAMPLES
    ):
        result[
            "bootstrap_status"
        ] = "too_few_samples"

        for metric in REGRESSION_METRICS:
            result[
                f"{metric}_ci_lower"
            ] = np.nan
            result[
                f"{metric}_ci_upper"
            ] = np.nan
            result[
                f"{metric}_valid_bootstrap"
            ] = 0

        return result

    if (
        n_papers
        < MIN_BOOTSTRAP_PAPERS
    ):
        result[
            "bootstrap_status"
        ] = "too_few_papers"

        for metric in REGRESSION_METRICS:
            result[
                f"{metric}_ci_lower"
            ] = np.nan
            result[
                f"{metric}_ci_upper"
            ] = np.nan
            result[
                f"{metric}_valid_bootstrap"
            ] = 0

        return result

    (
        paper_ids,
        clusters,
    ) = prepare_regression_clusters(
        frame
    )

    rng = np.random.default_rng(
        seed
    )

    distributions: Dict[
        str,
        List[float],
    ] = {
        metric: []
        for metric in REGRESSION_METRICS
    }

    for _ in range(
        repeats
    ):
        sampled_papers = rng.choice(
            paper_ids,
            size=len(paper_ids),
            replace=True,
        )

        y_true_parts: List[
            np.ndarray
        ] = []
        y_pred_parts: List[
            np.ndarray
        ] = []

        for paper_id in sampled_papers:
            (
                true_values,
                pred_values,
            ) = clusters[
                str(paper_id)
            ]

            y_true_parts.append(
                true_values
            )
            y_pred_parts.append(
                pred_values
            )

        y_true = np.concatenate(
            y_true_parts
        )
        y_pred = np.concatenate(
            y_pred_parts
        )

        metrics = evaluate_regression(
            y_true,
            y_pred,
        )

        for metric in REGRESSION_METRICS:
            value = metrics[
                metric
            ]
            if np.isfinite(value):
                distributions[
                    metric
                ].append(
                    float(value)
                )

    for metric in REGRESSION_METRICS:
        (
            lower,
            upper,
            valid,
        ) = percentile_interval(
            distributions[
                metric
            ],
            CI_LEVEL,
        )

        result[
            f"{metric}_ci_lower"
        ] = lower
        result[
            f"{metric}_ci_upper"
        ] = upper
        result[
            f"{metric}_valid_bootstrap"
        ] = valid

    minimum_valid = min(
        result[
            f"{metric}_valid_bootstrap"
        ]
        for metric in REGRESSION_METRICS
    )

    if minimum_valid < max(
        100,
        int(
            0.50
            * repeats
        ),
    ):
        result[
            "bootstrap_status"
        ] = (
            "unstable_many_invalid_replicates"
        )

    return result


def regression_bootstrap_analysis(
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
        ],
        dropna=False,
        sort=False,
    )

    total_groups = len(
        grouped
    )
    completed = 0

    for (
        source_model,
        target,
    ), frame in grouped:
        group_seed = (
            seed
            + sum(
                ord(character)
                for character in str(
                    source_model
                )
            )
            + 10
            * sum(
                ord(character)
                for character in str(
                    target
                )
            )
        )

        result = bootstrap_regression_frame(
            frame=frame,
            repeats=repeats,
            seed=group_seed,
        )

        rows.append({
            "source_model": source_model,
            "source_label": SOURCE_LABELS.get(
                source_model,
                source_model,
            ),
            "target": target,
            "target_label": TARGET_LABELS.get(
                target,
                target,
            ),
            **result,
        })

        completed += 1
        print(
            f"[BOOTSTRAP-REG] "
            f"{completed}/{total_groups} "
            f"{source_model} / {target} / "
            f"{result['bootstrap_status']}"
        )

    return pd.DataFrame(
        rows
    )


def merge_regression_metrics_ci(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "source_model",
        "source_label",
        "target",
        "target_label",
    ]

    return metrics.merge(
        bootstrap,
        on=keys,
        how="left",
        suffixes=(
            "",
            "_bootstrap",
        ),
    )


# =============================================================================
# 9. Quantile metrics and bootstrap
# =============================================================================

def build_quantile_scopes(
    predictions: pd.DataFrame,
) -> List[
    Tuple[
        str,
        pd.DataFrame,
    ]
]:
    return [
        (
            "overall",
            predictions,
        ),
        (
            "L1",
            predictions[
                predictions[
                    "data_level"
                ]
                == "L1"
            ],
        ),
        (
            "L2",
            predictions[
                predictions[
                    "data_level"
                ]
                == "L2"
            ],
        ),
    ]


def quantile_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    for (
        scope,
        frame,
    ) in build_quantile_scopes(
        predictions
    ):
        if frame.empty:
            rows.append({
                "scope": scope,
                "status": (
                    "no_valid_external_samples"
                ),
                "n_samples": 0,
                "n_papers": 0,
            })
            continue

        metrics = evaluate_quantiles(
            frame[
                "y_true"
            ].to_numpy(dtype=float),
            frame[
                "q10_raw"
            ].to_numpy(dtype=float),
            frame[
                "q50_raw"
            ].to_numpy(dtype=float),
            frame[
                "q90_raw"
            ].to_numpy(dtype=float),
        )

        rows.append({
            "scope": scope,
            "status": "ok",
            "n_papers": int(
                frame["paper_id"]
                .nunique()
            ),
            **metrics,
        })

    return pd.DataFrame(
        rows
    )


def prepare_quantile_clusters(
    frame: pd.DataFrame,
) -> Tuple[
    List[str],
    Dict[
        str,
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ],
]:
    clusters: Dict[
        str,
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    for (
        paper_id,
        paper_frame,
    ) in frame.groupby(
        "paper_id",
        sort=False,
    ):
        clusters[
            str(paper_id)
        ] = (
            paper_frame[
                "y_true"
            ].to_numpy(dtype=float),
            paper_frame[
                "q10_raw"
            ].to_numpy(dtype=float),
            paper_frame[
                "q50_raw"
            ].to_numpy(dtype=float),
            paper_frame[
                "q90_raw"
            ].to_numpy(dtype=float),
        )

    paper_ids = list(
        clusters.keys()
    )

    return (
        paper_ids,
        clusters,
    )


def bootstrap_quantile_frame(
    frame: pd.DataFrame,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    n_samples = int(
        len(frame)
    )
    n_papers = int(
        frame["paper_id"]
        .nunique()
    )

    result: Dict[str, Any] = {
        "bootstrap_status": "ok",
        "bootstrap_requested": int(
            repeats
        ),
        "n_samples": n_samples,
        "n_papers": n_papers,
    }

    if (
        n_samples
        < MIN_BOOTSTRAP_SAMPLES
    ):
        result[
            "bootstrap_status"
        ] = "too_few_samples"

        for metric in QUANTILE_METRICS:
            result[
                f"{metric}_ci_lower"
            ] = np.nan
            result[
                f"{metric}_ci_upper"
            ] = np.nan
            result[
                f"{metric}_valid_bootstrap"
            ] = 0

        return result

    if (
        n_papers
        < MIN_BOOTSTRAP_PAPERS
    ):
        result[
            "bootstrap_status"
        ] = "too_few_papers"

        for metric in QUANTILE_METRICS:
            result[
                f"{metric}_ci_lower"
            ] = np.nan
            result[
                f"{metric}_ci_upper"
            ] = np.nan
            result[
                f"{metric}_valid_bootstrap"
            ] = 0

        return result

    (
        paper_ids,
        clusters,
    ) = prepare_quantile_clusters(
        frame
    )

    rng = np.random.default_rng(
        seed
    )

    distributions: Dict[
        str,
        List[float],
    ] = {
        metric: []
        for metric in QUANTILE_METRICS
    }

    for _ in range(
        repeats
    ):
        sampled_papers = rng.choice(
            paper_ids,
            size=len(paper_ids),
            replace=True,
        )

        y_parts: List[
            np.ndarray
        ] = []
        q10_parts: List[
            np.ndarray
        ] = []
        q50_parts: List[
            np.ndarray
        ] = []
        q90_parts: List[
            np.ndarray
        ] = []

        for paper_id in sampled_papers:
            (
                y_values,
                q10_values,
                q50_values,
                q90_values,
            ) = clusters[
                str(paper_id)
            ]

            y_parts.append(
                y_values
            )
            q10_parts.append(
                q10_values
            )
            q50_parts.append(
                q50_values
            )
            q90_parts.append(
                q90_values
            )

        metrics = evaluate_quantiles(
            np.concatenate(
                y_parts
            ),
            np.concatenate(
                q10_parts
            ),
            np.concatenate(
                q50_parts
            ),
            np.concatenate(
                q90_parts
            ),
        )

        for metric in QUANTILE_METRICS:
            value = metrics.get(
                metric,
                np.nan,
            )
            if np.isfinite(value):
                distributions[
                    metric
                ].append(
                    float(value)
                )

    for metric in QUANTILE_METRICS:
        (
            lower,
            upper,
            valid,
        ) = percentile_interval(
            distributions[
                metric
            ],
            CI_LEVEL,
        )

        result[
            f"{metric}_ci_lower"
        ] = lower
        result[
            f"{metric}_ci_upper"
        ] = upper
        result[
            f"{metric}_valid_bootstrap"
        ] = valid

    key_valid_counts = [
        result[
            f"{metric}_valid_bootstrap"
        ]
        for metric in (
            "q50_r2",
            "q50_spearman",
            "picp_80_ordered",
        )
    ]

    if min(
        key_valid_counts
    ) < max(
        100,
        int(
            0.50
            * repeats
        ),
    ):
        result[
            "bootstrap_status"
        ] = (
            "unstable_many_invalid_replicates"
        )

    return result


def quantile_bootstrap_analysis(
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    scopes = build_quantile_scopes(
        predictions
    )

    for index, (
        scope,
        frame,
    ) in enumerate(
        scopes,
        start=1,
    ):
        scope_seed = (
            seed
            + 10000
            * index
        )

        result = bootstrap_quantile_frame(
            frame=frame,
            repeats=repeats,
            seed=scope_seed,
        )

        rows.append({
            "scope": scope,
            **result,
        })

        print(
            f"[BOOTSTRAP-Q] "
            f"{scope} / "
            f"{result['bootstrap_status']}"
        )

    return pd.DataFrame(
        rows
    )


def merge_quantile_metrics_ci(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    return metrics.merge(
        bootstrap,
        on=[
            "scope",
        ],
        how="left",
        suffixes=(
            "",
            "_bootstrap",
        ),
    )


# =============================================================================
# 10. Growth-mode stratification
# =============================================================================

def regression_metrics_by_growth_mode(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
            "growth_mode_normalized",
        ],
        dropna=False,
        sort=False,
    )

    for (
        source_model,
        target,
        growth_mode,
    ), frame in grouped:
        metrics = evaluate_regression(
            frame["y_true"]
            .to_numpy(dtype=float),
            frame["y_pred"]
            .to_numpy(dtype=float),
        )

        n_samples = metrics[
            "n_samples"
        ]
        n_papers = int(
            frame["paper_id"]
            .nunique()
        )

        reliable_r2 = (
            n_samples
            >= MIN_MODE_SAMPLES_FOR_R2
            and n_papers
            >= MIN_MODE_PAPERS_FOR_R2
            and np.isfinite(
                metrics["r2"]
            )
        )

        rows.append({
            "source_model": source_model,
            "source_label": SOURCE_LABELS.get(
                source_model,
                source_model,
            ),
            "target": target,
            "target_label": TARGET_LABELS.get(
                target,
                target,
            ),
            "growth_mode": growth_mode,
            "n_samples": n_samples,
            "n_papers": n_papers,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "spearman": metrics[
                "spearman"
            ],
            "r2_reliable_by_minimum_count_rule": bool(
                reliable_r2
            ),
        })

    return pd.DataFrame(
        rows
    )


def quantile_metrics_by_growth_mode(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    grouped = predictions.groupby(
        "growth_mode_normalized",
        dropna=False,
        sort=False,
    )

    for (
        growth_mode,
        frame,
    ) in grouped:
        metrics = evaluate_quantiles(
            frame[
                "y_true"
            ].to_numpy(dtype=float),
            frame[
                "q10_raw"
            ].to_numpy(dtype=float),
            frame[
                "q50_raw"
            ].to_numpy(dtype=float),
            frame[
                "q90_raw"
            ].to_numpy(dtype=float),
        )

        rows.append({
            "growth_mode": growth_mode,
            "n_papers": int(
                frame["paper_id"]
                .nunique()
            ),
            **metrics,
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 11. Sample statistics
# =============================================================================

def build_sample_statistics(
    regression_predictions: pd.DataFrame,
    l12_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    for (
        source_model,
        target,
    ), frame in regression_predictions.groupby(
        [
            "source_model",
            "target",
        ],
        dropna=False,
        sort=False,
    ):
        rows.append({
            "analysis_type": "regression",
            "source_model": source_model,
            "target": target,
            "scope": "overall",
            "n_samples": int(
                len(frame)
            ),
            "n_papers": int(
                frame["paper_id"]
                .nunique()
            ),
            "n_L1": int(
                (
                    frame["data_level"]
                    == "L1"
                ).sum()
            ),
            "n_L2": int(
                (
                    frame["data_level"]
                    == "L2"
                ).sum()
            ),
        })

    for (
        scope,
        frame,
    ) in build_quantile_scopes(
        l12_predictions
    ):
        rows.append({
            "analysis_type": "quantile",
            "source_model": "L12_quantile",
            "target": "y_composite_final_score",
            "scope": scope,
            "n_samples": int(
                len(frame)
            ),
            "n_papers": int(
                frame["paper_id"]
                .nunique()
            ),
            "n_L1": int(
                (
                    frame["data_level"]
                    == "L1"
                ).sum()
            ),
            "n_L2": int(
                (
                    frame["data_level"]
                    == "L2"
                ).sum()
            ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 12. Figures
# =============================================================================

def safe_filename(
    text: str,
) -> str:
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(text),
    )
    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip(
        "._-"
    )

    return (
        value
        if value
        else "figure"
    )


def plot_parity(
    frame: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    source_data_path: Path,
    dpi: int,
) -> None:
    if frame.empty:
        return

    y_true = pd.to_numeric(
        frame["y_true"],
        errors="coerce",
    ).to_numpy(dtype=float)
    y_pred = pd.to_numeric(
        frame["y_pred"],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) == 0:
        return

    minimum = float(
        min(
            np.min(y_true),
            np.min(y_pred),
        )
    )
    maximum = float(
        max(
            np.max(y_true),
            np.max(y_pred),
        )
    )

    if math.isclose(
        minimum,
        maximum,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        delta = (
            1.0
            if minimum == 0
            else abs(minimum) * 0.05
        )
        minimum -= delta
        maximum += delta

    metrics = evaluate_regression(
        y_true,
        y_pred,
    )

    figure, axis = plt.subplots(
        figsize=(
            5.0,
            5.0,
        )
    )

    axis.scatter(
        y_true,
        y_pred,
        s=28,
        alpha=0.75,
    )
    axis.plot(
        [
            minimum,
            maximum,
        ],
        [
            minimum,
            maximum,
        ],
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_xlim(
        minimum,
        maximum,
    )
    axis.set_ylim(
        minimum,
        maximum,
    )

    axis.set_xlabel(
        x_label
    )
    axis.set_ylabel(
        y_label
    )
    axis.set_title(
        title
    )

    annotation = (
        f"n = {metrics['n_samples']}\n"
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"Spearman = {metrics['spearman']:.3f}"
    )

    axis.text(
        0.05,
        0.95,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )

    source = frame.copy()
    source.to_csv(
        source_data_path,
        index=False,
        encoding="utf-8-sig",
    )


def make_parity_figures(
    regression_predictions: pd.DataFrame,
    figures_dir: Path,
    source_data_dir: Path,
    dpi: int,
) -> List[str]:
    created: List[str] = []

    grouped = regression_predictions.groupby(
        [
            "source_model",
            "target",
        ],
        dropna=False,
        sort=False,
    )

    for (
        source_model,
        target,
    ), frame in grouped:
        source_label = SOURCE_LABELS.get(
            source_model,
            source_model,
        )
        target_label = TARGET_LABELS.get(
            target,
            target,
        )

        stem = safe_filename(
            f"parity_{source_model}_{target}"
        )

        figure_path = (
            figures_dir
            / f"{stem}.png"
        )
        source_path = (
            source_data_dir
            / f"{stem}.csv"
        )

        plot_parity(
            frame=frame,
            title=(
                f"{source_label}: "
                f"{target_label}"
            ),
            x_label=(
                f"Observed {target_label}"
            ),
            y_label=(
                f"Predicted {target_label}"
            ),
            output_path=figure_path,
            source_data_path=source_path,
            dpi=dpi,
        )

        if figure_path.is_file():
            created.append(
                str(figure_path)
            )

    return created


def plot_quantile_calibration(
    l12_predictions: pd.DataFrame,
    figures_dir: Path,
    source_data_dir: Path,
    dpi: int,
) -> Optional[str]:
    if l12_predictions.empty:
        return None

    metrics = evaluate_quantiles(
        l12_predictions[
            "y_true"
        ].to_numpy(dtype=float),
        l12_predictions[
            "q10_raw"
        ].to_numpy(dtype=float),
        l12_predictions[
            "q50_raw"
        ].to_numpy(dtype=float),
        l12_predictions[
            "q90_raw"
        ].to_numpy(dtype=float),
    )

    source = pd.DataFrame({
        "nominal_quantile": [
            0.1,
            0.5,
            0.9,
        ],
        "empirical_coverage": [
            metrics[
                "empirical_coverage_q10"
            ],
            metrics[
                "empirical_coverage_q50"
            ],
            metrics[
                "empirical_coverage_q90"
            ],
        ],
    })

    figure, axis = plt.subplots(
        figsize=(
            5.0,
            5.0,
        )
    )

    axis.plot(
        source[
            "nominal_quantile"
        ],
        source[
            "empirical_coverage"
        ],
        marker="o",
    )
    axis.plot(
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_xlim(
        0.0,
        1.0,
    )
    axis.set_ylim(
        0.0,
        1.0,
    )
    axis.set_xlabel(
        "Nominal quantile"
    )
    axis.set_ylabel(
        "Empirical coverage"
    )
    axis.set_title(
        "External quantile calibration"
    )

    figure.tight_layout()

    figure_path = (
        figures_dir
        / "quantile_calibration.png"
    )
    source_path = (
        source_data_dir
        / "quantile_calibration.csv"
    )

    figure.savefig(
        figure_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )

    source.to_csv(
        source_path,
        index=False,
        encoding="utf-8-sig",
    )

    return str(
        figure_path
    )


# =============================================================================
# 13. 论文表格
# =============================================================================

def make_regression_paper_table(
    metrics_with_ci: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_with_ci.empty:
        return pd.DataFrame()

    columns = [
        "source_model",
        "source_label",
        "target",
        "target_label",
        "n_samples",
        "n_papers",
        "mae",
        "mae_ci_lower",
        "mae_ci_upper",
        "rmse",
        "rmse_ci_lower",
        "rmse_ci_upper",
        "r2",
        "r2_ci_lower",
        "r2_ci_upper",
        "spearman",
        "spearman_ci_lower",
        "spearman_ci_upper",
        "bootstrap_status",
    ]

    available = [
        column
        for column in columns
        if column
        in metrics_with_ci.columns
    ]

    return (
        metrics_with_ci[
            available
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def make_quantile_paper_table(
    metrics_with_ci: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_with_ci.empty:
        return pd.DataFrame()

    columns = [
        "scope",
        "status",
        "n_samples",
        "n_papers",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "empirical_coverage_q10",
        "empirical_coverage_q50",
        "empirical_coverage_q90",
        "q50_mae",
        "q50_rmse",
        "q50_r2",
        "q50_r2_ci_lower",
        "q50_r2_ci_upper",
        "q50_spearman",
        "q50_spearman_ci_lower",
        "q50_spearman_ci_upper",
        "picp_80_ordered",
        "picp_80_ordered_ci_lower",
        "picp_80_ordered_ci_upper",
        "mpiw_80_ordered",
        "winkler_80_ordered",
        "any_crossing_rate",
        "bootstrap_status",
    ]

    available = [
        column
        for column in columns
        if column
        in metrics_with_ci.columns
    ]

    return (
        metrics_with_ci[
            available
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 14. 主程序
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze independent external-validation predictions "
            "with paper-level bootstrap confidence intervals."
        )
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "02_validate_external_models.py 输出目录；"
            "默认 outputs/evaluation/external_validation。"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "分析输出目录；默认与 input_dir 相同。"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
        help=(
            "以 paper_id 为单位的 bootstrap 次数，默认2000。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "bootstrap 随机种子。"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=(
            "输出图像 DPI，默认600。"
        ),
    )

    args = parser.parse_args()

    if args.bootstrap <= 0:
        raise ValueError(
            "--bootstrap 必须大于0。"
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi 必须大于0。"
        )

    input_dir = resolve_project_path(
        args.input_dir,
        DEFAULT_INPUT_DIR,
    )

    output_dir = (
        resolve_project_path(
            args.output_dir,
            DEFAULT_OUTPUT_DIR,
        )
        if args.output_dir
        else input_dir
    )

    figures_dir = (
        output_dir
        / "figures"
    )
    source_data_dir = (
        output_dir
        / "source_data"
    )
    tables_dir = (
        output_dir
        / "tables"
    )
    manifests_dir = (
        output_dir
        / "manifests"
    )

    for directory in (
        output_dir,
        figures_dir,
        source_data_dir,
        tables_dir,
        manifests_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    (
        l1_target,
        l1_composite,
        l12,
        validation_summary,
    ) = load_and_standardize_inputs(
        input_dir
    )

    audit = audit_external_validation(
        l1_target,
        l1_composite,
        l12,
        validation_summary,
    )

    audit_path = (
        manifests_dir
        / "external_analysis_audit.json"
    )
    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            ensure_ascii=False,
            indent=2,
        )

    regression_predictions = (
        build_regression_predictions(
            l1_target,
            l1_composite,
            l12,
        )
    )

    regression_predictions_path = (
        source_data_dir
        / "external_regression_predictions.csv"
    )
    regression_predictions.to_csv(
        regression_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    l12_source_path = (
        source_data_dir
        / "external_l12_quantile_predictions.csv"
    )
    l12.to_csv(
        l12_source_path,
        index=False,
        encoding="utf-8-sig",
    )

    regression_metrics_df = (
        regression_metrics(
            regression_predictions
        )
    )

    regression_metrics_path = (
        output_dir
        / "external_regression_metrics.csv"
    )
    regression_metrics_df.to_csv(
        regression_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    regression_bootstrap_df = (
        regression_bootstrap_analysis(
            predictions=regression_predictions,
            repeats=args.bootstrap,
            seed=args.seed,
        )
    )

    regression_bootstrap_path = (
        output_dir
        / "external_regression_bootstrap_ci.csv"
    )
    regression_bootstrap_df.to_csv(
        regression_bootstrap_path,
        index=False,
        encoding="utf-8-sig",
    )

    regression_with_ci = (
        merge_regression_metrics_ci(
            regression_metrics_df,
            regression_bootstrap_df,
        )
    )

    regression_with_ci_path = (
        output_dir
        / "external_regression_metrics_with_ci.csv"
    )
    regression_with_ci.to_csv(
        regression_with_ci_path,
        index=False,
        encoding="utf-8-sig",
    )

    quantile_metrics_df = (
        quantile_metrics(
            l12
        )
    )

    quantile_metrics_path = (
        output_dir
        / "external_quantile_metrics.csv"
    )
    quantile_metrics_df.to_csv(
        quantile_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    quantile_bootstrap_df = (
        quantile_bootstrap_analysis(
            predictions=l12,
            repeats=args.bootstrap,
            seed=args.seed,
        )
    )

    quantile_bootstrap_path = (
        output_dir
        / "external_quantile_bootstrap_ci.csv"
    )
    quantile_bootstrap_df.to_csv(
        quantile_bootstrap_path,
        index=False,
        encoding="utf-8-sig",
    )

    quantile_with_ci = (
        merge_quantile_metrics_ci(
            quantile_metrics_df,
            quantile_bootstrap_df,
        )
    )

    quantile_with_ci_path = (
        output_dir
        / "external_quantile_metrics_with_ci.csv"
    )
    quantile_with_ci.to_csv(
        quantile_with_ci_path,
        index=False,
        encoding="utf-8-sig",
    )

    regression_mode_df = (
        regression_metrics_by_growth_mode(
            regression_predictions
        )
    )

    regression_mode_path = (
        output_dir
        / "external_metrics_by_growth_mode.csv"
    )
    regression_mode_df.to_csv(
        regression_mode_path,
        index=False,
        encoding="utf-8-sig",
    )

    quantile_mode_df = (
        quantile_metrics_by_growth_mode(
            l12
        )
    )

    quantile_mode_path = (
        output_dir
        / "external_quantile_metrics_by_growth_mode.csv"
    )
    quantile_mode_df.to_csv(
        quantile_mode_path,
        index=False,
        encoding="utf-8-sig",
    )

    sample_statistics_df = (
        build_sample_statistics(
            regression_predictions,
            l12,
        )
    )

    sample_statistics_path = (
        output_dir
        / "external_sample_statistics.csv"
    )
    sample_statistics_df.to_csv(
        sample_statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    regression_paper_table = (
        make_regression_paper_table(
            regression_with_ci
        )
    )
    regression_paper_table_path = (
        tables_dir
        / "external_regression_table.csv"
    )
    regression_paper_table.to_csv(
        regression_paper_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    quantile_paper_table = (
        make_quantile_paper_table(
            quantile_with_ci
        )
    )
    quantile_paper_table_path = (
        tables_dir
        / "external_quantile_table.csv"
    )
    quantile_paper_table.to_csv(
        quantile_paper_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    parity_figures = (
        make_parity_figures(
            regression_predictions=regression_predictions,
            figures_dir=figures_dir,
            source_data_dir=source_data_dir,
            dpi=args.dpi,
        )
    )

    calibration_figure = (
        plot_quantile_calibration(
            l12_predictions=l12,
            figures_dir=figures_dir,
            source_data_dir=source_data_dir,
            dpi=args.dpi,
        )
    )

    analysis_summary = {
        "input_dir": str(
            input_dir
        ),
        "output_dir": str(
            output_dir
        ),
        "bootstrap_repeats": int(
            args.bootstrap
        ),
        "bootstrap_seed": int(
            args.seed
        ),
        "bootstrap_unit": "paper_id",
        "confidence_level": CI_LEVEL,
        "external_rows": validation_summary.get(
            "external_rows"
        ),
        "external_papers": validation_summary.get(
            "external_papers"
        ),
        "training_overlap_papers": validation_summary.get(
            "training_overlap_papers"
        ),
        "retraining_performed": validation_summary.get(
            "retraining_performed"
        ),
        "normalization_refit_on_external_data": validation_summary.get(
            "normalization_refit_on_external_data"
        ),
        "external_feature_schema_exact_match": validation_summary.get(
            "external_feature_schema_exact_match"
        ),
        "regression_prediction_rows": int(
            len(
                regression_predictions
            )
        ),
        "l12_quantile_prediction_rows": int(
            len(l12)
        ),
        "regression_metric_groups": int(
            len(
                regression_metrics_df
            )
        ),
        "quantile_scopes": int(
            len(
                quantile_metrics_df
            )
        ),
        "parity_figures": parity_figures,
        "quantile_calibration_figure": calibration_figure,
        "raw_quantile_definition": {
            "q10": (
                "raw conditional q10 prediction; "
                "used for pinball loss and empirical coverage"
            ),
            "q50": (
                "raw conditional q50 prediction; "
                "used as the central ranking/regression score"
            ),
            "q90": (
                "raw conditional q90 prediction; "
                "used for pinball loss and empirical coverage"
            ),
        },
        "ordered_quantile_definition": (
            "Per-sample sorted q10/q50/q90 values are used only "
            "for descriptive interval diagnostics such as PICP, MPIW "
            "and Winkler score; they do not replace the raw q50 ranking score."
        ),
        "growth_mode_r2_minimum_rule": {
            "minimum_samples": MIN_MODE_SAMPLES_FOR_R2,
            "minimum_papers": MIN_MODE_PAPERS_FOR_R2,
        },
    }

    summary_path = (
        output_dir
        / "external_analysis_summary.json"
    )
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analysis_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 96
    )
    print(
        "External Validation Analysis Finished"
    )
    print(
        "=" * 96
    )
    print(
        f"[OK] input dir             : {input_dir}"
    )
    print(
        f"[OK] external papers       : "
        f"{validation_summary.get('external_papers')}"
    )
    print(
        f"[OK] regression rows       : "
        f"{len(regression_predictions)}"
    )
    print(
        f"[OK] L12 quantile rows     : "
        f"{len(l12)}"
    )
    print(
        f"[OK] bootstrap repeats     : "
        f"{args.bootstrap}"
    )
    print(
        f"[OK] bootstrap unit        : paper_id"
    )
    print(
        f"[OK] regression metrics    : "
        f"{regression_with_ci_path}"
    )
    print(
        f"[OK] quantile metrics      : "
        f"{quantile_with_ci_path}"
    )
    print(
        f"[OK] growth-mode metrics   : "
        f"{regression_mode_path}"
    )
    print(
        f"[OK] quantile by mode      : "
        f"{quantile_mode_path}"
    )
    print(
        f"[OK] figures               : "
        f"{figures_dir}"
    )
    print(
        f"[OK] tables                : "
        f"{tables_dir}"
    )
    print(
        f"[OK] source data           : "
        f"{source_data_dir}"
    )
    print(
        f"[OK] manifests             : "
        f"{manifests_dir}"
    )
    print(
        f"[OK] summary               : "
        f"{summary_path}"
    )
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
