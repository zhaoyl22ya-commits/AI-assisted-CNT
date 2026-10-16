# -*- coding: utf-8 -*-
"""
validate_generalization.py

CNT 3.0 — Article-ready validation analysis

本代码不重新训练模型。它读取已经由以下代码生成的验证结果：
    models/training/train_l1_targets.py
    models/training/train_l1_composite.py

并将随机4:1、跨论文五折和跨年份验证统一整理为可以直接用于论文的：
    1. 三种验证指标表；
    2. 以 paper_id 为重采样单位的 bootstrap 95% 置信区间；
    3. A–E 生长模式分层结果；
    4. paper-grouped OOF parity plots；
    5. temporal holdout parity plots；
    6. 主文表格、补充表格和每张图对应的 source data；
    7. 完整输入与划分审计记录。

默认输入目录
------------
<project_dir>/
├─ outputs/models/l1_targets/
│  ├─ L1_metrics_fold.csv
│  ├─ L1_metrics_summary.csv
│  ├─ L1_predictions.csv
│  └─ split_manifest.csv
└─ outputs/models/l1_composite/
   ├─ L1_composite_metrics_fold.csv
   ├─ L1_composite_metrics_summary.csv
   ├─ L1_composite_predictions.csv
   └─ composite_split_manifest.csv

默认输出目录
------------
<project_dir>/outputs/external_validation/generalization/
├─ figures/
├─ source_data/
├─ tables/
├─ manifests/
├─ three_split_metrics_fold_recomputed.csv
├─ three_split_metrics_pooled.csv
├─ three_split_bootstrap_ci.csv
├─ three_split_metrics_with_ci.csv
├─ metrics_by_growth_mode.csv
├─ target_sample_statistics.csv
├─ random_holdout_predictions.csv
├─ paper_grouped_oof_predictions.csv
└─ temporal_holdout_predictions.csv

运行
----
python validate_generalization.py

可选参数
--------
python validate_generalization.py --bootstrap 2000 --dpi 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. 固定定义
# =============================================================================

SPLIT_ORDER = [
    "random_holdout",
    "paper_grouped_cv",
    "temporal_holdout",
]

SPLIT_LABELS = {
    "random_holdout": "Random 4:1",
    "paper_grouped_cv": "Paper-grouped CV",
    "temporal_holdout": "Temporal holdout",
}

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
    "y_diameter_nm": "Diameter",
    "y_length_mm": "Length",
    "y_bundle_density_per_um": "Bundle density",
    "y_cnt_density_per_um": "CNT density",
    "y_ig_id_ratio_unified": r"$I_G/I_D$",
    "y_orientation_score": "Orientation",
    "y_purity_score": "Purity",
    "y_composite_final_score": "Composite",
}

TARGET_TABLE_LABELS = {
    "y_growth_quality_score": "Growth quality score",
    "y_diameter_nm": "Diameter (nm)",
    "y_length_mm": "Length (mm)",
    "y_bundle_density_per_um": "Bundle density (μm⁻¹)",
    "y_cnt_density_per_um": "CNT density (μm⁻¹)",
    "y_ig_id_ratio_unified": "IG/ID ratio",
    "y_orientation_score": "Orientation score",
    "y_purity_score": "Purity score",
    "y_composite_final_score": "Composite score",
}

METRIC_ORDER = ["mae", "rmse", "r2", "spearman"]
CI_LEVEL = 0.95

MIN_BOOTSTRAP_SAMPLES = 10
MIN_BOOTSTRAP_PAPERS = 3

MIN_MODE_SAMPLES_FOR_R2 = 10
MIN_MODE_PAPERS_FOR_R2 = 3

DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_RANDOM_SEED = 20260722
DEFAULT_DPI = 600


# =============================================================================
# 2. 路径与数据结构
# =============================================================================

@dataclass(frozen=True)
class Paths:
    project_dir: str
    ordinary_dir: str
    composite_dir: str
    ordinary_predictions: str
    ordinary_metrics_fold: str
    ordinary_metrics_summary: str
    ordinary_split_manifest: str
    composite_predictions: str
    composite_metrics_fold: str
    composite_metrics_summary: str
    composite_split_manifest: str
    output_dir: str
    figures_dir: str
    source_data_dir: str
    tables_dir: str
    manifests_dir: str


def build_paths(project_dir: str, output_dir: Optional[str] = None) -> Paths:
    project_dir = os.path.abspath(project_dir)

    ordinary_dir = os.path.join(
        project_dir,
        "outputs", "models", "l1_targets",
    )
    composite_dir = os.path.join(
        project_dir,
        "outputs", "models", "l1_composite",
    )

    resolved_output = (
        os.path.abspath(output_dir)
        if output_dir
        else os.path.join(project_dir, "outputs", "external_validation", "generalization")
    )

    return Paths(
        project_dir=project_dir,
        ordinary_dir=ordinary_dir,
        composite_dir=composite_dir,
        ordinary_predictions=os.path.join(
            ordinary_dir,
            "L1_predictions.csv",
        ),
        ordinary_metrics_fold=os.path.join(
            ordinary_dir,
            "L1_metrics_fold.csv",
        ),
        ordinary_metrics_summary=os.path.join(
            ordinary_dir,
            "L1_metrics_summary.csv",
        ),
        ordinary_split_manifest=os.path.join(
            ordinary_dir,
            "split_manifest.csv",
        ),
        composite_predictions=os.path.join(
            composite_dir,
            "L1_composite_predictions.csv",
        ),
        composite_metrics_fold=os.path.join(
            composite_dir,
            "L1_composite_metrics_fold.csv",
        ),
        composite_metrics_summary=os.path.join(
            composite_dir,
            "L1_composite_metrics_summary.csv",
        ),
        composite_split_manifest=os.path.join(
            composite_dir,
            "composite_split_manifest.csv",
        ),
        output_dir=resolved_output,
        figures_dir=os.path.join(resolved_output, "figures"),
        source_data_dir=os.path.join(resolved_output, "source_data"),
        tables_dir=os.path.join(resolved_output, "tables"),
        manifests_dir=os.path.join(resolved_output, "manifests"),
    )


def create_output_directories(paths: Paths) -> None:
    for directory in [
        paths.output_dir,
        paths.figures_dir,
        paths.source_data_dir,
        paths.tables_dir,
        paths.manifests_dir,
    ]:
        os.makedirs(directory, exist_ok=True)


def ensure_inputs_exist(paths: Paths) -> None:
    required = [
        paths.ordinary_predictions,
        paths.ordinary_metrics_fold,
        paths.ordinary_split_manifest,
        paths.composite_predictions,
        paths.composite_metrics_fold,
        paths.composite_split_manifest,
    ]

    missing = [
        path
        for path in required
        if not os.path.isfile(path)
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要输入文件：\n"
            + "\n".join(missing)
        )


# =============================================================================
# 3. 基础统计函数
# =============================================================================

def to_numeric_array(values: Iterable[Any]) -> np.ndarray:
    return pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return float("nan")

    if np.allclose(y_true, y_true[0]):
        return float("nan")

    return float(r2_score(y_true, y_pred))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
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

    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        return float("nan")

    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    finite = (
        np.isfinite(y_true_array)
        & np.isfinite(y_pred_array)
    )

    y_true_array = y_true_array[finite]
    y_pred_array = y_pred_array[finite]

    if len(y_true_array) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }

    mae = float(
        mean_absolute_error(
            y_true_array,
            y_pred_array,
        )
    )
    rmse = float(
        math.sqrt(
            mean_squared_error(
                y_true_array,
                y_pred_array,
            )
        )
    )

    return {
        "n": int(len(y_true_array)),
        "mae": mae,
        "rmse": rmse,
        "r2": safe_r2(y_true_array, y_pred_array),
        "spearman": safe_spearman(y_true_array, y_pred_array),
    }


def percentile_interval(
    values: Sequence[float],
    confidence_level: float = CI_LEVEL,
) -> Tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return np.nan, np.nan, 0

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(array, alpha / 2.0))
    upper = float(np.quantile(array, 1.0 - alpha / 2.0))

    return lower, upper, int(len(array))


# =============================================================================
# 4. 输入标准化
# =============================================================================

def normalize_growth_mode(value: Any) -> str:
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
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    standalone = re.search(r"\b([A-E])\b", upper)
    if standalone:
        return standalone.group(1)

    return raw


def standardize_prediction_table(
    frame: pd.DataFrame,
    source_model: str,
) -> pd.DataFrame:
    required = [
        "sample_id",
        "paper_id",
        "target",
        "split_type",
        "fold",
        "y_true",
        "y_pred",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source_model} predictions 缺少必要列：{missing}"
        )

    out = frame.copy()

    optional_defaults: Dict[str, Any] = {
        "paper_year_int": np.nan,
        "growth_mode": "Unknown",
        "data_level": "",
        "source_file": "",
        "synthesis_id": "",
        "temporal_cutoff_year": np.nan,
    }

    for column, default in optional_defaults.items():
        if column not in out.columns:
            out[column] = default

    out["sample_id"] = out["sample_id"].fillna("").astype(str)
    out["paper_id"] = out["paper_id"].fillna("").astype(str)
    out["target"] = out["target"].astype(str)
    out["split_type"] = out["split_type"].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    out["paper_year_int"] = pd.to_numeric(
        out["paper_year_int"],
        errors="coerce",
    )
    out["temporal_cutoff_year"] = pd.to_numeric(
        out["temporal_cutoff_year"],
        errors="coerce",
    )
    out["growth_mode_raw"] = out["growth_mode"].fillna("Unknown").astype(str)
    out["growth_mode_normalized"] = out["growth_mode"].map(
        normalize_growth_mode
    )
    out["source_model"] = source_model
    out["residual"] = out["y_pred"] - out["y_true"]
    out["absolute_error"] = np.abs(out["residual"])

    invalid = (
        ~np.isfinite(out["y_true"])
        | ~np.isfinite(out["y_pred"])
    )

    if invalid.any():
        print(
            f"[WARN] {source_model}: 删除 "
            f"{int(invalid.sum())} 条无有限真实值/预测值的记录。"
        )
        out = out.loc[~invalid].copy()

    return out


def load_predictions(paths: Paths) -> pd.DataFrame:
    ordinary = pd.read_csv(
        paths.ordinary_predictions,
        low_memory=False,
    )
    composite = pd.read_csv(
        paths.composite_predictions,
        low_memory=False,
    )

    ordinary = standardize_prediction_table(
        ordinary,
        "L1_multi_target",
    )
    composite = standardize_prediction_table(
        composite,
        "L1_composite",
    )

    combined = pd.concat(
        [ordinary, composite],
        ignore_index=True,
        sort=False,
    )

    combined = combined[
        combined["split_type"].isin(SPLIT_ORDER)
    ].copy()

    target_rank = {
        target: index
        for index, target in enumerate(TARGET_ORDER)
    }
    split_rank = {
        split_type: index
        for index, split_type in enumerate(SPLIT_ORDER)
    }

    combined["_target_order"] = combined["target"].map(
        lambda value: target_rank.get(value, len(target_rank))
    )
    combined["_split_order"] = combined["split_type"].map(
        lambda value: split_rank.get(value, len(split_rank))
    )

    combined = combined.sort_values(
        [
            "_target_order",
            "_split_order",
            "fold",
            "paper_id",
            "sample_id",
        ]
    ).drop(
        columns=["_target_order", "_split_order"]
    )

    return combined.reset_index(drop=True)


def standardize_manifest(
    frame: pd.DataFrame,
    source_model: str,
) -> pd.DataFrame:
    required = [
        "sample_id",
        "paper_id",
        "target",
        "split_type",
        "fold",
        "role",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source_model} split manifest 缺少必要列：{missing}"
        )

    out = frame.copy()

    for column, default in {
        "paper_year_int": np.nan,
        "growth_mode": "Unknown",
        "data_level": "",
        "temporal_cutoff_year": np.nan,
    }.items():
        if column not in out.columns:
            out[column] = default

    out["sample_id"] = out["sample_id"].fillna("").astype(str)
    out["paper_id"] = out["paper_id"].fillna("").astype(str)
    out["target"] = out["target"].astype(str)
    out["split_type"] = out["split_type"].astype(str)
    out["role"] = out["role"].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce")
    out["paper_year_int"] = pd.to_numeric(
        out["paper_year_int"],
        errors="coerce",
    )
    out["temporal_cutoff_year"] = pd.to_numeric(
        out["temporal_cutoff_year"],
        errors="coerce",
    )
    out["source_model"] = source_model

    return out


def load_manifests(paths: Paths) -> pd.DataFrame:
    ordinary = standardize_manifest(
        pd.read_csv(
            paths.ordinary_split_manifest,
            low_memory=False,
        ),
        "L1_multi_target",
    )

    composite = standardize_manifest(
        pd.read_csv(
            paths.composite_split_manifest,
            low_memory=False,
        ),
        "L1_composite",
    )

    return pd.concat(
        [ordinary, composite],
        ignore_index=True,
        sort=False,
    )


# =============================================================================
# 5. 划分与预测审计
# =============================================================================

def audit_predictions_and_splits(
    predictions: pd.DataFrame,
    manifests: pd.DataFrame,
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {
        "status": "ok",
        "checks": [],
        "warnings": [],
    }

    blank_paper = predictions["paper_id"].str.len() == 0
    if blank_paper.any():
        raise RuntimeError(
            f"预测表存在 {int(blank_paper.sum())} 条空 paper_id。"
        )

    duplicate_keys = predictions.duplicated(
        subset=[
            "source_model",
            "target",
            "split_type",
            "fold",
            "sample_id",
        ],
        keep=False,
    )

    if duplicate_keys.any():
        examples = predictions.loc[
            duplicate_keys,
            [
                "source_model",
                "target",
                "split_type",
                "fold",
                "sample_id",
            ],
        ].head(10)

        raise RuntimeError(
            "预测表存在重复的 target/split/fold/sample_id：\n"
            + examples.to_string(index=False)
        )

    audit["checks"].append({
        "name": "prediction_key_uniqueness",
        "status": "passed",
        "rows": int(len(predictions)),
    })

    grouped_manifest = manifests[
        manifests["split_type"] == "paper_grouped_cv"
    ]

    grouped_checks: List[Dict[str, Any]] = []

    for (
        source_model,
        target,
        fold,
    ), group in grouped_manifest.groupby(
        ["source_model", "target", "fold"],
        dropna=False,
    ):
        train_papers = set(
            group.loc[
                group["role"] == "train",
                "paper_id",
            ].astype(str)
        )
        test_papers = set(
            group.loc[
                group["role"] == "test",
                "paper_id",
            ].astype(str)
        )

        overlap = train_papers.intersection(test_papers)

        if overlap:
            raise RuntimeError(
                f"{source_model}/{target}/fold={fold} "
                f"存在 {len(overlap)} 篇训练—测试论文重叠。"
            )

        test_ids_manifest = set(
            group.loc[
                group["role"] == "test",
                "sample_id",
            ].astype(str)
        )

        test_ids_prediction = set(
            predictions.loc[
                (
                    (predictions["source_model"] == source_model)
                    & (predictions["target"] == target)
                    & (
                        predictions["split_type"]
                        == "paper_grouped_cv"
                    )
                    & (predictions["fold"] == fold)
                ),
                "sample_id",
            ].astype(str)
        )

        if test_ids_manifest != test_ids_prediction:
            raise RuntimeError(
                f"{source_model}/{target}/fold={fold} "
                "manifest测试样本与prediction样本不一致。"
            )

        grouped_checks.append({
            "source_model": source_model,
            "target": target,
            "fold": int(fold),
            "train_papers": len(train_papers),
            "test_papers": len(test_papers),
            "paper_overlap": 0,
            "test_samples": len(test_ids_manifest),
        })

    audit["checks"].append({
        "name": "paper_grouped_no_paper_leakage",
        "status": "passed",
        "fold_checks": grouped_checks,
    })

    grouped_predictions = predictions[
        predictions["split_type"] == "paper_grouped_cv"
    ]

    duplicated_oof = grouped_predictions.duplicated(
        subset=[
            "source_model",
            "target",
            "sample_id",
        ],
        keep=False,
    )

    if duplicated_oof.any():
        examples = grouped_predictions.loc[
            duplicated_oof,
            ["source_model", "target", "sample_id", "fold"],
        ].head(10)

        raise RuntimeError(
            "paper-grouped OOF 中同一样本重复进入测试集：\n"
            + examples.to_string(index=False)
        )

    audit["checks"].append({
        "name": "paper_grouped_oof_once_per_sample",
        "status": "passed",
        "rows": int(len(grouped_predictions)),
    })

    temporal_manifest = manifests[
        manifests["split_type"] == "temporal_holdout"
    ]

    temporal_checks: List[Dict[str, Any]] = []

    for (
        source_model,
        target,
        fold,
    ), group in temporal_manifest.groupby(
        ["source_model", "target", "fold"],
        dropna=False,
    ):
        train_years = group.loc[
            group["role"] == "train",
            "paper_year_int",
        ].dropna()

        test_years = group.loc[
            group["role"] == "test",
            "paper_year_int",
        ].dropna()

        if not train_years.empty and not test_years.empty:
            train_max = int(train_years.max())
            test_min = int(test_years.min())

            if train_max >= test_min:
                raise RuntimeError(
                    f"{source_model}/{target} 时间划分错误："
                    f"train_max={train_max}, test_min={test_min}"
                )

            temporal_checks.append({
                "source_model": source_model,
                "target": target,
                "fold": int(fold),
                "train_year_max": train_max,
                "test_year_min": test_min,
                "n_train": int((group["role"] == "train").sum()),
                "n_test": int((group["role"] == "test").sum()),
            })

    audit["checks"].append({
        "name": "temporal_chronology",
        "status": "passed",
        "target_checks": temporal_checks,
    })

    return audit


# =============================================================================
# 6. 指标重算
# =============================================================================

def recompute_fold_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
            "split_type",
            "fold",
        ],
        dropna=False,
        sort=False,
    )

    for (
        source_model,
        target,
        split_type,
        fold,
    ), frame in grouped:
        metrics = evaluate_regression(
            frame["y_true"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=float),
        )

        rows.append({
            "source_model": source_model,
            "target": target,
            "target_label": TARGET_TABLE_LABELS.get(
                target,
                target,
            ),
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(
                split_type,
                split_type,
            ),
            "fold": int(fold),
            "n_samples": metrics["n"],
            "n_papers": int(frame["paper_id"].nunique()),
            "year_min": (
                int(frame["paper_year_int"].min())
                if frame["paper_year_int"].notna().any()
                else np.nan
            ),
            "year_max": (
                int(frame["paper_year_int"].max())
                if frame["paper_year_int"].notna().any()
                else np.nan
            ),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "spearman": metrics["spearman"],
        })

    return pd.DataFrame(rows)


def recompute_pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
            "split_type",
        ],
        dropna=False,
        sort=False,
    )

    for (
        source_model,
        target,
        split_type,
    ), frame in grouped:
        metrics = evaluate_regression(
            frame["y_true"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=float),
        )

        rows.append({
            "source_model": source_model,
            "target": target,
            "target_label": TARGET_TABLE_LABELS.get(
                target,
                target,
            ),
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(
                split_type,
                split_type,
            ),
            "n_samples": metrics["n"],
            "n_papers": int(frame["paper_id"].nunique()),
            "n_folds": int(frame["fold"].nunique()),
            "year_min": (
                int(frame["paper_year_int"].min())
                if frame["paper_year_int"].notna().any()
                else np.nan
            ),
            "year_max": (
                int(frame["paper_year_int"].max())
                if frame["paper_year_int"].notna().any()
                else np.nan
            ),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "spearman": metrics["spearman"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# 7. Cluster bootstrap 95% CI
# =============================================================================

def prepare_paper_clusters(
    frame: pd.DataFrame,
) -> Tuple[List[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    clusters: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for paper_id, paper_frame in frame.groupby(
        "paper_id",
        sort=False,
    ):
        clusters[str(paper_id)] = (
            paper_frame["y_true"].to_numpy(dtype=float),
            paper_frame["y_pred"].to_numpy(dtype=float),
        )

    paper_ids = list(clusters.keys())
    return paper_ids, clusters


def cluster_bootstrap_metrics(
    frame: pd.DataFrame,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    n_samples = int(len(frame))
    n_papers = int(frame["paper_id"].nunique())

    result: Dict[str, Any] = {
        "bootstrap_status": "ok",
        "bootstrap_requested": int(repeats),
        "n_samples": n_samples,
        "n_papers": n_papers,
    }

    if n_samples < MIN_BOOTSTRAP_SAMPLES:
        result["bootstrap_status"] = "too_few_samples"
        for metric in METRIC_ORDER:
            result[f"{metric}_ci_lower"] = np.nan
            result[f"{metric}_ci_upper"] = np.nan
            result[f"{metric}_valid_bootstrap"] = 0
        return result

    if n_papers < MIN_BOOTSTRAP_PAPERS:
        result["bootstrap_status"] = "too_few_papers"
        for metric in METRIC_ORDER:
            result[f"{metric}_ci_lower"] = np.nan
            result[f"{metric}_ci_upper"] = np.nan
            result[f"{metric}_valid_bootstrap"] = 0
        return result

    paper_ids, clusters = prepare_paper_clusters(frame)
    rng = np.random.default_rng(seed)

    distributions: Dict[str, List[float]] = {
        metric: []
        for metric in METRIC_ORDER
    }

    for _ in range(repeats):
        sampled_papers = rng.choice(
            paper_ids,
            size=len(paper_ids),
            replace=True,
        )

        y_true_parts: List[np.ndarray] = []
        y_pred_parts: List[np.ndarray] = []

        for paper_id in sampled_papers:
            true_values, pred_values = clusters[str(paper_id)]
            y_true_parts.append(true_values)
            y_pred_parts.append(pred_values)

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)

        metrics = evaluate_regression(y_true, y_pred)

        for metric in METRIC_ORDER:
            value = metrics[metric]
            if np.isfinite(value):
                distributions[metric].append(float(value))

    for metric in METRIC_ORDER:
        lower, upper, valid = percentile_interval(
            distributions[metric],
            confidence_level=CI_LEVEL,
        )
        result[f"{metric}_ci_lower"] = lower
        result[f"{metric}_ci_upper"] = upper
        result[f"{metric}_valid_bootstrap"] = valid

    minimum_valid = min(
        result[f"{metric}_valid_bootstrap"]
        for metric in METRIC_ORDER
    )

    if minimum_valid < max(100, int(0.50 * repeats)):
        result["bootstrap_status"] = "unstable_many_invalid_replicates"

    return result


def run_bootstrap_analysis(
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
            "split_type",
        ],
        dropna=False,
        sort=False,
    )

    total_groups = len(grouped)
    completed = 0

    for (
        source_model,
        target,
        split_type,
    ), frame in grouped:
        group_seed = (
            seed
            + sum(ord(character) for character in target)
            + 1000 * SPLIT_ORDER.index(split_type)
        )

        result = cluster_bootstrap_metrics(
            frame=frame,
            repeats=repeats,
            seed=group_seed,
        )

        rows.append({
            "source_model": source_model,
            "target": target,
            "target_label": TARGET_TABLE_LABELS.get(
                target,
                target,
            ),
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(
                split_type,
                split_type,
            ),
            **result,
        })

        completed += 1
        print(
            f"[BOOTSTRAP] {completed}/{total_groups} "
            f"{target} / {split_type} / {result['bootstrap_status']}"
        )

    return pd.DataFrame(rows)


def merge_metrics_and_ci(
    pooled_metrics: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "source_model",
        "target",
        "target_label",
        "split_type",
        "split_label",
    ]

    merged = pooled_metrics.merge(
        bootstrap_ci,
        on=keys,
        how="left",
        suffixes=("", "_bootstrap"),
    )

    return merged


# =============================================================================
# 8. 生长模式分层结果
# =============================================================================

def metrics_by_growth_mode(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grouped = predictions.groupby(
        [
            "source_model",
            "target",
            "split_type",
            "growth_mode_normalized",
        ],
        dropna=False,
        sort=False,
    )

    for (
        source_model,
        target,
        split_type,
        growth_mode,
    ), frame in grouped:
        metrics = evaluate_regression(
            frame["y_true"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=float),
        )

        n_samples = metrics["n"]
        n_papers = int(frame["paper_id"].nunique())

        reliable_r2 = (
            n_samples >= MIN_MODE_SAMPLES_FOR_R2
            and n_papers >= MIN_MODE_PAPERS_FOR_R2
            and np.isfinite(metrics["r2"])
        )

        rows.append({
            "source_model": source_model,
            "target": target,
            "target_label": TARGET_TABLE_LABELS.get(
                target,
                target,
            ),
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(
                split_type,
                split_type,
            ),
            "growth_mode": growth_mode,
            "n_samples": n_samples,
            "n_papers": n_papers,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "spearman": metrics["spearman"],
            "r2_interpretation_status": (
                "usable"
                if reliable_r2
                else "descriptive_only_small_group"
            ),
        })

    result = pd.DataFrame(rows)

    mode_order = {
        mode: index
        for index, mode in enumerate(
            ["A", "B", "C", "D", "E", "Unknown"]
        )
    }

    result["_mode_order"] = result["growth_mode"].map(
        lambda value: mode_order.get(value, len(mode_order))
    )
    result["_target_order"] = result["target"].map(
        lambda value: (
            TARGET_ORDER.index(value)
            if value in TARGET_ORDER
            else len(TARGET_ORDER)
        )
    )
    result["_split_order"] = result["split_type"].map(
        lambda value: (
            SPLIT_ORDER.index(value)
            if value in SPLIT_ORDER
            else len(SPLIT_ORDER)
        )
    )

    result = result.sort_values(
        ["_target_order", "_split_order", "_mode_order"]
    ).drop(
        columns=["_target_order", "_split_order", "_mode_order"]
    )

    return result.reset_index(drop=True)


# =============================================================================
# 9. 样本统计与主文表格
# =============================================================================

def build_target_sample_statistics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (
        target,
        split_type,
    ), frame in predictions.groupby(
        ["target", "split_type"],
        sort=False,
    ):
        mode_counts = (
            frame["growth_mode_normalized"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        rows.append({
            "target": target,
            "target_label": TARGET_TABLE_LABELS.get(target, target),
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(split_type, split_type),
            "n_samples": int(len(frame)),
            "n_papers": int(frame["paper_id"].nunique()),
            "n_folds": int(frame["fold"].nunique()),
            "year_min": (
                int(frame["paper_year_int"].min())
                if frame["paper_year_int"].notna().any()
                else np.nan
            ),
            "year_max": (
                int(frame["paper_year_int"].max())
                if frame["paper_year_int"].notna().any()
                else np.nan
            ),
            "growth_mode_counts_json": json.dumps(
                mode_counts,
                ensure_ascii=False,
                sort_keys=True,
            ),
        })

    return pd.DataFrame(rows)


def format_estimate_ci(
    estimate: Any,
    lower: Any,
    upper: Any,
    digits: int = 3,
) -> str:
    if not np.isfinite(pd.to_numeric(estimate, errors="coerce")):
        return ""

    estimate_value = float(estimate)

    if (
        np.isfinite(pd.to_numeric(lower, errors="coerce"))
        and np.isfinite(pd.to_numeric(upper, errors="coerce"))
    ):
        return (
            f"{estimate_value:.{digits}f} "
            f"[{float(lower):.{digits}f}, {float(upper):.{digits}f}]"
        )

    return f"{estimate_value:.{digits}f}"


def build_main_table(
    metrics_with_ci: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for target in TARGET_ORDER:
        target_frame = metrics_with_ci[
            metrics_with_ci["target"] == target
        ]

        if target_frame.empty:
            continue

        row: Dict[str, Any] = {
            "Target": TARGET_TABLE_LABELS.get(target, target),
        }

        for split_type in SPLIT_ORDER:
            split_frame = target_frame[
                target_frame["split_type"] == split_type
            ]

            prefix = SPLIT_LABELS[split_type]

            if split_frame.empty:
                row[f"{prefix}: n"] = ""
                row[f"{prefix}: R² [95% CI]"] = ""
                row[f"{prefix}: Spearman [95% CI]"] = ""
                row[f"{prefix}: MAE [95% CI]"] = ""
                row[f"{prefix}: RMSE [95% CI]"] = ""
                continue

            item = split_frame.iloc[0]

            row[f"{prefix}: n"] = int(item["n_samples"])
            row[f"{prefix}: R² [95% CI]"] = format_estimate_ci(
                item["r2"],
                item["r2_ci_lower"],
                item["r2_ci_upper"],
            )
            row[f"{prefix}: Spearman [95% CI]"] = format_estimate_ci(
                item["spearman"],
                item["spearman_ci_lower"],
                item["spearman_ci_upper"],
            )
            row[f"{prefix}: MAE [95% CI]"] = format_estimate_ci(
                item["mae"],
                item["mae_ci_lower"],
                item["mae_ci_upper"],
            )
            row[f"{prefix}: RMSE [95% CI]"] = format_estimate_ci(
                item["rmse"],
                item["rmse_ci_lower"],
                item["rmse_ci_upper"],
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 10. 绘图基础
# =============================================================================

def apply_publication_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)


def save_figure(
    fig: plt.Figure,
    base_path_without_extension: str,
    dpi: int,
) -> None:
    fig.savefig(
        base_path_without_extension + ".png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        base_path_without_extension + ".pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def ordered_targets_present(frame: pd.DataFrame) -> List[str]:
    present = set(frame["target"].astype(str))
    ordered = [
        target
        for target in TARGET_ORDER
        if target in present
    ]
    extras = sorted(present.difference(ordered))
    return ordered + extras


# =============================================================================
# 11. Figure 3a / 3b：三种验证指标
# =============================================================================

def plot_three_split_metric(
    metrics_with_ci: pd.DataFrame,
    metric: str,
    output_base: str,
    source_data_path: str,
    dpi: int,
) -> None:
    targets = ordered_targets_present(metrics_with_ci)

    source = metrics_with_ci[
        metrics_with_ci["target"].isin(targets)
        & metrics_with_ci["split_type"].isin(SPLIT_ORDER)
    ].copy()

    source = source[
        [
            "target",
            "target_label",
            "split_type",
            "split_label",
            "n_samples",
            "n_papers",
            metric,
            f"{metric}_ci_lower",
            f"{metric}_ci_upper",
            "bootstrap_status",
        ]
    ]

    source.to_csv(
        source_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    x = np.arange(len(targets), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(SPLIT_ORDER))

    fig, ax = plt.subplots(figsize=(13.0, 5.8))

    for split_index, split_type in enumerate(SPLIT_ORDER):
        values: List[float] = []
        lower_errors: List[float] = []
        upper_errors: List[float] = []

        for target in targets:
            row = source[
                (
                    (source["target"] == target)
                    & (source["split_type"] == split_type)
                )
            ]

            if row.empty:
                values.append(np.nan)
                lower_errors.append(np.nan)
                upper_errors.append(np.nan)
                continue

            item = row.iloc[0]
            estimate = float(item[metric])
            lower = float(item[f"{metric}_ci_lower"])
            upper = float(item[f"{metric}_ci_upper"])

            values.append(estimate)
            lower_errors.append(
                max(0.0, estimate - lower)
                if np.isfinite(lower)
                else np.nan
            )
            upper_errors.append(
                max(0.0, upper - estimate)
                if np.isfinite(upper)
                else np.nan
            )

        positions = x + offsets[split_index]
        values_array = np.asarray(values, dtype=float)

        error_array = np.vstack([
            np.asarray(lower_errors, dtype=float),
            np.asarray(upper_errors, dtype=float),
        ])

        finite = np.isfinite(values_array)

        ax.errorbar(
            positions[finite],
            values_array[finite],
            yerr=error_array[:, finite],
            fmt="o",
            capsize=3,
            linewidth=1.2,
            markersize=5.5,
            label=SPLIT_LABELS[split_type],
        )

    ax.axhline(0.0, linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_LABELS.get(target, target) for target in targets],
        rotation=25,
        ha="right",
    )

    ylabel = {
        "r2": r"$R^2$",
        "spearman": "Spearman correlation",
        "mae": "MAE",
        "rmse": "RMSE",
    }.get(metric, metric)

    title = {
        "r2": "Predictive performance across three validation settings",
        "spearman": "Ranking performance across three validation settings",
        "mae": "Prediction error across three validation settings",
        "rmse": "Prediction error across three validation settings",
    }.get(metric, metric)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    apply_publication_axes(ax)

    fig.tight_layout()
    save_figure(fig, output_base, dpi)


# =============================================================================
# 12. Figure 3c / 3d：Parity plots
# =============================================================================

def parity_annotation(
    frame: pd.DataFrame,
) -> str:
    metrics = evaluate_regression(
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
    )

    r2_text = (
        f"{metrics['r2']:.2f}"
        if np.isfinite(metrics["r2"])
        else "NA"
    )
    spearman_text = (
        f"{metrics['spearman']:.2f}"
        if np.isfinite(metrics["spearman"])
        else "NA"
    )

    return (
        f"$R^2$ = {r2_text}\n"
        f"Spearman = {spearman_text}\n"
        f"n = {metrics['n']}"
    )


def robust_axis_limits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    values = np.concatenate([
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
    ])
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if math.isclose(minimum, maximum):
        padding = max(1.0, abs(minimum) * 0.10)
        return minimum - padding, maximum + padding

    padding = 0.06 * (maximum - minimum)
    return minimum - padding, maximum + padding


def plot_parity_grid(
    predictions: pd.DataFrame,
    split_type: str,
    output_base: str,
    source_data_path: str,
    dpi: int,
) -> None:
    subset = predictions[
        predictions["split_type"] == split_type
    ].copy()

    targets = ordered_targets_present(subset)

    if not targets:
        print(f"[WARN] {split_type} 没有预测，跳过parity图。")
        return

    subset.to_csv(
        source_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    n_targets = len(targets)
    n_columns = 3
    n_rows = int(math.ceil(n_targets / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.5 * n_columns, 4.1 * n_rows),
        squeeze=False,
    )

    for index, target in enumerate(targets):
        row_index = index // n_columns
        column_index = index % n_columns
        ax = axes[row_index][column_index]

        frame = subset[subset["target"] == target]

        ax.scatter(
            frame["y_true"],
            frame["y_pred"],
            s=20,
            alpha=0.68,
            edgecolors="none",
        )

        lower, upper = robust_axis_limits(
            frame["y_true"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=float),
        )

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        ax.text(
            0.04,
            0.96,
            parity_annotation(frame),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

        ax.set_title(TARGET_LABELS.get(target, target))
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.grid(linewidth=0.5, alpha=0.20)

    for empty_index in range(n_targets, n_rows * n_columns):
        row_index = empty_index // n_columns
        column_index = empty_index % n_columns
        axes[row_index][column_index].axis("off")

    fig.suptitle(
        {
            "paper_grouped_cv": "Paper-grouped out-of-fold predictions",
            "temporal_holdout": "Temporal holdout predictions",
            "random_holdout": "Random 4:1 holdout predictions",
        }.get(split_type, split_type),
        y=1.01,
        fontsize=15,
    )

    fig.tight_layout()
    save_figure(fig, output_base, dpi)


# =============================================================================
# 13. Supplementary：模式性能热图
# =============================================================================

def build_mode_matrix(
    mode_metrics: pd.DataFrame,
    split_type: str,
    metric: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    subset = mode_metrics[
        mode_metrics["split_type"] == split_type
    ].copy()

    targets = ordered_targets_present(subset)

    preferred_modes = ["A", "B", "C", "D", "E"]
    existing_modes = list(
        subset["growth_mode"].dropna().astype(str).unique()
    )

    modes = [
        mode
        for mode in preferred_modes
        if mode in existing_modes
    ]

    modes += sorted(
        mode
        for mode in existing_modes
        if mode not in modes
    )

    matrix = pd.DataFrame(
        np.nan,
        index=targets,
        columns=modes,
        dtype=float,
    )

    for _, row in subset.iterrows():
        target = str(row["target"])
        mode = str(row["growth_mode"])

        if target in matrix.index and mode in matrix.columns:
            matrix.loc[target, mode] = row[metric]

    return matrix, targets, modes


def plot_mode_heatmaps(
    mode_metrics: pd.DataFrame,
    output_base: str,
    source_data_path: str,
    dpi: int,
) -> None:
    source = mode_metrics[
        mode_metrics["split_type"] == "paper_grouped_cv"
    ].copy()

    source.to_csv(
        source_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    if source.empty:
        print("[WARN] 无paper-grouped模式数据，跳过模式热图。")
        return

    r2_matrix, targets, modes = build_mode_matrix(
        mode_metrics,
        "paper_grouped_cv",
        "r2",
    )
    spearman_matrix, _, _ = build_mode_matrix(
        mode_metrics,
        "paper_grouped_cv",
        "spearman",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.2 * max(4, len(modes)), 0.7 * len(targets) + 3.0),
    )

    for ax, matrix, title in [
        (axes[0], r2_matrix, r"Paper-grouped $R^2$"),
        (axes[1], spearman_matrix, "Paper-grouped Spearman"),
    ]:
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
        )

        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_yticks(np.arange(len(targets)))
        ax.set_yticklabels(
            [TARGET_LABELS.get(target, target) for target in targets]
        )
        ax.set_xlabel("Growth mode")
        ax.set_title(title)

        for row_index in range(len(targets)):
            for column_index in range(len(modes)):
                value = matrix.iloc[row_index, column_index]

                if np.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Mode-resolved performance of the global model",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, output_base, dpi)


# =============================================================================
# 14. 文件哈希与运行清单
# =============================================================================

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def build_input_manifest(paths: Paths) -> List[Dict[str, Any]]:
    input_paths = [
        paths.ordinary_predictions,
        paths.ordinary_metrics_fold,
        paths.ordinary_metrics_summary,
        paths.ordinary_split_manifest,
        paths.composite_predictions,
        paths.composite_metrics_fold,
        paths.composite_metrics_summary,
        paths.composite_split_manifest,
    ]

    rows: List[Dict[str, Any]] = []

    for path in input_paths:
        rows.append({
            "path": path,
            "exists": os.path.isfile(path),
            "size_bytes": (
                os.path.getsize(path)
                if os.path.isfile(path)
                else np.nan
            ),
            "sha256": (
                sha256_file(path)
                if os.path.isfile(path)
                else ""
            ),
        })

    return rows


# =============================================================================
# 15. 主程序
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将L1随机、跨论文和跨年份验证整理成论文Figure 3与表格。"
        )
    )

    parser.add_argument(
        "--project_dir",
        default=None,
        help="项目目录；默认使用当前脚本所在目录。",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录；默认 <project_dir>/outputs/external_validation/generalization。",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
        help="以paper_id为单位的bootstrap次数，默认2000。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="bootstrap随机种子。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PNG输出分辨率，默认600 dpi。",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = (
        os.path.abspath(args.project_dir)
        if args.project_dir
        else os.path.abspath(os.path.join(script_dir, "..", ".."))
    )

    paths = build_paths(
        project_dir=project_dir,
        output_dir=args.output_dir,
    )

    ensure_inputs_exist(paths)
    create_output_directories(paths)

    print("=" * 96)
    print("[ARTICLE VALIDATION START]")
    print(f"Project directory : {paths.project_dir}")
    print(f"Ordinary results  : {paths.ordinary_dir}")
    print(f"Composite results : {paths.composite_dir}")
    print(f"Bootstrap repeats : {args.bootstrap}")
    print(f"Random seed       : {args.seed}")
    print(f"PNG dpi           : {args.dpi}")
    print(f"Output            : {paths.output_dir}")
    print("=" * 96)

    predictions = load_predictions(paths)
    manifests = load_manifests(paths)

    audit = audit_predictions_and_splits(
        predictions=predictions,
        manifests=manifests,
    )

    fold_metrics = recompute_fold_metrics(predictions)
    pooled_metrics = recompute_pooled_metrics(predictions)

    bootstrap_ci = run_bootstrap_analysis(
        predictions=predictions,
        repeats=args.bootstrap,
        seed=args.seed,
    )

    metrics_with_ci = merge_metrics_and_ci(
        pooled_metrics=pooled_metrics,
        bootstrap_ci=bootstrap_ci,
    )

    mode_metrics = metrics_by_growth_mode(predictions)
    sample_statistics = build_target_sample_statistics(predictions)
    main_table = build_main_table(metrics_with_ci)

    # -------------------------------------------------------------------------
    # 保存核心数据
    # -------------------------------------------------------------------------

    fold_metrics_path = os.path.join(
        paths.output_dir,
        "three_split_metrics_fold_recomputed.csv",
    )
    pooled_metrics_path = os.path.join(
        paths.output_dir,
        "three_split_metrics_pooled.csv",
    )
    bootstrap_path = os.path.join(
        paths.output_dir,
        "three_split_bootstrap_ci.csv",
    )
    metrics_ci_path = os.path.join(
        paths.output_dir,
        "three_split_metrics_with_ci.csv",
    )
    mode_metrics_path = os.path.join(
        paths.output_dir,
        "metrics_by_growth_mode.csv",
    )
    statistics_path = os.path.join(
        paths.output_dir,
        "target_sample_statistics.csv",
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics.to_csv(
        pooled_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_ci.to_csv(
        bootstrap_path,
        index=False,
        encoding="utf-8-sig",
    )
    metrics_with_ci.to_csv(
        metrics_ci_path,
        index=False,
        encoding="utf-8-sig",
    )
    mode_metrics.to_csv(
        mode_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    sample_statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 保存三种预测集
    # -------------------------------------------------------------------------

    prediction_output_paths: Dict[str, str] = {}

    for split_type, filename in [
        ("random_holdout", "random_holdout_predictions.csv"),
        ("paper_grouped_cv", "paper_grouped_oof_predictions.csv"),
        ("temporal_holdout", "temporal_holdout_predictions.csv"),
    ]:
        output_path = os.path.join(paths.output_dir, filename)

        predictions[
            predictions["split_type"] == split_type
        ].to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        prediction_output_paths[split_type] = output_path

    # -------------------------------------------------------------------------
    # 保存文章表格
    # -------------------------------------------------------------------------

    main_table_path = os.path.join(
        paths.tables_dir,
        "table_main_three_split_validation.csv",
    )
    si_mode_table_path = os.path.join(
        paths.tables_dir,
        "table_si_metrics_by_growth_mode.csv",
    )
    si_fold_table_path = os.path.join(
        paths.tables_dir,
        "table_si_fold_level_metrics.csv",
    )
    si_bootstrap_table_path = os.path.join(
        paths.tables_dir,
        "table_si_bootstrap_confidence_intervals.csv",
    )

    main_table.to_csv(
        main_table_path,
        index=False,
        encoding="utf-8-sig",
    )
    mode_metrics.to_csv(
        si_mode_table_path,
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        si_fold_table_path,
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_ci.to_csv(
        si_bootstrap_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 绘图与source data
    # -------------------------------------------------------------------------

    figure_files: Dict[str, Dict[str, str]] = {}

    for metric, figure_name in [
        ("r2", "fig3a_three_split_r2"),
        ("spearman", "fig3b_three_split_spearman"),
    ]:
        figure_base = os.path.join(
            paths.figures_dir,
            figure_name,
        )
        source_path = os.path.join(
            paths.source_data_dir,
            figure_name + "_source_data.csv",
        )

        plot_three_split_metric(
            metrics_with_ci=metrics_with_ci,
            metric=metric,
            output_base=figure_base,
            source_data_path=source_path,
            dpi=args.dpi,
        )

        figure_files[figure_name] = {
            "png": figure_base + ".png",
            "pdf": figure_base + ".pdf",
            "source_data": source_path,
        }

    for split_type, figure_name in [
        ("paper_grouped_cv", "fig3c_grouped_parity"),
        ("temporal_holdout", "fig3d_temporal_parity"),
    ]:
        figure_base = os.path.join(
            paths.figures_dir,
            figure_name,
        )
        source_path = os.path.join(
            paths.source_data_dir,
            figure_name + "_source_data.csv",
        )

        plot_parity_grid(
            predictions=predictions,
            split_type=split_type,
            output_base=figure_base,
            source_data_path=source_path,
            dpi=args.dpi,
        )

        figure_files[figure_name] = {
            "png": figure_base + ".png",
            "pdf": figure_base + ".pdf",
            "source_data": source_path,
        }

    mode_figure_name = "figS1_metrics_by_growth_mode"
    mode_figure_base = os.path.join(
        paths.figures_dir,
        mode_figure_name,
    )
    mode_source_path = os.path.join(
        paths.source_data_dir,
        mode_figure_name + "_source_data.csv",
    )

    plot_mode_heatmaps(
        mode_metrics=mode_metrics,
        output_base=mode_figure_base,
        source_data_path=mode_source_path,
        dpi=args.dpi,
    )

    figure_files[mode_figure_name] = {
        "png": mode_figure_base + ".png",
        "pdf": mode_figure_base + ".pdf",
        "source_data": mode_source_path,
    }

    # -------------------------------------------------------------------------
    # 审计与运行清单
    # -------------------------------------------------------------------------

    input_manifest = build_input_manifest(paths)

    input_manifest_path = os.path.join(
        paths.manifests_dir,
        "input_file_manifest.csv",
    )
    audit_path = os.path.join(
        paths.manifests_dir,
        "validation_audit.json",
    )
    run_manifest_path = os.path.join(
        paths.manifests_dir,
        "article_validation_run_manifest.json",
    )

    pd.DataFrame(input_manifest).to_csv(
        input_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    with open(audit_path, "w", encoding="utf-8") as file:
        json.dump(
            audit,
            file,
            ensure_ascii=False,
            indent=2,
        )

    run_manifest = {
        "analysis": "CNT 3.0 article-ready three-split validation",
        "project_dir": paths.project_dir,
        "output_dir": paths.output_dir,
        "bootstrap": {
            "cluster": "paper_id",
            "repeats": int(args.bootstrap),
            "confidence_level": CI_LEVEL,
            "seed": int(args.seed),
            "minimum_samples": MIN_BOOTSTRAP_SAMPLES,
            "minimum_papers": MIN_BOOTSTRAP_PAPERS,
        },
        "validation_settings": {
            split_type: SPLIT_LABELS[split_type]
            for split_type in SPLIT_ORDER
        },
        "targets": [
            target
            for target in TARGET_ORDER
            if target in set(predictions["target"])
        ],
        "prediction_rows": int(len(predictions)),
        "unique_papers": int(predictions["paper_id"].nunique()),
        "outputs": {
            "fold_metrics": fold_metrics_path,
            "pooled_metrics": pooled_metrics_path,
            "bootstrap_ci": bootstrap_path,
            "metrics_with_ci": metrics_ci_path,
            "mode_metrics": mode_metrics_path,
            "sample_statistics": statistics_path,
            "predictions": prediction_output_paths,
            "main_table": main_table_path,
            "si_mode_table": si_mode_table_path,
            "si_fold_table": si_fold_table_path,
            "si_bootstrap_table": si_bootstrap_table_path,
            "figures": figure_files,
            "audit": audit_path,
            "input_manifest": input_manifest_path,
        },
        "interpretation_notes": {
            "random_holdout": (
                "Baseline 80/20 row-level holdout; papers may be represented "
                "on both sides and this result should not be the primary "
                "generalization claim."
            ),
            "paper_grouped_cv": (
                "Primary retrospective generalization result; every paper is "
                "restricted to one fold."
            ),
            "temporal_holdout": (
                "Prospective-like external_validation using earlier literature for "
                "training and later literature for testing."
            ),
            "bootstrap": (
                "Confidence intervals are clustered by paper_id rather than "
                "resampling individual synthesis rows."
            ),
            "mode_metrics": (
                "Groups below the configured sample/paper thresholds are "
                "marked descriptive_only_small_group."
            ),
        },
    }

    with open(run_manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            run_manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 96)
    print("[ARTICLE VALIDATION FINISHED]")
    print(f"Fold metrics       : {fold_metrics_path}")
    print(f"Pooled metrics     : {pooled_metrics_path}")
    print(f"Bootstrap CI       : {bootstrap_path}")
    print(f"Metrics with CI    : {metrics_ci_path}")
    print(f"Mode metrics       : {mode_metrics_path}")
    print(f"Main table         : {main_table_path}")
    print(f"Figures            : {paths.figures_dir}")
    print(f"Source data        : {paths.source_data_dir}")
    print(f"Audit              : {audit_path}")
    print(f"Run manifest       : {run_manifest_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
