# -*- coding: utf-8 -*-
"""
composite_sensitivity.py

CNT 3.0 — Composite-label completeness sensitivity analysis

Purpose
-------
The current composite target is a renormalized weighted combination of up to
five reported CNT outcomes:

    0.30 × growth quality
  + 0.20 × IG/ID
  + 0.20 × orientation
  + 0.15 × purity
  + 0.15 × bundle density

The formal L1-composite and L12 pipelines currently require at least two
reported components. This sensitivity analysis additionally tests thresholds
1, 2 and 3 to quantify the effect of label completeness.

What this script does
---------------------
1. Reuses the frozen L1/L12 split roles for existing samples and extends the
   same split design to threshold-1-only samples.
2. Fits IG/ID and bundle-density normalization on each outer training set only.
3. Retrains:
       - L1 point-composite model on L1 rows;
       - L12 q10/q50/q90 composite models on L1+L2 rows.
4. Evaluates random holdout, paper-grouped CV and temporal holdout.
5. Repeats the historical replay at the same selected cutoff years used by
   3model-retrospective-ranking.py, or at user-specified cutoffs.
6. Compares sample retention, q50 prediction performance, interval coverage,
   q50/q90 historical NDCG, Recall and enrichment.
7. Produces tables, figures, source data, manifests and optional model files.
8. Supports checkpoint/resume. Use --fresh when data, thresholds or settings
   have changed.

This script does NOT
--------------------
- overwrite the existing L1/L12 results;
- modify N/C/F/P features;
- modify L123;
- automatically declare one threshold scientifically correct;
- train deployment models for L123. Deployment models should be exported only
  after the sensitivity results have been reviewed.

Default inputs
--------------
<project_dir>/
├─ data/11_flattened/
│  ├─ ml_dataset_3p0_full60D.csv
│  └─ feature_schema.json
├─ outputs/models/l1_composite/
│  ├─ composite_split_manifest.csv
│  └─ L1_composite_predictions.csv                  # optional audit input
├─ outputs/models/l12_quantile/
│  ├─ L12_split_manifest.csv
│  └─ L12_quantile_predictions.csv                  # optional audit input
└─ outputs/external_validation/retrospective_ranking/
   ├─ cutoff_selection_manifest.csv                 # optional; defaults below
   └─ historical_replay_predictions.csv             # optional audit input

Default outputs
---------------
<project_dir>/outputs/external_validation/composite_sensitivity/
├─ figures/
├─ source_data/
├─ tables/
├─ manifests/
├─ validation_models/                               # only --save_models
├─ historical_models/                               # only --save_models
├─ composite_sensitivity_component_coverage.csv
├─ composite_sensitivity_metrics_fold.csv
├─ composite_sensitivity_metrics_pooled.csv
├─ composite_sensitivity_bootstrap_ci.csv
├─ composite_sensitivity_predictions.csv
├─ composite_sensitivity_normalization_manifest.csv
├─ composite_sensitivity_split_retention_audit.csv
├─ composite_sensitivity_historical_predictions.csv
├─ composite_sensitivity_historical_metrics.csv
├─ composite_sensitivity_historical_summary.csv
├─ composite_threshold_comparison_table.csv
└─ composite_sensitivity_run_summary.json

Run
---
Formal run:
    python composite_sensitivity.py

Quick check:
    python composite_sensitivity.py --bootstrap 200 --random_repeats 200 --cutoffs 2017 2020

Restart from zero:
    python composite_sensitivity.py --fresh

Save all validation and historical models:
    python composite_sensitivity.py --save_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# =============================================================================
# 1. Fixed scientific definitions
# =============================================================================

SEED = 42
DEFAULT_RANDOM_SEED = 20260723
DEFAULT_DPI = 600

INNER_VALID_SIZE = 0.15
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200

MIN_OUTER_TRAIN_SAMPLES = 60
MIN_OUTER_TEST_SAMPLES = 20
MIN_OUTER_TRAIN_PAPERS = 5
MIN_OUTER_TEST_PAPERS = 2

MIN_HISTORICAL_TRAIN_SAMPLES = 100
MIN_HISTORICAL_FUTURE_SAMPLES = 25
MIN_HISTORICAL_TRAIN_PAPERS = 10
MIN_HISTORICAL_FUTURE_PAPERS = 3

FORMAL_BASELINE_THRESHOLD = 2
DEFAULT_THRESHOLDS = [1, 2, 3]
DEFAULT_CUTOFFS = [2012, 2015, 2017, 2020]
DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_RANDOM_RANKING_REPEATS = 1000
DEFAULT_POSITIVE_FRACTION = 0.20
DEFAULT_SCREEN_FRACTION = 0.10
BASELINE_REPRODUCTION_TOLERANCE = 1e-6

TARGET_NAME = "y_composite_final_score"

POINT_FAMILY = "L1_point"
QUANTILE_FAMILY = "L12_quantile"
MODEL_FAMILY_ORDER = [POINT_FAMILY, QUANTILE_FAMILY]

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

QUANTILES: Dict[str, float] = {
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
}

WEIGHTS: Dict[str, float] = {
    "gq": 0.30,
    "igid": 0.20,
    "ori": 0.20,
    "purity": 0.15,
    "bundle": 0.15,
}

MISSING_POLICY = "renorm"

PREFERRED_IGID_COL = "y_ig_id_ratio"
LEGACY_IGID_COL = "y_IG_ID"
UNIFIED_IGID_COL = "y_ig_id_ratio_unified"

COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

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

POINT_PARAMS: Dict[str, Any] = {
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

QUANTILE_BASE_PARAMS: Dict[str, Any] = {
    "objective": "quantile",
    "metric": "quantile",
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

HISTORICAL_STRATEGIES = {
    "q50_balanced": "q50_raw",
    "q90_potential": "q90_raw",
}

HISTORICAL_STRATEGY_LABELS = {
    "q50_balanced": "q50 balanced",
    "q90_potential": "q90 potential",
}


# =============================================================================
# 2. Paths and file helpers
# =============================================================================

@dataclass(frozen=True)
class Paths:
    project_dir: str
    dataset: str
    schema: str

    l1_dir: str
    l1_split_manifest: str
    l1_baseline_predictions: str

    l12_dir: str
    l12_split_manifest: str
    l12_baseline_predictions: str

    retrospective_dir: str
    retrospective_cutoff_manifest: str
    retrospective_baseline_predictions: str

    output_dir: str
    figures_dir: str
    source_data_dir: str
    tables_dir: str
    manifests_dir: str
    validation_models_dir: str
    historical_models_dir: str

    coverage: str
    metrics_fold: str
    metrics_pooled: str
    bootstrap_ci: str
    predictions: str
    normalizations: str
    split_retention_audit: str
    historical_predictions: str
    historical_metrics: str
    historical_summary: str
    random_baseline: str
    comparison_table: str
    baseline_audit: str
    historical_baseline_audit: str
    checkpoint_config: str
    run_summary: str


def build_paths(project_dir: str, output_dir: Optional[str]) -> Paths:
    project_dir = os.path.abspath(project_dir)
    output_dir = (
        os.path.abspath(output_dir)
        if output_dir
        else os.path.join(project_dir, "outputs", "external_validation", "composite_sensitivity")
    )

    data_dir = os.path.join(project_dir, "data", "11_flattened")
    l1_dir = os.path.join(project_dir, "outputs", "models", "l1_composite")
    l12_dir = os.path.join(project_dir, "outputs", "models", "l12_quantile")
    retrospective_dir = os.path.join(
        project_dir,
        "outputs",
        "external_validation",
        "retrospective_ranking",
    )

    return Paths(
        project_dir=project_dir,
        dataset=os.path.join(data_dir, "ml_dataset_3p0_full60D.csv"),
        schema=os.path.join(data_dir, "feature_schema.json"),

        l1_dir=l1_dir,
        l1_split_manifest=os.path.join(l1_dir, "composite_split_manifest.csv"),
        l1_baseline_predictions=os.path.join(l1_dir, "L1_composite_predictions.csv"),

        l12_dir=l12_dir,
        l12_split_manifest=os.path.join(l12_dir, "L12_split_manifest.csv"),
        l12_baseline_predictions=os.path.join(l12_dir, "L12_quantile_predictions.csv"),

        retrospective_dir=retrospective_dir,
        retrospective_cutoff_manifest=os.path.join(
            retrospective_dir,
            "cutoff_selection_manifest.csv",
        ),
        retrospective_baseline_predictions=os.path.join(
            retrospective_dir,
            "historical_replay_predictions.csv",
        ),

        output_dir=output_dir,
        figures_dir=os.path.join(output_dir, "figures"),
        source_data_dir=os.path.join(output_dir, "source_data"),
        tables_dir=os.path.join(output_dir, "tables"),
        manifests_dir=os.path.join(output_dir, "manifests"),
        validation_models_dir=os.path.join(output_dir, "validation_models"),
        historical_models_dir=os.path.join(output_dir, "historical_models"),

        coverage=os.path.join(
            output_dir,
            "composite_sensitivity_component_coverage.csv",
        ),
        metrics_fold=os.path.join(
            output_dir,
            "composite_sensitivity_metrics_fold.csv",
        ),
        metrics_pooled=os.path.join(
            output_dir,
            "composite_sensitivity_metrics_pooled.csv",
        ),
        bootstrap_ci=os.path.join(
            output_dir,
            "composite_sensitivity_bootstrap_ci.csv",
        ),
        predictions=os.path.join(
            output_dir,
            "composite_sensitivity_predictions.csv",
        ),
        normalizations=os.path.join(
            output_dir,
            "composite_sensitivity_normalization_manifest.csv",
        ),
        split_retention_audit=os.path.join(
            output_dir,
            "composite_sensitivity_split_retention_audit.csv",
        ),
        historical_predictions=os.path.join(
            output_dir,
            "composite_sensitivity_historical_predictions.csv",
        ),
        historical_metrics=os.path.join(
            output_dir,
            "composite_sensitivity_historical_metrics.csv",
        ),
        historical_summary=os.path.join(
            output_dir,
            "composite_sensitivity_historical_summary.csv",
        ),
        random_baseline=os.path.join(
            output_dir,
            "composite_sensitivity_random_ranking_baseline.csv",
        ),
        comparison_table=os.path.join(
            output_dir,
            "composite_threshold_comparison_table.csv",
        ),
        baseline_audit=os.path.join(
            output_dir,
            "composite_sensitivity_threshold2_reproduction_audit.csv",
        ),
        historical_baseline_audit=os.path.join(
            output_dir,
            "composite_sensitivity_historical_threshold1_reproduction_audit.csv",
        ),
        checkpoint_config=os.path.join(
            output_dir,
            "composite_sensitivity_checkpoint_config.json",
        ),
        run_summary=os.path.join(
            output_dir,
            "composite_sensitivity_run_summary.json",
        ),
    )


def ensure_required_inputs(paths: Paths) -> None:
    required = [
        paths.dataset,
        paths.schema,
        paths.l1_split_manifest,
        paths.l12_split_manifest,
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "缺少必要输入文件：\n" + "\n".join(missing)
        )


def create_output_dirs(paths: Paths, save_models: bool) -> None:
    directories = [
        paths.output_dir,
        paths.figures_dir,
        paths.source_data_dir,
        paths.tables_dir,
        paths.manifests_dir,
    ]
    if save_models:
        directories.extend(
            [
                paths.validation_models_dir,
                paths.historical_models_dir,
            ]
        )
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".csv",
        dir=directory,
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_write_json(data: Any, path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".json",
        dir=directory,
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_optional_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


# =============================================================================
# 3. Dataset and composite target
# =============================================================================


def load_feature_schema(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        schema = json.load(file)
    columns = schema.get("model_feature_columns")
    if not isinstance(columns, list) or not columns:
        raise RuntimeError(
            "feature_schema.json 缺少有效 model_feature_columns。"
        )
    columns = [str(column) for column in columns]
    if len(columns) != len(set(columns)):
        raise RuntimeError("feature_schema.json 中存在重复特征。")
    return columns


def ensure_igid_unified(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = (
        pd.to_numeric(frame[PREFERRED_IGID_COL], errors="coerce")
        if PREFERRED_IGID_COL in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    legacy = (
        pd.to_numeric(frame[LEGACY_IGID_COL], errors="coerce")
        if LEGACY_IGID_COL in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    output = frame.copy()
    output[UNIFIED_IGID_COL] = preferred.where(preferred.notna(), legacy)
    return output


def load_dataset(path: str, feature_columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)

    required_metadata = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "data_level",
    ]
    missing_metadata = [
        column for column in required_metadata if column not in frame.columns
    ]
    if missing_metadata:
        raise RuntimeError(f"数据集缺少元数据：{missing_metadata}")

    missing_features = [
        column for column in feature_columns if column not in frame.columns
    ]
    if missing_features:
        raise RuntimeError(
            "数据集缺少schema特征，示例："
            f"{missing_features[:10]}"
        )

    frame = ensure_igid_unified(frame)

    defaults = {
        "growth_mode": "",
        "source_file": "",
        "synthesis_id": "",
        "doi": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame["sample_id"] = frame["sample_id"].fillna("").astype(str)
    frame["paper_id"] = frame["paper_id"].fillna("").astype(str)
    frame["paper_year_int"] = pd.to_numeric(
        frame["paper_year_int"],
        errors="coerce",
    )
    frame["data_level"] = (
        frame["data_level"].fillna("").astype(str).str.upper()
    )

    if (frame["sample_id"].str.len() == 0).any():
        raise RuntimeError("存在空 sample_id。")
    if (frame["paper_id"].str.len() == 0).any():
        raise RuntimeError("存在空 paper_id。")
    if frame["sample_id"].duplicated().any():
        examples = frame.loc[
            frame["sample_id"].duplicated(keep=False),
            "sample_id",
        ].head(10).tolist()
        raise RuntimeError(f"sample_id不唯一，示例：{examples}")

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.reset_index(drop=True)


def prepare_composite_base(
    dataset: pd.DataFrame,
    levels: Sequence[str],
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = dataset[
        dataset["data_level"].isin([str(level).upper() for level in levels])
    ].copy()

    for column in [
        COL_GQ,
        UNIFIED_IGID_COL,
        COL_ORI,
        COL_PURITY,
        COL_BUNDLE,
    ]:
        if column not in work.columns:
            work[column] = np.nan

    work[RAW_GQ] = pd.to_numeric(work[COL_GQ], errors="coerce")
    work[RAW_IGID] = pd.to_numeric(work[UNIFIED_IGID_COL], errors="coerce")
    work[RAW_ORI] = pd.to_numeric(work[COL_ORI], errors="coerce")
    work[RAW_PURITY] = pd.to_numeric(work[COL_PURITY], errors="coerce")
    work[RAW_BUNDLE] = pd.to_numeric(work[COL_BUNDLE], errors="coerce")

    bundle = work[RAW_BUNDLE].to_numpy(dtype=float)
    work[RAW_BUNDLE_LOG] = np.where(
        np.isfinite(bundle) & (bundle >= 0.0),
        np.log1p(bundle),
        np.nan,
    )

    availability = pd.DataFrame(
        {
            "gq": work[RAW_GQ].notna(),
            "igid": work[RAW_IGID].notna(),
            "ori": work[RAW_ORI].notna(),
            "purity": work[RAW_PURITY].notna(),
            "bundle": work[RAW_BUNDLE_LOG].notna(),
        },
        index=work.index,
    )
    work[RAW_COMPONENT_COUNT] = availability.sum(axis=1).astype(int)
    work = work[work[RAW_COMPONENT_COUNT] >= 1].copy().reset_index(drop=True)

    X = work[list(feature_columns)].copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.reset_index(drop=True)

    return work, X


def minmax_parameters(series: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
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
    usable = not math.isclose(
        minimum,
        maximum,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    return {
        "minimum": minimum,
        "maximum": maximum,
        "usable": usable,
        "finite_count": int(len(finite)),
        "reason": "ok" if usable else "constant_values",
    }


def fit_composite_normalizer(
    work: pd.DataFrame,
    train_indices: np.ndarray,
) -> Dict[str, Any]:
    train_indices = np.asarray(train_indices, dtype=int)
    train = work.iloc[train_indices]
    igid = minmax_parameters(train[RAW_IGID])
    bundle = minmax_parameters(train[RAW_BUNDLE_LOG])
    return {
        "igid_min": igid["minimum"],
        "igid_max": igid["maximum"],
        "igid_usable": bool(igid["usable"]),
        "igid_finite_count": int(igid["finite_count"]),
        "igid_reason": igid["reason"],
        "bundle_log_min": bundle["minimum"],
        "bundle_log_max": bundle["maximum"],
        "bundle_usable": bool(bundle["usable"]),
        "bundle_finite_count": int(bundle["finite_count"]),
        "bundle_reason": bundle["reason"],
        "fit_sample_count": int(len(train_indices)),
        "fit_paper_count": int(train["paper_id"].nunique()),
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
    output = (
        pd.to_numeric(series, errors="coerce") - float(minimum)
    ) / (float(maximum) - float(minimum))
    return output.clip(lower=0.0, upper=1.0)


def transform_composite(
    work: pd.DataFrame,
    normalizer: Mapping[str, Any],
    minimum_component_count: int,
) -> pd.DataFrame:
    output = pd.DataFrame(index=work.index)

    output["gq_01"] = (
        pd.to_numeric(work[RAW_GQ], errors="coerce") / 10.0
    ).clip(lower=0.0, upper=1.0)
    output["igid_norm01"] = apply_minmax(
        work[RAW_IGID],
        normalizer.get("igid_min"),
        normalizer.get("igid_max"),
        bool(normalizer.get("igid_usable", False)),
    )
    output["ori_01"] = pd.to_numeric(
        work[RAW_ORI], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    output["purity_01"] = pd.to_numeric(
        work[RAW_PURITY], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    output["bundle_norm01"] = apply_minmax(
        work[RAW_BUNDLE_LOG],
        normalizer.get("bundle_log_min"),
        normalizer.get("bundle_log_max"),
        bool(normalizer.get("bundle_usable", False)),
    )

    components = pd.DataFrame(
        {
            component: output[column]
            for component, column in COMPONENT_COLUMNS.items()
        },
        index=work.index,
    )
    weights = pd.Series(WEIGHTS, dtype=float)

    available = components.notna()
    component_count = available.sum(axis=1).astype(int)
    weight_sum = available.mul(weights, axis=1).sum(axis=1)
    numerator = components.mul(weights, axis=1).sum(axis=1, skipna=True)

    composite = pd.Series(
        np.where(weight_sum > 0, numerator / weight_sum, np.nan),
        index=work.index,
        dtype=float,
    )
    composite = composite.where(
        component_count >= int(minimum_component_count),
        np.nan,
    )

    output[TARGET_NAME] = composite
    output["composite_component_count"] = component_count
    output["composite_weight_sum_used"] = weight_sum.where(
        composite.notna(),
        np.nan,
    )
    return output


# =============================================================================
# 4. Metrics
# =============================================================================


def safe_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    y_true_array = y_true_array[finite]
    y_pred_array = y_pred_array[finite]
    if len(y_true_array) < 2 or np.allclose(y_true_array, y_true_array[0]):
        return float("nan")
    return float(r2_score(y_true_array, y_pred_array))


def safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    y_true_array = y_true_array[finite]
    y_pred_array = y_pred_array[finite]
    if len(y_true_array) < 2:
        return float("nan")
    true_rank = pd.Series(y_true_array).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred_array).rank(method="average").to_numpy(dtype=float)
    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        return float("nan")
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def point_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
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
    return {
        "n": int(len(y_true_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": safe_r2(y_true_array, y_pred_array),
        "spearman": safe_spearman(y_true_array, y_pred_array),
    }


def pinball_loss(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    alpha: float,
) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    error = y_true_array - y_pred_array
    loss = np.maximum(alpha * error, (alpha - 1.0) * error)
    return float(np.mean(loss))


def quantile_metrics(
    y_true: Sequence[float],
    q10_raw: Sequence[float],
    q50_raw: Sequence[float],
    q90_raw: Sequence[float],
) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    q10_array = np.asarray(q10_raw, dtype=float)
    q50_array = np.asarray(q50_raw, dtype=float)
    q90_array = np.asarray(q90_raw, dtype=float)

    finite = (
        np.isfinite(y_true_array)
        & np.isfinite(q10_array)
        & np.isfinite(q50_array)
        & np.isfinite(q90_array)
    )
    y_true_array = y_true_array[finite]
    q10_array = q10_array[finite]
    q50_array = q50_array[finite]
    q90_array = q90_array[finite]

    if len(y_true_array) == 0:
        return {
            "n": 0,
            "q50_mae": np.nan,
            "q50_rmse": np.nan,
            "q50_r2": np.nan,
            "q50_spearman": np.nan,
            "pinball_q10": np.nan,
            "pinball_q50": np.nan,
            "pinball_q90": np.nan,
            "empirical_coverage_q10": np.nan,
            "empirical_coverage_q50": np.nan,
            "empirical_coverage_q90": np.nan,
            "picp_80": np.nan,
            "mpiw_80": np.nan,
            "crossing_rate": np.nan,
        }

    ordered = np.sort(
        np.column_stack([q10_array, q50_array, q90_array]),
        axis=1,
    )
    lower = ordered[:, 0]
    upper = ordered[:, 2]
    width = upper - lower
    covered = (y_true_array >= lower) & (y_true_array <= upper)
    crossing = (q10_array > q50_array) | (q50_array > q90_array)

    q50_point = point_metrics(y_true_array, q50_array)
    return {
        "n": int(len(y_true_array)),
        "q50_mae": q50_point["mae"],
        "q50_rmse": q50_point["rmse"],
        "q50_r2": q50_point["r2"],
        "q50_spearman": q50_point["spearman"],
        "pinball_q10": pinball_loss(y_true_array, q10_array, 0.10),
        "pinball_q50": pinball_loss(y_true_array, q50_array, 0.50),
        "pinball_q90": pinball_loss(y_true_array, q90_array, 0.90),
        "empirical_coverage_q10": float(np.mean(y_true_array <= q10_array)),
        "empirical_coverage_q50": float(np.mean(y_true_array <= q50_array)),
        "empirical_coverage_q90": float(np.mean(y_true_array <= q90_array)),
        "picp_80": float(np.mean(covered)),
        "mpiw_80": float(np.mean(width)),
        "crossing_rate": float(np.mean(crossing)),
    }


# =============================================================================
# 5. Frozen split manifests
# =============================================================================


def load_split_manifest(path: str, expected_levels: Sequence[str]) -> pd.DataFrame:
    manifest = pd.read_csv(path, low_memory=False)
    required = ["sample_id", "split_type", "fold", "role"]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise RuntimeError(f"划分清单缺少列：{missing}; 文件={path}")

    manifest = manifest.copy()
    manifest["sample_id"] = manifest["sample_id"].fillna("").astype(str)
    manifest["split_type"] = manifest["split_type"].astype(str)
    manifest["role"] = manifest["role"].astype(str).str.lower()
    manifest["fold"] = pd.to_numeric(manifest["fold"], errors="raise").astype(int)

    manifest = manifest[
        manifest["split_type"].isin(SPLIT_ORDER)
        & manifest["role"].isin(["train", "test"])
    ].copy()

    if "data_level" in manifest.columns:
        allowed = {str(level).upper() for level in expected_levels}
        actual = set(
            manifest["data_level"].dropna().astype(str).str.upper().unique()
        )
        unexpected = actual - allowed
        if unexpected:
            raise RuntimeError(
                f"划分清单包含不应出现的数据层级：{sorted(unexpected)}"
            )

    duplicate = manifest.duplicated(
        subset=["split_type", "fold", "role", "sample_id"],
        keep=False,
    )
    if duplicate.any():
        raise RuntimeError(
            "划分清单在同一split/fold/role内包含重复sample_id。"
        )

    return manifest.reset_index(drop=True)


def split_tasks_from_manifest(manifest: pd.DataFrame) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    grouped = manifest.groupby(["split_type", "fold"], sort=False)

    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    keys = sorted(
        grouped.groups.keys(),
        key=lambda item: (split_rank.get(item[0], 999), int(item[1])),
    )

    for split_type, fold in keys:
        group = grouped.get_group((split_type, fold))
        train_ids = group.loc[group["role"] == "train", "sample_id"].tolist()
        test_ids = group.loc[group["role"] == "test", "sample_id"].tolist()

        cutoff: Optional[int] = None
        if "temporal_cutoff_year" in group.columns:
            values = pd.to_numeric(
                group["temporal_cutoff_year"],
                errors="coerce",
            ).dropna().unique()
            if len(values) == 1:
                cutoff = int(values[0])

        if set(train_ids) & set(test_ids):
            raise RuntimeError(
                f"{split_type}/fold={fold}训练与测试sample_id重叠。"
            )

        tasks.append(
            {
                "split_type": str(split_type),
                "fold": int(fold),
                "temporal_cutoff_year": cutoff,
                "train_ids": train_ids,
                "test_ids": test_ids,
            }
        )

    return tasks


def extend_tasks_for_threshold1(
    tasks: Sequence[Mapping[str, Any]],
    work: pd.DataFrame,
) -> List[Dict[str, Any]]:
    extended = [
        {**dict(task), "train_ids": list(task["train_ids"]), "test_ids": list(task["test_ids"])}
        for task in tasks
    ]
    frozen_ids = {
        str(sample_id)
        for task in extended
        for sample_id in list(task["train_ids"]) + list(task["test_ids"])
    }
    extra = work[
        ~work["sample_id"].astype(str).isin(frozen_ids)
    ][["sample_id", "paper_id", "paper_year_int"]].copy()
    if extra.empty:
        return extended

    extra["sample_id"] = extra["sample_id"].astype(str)
    extra["paper_id"] = extra["paper_id"].astype(str)

    for task in extended:
        if task["split_type"] == "random_holdout":
            total = len(task["train_ids"]) + len(task["test_ids"])
            fraction = len(task["test_ids"]) / total if total else 0.20
            ids = extra["sample_id"].tolist()
            if len(ids) > 1:
                train_ids, test_ids = train_test_split(
                    ids, test_size=fraction, random_state=SEED
                )
                task["train_ids"].extend(train_ids)
                task["test_ids"].extend(test_ids)
            else:
                task["train_ids"].extend(ids)

    grouped = [task for task in extended if task["split_type"] == "paper_grouped_cv"]
    if grouped:
        sample_to_paper = dict(zip(work["sample_id"].astype(str), work["paper_id"].astype(str)))
        paper_to_fold: Dict[str, int] = {}
        fold_load = {}
        for task in grouped:
            fold = int(task["fold"])
            fold_load[fold] = len(task["test_ids"])
            for sample_id in task["test_ids"]:
                paper = sample_to_paper.get(str(sample_id))
                if paper is not None:
                    paper_to_fold[paper] = fold

        extra_by_paper = {
            paper: group["sample_id"].tolist()
            for paper, group in extra.groupby("paper_id", sort=True)
        }
        new_papers = sorted(
            (paper for paper in extra_by_paper if paper not in paper_to_fold),
            key=lambda paper: (-len(extra_by_paper[paper]), paper),
        )
        for paper in new_papers:
            fold = min(fold_load, key=lambda value: (fold_load[value], value))
            paper_to_fold[paper] = fold
            fold_load[fold] += len(extra_by_paper[paper])

        for task in grouped:
            fold = int(task["fold"])
            for paper, ids in extra_by_paper.items():
                target = task["test_ids"] if paper_to_fold[paper] == fold else task["train_ids"]
                target.extend(ids)

    years = pd.to_numeric(extra["paper_year_int"], errors="coerce")
    for task in extended:
        if task["split_type"] == "temporal_holdout" and task.get("temporal_cutoff_year") is not None:
            cutoff = int(task["temporal_cutoff_year"])
            task["train_ids"].extend(extra.loc[years <= cutoff, "sample_id"].tolist())
            task["test_ids"].extend(extra.loc[years > cutoff, "sample_id"].tolist())

    return extended


def ids_to_indices(
    work: pd.DataFrame,
    sample_ids: Sequence[str],
    minimum_component_count: int,
) -> np.ndarray:
    eligible = work[RAW_COMPONENT_COUNT] >= int(minimum_component_count)
    id_to_index = {
        sample_id: int(index)
        for index, sample_id in enumerate(work["sample_id"].astype(str))
        if bool(eligible.iloc[index])
    }
    indices = [id_to_index[sample_id] for sample_id in sample_ids if sample_id in id_to_index]
    return np.asarray(indices, dtype=int)


# =============================================================================
# 6. Model training
# =============================================================================


def make_inner_group_split(
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
        relative_fit, relative_valid = next(
            splitter.split(
                np.zeros(len(outer_train_indices)),
                groups=groups.to_numpy(),
            )
        )
        return (
            outer_train_indices[relative_fit],
            outer_train_indices[relative_valid],
            "group_shuffle",
        )

    relative_indices = np.arange(len(outer_train_indices))
    relative_fit, relative_valid = train_test_split(
        relative_indices,
        test_size=INNER_VALID_SIZE,
        random_state=SEED,
    )
    return (
        outer_train_indices[relative_fit],
        outer_train_indices[relative_valid],
        "random_fallback",
    )


def fit_point_model(
    X: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[lgb.Booster, int, str]:
    fit_indices, valid_indices, method = make_inner_group_split(
        train_indices,
        groups_all,
    )
    train_set = lgb.Dataset(
        X.iloc[fit_indices],
        label=y[fit_indices],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X.iloc[valid_indices],
        label=y[valid_indices],
        reference=train_set,
        free_raw_data=False,
    )
    model = lgb.train(
        params=dict(POINT_PARAMS),
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
        model.best_iteration
        if model.best_iteration and model.best_iteration > 0
        else MAX_BOOST_ROUNDS
    )
    return model, best_iteration, method


def quantile_params(alpha: float) -> Dict[str, Any]:
    params = dict(QUANTILE_BASE_PARAMS)
    params["alpha"] = float(alpha)
    return params


def fit_quantile_models(
    X: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[Dict[str, lgb.Booster], Dict[str, int], str]:
    fit_indices, valid_indices, method = make_inner_group_split(
        train_indices,
        groups_all,
    )

    models: Dict[str, lgb.Booster] = {}
    iterations: Dict[str, int] = {}

    for name, alpha in QUANTILES.items():
        train_set = lgb.Dataset(
            X.iloc[fit_indices],
            label=y[fit_indices],
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X.iloc[valid_indices],
            label=y[valid_indices],
            reference=train_set,
            free_raw_data=False,
        )
        model = lgb.train(
            params=quantile_params(alpha),
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
            model.best_iteration
            if model.best_iteration and model.best_iteration > 0
            else MAX_BOOST_ROUNDS
        )
        models[name] = model
        iterations[name] = best_iteration

    return models, iterations, method


def save_validation_model(
    model: lgb.Booster,
    paths: Paths,
    threshold: int,
    model_family: str,
    split_type: str,
    fold: int,
    model_name: str,
) -> str:
    directory = os.path.join(
        paths.validation_models_dir,
        f"min_components_{threshold}",
        model_family,
        split_type,
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{model_name}_fold{fold}.txt")
    model.save_model(path)
    return path


def validation_task_key(
    threshold: int,
    model_family: str,
    split_type: str,
    fold: int,
) -> str:
    return f"{threshold}|{model_family}|{split_type}|{fold}"


def run_validation_task(
    threshold: int,
    model_family: str,
    task: Mapping[str, Any],
    work: pd.DataFrame,
    X: pd.DataFrame,
    paths: Paths,
    save_models: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    split_type = str(task["split_type"])
    fold = int(task["fold"])
    cutoff = task.get("temporal_cutoff_year")

    train_indices_initial = ids_to_indices(
        work,
        task["train_ids"],
        threshold,
    )
    test_indices_initial = ids_to_indices(
        work,
        task["test_ids"],
        threshold,
    )

    audit = {
        "minimum_component_count": int(threshold),
        "model_family": model_family,
        "split_type": split_type,
        "fold": fold,
        "temporal_cutoff_year": cutoff,
        "manifest_train_rows": int(len(task["train_ids"])),
        "manifest_test_rows": int(len(task["test_ids"])),
        "threshold_retained_train_rows_before_normalization": int(
            len(train_indices_initial)
        ),
        "threshold_retained_test_rows_before_normalization": int(
            len(test_indices_initial)
        ),
        "train_test_sample_overlap": int(
            len(
                set(work.iloc[train_indices_initial]["sample_id"])
                & set(work.iloc[test_indices_initial]["sample_id"])
            )
        ),
        "train_test_paper_overlap": int(
            len(
                set(work.iloc[train_indices_initial]["paper_id"])
                & set(work.iloc[test_indices_initial]["paper_id"])
            )
        ),
    }

    if len(train_indices_initial) == 0 or len(test_indices_initial) == 0:
        metric = {
            "minimum_component_count": threshold,
            "model_family": model_family,
            "split_type": split_type,
            "fold": fold,
            "status": "skipped_empty_after_threshold",
            "temporal_cutoff_year": cutoff,
        }
        audit["status"] = metric["status"]
        return metric, pd.DataFrame(), {}, audit

    normalizer = fit_composite_normalizer(work, train_indices_initial)
    transformed = transform_composite(work, normalizer, threshold)
    y = pd.to_numeric(transformed[TARGET_NAME], errors="coerce").to_numpy(dtype=float)

    train_indices = train_indices_initial[np.isfinite(y[train_indices_initial])]
    test_indices = test_indices_initial[np.isfinite(y[test_indices_initial])]

    n_train_papers = int(work.iloc[train_indices]["paper_id"].nunique())
    n_test_papers = int(work.iloc[test_indices]["paper_id"].nunique())

    audit.update(
        {
            "valid_train_rows_after_normalization": int(len(train_indices)),
            "valid_test_rows_after_normalization": int(len(test_indices)),
            "valid_train_papers": n_train_papers,
            "valid_test_papers": n_test_papers,
            "all_train_component_count_ge_threshold": bool(
                (
                    transformed.iloc[train_indices]["composite_component_count"]
                    >= threshold
                ).all()
            )
            if len(train_indices) > 0
            else False,
            "all_test_component_count_ge_threshold": bool(
                (
                    transformed.iloc[test_indices]["composite_component_count"]
                    >= threshold
                ).all()
            )
            if len(test_indices) > 0
            else False,
        }
    )

    usable = (
        len(train_indices) >= MIN_OUTER_TRAIN_SAMPLES
        and len(test_indices) >= MIN_OUTER_TEST_SAMPLES
        and n_train_papers >= MIN_OUTER_TRAIN_PAPERS
        and n_test_papers >= MIN_OUTER_TEST_PAPERS
    )

    if not usable:
        metric = {
            "minimum_component_count": threshold,
            "model_family": model_family,
            "split_type": split_type,
            "fold": fold,
            "status": "skipped_too_few_valid_samples_or_papers",
            "temporal_cutoff_year": cutoff,
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "n_train_papers": n_train_papers,
            "n_test_papers": n_test_papers,
        }
        audit["status"] = metric["status"]
        normalization_row = {
            "minimum_component_count": threshold,
            "model_family": model_family,
            "split_type": split_type,
            "fold": fold,
            "temporal_cutoff_year": cutoff,
            **normalizer,
        }
        return metric, pd.DataFrame(), normalization_row, audit

    if model_family == POINT_FAMILY:
        model, best_iteration, inner_method = fit_point_model(
            X,
            y,
            train_indices,
            work["paper_id"],
        )
        y_pred = np.asarray(
            model.predict(
                X.iloc[test_indices],
                num_iteration=best_iteration,
            ),
            dtype=float,
        )
        y_true = y[test_indices]
        metrics = point_metrics(y_true, y_pred)
        model_path = ""
        if save_models:
            model_path = save_validation_model(
                model,
                paths,
                threshold,
                model_family,
                split_type,
                fold,
                "L1_composite",
            )

        prediction = work.iloc[test_indices][
            [
                "sample_id",
                "paper_id",
                "paper_year_int",
                "data_level",
                "growth_mode",
                "source_file",
                "synthesis_id",
            ]
        ].copy()
        prediction["minimum_component_count"] = threshold
        prediction["model_family"] = model_family
        prediction["split_type"] = split_type
        prediction["fold"] = fold
        prediction["temporal_cutoff_year"] = cutoff
        prediction["y_true"] = y_true
        prediction["y_pred"] = y_pred
        prediction["absolute_error"] = np.abs(y_true - y_pred)
        prediction["component_count"] = transformed.iloc[test_indices][
            "composite_component_count"
        ].to_numpy(dtype=int)
        prediction["weight_sum_used"] = transformed.iloc[test_indices][
            "composite_weight_sum_used"
        ].to_numpy(dtype=float)

        metric = {
            "minimum_component_count": threshold,
            "model_family": model_family,
            "split_type": split_type,
            "fold": fold,
            "status": "ok",
            "temporal_cutoff_year": cutoff,
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "n_train_papers": n_train_papers,
            "n_test_papers": n_test_papers,
            "inner_split_method": inner_method,
            "best_iteration": int(best_iteration),
            "model_path": model_path,
            **metrics,
        }

    elif model_family == QUANTILE_FAMILY:
        models, iterations, inner_method = fit_quantile_models(
            X,
            y,
            train_indices,
            work["paper_id"],
        )
        q10 = np.asarray(
            models["q10"].predict(
                X.iloc[test_indices],
                num_iteration=iterations["q10"],
            ),
            dtype=float,
        )
        q50 = np.asarray(
            models["q50"].predict(
                X.iloc[test_indices],
                num_iteration=iterations["q50"],
            ),
            dtype=float,
        )
        q90 = np.asarray(
            models["q90"].predict(
                X.iloc[test_indices],
                num_iteration=iterations["q90"],
            ),
            dtype=float,
        )
        y_true = y[test_indices]
        metrics = quantile_metrics(y_true, q10, q50, q90)

        model_paths: Dict[str, str] = {name: "" for name in QUANTILES}
        if save_models:
            for name, model in models.items():
                model_paths[name] = save_validation_model(
                    model,
                    paths,
                    threshold,
                    model_family,
                    split_type,
                    fold,
                    f"L12_composite_{name}",
                )

        ordered = np.sort(np.column_stack([q10, q50, q90]), axis=1)
        lower = ordered[:, 0]
        middle = ordered[:, 1]
        upper = ordered[:, 2]

        prediction = work.iloc[test_indices][
            [
                "sample_id",
                "paper_id",
                "paper_year_int",
                "data_level",
                "growth_mode",
                "source_file",
                "synthesis_id",
            ]
        ].copy()
        prediction["minimum_component_count"] = threshold
        prediction["model_family"] = model_family
        prediction["split_type"] = split_type
        prediction["fold"] = fold
        prediction["temporal_cutoff_year"] = cutoff
        prediction["y_true"] = y_true
        prediction["q10_raw"] = q10
        prediction["q50_raw"] = q50
        prediction["q90_raw"] = q90
        prediction["q10_ordered"] = lower
        prediction["q50_ordered"] = middle
        prediction["q90_ordered"] = upper
        prediction["interval_width_ordered"] = upper - lower
        prediction["covered_by_ordered_q10_q90"] = (
            (y_true >= lower) & (y_true <= upper)
        )
        prediction["any_crossing"] = (q10 > q50) | (q50 > q90)
        prediction["absolute_error_q50"] = np.abs(y_true - q50)
        prediction["component_count"] = transformed.iloc[test_indices][
            "composite_component_count"
        ].to_numpy(dtype=int)
        prediction["weight_sum_used"] = transformed.iloc[test_indices][
            "composite_weight_sum_used"
        ].to_numpy(dtype=float)

        metric = {
            "minimum_component_count": threshold,
            "model_family": model_family,
            "split_type": split_type,
            "fold": fold,
            "status": "ok",
            "temporal_cutoff_year": cutoff,
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "n_train_papers": n_train_papers,
            "n_test_papers": n_test_papers,
            "inner_split_method": inner_method,
            "best_iteration_q10": int(iterations["q10"]),
            "best_iteration_q50": int(iterations["q50"]),
            "best_iteration_q90": int(iterations["q90"]),
            "model_path_q10": model_paths["q10"],
            "model_path_q50": model_paths["q50"],
            "model_path_q90": model_paths["q90"],
            **metrics,
        }

    else:
        raise ValueError(f"未知model_family：{model_family}")

    normalization_row = {
        "minimum_component_count": threshold,
        "model_family": model_family,
        "split_type": split_type,
        "fold": fold,
        "temporal_cutoff_year": cutoff,
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        **normalizer,
    }

    audit["status"] = "ok"
    return metric, prediction.reset_index(drop=True), normalization_row, audit


# =============================================================================
# 7. Pooled metrics and clustered bootstrap
# =============================================================================


def pooled_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()

    group_columns = [
        "minimum_component_count",
        "model_family",
        "split_type",
    ]

    for keys, group in predictions.groupby(group_columns, sort=False):
        threshold, model_family, split_type = keys
        common = {
            "minimum_component_count": int(threshold),
            "model_family": model_family,
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(split_type, split_type),
            "n": int(len(group)),
            "papers": int(group["paper_id"].nunique()),
            "folds": int(group["fold"].nunique()),
        }
        if model_family == POINT_FAMILY:
            rows.append(
                {
                    **common,
                    **point_metrics(group["y_true"], group["y_pred"]),
                }
            )
        elif model_family == QUANTILE_FAMILY:
            rows.append(
                {
                    **common,
                    **quantile_metrics(
                        group["y_true"],
                        group["q10_raw"],
                        group["q50_raw"],
                        group["q90_raw"],
                    ),
                }
            )

    return pd.DataFrame(rows)


def bootstrap_metric_values(
    frame: pd.DataFrame,
    model_family: str,
    repeats: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    paper_groups = {
        paper_id: group.index.to_numpy(dtype=int)
        for paper_id, group in frame.groupby("paper_id", sort=False)
    }
    paper_ids = list(paper_groups)
    rng = np.random.default_rng(seed)

    if model_family == POINT_FAMILY:
        metric_names = ["mae", "rmse", "r2", "spearman"]
    else:
        metric_names = [
            "q50_mae",
            "q50_rmse",
            "q50_r2",
            "q50_spearman",
            "picp_80",
            "mpiw_80",
        ]

    values: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(repeats):
        sampled_papers = rng.choice(paper_ids, size=len(paper_ids), replace=True)
        sampled_indices = np.concatenate(
            [paper_groups[paper_id] for paper_id in sampled_papers]
        )
        sample = frame.loc[sampled_indices]

        if model_family == POINT_FAMILY:
            metrics = point_metrics(sample["y_true"], sample["y_pred"])
        else:
            metrics = quantile_metrics(
                sample["y_true"],
                sample["q10_raw"],
                sample["q50_raw"],
                sample["q90_raw"],
            )

        for name in metric_names:
            value = metrics.get(name, np.nan)
            if np.isfinite(value):
                values[name].append(float(value))

    return {
        name: np.asarray(metric_values, dtype=float)
        for name, metric_values in values.items()
    }


def run_cluster_bootstrap(
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()

    group_columns = [
        "minimum_component_count",
        "model_family",
        "split_type",
    ]

    for group_index, (keys, group) in enumerate(
        predictions.groupby(group_columns, sort=False),
        start=1,
    ):
        threshold, model_family, split_type = keys
        n_papers = int(group["paper_id"].nunique())
        status = "ok"

        if len(group) < 10 or n_papers < 3:
            status = "skipped_too_few_samples_or_papers"
            rows.append(
                {
                    "minimum_component_count": int(threshold),
                    "model_family": model_family,
                    "split_type": split_type,
                    "metric": "",
                    "estimate": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "valid_repeats": 0,
                    "requested_repeats": int(repeats),
                    "status": status,
                }
            )
            continue

        distributions = bootstrap_metric_values(
            frame=group.reset_index(drop=True),
            model_family=model_family,
            repeats=repeats,
            seed=seed + group_index * 1009,
        )

        if model_family == POINT_FAMILY:
            estimates = point_metrics(group["y_true"], group["y_pred"])
        else:
            estimates = quantile_metrics(
                group["y_true"],
                group["q10_raw"],
                group["q50_raw"],
                group["q90_raw"],
            )

        for metric_name, distribution in distributions.items():
            distribution = distribution[np.isfinite(distribution)]
            rows.append(
                {
                    "minimum_component_count": int(threshold),
                    "model_family": model_family,
                    "split_type": split_type,
                    "metric": metric_name,
                    "estimate": estimates.get(metric_name, np.nan),
                    "ci_lower": float(np.quantile(distribution, 0.025))
                    if len(distribution) > 0
                    else np.nan,
                    "ci_upper": float(np.quantile(distribution, 0.975))
                    if len(distribution) > 0
                    else np.nan,
                    "valid_repeats": int(len(distribution)),
                    "requested_repeats": int(repeats),
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 8. Historical replay
# =============================================================================


def resolve_cutoffs(paths: Paths, requested: Optional[Sequence[int]]) -> List[int]:
    if requested:
        return sorted(set(int(value) for value in requested))

    if os.path.isfile(paths.retrospective_cutoff_manifest):
        manifest = pd.read_csv(paths.retrospective_cutoff_manifest, low_memory=False)
        if "cutoff_year" in manifest.columns:
            selected_mask = pd.Series(True, index=manifest.index)
            if "selected" in manifest.columns:
                selected_text = manifest["selected"].astype(str).str.lower()
                selected_mask = selected_text.isin(["true", "1", "yes"])
            selected = pd.to_numeric(
                manifest.loc[selected_mask, "cutoff_year"],
                errors="coerce",
            ).dropna().astype(int).tolist()
            if selected:
                return sorted(set(selected))

    return list(DEFAULT_CUTOFFS)


def make_historical_inner_split(
    training: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, str]:
    years = pd.to_numeric(training["paper_year_int"], errors="coerce")
    unique_years = sorted(int(year) for year in years.dropna().unique())
    target_valid = max(20, int(math.ceil(INNER_VALID_SIZE * len(training))))
    selected_years: List[int] = []
    count = 0

    for year in reversed(unique_years):
        selected_years.append(year)
        count += int((years == year).sum())
        fit_mask = years < min(selected_years)
        valid_mask = years.isin(selected_years)
        if (
            count >= target_valid
            and int(fit_mask.sum()) >= 30
            and training.loc[fit_mask, "paper_id"].nunique() >= 3
            and training.loc[valid_mask, "paper_id"].nunique() >= 2
        ):
            return (
                np.where(fit_mask.to_numpy())[0],
                np.where(valid_mask.to_numpy())[0],
                "inner_temporal_year_block",
            )

    groups = training["paper_id"].fillna("").astype(str)
    if groups.nunique() < 2:
        relative = np.arange(len(training))
        fit, valid = train_test_split(
            relative,
            test_size=INNER_VALID_SIZE,
            random_state=SEED,
        )
        return np.asarray(fit), np.asarray(valid), "inner_random_fallback"

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VALID_SIZE,
        random_state=SEED,
    )
    fit, valid = next(
        splitter.split(
            np.zeros(len(training)),
            groups=groups.to_numpy(),
        )
    )
    return np.asarray(fit), np.asarray(valid), "inner_group_shuffle_fallback"


def select_historical_iteration(
    X: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
    alpha: float,
) -> Tuple[int, str]:
    fit, valid, method = make_historical_inner_split(metadata)
    train_set = lgb.Dataset(
        X.iloc[fit],
        label=y[fit],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X.iloc[valid],
        label=y[valid],
        reference=train_set,
        free_raw_data=False,
    )
    model = lgb.train(
        params=quantile_params(alpha),
        train_set=train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[train_set, valid_set],
        valid_names=["inner_fit", "inner_valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    iteration = int(
        model.best_iteration
        if model.best_iteration and model.best_iteration > 0
        else MAX_BOOST_ROUNDS
    )
    return iteration, method


def fit_historical_final_quantile(
    X: pd.DataFrame,
    y: np.ndarray,
    alpha: float,
    iteration: int,
) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y, free_raw_data=False)
    return lgb.train(
        params=quantile_params(alpha),
        train_set=dataset,
        num_boost_round=int(iteration),
        callbacks=[lgb.log_evaluation(period=0)],
    )


def save_historical_model(
    model: lgb.Booster,
    paths: Paths,
    threshold: int,
    cutoff: int,
    quantile_name: str,
) -> str:
    directory = os.path.join(
        paths.historical_models_dir,
        f"min_components_{threshold}",
        f"cutoff_{cutoff}",
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"L12_composite_{quantile_name}.txt")
    model.save_model(path)
    return path


def dcg(relevance: Sequence[float], k: int) -> float:
    values = np.asarray(relevance, dtype=float)[:k]
    if len(values) == 0:
        return float("nan")
    discounts = np.log2(np.arange(2, len(values) + 2, dtype=float))
    gains = np.power(2.0, values) - 1.0
    return float(np.sum(gains / discounts))


def ndcg(
    ranked_relevance: Sequence[float],
    all_relevance: Sequence[float],
    k: int,
) -> float:
    actual = dcg(ranked_relevance, k)
    ideal = dcg(np.sort(np.asarray(all_relevance, dtype=float))[::-1], k)
    if not np.isfinite(actual) or not np.isfinite(ideal) or ideal <= 0:
        return float("nan")
    return float(actual / ideal)


def evaluate_historical_strategy(
    frame: pd.DataFrame,
    threshold: int,
    cutoff: int,
    strategy: str,
    positive_fraction: float,
    screen_fraction: float,
) -> Dict[str, Any]:
    score_column = HISTORICAL_STRATEGIES[strategy]
    ranked = frame.sort_values(
        score_column,
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    pool_size = len(ranked)
    k = max(1, min(pool_size, int(math.ceil(screen_fraction * pool_size))))
    true_threshold = float(
        ranked["y_true_replay"].quantile(1.0 - positive_fraction)
    )
    positive = ranked["y_true_replay"] >= true_threshold
    total_positive = int(positive.sum())
    prevalence = total_positive / pool_size if pool_size > 0 else np.nan

    top = ranked.iloc[:k]
    hits = int((top["y_true_replay"] >= true_threshold).sum())
    precision = hits / k if k > 0 else np.nan
    recall = hits / total_positive if total_positive > 0 else np.nan
    enrichment = (
        precision / prevalence
        if np.isfinite(prevalence) and prevalence > 0
        else np.nan
    )

    true_rank = ranked["y_true_replay"].rank(method="average", ascending=False)
    continuous_relevance = 1.0 - (true_rank - 1.0) / max(1.0, pool_size - 1.0)
    binary_relevance = positive.astype(float).to_numpy()

    return {
        "minimum_component_count": int(threshold),
        "cutoff_year": int(cutoff),
        "strategy": strategy,
        "strategy_label": HISTORICAL_STRATEGY_LABELS[strategy],
        "score_column": score_column,
        "pool_size": int(pool_size),
        "future_papers": int(ranked["paper_id"].nunique()),
        "screen_fraction": float(screen_fraction),
        "k": int(k),
        "positive_fraction": float(positive_fraction),
        "true_positive_threshold": true_threshold,
        "total_positives": total_positive,
        "positive_prevalence": prevalence,
        "hits_at_k": hits,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_rate_at_k": float(hits > 0),
        "enrichment_factor_at_k": enrichment,
        "ndcg_binary_at_k": ndcg(
            binary_relevance,
            binary_relevance,
            k,
        ),
        "ndcg_continuous_at_k": ndcg(
            continuous_relevance.to_numpy(dtype=float),
            continuous_relevance.to_numpy(dtype=float),
            k,
        ),
        "spearman_all_future": safe_spearman(
            ranked["y_true_replay"],
            ranked[score_column],
        ),
    }


def random_historical_baseline(
    frame: pd.DataFrame,
    threshold: int,
    cutoff: int,
    positive_fraction: float,
    screen_fraction: float,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    pool_size = len(frame)
    k = max(1, min(pool_size, int(math.ceil(screen_fraction * pool_size))))
    true_threshold = float(
        frame["y_true_replay"].quantile(1.0 - positive_fraction)
    )
    positive = (frame["y_true_replay"] >= true_threshold).to_numpy(dtype=bool)
    total_positive = int(positive.sum())
    prevalence = total_positive / pool_size if pool_size > 0 else np.nan

    true_rank = frame["y_true_replay"].rank(method="average", ascending=False)
    continuous_relevance = (
        1.0 - (true_rank - 1.0) / max(1.0, pool_size - 1.0)
    ).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    distributions: Dict[str, List[float]] = {
        "precision_at_k": [],
        "recall_at_k": [],
        "hit_rate_at_k": [],
        "enrichment_factor_at_k": [],
        "ndcg_binary_at_k": [],
        "ndcg_continuous_at_k": [],
    }

    for _ in range(repeats):
        order = rng.permutation(pool_size)
        positive_ranked = positive[order]
        relevance_ranked = continuous_relevance[order]
        hits = int(positive_ranked[:k].sum())
        precision = hits / k if k > 0 else np.nan
        recall = hits / total_positive if total_positive > 0 else np.nan
        enrichment = (
            precision / prevalence
            if np.isfinite(prevalence) and prevalence > 0
            else np.nan
        )

        distributions["precision_at_k"].append(precision)
        distributions["recall_at_k"].append(recall)
        distributions["hit_rate_at_k"].append(float(hits > 0))
        distributions["enrichment_factor_at_k"].append(enrichment)
        distributions["ndcg_binary_at_k"].append(
            ndcg(positive_ranked.astype(float), positive.astype(float), k)
        )
        distributions["ndcg_continuous_at_k"].append(
            ndcg(relevance_ranked, continuous_relevance, k)
        )

    row: Dict[str, Any] = {
        "minimum_component_count": int(threshold),
        "cutoff_year": int(cutoff),
        "pool_size": int(pool_size),
        "k": int(k),
        "positive_fraction": float(positive_fraction),
        "screen_fraction": float(screen_fraction),
        "random_repeats": int(repeats),
    }

    for metric_name, values in distributions.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        row[f"random_{metric_name}_mean"] = (
            float(np.mean(array)) if len(array) > 0 else np.nan
        )
        row[f"random_{metric_name}_ci_lower"] = (
            float(np.quantile(array, 0.025)) if len(array) > 0 else np.nan
        )
        row[f"random_{metric_name}_ci_upper"] = (
            float(np.quantile(array, 0.975)) if len(array) > 0 else np.nan
        )

    return row


def run_historical_cutoff(
    work: pd.DataFrame,
    X: pd.DataFrame,
    feature_columns: Sequence[str],
    threshold: int,
    cutoff: int,
    paths: Paths,
    save_models: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    threshold_mask = work[RAW_COMPONENT_COUNT] >= int(threshold)
    dated_mask = work["paper_year_int"].notna()
    subset_indices = np.where((threshold_mask & dated_mask).to_numpy())[0]
    subset = work.iloc[subset_indices].copy().reset_index(drop=True)
    X_subset = X.iloc[subset_indices].copy().reset_index(drop=True)

    early_initial = np.where(
        (subset["paper_year_int"] <= cutoff).to_numpy()
    )[0]
    future_initial = np.where(
        (subset["paper_year_int"] > cutoff).to_numpy()
    )[0]

    if len(early_initial) == 0 or len(future_initial) == 0:
        raise RuntimeError(
            f"threshold={threshold}, cutoff={cutoff}: 无训练或未来样本。"
        )

    normalizer = fit_composite_normalizer(subset, early_initial)
    transformed = transform_composite(subset, normalizer, threshold)
    y = pd.to_numeric(transformed[TARGET_NAME], errors="coerce").to_numpy(dtype=float)

    early_indices = early_initial[np.isfinite(y[early_initial])]
    future_indices = future_initial[np.isfinite(y[future_initial])]

    early = subset.iloc[early_indices].copy().reset_index(drop=True)
    future = subset.iloc[future_indices].copy().reset_index(drop=True)
    X_early = X_subset.iloc[early_indices].copy().reset_index(drop=True)
    X_future = X_subset.iloc[future_indices].copy().reset_index(drop=True)
    y_early = y[early_indices]
    y_future = y[future_indices]

    if (
        len(early) < MIN_HISTORICAL_TRAIN_SAMPLES
        or len(future) < MIN_HISTORICAL_FUTURE_SAMPLES
        or early["paper_id"].nunique() < MIN_HISTORICAL_TRAIN_PAPERS
        or future["paper_id"].nunique() < MIN_HISTORICAL_FUTURE_PAPERS
    ):
        raise RuntimeError(
            f"threshold={threshold}, cutoff={cutoff}: "
            "历史训练或未来候选样本/论文不足。"
        )

    prediction_columns: Dict[str, np.ndarray] = {}
    model_rows: List[Dict[str, Any]] = []

    for name, alpha in QUANTILES.items():
        iteration, inner_method = select_historical_iteration(
            X_early,
            y_early,
            early,
            alpha,
        )
        model = fit_historical_final_quantile(
            X_early,
            y_early,
            alpha,
            iteration,
        )
        prediction_columns[name] = np.asarray(
            model.predict(X_future, num_iteration=iteration),
            dtype=float,
        )
        model_path = ""
        if save_models:
            model_path = save_historical_model(
                model,
                paths,
                threshold,
                cutoff,
                name,
            )
        model_rows.append(
            {
                "minimum_component_count": int(threshold),
                "cutoff_year": int(cutoff),
                "quantile_name": name,
                "alpha": float(alpha),
                "n_train": int(len(early)),
                "n_train_papers": int(early["paper_id"].nunique()),
                "n_future": int(len(future)),
                "n_future_papers": int(future["paper_id"].nunique()),
                "best_iteration": int(iteration),
                "inner_split_method": inner_method,
                "model_path": model_path,
                "igid_min": normalizer.get("igid_min"),
                "igid_max": normalizer.get("igid_max"),
                "bundle_log_min": normalizer.get("bundle_log_min"),
                "bundle_log_max": normalizer.get("bundle_log_max"),
            }
        )

    output = future[
        [
            "sample_id",
            "paper_id",
            "paper_year_int",
            "data_level",
            "growth_mode",
            "source_file",
            "synthesis_id",
        ]
    ].copy().reset_index(drop=True)
    output["minimum_component_count"] = int(threshold)
    output["cutoff_year"] = int(cutoff)
    output["y_true_replay"] = y_future
    output["q10_raw"] = prediction_columns["q10"]
    output["q50_raw"] = prediction_columns["q50"]
    output["q90_raw"] = prediction_columns["q90"]
    output["component_count"] = transformed.iloc[future_indices][
        "composite_component_count"
    ].to_numpy(dtype=int)
    output["weight_sum_used"] = transformed.iloc[future_indices][
        "composite_weight_sum_used"
    ].to_numpy(dtype=float)

    return output, pd.DataFrame(model_rows)


def summarize_historical_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_columns = ["minimum_component_count", "strategy", "strategy_label"]
    numeric_metrics = [
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "enrichment_factor_at_k",
        "ndcg_binary_at_k",
        "ndcg_continuous_at_k",
        "spearman_all_future",
    ]

    for keys, group in metrics.groupby(group_columns, sort=False):
        threshold, strategy, strategy_label = keys
        row: Dict[str, Any] = {
            "minimum_component_count": int(threshold),
            "strategy": strategy,
            "strategy_label": strategy_label,
            "cutoff_count": int(group["cutoff_year"].nunique()),
            "mean_pool_size": float(group["pool_size"].mean()),
        }
        for metric_name in numeric_metrics:
            row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            row[f"{metric_name}_std"] = (
                float(group[metric_name].std(ddof=1))
                if len(group) > 1
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 9. Coverage, audits and comparison table
# =============================================================================


def build_component_coverage(
    dataset: pd.DataFrame,
    thresholds: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_family, levels in [
        (POINT_FAMILY, ["L1"]),
        (QUANTILE_FAMILY, ["L1", "L2"]),
    ]:
        frame = dataset[dataset["data_level"].isin(levels)].copy()
        base_count = int((frame[RAW_COMPONENT_COUNT] >= 1).sum())
        base_papers = int(
            frame.loc[frame[RAW_COMPONENT_COUNT] >= 1, "paper_id"].nunique()
        )
        for threshold in thresholds:
            retained = frame[frame[RAW_COMPONENT_COUNT] >= threshold]
            rows.append(
                {
                    "model_family": model_family,
                    "levels": "+".join(levels),
                    "minimum_component_count": int(threshold),
                    "samples": int(len(retained)),
                    "papers": int(retained["paper_id"].nunique()),
                    "dated_samples": int(retained["paper_year_int"].notna().sum()),
                    "sample_retention_vs_threshold1": (
                        len(retained) / base_count if base_count > 0 else np.nan
                    ),
                    "paper_retention_vs_threshold1": (
                        retained["paper_id"].nunique() / base_papers
                        if base_papers > 0
                        else np.nan
                    ),
                    "component_count_1": int(
                        (retained[RAW_COMPONENT_COUNT] == 1).sum()
                    ),
                    "component_count_2": int(
                        (retained[RAW_COMPONENT_COUNT] == 2).sum()
                    ),
                    "component_count_3": int(
                        (retained[RAW_COMPONENT_COUNT] == 3).sum()
                    ),
                    "component_count_4": int(
                        (retained[RAW_COMPONENT_COUNT] == 4).sum()
                    ),
                    "component_count_5": int(
                        (retained[RAW_COMPONENT_COUNT] == 5).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_columns(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    keys: Sequence[str],
    pairs: Sequence[Tuple[str, str]],
    context: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if current.empty or baseline.empty:
        for current_column, baseline_column in pairs:
            rows.append(
                {
                    **context,
                    "current_column": current_column,
                    "baseline_column": baseline_column,
                    "shared_rows": 0,
                    "max_absolute_difference": np.nan,
                    "mean_absolute_difference": np.nan,
                    "all_within_tolerance": False,
                    "status": "missing_optional_baseline_or_current",
                }
            )
        return rows

    merged = current.merge(
        baseline,
        on=list(keys),
        how="inner",
        suffixes=("_current", "_baseline"),
    )

    for current_column, baseline_column in pairs:
        current_name = (
            f"{current_column}_current"
            if current_column in baseline.columns
            else current_column
        )
        baseline_name = (
            f"{baseline_column}_baseline"
            if baseline_column in current.columns
            else baseline_column
        )

        if current_name not in merged.columns or baseline_name not in merged.columns:
            rows.append(
                {
                    **context,
                    "current_column": current_column,
                    "baseline_column": baseline_column,
                    "shared_rows": int(len(merged)),
                    "max_absolute_difference": np.nan,
                    "mean_absolute_difference": np.nan,
                    "all_within_tolerance": False,
                    "status": "comparison_column_missing",
                }
            )
            continue

        left = pd.to_numeric(merged[current_name], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[baseline_name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        differences = np.abs(left[finite] - right[finite])

        rows.append(
            {
                **context,
                "current_column": current_column,
                "baseline_column": baseline_column,
                "shared_rows": int(finite.sum()),
                "max_absolute_difference": (
                    float(np.max(differences)) if len(differences) > 0 else np.nan
                ),
                "mean_absolute_difference": (
                    float(np.mean(differences)) if len(differences) > 0 else np.nan
                ),
                "all_within_tolerance": bool(
                    len(differences) > 0
                    and np.all(differences <= BASELINE_REPRODUCTION_TOLERANCE)
                ),
                "status": "ok" if len(differences) > 0 else "no_finite_shared_rows",
            }
        )

    return rows


def build_baseline_reproduction_audit(
    predictions: pd.DataFrame,
    paths: Paths,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    current_l1 = predictions[
        (predictions["minimum_component_count"] == FORMAL_BASELINE_THRESHOLD)
        & (predictions["model_family"] == POINT_FAMILY)
    ].copy()
    baseline_l1 = load_optional_csv(paths.l1_baseline_predictions)
    rows.extend(
        compare_columns(
            current=current_l1,
            baseline=baseline_l1,
            keys=["sample_id", "split_type", "fold"],
            pairs=[("y_true", "y_true"), ("y_pred", "y_pred")],
            context={"model_family": POINT_FAMILY},
        )
    )

    current_l12 = predictions[
        (predictions["minimum_component_count"] == FORMAL_BASELINE_THRESHOLD)
        & (predictions["model_family"] == QUANTILE_FAMILY)
    ].copy()
    baseline_l12 = load_optional_csv(paths.l12_baseline_predictions)
    rows.extend(
        compare_columns(
            current=current_l12,
            baseline=baseline_l12,
            keys=["sample_id", "split_type", "fold"],
            pairs=[
                ("y_true", "y_true"),
                ("q10_raw", "q10_raw"),
                ("q50_raw", "q50_raw"),
                ("q90_raw", "q90_raw"),
            ],
            context={"model_family": QUANTILE_FAMILY},
        )
    )

    return pd.DataFrame(rows)


def build_historical_baseline_audit(
    historical_predictions: pd.DataFrame,
    paths: Paths,
) -> pd.DataFrame:
    current = historical_predictions[
        historical_predictions["minimum_component_count"] == 1
    ].copy()
    baseline = load_optional_csv(paths.retrospective_baseline_predictions)
    return pd.DataFrame(
        compare_columns(
            current=current,
            baseline=baseline,
            keys=["sample_id", "cutoff_year"],
            pairs=[
                ("y_true_replay", "y_true_replay"),
                ("q10_raw", "pred_q10_raw"),
                ("q50_raw", "pred_q50_raw"),
                ("q90_raw", "pred_q90_raw"),
            ],
            context={"analysis": "historical_replay"},
        )
    )


def build_comparison_table(
    coverage: pd.DataFrame,
    pooled: pd.DataFrame,
    historical_summary: pd.DataFrame,
    thresholds: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for threshold in thresholds:
        row: Dict[str, Any] = {"minimum_component_count": int(threshold)}

        for model_family, prefix in [
            (POINT_FAMILY, "L1"),
            (QUANTILE_FAMILY, "L12"),
        ]:
            coverage_row = coverage[
                (coverage["minimum_component_count"] == threshold)
                & (coverage["model_family"] == model_family)
            ]
            if not coverage_row.empty:
                item = coverage_row.iloc[0]
                row[f"{prefix}_samples"] = item["samples"]
                row[f"{prefix}_papers"] = item["papers"]
                row[f"{prefix}_sample_retention"] = item[
                    "sample_retention_vs_threshold1"
                ]

        for model_family, prefix in [
            (POINT_FAMILY, "L1_point"),
            (QUANTILE_FAMILY, "L12_q50"),
        ]:
            for split_type, split_prefix in [
                ("paper_grouped_cv", "grouped"),
                ("temporal_holdout", "temporal"),
            ]:
                metric_row = pooled[
                    (pooled["minimum_component_count"] == threshold)
                    & (pooled["model_family"] == model_family)
                    & (pooled["split_type"] == split_type)
                ]
                if metric_row.empty:
                    continue
                item = metric_row.iloc[0]
                if model_family == POINT_FAMILY:
                    row[f"{prefix}_{split_prefix}_mae"] = item.get("mae", np.nan)
                    row[f"{prefix}_{split_prefix}_r2"] = item.get("r2", np.nan)
                    row[f"{prefix}_{split_prefix}_spearman"] = item.get(
                        "spearman", np.nan
                    )
                else:
                    row[f"{prefix}_{split_prefix}_mae"] = item.get(
                        "q50_mae", np.nan
                    )
                    row[f"{prefix}_{split_prefix}_r2"] = item.get(
                        "q50_r2", np.nan
                    )
                    row[f"{prefix}_{split_prefix}_spearman"] = item.get(
                        "q50_spearman", np.nan
                    )
                    row[f"{prefix}_{split_prefix}_picp80"] = item.get(
                        "picp_80", np.nan
                    )

        for strategy, prefix in [
            ("q50_balanced", "historical_q50"),
            ("q90_potential", "historical_q90"),
        ]:
            history = historical_summary[
                (historical_summary["minimum_component_count"] == threshold)
                & (historical_summary["strategy"] == strategy)
            ]
            if history.empty:
                continue
            item = history.iloc[0]
            row[f"{prefix}_cutoff_count"] = item.get("cutoff_count", np.nan)
            row[f"{prefix}_ndcg_mean"] = item.get(
                "ndcg_continuous_at_k_mean", np.nan
            )
            row[f"{prefix}_recall_mean"] = item.get(
                "recall_at_k_mean", np.nan
            )
            row[f"{prefix}_enrichment_mean"] = item.get(
                "enrichment_factor_at_k_mean", np.nan
            )
            row[f"{prefix}_spearman_mean"] = item.get(
                "spearman_all_future_mean", np.nan
            )

        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        "minimum_component_count"
    ).reset_index(drop=True)

    if not comparison.empty:
        baseline = comparison[
            comparison["minimum_component_count"] == 1
        ]
        if not baseline.empty:
            base = baseline.iloc[0]
            for column in [
                "L1_point_grouped_spearman",
                "L1_point_temporal_spearman",
                "L12_q50_grouped_spearman",
                "L12_q50_temporal_spearman",
                "historical_q50_ndcg_mean",
                "historical_q50_enrichment_mean",
                "historical_q90_ndcg_mean",
                "historical_q90_enrichment_mean",
            ]:
                if column in comparison.columns:
                    comparison[f"delta_vs_threshold1__{column}"] = (
                        comparison[column] - base.get(column, np.nan)
                    )

    return comparison


# =============================================================================
# 10. Figures
# =============================================================================


def apply_axes(ax: plt.Axes, grid_axis: Optional[str] = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    if grid_axis:
        ax.grid(axis=grid_axis, linewidth=0.5, alpha=0.25)


def save_figure(fig: plt.Figure, base_path: str, dpi: int) -> None:
    fig.savefig(base_path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sample_retention(
    coverage: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    source = coverage.copy()
    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            "fig_composite_sensitivity_sample_retention_source_data.csv",
        ),
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for model_family in MODEL_FAMILY_ORDER:
        group = source[source["model_family"] == model_family].sort_values(
            "minimum_component_count"
        )
        ax.plot(
            group["minimum_component_count"],
            group["sample_retention_vs_threshold1"],
            marker="o",
            linewidth=1.6,
            label=model_family,
        )
    ax.set_xticks(sorted(source["minimum_component_count"].unique()))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Minimum number of composite components")
    ax.set_ylabel("Sample retention relative to threshold 1")
    ax.set_title("Composite-label sample retention")
    ax.legend(frameon=False)
    apply_axes(ax, "both")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig_composite_sensitivity_sample_retention",
        ),
        dpi,
    )


def plot_validation_spearman(
    pooled: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for _, item in pooled.iterrows():
        if item["split_type"] not in ["paper_grouped_cv", "temporal_holdout"]:
            continue
        metric = (
            item.get("spearman", np.nan)
            if item["model_family"] == POINT_FAMILY
            else item.get("q50_spearman", np.nan)
        )
        rows.append(
            {
                "minimum_component_count": int(item["minimum_component_count"]),
                "model_family": item["model_family"],
                "split_type": item["split_type"],
                "split_label": SPLIT_LABELS[item["split_type"]],
                "spearman": metric,
            }
        )
    source = pd.DataFrame(rows)
    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            "fig_composite_sensitivity_validation_spearman_source_data.csv",
        ),
    )

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    for (model_family, split_type), group in source.groupby(
        ["model_family", "split_type"],
        sort=False,
    ):
        group = group.sort_values("minimum_component_count")
        ax.plot(
            group["minimum_component_count"],
            group["spearman"],
            marker="o",
            linewidth=1.6,
            label=f"{model_family} — {SPLIT_LABELS[split_type]}",
        )
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xticks(sorted(source["minimum_component_count"].unique()))
    ax.set_xlabel("Minimum number of composite components")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Composite sensitivity under strict validation")
    ax.legend(frameon=False, ncol=2)
    apply_axes(ax, "both")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig_composite_sensitivity_validation_spearman",
        ),
        dpi,
    )


def plot_validation_mae(
    pooled: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    source = pooled[
        (pooled["model_family"] == QUANTILE_FAMILY)
        & pooled["split_type"].isin(["paper_grouped_cv", "temporal_holdout"])
    ][
        [
            "minimum_component_count",
            "split_type",
            "split_label",
            "q50_mae",
        ]
    ].copy()
    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            "fig_composite_sensitivity_q50_mae_source_data.csv",
        ),
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for split_type, group in source.groupby("split_type", sort=False):
        group = group.sort_values("minimum_component_count")
        ax.plot(
            group["minimum_component_count"],
            group["q50_mae"],
            marker="o",
            linewidth=1.6,
            label=SPLIT_LABELS[split_type],
        )
    ax.set_xticks(sorted(source["minimum_component_count"].unique()))
    ax.set_xlabel("Minimum number of composite components")
    ax.set_ylabel("q50 MAE")
    ax.set_title("Composite q50 error versus label completeness")
    ax.legend(frameon=False)
    apply_axes(ax, "both")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "fig_composite_sensitivity_q50_mae"),
        dpi,
    )


def plot_historical_metric(
    historical_metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    paths: Paths,
    dpi: int,
) -> None:
    source = historical_metrics.copy()
    atomic_write_csv(
        source,
        os.path.join(paths.source_data_dir, f"{filename}_source_data.csv"),
    )

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    for (threshold, strategy), group in source.groupby(
        ["minimum_component_count", "strategy"],
        sort=False,
    ):
        group = group.sort_values("cutoff_year")
        ax.plot(
            group["cutoff_year"],
            group[metric],
            marker="o",
            linewidth=1.5,
            label=(
                f"≥{threshold} components — "
                f"{HISTORICAL_STRATEGY_LABELS[strategy]}"
            ),
        )
    ax.set_xlabel("Historical cutoff year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    apply_axes(ax, "both")
    fig.tight_layout()
    save_figure(fig, os.path.join(paths.figures_dir, filename), dpi)


def plot_historical_enrichment_summary(
    historical_summary: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    source = historical_summary.copy()
    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            "fig_composite_sensitivity_historical_enrichment_source_data.csv",
        ),
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    thresholds = sorted(source["minimum_component_count"].unique())
    x = np.arange(len(thresholds), dtype=float)
    width = 0.36

    for index, strategy in enumerate(["q50_balanced", "q90_potential"]):
        values: List[float] = []
        errors: List[float] = []
        for threshold in thresholds:
            row = source[
                (source["minimum_component_count"] == threshold)
                & (source["strategy"] == strategy)
            ]
            values.append(
                float(row.iloc[0]["enrichment_factor_at_k_mean"])
                if not row.empty
                else np.nan
            )
            errors.append(
                float(row.iloc[0]["enrichment_factor_at_k_std"])
                if not row.empty
                else np.nan
            )
        offset = (index - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=HISTORICAL_STRATEGY_LABELS[strategy],
        )

    ax.axhline(1.0, linewidth=1.0, linestyle="--", label="Random expectation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{threshold}" for threshold in thresholds])
    ax.set_xlabel("Minimum number of composite components")
    ax.set_ylabel("Mean enrichment factor@Top 10%")
    ax.set_title("Historical enrichment sensitivity")
    ax.legend(frameon=False)
    apply_axes(ax, "y")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig_composite_sensitivity_historical_enrichment",
        ),
        dpi,
    )


# =============================================================================
# 11. Checkpoint configuration
# =============================================================================


def configuration_payload(
    paths: Paths,
    thresholds: Sequence[int],
    cutoffs: Sequence[int],
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "dataset_sha256": sha256_file(paths.dataset),
        "schema_sha256": sha256_file(paths.schema),
        "l1_manifest_sha256": sha256_file(paths.l1_split_manifest),
        "l12_manifest_sha256": sha256_file(paths.l12_split_manifest),
        "formal_baseline_threshold": int(FORMAL_BASELINE_THRESHOLD),
        "threshold1_split_extension": True,
        "thresholds": [int(value) for value in thresholds],
        "cutoffs": [int(value) for value in cutoffs],
        "feature_columns": list(feature_columns),
        "bootstrap": int(args.bootstrap),
        "random_repeats": int(args.random_repeats),
        "positive_fraction": float(args.positive_fraction),
        "screen_fraction": float(args.screen_fraction),
        "seed": int(args.seed),
        "point_params": POINT_PARAMS,
        "quantile_base_params": QUANTILE_BASE_PARAMS,
        "weights": WEIGHTS,
        "missing_policy": MISSING_POLICY,
    }


def payload_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def prepare_checkpoint(paths: Paths, payload: Mapping[str, Any]) -> None:
    current_hash = payload_hash(payload)
    if os.path.isfile(paths.checkpoint_config):
        with open(paths.checkpoint_config, "r", encoding="utf-8") as file:
            previous = json.load(file)
        if previous.get("configuration_hash") != current_hash:
            raise RuntimeError(
                "现有断点结果与当前数据、阈值、cutoff或参数不一致。"
                "请使用 --fresh 重新运行。"
            )
    else:
        atomic_write_json(
            {
                "configuration_hash": current_hash,
                "configuration": payload,
            },
            paths.checkpoint_config,
        )


# =============================================================================
# 12. Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "测试Composite至少需要1、2或3个组成指标时，"
            "严格验证与历史回放结论是否稳定。"
        )
    )
    parser.add_argument(
        "--project_dir",
        default=None,
        help="项目根目录；默认由 models/external_validation/ 向上两级自动定位。",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录；默认 outputs/external_validation/composite_sensitivity。",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLDS,
        help="最低Composite组成指标数，默认 1 2 3。",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=None,
        help=(
            "历史回放年份；默认读取已有cutoff_selection_manifest，"
            "读取失败时使用2012 2015 2017 2020。"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
        help="按paper_id聚类bootstrap次数，默认2000。",
    )
    parser.add_argument(
        "--random_repeats",
        type=int,
        default=DEFAULT_RANDOM_RANKING_REPEATS,
        help="每个历史候选池随机排序次数，默认1000。",
    )
    parser.add_argument(
        "--positive_fraction",
        type=float,
        default=DEFAULT_POSITIVE_FRACTION,
        help="历史候选池真实高质量比例，默认0.20。",
    )
    parser.add_argument(
        "--screen_fraction",
        type=float,
        default=DEFAULT_SCREEN_FRACTION,
        help="历史回放筛选预算比例，默认0.10。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="保存全部验证与历史模型。",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="删除现有输出并完全重算。",
    )
    args = parser.parse_args()

    thresholds = sorted(set(int(value) for value in args.thresholds))
    if not thresholds or any(value < 1 or value > 5 for value in thresholds):
        raise ValueError("--thresholds必须是1–5之间的整数。")
    if 1 not in thresholds:
        raise ValueError("敏感性分析必须包含threshold=1作为基线。")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap必须大于0。")
    if args.random_repeats < 1:
        raise ValueError("--random_repeats必须大于0。")
    if not 0.0 < args.positive_fraction < 1.0:
        raise ValueError("--positive_fraction必须位于0和1之间。")
    if not 0.0 < args.screen_fraction < 1.0:
        raise ValueError("--screen_fraction必须位于0和1之间。")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    project_dir = (
        os.path.abspath(args.project_dir)
        if args.project_dir
        else default_project_dir
    )
    paths = build_paths(project_dir, args.output_dir)
    ensure_required_inputs(paths)

    if args.fresh and os.path.isdir(paths.output_dir):
        shutil.rmtree(paths.output_dir)
    create_output_dirs(paths, args.save_models)

    feature_columns = load_feature_schema(paths.schema)
    dataset = load_dataset(paths.dataset, feature_columns)

    # Build raw component counts once on the entire dataset.
    all_work, _ = prepare_composite_base(
        dataset,
        levels=sorted(set(dataset["data_level"].unique())),
        feature_columns=feature_columns,
    )
    component_count_lookup = all_work.set_index("sample_id")[RAW_COMPONENT_COUNT]
    dataset[RAW_COMPONENT_COUNT] = dataset["sample_id"].map(component_count_lookup)

    l1_work, l1_X = prepare_composite_base(
        dataset,
        levels=["L1"],
        feature_columns=feature_columns,
    )
    l12_work, l12_X = prepare_composite_base(
        dataset,
        levels=["L1", "L2"],
        feature_columns=feature_columns,
    )

    l1_manifest = load_split_manifest(paths.l1_split_manifest, ["L1"])
    l12_manifest = load_split_manifest(paths.l12_split_manifest, ["L1", "L2"])
    l1_tasks = split_tasks_from_manifest(l1_manifest)
    l12_tasks = split_tasks_from_manifest(l12_manifest)
    l1_tasks_threshold1 = extend_tasks_for_threshold1(l1_tasks, l1_work)
    l12_tasks_threshold1 = extend_tasks_for_threshold1(l12_tasks, l12_work)
    cutoffs = resolve_cutoffs(paths, args.cutoffs)

    payload = configuration_payload(
        paths,
        thresholds,
        cutoffs,
        feature_columns,
        args,
    )
    prepare_checkpoint(paths, payload)

    coverage = build_component_coverage(dataset, thresholds)
    atomic_write_csv(coverage, paths.coverage)

    existing_metrics = load_optional_csv(paths.metrics_fold)
    existing_predictions = load_optional_csv(paths.predictions)
    existing_normalizations = load_optional_csv(paths.normalizations)
    existing_audit = load_optional_csv(paths.split_retention_audit)

    completed_validation: Set[str] = set()
    if not existing_metrics.empty:
        ok_rows = existing_metrics[existing_metrics.get("status", "") == "ok"]
        for _, row in ok_rows.iterrows():
            completed_validation.add(
                validation_task_key(
                    int(row["minimum_component_count"]),
                    str(row["model_family"]),
                    str(row["split_type"]),
                    int(row["fold"]),
                )
            )

    metric_rows = existing_metrics.to_dict("records") if not existing_metrics.empty else []
    prediction_frames = [existing_predictions] if not existing_predictions.empty else []
    normalization_rows = (
        existing_normalizations.to_dict("records")
        if not existing_normalizations.empty
        else []
    )
    audit_rows = existing_audit.to_dict("records") if not existing_audit.empty else []

    validation_plan: List[Tuple[int, str, Mapping[str, Any], pd.DataFrame, pd.DataFrame]] = []
    for threshold in thresholds:
        current_l1_tasks = l1_tasks_threshold1 if threshold == 1 else l1_tasks
        current_l12_tasks = l12_tasks_threshold1 if threshold == 1 else l12_tasks
        for task in current_l1_tasks:
            validation_plan.append((threshold, POINT_FAMILY, task, l1_work, l1_X))
        for task in current_l12_tasks:
            validation_plan.append((threshold, QUANTILE_FAMILY, task, l12_work, l12_X))

    print("=" * 108)
    print("[COMPOSITE SENSITIVITY ANALYSIS START]")
    print(f"Project directory      : {paths.project_dir}")
    print(f"Thresholds             : {thresholds}")
    print(f"Historical cutoffs     : {cutoffs}")
    print(f"Feature count          : {len(feature_columns)}")
    print(f"L1 raw rows            : {len(l1_work)}")
    print(f"L1+L2 raw rows         : {len(l12_work)}")
    print(f"Validation tasks       : {len(validation_plan)}")
    print(f"Bootstrap repeats      : {args.bootstrap}")
    print(f"Random ranking repeats : {args.random_repeats}")
    print(f"Save models            : {args.save_models}")
    print(f"Output                  : {paths.output_dir}")
    print("=" * 108)

    for position, (threshold, family, task, work, X) in enumerate(
        validation_plan,
        start=1,
    ):
        key = validation_task_key(
            threshold,
            family,
            str(task["split_type"]),
            int(task["fold"]),
        )
        if key in completed_validation:
            print(
                f"[VALIDATION {position}/{len(validation_plan)}] "
                f"{key} already completed, skip."
            )
            continue

        print(
            f"[VALIDATION {position}/{len(validation_plan)}] "
            f"min_components={threshold} / {family} / "
            f"{task['split_type']} / fold={task['fold']}"
        )

        metric, prediction, normalization, audit = run_validation_task(
            threshold=threshold,
            model_family=family,
            task=task,
            work=work,
            X=X,
            paths=paths,
            save_models=args.save_models,
        )
        metric_rows.append(metric)
        if not prediction.empty:
            prediction_frames.append(prediction)
        if normalization:
            normalization_rows.append(normalization)
        audit_rows.append(audit)

        metrics_frame = pd.DataFrame(metric_rows)
        predictions_frame = (
            pd.concat(prediction_frames, ignore_index=True, sort=False)
            if prediction_frames
            else pd.DataFrame()
        )
        normalizations_frame = pd.DataFrame(normalization_rows)
        audit_frame = pd.DataFrame(audit_rows)

        atomic_write_csv(metrics_frame, paths.metrics_fold)
        atomic_write_csv(predictions_frame, paths.predictions)
        atomic_write_csv(normalizations_frame, paths.normalizations)
        atomic_write_csv(audit_frame, paths.split_retention_audit)

    metrics_fold = pd.read_csv(paths.metrics_fold, low_memory=False)
    predictions = pd.read_csv(paths.predictions, low_memory=False)
    normalizations = pd.read_csv(paths.normalizations, low_memory=False)
    split_audit = pd.read_csv(paths.split_retention_audit, low_memory=False)

    pooled = pooled_metrics_from_predictions(predictions)
    bootstrap_ci = run_cluster_bootstrap(
        predictions,
        repeats=args.bootstrap,
        seed=args.seed,
    )
    atomic_write_csv(pooled, paths.metrics_pooled)
    atomic_write_csv(bootstrap_ci, paths.bootstrap_ci)

    # Formal threshold-2 reproduction audits.
    baseline_audit = build_baseline_reproduction_audit(predictions, paths)
    atomic_write_csv(baseline_audit, paths.baseline_audit)

    # -------------------------------------------------------------------------
    # Historical replay with checkpoint/resume.
    # -------------------------------------------------------------------------
    historical_existing = load_optional_csv(paths.historical_predictions)
    completed_historical: Set[Tuple[int, int]] = set()
    if not historical_existing.empty:
        completed_historical = set(
            (int(threshold), int(cutoff))
            for threshold, cutoff in historical_existing[
                ["minimum_component_count", "cutoff_year"]
            ].drop_duplicates().itertuples(index=False, name=None)
        )

    historical_frames = [historical_existing] if not historical_existing.empty else []
    historical_model_manifest_path = os.path.join(
        paths.manifests_dir,
        "composite_sensitivity_historical_model_manifest.csv",
    )
    historical_models_existing = load_optional_csv(historical_model_manifest_path)
    historical_model_frames = (
        [historical_models_existing]
        if not historical_models_existing.empty
        else []
    )

    historical_plan = [(threshold, cutoff) for threshold in thresholds for cutoff in cutoffs]

    for position, (threshold, cutoff) in enumerate(historical_plan, start=1):
        if (threshold, cutoff) in completed_historical:
            print(
                f"[HISTORICAL {position}/{len(historical_plan)}] "
                f"min_components={threshold}, cutoff={cutoff} already completed, skip."
            )
            continue

        print(
            f"[HISTORICAL {position}/{len(historical_plan)}] "
            f"min_components={threshold}, cutoff={cutoff}"
        )

        try:
            historical_prediction, model_manifest = run_historical_cutoff(
                work=l12_work,
                X=l12_X,
                feature_columns=feature_columns,
                threshold=threshold,
                cutoff=cutoff,
                paths=paths,
                save_models=args.save_models,
            )
        except RuntimeError as error:
            print(f"    [SKIP] {error}")
            continue

        historical_frames.append(historical_prediction)
        historical_model_frames.append(model_manifest)

        combined_historical = pd.concat(
            historical_frames,
            ignore_index=True,
            sort=False,
        )
        combined_models = pd.concat(
            historical_model_frames,
            ignore_index=True,
            sort=False,
        )
        atomic_write_csv(combined_historical, paths.historical_predictions)
        atomic_write_csv(combined_models, historical_model_manifest_path)

        print(
            f"    future samples={len(historical_prediction)}, "
            f"future papers={historical_prediction['paper_id'].nunique()}"
        )

    historical_predictions = load_optional_csv(paths.historical_predictions)
    if historical_predictions.empty:
        raise RuntimeError("没有生成任何历史回放结果。")

    historical_metric_rows: List[Dict[str, Any]] = []
    random_baseline_rows: List[Dict[str, Any]] = []

    for group_index, ((threshold, cutoff), group) in enumerate(
        historical_predictions.groupby(
            ["minimum_component_count", "cutoff_year"],
            sort=True,
        ),
        start=1,
    ):
        threshold = int(threshold)
        cutoff = int(cutoff)
        group = group.reset_index(drop=True)

        random_row = random_historical_baseline(
            frame=group,
            threshold=threshold,
            cutoff=cutoff,
            positive_fraction=args.positive_fraction,
            screen_fraction=args.screen_fraction,
            repeats=args.random_repeats,
            seed=args.seed + group_index * 2029,
        )
        random_baseline_rows.append(random_row)

        for strategy in HISTORICAL_STRATEGIES:
            metric = evaluate_historical_strategy(
                frame=group,
                threshold=threshold,
                cutoff=cutoff,
                strategy=strategy,
                positive_fraction=args.positive_fraction,
                screen_fraction=args.screen_fraction,
            )
            for random_metric in [
                "precision_at_k",
                "recall_at_k",
                "hit_rate_at_k",
                "enrichment_factor_at_k",
                "ndcg_binary_at_k",
                "ndcg_continuous_at_k",
            ]:
                random_mean = random_row.get(f"random_{random_metric}_mean", np.nan)
                metric[f"random_{random_metric}_mean"] = random_mean
                metric[f"{random_metric}_minus_random_mean"] = (
                    metric[random_metric] - random_mean
                    if np.isfinite(random_mean)
                    else np.nan
                )
            historical_metric_rows.append(metric)

    historical_metrics = pd.DataFrame(historical_metric_rows)
    random_baseline = pd.DataFrame(random_baseline_rows)
    historical_summary = summarize_historical_metrics(historical_metrics)

    atomic_write_csv(historical_metrics, paths.historical_metrics)
    atomic_write_csv(random_baseline, paths.random_baseline)
    atomic_write_csv(historical_summary, paths.historical_summary)

    historical_baseline_audit = build_historical_baseline_audit(
        historical_predictions,
        paths,
    )
    atomic_write_csv(historical_baseline_audit, paths.historical_baseline_audit)

    comparison = build_comparison_table(
        coverage,
        pooled,
        historical_summary,
        thresholds,
    )
    atomic_write_csv(comparison, paths.comparison_table)

    # Article/SI tables.
    atomic_write_csv(
        comparison,
        os.path.join(
            paths.tables_dir,
            "table_main_composite_threshold_comparison.csv",
        ),
    )
    atomic_write_csv(
        pooled,
        os.path.join(
            paths.tables_dir,
            "table_si_composite_sensitivity_validation.csv",
        ),
    )
    atomic_write_csv(
        historical_metrics,
        os.path.join(
            paths.tables_dir,
            "table_si_composite_sensitivity_historical_replay.csv",
        ),
    )
    atomic_write_csv(
        coverage,
        os.path.join(
            paths.tables_dir,
            "table_si_composite_component_coverage.csv",
        ),
    )

    # Figures.
    plot_sample_retention(coverage, paths, args.dpi)
    plot_validation_spearman(pooled, paths, args.dpi)
    plot_validation_mae(pooled, paths, args.dpi)
    plot_historical_metric(
        historical_metrics,
        metric="ndcg_continuous_at_k",
        ylabel="Continuous NDCG@Top 10%",
        title="Historical NDCG sensitivity to composite-label completeness",
        filename="fig_composite_sensitivity_historical_ndcg",
        paths=paths,
        dpi=args.dpi,
    )
    plot_historical_metric(
        historical_metrics,
        metric="recall_at_k",
        ylabel="Recall@Top 10%",
        title="Historical recovery sensitivity to composite-label completeness",
        filename="fig_composite_sensitivity_historical_recall",
        paths=paths,
        dpi=args.dpi,
    )
    plot_historical_enrichment_summary(historical_summary, paths, args.dpi)

    # Input and split manifest audit files.
    input_manifest = pd.DataFrame(
        [
            {
                "path": path,
                "exists": os.path.isfile(path),
                "size_bytes": os.path.getsize(path) if os.path.isfile(path) else np.nan,
                "sha256": sha256_file(path) if os.path.isfile(path) else "",
            }
            for path in [
                paths.dataset,
                paths.schema,
                paths.l1_split_manifest,
                paths.l12_split_manifest,
                paths.retrospective_cutoff_manifest,
                paths.l1_baseline_predictions,
                paths.l12_baseline_predictions,
                paths.retrospective_baseline_predictions,
            ]
        ]
    )
    atomic_write_csv(
        input_manifest,
        os.path.join(paths.manifests_dir, "input_file_manifest.csv"),
    )

    frozen_split_summary = pd.DataFrame(
        [
            {
                "model_family": POINT_FAMILY,
                "manifest": paths.l1_split_manifest,
                "rows": int(len(l1_manifest)),
                "split_types": ",".join(sorted(l1_manifest["split_type"].unique())),
                "folds": int(l1_manifest[["split_type", "fold"]].drop_duplicates().shape[0]),
            },
            {
                "model_family": QUANTILE_FAMILY,
                "manifest": paths.l12_split_manifest,
                "rows": int(len(l12_manifest)),
                "split_types": ",".join(sorted(l12_manifest["split_type"].unique())),
                "folds": int(l12_manifest[["split_type", "fold"]].drop_duplicates().shape[0]),
            },
        ]
    )
    atomic_write_csv(
        frozen_split_summary,
        os.path.join(paths.manifests_dir, "frozen_split_manifest_summary.csv"),
    )

    run_summary = {
        "analysis": "CNT 3.0 composite-label completeness sensitivity analysis",
        "project_dir": paths.project_dir,
        "output_dir": paths.output_dir,
        "thresholds": thresholds,
        "historical_cutoffs": cutoffs,
        "feature_count": int(len(feature_columns)),
        "weights": WEIGHTS,
        "missing_policy": MISSING_POLICY,
        "validation": {
            "split_source": {
                POINT_FAMILY: paths.l1_split_manifest,
                QUANTILE_FAMILY: paths.l12_split_manifest,
            },
            "rule": (
                "Formal-manifest sample roles are preserved. Threshold-1-only samples "
                "are added deterministically. Composite normalization is fitted only "
                "on the threshold-specific outer training rows."
            ),
            "bootstrap_cluster": "paper_id",
            "bootstrap_repeats": int(args.bootstrap),
        },
        "historical_replay": {
            "cutoffs": cutoffs,
            "positive_fraction": float(args.positive_fraction),
            "screen_fraction": float(args.screen_fraction),
            "random_repeats": int(args.random_repeats),
            "strategies": HISTORICAL_STRATEGY_LABELS,
        },
        "scientific_decision_rule": (
            "Review sample retention, paper-grouped and temporal q50 performance, "
            "and q50/q90 historical enrichment jointly. Do not choose the threshold "
            "from a single metric."
        ),
        "outputs": {
            "coverage": paths.coverage,
            "metrics_fold": paths.metrics_fold,
            "metrics_pooled": paths.metrics_pooled,
            "bootstrap_ci": paths.bootstrap_ci,
            "predictions": paths.predictions,
            "normalizations": paths.normalizations,
            "split_retention_audit": paths.split_retention_audit,
            "historical_predictions": paths.historical_predictions,
            "historical_metrics": paths.historical_metrics,
            "historical_summary": paths.historical_summary,
            "comparison_table": paths.comparison_table,
            "threshold2_reproduction_audit": paths.baseline_audit,
            "historical_threshold1_reproduction_audit": paths.historical_baseline_audit,
            "figures_dir": paths.figures_dir,
            "source_data_dir": paths.source_data_dir,
            "tables_dir": paths.tables_dir,
            "manifests_dir": paths.manifests_dir,
        },
        "next_step_boundary": (
            "This analysis selects the composite-label threshold. It does not yet "
            "export deployment models or modify L123."
        ),
    }
    atomic_write_json(run_summary, paths.run_summary)

    print("\n" + "=" * 108)
    print("[COMPOSITE SENSITIVITY ANALYSIS FINISHED]")
    print(f"Coverage                 : {paths.coverage}")
    print(f"Fold metrics             : {paths.metrics_fold}")
    print(f"Pooled metrics           : {paths.metrics_pooled}")
    print(f"Bootstrap CI             : {paths.bootstrap_ci}")
    print(f"Predictions              : {paths.predictions}")
    print(f"Historical predictions   : {paths.historical_predictions}")
    print(f"Historical metrics       : {paths.historical_metrics}")
    print(f"Historical summary       : {paths.historical_summary}")
    print(f"Comparison table         : {paths.comparison_table}")
    print(f"Figures                  : {paths.figures_dir}")
    print(f"Source data              : {paths.source_data_dir}")
    print(f"Tables                   : {paths.tables_dir}")
    print(f"Run summary              : {paths.run_summary}")
    print("=" * 108)


if __name__ == "__main__":
    main()
