# -*- coding: utf-8 -*-
"""
uncertainty.py

CNT 3.0 — Article-ready uncertainty calibration and risk-validity analysis

This script does not retrain the L12 quantile models. It reads the existing
q10/q50/q90 sample-level predictions and generates article-ready analyses:

1. Calibration of q10/q50/q90 under random, paper-grouped and temporal splits.
2. PICP, MPIW and Winkler score for the ordered q10-q90 interval.
3. Whether interval width identifies genuinely difficult samples.
4. L1 versus L2 uncertainty behaviour.
5. Growth-mode- and component-count-resolved calibration.
6. Quantile-crossing audit.
7. Paper-cluster bootstrap 95% confidence intervals.
8. PNG/PDF figures, source-data CSV files, tables and audit manifests.

Definitions
-----------
- Quantile calibration uses raw q10/q50/q90 predictions.
- q50 point error uses raw q50.
- Interval coverage/width/Winkler use per-sample ordered q10-q90.
- Quantile crossing is always reported from the raw predictions.
- uncertainty = q90_ordered - q10_ordered.
- actual_error = |y_true - q50_raw|.
- Width quartiles are assigned within each split/fold to reduce fold-scale bias.
- Bootstrap resamples complete paper_id groups, not individual synthesis rows.

Default input directory
-----------------------
outputs/models/l12_quantile/
    L12_quantile_predictions.csv
    L12_quantile_metrics_fold.csv
    L12_split_manifest.csv

Default output directory
------------------------
outputs/external_validation/uncertainty/

Run
---
python uncertainty.py
python uncertainty.py --bootstrap 200
python uncertainty.py --bootstrap 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. Configuration
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

LEVEL_ORDER = ["All", "L1", "L2"]

QUANTILES = {
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
}

NOMINAL_INTERVAL_COVERAGE = 0.80
INTERVAL_MIS_COVERAGE = 0.20

DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_RANDOM_SEED = 20260722
DEFAULT_DPI = 600

MIN_BOOTSTRAP_SAMPLES = 20
MIN_BOOTSTRAP_PAPERS = 5
MIN_GROUP_SAMPLES = 10
MIN_GROUP_PAPERS = 3

SEVERE_ERROR_QUANTILE = 0.80
DERIVED_COLUMN_TOLERANCE = 1e-10
METRIC_COMPARISON_TOLERANCE = 1e-8

QUARTILE_LABELS = [
    "Q1 lowest uncertainty",
    "Q2",
    "Q3",
    "Q4 highest uncertainty",
]

BOOTSTRAP_METRICS = [
    "empirical_coverage_q10",
    "empirical_coverage_q50",
    "empirical_coverage_q90",
    "q50_mae",
    "q50_r2",
    "q50_spearman",
    "picp_80_ordered",
    "mpiw_80_ordered",
    "winkler_80_ordered",
    "uncertainty_error_spearman",
    "any_crossing_rate",
]


# =============================================================================
# 2. Paths
# =============================================================================

@dataclass(frozen=True)
class Paths:
    project_dir: str
    l12_dir: str
    predictions: str
    metrics_fold_existing: str
    split_manifest: str
    output_dir: str
    figures_dir: str
    source_data_dir: str
    tables_dir: str
    manifests_dir: str


def build_paths(project_dir: str, output_dir: Optional[str]) -> Paths:
    project_dir = os.path.abspath(project_dir)
    l12_dir = os.path.join(
        project_dir,
        "outputs", "models", "l12_quantile",
    )
    resolved_output = (
        os.path.abspath(output_dir)
        if output_dir
        else os.path.join(project_dir, "outputs", "external_validation", "uncertainty")
    )

    return Paths(
        project_dir=project_dir,
        l12_dir=l12_dir,
        predictions=os.path.join(l12_dir, "L12_quantile_predictions.csv"),
        metrics_fold_existing=os.path.join(
            l12_dir,
            "L12_quantile_metrics_fold.csv",
        ),
        split_manifest=os.path.join(l12_dir, "L12_split_manifest.csv"),
        output_dir=resolved_output,
        figures_dir=os.path.join(resolved_output, "figures"),
        source_data_dir=os.path.join(resolved_output, "source_data"),
        tables_dir=os.path.join(resolved_output, "tables"),
        manifests_dir=os.path.join(resolved_output, "manifests"),
    )


def ensure_inputs_exist(paths: Paths) -> None:
    required = [
        paths.predictions,
        paths.metrics_fold_existing,
        paths.split_manifest,
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(missing)
        )


def create_output_dirs(paths: Paths) -> None:
    for directory in [
        paths.output_dir,
        paths.figures_dir,
        paths.source_data_dir,
        paths.tables_dir,
        paths.manifests_dir,
    ]:
        os.makedirs(directory, exist_ok=True)


# =============================================================================
# 3. General utilities
# =============================================================================

def atomic_write_csv(frame: pd.DataFrame, path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".csv",
        dir=directory,
    )
    os.close(fd)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def atomic_write_json(data: Any, path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".json",
        dir=directory,
    )
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(parse_bool).astype(bool)


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if len(x_array) < 2:
        return float("nan")
    if np.std(x_array) == 0 or np.std(y_array) == 0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if len(x_array) < 2:
        return float("nan")
    x_rank = pd.Series(x_array).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y_array).rank(method="average").to_numpy(dtype=float)
    return safe_pearson(x_rank, y_rank)


def safe_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    y_true_array = y_true_array[finite]
    y_pred_array = y_pred_array[finite]
    if len(y_true_array) < 2 or np.allclose(y_true_array, y_true_array[0]):
        return float("nan")
    residual = float(np.sum((y_true_array - y_pred_array) ** 2))
    total = float(np.sum((y_true_array - np.mean(y_true_array)) ** 2))
    if math.isclose(total, 0.0):
        return float("nan")
    return float(1.0 - residual / total)


def pinball_loss(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    alpha: float,
) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    loss = np.maximum(alpha * error, (alpha - 1.0) * error)
    return float(np.mean(loss))


def winkler_score_array(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    alpha: float = INTERVAL_MIS_COVERAGE,
) -> np.ndarray:
    y_true_array = np.asarray(y_true, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    width = upper_array - lower_array
    score = width.copy()
    below = y_true_array < lower_array
    above = y_true_array > upper_array
    score[below] += 2.0 / alpha * (lower_array[below] - y_true_array[below])
    score[above] += 2.0 / alpha * (y_true_array[above] - upper_array[above])
    return score


def percentile_interval(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> Tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, 0
    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
        int(len(array)),
    )


def normalize_growth_mode(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    raw = str(value).strip()
    if not raw:
        return "Unknown"
    upper = raw.upper()
    for pattern in [
        r"^([A-E])$",
        r"^MODE[\s_\-:]*([A-E])$",
        r"^([A-E])[\s_\-:]",
        r"[\s_\-:]([A-E])$",
    ]:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    standalone = re.search(r"\b([A-E])\b", upper)
    if standalone:
        return standalone.group(1)
    return raw


# =============================================================================
# 4. Load and enrich L12 predictions
# =============================================================================

def load_predictions(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "data_level",
        "growth_mode",
        "target",
        "split_type",
        "fold",
        "y_true",
        "q10_raw",
        "q50_raw",
        "q90_raw",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            "L12_quantile_predictions.csv is missing columns: " + str(missing)
        )

    optional_defaults: Dict[str, Any] = {
        "source_file": "",
        "synthesis_id": "",
        "temporal_cutoff_year": np.nan,
        "component_count": np.nan,
        "weight_sum_used": np.nan,
        "q10_ordered": np.nan,
        "q50_ordered": np.nan,
        "q90_ordered": np.nan,
        "covered_by_ordered_q10_q90": False,
        "interval_width_ordered": np.nan,
        "winkler_score_ordered": np.nan,
        "crossing_q10_q50": False,
        "crossing_q50_q90": False,
        "any_crossing": False,
    }
    for column, default in optional_defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].fillna("").astype(str)
    frame["paper_id"] = frame["paper_id"].fillna("").astype(str)
    frame["split_type"] = frame["split_type"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["paper_year_int"] = pd.to_numeric(
        frame["paper_year_int"],
        errors="coerce",
    )
    frame["temporal_cutoff_year"] = pd.to_numeric(
        frame["temporal_cutoff_year"],
        errors="coerce",
    )
    frame["data_level"] = (
        frame["data_level"].fillna("").astype(str).str.strip().str.upper()
    )
    frame["growth_mode_raw"] = frame["growth_mode"].fillna("").astype(str)
    frame["growth_mode_normalized"] = frame["growth_mode"].map(
        normalize_growth_mode
    )

    numeric_columns = [
        "y_true",
        "q10_raw",
        "q50_raw",
        "q90_raw",
        "q10_ordered",
        "q50_ordered",
        "q90_ordered",
        "interval_width_ordered",
        "winkler_score_ordered",
        "component_count",
        "weight_sum_used",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    finite = (
        np.isfinite(frame["y_true"])
        & np.isfinite(frame["q10_raw"])
        & np.isfinite(frame["q50_raw"])
        & np.isfinite(frame["q90_raw"])
    )
    if not finite.all():
        print(f"[WARN] Dropping {int((~finite).sum())} non-finite prediction rows.")
        frame = frame.loc[finite].copy()

    if (frame["sample_id"].str.len() == 0).any():
        raise RuntimeError("Empty sample_id exists in prediction file.")
    if (frame["paper_id"].str.len() == 0).any():
        raise RuntimeError("Empty paper_id exists in prediction file.")

    unknown = sorted(set(frame["split_type"]).difference(SPLIT_ORDER))
    if unknown:
        print(f"[WARN] Ignoring unknown split types: {unknown}")
    frame = frame[frame["split_type"].isin(SPLIT_ORDER)].copy()
    if frame.empty:
        raise RuntimeError("No usable predictions were found.")
    return frame.reset_index(drop=True)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    raw = out[["q10_raw", "q50_raw", "q90_raw"]].to_numpy(dtype=float)
    ordered = np.sort(raw, axis=1)

    out["q10_ordered_recomputed"] = ordered[:, 0]
    out["q50_ordered_recomputed"] = ordered[:, 1]
    out["q90_ordered_recomputed"] = ordered[:, 2]

    out["crossing_q10_q50_recomputed"] = out["q10_raw"] > out["q50_raw"]
    out["crossing_q50_q90_recomputed"] = out["q50_raw"] > out["q90_raw"]
    out["any_crossing_recomputed"] = (
        out["crossing_q10_q50_recomputed"]
        | out["crossing_q50_q90_recomputed"]
    )

    out["covered80_recomputed"] = (
        (out["y_true"] >= out["q10_ordered_recomputed"])
        & (out["y_true"] <= out["q90_ordered_recomputed"])
    )
    out["uncertainty_width"] = (
        out["q90_ordered_recomputed"] - out["q10_ordered_recomputed"]
    )
    out["absolute_error_q50_raw"] = np.abs(out["y_true"] - out["q50_raw"])
    out["signed_error_q50_raw"] = out["q50_raw"] - out["y_true"]
    out["absolute_error_q50_ordered"] = np.abs(
        out["y_true"] - out["q50_ordered_recomputed"]
    )
    out["winkler80_recomputed"] = winkler_score_array(
        out["y_true"],
        out["q10_ordered_recomputed"],
        out["q90_ordered_recomputed"],
    )
    out["y_le_q10_raw"] = out["y_true"] <= out["q10_raw"]
    out["y_le_q50_raw"] = out["y_true"] <= out["q50_raw"]
    out["y_le_q90_raw"] = out["y_true"] <= out["q90_raw"]
    return out


def load_manifest(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "data_level",
        "target",
        "split_type",
        "fold",
        "role",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError("L12 split manifest is missing columns: " + str(missing))

    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].fillna("").astype(str)
    frame["paper_id"] = frame["paper_id"].fillna("").astype(str)
    frame["split_type"] = frame["split_type"].astype(str)
    frame["role"] = frame["role"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["paper_year_int"] = pd.to_numeric(
        frame["paper_year_int"],
        errors="coerce",
    )
    return frame[frame["split_type"].isin(SPLIT_ORDER)].copy()


# =============================================================================
# 5. Audits
# =============================================================================

def audit_predictions(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {
        "status": "ok",
        "checks": [],
        "warnings": [],
    }

    duplicate = predictions.duplicated(
        subset=["split_type", "fold", "sample_id"],
        keep=False,
    )
    if duplicate.any():
        examples = predictions.loc[
            duplicate,
            ["split_type", "fold", "sample_id"],
        ].head(10)
        raise RuntimeError(
            "Duplicate split/fold/sample_id rows:\n"
            + examples.to_string(index=False)
        )

    audit["checks"].append({
        "name": "prediction_key_uniqueness",
        "status": "passed",
        "rows": int(len(predictions)),
    })

    grouped = predictions[predictions["split_type"] == "paper_grouped_cv"]
    grouped_duplicate = grouped.duplicated(subset=["sample_id"], keep=False)
    if grouped_duplicate.any():
        raise RuntimeError("A paper-grouped OOF sample appears in more than one fold.")

    split_checks: List[Dict[str, Any]] = []
    for (split_type, fold), split_frame in manifest.groupby(
        ["split_type", "fold"],
        sort=False,
    ):
        train = split_frame[split_frame["role"] == "train"]
        test = split_frame[split_frame["role"] == "test"]
        train_papers = set(train["paper_id"].astype(str))
        test_papers = set(test["paper_id"].astype(str))
        overlap = train_papers.intersection(test_papers)

        if split_type == "paper_grouped_cv" and overlap:
            raise RuntimeError(
                f"{split_type}/fold={fold} has {len(overlap)} overlapping papers."
            )

        if split_type == "temporal_holdout" and overlap:
            audit["warnings"].append({
                "name": "temporal_paper_overlap",
                "fold": int(fold),
                "overlapping_papers": int(len(overlap)),
                "note": (
                    "The frozen L12 temporal protocol is year-based at the sample level; "
                    "paper overlap is therefore audited and reported but is not treated "
                    "as a fatal inconsistency here."
                ),
            })

        manifest_test_ids = set(test["sample_id"].astype(str))
        prediction_test_ids = set(
            predictions.loc[
                (predictions["split_type"] == split_type)
                & (predictions["fold"] == fold),
                "sample_id",
            ].astype(str)
        )
        if manifest_test_ids != prediction_test_ids:
            raise RuntimeError(
                f"{split_type}/fold={fold} manifest-test and prediction IDs differ."
            )

        chronology = "not_applicable"
        train_year_max = None
        test_year_min = None
        if split_type == "temporal_holdout":
            train_years = train["paper_year_int"].dropna()
            test_years = test["paper_year_int"].dropna()
            if not train_years.empty and not test_years.empty:
                train_year_max = int(train_years.max())
                test_year_min = int(test_years.min())
                if train_year_max >= test_year_min:
                    raise RuntimeError(
                        "Temporal chronology failed: "
                        f"train_max={train_year_max}, test_min={test_year_min}"
                    )
                chronology = "passed"

        split_checks.append({
            "split_type": split_type,
            "fold": int(fold),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_train_papers": int(len(train_papers)),
            "n_test_papers": int(len(test_papers)),
            "paper_overlap": int(len(overlap)),
            "sample_match": True,
            "chronology": chronology,
            "train_year_max": train_year_max,
            "test_year_min": test_year_min,
        })

    audit["checks"].append({
        "name": "split_manifest_consistency",
        "status": "passed",
        "details": split_checks,
    })

    numeric_checks: Dict[str, Any] = {}
    for stored, recomputed in [
        ("q10_ordered", "q10_ordered_recomputed"),
        ("q50_ordered", "q50_ordered_recomputed"),
        ("q90_ordered", "q90_ordered_recomputed"),
        ("interval_width_ordered", "uncertainty_width"),
        ("winkler_score_ordered", "winkler80_recomputed"),
    ]:
        stored_values = pd.to_numeric(predictions[stored], errors="coerce").to_numpy(float)
        recomputed_values = predictions[recomputed].to_numpy(float)
        comparable = np.isfinite(stored_values) & np.isfinite(recomputed_values)
        max_difference = (
            float(np.max(np.abs(stored_values[comparable] - recomputed_values[comparable])))
            if comparable.any()
            else np.nan
        )
        numeric_checks[stored] = {
            "comparable_rows": int(comparable.sum()),
            "max_absolute_difference": max_difference,
            "within_tolerance": (
                bool(max_difference <= DERIVED_COLUMN_TOLERANCE)
                if np.isfinite(max_difference)
                else None
            ),
        }

    boolean_checks: Dict[str, Any] = {}
    for stored, recomputed in [
        ("covered_by_ordered_q10_q90", "covered80_recomputed"),
        ("crossing_q10_q50", "crossing_q10_q50_recomputed"),
        ("crossing_q50_q90", "crossing_q50_q90_recomputed"),
        ("any_crossing", "any_crossing_recomputed"),
    ]:
        mismatch = parse_bool_series(predictions[stored]) != predictions[recomputed].astype(bool)
        boolean_checks[stored] = {
            "rows": int(len(predictions)),
            "mismatch_rows": int(mismatch.sum()),
            "exact_match": bool(not mismatch.any()),
        }

    audit["checks"].append({
        "name": "stored_derived_columns_consistency",
        "status": "passed",
        "numeric": numeric_checks,
        "boolean": boolean_checks,
    })
    return audit


# =============================================================================
# 6. Core uncertainty metrics
# =============================================================================

def evaluate_uncertainty(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"n_samples": 0, "n_papers": 0}

    y_true = frame["y_true"].to_numpy(dtype=float)
    q10 = frame["q10_raw"].to_numpy(dtype=float)
    q50 = frame["q50_raw"].to_numpy(dtype=float)
    q90 = frame["q90_raw"].to_numpy(dtype=float)
    lower = frame["q10_ordered_recomputed"].to_numpy(dtype=float)
    upper = frame["q90_ordered_recomputed"].to_numpy(dtype=float)
    width = frame["uncertainty_width"].to_numpy(dtype=float)
    absolute_error = frame["absolute_error_q50_raw"].to_numpy(dtype=float)

    coverage_q10 = float(np.mean(y_true <= q10))
    coverage_q50 = float(np.mean(y_true <= q50))
    coverage_q90 = float(np.mean(y_true <= q90))
    covered80 = (y_true >= lower) & (y_true <= upper)
    picp80 = float(np.mean(covered80))
    mpiw80 = float(np.mean(width))
    y_range = float(np.max(y_true) - np.min(y_true))
    winkler = winkler_score_array(y_true, lower, upper)

    return {
        "n_samples": int(len(frame)),
        "n_papers": int(frame["paper_id"].nunique()),
        "pinball_q10": pinball_loss(y_true, q10, 0.10),
        "pinball_q50": pinball_loss(y_true, q50, 0.50),
        "pinball_q90": pinball_loss(y_true, q90, 0.90),
        "empirical_coverage_q10": coverage_q10,
        "empirical_coverage_q50": coverage_q50,
        "empirical_coverage_q90": coverage_q90,
        "signed_calibration_error_q10": coverage_q10 - 0.10,
        "signed_calibration_error_q50": coverage_q50 - 0.50,
        "signed_calibration_error_q90": coverage_q90 - 0.90,
        "absolute_calibration_error_q10": abs(coverage_q10 - 0.10),
        "absolute_calibration_error_q50": abs(coverage_q50 - 0.50),
        "absolute_calibration_error_q90": abs(coverage_q90 - 0.90),
        "mean_absolute_quantile_calibration_error": float(
            np.mean([
                abs(coverage_q10 - 0.10),
                abs(coverage_q50 - 0.50),
                abs(coverage_q90 - 0.90),
            ])
        ),
        "q50_mae": float(np.mean(absolute_error)),
        "q50_rmse": float(math.sqrt(np.mean((y_true - q50) ** 2))),
        "q50_r2": safe_r2(y_true, q50),
        "q50_spearman": safe_spearman(y_true, q50),
        "q50_bias": float(np.mean(q50 - y_true)),
        "picp_80_ordered": picp80,
        "picp_80_minus_nominal": picp80 - NOMINAL_INTERVAL_COVERAGE,
        "picp_80_absolute_calibration_error": abs(
            picp80 - NOMINAL_INTERVAL_COVERAGE
        ),
        "mpiw_80_ordered": mpiw80,
        "median_interval_width_ordered": float(np.median(width)),
        "nmpiw_by_observed_y_range": (
            mpiw80 / y_range if y_range > 0 else np.nan
        ),
        "winkler_80_ordered": float(np.mean(winkler)),
        "uncertainty_error_pearson": safe_pearson(width, absolute_error),
        "uncertainty_error_spearman": safe_spearman(width, absolute_error),
        "crossing_q10_q50_rate": float(
            frame["crossing_q10_q50_recomputed"].mean()
        ),
        "crossing_q50_q90_rate": float(
            frame["crossing_q50_q90_recomputed"].mean()
        ),
        "any_crossing_rate": float(frame["any_crossing_recomputed"].mean()),
    }


def evaluate_groups(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(
        list(group_columns),
        dropna=False,
        sort=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            column: value
            for column, value in zip(group_columns, keys)
        }
        row.update(evaluate_uncertainty(group))
        row["interpretation_status"] = (
            "usable"
            if (
                row["n_samples"] >= MIN_GROUP_SAMPLES
                and row["n_papers"] >= MIN_GROUP_PAPERS
            )
            else "descriptive_only_small_group"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = evaluate_groups(frame, ["split_type", "fold"])
    result["split_label"] = result["split_type"].map(SPLIT_LABELS)
    return result


def build_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    result = evaluate_groups(frame, ["split_type"])
    result["split_label"] = result["split_type"].map(SPLIT_LABELS)
    rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    result["_rank"] = result["split_type"].map(rank)
    return result.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)


def build_by_level(frame: pd.DataFrame) -> pd.DataFrame:
    all_rows = evaluate_groups(
        frame.assign(data_level_analysis="All"),
        ["split_type", "data_level_analysis"],
    )
    level_frame = frame.rename(columns={"data_level": "data_level_analysis"})
    level_rows = evaluate_groups(
        level_frame,
        ["split_type", "data_level_analysis"],
    )
    result = pd.concat([all_rows, level_rows], ignore_index=True, sort=False)
    result["split_label"] = result["split_type"].map(SPLIT_LABELS)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    level_rank = {name: index for index, name in enumerate(LEVEL_ORDER)}
    result["_split_rank"] = result["split_type"].map(split_rank)
    result["_level_rank"] = result["data_level_analysis"].map(
        lambda value: level_rank.get(value, len(level_rank))
    )
    return (
        result.sort_values(["_split_rank", "_level_rank"])
        .drop(columns=["_split_rank", "_level_rank"])
        .reset_index(drop=True)
    )


def build_level_comparison(by_level: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_type in SPLIT_ORDER:
        subset = by_level[by_level["split_type"] == split_type]
        l1 = subset[subset["data_level_analysis"] == "L1"]
        l2 = subset[subset["data_level_analysis"] == "L2"]
        if l1.empty or l2.empty:
            continue
        a = l1.iloc[0]
        b = l2.iloc[0]
        rows.append({
            "split_type": split_type,
            "split_label": SPLIT_LABELS[split_type],
            "L1_n_samples": int(a["n_samples"]),
            "L2_n_samples": int(b["n_samples"]),
            "L1_q50_mae": a["q50_mae"],
            "L2_q50_mae": b["q50_mae"],
            "L2_minus_L1_q50_mae": b["q50_mae"] - a["q50_mae"],
            "L1_picp80": a["picp_80_ordered"],
            "L2_picp80": b["picp_80_ordered"],
            "L2_minus_L1_picp80": (
                b["picp_80_ordered"] - a["picp_80_ordered"]
            ),
            "L1_mpiw80": a["mpiw_80_ordered"],
            "L2_mpiw80": b["mpiw_80_ordered"],
            "L2_minus_L1_mpiw80": (
                b["mpiw_80_ordered"] - a["mpiw_80_ordered"]
            ),
            "L1_uncertainty_error_spearman": a[
                "uncertainty_error_spearman"
            ],
            "L2_uncertainty_error_spearman": b[
                "uncertainty_error_spearman"
            ],
        })
    return pd.DataFrame(rows)


def build_by_growth_mode(frame: pd.DataFrame) -> pd.DataFrame:
    result = evaluate_groups(
        frame,
        ["split_type", "growth_mode_normalized"],
    ).rename(columns={"growth_mode_normalized": "growth_mode"})
    result["split_label"] = result["split_type"].map(SPLIT_LABELS)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    mode_rank = {
        name: index
        for index, name in enumerate(["A", "B", "C", "D", "E", "Unknown"])
    }
    result["_split_rank"] = result["split_type"].map(split_rank)
    result["_mode_rank"] = result["growth_mode"].map(
        lambda value: mode_rank.get(value, len(mode_rank))
    )
    return (
        result.sort_values(["_split_rank", "_mode_rank"])
        .drop(columns=["_split_rank", "_mode_rank"])
        .reset_index(drop=True)
    )


def build_by_component_count(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame[frame["component_count"].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    usable["component_count_int"] = usable["component_count"].round().astype(int)
    result = evaluate_groups(
        usable,
        ["split_type", "component_count_int"],
    )
    result["split_label"] = result["split_type"].map(SPLIT_LABELS)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    result["_split_rank"] = result["split_type"].map(split_rank)
    return (
        result.sort_values(["_split_rank", "component_count_int"])
        .drop(columns=["_split_rank"])
        .reset_index(drop=True)
    )


def build_crossing_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    overall = evaluate_groups(
        frame.assign(analysis_group="All"),
        ["split_type", "analysis_group"],
    )
    by_level = evaluate_groups(
        frame.rename(columns={"data_level": "analysis_group"}),
        ["split_type", "analysis_group"],
    )
    result = pd.concat([overall, by_level], ignore_index=True, sort=False)
    result["split_label"] = result["split_type"].map(SPLIT_LABELS)
    return result[[
        "split_type",
        "split_label",
        "analysis_group",
        "n_samples",
        "n_papers",
        "crossing_q10_q50_rate",
        "crossing_q50_q90_rate",
        "any_crossing_rate",
        "interpretation_status",
    ]]


# =============================================================================
# 7. Width quartiles and risk validity
# =============================================================================

def assign_uncertainty_quartiles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["uncertainty_percentile_within_fold"] = np.nan
    out["uncertainty_quartile"] = pd.Series(index=out.index, dtype="object")
    out["fold_severe_error_threshold"] = np.nan
    out["severe_error_within_fold"] = False

    for (_, _), indices in out.groupby(["split_type", "fold"]).groups.items():
        index = pd.Index(indices)
        width = out.loc[index, "uncertainty_width"]
        error = out.loc[index, "absolute_error_q50_raw"]
        percentile = width.rank(method="average", pct=True).clip(
            lower=np.finfo(float).eps,
            upper=1.0,
        )
        quartile = pd.cut(
            percentile,
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=QUARTILE_LABELS,
            include_lowest=True,
        )
        threshold = float(error.quantile(SEVERE_ERROR_QUANTILE))
        out.loc[index, "uncertainty_percentile_within_fold"] = percentile.to_numpy()
        out.loc[index, "uncertainty_quartile"] = quartile.astype(str).to_numpy()
        out.loc[index, "fold_severe_error_threshold"] = threshold
        out.loc[index, "severe_error_within_fold"] = (
            error >= threshold
        ).to_numpy(dtype=bool)
    return out


def build_quartile_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (split_type, quartile), group in frame.groupby(
        ["split_type", "uncertainty_quartile"],
        sort=False,
        dropna=False,
    ):
        row = {
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(split_type, split_type),
            "uncertainty_quartile": quartile,
        }
        row.update(evaluate_uncertainty(group))
        row.update({
            "mean_uncertainty_width": float(group["uncertainty_width"].mean()),
            "median_uncertainty_width": float(
                group["uncertainty_width"].median()
            ),
            "mean_absolute_error": float(
                group["absolute_error_q50_raw"].mean()
            ),
            "median_absolute_error": float(
                group["absolute_error_q50_raw"].median()
            ),
            "severe_error_rate": float(
                group["severe_error_within_fold"].mean()
            ),
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    quartile_rank = {
        name: index for index, name in enumerate(QUARTILE_LABELS)
    }
    result["_split_rank"] = result["split_type"].map(split_rank)
    result["_quartile_rank"] = result["uncertainty_quartile"].map(
        quartile_rank
    )
    return (
        result.sort_values(["_split_rank", "_quartile_rank"])
        .drop(columns=["_split_rank", "_quartile_rank"])
        .reset_index(drop=True)
    )


def build_quartile_summary(quartiles: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_type in SPLIT_ORDER:
        subset = quartiles[quartiles["split_type"] == split_type].copy()
        q1 = subset[
            subset["uncertainty_quartile"] == QUARTILE_LABELS[0]
        ]
        q4 = subset[
            subset["uncertainty_quartile"] == QUARTILE_LABELS[-1]
        ]
        if q1.empty or q4.empty:
            continue
        a = q1.iloc[0]
        b = q4.iloc[0]
        q1_mae = float(a["mean_absolute_error"])
        q4_mae = float(b["mean_absolute_error"])
        rank = {name: index for index, name in enumerate(QUARTILE_LABELS)}
        subset["_rank"] = subset["uncertainty_quartile"].map(rank)
        subset = subset.sort_values("_rank")
        mae_values = subset["mean_absolute_error"].to_numpy(dtype=float)
        adjacent_increases = int(np.sum(np.diff(mae_values) >= 0))
        rows.append({
            "split_type": split_type,
            "split_label": SPLIT_LABELS[split_type],
            "Q1_mean_absolute_error": q1_mae,
            "Q4_mean_absolute_error": q4_mae,
            "Q4_to_Q1_mae_ratio": (
                q4_mae / q1_mae if q1_mae > 0 else np.nan
            ),
            "Q1_severe_error_rate": a["severe_error_rate"],
            "Q4_severe_error_rate": b["severe_error_rate"],
            "Q4_minus_Q1_severe_error_rate": (
                b["severe_error_rate"] - a["severe_error_rate"]
            ),
            "adjacent_mae_increases_out_of_3": adjacent_increases,
            "mae_monotonic_non_decreasing": bool(
                adjacent_increases == max(0, len(mae_values) - 1)
            ),
        })
    return pd.DataFrame(rows)


def build_error_relation(
    frame: pd.DataFrame,
    quartile_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_type, group in frame.groupby("split_type", sort=False):
        row: Dict[str, Any] = {
            "split_type": split_type,
            "split_label": SPLIT_LABELS.get(split_type, split_type),
            "n_samples": int(len(group)),
            "n_papers": int(group["paper_id"].nunique()),
            "pearson_width_vs_absolute_error": safe_pearson(
                group["uncertainty_width"],
                group["absolute_error_q50_raw"],
            ),
            "spearman_width_vs_absolute_error": safe_spearman(
                group["uncertainty_width"],
                group["absolute_error_q50_raw"],
            ),
            "pearson_width_vs_squared_error": safe_pearson(
                group["uncertainty_width"],
                group["signed_error_q50_raw"] ** 2,
            ),
            "spearman_width_vs_squared_error": safe_spearman(
                group["uncertainty_width"],
                group["signed_error_q50_raw"] ** 2,
            ),
        }
        summary = quartile_summary[
            quartile_summary["split_type"] == split_type
        ]
        if not summary.empty:
            for column in [
                "Q4_to_Q1_mae_ratio",
                "Q4_minus_Q1_severe_error_rate",
                "adjacent_mae_increases_out_of_3",
                "mae_monotonic_non_decreasing",
            ]:
                row[column] = summary.iloc[0][column]
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 8. Paper-cluster bootstrap
# =============================================================================

def cluster_bootstrap(
    frame: pd.DataFrame,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "bootstrap_requested": int(repeats),
        "n_samples": int(len(frame)),
        "n_papers": int(frame["paper_id"].nunique()),
        "bootstrap_status": "ok",
    }

    if repeats <= 0:
        result["bootstrap_status"] = "disabled"
    elif len(frame) < MIN_BOOTSTRAP_SAMPLES:
        result["bootstrap_status"] = "too_few_samples"
    elif frame["paper_id"].nunique() < MIN_BOOTSTRAP_PAPERS:
        result["bootstrap_status"] = "too_few_papers"

    if result["bootstrap_status"] != "ok":
        for metric in BOOTSTRAP_METRICS:
            result[f"{metric}_ci_lower"] = np.nan
            result[f"{metric}_ci_upper"] = np.nan
            result[f"{metric}_valid_bootstrap"] = 0
        return result

    papers = list(frame["paper_id"].astype(str).unique())
    paper_frames = {
        paper: frame[frame["paper_id"].astype(str) == paper].copy()
        for paper in papers
    }
    rng = np.random.default_rng(seed)
    distributions: Dict[str, List[float]] = {
        metric: [] for metric in BOOTSTRAP_METRICS
    }

    for _ in range(repeats):
        sampled_papers = rng.choice(papers, size=len(papers), replace=True)
        sampled = pd.concat(
            [paper_frames[str(paper)] for paper in sampled_papers],
            ignore_index=True,
        )
        metrics = evaluate_uncertainty(sampled)
        for metric in BOOTSTRAP_METRICS:
            value = metrics.get(metric, np.nan)
            if np.isfinite(value):
                distributions[metric].append(float(value))

    for metric in BOOTSTRAP_METRICS:
        lower, upper, valid = percentile_interval(distributions[metric])
        result[f"{metric}_ci_lower"] = lower
        result[f"{metric}_ci_upper"] = upper
        result[f"{metric}_valid_bootstrap"] = valid

    minimum_valid = min(
        result[f"{metric}_valid_bootstrap"]
        for metric in BOOTSTRAP_METRICS
    )
    if minimum_valid < max(100, int(0.50 * repeats)):
        result["bootstrap_status"] = "unstable_many_invalid_replicates"
    return result


def run_bootstrap(
    frame: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    groups: List[Tuple[str, str, pd.DataFrame]] = []
    for split_type in SPLIT_ORDER:
        split = frame[frame["split_type"] == split_type]
        if split.empty:
            continue
        groups.append((split_type, "All", split))
        for level in ["L1", "L2"]:
            level_frame = split[split["data_level"] == level]
            if not level_frame.empty:
                groups.append((split_type, level, level_frame))

    rows: List[Dict[str, Any]] = []
    for index, (split_type, level, group) in enumerate(groups, start=1):
        group_seed = (
            seed
            + 1000 * SPLIT_ORDER.index(split_type)
            + 100 * LEVEL_ORDER.index(level)
        )
        metrics = cluster_bootstrap(group, repeats, group_seed)
        rows.append({
            "split_type": split_type,
            "split_label": SPLIT_LABELS[split_type],
            "data_level_analysis": level,
            **metrics,
        })
        print(
            f"[BOOTSTRAP] {index}/{len(groups)} "
            f"{split_type} / {level} / {metrics['bootstrap_status']}"
        )
    return pd.DataFrame(rows)


# =============================================================================
# 9. Existing-metric reproduction audit
# =============================================================================

def audit_existing_metrics(
    recomputed: pd.DataFrame,
    existing_path: str,
) -> pd.DataFrame:
    existing = pd.read_csv(existing_path, low_memory=False)
    keys = ["split_type", "fold"]
    if any(key not in existing.columns for key in keys):
        return pd.DataFrame([{
            "status": "existing_metrics_missing_keys",
        }])

    shared = [
        metric
        for metric in [
            "pinball_q10",
            "pinball_q50",
            "pinball_q90",
            "empirical_coverage_q10",
            "empirical_coverage_q50",
            "empirical_coverage_q90",
            "q50_mae",
            "q50_rmse",
            "q50_r2",
            "q50_spearman",
            "picp_80_ordered",
            "mpiw_80_ordered",
            "winkler_80_ordered",
            "crossing_q10_q50_rate",
            "crossing_q50_q90_rate",
            "any_crossing_rate",
        ]
        if metric in existing.columns and metric in recomputed.columns
    ]

    merged = recomputed.merge(
        existing[keys + shared],
        on=keys,
        how="outer",
        suffixes=("_recomputed", "_existing"),
        indicator=True,
    )

    rows: List[Dict[str, Any]] = []
    for _, item in merged.iterrows():
        row: Dict[str, Any] = {
            "split_type": item.get("split_type"),
            "fold": item.get("fold"),
            "merge_status": item.get("_merge"),
        }
        all_within = item.get("_merge") == "both"
        for metric in shared:
            a = pd.to_numeric(
                pd.Series([item.get(f"{metric}_recomputed")]),
                errors="coerce",
            ).iloc[0]
            b = pd.to_numeric(
                pd.Series([item.get(f"{metric}_existing")]),
                errors="coerce",
            ).iloc[0]
            difference = (
                abs(float(a) - float(b))
                if np.isfinite(a) and np.isfinite(b)
                else np.nan
            )
            row[f"{metric}_absolute_difference"] = difference
            if np.isfinite(difference) and difference > METRIC_COMPARISON_TOLERANCE:
                all_within = False
        row["all_shared_metrics_within_tolerance"] = bool(all_within)
        row["status"] = "passed" if all_within else "check_difference"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 10. Publication tables
# =============================================================================

def format_estimate_ci(
    estimate: Any,
    lower: Any,
    upper: Any,
    digits: int = 3,
) -> str:
    estimate_value = pd.to_numeric(
        pd.Series([estimate]),
        errors="coerce",
    ).iloc[0]
    lower_value = pd.to_numeric(
        pd.Series([lower]),
        errors="coerce",
    ).iloc[0]
    upper_value = pd.to_numeric(
        pd.Series([upper]),
        errors="coerce",
    ).iloc[0]
    if not np.isfinite(estimate_value):
        return ""
    if np.isfinite(lower_value) and np.isfinite(upper_value):
        return (
            f"{float(estimate_value):.{digits}f} "
            f"[{float(lower_value):.{digits}f}, "
            f"{float(upper_value):.{digits}f}]"
        )
    return f"{float(estimate_value):.{digits}f}"


def build_main_table(
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    error_relation: pd.DataFrame,
) -> pd.DataFrame:
    overall_bootstrap = bootstrap[
        bootstrap["data_level_analysis"] == "All"
    ].copy()
    merged = calibration.merge(
        overall_bootstrap,
        on=["split_type", "split_label"],
        how="left",
        suffixes=("", "_bootstrap"),
    )
    merged = merged.merge(
        error_relation[[
            "split_type",
            "Q4_to_Q1_mae_ratio",
            "Q4_minus_Q1_severe_error_rate",
            "mae_monotonic_non_decreasing",
        ]],
        on="split_type",
        how="left",
    )

    rows: List[Dict[str, Any]] = []
    for _, item in merged.iterrows():
        rows.append({
            "Validation setting": item["split_label"],
            "Samples": int(item["n_samples"]),
            "Papers": int(item["n_papers"]),
            "q10 empirical coverage": format_estimate_ci(
                item["empirical_coverage_q10"],
                item.get("empirical_coverage_q10_ci_lower"),
                item.get("empirical_coverage_q10_ci_upper"),
            ),
            "q50 empirical coverage": format_estimate_ci(
                item["empirical_coverage_q50"],
                item.get("empirical_coverage_q50_ci_lower"),
                item.get("empirical_coverage_q50_ci_upper"),
            ),
            "q90 empirical coverage": format_estimate_ci(
                item["empirical_coverage_q90"],
                item.get("empirical_coverage_q90_ci_lower"),
                item.get("empirical_coverage_q90_ci_upper"),
            ),
            "80% PICP [95% CI]": format_estimate_ci(
                item["picp_80_ordered"],
                item.get("picp_80_ordered_ci_lower"),
                item.get("picp_80_ordered_ci_upper"),
            ),
            "MPIW [95% CI]": format_estimate_ci(
                item["mpiw_80_ordered"],
                item.get("mpiw_80_ordered_ci_lower"),
                item.get("mpiw_80_ordered_ci_upper"),
            ),
            "Winkler [95% CI]": format_estimate_ci(
                item["winkler_80_ordered"],
                item.get("winkler_80_ordered_ci_lower"),
                item.get("winkler_80_ordered_ci_upper"),
            ),
            "q50 MAE [95% CI]": format_estimate_ci(
                item["q50_mae"],
                item.get("q50_mae_ci_lower"),
                item.get("q50_mae_ci_upper"),
            ),
            "Width-error Spearman [95% CI]": format_estimate_ci(
                item["uncertainty_error_spearman"],
                item.get("uncertainty_error_spearman_ci_lower"),
                item.get("uncertainty_error_spearman_ci_upper"),
            ),
            "Q4/Q1 MAE ratio": item.get("Q4_to_Q1_mae_ratio", np.nan),
            "Any crossing rate": item["any_crossing_rate"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 11. Plotting
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


def plot_quantile_calibration(
    calibration: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for _, item in calibration.iterrows():
        for name, nominal in QUANTILES.items():
            rows.append({
                "split_type": item["split_type"],
                "split_label": item["split_label"],
                "quantile": name,
                "nominal_quantile": nominal,
                "empirical_coverage": item[f"empirical_coverage_{name}"],
                "signed_calibration_error": item[
                    f"signed_calibration_error_{name}"
                ],
                "absolute_calibration_error": item[
                    f"absolute_calibration_error_{name}"
                ],
                "n_samples": item["n_samples"],
                "n_papers": item["n_papers"],
            })
    source = pd.DataFrame(rows)
    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            "fig5a_quantile_calibration_source_data.csv",
        ),
    )

    fig, ax = plt.subplots(figsize=(7.4, 6.3))
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.1,
        label="Ideal calibration",
    )
    for split_type in SPLIT_ORDER:
        subset = source[
            source["split_type"] == split_type
        ].sort_values("nominal_quantile")
        if subset.empty:
            continue
        ax.plot(
            subset["nominal_quantile"],
            subset["empirical_coverage"],
            marker="o",
            linewidth=1.7,
            label=SPLIT_LABELS[split_type],
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical fraction below prediction")
    ax.set_title("Calibration of q10, q50 and q90 predictions")
    ax.legend(frameon=False)
    apply_axes(ax, "both")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "fig5a_quantile_calibration"),
        dpi,
    )


def plot_interval_coverage_width(
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    overall_bootstrap = bootstrap[
        bootstrap["data_level_analysis"] == "All"
    ]
    source = calibration.merge(
        overall_bootstrap,
        on=["split_type", "split_label"],
        how="left",
        suffixes=("", "_bootstrap"),
    )
    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            "fig5b_interval_coverage_width_source_data.csv",
        ),
    )

    x = np.arange(len(SPLIT_ORDER), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    for ax, metric, lower_name, upper_name, ylabel, title in [
        (
            axes[0],
            "picp_80_ordered",
            "picp_80_ordered_ci_lower",
            "picp_80_ordered_ci_upper",
            "PICP",
            "Coverage of ordered q10-q90 interval",
        ),
        (
            axes[1],
            "mpiw_80_ordered",
            "mpiw_80_ordered_ci_lower",
            "mpiw_80_ordered_ci_upper",
            "Mean prediction interval width",
            "Sharpness of ordered q10-q90 interval",
        ),
    ]:
        values: List[float] = []
        lower_errors: List[float] = []
        upper_errors: List[float] = []
        for split_type in SPLIT_ORDER:
            row = source[source["split_type"] == split_type]
            if row.empty:
                values.append(np.nan)
                lower_errors.append(np.nan)
                upper_errors.append(np.nan)
                continue
            item = row.iloc[0]
            value = float(item[metric])
            lower = item.get(lower_name, np.nan)
            upper = item.get(upper_name, np.nan)
            values.append(value)
            lower_errors.append(
                max(0.0, value - float(lower)) if np.isfinite(lower) else np.nan
            )
            upper_errors.append(
                max(0.0, float(upper) - value) if np.isfinite(upper) else np.nan
            )
        ax.errorbar(
            x,
            np.asarray(values, dtype=float),
            yerr=np.vstack([
                np.asarray(lower_errors, dtype=float),
                np.asarray(upper_errors, dtype=float),
            ]),
            fmt="o",
            capsize=4,
            markersize=7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SPLIT_LABELS[name] for name in SPLIT_ORDER],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        apply_axes(ax, "y")

    axes[0].axhline(
        NOMINAL_INTERVAL_COVERAGE,
        linestyle="--",
        linewidth=1.1,
        label="Nominal 80%",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "fig5b_interval_coverage_width"),
        dpi,
    )


def plot_uncertainty_error_relation(
    frame: pd.DataFrame,
    quartiles: pd.DataFrame,
    relation: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    grouped = frame[frame["split_type"] == "paper_grouped_cv"].copy()
    scatter_source = grouped[[
        "sample_id",
        "paper_id",
        "data_level",
        "growth_mode_normalized",
        "fold",
        "y_true",
        "q50_raw",
        "uncertainty_width",
        "absolute_error_q50_raw",
        "uncertainty_quartile",
        "severe_error_within_fold",
    ]]
    atomic_write_csv(
        scatter_source,
        os.path.join(
            paths.source_data_dir,
            "fig5c_uncertainty_error_scatter_source_data.csv",
        ),
    )
    atomic_write_csv(
        quartiles,
        os.path.join(
            paths.source_data_dir,
            "fig5c_uncertainty_quartile_source_data.csv",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    axes[0].scatter(
        grouped["uncertainty_width"],
        grouped["absolute_error_q50_raw"],
        s=18,
        alpha=0.55,
        edgecolors="none",
    )
    x = grouped["uncertainty_width"].to_numpy(dtype=float)
    y = grouped["absolute_error_q50_raw"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 2 and np.std(x[finite]) > 0:
        slope, intercept = np.polyfit(x[finite], y[finite], 1)
        x_line = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 100)
        axes[0].plot(x_line, slope * x_line + intercept, linewidth=1.5)

    grouped_relation = relation[
        relation["split_type"] == "paper_grouped_cv"
    ]
    if not grouped_relation.empty:
        rho = grouped_relation.iloc[0]["spearman_width_vs_absolute_error"]
        axes[0].text(
            0.04,
            0.96,
            f"Spearman = {rho:.2f}\nn = {len(grouped)}",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
        )
    axes[0].set_xlabel("Ordered q10-q90 interval width")
    axes[0].set_ylabel(r"$|y - q50|$")
    axes[0].set_title("Cross-paper uncertainty versus actual error")
    apply_axes(axes[0], "both")

    x_quartile = np.arange(4, dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(SPLIT_ORDER))
    for split_index, split_type in enumerate(SPLIT_ORDER):
        subset = quartiles[quartiles["split_type"] == split_type]
        values: List[float] = []
        for quartile in QUARTILE_LABELS:
            row = subset[subset["uncertainty_quartile"] == quartile]
            values.append(
                float(row["mean_absolute_error"].iloc[0])
                if not row.empty
                else np.nan
            )
        axes[1].plot(
            x_quartile + offsets[split_index],
            np.asarray(values, dtype=float),
            marker="o",
            linewidth=1.5,
            label=SPLIT_LABELS[split_type],
        )
    axes[1].set_xticks(x_quartile)
    axes[1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    axes[1].set_xlabel("Uncertainty-width quartile within each fold")
    axes[1].set_ylabel("Mean absolute q50 error")
    axes[1].set_title("Error stratification by predicted uncertainty")
    axes[1].legend(frameon=False)
    apply_axes(axes[1], "y")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "fig5c_uncertainty_error_relation"),
        dpi,
    )


def plot_by_level(by_level: pd.DataFrame, paths: Paths, dpi: int) -> None:
    atomic_write_csv(
        by_level,
        os.path.join(
            paths.source_data_dir,
            "figS_uncertainty_by_level_source_data.csv",
        ),
    )
    x = np.arange(len(SPLIT_ORDER), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    for ax, metric, ylabel, title in [
        (axes[0], "picp_80_ordered", "PICP", "80% interval coverage"),
        (axes[1], "mpiw_80_ordered", "MPIW", "Interval width"),
        (axes[2], "q50_mae", "q50 MAE", "Median-model error"),
    ]:
        for level_index, level in enumerate(["L1", "L2"]):
            values: List[float] = []
            for split_type in SPLIT_ORDER:
                row = by_level[
                    (by_level["split_type"] == split_type)
                    & (by_level["data_level_analysis"] == level)
                ]
                values.append(
                    float(row[metric].iloc[0]) if not row.empty else np.nan
                )
            offset = (level_index - 0.5) * width
            ax.bar(
                x + offset,
                np.asarray(values, dtype=float),
                width=width,
                label=level,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SPLIT_LABELS[name] for name in SPLIT_ORDER],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        apply_axes(ax, "y")
    axes[0].axhline(
        NOMINAL_INTERVAL_COVERAGE,
        linestyle="--",
        linewidth=1.0,
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "figS_uncertainty_by_level"),
        dpi,
    )


def mode_matrix(
    by_mode: pd.DataFrame,
    metric: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    present = list(by_mode["growth_mode"].dropna().astype(str).unique())
    modes = [mode for mode in ["A", "B", "C", "D", "E"] if mode in present]
    modes += sorted(mode for mode in present if mode not in modes)
    matrix = pd.DataFrame(np.nan, index=SPLIT_ORDER, columns=modes, dtype=float)
    for _, row in by_mode.iterrows():
        split_type = str(row["split_type"])
        mode = str(row["growth_mode"])
        if split_type in matrix.index and mode in matrix.columns:
            matrix.loc[split_type, mode] = row[metric]
    return matrix, list(SPLIT_ORDER), modes


def plot_by_mode(by_mode: pd.DataFrame, paths: Paths, dpi: int) -> None:
    atomic_write_csv(
        by_mode,
        os.path.join(
            paths.source_data_dir,
            "figS_uncertainty_by_mode_source_data.csv",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    for ax, metric, title in [
        (axes[0], "picp_80_ordered", "PICP"),
        (axes[1], "mpiw_80_ordered", "MPIW"),
        (
            axes[2],
            "uncertainty_error_spearman",
            "Width-error Spearman",
        ),
    ]:
        matrix, splits, modes = mode_matrix(by_mode, metric)
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_yticks(np.arange(len(splits)))
        ax.set_yticklabels([SPLIT_LABELS[name] for name in splits])
        ax.set_xlabel("Growth mode")
        ax.set_title(title)
        for row_index in range(len(splits)):
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
        "Growth-mode-resolved uncertainty calibration",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "figS_uncertainty_by_mode"),
        dpi,
    )


def plot_crossing(crossing: pd.DataFrame, paths: Paths, dpi: int) -> None:
    atomic_write_csv(
        crossing,
        os.path.join(
            paths.source_data_dir,
            "figS_quantile_crossing_source_data.csv",
        ),
    )
    overall = crossing[crossing["analysis_group"] == "All"]
    x = np.arange(len(SPLIT_ORDER), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    for index, (metric, label) in enumerate([
        ("crossing_q10_q50_rate", "q10 > q50"),
        ("crossing_q50_q90_rate", "q50 > q90"),
        ("any_crossing_rate", "Any crossing"),
    ]):
        values: List[float] = []
        for split_type in SPLIT_ORDER:
            row = overall[overall["split_type"] == split_type]
            values.append(
                float(row[metric].iloc[0]) if not row.empty else np.nan
            )
        ax.bar(
            x + (index - 1.0) * width,
            np.asarray(values, dtype=float),
            width=width,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SPLIT_LABELS[name] for name in SPLIT_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Crossing rate")
    ax.set_title("Raw quantile crossing audit")
    ax.legend(frameon=False)
    apply_axes(ax, "y")
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(paths.figures_dir, "figS_quantile_crossing"),
        dpi,
    )


def plot_by_component_count(
    by_component: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> bool:
    if by_component.empty:
        print("[WARN] component_count is unavailable; skipping component figure.")
        return False
    atomic_write_csv(
        by_component,
        os.path.join(
            paths.source_data_dir,
            "figS_uncertainty_by_component_count_source_data.csv",
        ),
    )
    grouped = by_component[
        by_component["split_type"] == "paper_grouped_cv"
    ].sort_values("component_count_int")
    if grouped.empty:
        return False
    x = grouped["component_count_int"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, metric, ylabel, title in [
        (axes[0], "q50_mae", "q50 MAE", "Prediction error"),
        (axes[1], "mpiw_80_ordered", "MPIW", "Interval width"),
        (axes[2], "picp_80_ordered", "PICP", "Interval coverage"),
    ]:
        ax.plot(
            x,
            grouped[metric].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
        )
        ax.set_xticks(x)
        ax.set_xlabel("Available composite components")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        apply_axes(ax, "y")
    axes[2].axhline(
        NOMINAL_INTERVAL_COVERAGE,
        linestyle="--",
        linewidth=1.0,
    )
    fig.suptitle(
        "Paper-grouped uncertainty by composite-label completeness",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "figS_uncertainty_by_component_count",
        ),
        dpi,
    )
    return True


# =============================================================================
# 12. Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate article-ready L12 quantile calibration and risk-validity "
            "analyses from existing q10/q50/q90 predictions."
        )
    )
    parser.add_argument(
        "--project_dir",
        default=None,
        help="Project directory; default is the repository root resolved from this script.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory; default is <project_dir>/outputs/external_validation/uncertainty.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
        help="Number of paper-cluster bootstrap repeats; 0 disables bootstrap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Bootstrap random seed.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PNG resolution.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = (
        os.path.abspath(args.project_dir)
        if args.project_dir
        else os.path.abspath(os.path.join(script_dir, "..", ".."))
    )
    paths = build_paths(project_dir, args.output_dir)
    ensure_inputs_exist(paths)
    create_output_dirs(paths)

    print("=" * 100)
    print("[ARTICLE UNCERTAINTY ANALYSIS START]")
    print(f"Project directory : {paths.project_dir}")
    print(f"L12 result dir    : {paths.l12_dir}")
    print(f"Bootstrap repeats : {args.bootstrap}")
    print(f"Random seed       : {args.seed}")
    print(f"PNG dpi           : {args.dpi}")
    print(f"Output            : {paths.output_dir}")
    print("=" * 100)

    predictions = add_derived_columns(load_predictions(paths.predictions))
    manifest = load_manifest(paths.split_manifest)
    audit = audit_predictions(predictions, manifest)
    predictions = assign_uncertainty_quartiles(predictions)

    fold_metrics = build_fold_metrics(predictions)
    calibration = build_calibration(predictions)
    by_level = build_by_level(predictions)
    level_comparison = build_level_comparison(by_level)
    by_mode = build_by_growth_mode(predictions)
    by_component = build_by_component_count(predictions)
    crossing = build_crossing_analysis(predictions)
    quartiles = build_quartile_analysis(predictions)
    quartile_summary = build_quartile_summary(quartiles)
    error_relation = build_error_relation(predictions, quartile_summary)
    bootstrap = run_bootstrap(predictions, args.bootstrap, args.seed)
    metric_audit = audit_existing_metrics(
        fold_metrics,
        paths.metrics_fold_existing,
    )

    output_files = {
        "fold_metrics": os.path.join(
            paths.output_dir,
            "uncertainty_metrics_fold.csv",
        ),
        "calibration": os.path.join(
            paths.output_dir,
            "uncertainty_calibration.csv",
        ),
        "bootstrap": os.path.join(
            paths.output_dir,
            "uncertainty_bootstrap_ci.csv",
        ),
        "error_relation": os.path.join(
            paths.output_dir,
            "uncertainty_error_relation.csv",
        ),
        "quartiles": os.path.join(
            paths.output_dir,
            "uncertainty_quartile_analysis.csv",
        ),
        "quartile_summary": os.path.join(
            paths.output_dir,
            "uncertainty_quartile_summary.csv",
        ),
        "by_level": os.path.join(
            paths.output_dir,
            "uncertainty_by_level.csv",
        ),
        "level_comparison": os.path.join(
            paths.output_dir,
            "uncertainty_level_comparison.csv",
        ),
        "by_mode": os.path.join(
            paths.output_dir,
            "uncertainty_by_growth_mode.csv",
        ),
        "by_component": os.path.join(
            paths.output_dir,
            "uncertainty_by_component_count.csv",
        ),
        "crossing": os.path.join(
            paths.output_dir,
            "quantile_crossing_analysis.csv",
        ),
        "predictions_enriched": os.path.join(
            paths.output_dir,
            "L12_predictions_enriched_for_uncertainty_analysis.csv",
        ),
    }

    for frame, key in [
        (fold_metrics, "fold_metrics"),
        (calibration, "calibration"),
        (bootstrap, "bootstrap"),
        (error_relation, "error_relation"),
        (quartiles, "quartiles"),
        (quartile_summary, "quartile_summary"),
        (by_level, "by_level"),
        (level_comparison, "level_comparison"),
        (by_mode, "by_mode"),
        (by_component, "by_component"),
        (crossing, "crossing"),
        (predictions, "predictions_enriched"),
    ]:
        atomic_write_csv(frame, output_files[key])

    audit_path = os.path.join(
        paths.manifests_dir,
        "uncertainty_validation_audit.json",
    )
    metric_audit_path = os.path.join(
        paths.manifests_dir,
        "existing_L12_metrics_reproduction_audit.csv",
    )
    input_manifest_path = os.path.join(
        paths.manifests_dir,
        "input_file_manifest.csv",
    )
    atomic_write_json(audit, audit_path)
    atomic_write_csv(metric_audit, metric_audit_path)
    atomic_write_csv(
        pd.DataFrame([
            {
                "path": path,
                "exists": os.path.isfile(path),
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
            for path in [
                paths.predictions,
                paths.metrics_fold_existing,
                paths.split_manifest,
            ]
        ]),
        input_manifest_path,
    )

    main_table = build_main_table(
        calibration,
        bootstrap,
        error_relation,
    )
    table_paths = {
        "main": os.path.join(
            paths.tables_dir,
            "table_main_uncertainty_validation.csv",
        ),
        "fold": os.path.join(
            paths.tables_dir,
            "table_si_uncertainty_fold_metrics.csv",
        ),
        "level": os.path.join(
            paths.tables_dir,
            "table_si_uncertainty_by_level.csv",
        ),
        "mode": os.path.join(
            paths.tables_dir,
            "table_si_uncertainty_by_growth_mode.csv",
        ),
        "component": os.path.join(
            paths.tables_dir,
            "table_si_uncertainty_by_component_count.csv",
        ),
        "quartile": os.path.join(
            paths.tables_dir,
            "table_si_uncertainty_quartiles.csv",
        ),
        "crossing": os.path.join(
            paths.tables_dir,
            "table_si_quantile_crossing.csv",
        ),
        "bootstrap": os.path.join(
            paths.tables_dir,
            "table_si_uncertainty_bootstrap_ci.csv",
        ),
    }
    for frame, key in [
        (main_table, "main"),
        (fold_metrics, "fold"),
        (by_level, "level"),
        (by_mode, "mode"),
        (by_component, "component"),
        (quartiles, "quartile"),
        (crossing, "crossing"),
        (bootstrap, "bootstrap"),
    ]:
        atomic_write_csv(frame, table_paths[key])

    plot_quantile_calibration(calibration, paths, args.dpi)
    plot_interval_coverage_width(calibration, bootstrap, paths, args.dpi)
    plot_uncertainty_error_relation(
        predictions,
        quartiles,
        error_relation,
        paths,
        args.dpi,
    )
    plot_by_level(by_level, paths, args.dpi)
    plot_by_mode(by_mode, paths, args.dpi)
    plot_crossing(crossing, paths, args.dpi)
    component_figure = plot_by_component_count(
        by_component,
        paths,
        args.dpi,
    )

    summary_path = os.path.join(
        paths.output_dir,
        "uncertainty_analysis_summary.json",
    )
    summary = {
        "analysis": (
            "CNT 3.0 article-ready L12 uncertainty calibration and "
            "risk-validity analysis"
        ),
        "project_dir": paths.project_dir,
        "l12_dir": paths.l12_dir,
        "output_dir": paths.output_dir,
        "prediction_rows": int(len(predictions)),
        "samples_by_split": {
            split_type: int(
                predictions.loc[
                    predictions["split_type"] == split_type,
                    "sample_id",
                ].nunique()
            )
            for split_type in SPLIT_ORDER
        },
        "papers_by_split": {
            split_type: int(
                predictions.loc[
                    predictions["split_type"] == split_type,
                    "paper_id",
                ].nunique()
            )
            for split_type in SPLIT_ORDER
        },
        "bootstrap": {
            "repeats": int(args.bootstrap),
            "seed": int(args.seed),
            "cluster": "paper_id",
            "confidence_level": 0.95,
        },
        "definitions": {
            "quantile_calibration": "Raw q10/q50/q90 predictions.",
            "point_error": "Absolute error of raw q50.",
            "interval": "Per-sample ordered q10-q90 interval.",
            "uncertainty": "q90_ordered - q10_ordered.",
            "risk_validity": (
                "Width-error correlation plus within-fold width quartiles."
            ),
            "severe_error": (
                "Absolute q50 error at or above the within-fold 80th percentile."
            ),
        },
        "outputs": {
            **output_files,
            "tables": table_paths,
            "figures_dir": paths.figures_dir,
            "source_data_dir": paths.source_data_dir,
            "audit": audit_path,
            "metric_reproduction_audit": metric_audit_path,
            "input_manifest": input_manifest_path,
        },
        "figures": {
            "fig5a_quantile_calibration": True,
            "fig5b_interval_coverage_width": True,
            "fig5c_uncertainty_error_relation": True,
            "figS_uncertainty_by_level": True,
            "figS_uncertainty_by_mode": True,
            "figS_quantile_crossing": True,
            "figS_uncertainty_by_component_count": component_figure,
        },
        "interpretation_rules": {
            "calibration": (
                "Empirical q10/q50/q90 coverage should approach 0.10/0.50/0.90."
            ),
            "interval": (
                "PICP should approach 0.80; MPIW and Winkler must be read jointly."
            ),
            "useful_uncertainty": (
                "Width-error Spearman should be positive and Q4 should be more "
                "error-prone than Q1."
            ),
            "crossing": (
                "Raw crossing is reported; ordering is used only for interval metrics."
            ),
        },
    }
    atomic_write_json(summary, summary_path)

    print("\n" + "=" * 100)
    print("[ARTICLE UNCERTAINTY ANALYSIS FINISHED]")
    print(f"Fold metrics       : {output_files['fold_metrics']}")
    print(f"Calibration        : {output_files['calibration']}")
    print(f"Bootstrap CI       : {output_files['bootstrap']}")
    print(f"Error relation     : {output_files['error_relation']}")
    print(f"Quartiles          : {output_files['quartiles']}")
    print(f"By level           : {output_files['by_level']}")
    print(f"By growth mode     : {output_files['by_mode']}")
    print(f"By components      : {output_files['by_component']}")
    print(f"Crossing analysis  : {output_files['crossing']}")
    print(f"Figures            : {paths.figures_dir}")
    print(f"Source data        : {paths.source_data_dir}")
    print(f"Tables             : {paths.tables_dir}")
    print(f"Audit              : {audit_path}")
    print(f"Run summary        : {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
