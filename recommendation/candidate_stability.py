# -*- coding: utf-8 -*-
"""
candidate_stability.py

CNT 3.0 — Candidate ranking stability across paper-grouped fold models

Purpose
-------
Evaluate whether L123 catalyst-route candidates remain highly ranked across the
five paper-grouped validation-fold q50/q90 models saved by
composite_sensitivity.py.

The script first reuses saved paper-grouped fold models when they exist. If the
upstream sensitivity run did not save model files, it rebuilds only the required
Composite>=2 paper-grouped q50/q90 fold models from the frozen L12 split manifest
using the exact training definitions in composite_sensitivity.py, saves
them locally, and then performs the stability analysis.

Fold-wise variation is not treated as a prediction confidence interval. It
measures ranking sensitivity to changes in the paper-level training composition.

Candidate-pool behavior
-----------------------
Candidate stability is defined only for a completed route-level candidate pool.
The default ``--candidates auto`` first looks for the formal current-project
candidate pool. If none exists, the script writes an explicit SKIPPED summary
and exits with code 0; it never fabricates stability results.

Temporary app runs are not used silently. They can be selected explicitly
with ``--l123_run <run_directory>`` or included in auto-discovery with
``--include_latest_app``.

Default inputs when a candidate pool is available
-------------------------------------------------
<project_dir>/outputs/recommendation/rank_candidate_routes/all_route_predictions.csv
<project_dir>/outputs/external_validation/composite_sensitivity/
    composite_sensitivity_metrics_fold.csv
    validation_models/min_components_2/L12_quantile/paper_grouped_cv/
<project_dir>/data/11_flattened/ml_dataset_3p0_full60D.csv
<project_dir>/data/11_flattened/feature_schema.json
<project_dir>/outputs/recommendation/optimize_physical_state/process_packages_with_physical_state.csv

Default outputs
---------------
<project_dir>/outputs/recommendation/candidate_stability/
    candidate_fold_predictions.csv
    candidate_rank_stability.csv
    catalyst_fold_rankings.csv
    catalyst_rank_stability.csv
    topk_jaccard.csv
    topk_jaccard_summary.csv
    fold_model_manifest.csv
    domain_thresholds.csv
    final_stable_candidates.csv
    fig_candidate_rank_stability.png
    fig_topk_jaccard.png
    candidate_stability_summary.json

Run in the full project pipeline
--------------------------------
python recommendation/candidate_stability.py

If no formal candidate pool exists, the run is marked SKIPPED and exits normally.

Run on one explicit app result
-------------------------------------
python recommendation/candidate_stability.py --l123_run outputs/app/run_xxx

Allow auto-discovery of the newest temporary app candidate pool
----------------------------------------------------------------------
python recommendation/candidate_stability.py --include_latest_app

Require a candidate pool instead of skipping
--------------------------------------------
python recommendation/candidate_stability.py --require_candidates

Use another sensitivity output directory
----------------------------------------
python recommendation/candidate_stability.py --sensitivity_dir outputs/external_validation/composite_sensitivity
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# 1. Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_CANDIDATES = "auto"
DEFAULT_SENSITIVITY_DIR = (
    PROJECT_DIR / "outputs" / "external_validation" / "composite_sensitivity"
)
DEFAULT_SENSITIVITY_SCRIPT = (
    PROJECT_DIR / "models" / "external_validation" / "composite_sensitivity.py"
)
DEFAULT_L12_SPLIT_MANIFEST = (
    PROJECT_DIR / "outputs" / "models" / "l12_quantile" / "L12_split_manifest.csv"
)
DEFAULT_DATASET = PROJECT_DIR / "data" / "11_flattened" / "ml_dataset_3p0_full60D.csv"
DEFAULT_SCHEMA = PROJECT_DIR / "data" / "11_flattened" / "feature_schema.json"
DEFAULT_BOUNDARY_PATH = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "optimize_physical_state"
    / "process_packages_with_physical_state.csv"
)
DEFAULT_BOUNDARY = "auto"
SCRIPT_REVISION = "current_repo_20261001"
DEFAULT_OUTDIR = PROJECT_DIR / "outputs" / "recommendation" / "candidate_stability"

MIN_COMPONENT_COUNT = 2
EXPECTED_FOLDS = 5
CATALYST_AGGREGATE_TOP_N = 5
TOP_K_VALUES = (10, 20)
DEFAULT_DPI = 600

N_COLS = [f"N{i}" for i in range(1, 19)]
C_COLS = [f"C{i}" for i in range(1, 13)]
F_COLS = [f"F{i}" for i in range(1, 19)]
P_COLS = [f"P{i}" for i in range(1, 13)]
PHYSICAL_COLS = N_COLS + C_COLS + F_COLS + P_COLS
MISS_COLS = [f"miss_{column}" for column in PHYSICAL_COLS]
MODEL_COLS = PHYSICAL_COLS + MISS_COLS

PREFERRED_IGID_COL = "y_ig_id_ratio"
LEGACY_IGID_COL = "y_IG_ID"
COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

FREQUENCY_DENOMINATOR = float(EXPECTED_FOLDS)

PROFILE_COLUMNS = [
    "recipe_key",
    "metal_set",
    "metal1",
    "metal2",
    "metal3",
    "frac1",
    "frac2",
    "frac3",
    "candidate_growth_mode",
    "growth_mode",
    "source_level",
    "candidate_state_id",
]


# =============================================================================
# 2. Paths and basic helpers
# =============================================================================

@dataclass(frozen=True)
class Paths:
    project_dir: Path
    candidates: Path
    sensitivity_dir: Path
    metrics_fold: Path
    validation_model_dir: Path
    sensitivity_script: Path
    l12_split_manifest: Path
    generated_model_dir: Path
    dataset: Path
    schema: Path
    boundary: Path
    outdir: Path
    fold_predictions: Path
    candidate_stability: Path
    catalyst_fold_rankings: Path
    catalyst_stability: Path
    jaccard: Path
    jaccard_summary: Path
    model_manifest: Path
    domain_thresholds: Path
    final_candidates: Path
    fig_stability: Path
    fig_jaccard: Path
    summary: Path


def resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def build_paths(args: argparse.Namespace) -> Paths:
    project_dir = Path(args.project_dir).expanduser().resolve()
    sensitivity_dir = resolve_path(project_dir, args.sensitivity_dir)
    outdir = resolve_path(project_dir, args.outdir)
    return Paths(
        project_dir=project_dir,
        candidates=resolve_path(project_dir, args.candidates),
        sensitivity_dir=sensitivity_dir,
        metrics_fold=sensitivity_dir / "composite_sensitivity_metrics_fold.csv",
        validation_model_dir=(
            sensitivity_dir
            / "validation_models"
            / f"min_components_{args.min_components}"
            / "L12_quantile"
            / "paper_grouped_cv"
        ),
        sensitivity_script=resolve_path(project_dir, args.sensitivity_script),
        l12_split_manifest=resolve_path(project_dir, args.l12_manifest),
        generated_model_dir=outdir / "fold_models",
        dataset=resolve_path(project_dir, args.dataset),
        schema=resolve_path(project_dir, args.schema),
        boundary=resolve_path(project_dir, args.boundary_csv),
        outdir=outdir,
        fold_predictions=outdir / "candidate_fold_predictions.csv",
        candidate_stability=outdir / "candidate_rank_stability.csv",
        catalyst_fold_rankings=outdir / "catalyst_fold_rankings.csv",
        catalyst_stability=outdir / "catalyst_rank_stability.csv",
        jaccard=outdir / "topk_jaccard.csv",
        jaccard_summary=outdir / "topk_jaccard_summary.csv",
        model_manifest=outdir / "fold_model_manifest.csv",
        domain_thresholds=outdir / "domain_thresholds.csv",
        final_candidates=outdir / "final_stable_candidates.csv",
        fig_stability=outdir / "fig_candidate_rank_stability.png",
        fig_jaccard=outdir / "fig_topk_jaccard.png",
        summary=outdir / "candidate_stability_summary.json",
    )


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found:\n  {path}")


def read_csv_smart(path: Path) -> pd.DataFrame:
    ensure_file(path, "CSV")
    last_error: Optional[Exception] = None
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read CSV:\n  {path}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns:\n  " + "\n  ".join(missing))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_mode(value: Any) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    token = str(value).strip().upper()
    direct = {
        "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    }
    if token in direct:
        return direct[token]
    for letter in "ABCDE":
        if token.startswith((f"{letter}_", f"{letter}-", f"MODE_{letter}")):
            return letter
    return token if token else "UNKNOWN"


def quantile_iqr(values: pd.Series) -> float:
    values = numeric(values).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def unique_names(columns: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(columns))



# =============================================================================
# 2b. Optional candidate-pool resolution
# =============================================================================

@dataclass(frozen=True)
class CandidatePoolResolution:
    candidates: Optional[Path]
    boundary: Optional[Path]
    source: str
    searched_locations: List[str]


def infer_boundary_from_candidates(candidate_path: Path) -> Path:
    """
    Infer the matching physical-state recommendation table from a route-ranking pool.
    The inferred file is optional; missing boundary information does not block
    the fold-stability analysis.
    """
    candidate_path = candidate_path.expanduser().resolve()
    stage001 = candidate_path.parent

    if stage001.name == "rank_candidate_routes":
        return (
            stage001.parent
            / "optimize_physical_state"
            / "process_packages_with_physical_state.csv"
        ).resolve()

    return DEFAULT_BOUNDARY_PATH.resolve()


def candidate_pool_from_run(root: Path) -> Tuple[Path, Path, str] | None:
    """
    Resolve a candidate pool from a current-project or app run root.
    """
    root = root.expanduser().resolve()

    layouts = [
        (
            "current_project",
            root
            / "outputs"
            / "recommendation"
            / "rank_candidate_routes"
            / "all_route_predictions.csv",
            root
            / "outputs"
            / "recommendation"
            / "optimize_physical_state"
            / "process_packages_with_physical_state.csv",
        ),
        (
            "app_run",
            root / "rank_candidate_routes" / "all_route_predictions.csv",
            root
            / "optimize_physical_state"
            / "process_packages_with_physical_state.csv",
        ),
    ]

    # Also accept the route-ranking stage directory itself.
    if root.name == "rank_candidate_routes":
        layouts.insert(
            0,
            (
                "rank_candidate_routes_stage",
                root / "all_route_predictions.csv",
                root.parent
                / "optimize_physical_state"
                / "process_packages_with_physical_state.csv",
            ),
        )

    for layout, candidates, boundary in layouts:
        if candidates.is_file():
            return candidates.resolve(), boundary.resolve(), layout

    return None


def newest_candidate_pool_under(
    base_dirs: Sequence[Path],
) -> Tuple[Path, Path, str] | None:
    matches: List[Path] = []

    for base in base_dirs:
        if not base.is_dir():
            continue
        matches.extend(
            path.resolve()
            for path in base.rglob("all_route_predictions.csv")
            if path.is_file()
            and path.parent.name == "rank_candidate_routes"
        )

    if not matches:
        return None

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    candidates = matches[0]
    return (
        candidates,
        infer_boundary_from_candidates(candidates),
        "latest_temporary_app",
    )


def resolve_candidate_pool(
    args: argparse.Namespace,
    project_dir: Path,
) -> CandidatePoolResolution:
    """
    Resolve a formal candidate pool without silently coupling the analysis to
    the most recent app request.
    """
    searched: List[str] = []

    l123_run = str(args.l123_run).strip()
    if l123_run:
        root = resolve_path(project_dir, l123_run)
        searched.append(str(root))
        if not root.is_dir():
            raise FileNotFoundError(
                f"Selected L123 run directory not found:\n  {root}"
            )

        resolved = candidate_pool_from_run(root)
        if resolved is None:
            raise FileNotFoundError(
                "The selected run does not contain "
                "rank_candidate_routes/all_route_predictions.csv:\n"
                f"  {root}"
            )

        candidates, inferred_boundary, layout = resolved
        boundary = (
            resolve_path(project_dir, args.boundary_csv)
            if str(args.boundary_csv).strip().lower() != "auto"
            else inferred_boundary
        )
        return CandidatePoolResolution(
            candidates=candidates,
            boundary=boundary,
            source=f"explicit_l123_run:{layout}",
            searched_locations=searched,
        )

    candidate_token = str(args.candidates).strip()
    if candidate_token.lower() != "auto":
        candidates = resolve_path(project_dir, candidate_token)
        searched.append(str(candidates))
        if not candidates.is_file():
            raise FileNotFoundError(
                f"Explicit candidate pool not found:\n  {candidates}"
            )

        boundary = (
            resolve_path(project_dir, args.boundary_csv)
            if str(args.boundary_csv).strip().lower() != "auto"
            else infer_boundary_from_candidates(candidates)
        )
        return CandidatePoolResolution(
            candidates=candidates,
            boundary=boundary,
            source="explicit_candidate_csv",
            searched_locations=searched,
        )

    # Formal current-project pool first. Temporary app runs are intentionally
    # excluded unless --include_latest_app is explicitly requested.
    formal_candidates = [
        (
            project_dir
            / "outputs"
            / "recommendation"
            / "rank_candidate_routes"
            / "all_route_predictions.csv"
        ),
    ]

    for candidates in formal_candidates:
        candidates = candidates.resolve()
        searched.append(str(candidates))
        if candidates.is_file():
            boundary = (
                resolve_path(project_dir, args.boundary_csv)
                if str(args.boundary_csv).strip().lower() != "auto"
                else infer_boundary_from_candidates(candidates)
            )
            return CandidatePoolResolution(
                candidates=candidates,
                boundary=boundary,
                source="auto_formal_candidate_pool",
                searched_locations=searched,
            )

    if args.include_latest_app:
        temporary_bases = [
            project_dir / "outputs" / "app",
        ]
        searched.extend(str(path.resolve()) for path in temporary_bases)
        resolved = newest_candidate_pool_under(temporary_bases)
        if resolved is not None:
            candidates, inferred_boundary, source = resolved
            boundary = (
                resolve_path(project_dir, args.boundary_csv)
                if str(args.boundary_csv).strip().lower() != "auto"
                else inferred_boundary
            )
            return CandidatePoolResolution(
                candidates=candidates,
                boundary=boundary,
                source=source,
                searched_locations=searched,
            )

    return CandidatePoolResolution(
        candidates=None,
        boundary=None,
        source="not_available",
        searched_locations=searched,
    )


def remove_stale_result_files(paths: Paths) -> None:
    """
    Remove old result tables/figures when the current analysis is skipped.
    Auto-trained fold-model files are retained for reuse.
    """
    result_files = [
        paths.fold_predictions,
        paths.candidate_stability,
        paths.catalyst_fold_rankings,
        paths.catalyst_stability,
        paths.jaccard,
        paths.jaccard_summary,
        paths.model_manifest,
        paths.domain_thresholds,
        paths.final_candidates,
        paths.fig_stability,
        paths.fig_jaccard,
        paths.summary,
        paths.outdir / "candidate_stability_status.csv",
    ]

    for path in result_files:
        if path.is_file():
            path.unlink()


def write_skipped_candidate_analysis(
    paths: Paths,
    resolution: CandidatePoolResolution,
) -> None:
    """
    Record a scientifically explicit skip instead of failing the whole pipeline.
    """
    paths.outdir.mkdir(parents=True, exist_ok=True)
    remove_stale_result_files(paths)

    reason = (
        "No completed route-level candidate pool was found. Candidate ranking "
        "stability is undefined without catalyst-route candidates, so the "
        "analysis was skipped rather than fabricating a result."
    )
    status = pd.DataFrame(
        [
            {
                "status": "SKIPPED_NO_CANDIDATE_POOL",
                "candidate_source": resolution.source,
                "reason": reason,
                "exit_code": 0,
            }
        ]
    )
    write_csv(
        status,
        paths.outdir / "candidate_stability_status.csv",
    )

    summary = {
        "script_revision": SCRIPT_REVISION,
        "analysis": "CNT 3.0 candidate ranking stability",
        "status": "skipped_no_candidate_pool",
        "candidate_source": resolution.source,
        "candidate_pool": None,
        "reason": reason,
        "searched_locations": resolution.searched_locations,
        "scientific_boundary": (
            "Candidate stability requires a fixed route-level candidate pool. "
            "The dataset or trained fold models alone cannot define "
            "candidate ranks or Top-K overlap."
        ),
        "how_to_run": {
            "formal_pool": (
                "python recommendation/candidate_stability.py "
                "--candidates <rank_candidate_routes/all_route_predictions.csv>"
            ),
            "specific_app_run": (
                "python recommendation/candidate_stability.py "
                "--l123_run <outputs/app/run_xxx>"
            ),
            "latest_app_opt_in": (
                "python recommendation/candidate_stability.py --include_latest_app"
            ),
        },
        "output_directory": str(paths.outdir),
    }
    write_json(summary, paths.summary)

    print("=" * 108)
    print("[CANDIDATE STABILITY SKIPPED]")
    print(f"Script revision      : {SCRIPT_REVISION}")
    print(f"Project directory    : {paths.project_dir}")
    print("Candidate pool       : NOT AVAILABLE")
    print("Status               : SKIPPED_NO_CANDIDATE_POOL")
    print(f"Output               : {paths.outdir}")
    print("-" * 108)
    print(reason)
    print(
        "This is expected when the project has not completed the formal "
        "route-level candidate screening."
    )
    print(
        "Use --l123_run <run_directory> for one selected app run, "
        "or --include_latest_app to opt in to the latest app candidate pool."
    )
    print("=" * 108)


# =============================================================================
# 3. Feature schema and candidate pool
# =============================================================================


def load_schema(path: Path) -> List[str]:
    ensure_file(path, "Feature schema")
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    columns = schema.get("model_feature_columns")
    if not isinstance(columns, list) or not columns:
        raise RuntimeError("feature_schema.json lacks model_feature_columns.")
    columns = [str(column) for column in columns]
    if len(columns) != len(set(columns)):
        raise RuntimeError("model_feature_columns contains duplicate names.")
    if len(columns) != 120:
        raise RuntimeError(f"Expected 120 model features, found {len(columns)}.")
    return columns


def prepare_candidate_pool(path: Path, feature_columns: Sequence[str]) -> pd.DataFrame:
    frame = read_csv_smart(path).copy()
    require_columns(frame, ["catalyst_id", "template_id", *feature_columns], "Candidate pool")

    for column in ["catalyst_id", "template_id"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        invalid = frame[column].eq("") | frame[column].str.upper().eq("NAN")
        if invalid.any():
            raise RuntimeError(f"{column} has {int(invalid.sum())} invalid rows.")

    if "combination_id" not in frame.columns:
        frame["combination_id"] = frame["catalyst_id"] + "||" + frame["template_id"]
    frame["combination_id"] = frame["combination_id"].fillna("").astype(str).str.strip()

    duplicates = int(frame["combination_id"].duplicated().sum())
    if duplicates:
        raise RuntimeError(
            f"combination_id has {duplicates} duplicates. Fix rank_candidate_routes output first."
        )

    for column in feature_columns:
        frame[column] = numeric(frame[column])

    for feature in PHYSICAL_COLS:
        miss = f"miss_{feature}"
        frame[miss] = numeric(frame[miss]).fillna(1).gt(0).astype(int)
        frame.loc[frame[feature].isna(), miss] = 1
        frame.loc[frame[miss].eq(1), feature] = np.nan

    # Consolidate blocks after repeated numeric coercion to avoid pandas
    # fragmentation warnings on the 120D candidate table.
    frame = frame.copy()
    if "growth_mode" not in frame.columns:
        if "candidate_growth_mode" in frame.columns:
            frame["growth_mode"] = frame["candidate_growth_mode"]
        else:
            frame["growth_mode"] = "UNKNOWN"
    frame = frame.assign(
        growth_mode_normalized=frame["growth_mode"].map(normalize_mode)
    )

    return frame.reset_index(drop=True)


# =============================================================================
# 4. Fold model discovery and validation
# =============================================================================


def model_path_from_metric(
    value: Any,
    sensitivity_dir: Path,
) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NAN":
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = sensitivity_dir / path
    path = path.resolve()
    return path if path.is_file() else None


def split_manifest_fold_ids(path: Path) -> List[int]:
    """Read the actual paper-grouped fold ids from the frozen L12 manifest."""
    manifest = read_csv_smart(path)
    require_columns(manifest, ["split_type", "fold", "role", "sample_id"], "L12 split manifest")
    subset = manifest[manifest["split_type"].astype(str).eq("paper_grouped_cv")].copy()
    folds = sorted(numeric(subset["fold"]).dropna().astype(int).unique().tolist())
    return folds


def discover_fold_models(paths: Paths, min_components: int) -> pd.DataFrame:
    """Discover saved models without assuming that the upstream run used --save_models."""
    metric_rows = pd.DataFrame()
    if paths.metrics_fold.is_file():
        metrics = read_csv_smart(paths.metrics_fold)
        required = [
            "minimum_component_count",
            "model_family",
            "split_type",
            "fold",
            "status",
        ]
        require_columns(metrics, required, "Sensitivity fold metrics")
        metric_rows = metrics[
            numeric(metrics["minimum_component_count"]).eq(min_components)
            & metrics["model_family"].astype(str).eq("L12_quantile")
            & metrics["split_type"].astype(str).eq("paper_grouped_cv")
            & metrics["status"].astype(str).eq("ok")
        ].copy()
        metric_rows["fold"] = numeric(metric_rows["fold"]).astype(int)
        metric_rows = metric_rows.sort_values("fold").drop_duplicates("fold", keep="last")

    candidate_folds: List[int] = []
    if not metric_rows.empty:
        candidate_folds = sorted(metric_rows["fold"].astype(int).unique().tolist())
    elif paths.l12_split_manifest.is_file():
        candidate_folds = split_manifest_fold_ids(paths.l12_split_manifest)

    if not candidate_folds:
        for directory in [paths.validation_model_dir, paths.generated_model_dir]:
            for path in sorted(directory.glob("L12_composite_q50_fold*.txt")):
                token = path.stem.rsplit("fold", 1)[-1]
                if token.isdigit():
                    candidate_folds.append(int(token))
        candidate_folds = sorted(set(candidate_folds))

    rows: List[Dict[str, Any]] = []
    for fold in candidate_folds:
        metric = metric_rows[metric_rows["fold"].eq(fold)]
        item = metric.iloc[0] if not metric.empty else pd.Series(dtype=object)

        q50 = model_path_from_metric(item.get("model_path_q50"), paths.sensitivity_dir)
        q90 = model_path_from_metric(item.get("model_path_q90"), paths.sensitivity_dir)
        model_source = "sensitivity_metrics"

        if q50 is None:
            candidates = [
                paths.validation_model_dir / f"L12_composite_q50_fold{fold}.txt",
                paths.generated_model_dir / f"L12_composite_q50_fold{fold}.txt",
            ]
            q50 = next((path.resolve() for path in candidates if path.is_file()), None)
        if q90 is None:
            candidates = [
                paths.validation_model_dir / f"L12_composite_q90_fold{fold}.txt",
                paths.generated_model_dir / f"L12_composite_q90_fold{fold}.txt",
            ]
            q90 = next((path.resolve() for path in candidates if path.is_file()), None)

        if q50 is not None and paths.generated_model_dir in q50.parents:
            model_source = "candidate_stability_auto_trained"
        elif q50 is None or q90 is None:
            model_source = "missing"

        rows.append(
            {
                "fold": int(fold),
                "q50_model_path": str(q50) if q50 is not None else "",
                "q90_model_path": str(q90) if q90 is not None else "",
                "best_iteration_q50": item.get("best_iteration_q50", np.nan),
                "best_iteration_q90": item.get("best_iteration_q90", np.nan),
                "q50_exists": bool(q50 is not None and q50.is_file()),
                "q90_exists": bool(q90 is not None and q90.is_file()),
                "q50_sha256": sha256_file(q50) if q50 is not None and q50.is_file() else "",
                "q90_sha256": sha256_file(q90) if q90 is not None and q90.is_file() else "",
                "model_source": model_source,
            }
        )

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def load_sensitivity_module(path: Path) -> ModuleType:
    ensure_file(path, "Upstream sensitivity script")
    spec = importlib.util.spec_from_file_location("cnt_composite_sensitivity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream sensitivity script:\n  {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train_required_fold_models(
    paths: Paths,
    min_components: int,
    feature_columns: Sequence[str],
    existing_manifest: pd.DataFrame,
    force_retrain: bool = False,
) -> pd.DataFrame:
    """
    Rebuild only q50/q90 paper-grouped fold models when the upstream run did not
    save them. Training definitions and frozen splits are taken directly from
    composite_sensitivity.py.
    """
    upstream = load_sensitivity_module(paths.sensitivity_script)
    ensure_file(paths.l12_split_manifest, "Frozen L12 split manifest")

    dataset = upstream.load_dataset(str(paths.dataset), feature_columns)
    work, X = upstream.prepare_composite_base(
        dataset,
        levels=["L1", "L2"],
        feature_columns=feature_columns,
    )
    split_manifest = upstream.load_split_manifest(
        str(paths.l12_split_manifest),
        ["L1", "L2"],
    )
    tasks = [
        task
        for task in upstream.split_tasks_from_manifest(split_manifest)
        if str(task["split_type"]) == "paper_grouped_cv"
    ]
    tasks = sorted(tasks, key=lambda item: int(item["fold"]))

    if len(tasks) != EXPECTED_FOLDS:
        raise RuntimeError(
            f"Expected {EXPECTED_FOLDS} paper-grouped tasks in L12 manifest, "
            f"found {len(tasks)}."
        )

    paths.generated_model_dir.mkdir(parents=True, exist_ok=True)
    existing_by_fold = {
        int(row["fold"]): row
        for _, row in existing_manifest.iterrows()
    }
    rows: List[Dict[str, Any]] = []

    for position, task in enumerate(tasks, start=1):
        fold = int(task["fold"])
        old = existing_by_fold.get(fold)
        old_q50 = Path(str(old.get("q50_model_path", ""))) if old is not None else Path()
        old_q90 = Path(str(old.get("q90_model_path", ""))) if old is not None else Path()
        reuse = (
            not force_retrain
            and old is not None
            and bool(old.get("q50_exists", False))
            and bool(old.get("q90_exists", False))
            and old_q50.is_file()
            and old_q90.is_file()
        )
        if reuse:
            rows.append(dict(old))
            print(f"[FOLD MODEL {position}/{len(tasks)}] fold={fold} existing models reused")
            continue

        print(f"[FOLD MODEL {position}/{len(tasks)}] fold={fold} rebuilding q50/q90")
        train_initial = upstream.ids_to_indices(
            work,
            task["train_ids"],
            min_components,
        )
        if len(train_initial) == 0:
            raise RuntimeError(f"fold={fold}: no retained outer-training rows.")

        normalizer = upstream.fit_composite_normalizer(work, train_initial)
        transformed = upstream.transform_composite(work, normalizer, min_components)
        y = pd.to_numeric(
            transformed[upstream.TARGET_NAME], errors="coerce"
        ).to_numpy(dtype=float)
        train_indices = train_initial[np.isfinite(y[train_initial])]
        n_train_papers = int(work.iloc[train_indices]["paper_id"].nunique())

        if (
            len(train_indices) < upstream.MIN_OUTER_TRAIN_SAMPLES
            or n_train_papers < upstream.MIN_OUTER_TRAIN_PAPERS
        ):
            raise RuntimeError(
                f"fold={fold}: too few valid Composite>={min_components} "
                f"training rows or papers ({len(train_indices)} rows, "
                f"{n_train_papers} papers)."
            )

        fit_indices, valid_indices, inner_method = upstream.make_inner_group_split(
            train_indices,
            work["paper_id"],
        )
        model_paths: Dict[str, Path] = {}
        best_iterations: Dict[str, int] = {}

        for name, alpha in [("q50", 0.50), ("q90", 0.90)]:
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
                params=upstream.quantile_params(alpha),
                train_set=train_set,
                num_boost_round=upstream.MAX_BOOST_ROUNDS,
                valid_sets=[train_set, valid_set],
                valid_names=["inner_fit", "inner_valid"],
                callbacks=[
                    lgb.early_stopping(
                        upstream.EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = int(
                model.best_iteration
                if model.best_iteration and model.best_iteration > 0
                else upstream.MAX_BOOST_ROUNDS
            )
            model_path = (
                paths.generated_model_dir
                / f"L12_composite_{name}_fold{fold}.txt"
            )
            model.save_model(str(model_path))
            model_paths[name] = model_path.resolve()
            best_iterations[name] = best_iteration

        rows.append(
            {
                "fold": fold,
                "q50_model_path": str(model_paths["q50"]),
                "q90_model_path": str(model_paths["q90"]),
                "best_iteration_q50": best_iterations["q50"],
                "best_iteration_q90": best_iterations["q90"],
                "q50_exists": True,
                "q90_exists": True,
                "q50_sha256": sha256_file(model_paths["q50"]),
                "q90_sha256": sha256_file(model_paths["q90"]),
                "model_source": "candidate_stability_auto_trained",
                "n_outer_train": int(len(train_indices)),
                "n_outer_train_papers": n_train_papers,
                "n_inner_fit": int(len(fit_indices)),
                "n_inner_valid": int(len(valid_indices)),
                "inner_split_method": inner_method,
                "igid_min": normalizer.get("igid_min"),
                "igid_max": normalizer.get("igid_max"),
                "bundle_log_min": normalizer.get("bundle_log_min"),
                "bundle_log_max": normalizer.get("bundle_log_max"),
            }
        )

    manifest = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    if len(manifest) != EXPECTED_FOLDS:
        raise RuntimeError(
            f"Expected {EXPECTED_FOLDS} fold model pairs, found {len(manifest)}."
        )
    if not manifest[["q50_exists", "q90_exists"]].all(axis=None):
        raise RuntimeError("One or more rebuilt fold models are missing.")
    return manifest


def ensure_fold_models(
    paths: Paths,
    min_components: int,
    feature_columns: Sequence[str],
    force_retrain: bool = False,
) -> pd.DataFrame:
    discovered = discover_fold_models(paths, min_components)
    complete = (
        len(discovered) == EXPECTED_FOLDS
        and not discovered.empty
        and discovered[["q50_exists", "q90_exists"]].all(axis=None)
    )
    if complete and not force_retrain:
        return discovered

    print(
        "[MODEL NOTICE] The sensitivity analysis did not save all q50/q90 fold "
        "models. Rebuilding the required Composite>=2 paper-grouped models "
        "from the frozen L12 manifest."
    )
    return train_required_fold_models(
        paths=paths,
        min_components=min_components,
        feature_columns=feature_columns,
        existing_manifest=discovered,
        force_retrain=force_retrain,
    )


def model_feature_order(
    model: lgb.Booster,
    schema_columns: Sequence[str],
    label: str,
) -> List[str]:
    names = [str(name) for name in model.feature_name()]
    if model.num_feature() != len(schema_columns):
        raise RuntimeError(
            f"{label} expects {model.num_feature()} features; schema has {len(schema_columns)}."
        )
    if set(names) == set(schema_columns):
        return names
    if all(name.startswith("Column_") for name in names):
        return list(schema_columns)
    raise RuntimeError(f"{label} feature names do not match feature_schema.json.")


# =============================================================================
# 5. Fold prediction, ranks and Top-K membership
# =============================================================================


def tie_inclusive_topk_ids(
    frame: pd.DataFrame,
    score_column: str,
    k: int,
    id_column: str,
) -> set[str]:
    scores = numeric(frame[score_column])
    finite = frame.loc[scores.notna(), [id_column, score_column]].copy()
    if finite.empty:
        return set()
    k_effective = max(1, min(int(k), len(finite)))
    sorted_scores = numeric(finite[score_column]).sort_values(ascending=False, kind="stable")
    threshold = float(sorted_scores.iloc[k_effective - 1])
    return set(
        finite.loc[numeric(finite[score_column]).ge(threshold), id_column].astype(str)
    )


def predict_folds(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    model_manifest: pd.DataFrame,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    base_columns = [
        column
        for column in [
            "combination_id",
            "catalyst_id",
            "template_id",
            "growth_mode",
            "growth_mode_normalized",
            *PROFILE_COLUMNS,
        ]
        if column in candidates.columns
    ]

    for _, row in model_manifest.iterrows():
        fold = int(row["fold"])
        q50_model = lgb.Booster(model_file=str(row["q50_model_path"]))
        q90_model = lgb.Booster(model_file=str(row["q90_model_path"]))
        q50_order = model_feature_order(q50_model, feature_columns, f"q50 fold {fold}")
        q90_order = model_feature_order(q90_model, feature_columns, f"q90 fold {fold}")

        q50 = np.asarray(q50_model.predict(candidates[q50_order]), dtype=float)
        q90 = np.asarray(q90_model.predict(candidates[q90_order]), dtype=float)
        if not np.isfinite(q50).all() or not np.isfinite(q90).all():
            raise RuntimeError(f"Non-finite predictions detected in fold {fold}.")

        fold_frame = candidates[base_columns].copy()
        fold_frame["fold"] = fold
        fold_frame["q50_prediction"] = q50
        fold_frame["q90_prediction"] = q90
        fold_frame["q50_rank"] = pd.Series(q50).rank(method="average", ascending=False)
        fold_frame["q90_rank"] = pd.Series(q90).rank(method="average", ascending=False)

        denominator = max(1, len(fold_frame) - 1)
        fold_frame["q50_rank_percentile"] = (fold_frame["q50_rank"] - 1.0) / denominator
        fold_frame["q90_rank_percentile"] = (fold_frame["q90_rank"] - 1.0) / denominator

        for k in TOP_K_VALUES:
            q50_ids = tie_inclusive_topk_ids(
                fold_frame, "q50_prediction", k, "combination_id"
            )
            q90_ids = tie_inclusive_topk_ids(
                fold_frame, "q90_prediction", k, "combination_id"
            )
            fold_frame[f"is_top{k}_q50"] = fold_frame["combination_id"].isin(q50_ids)
            fold_frame[f"is_top{k}_q90"] = fold_frame["combination_id"].isin(q90_ids)

        frames.append(fold_frame)
        print(
            f"[FOLD {fold}] candidates={len(fold_frame)}, "
            f"q50 Top10={int(fold_frame['is_top10_q50'].sum())}, "
            f"q90 Top10={int(fold_frame['is_top10_q90'].sum())}"
        )

    output = pd.concat(frames, ignore_index=True, sort=False)
    expected_rows = len(candidates) * len(model_manifest)
    if len(output) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} fold predictions, found {len(output)}.")
    return output


# =============================================================================
# 6. Candidate-level stability
# =============================================================================


def first_profile(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    columns = unique_names(
        column
        for column in [
            key,
            "catalyst_id",
            "template_id",
            "growth_mode",
            "growth_mode_normalized",
            *PROFILE_COLUMNS,
        ]
        if column in frame.columns
    )
    return frame[columns].drop_duplicates(key, keep="first")


def summarize_candidate_stability(fold_predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_predictions.groupby("combination_id", sort=False)
    summary = grouped.agg(
        q50_mean=("q50_prediction", "mean"),
        q50_std=("q50_prediction", "std"),
        q90_mean=("q90_prediction", "mean"),
        q90_std=("q90_prediction", "std"),
        median_q50_rank=("q50_rank", "median"),
        median_q90_rank=("q90_rank", "median"),
        median_q50_rank_percentile=("q50_rank_percentile", "median"),
        median_q90_rank_percentile=("q90_rank_percentile", "median"),
        top10_frequency_q50=("is_top10_q50", "mean"),
        top10_frequency_q90=("is_top10_q90", "mean"),
        top20_frequency_q50=("is_top20_q50", "mean"),
        top20_frequency_q90=("is_top20_q90", "mean"),
        fold_count=("fold", "nunique"),
    ).reset_index()

    iqr_rows = grouped.agg(
        q50_rank_IQR=("q50_rank", quantile_iqr),
        q90_rank_IQR=("q90_rank", quantile_iqr),
        q50_rank_percentile_IQR=("q50_rank_percentile", quantile_iqr),
        q90_rank_percentile_IQR=("q90_rank_percentile", quantile_iqr),
    ).reset_index()

    summary = summary.merge(iqr_rows, on="combination_id", how="left", validate="one_to_one")
    profile = first_profile(fold_predictions, "combination_id")
    summary = profile.merge(summary, on="combination_id", how="right", validate="one_to_one")

    if not summary["fold_count"].eq(EXPECTED_FOLDS).all():
        raise RuntimeError("Some candidates do not have predictions from all five folds.")

    return summary.sort_values(
        ["median_q50_rank", "median_q90_rank", "q50_rank_IQR"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# 7. Catalyst-level fold ranking and stability
# =============================================================================


def catalyst_profile(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = unique_names(
        column for column in ["catalyst_id", *PROFILE_COLUMNS] if column in candidates
    )
    return candidates[columns].drop_duplicates("catalyst_id", keep="first")


def build_catalyst_fold_rankings(
    fold_predictions: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for fold, fold_frame in fold_predictions.groupby("fold", sort=True):
        profile = catalyst_profile(fold_frame)
        q50_ordered = fold_frame.sort_values(
            ["catalyst_id", "q50_prediction", "template_id"],
            ascending=[True, False, True],
            kind="stable",
        ).copy()
        q90_ordered = fold_frame.sort_values(
            ["catalyst_id", "q90_prediction", "template_id"],
            ascending=[True, False, True],
            kind="stable",
        ).copy()

        q50_ordered["_route_rank"] = q50_ordered.groupby("catalyst_id").cumcount() + 1
        q90_ordered["_route_rank"] = q90_ordered.groupby("catalyst_id").cumcount() + 1

        q50_scores = (
            q50_ordered[q50_ordered["_route_rank"].le(top_n)]
            .groupby("catalyst_id", as_index=False)["q50_prediction"]
            .mean()
            .rename(columns={"q50_prediction": "q50_catalyst_score"})
        )
        q90_scores = (
            q90_ordered[q90_ordered["_route_rank"].le(top_n)]
            .groupby("catalyst_id", as_index=False)["q90_prediction"]
            .mean()
            .rename(columns={"q90_prediction": "q90_catalyst_score"})
        )

        result = q50_scores.merge(q90_scores, on="catalyst_id", how="outer")
        result["fold"] = int(fold)
        result["q50_catalyst_rank"] = result["q50_catalyst_score"].rank(
            method="average", ascending=False
        )
        result["q90_catalyst_rank"] = result["q90_catalyst_score"].rank(
            method="average", ascending=False
        )

        catalyst_count = len(result)
        denominator = max(1, catalyst_count - 1)
        result["q50_catalyst_rank_percentile"] = (
            result["q50_catalyst_rank"] - 1.0
        ) / denominator
        result["q90_catalyst_rank_percentile"] = (
            result["q90_catalyst_rank"] - 1.0
        ) / denominator

        q50_top5 = tie_inclusive_topk_ids(
            result, "q50_catalyst_score", min(5, catalyst_count), "catalyst_id"
        )
        q90_top5 = tie_inclusive_topk_ids(
            result, "q90_catalyst_score", min(5, catalyst_count), "catalyst_id"
        )
        result["is_top5_catalyst_q50"] = result["catalyst_id"].isin(q50_top5)
        result["is_top5_catalyst_q90"] = result["catalyst_id"].isin(q90_top5)
        result = result.merge(profile, on="catalyst_id", how="left")
        rows.append(result)

    return pd.concat(rows, ignore_index=True, sort=False)


def summarize_catalyst_stability(catalyst_folds: pd.DataFrame) -> pd.DataFrame:
    grouped = catalyst_folds.groupby("catalyst_id", sort=False)
    summary = grouped.agg(
        q50_catalyst_score_mean=("q50_catalyst_score", "mean"),
        q50_catalyst_score_std=("q50_catalyst_score", "std"),
        q90_catalyst_score_mean=("q90_catalyst_score", "mean"),
        q90_catalyst_score_std=("q90_catalyst_score", "std"),
        median_q50_catalyst_rank=("q50_catalyst_rank", "median"),
        median_q90_catalyst_rank=("q90_catalyst_rank", "median"),
        median_q50_catalyst_rank_percentile=("q50_catalyst_rank_percentile", "median"),
        median_q90_catalyst_rank_percentile=("q90_catalyst_rank_percentile", "median"),
        top5_catalyst_frequency_q50=("is_top5_catalyst_q50", "mean"),
        top5_catalyst_frequency_q90=("is_top5_catalyst_q90", "mean"),
        fold_count=("fold", "nunique"),
    ).reset_index()

    iqrs = grouped.agg(
        q50_catalyst_rank_IQR=("q50_catalyst_rank", quantile_iqr),
        q90_catalyst_rank_IQR=("q90_catalyst_rank", quantile_iqr),
        q50_catalyst_rank_percentile_IQR=("q50_catalyst_rank_percentile", quantile_iqr),
        q90_catalyst_rank_percentile_IQR=("q90_catalyst_rank_percentile", quantile_iqr),
    ).reset_index()
    summary = summary.merge(iqrs, on="catalyst_id", how="left", validate="one_to_one")
    profile = catalyst_profile(catalyst_folds)
    summary = profile.merge(summary, on="catalyst_id", how="right", validate="one_to_one")
    return summary.sort_values(
        ["median_q50_catalyst_rank", "median_q90_catalyst_rank"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# 8. Top-K Jaccard
# =============================================================================


def compute_topk_jaccard(fold_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    folds = sorted(fold_predictions["fold"].astype(int).unique())

    for target, score_column in [("q50", "q50_prediction"), ("q90", "q90_prediction")]:
        for k in TOP_K_VALUES:
            sets: Dict[int, set[str]] = {}
            for fold in folds:
                group = fold_predictions[fold_predictions["fold"].eq(fold)]
                sets[fold] = tie_inclusive_topk_ids(
                    group, score_column, k, "combination_id"
                )

            for fold_a, fold_b in combinations(folds, 2):
                set_a = sets[fold_a]
                set_b = sets[fold_b]
                intersection = set_a & set_b
                union = set_a | set_b
                rows.append(
                    {
                        "target": target,
                        "k": int(k),
                        "fold_a": int(fold_a),
                        "fold_b": int(fold_b),
                        "topk_size_a": int(len(set_a)),
                        "topk_size_b": int(len(set_b)),
                        "intersection_size": int(len(intersection)),
                        "union_size": int(len(union)),
                        "jaccard": float(len(intersection) / len(union)) if union else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def summarize_jaccard(jaccard: pd.DataFrame) -> pd.DataFrame:
    return (
        jaccard.groupby(["target", "k"], as_index=False)["jaccard"]
        .agg(
            mean_jaccard="mean",
            median_jaccard="median",
            minimum_jaccard="min",
            maximum_jaccard="max",
            pair_count="count",
        )
    )


# =============================================================================
# 9. L123 candidate domain support
# =============================================================================


def component_count(frame: pd.DataFrame) -> pd.Series:
    igid = (
        numeric(frame[PREFERRED_IGID_COL])
        if PREFERRED_IGID_COL in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    if LEGACY_IGID_COL in frame.columns:
        igid = igid.where(igid.notna(), numeric(frame[LEGACY_IGID_COL]))

    columns = {
        "gq": numeric(frame[COL_GQ]) if COL_GQ in frame else pd.Series(np.nan, index=frame.index),
        "igid": igid,
        "ori": numeric(frame[COL_ORI]) if COL_ORI in frame else pd.Series(np.nan, index=frame.index),
        "purity": numeric(frame[COL_PURITY]) if COL_PURITY in frame else pd.Series(np.nan, index=frame.index),
        "bundle": numeric(frame[COL_BUNDLE]) if COL_BUNDLE in frame else pd.Series(np.nan, index=frame.index),
    }
    return pd.DataFrame(columns).notna().sum(axis=1).astype(int)


def prepare_support_reference(dataset_path: Path) -> pd.DataFrame:
    data = read_csv_smart(dataset_path).copy()
    require_columns(
        data,
        ["sample_id", "paper_id", "data_level", "growth_mode", *MODEL_COLS],
        "Support dataset",
    )
    data["data_level"] = data["data_level"].fillna("").astype(str).str.upper()
    data = data[data["data_level"].isin(["L1", "L2"])].copy()
    data["composite_component_count"] = component_count(data)
    data = data[data["composite_component_count"].ge(MIN_COMPONENT_COUNT)].copy()
    data["sample_id"] = data["sample_id"].fillna("").astype(str)
    data["paper_id"] = data["paper_id"].fillna("").astype(str)
    data["growth_mode_normalized"] = data["growth_mode"].map(normalize_mode)

    for column in MODEL_COLS:
        data[column] = numeric(data[column])
    for feature in PHYSICAL_COLS:
        miss = f"miss_{feature}"
        data[miss] = data[miss].fillna(1).gt(0).astype(int)
        data.loc[data[feature].isna(), miss] = 1
        data.loc[data[miss].eq(1), feature] = 0.0
        data[feature] = data[feature].fillna(0.0)
    return data.reset_index(drop=True)


def restore_observed_physical(frame: pd.DataFrame) -> pd.DataFrame:
    physical = frame[PHYSICAL_COLS].apply(pd.to_numeric, errors="coerce").copy()
    for feature in PHYSICAL_COLS:
        miss = numeric(frame[f"miss_{feature}"]).fillna(1).gt(0)
        physical.loc[miss, feature] = np.nan
    return physical


@dataclass
class SupportModel:
    mode: str
    feature_columns: List[str]
    median: pd.Series
    scale: pd.Series
    reference_scaled: np.ndarray
    reference_rows: pd.DataFrame
    nn_model: NearestNeighbors
    train_nn_distances: np.ndarray
    p95: float
    p99: float
    missing_matrix: np.ndarray
    missing_nn_model: NearestNeighbors


def fit_support_model(reference: pd.DataFrame, mode: str) -> SupportModel:
    raw = restore_observed_physical(reference)
    observed_count = raw.notna().sum(axis=0)
    observed_values_unique = raw.nunique(dropna=True)
    usable = [
        column
        for column in PHYSICAL_COLS
        if observed_count[column] >= 2 and observed_values_unique[column] >= 2
    ]
    if not usable:
        raise RuntimeError(f"No usable support features for growth mode {mode}.")

    raw = raw[usable]
    median = raw.median(skipna=True).fillna(0.0)
    iqr = raw.quantile(0.75) - raw.quantile(0.25)
    std = raw.std(ddof=0)
    scale = iqr.where(iqr.abs().gt(1e-12), std)
    scale = scale.where(scale.abs().gt(1e-12), 1.0).fillna(1.0)
    filled = raw.fillna(median)
    scaled = ((filled - median) / scale).to_numpy(dtype=float)

    # Remove duplicate physical vectors for the leave-one-state-out threshold.
    unique_scaled, unique_indices = np.unique(scaled, axis=0, return_index=True)
    unique_rows = reference.iloc[unique_indices].reset_index(drop=True)
    if len(unique_scaled) < 2:
        raise RuntimeError(f"Too few unique support states for growth mode {mode}.")

    threshold_nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    threshold_nn.fit(unique_scaled)
    threshold_distances, _ = threshold_nn.kneighbors(unique_scaled)
    train_nn = threshold_distances[:, 1]
    p95 = float(np.quantile(train_nn, 0.95))
    p99 = float(np.quantile(train_nn, 0.99))

    candidate_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    candidate_nn.fit(unique_scaled)

    missing = (
        reference.iloc[unique_indices][MISS_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(1)
        .gt(0)
        .astype(float)
        .to_numpy()
    )
    missing_nn = NearestNeighbors(n_neighbors=1, metric="hamming")
    missing_nn.fit(missing)

    return SupportModel(
        mode=mode,
        feature_columns=usable,
        median=median,
        scale=scale,
        reference_scaled=unique_scaled,
        reference_rows=unique_rows,
        nn_model=candidate_nn,
        train_nn_distances=train_nn,
        p95=p95,
        p99=p99,
        missing_matrix=missing,
        missing_nn_model=missing_nn,
    )


def fit_support_models(reference: pd.DataFrame) -> Tuple[Dict[str, SupportModel], pd.DataFrame]:
    models: Dict[str, SupportModel] = {}
    rows: List[Dict[str, Any]] = []

    global_model = fit_support_model(reference, "ALL")
    models["ALL"] = global_model

    for mode in sorted(reference["growth_mode_normalized"].unique()):
        subset = reference[reference["growth_mode_normalized"].eq(mode)].copy()
        if len(subset) < 20:
            continue
        try:
            models[mode] = fit_support_model(subset, mode)
        except RuntimeError:
            continue

    for mode, model in models.items():
        rows.append(
            {
                "growth_mode": mode,
                "reference_rows": len(model.reference_rows),
                "usable_physical_features": len(model.feature_columns),
                "train_nn_p50": float(np.quantile(model.train_nn_distances, 0.50)),
                "train_nn_p95": model.p95,
                "train_nn_p99": model.p99,
            }
        )
    return models, pd.DataFrame(rows)


def empirical_percentile(reference_distances: np.ndarray, value: float) -> float:
    return float(100.0 * np.mean(reference_distances <= value))


def add_domain_support(
    candidates: pd.DataFrame,
    models: Mapping[str, SupportModel],
) -> pd.DataFrame:
    output = candidates[["combination_id", "growth_mode_normalized", *MODEL_COLS]].copy()
    records: List[pd.DataFrame] = []

    for mode, group in output.groupby("growth_mode_normalized", sort=False):
        model = models.get(mode, models["ALL"])
        raw = restore_observed_physical(group)
        filled = raw[model.feature_columns].fillna(model.median)
        scaled = ((filled - model.median) / model.scale).to_numpy(dtype=float)
        distances, indices = model.nn_model.kneighbors(scaled)
        nearest = model.reference_rows.iloc[indices[:, 0]].reset_index(drop=True)

        missing = (
            group[MISS_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(1)
            .gt(0)
            .astype(float)
            .to_numpy()
        )
        missing_distances, _ = model.missing_nn_model.kneighbors(missing)

        block = group[["combination_id"]].copy().reset_index(drop=True)
        block["support_reference_mode"] = model.mode
        block["support_distance_full"] = distances[:, 0]
        block["support_percentile_full"] = [
            empirical_percentile(model.train_nn_distances, value)
            for value in distances[:, 0]
        ]
        block["domain_status"] = np.select(
            [
                block["support_distance_full"].le(model.p95),
                block["support_distance_full"].le(model.p99),
            ],
            ["supported", "near-boundary"],
            default="out-of-domain",
        )
        block["nearest_training_sample"] = nearest["sample_id"].astype(str).to_numpy()
        block["nearest_training_paper"] = nearest["paper_id"].astype(str).to_numpy()
        block["missingness_distance"] = missing_distances[:, 0]
        records.append(block)

    result = pd.concat(records, ignore_index=True, sort=False)
    if result["combination_id"].duplicated().any():
        raise RuntimeError("Domain support produced duplicate combination_id values.")
    return result


# =============================================================================
# 10. Boundary merge and final classification
# =============================================================================


def load_boundary_status(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["combination_id", "q50_search_status", "q90_search_status"])
    frame = read_csv_smart(path)
    require_columns(frame, ["combination_id"], "Physical-state boundary table")
    columns = [
        column
        for column in [
            "combination_id",
            "q50_search_status",
            "q90_search_status",
            "recommended_np_size_nm_balanced",
            "recommended_C_over_M_balanced",
            "recommended_np_size_nm_potential",
            "recommended_C_over_M_potential",
        ]
        if column in frame.columns
    ]
    return frame[columns].drop_duplicates("combination_id", keep="first")


def classify_final_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["domain_status"] = output["domain_status"].fillna("not_available")
    output["q50_search_status"] = output.get(
        "q50_search_status", pd.Series("not_evaluated", index=output.index)
    ).fillna("not_evaluated")
    output["q90_search_status"] = output.get(
        "q90_search_status", pd.Series("not_evaluated", index=output.index)
    ).fillna("not_evaluated")

    output["boundary_limited_q50"] = ~output["q50_search_status"].isin(
        ["interior_grid_maximum", "not_evaluated"]
    )
    output["boundary_limited_q90"] = ~output["q90_search_status"].isin(
        ["interior_grid_maximum", "not_evaluated"]
    )
    output["any_boundary_limited"] = (
        output["boundary_limited_q50"] | output["boundary_limited_q90"]
    )

    domain_ok = ~output["domain_status"].eq("out-of-domain")
    stable_q50 = output["top20_frequency_q50"].ge(0.50)
    stable_q90 = output["top20_frequency_q90"].ge(0.50)
    high_consensus = (
        domain_ok
        & output["top10_frequency_q50"].ge(0.60)
        & output["top10_frequency_q90"].ge(0.60)
    )

    output["is_formal_stable_candidate"] = domain_ok & (stable_q50 | stable_q90)
    output["is_high_consensus_candidate"] = high_consensus

    conditions = [
        output["domain_status"].eq("out-of-domain"),
        high_consensus & output["any_boundary_limited"],
        high_consensus,
        domain_ok & stable_q50 & stable_q90 & output["any_boundary_limited"],
        domain_ok & stable_q50 & stable_q90,
        domain_ok & stable_q50 & output["any_boundary_limited"],
        domain_ok & stable_q90 & output["any_boundary_limited"],
        domain_ok & stable_q50,
        domain_ok & stable_q90,
        output["domain_status"].eq("near-boundary"),
    ]
    labels = [
        "out_of_domain_excluded",
        "high_consensus_boundary_limited",
        "high_consensus",
        "stable_both_boundary_limited",
        "stable_both",
        "stable_balanced_boundary_limited",
        "stable_potential_boundary_limited",
        "stable_balanced",
        "stable_potential",
        "near_boundary_exploratory",
    ]
    output["stability_class"] = np.select(
        conditions, labels, default="insufficient_consensus"
    )

    output["selection_reason"] = np.select(
        [
            high_consensus,
            stable_q50 & stable_q90,
            stable_q50,
            stable_q90,
        ],
        [
            "Top10 in at least 3/5 folds for both q50 and q90",
            "Top20 in at least 3/5 folds for both q50 and q90",
            "Top20 in at least 3/5 folds for q50",
            "Top20 in at least 3/5 folds for q90",
        ],
        default="Did not meet repeated Top-K criterion",
    )

    priority = {
        "high_consensus": 1,
        "high_consensus_boundary_limited": 2,
        "stable_both": 3,
        "stable_both_boundary_limited": 4,
        "stable_balanced": 5,
        "stable_potential": 6,
        "stable_balanced_boundary_limited": 7,
        "stable_potential_boundary_limited": 8,
        "near_boundary_exploratory": 9,
        "insufficient_consensus": 10,
        "out_of_domain_excluded": 11,
    }
    output["_class_priority"] = output["stability_class"].map(priority).fillna(99)
    output["_best_top10_frequency"] = output[
        ["top10_frequency_q50", "top10_frequency_q90"]
    ].max(axis=1)
    output["_best_median_rank"] = output[
        ["median_q50_rank", "median_q90_rank"]
    ].min(axis=1)

    output = output.sort_values(
        ["_class_priority", "_best_top10_frequency", "_best_median_rank", "q50_mean", "q90_mean"],
        ascending=[True, False, True, False, False],
        kind="stable",
    ).drop(columns=["_class_priority", "_best_top10_frequency", "_best_median_rank"])
    return output.reset_index(drop=True)


# =============================================================================
# 11. Figures
# =============================================================================


def apply_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    axis.grid(alpha=0.25, linewidth=0.5)


def plot_candidate_stability(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), layout="constrained")

    domain_markers = {
        "supported": "o",
        "near-boundary": "o",
        "out-of-domain": "x",
        "not_available": "+",
    }

    for axis, target in zip(axes, ["q50", "q90"]):
        for status, marker in domain_markers.items():
            subset = frame[frame["domain_status"].eq(status)]
            if subset.empty:
                continue
            axis.scatter(
                subset[f"median_{target}_rank"],
                subset[f"{target}_rank_IQR"],
                s=35 + 170 * subset[f"top20_frequency_{target}"],
                marker=marker,
                alpha=0.75,
                label=status,
            )

        label_rows = frame[
            frame["is_high_consensus_candidate"]
            | frame["is_formal_stable_candidate"]
        ].head(12)
        for _, row in label_rows.iterrows():
            axis.annotate(
                str(row["combination_id"]),
                (row[f"median_{target}_rank"], row[f"{target}_rank_IQR"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        axis.set_xlabel(f"Median {target} rank across folds")
        axis.set_ylabel(f"{target} rank IQR")
        axis.set_title(f"{target} candidate ranking stability")
        axis.invert_xaxis()
        apply_axes(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_heatmaps(
    jaccard: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    folds = sorted(set(jaccard["fold_a"]) | set(jaccard["fold_b"]))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), layout="constrained")

    panels = [("q50", 10), ("q50", 20), ("q90", 10), ("q90", 20)]
    for axis, (target, k) in zip(axes.ravel(), panels):
        matrix = pd.DataFrame(np.eye(len(folds)), index=folds, columns=folds)
        subset = jaccard[(jaccard["target"].eq(target)) & (jaccard["k"].eq(k))]
        for _, row in subset.iterrows():
            a, b = int(row["fold_a"]), int(row["fold_b"])
            matrix.loc[a, b] = row["jaccard"]
            matrix.loc[b, a] = row["jaccard"]

        image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(folds)), labels=folds)
        axis.set_yticks(range(len(folds)), labels=folds)
        axis.set_xlabel("Fold")
        axis.set_ylabel("Fold")
        axis.set_title(f"{target} Top-{k} Jaccard")
        for i in range(len(folds)):
            for j in range(len(folds)):
                axis.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Jaccard index")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 12. Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CNT 3.0 candidate ranking stability across paper-grouped fold "
            "models. Without a completed candidate pool, the default behavior "
            "is to write a SKIPPED summary and exit normally."
        )
    )
    parser.add_argument("--project_dir", default=str(PROJECT_DIR))
    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
        help=(
            "rank_candidate_routes/all_route_predictions.csv or 'auto'. Auto uses the "
            "formal current-project output unless --include_latest_app is set."
        ),
    )
    parser.add_argument(
        "--l123_run",
        default="",
        help=(
            "Explicit current-project or app run root containing rank_candidate_routes "
            "and optionally optimize_physical_state. Takes precedence over --candidates."
        ),
    )
    parser.add_argument(
        "--include_latest_app",
        action="store_true",
        help=(
            "When --candidates auto finds no formal pool, use the newest candidate "
            "pool found under outputs/app."
        ),
    )
    parser.add_argument(
        "--require_candidates",
        action="store_true",
        help=(
            "Raise FileNotFoundError when auto mode finds no candidate pool. "
            "Without this flag, the analysis is marked SKIPPED and exits 0."
        ),
    )
    parser.add_argument("--sensitivity_dir", default=str(DEFAULT_SENSITIVITY_DIR))
    parser.add_argument("--sensitivity_script", default=str(DEFAULT_SENSITIVITY_SCRIPT))
    parser.add_argument("--l12_manifest", default=str(DEFAULT_L12_SPLIT_MANIFEST))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument(
        "--boundary_csv",
        default=DEFAULT_BOUNDARY,
        help=(
            "Matching optimize_physical_state/process_packages_with_physical_state.csv or "
            "'auto'. Missing boundary data do not block the stability run."
        ),
    )
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--min_components", type=int, default=MIN_COMPONENT_COUNT)
    parser.add_argument("--catalyst_top_n", type=int, default=CATALYST_AGGREGATE_TOP_N)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--retrain_fold_models",
        action="store_true",
        help="Force rebuilding the five q50/q90 paper-grouped fold models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_components != 2:
        raise ValueError("Formal candidate stability is fixed to min_components=2.")
    if args.catalyst_top_n <= 0:
        raise ValueError("--catalyst_top_n must be positive.")

    project_dir = Path(args.project_dir).expanduser().resolve()
    resolution = resolve_candidate_pool(args, project_dir)

    if resolution.candidates is None:
        # build_paths still needs concrete placeholders, but none of these
        # candidate-specific paths are opened in the skipped branch.
        args.candidates = str(
            project_dir
            / "outputs"
            / "recommendation"
            / "rank_candidate_routes"
            / "all_route_predictions.csv"
        )
        args.boundary_csv = str(
            project_dir
            / "outputs"
            / "recommendation"
            / "optimize_physical_state"
            / "process_packages_with_physical_state.csv"
        )
        paths = build_paths(args)

        if args.require_candidates:
            searched = "\n".join(
                f"  - {path}" for path in resolution.searched_locations
            )
            raise FileNotFoundError(
                "No completed route-level candidate pool was found.\n"
                "Searched locations:\n"
                f"{searched or '  - none'}"
            )

        write_skipped_candidate_analysis(paths, resolution)
        return

    args.candidates = str(resolution.candidates)
    args.boundary_csv = str(
        resolution.boundary
        if resolution.boundary is not None
        else infer_boundary_from_candidates(resolution.candidates)
    )
    paths = build_paths(args)
    paths.outdir.mkdir(parents=True, exist_ok=True)

    ensure_file(paths.candidates, "Route-level candidate pool")
    ensure_file(paths.dataset, "CNT dataset")
    ensure_file(paths.schema, "Feature schema")
    ensure_file(paths.sensitivity_script, "Upstream sensitivity script")
    ensure_file(paths.l12_split_manifest, "Frozen L12 split manifest")

    feature_columns = load_schema(paths.schema)
    candidates = prepare_candidate_pool(paths.candidates, feature_columns)
    model_manifest = ensure_fold_models(
        paths,
        args.min_components,
        feature_columns,
        force_retrain=args.retrain_fold_models,
    )
    write_csv(model_manifest, paths.model_manifest)

    print("=" * 108)
    print("[CANDIDATE STABILITY START]")
    print(f"Project directory      : {paths.project_dir}")
    print(f"Candidate pool         : {paths.candidates}")
    print(f"Candidate source       : {resolution.source}")
    print(
        "Boundary table        : "
        f"{paths.boundary if paths.boundary.is_file() else 'NOT AVAILABLE'}"
    )
    print(f"Candidate rows         : {len(candidates)}")
    print(f"Unique catalysts       : {candidates['catalyst_id'].nunique()}")
    print(f"Unique routes          : {candidates['template_id'].nunique()}")
    print(f"Feature count          : {len(feature_columns)}")
    print(f"Fold models            : {len(model_manifest)} q50 + {len(model_manifest)} q90")
    print(f"Sensitivity directory  : {paths.sensitivity_dir}")
    print(f"Output                  : {paths.outdir}")
    print("=" * 108)

    fold_predictions = predict_folds(candidates, feature_columns, model_manifest)
    write_csv(fold_predictions, paths.fold_predictions)

    candidate_stability = summarize_candidate_stability(fold_predictions)

    catalyst_folds = build_catalyst_fold_rankings(
        fold_predictions, top_n=args.catalyst_top_n
    )
    catalyst_stability = summarize_catalyst_stability(catalyst_folds)
    write_csv(catalyst_folds, paths.catalyst_fold_rankings)
    write_csv(catalyst_stability, paths.catalyst_stability)

    jaccard = compute_topk_jaccard(fold_predictions)
    jaccard_summary = summarize_jaccard(jaccard)
    write_csv(jaccard, paths.jaccard)
    write_csv(jaccard_summary, paths.jaccard_summary)

    support_reference = prepare_support_reference(paths.dataset)
    support_models, threshold_table = fit_support_models(support_reference)
    write_csv(threshold_table, paths.domain_thresholds)
    domain = add_domain_support(candidates, support_models)

    boundary = load_boundary_status(paths.boundary)
    combined = candidate_stability.merge(
        domain, on="combination_id", how="left", validate="one_to_one"
    )
    if not boundary.empty:
        combined = combined.merge(
            boundary, on="combination_id", how="left", validate="one_to_one"
        )

    classified_candidates = classify_final_candidates(combined)
    final_candidates = classified_candidates[
        classified_candidates["is_formal_stable_candidate"]
    ].copy().reset_index(drop=True)
    write_csv(classified_candidates, paths.candidate_stability)
    write_csv(final_candidates, paths.final_candidates)

    plot_candidate_stability(classified_candidates, paths.fig_stability, args.dpi)
    plot_jaccard_heatmaps(jaccard, paths.fig_jaccard, args.dpi)

    class_counts = classified_candidates["stability_class"].value_counts().to_dict()
    summary = {
        "script_revision": SCRIPT_REVISION,
        "analysis": "CNT 3.0 candidate ranking stability",
        "candidate_source_type": resolution.source,
        "candidate_source": str(paths.candidates),
        "candidate_count": int(len(candidates)),
        "catalyst_count": int(candidates["catalyst_id"].nunique()),
        "route_count": int(candidates["template_id"].nunique()),
        "model_source": str(paths.sensitivity_dir),
        "upstream_sensitivity_script": str(paths.sensitivity_script),
        "frozen_l12_split_manifest": str(paths.l12_split_manifest),
        "fold_model_sources": model_manifest["model_source"].astype(str).value_counts().to_dict(),
        "folds": model_manifest["fold"].astype(int).tolist(),
        "minimum_component_count": int(args.min_components),
        "candidate_ranking": {
            "targets": ["raw q50", "raw q90"],
            "rank_ties": "average rank",
            "topk_ties": "all candidates tied at the Kth score are included",
            "topk_values": list(TOP_K_VALUES),
            "formal_rule": (
                "domain_status != out-of-domain and Top20 frequency >= 0.50 "
                "for q50 or q90; with five folds this means at least 3/5 folds"
            ),
            "high_consensus_rule": (
                "Top10 frequency >= 0.60 for both q50 and q90"
            ),
        },
        "catalyst_ranking": {
            "aggregate_top_n_routes": int(args.catalyst_top_n),
            "score": "mean of top-N route predictions within each fold",
        },
        "domain_support": {
            "reference_rows": int(len(support_reference)),
            "reference_levels": ["L1", "L2"],
            "minimum_composite_components": MIN_COMPONENT_COUNT,
            "physical_features": len(PHYSICAL_COLS),
            "thresholds": "mode-specific leave-one-state-out nearest-neighbor p95/p99",
            "missingness": "separate 60D Hamming nearest-neighbor distance",
        },
        "boundary_source": str(paths.boundary) if paths.boundary.is_file() else "not_available",
        "stability_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "scientific_boundary": (
            "Fold-wise dispersion measures sensitivity to paper-level training-set "
            "composition. It is not a calibrated prediction confidence interval and "
            "does not replace prospective experimental validation."
        ),
        "outputs": {
            "candidate_fold_predictions": str(paths.fold_predictions),
            "candidate_rank_stability": str(paths.candidate_stability),
            "catalyst_fold_rankings": str(paths.catalyst_fold_rankings),
            "catalyst_rank_stability": str(paths.catalyst_stability),
            "topk_jaccard": str(paths.jaccard),
            "topk_jaccard_summary": str(paths.jaccard_summary),
            "fold_model_manifest": str(paths.model_manifest),
            "domain_thresholds": str(paths.domain_thresholds),
            "final_stable_candidates": str(paths.final_candidates),
            "fig_candidate_rank_stability": str(paths.fig_stability),
            "fig_topk_jaccard": str(paths.fig_jaccard),
        },
    }
    write_json(summary, paths.summary)

    print("\n[RESULT SUMMARY]")
    print(f"Candidate fold predictions : {len(fold_predictions)}")
    print(f"Candidate stability rows   : {len(candidate_stability)}")
    print(f"Catalyst stability rows    : {len(catalyst_stability)}")
    print(f"Top-K Jaccard rows         : {len(jaccard)}")
    print(
        "Domain status              : "
        + ", ".join(
            f"{key}={value}"
            for key, value in classified_candidates["domain_status"].value_counts().to_dict().items()
        )
    )
    print(
        "Formal stable candidates  : "
        f"{len(final_candidates)}"
    )
    print(
        "High-consensus candidates : "
        f"{int(classified_candidates['is_high_consensus_candidate'].sum())}"
    )
    print(f"Output directory           : {paths.outdir}")
    print(f"Final stable candidates    : {paths.final_candidates}")
    print(f"Script revision            : {SCRIPT_REVISION}")
    print("\n[OK] Candidate stability completed.")
    print("=" * 108)


if __name__ == "__main__":
    main()
