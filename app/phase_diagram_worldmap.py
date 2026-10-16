# -*- coding: utf-8 -*-
"""
phase_diagram_worldmap.py

CNT 3.0 WorldMap

Recombine literature-observed catalyst base states with literature-observed
route states (F+P), rebuild the route-dependent N/C descriptors, predict L1
and q10/q50/q90 scores, and audit whether the resulting 60D physical state
remains close to the actual L1+L2 Composite>=2 training domain.

Key additions
-------------
1. N+C, F+P and full-60D physical nearest-training distances.
2. Missingness-pattern Hamming distance kept separate from physical distance.
3. Data-derived supported / near-boundary / out-of-domain thresholds from the
   leave-one-out nearest-neighbour distance distribution of training states.
4. Catalyst and route PCA fitted on unique observed states, not repeated
   candidate rows.
5. Aggregated catalyst-state and route-state tables to remove coordinate
   over-plotting.
6. Balanced = raw q50, Potential = raw q90, Conservative = raw q10.

The support labels are empirical training-domain labels. They are not
probabilities of correctness and are not used to delete or penalize candidates.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET = "data/11_flattened/ml_dataset_3p0_full60D.csv"
OUT_BASE = "outputs/app/phase_diagram_worldmap"

L1_MODEL = (
    "outputs/models/l1_composite/"
    "final_models/L1_composite_final_score.txt"
)
Q10_MODEL = (
    "outputs/models/l12_quantile/"
    "final_models/L12_composite_q10.txt"
)
Q50_MODEL = (
    "outputs/models/l12_quantile/"
    "final_models/L12_composite_q50.txt"
)
Q90_MODEL = (
    "outputs/models/l12_quantile/"
    "final_models/L12_composite_q90.txt"
)

N_COLS = [f"N{i}" for i in range(1, 19)]
C_COLS = [f"C{i}" for i in range(1, 13)]
F_COLS = [f"F{i}" for i in range(1, 19)]
P_COLS = [f"P{i}" for i in range(1, 13)]

PHYSICAL_COLS = N_COLS + C_COLS + F_COLS + P_COLS
MISS_COLS = [f"miss_{column}" for column in PHYSICAL_COLS]
MODEL_FEATURES = PHYSICAL_COLS + MISS_COLS

NC_COLS = N_COLS + C_COLS
FP_COLS = F_COLS + P_COLS

# Only these N/C descriptors change when the same catalyst state is placed on a
# different literature-observed route. The remaining N/C descriptors define the
# route-independent catalyst/carbon base state used by the WorldMap.
ROUTE_DEPENDENT_NC_COLS = ["N9", "C1", "C2", "C5", "C6", "C7", "C9"]
N_BASE_COLS = [column for column in N_COLS if column not in ROUTE_DEPENDENT_NC_COLS]
C_BASE_COLS = [column for column in C_COLS if column not in ROUTE_DEPENDENT_NC_COLS]
NC_BASE_COLS = N_BASE_COLS + C_BASE_COLS

NC_STATE_COLS = NC_COLS + [f"miss_{column}" for column in NC_COLS]
NC_BASE_STATE_COLS = NC_BASE_COLS + [f"miss_{column}" for column in NC_BASE_COLS]
FP_STATE_COLS = FP_COLS + [f"miss_{column}" for column in FP_COLS]

MIN_COMPONENT_COUNT = 2
SUPPORT_P95 = 0.95
SUPPORT_P99 = 0.99
ROUND_STATE_DECIMALS = 8
ROUND_DISTANCE_DECIMALS = 12

PREFERRED_IGID_COL = "y_ig_id_ratio"
LEGACY_IGID_COL = "y_IG_ID"
COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

META_COLS = [
    "paper_id",
    "paper_doi",
    "paper_title",
    "paper_year_int",
    "paper_year",
    "paper_journal",
    "data_level",
    "sample_id",
    "synthesis_id",
    "source_file",
]

DOMAIN_ORDER = ["supported", "near-boundary", "out-of-domain"]
DOMAIN_LABELS = {
    "supported": "Supported",
    "near-boundary": "Near boundary",
    "out-of-domain": "Out of domain",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(project_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def read_csv_smart(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV:\n  {path}")

    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read CSV:\n  {path}")


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mode_number(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan

    token = str(value).strip().upper()
    direct = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    }
    if token in direct:
        return float(direct[token])

    for letter, number in zip("ABCDE", range(1, 6)):
        if f"MODE {letter}" in token or f"MODE_{letter}" in token:
            return float(number)
    return np.nan


def mode_letter(value: int) -> str:
    return {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(int(value), "?")


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{label} is missing columns:\n  {missing}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def copy_figure(source: Path, targets: Sequence[Path]) -> None:
    for target in targets:
        if target != source:
            shutil.copyfile(source, target)


# =============================================================================
# DATA PREPARATION
# =============================================================================


def composite_component_count(df: pd.DataFrame) -> pd.Series:
    required = [COL_GQ, COL_ORI, COL_PURITY, COL_BUNDLE]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(
            "Cannot reproduce the Composite>=2 training reference because the "
            f"dataset is missing target columns: {missing}"
        )

    preferred = (
        to_num(df[PREFERRED_IGID_COL])
        if PREFERRED_IGID_COL in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    legacy = (
        to_num(df[LEGACY_IGID_COL])
        if LEGACY_IGID_COL in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    if preferred.notna().sum() == 0 and legacy.notna().sum() == 0:
        raise RuntimeError(
            "Cannot reproduce the Composite>=2 training reference because both "
            f"{PREFERRED_IGID_COL!r} and {LEGACY_IGID_COL!r} are absent or empty."
        )
    igid = preferred.where(preferred.notna(), legacy)

    bundle = to_num(df[COL_BUNDLE])
    available = pd.DataFrame(
        {
            "gq": to_num(df[COL_GQ]).notna(),
            "igid": igid.notna(),
            "ori": to_num(df[COL_ORI]).notna(),
            "purity": to_num(df[COL_PURITY]).notna(),
            "bundle": bundle.notna() & bundle.ge(0),
        },
        index=df.index,
    )
    return available.sum(axis=1).astype(int)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        ["growth_mode", "data_level", *MODEL_FEATURES],
        "CNT 3.0 dataset",
    )

    data = df.copy()
    data["_composite_component_count"] = composite_component_count(data)

    data["data_level"] = (
        data["data_level"].fillna("").astype(str).str.strip().str.upper()
    )
    data = data[data["data_level"].isin(["L1", "L2"])].copy()

    data["growth_mode"] = data["growth_mode"].map(mode_number)
    data = data[data["growth_mode"].isin([1, 2, 3, 4, 5])].copy()
    data["growth_mode"] = data["growth_mode"].astype(int)
    data["growth_mode_letter"] = data["growth_mode"].map(mode_letter)

    if "sample_id" not in data.columns:
        data["sample_id"] = [f"ROW_{index:06d}" for index in range(1, len(data) + 1)]
    data["sample_id"] = data["sample_id"].fillna("").astype(str)
    empty_sample = data["sample_id"].str.len().eq(0)
    data.loc[empty_sample, "sample_id"] = [
        f"ROW_{index:06d}" for index in range(1, int(empty_sample.sum()) + 1)
    ]

    if "paper_id" not in data.columns:
        fallback = None
        for column in ["paper_doi", "source_file", "paper_title"]:
            if column in data.columns:
                fallback = data[column].fillna("").astype(str)
                break
        data["paper_id"] = fallback if fallback is not None else "UNKNOWN_PAPER"
    data["paper_id"] = data["paper_id"].fillna("UNKNOWN_PAPER").astype(str)

    for column in MODEL_FEATURES:
        data[column] = to_num(data[column])

    # Frozen model input convention: missing physical value -> 0, miss_* -> 1.
    for feature in PHYSICAL_COLS:
        miss = f"miss_{feature}"
        data[miss] = data[miss].fillna(1).gt(0).astype(int)
        data.loc[data[feature].isna(), miss] = 1
        data.loc[data[miss].eq(1), feature] = 0.0
        data[feature] = data[feature].fillna(0.0)

    if data.empty:
        raise RuntimeError("No valid L1/L2 CNT 3.0 rows remain.")

    return data.reset_index(drop=True)


def build_support_reference(data: pd.DataFrame) -> pd.DataFrame:
    reference = data[
        data["_composite_component_count"].ge(MIN_COMPONENT_COUNT)
    ].copy()
    if len(reference) < 2:
        raise RuntimeError("Composite>=2 support reference contains fewer than 2 rows.")
    return reference.reset_index(drop=True)


def add_representative_meta(
    states: pd.DataFrame,
    source: pd.DataFrame,
    key: str,
    prefix: str,
) -> pd.DataFrame:
    meta = [column for column in META_COLS if column in source.columns]
    if not meta:
        return states

    representative = (
        source[[key, *meta]]
        .drop_duplicates(key, keep="first")
        .rename(columns={column: f"{prefix}_{column}" for column in meta})
    )
    return states.merge(representative, on=key, how="left")


# =============================================================================
# OBSERVED N+C AND F+P STATES
# =============================================================================


def build_catalyst_states(data: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = data.copy()

    # A catalyst state must not be distinguished only because its source paper used
    # a different temperature/gas route. Those route-dependent dimensions are
    # rebuilt after catalyst × route pairing.
    key_frame = work[NC_BASE_STATE_COLS].copy()
    for column in NC_BASE_COLS:
        key_frame[column] = key_frame[column].round(ROUND_STATE_DECIMALS)

    work["_catalyst_key"] = key_frame.astype(str).agg("|".join, axis=1)
    frequency = (
        work.groupby("_catalyst_key").size().reset_index(name="catalyst_frequency")
    )

    states = (
        work.drop_duplicates("_catalyst_key", keep="first")
        .merge(frequency, on="_catalyst_key", how="left")
        .sort_values("catalyst_frequency", ascending=False, kind="stable")
        .head(top_n)
        .copy()
    )
    states = states.drop(columns=META_COLS, errors="ignore")
    states = add_representative_meta(states, work, "_catalyst_key", "catalyst")
    states["catalyst_state_id"] = [
        f"CATSTATE_{index:04d}" for index in range(1, len(states) + 1)
    ]

    meta = [
        column
        for column in states.columns
        if column.startswith("catalyst_")
        and column not in {"catalyst_state_id", "catalyst_frequency"}
    ]
    return states[
        [
            "catalyst_state_id",
            "_catalyst_key",
            "catalyst_frequency",
            *NC_STATE_COLS,
            *meta,
        ]
    ].reset_index(drop=True)


def build_route_states(
    data: pd.DataFrame,
    top_routes: int,
    min_per_mode: int,
) -> pd.DataFrame:
    work = data.copy()
    key_frame = work[["growth_mode", *FP_STATE_COLS]].copy()
    for column in FP_COLS:
        key_frame[column] = key_frame[column].round(ROUND_STATE_DECIMALS)

    work["_route_key"] = key_frame.astype(str).agg("|".join, axis=1)
    frequency = work.groupby("_route_key").size().reset_index(name="route_frequency")

    routes = (
        work.drop_duplicates("_route_key", keep="first")
        .merge(frequency, on="_route_key", how="left")
        .sort_values("route_frequency", ascending=False, kind="stable")
        .copy()
    )
    routes = routes.drop(columns=META_COLS, errors="ignore")
    routes = add_representative_meta(routes, work, "_route_key", "route")

    selected = []
    for mode in range(1, 6):
        subset = routes[routes["growth_mode"].eq(mode)].head(min_per_mode)
        if not subset.empty:
            selected.append(subset)

    picked = pd.concat(selected, ignore_index=False) if selected else routes.head(min_per_mode)
    if len(picked) < top_routes:
        extra = routes[~routes.index.isin(picked.index)].head(top_routes - len(picked))
        picked = pd.concat([picked, extra], ignore_index=False)

    picked = (
        picked.drop_duplicates("_route_key").head(top_routes).reset_index(drop=True)
    )
    picked["route_state_id"] = [
        f"ROUTE_{index:04d}" for index in range(1, len(picked) + 1)
    ]

    meta = [
        column
        for column in picked.columns
        if column.startswith("route_")
        and column not in {"route_state_id", "route_frequency"}
    ]
    return picked[
        [
            "route_state_id",
            "_route_key",
            "route_frequency",
            "growth_mode",
            "growth_mode_letter",
            *FP_STATE_COLS,
            *meta,
        ]
    ].reset_index(drop=True)


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================


def rebuild_route_dependent_nc(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the N/C descriptors that are determined by the selected route.

    The WorldMap keeps the literature-observed catalyst base state (composition,
    nanoparticle state, C/M and S/M) but must not keep the source paper's
    temperature/gas descriptors after pairing that catalyst with another route.

    Formal route mapping used by the recommendation pipeline:
      P1  -> growth temperature
      P8  -> carbon partial pressure
      P9  -> hydrogen partial pressure
      P10 -> p_C / p_H2
    """
    output = frame.copy()

    def observed(column: str) -> pd.Series:
        return to_num(output[f"miss_{column}"]).fillna(1).eq(0) & to_num(output[column]).notna()

    def copy_route(target: str, source: str) -> None:
        valid = observed(source)
        output[target] = np.where(valid, to_num(output[source]), 0.0)
        output[f"miss_{target}"] = (~valid).astype(int)

    # Temperature-linked state. N8 is the catalyst effective melting temperature.
    valid_t = observed("P1") & observed("N8") & to_num(output["N8"]).ne(0)
    ratio_t_tm = pd.Series(np.nan, index=output.index, dtype=float)
    ratio_t_tm.loc[valid_t] = (
        to_num(output.loc[valid_t, "P1"]) / to_num(output.loc[valid_t, "N8"])
    )

    output["N9"] = ratio_t_tm.fillna(0.0)
    output["miss_N9"] = (~valid_t).astype(int)

    copy_route("C1", "P1")
    output["C2"] = ratio_t_tm.fillna(0.0)
    output["miss_C2"] = (~valid_t).astype(int)

    # Route gas state: keep exactly the same mapping used by build_zc.py.
    copy_route("C5", "P8")
    copy_route("C6", "P9")
    copy_route("C7", "P10")

    # Catalyst carbide state is intrinsic to the catalyst; C8 is defined from N16.
    valid_c8 = observed("N16")
    output["C8"] = np.where(valid_c8, to_num(output["N16"]), 0.0)
    output["miss_C8"] = (~valid_c8).astype(int)

    valid_c9 = valid_c8 & valid_t
    output["C9"] = np.where(
        valid_c9,
        to_num(output["N16"]) * ratio_t_tm.fillna(0.0),
        0.0,
    )
    output["miss_C9"] = (~valid_c9).astype(int)

    return output


def generate_candidates(
    catalyst_states: pd.DataFrame,
    route_states: pd.DataFrame,
    routes_per_mode: int,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    outputs = []

    for mode in range(1, 6):
        routes = route_states[route_states["growth_mode"].eq(mode)].head(routes_per_mode)
        if routes.empty:
            continue

        catalysts = catalyst_states.copy()
        routes = routes.copy()
        catalysts["_merge"] = 1
        routes["_merge"] = 1
        combined = catalysts.merge(routes, on="_merge", how="inner").drop(columns="_merge")
        combined["growth_mode"] = mode
        combined["growth_mode_letter"] = mode_letter(mode)
        outputs.append(combined)

    if not outputs:
        raise RuntimeError("No candidate combinations were generated.")

    candidates = pd.concat(outputs, ignore_index=True)

    # The Cartesian product changes the route. Rebuild every N/C descriptor that
    # depends on route temperature or gas state before any model prediction.
    candidates = rebuild_route_dependent_nc(candidates)

    if len(candidates) > max_rows:
        candidates = candidates.sample(
            n=max_rows,
            random_state=random_state,
        ).reset_index(drop=True)

    candidates["combination_id"] = (
        candidates["catalyst_state_id"] + "||" + candidates["route_state_id"]
    )
    require_columns(candidates, MODEL_FEATURES, "Candidate table")
    return candidates


# =============================================================================
# MODELS AND PREDICTION
# =============================================================================


def load_models(
    project_dir: Path,
    l1_path: str,
    q10_path: str,
    q50_path: str,
    q90_path: str,
) -> Dict[str, lgb.Booster]:
    paths = {
        "L1": resolve_path(project_dir, l1_path),
        "q10": resolve_path(project_dir, q10_path),
        "q50": resolve_path(project_dir, q50_path),
        "q90": resolve_path(project_dir, q90_path),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n  " + "\n  ".join(missing))

    return {name: lgb.Booster(model_file=str(path)) for name, path in paths.items()}


def aligned_features(model: lgb.Booster, data: pd.DataFrame, label: str) -> pd.DataFrame:
    features = model.feature_name()
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise RuntimeError(
            f"{label} model features missing from candidates:\n  {missing}"
        )
    return data[features].copy()


def predict_all(
    candidates: pd.DataFrame,
    models: Dict[str, lgb.Booster],
) -> pd.DataFrame:
    output = candidates.copy()
    raw = {}

    for label in ["q10", "q50", "q90"]:
        raw[label] = np.asarray(
            models[label].predict(aligned_features(models[label], output, label)),
            dtype=float,
        )

    ordered = np.sort(np.column_stack([raw["q10"], raw["q50"], raw["q90"]]), axis=1)
    output["pred_q10_raw"] = raw["q10"]
    output["pred_q50_raw"] = raw["q50"]
    output["pred_q90_raw"] = raw["q90"]

    # Compatibility / audit columns only. Strategy identities remain raw models.
    output["q10_fix"] = ordered[:, 0]
    output["q50_fix"] = ordered[:, 1]
    output["q90_fix"] = ordered[:, 2]
    output["quantile_fixed"] = (
        (raw["q10"] > raw["q50"])
        | (raw["q50"] > raw["q90"])
        | (raw["q10"] > raw["q90"])
    ).astype(int)

    output["interval_width"] = raw["q90"] - raw["q10"]
    output["interval_width_ordered"] = ordered[:, 2] - ordered[:, 0]
    output["pred_L1_composite_final_score"] = np.asarray(
        models["L1"].predict(aligned_features(models["L1"], output, "L1")),
        dtype=float,
    )

    output["balanced_score"] = output["pred_q50_raw"]
    output["potential_score"] = output["pred_q90_raw"]
    output["conservative_score"] = output["pred_q10_raw"]
    return output


# =============================================================================
# SUPPORT-DOMAIN MODELS
# =============================================================================


def restore_physical_missing(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    matrix = frame[list(features)].apply(pd.to_numeric, errors="coerce").copy()
    for feature in features:
        miss = f"miss_{feature}"
        missing_mask = to_num(frame[miss]).fillna(1).gt(0)
        matrix.loc[missing_mask, feature] = np.nan
    return matrix


def missingness_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    columns = [f"miss_{feature}" for feature in features]
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(1)
        .gt(0)
        .astype(float)
        .to_numpy()
    )


def deduplicate_vectors(matrix: np.ndarray) -> np.ndarray:
    rounded = np.round(matrix, ROUND_DISTANCE_DECIMALS)
    return pd.DataFrame(rounded).drop_duplicates().index.to_numpy(dtype=int)


def trace_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["sample"] = frame["sample_id"].fillna("").astype(str)
    output["paper"] = frame["paper_id"].fillna("UNKNOWN_PAPER").astype(str)

    if "paper_year_int" in frame.columns:
        output["year"] = to_num(frame["paper_year_int"])
    elif "paper_year" in frame.columns:
        output["year"] = to_num(frame["paper_year"])
    else:
        output["year"] = np.nan
    return output


@dataclass
class PhysicalSupportSpace:
    label: str
    features: List[str]
    median: pd.Series
    scale: pd.Series
    model: NearestNeighbors
    reference_vectors: np.ndarray
    reference_meta: pd.DataFrame
    training_nn_distances: np.ndarray
    p95: float
    p99: float
    n_reference_rows: int


@dataclass
class MissingnessSupportSpace:
    model: NearestNeighbors
    reference_patterns: np.ndarray
    reference_meta: pd.DataFrame
    training_nn_distances: np.ndarray
    p95: float
    p99: float
    n_reference_rows: int


def fit_physical_support_space(
    reference: pd.DataFrame,
    features: Sequence[str],
    label: str,
) -> PhysicalSupportSpace:
    raw = restore_physical_missing(reference, features)
    median = raw.median().fillna(0.0)
    filled = raw.fillna(median)

    iqr = (filled.quantile(0.75) - filled.quantile(0.25)).abs()
    std = filled.std(ddof=0).abs()
    scale = iqr.where(iqr > 1e-12, std).where(lambda values: values > 1e-12, 1.0)
    scaled = ((filled - median) / scale).to_numpy(dtype=float)

    unique_index = deduplicate_vectors(scaled)
    unique_vectors = scaled[unique_index]
    unique_meta = trace_metadata(reference.iloc[unique_index]).reset_index(drop=True)

    if len(unique_vectors) < 2:
        raise RuntimeError(f"{label} support space has fewer than 2 unique states.")

    model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(unique_vectors)
    loo = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(unique_vectors)
    training_nn = loo.kneighbors(unique_vectors, return_distance=True)[0][:, 1]

    return PhysicalSupportSpace(
        label=label,
        features=list(features),
        median=median,
        scale=scale,
        model=model,
        reference_vectors=unique_vectors,
        reference_meta=unique_meta,
        training_nn_distances=training_nn,
        p95=float(np.quantile(training_nn, SUPPORT_P95)),
        p99=float(np.quantile(training_nn, SUPPORT_P99)),
        n_reference_rows=len(reference),
    )


def fit_missingness_support_space(reference: pd.DataFrame) -> MissingnessSupportSpace:
    patterns = missingness_matrix(reference, PHYSICAL_COLS)
    unique_index = deduplicate_vectors(patterns)
    unique_patterns = patterns[unique_index]
    unique_meta = trace_metadata(reference.iloc[unique_index]).reset_index(drop=True)

    if len(unique_patterns) == 1:
        training_nn = np.zeros(1, dtype=float)
    else:
        loo = NearestNeighbors(n_neighbors=2, metric="hamming").fit(unique_patterns)
        training_nn = loo.kneighbors(unique_patterns, return_distance=True)[0][:, 1]

    model = NearestNeighbors(n_neighbors=1, metric="hamming").fit(unique_patterns)
    return MissingnessSupportSpace(
        model=model,
        reference_patterns=unique_patterns,
        reference_meta=unique_meta,
        training_nn_distances=training_nn,
        p95=float(np.quantile(training_nn, SUPPORT_P95)),
        p99=float(np.quantile(training_nn, SUPPORT_P99)),
        n_reference_rows=len(reference),
    )


def empirical_percentile(values: np.ndarray, reference_distribution: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference_distribution, dtype=float))
    return 100.0 * np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def apply_physical_support(
    candidates: pd.DataFrame,
    space: PhysicalSupportSpace,
) -> pd.DataFrame:
    raw = restore_physical_missing(candidates, space.features)
    filled = raw.fillna(space.median)
    scaled = ((filled - space.median) / space.scale).to_numpy(dtype=float)
    distances, indices = space.model.kneighbors(scaled)
    nearest = space.reference_meta.iloc[indices[:, 0]].reset_index(drop=True)

    return pd.DataFrame(
        {
            "distance": distances[:, 0],
            "percentile": empirical_percentile(
                distances[:, 0],
                space.training_nn_distances,
            ),
            "nearest_sample": nearest["sample"].to_numpy(),
            "nearest_paper": nearest["paper"].to_numpy(),
            "nearest_year": nearest["year"].to_numpy(),
        }
    )


def apply_missingness_support(
    candidates: pd.DataFrame,
    space: MissingnessSupportSpace,
) -> pd.DataFrame:
    patterns = missingness_matrix(candidates, PHYSICAL_COLS)
    distances, indices = space.model.kneighbors(patterns)
    nearest = space.reference_meta.iloc[indices[:, 0]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "distance": distances[:, 0],
            "nearest_sample": nearest["sample"].to_numpy(),
            "nearest_paper": nearest["paper"].to_numpy(),
        }
    )


def classify_domain(distance: pd.Series, p95: float, p99: float) -> pd.Series:
    values = pd.to_numeric(distance, errors="coerce")
    return pd.Series(
        np.select(
            [values.le(p95), values.le(p99)],
            ["supported", "near-boundary"],
            default="out-of-domain",
        ),
        index=distance.index,
        dtype=str,
    )


def add_support_metrics(
    candidates: pd.DataFrame,
    nc_space: PhysicalSupportSpace,
    fp_space: PhysicalSupportSpace,
    full_space: PhysicalSupportSpace,
    missing_space: MissingnessSupportSpace,
) -> pd.DataFrame:
    output = candidates.copy()
    nc = apply_physical_support(output, nc_space)
    fp = apply_physical_support(output, fp_space)
    full = apply_physical_support(output, full_space)
    missing = apply_missingness_support(output, missing_space)

    output["support_distance_NC"] = nc["distance"].to_numpy()
    output["support_distance_FP"] = fp["distance"].to_numpy()
    output["support_distance_full"] = full["distance"].to_numpy()
    output["support_percentile_full"] = full["percentile"].to_numpy()
    output["domain_status"] = classify_domain(
        output["support_distance_full"],
        full_space.p95,
        full_space.p99,
    )
    output["nearest_training_sample"] = full["nearest_sample"].to_numpy()
    output["nearest_training_paper"] = full["nearest_paper"].to_numpy()
    output["nearest_training_year"] = full["nearest_year"].to_numpy()
    output["missingness_distance"] = missing["distance"].to_numpy()
    output["nearest_missingness_sample"] = missing["nearest_sample"].to_numpy()
    output["nearest_missingness_paper"] = missing["nearest_paper"].to_numpy()
    return output


def support_threshold_table(
    spaces: Sequence[PhysicalSupportSpace],
    missing_space: MissingnessSupportSpace,
) -> pd.DataFrame:
    rows = []
    for space in spaces:
        rows.append(
            {
                "space": space.label,
                "distance_metric": "robust-scaled Euclidean",
                "n_reference_rows": space.n_reference_rows,
                "n_unique_reference_states": len(space.reference_vectors),
                "n_features": len(space.features),
                "nn_distance_p50": float(np.quantile(space.training_nn_distances, 0.50)),
                "nn_distance_p95": space.p95,
                "nn_distance_p99": space.p99,
            }
        )

    rows.append(
        {
            "space": "missingness",
            "distance_metric": "Hamming",
            "n_reference_rows": missing_space.n_reference_rows,
            "n_unique_reference_states": len(missing_space.reference_patterns),
            "n_features": len(MISS_COLS),
            "nn_distance_p50": float(np.quantile(missing_space.training_nn_distances, 0.50)),
            "nn_distance_p95": missing_space.p95,
            "nn_distance_p99": missing_space.p99,
        }
    )
    return pd.DataFrame(rows)


# =============================================================================
# PCA ON UNIQUE STATES
# =============================================================================


def block_pca(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    usable = []
    for feature in feature_columns:
        observed = to_num(frame[f"miss_{feature}"]).fillna(1).eq(0)
        values = to_num(frame.loc[observed, feature]).dropna()
        if len(values) >= 2 and values.nunique() > 1:
            usable.append(feature)

    if not usable:
        return np.zeros((len(frame), 2)), np.zeros(2), []

    matrix = pd.DataFrame(index=frame.index)
    for feature in usable:
        missing = to_num(frame[f"miss_{feature}"]).fillna(1).gt(0)
        values = to_num(frame[feature]).copy()
        values.loc[missing] = np.nan
        values = values.fillna(values.median(skipna=True))
        std = values.std(ddof=1)
        if not np.isfinite(std) or std == 0:
            std = 1.0
        matrix[feature] = (values - values.mean()) / std

    array = matrix.to_numpy(dtype=float)
    array -= np.mean(array, axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(array, full_matrices=False)
    count = min(2, len(singular_values))
    coordinates = u[:, :count] * singular_values[:count]
    if count == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(frame))])

    variance = singular_values ** 2
    ratio = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    explained = np.array(
        [ratio[0] if len(ratio) > 0 else 0.0, ratio[1] if len(ratio) > 1 else 0.0]
    )
    return coordinates, explained, usable


def add_state_pcas(
    frame: pd.DataFrame,
    blocks: Dict[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    audit = []

    for name, columns in blocks.items():
        coords, explained, usable = block_pca(output, columns)
        output[f"{name}_PC1"] = coords[:, 0]
        output[f"{name}_PC2"] = coords[:, 1]
        audit.append(
            {
                "block": name,
                "n_state_rows": len(output),
                "n_requested_features": len(columns),
                "n_used_features": len(usable),
                "used_features": ",".join(usable),
                "excluded_features": ",".join(
                    [column for column in columns if column not in usable]
                ),
                "PC1_explained_variance_ratio": explained[0],
                "PC2_explained_variance_ratio": explained[1],
            }
        )

    return output, pd.DataFrame(audit)


def merge_state_coordinates(
    predictions: pd.DataFrame,
    catalyst_states: pd.DataFrame,
    route_states: pd.DataFrame,
) -> pd.DataFrame:
    catalyst_columns = [
        "catalyst_state_id",
        "nuc_PC1", "nuc_PC2",
        "cat_PC1", "cat_PC2",
        "NC_PC1", "NC_PC2",
    ]
    route_columns = [
        "route_state_id",
        "flow_PC1", "flow_PC2",
        "proc_PC1", "proc_PC2",
        "FP_PC1", "FP_PC2",
    ]
    output = predictions.merge(
        catalyst_states[catalyst_columns],
        on="catalyst_state_id",
        how="left",
        validate="many_to_one",
    )
    return output.merge(
        route_states[route_columns],
        on="route_state_id",
        how="left",
        validate="many_to_one",
    )


# =============================================================================
# AGGREGATED STATE TABLES
# =============================================================================


def add_domain_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for status in DOMAIN_ORDER:
        output[f"is_{status.replace('-', '_')}"] = output["domain_status"].eq(status).astype(float)
    return output


def aggregate_catalyst_states(
    predictions: pd.DataFrame,
    catalyst_states: pd.DataFrame,
) -> pd.DataFrame:
    work = add_domain_flags(predictions)
    grouped = work.groupby(
        ["catalyst_state_id", "growth_mode", "growth_mode_letter"],
        sort=False,
    )
    aggregated = grouped.agg(
        candidate_count=("combination_id", "size"),
        median_q50=("balanced_score", "median"),
        max_q50=("balanced_score", "max"),
        q25_q50=("balanced_score", lambda values: values.quantile(0.25)),
        q75_q50=("balanced_score", lambda values: values.quantile(0.75)),
        median_q90=("potential_score", "median"),
        max_q90=("potential_score", "max"),
        supported_fraction=("is_supported", "mean"),
        near_boundary_fraction=("is_near_boundary", "mean"),
        out_of_domain_fraction=("is_out_of_domain", "mean"),
        median_full_support_distance=("support_distance_full", "median"),
    ).reset_index()
    aggregated["q50_IQR"] = aggregated["q75_q50"] - aggregated["q25_q50"]

    state_columns = [
        "catalyst_state_id",
        "catalyst_frequency",
        "N11", "C3",
        "nuc_PC1", "nuc_PC2",
        "cat_PC1", "cat_PC2",
        "NC_PC1", "NC_PC2",
    ]
    return aggregated.merge(
        catalyst_states[[column for column in state_columns if column in catalyst_states.columns]],
        on="catalyst_state_id",
        how="left",
        validate="many_to_one",
    )


def aggregate_route_states(
    predictions: pd.DataFrame,
    route_states: pd.DataFrame,
) -> pd.DataFrame:
    work = add_domain_flags(predictions)
    grouped = work.groupby(
        ["route_state_id", "growth_mode", "growth_mode_letter"],
        sort=False,
    )
    aggregated = grouped.agg(
        candidate_count=("combination_id", "size"),
        median_q50=("balanced_score", "median"),
        max_q50=("balanced_score", "max"),
        q25_q50=("balanced_score", lambda values: values.quantile(0.25)),
        q75_q50=("balanced_score", lambda values: values.quantile(0.75)),
        median_q90=("potential_score", "median"),
        max_q90=("potential_score", "max"),
        supported_fraction=("is_supported", "mean"),
        near_boundary_fraction=("is_near_boundary", "mean"),
        out_of_domain_fraction=("is_out_of_domain", "mean"),
        median_full_support_distance=("support_distance_full", "median"),
    ).reset_index()
    aggregated["q50_IQR"] = aggregated["q75_q50"] - aggregated["q25_q50"]

    state_columns = [
        "route_state_id",
        "route_frequency",
        "flow_PC1", "flow_PC2",
        "proc_PC1", "proc_PC2",
        "FP_PC1", "FP_PC2",
    ]
    return aggregated.merge(
        route_states[[column for column in state_columns if column in route_states.columns]],
        on="route_state_id",
        how="left",
        validate="many_to_one",
    )


# =============================================================================
# FIGURES
# =============================================================================


def set_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#0b1020",
            "savefig.facecolor": "#0b1020",
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.edgecolor": "#c7d0ff",
            "xtick.color": "#c7d0ff",
            "ytick.color": "#c7d0ff",
            "grid.color": "#3a4a7a",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )


def mode_figure():
    return plt.subplots(2, 3, figsize=(18.5, 9.5), layout="constrained")


def scaled_sizes(values: pd.Series, minimum: float = 35.0, maximum: float = 190.0) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    if len(numeric) == 0 or np.nanmax(numeric) <= np.nanmin(numeric):
        return np.full(len(numeric), (minimum + maximum) / 2.0)
    scaled = (numeric - np.nanmin(numeric)) / (np.nanmax(numeric) - np.nanmin(numeric))
    return minimum + scaled * (maximum - minimum)


def save_supported_candidate_space(frame: pd.DataFrame, path: Path) -> None:
    set_dark_style()
    fig, axes = mode_figure()
    axes = axes.ravel()

    supported_scatter = None
    for index, mode in enumerate(range(1, 6)):
        ax = axes[index]
        subset = frame[frame["growth_mode"].eq(mode)].copy()
        if subset.empty:
            ax.set_axis_off()
            continue

        out = subset[subset["domain_status"].eq("out-of-domain")]
        near = subset[subset["domain_status"].eq("near-boundary")]
        supported = subset[subset["domain_status"].eq("supported")]

        if not out.empty:
            ax.scatter(
                out["balanced_score"],
                out["potential_score"],
                marker="x",
                s=28,
                alpha=0.45,
                label="Out of domain",
            )
        if not near.empty:
            ax.scatter(
                near["balanced_score"],
                near["potential_score"],
                facecolors="none",
                edgecolors="white",
                marker="o",
                s=32,
                linewidths=0.8,
                alpha=0.75,
                label="Near boundary",
            )
        if not supported.empty:
            supported_scatter = ax.scatter(
                supported["balanced_score"],
                supported["potential_score"],
                c=supported["pred_L1_composite_final_score"],
                cmap="plasma",
                marker="o",
                s=30,
                alpha=0.8,
                edgecolors="none",
                label="Supported",
            )

        ax.set_title(f"Mode {mode_letter(mode)}")
        ax.set_xlabel("Balanced score (raw q50)")
        ax.set_ylabel("Potential score (raw q90)")
        if index == 0:
            ax.legend(frameon=False)

    axes[5].set_axis_off()
    if supported_scatter is not None:
        bar = fig.colorbar(supported_scatter, ax=axes[:5], shrink=0.86)
        bar.set_label("L1 predicted composite score")

    fig.suptitle("CNT WorldMap candidate space with empirical training-domain support", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_aggregated_pca(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
) -> None:
    set_dark_style()
    fig, axes = mode_figure()
    axes = axes.ravel()

    color_values = pd.to_numeric(frame["median_q50"], errors="coerce")
    finite = color_values[np.isfinite(color_values)]
    vmin = float(finite.min()) if len(finite) else 0.0
    vmax = float(finite.max()) if len(finite) else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    for index, mode in enumerate(range(1, 6)):
        ax = axes[index]
        subset = frame[frame["growth_mode"].eq(mode)].copy()
        if subset.empty:
            ax.set_axis_off()
            continue

        rgba = cmap(norm(subset["median_q50"].to_numpy(dtype=float)))
        rgba[:, 3] = 0.25 + 0.75 * subset["supported_fraction"].clip(0, 1).to_numpy(dtype=float)
        ax.scatter(
            subset[x],
            subset[y],
            s=scaled_sizes(subset["candidate_count"]),
            c=rgba,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_title(f"Mode {mode_letter(mode)}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    axes[5].set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    bar = fig.colorbar(scalar, ax=axes[:5], shrink=0.86)
    bar.set_label("Median balanced score (raw q50)")
    fig.suptitle(
        title + "\nPoint size: candidate count; opacity: supported fraction",
        fontsize=16,
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_catalyst_physical_aggregated(frame: pd.DataFrame, path: Path) -> None:
    set_dark_style()
    fig, axes = mode_figure()
    axes = axes.ravel()

    for index, mode in enumerate(range(1, 6)):
        ax = axes[index]
        subset = frame[frame["growth_mode"].eq(mode)].copy()
        valid = (
            to_num(subset["N11"]).gt(0)
            & to_num(subset["C3"]).gt(0)
        )
        subset = subset[valid]
        if subset.empty:
            ax.set_axis_off()
            continue

        scatter = ax.scatter(
            subset["N11"],
            np.log10(subset["C3"]),
            c=subset["median_q50"],
            cmap="plasma",
            s=scaled_sizes(subset["candidate_count"]),
            alpha=0.35 + 0.65 * subset["supported_fraction"].clip(0, 1),
            edgecolors="white",
            linewidths=0.4,
        )
        ax.set_xscale("log")
        ax.set_title(f"Mode {mode_letter(mode)}")
        ax.set_xlabel("Catalyst particle size N11 (nm)")
        ax.set_ylabel("log10(raw C/M, C3)")

    axes[5].set_axis_off()
    if "scatter" in locals():
        bar = fig.colorbar(scatter, ax=axes[:5], shrink=0.86)
        bar.set_label("Median balanced score (raw q50)")
    fig.suptitle("Aggregated observed catalyst-state landscape", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_domain_distribution(frame: pd.DataFrame, path: Path) -> None:
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), layout="constrained")

    counts = (
        frame.groupby(["growth_mode_letter", "domain_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=DOMAIN_ORDER, fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    bottom = np.zeros(len(proportions))
    x = np.arange(len(proportions))
    for status in DOMAIN_ORDER:
        values = proportions[status].to_numpy(dtype=float)
        axes[0].bar(x, values, bottom=bottom, label=DOMAIN_LABELS[status])
        bottom += values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Mode {mode}" for mode in proportions.index])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Candidate fraction")
    axes[0].set_title("Training-domain status by growth mode")
    axes[0].legend(frameon=False)

    for status in DOMAIN_ORDER:
        values = frame.loc[
            frame["domain_status"].eq(status),
            "support_percentile_full",
        ]
        if len(values):
            axes[1].hist(values, bins=25, alpha=0.55, label=DOMAIN_LABELS[status])
    axes[1].axvline(95, linestyle="--", linewidth=1.0)
    axes[1].axvline(99, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Full-60D support distance percentile")
    axes[1].set_ylabel("Candidate count")
    axes[1].set_title("Candidate support distribution")
    axes[1].legend(frameon=False)

    fig.suptitle("CNT WorldMap empirical domain audit", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_dashboard(frame: pd.DataFrame, path: Path) -> bool:
    try:
        import plotly.express as px
    except ImportError:
        return False

    hover = [
        column
        for column in [
            "catalyst_state_id",
            "route_state_id",
            "growth_mode_letter",
            "domain_status",
            "support_distance_full",
            "support_percentile_full",
            "nearest_training_sample",
            "nearest_training_paper",
            "pred_q10_raw",
            "pred_q50_raw",
            "pred_q90_raw",
            "interval_width",
            "pred_L1_composite_final_score",
            "N11",
            "C3",
            "P1",
            "F1",
        ]
        if column in frame.columns
    ]

    fig = px.scatter(
        frame,
        x="balanced_score",
        y="potential_score",
        color="growth_mode_letter",
        symbol="domain_status",
        hover_data=hover,
        title="CNT 3.0 WorldMap — raw q50/q90 with empirical domain status",
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return True


def combine_pngs(paths: Sequence[Path], output_pdf: Path) -> None:
    with PdfPages(output_pdf) as pdf:
        for path in paths:
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.name)
            pdf.savefig(fig)
            plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNT 3.0 WorldMap with empirical training-domain support"
    )
    parser.add_argument("--project_dir", default=str(PROJECT_DIR))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out_base", default=OUT_BASE)
    parser.add_argument("--l1_model", default=L1_MODEL)
    parser.add_argument("--q10_model", default=Q10_MODEL)
    parser.add_argument("--q50_model", default=Q50_MODEL)
    parser.add_argument("--q90_model", default=Q90_MODEL)
    parser.add_argument("--top_routes", type=int, default=120)
    parser.add_argument("--min_routes_per_mode", type=int, default=20)
    parser.add_argument("--n_routes_per_mode", type=int, default=30)
    parser.add_argument("--n_catalyst_states", type=int, default=80)
    parser.add_argument("--max_rows", type=int, default=25000)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    dataset_path = resolve_path(project_dir, args.dataset)
    out_dir = resolve_path(project_dir, args.out_base) / f"run_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("[CNT WORLDMAP START]")
    print(f"Project directory       : {project_dir}")
    print(f"Dataset                 : {dataset_path}")
    print(f"Output directory        : {out_dir}")
    print(f"Composite support filter: >= {MIN_COMPONENT_COUNT} components")
    print("=" * 108)

    data = prepare_dataset(read_csv_smart(dataset_path))
    support_reference = build_support_reference(data)
    models = load_models(
        project_dir,
        args.l1_model,
        args.q10_model,
        args.q50_model,
        args.q90_model,
    )

    catalyst_states = build_catalyst_states(data, args.n_catalyst_states)
    route_states = build_route_states(
        data,
        args.top_routes,
        args.min_routes_per_mode,
    )

    catalyst_states, catalyst_pca_audit = add_state_pcas(
        catalyst_states,
        {"nuc": N_BASE_COLS, "cat": C_BASE_COLS, "NC": NC_BASE_COLS},
    )
    route_states, route_pca_audit = add_state_pcas(
        route_states,
        {"flow": F_COLS, "proc": P_COLS, "FP": FP_COLS},
    )
    pca_audit = pd.concat(
        [catalyst_pca_audit, route_pca_audit],
        ignore_index=True,
    )

    candidates = generate_candidates(
        catalyst_states,
        route_states,
        args.n_routes_per_mode,
        args.max_rows,
        args.random_state,
    )
    predictions = predict_all(candidates, models)

    print("[SUPPORT] fitting N+C reference space")
    nc_space = fit_physical_support_space(support_reference, NC_COLS, "NC")
    print("[SUPPORT] fitting F+P reference space")
    fp_space = fit_physical_support_space(support_reference, FP_COLS, "FP")
    print("[SUPPORT] fitting full 60D reference space")
    full_space = fit_physical_support_space(support_reference, PHYSICAL_COLS, "full")
    print("[SUPPORT] fitting 60D missingness reference space")
    missing_space = fit_missingness_support_space(support_reference)

    predictions = add_support_metrics(
        predictions,
        nc_space,
        fp_space,
        full_space,
        missing_space,
    )

    catalyst_aggregated = aggregate_catalyst_states(predictions, catalyst_states)
    route_aggregated = aggregate_route_states(predictions, route_states)
    threshold_table = support_threshold_table(
        [nc_space, fp_space, full_space],
        missing_space,
    )

    candidate_csv = out_dir / "PHASE_CANDIDATES_sampled.csv"
    supported_csv = out_dir / "PHASE_CANDIDATES_supported.csv"
    catalyst_csv = out_dir / "CATALYST_STATES.csv"
    route_csv = out_dir / "ROUTE_STATES.csv"
    catalyst_agg_csv = out_dir / "CATALYST_STATE_AGGREGATED.csv"
    route_agg_csv = out_dir / "ROUTE_STATE_AGGREGATED.csv"
    pca_csv = out_dir / "BLOCK_PCA_AUDIT.csv"
    threshold_csv = out_dir / "SUPPORT_THRESHOLDS.csv"

    write_csv(predictions, candidate_csv)
    write_csv(predictions[predictions["domain_status"].eq("supported")], supported_csv)
    write_csv(catalyst_states, catalyst_csv)
    write_csv(route_states, route_csv)
    write_csv(catalyst_aggregated, catalyst_agg_csv)
    write_csv(route_aggregated, route_agg_csv)
    write_csv(pca_audit, pca_csv)
    write_csv(threshold_table, threshold_csv)

    fig_supported = out_dir / "fig_worldmap_supported_candidate_space.png"
    fig_catalyst = out_dir / "fig_worldmap_catalyst_state_aggregated.png"
    fig_route = out_dir / "figS_worldmap_route_state_aggregated.png"
    fig_domain = out_dir / "figS_worldmap_domain_distribution.png"
    fig_physical = out_dir / "02_mode_atlas_physical_state.png"

    save_supported_candidate_space(predictions, fig_supported)
    save_aggregated_pca(
        catalyst_aggregated,
        "NC_PC1",
        "NC_PC2",
        "Aggregated catalyst / nucleation state atlas",
        fig_catalyst,
    )
    save_aggregated_pca(
        route_aggregated,
        "FP_PC1",
        "FP_PC2",
        "Aggregated process / flow route atlas",
        fig_route,
    )
    save_domain_distribution(predictions, fig_domain)
    save_catalyst_physical_aggregated(catalyst_aggregated, fig_physical)

    # Preserve legacy figure names used by earlier review scripts.
    copy_figure(fig_supported, [out_dir / "01_mode_atlas_q50_vs_q90.png"])
    copy_figure(fig_route, [out_dir / "03_mode_atlas_process_flow.png"])
    copy_figure(fig_catalyst, [out_dir / "04_mode_atlas_catalyst_nucleation.png"])

    dashboard = out_dir / "dashboard_phase_diagram.html"
    if not build_dashboard(predictions, dashboard):
        dashboard = out_dir / "dashboard_phase_diagram.SKIPPED.txt"
        dashboard.write_text(
            "Plotly is not installed. Install: pip install plotly",
            encoding="utf-8",
        )

    pdf_path = out_dir / "ALL_FIGURES.pdf"
    combine_pngs(
        [fig_supported, fig_catalyst, fig_route, fig_domain, fig_physical],
        pdf_path,
    )

    domain_counts = predictions["domain_status"].value_counts().to_dict()
    summary = {
        "project_directory": str(project_dir),
        "dataset": str(dataset_path),
        "models": {
            "L1": str(resolve_path(project_dir, args.l1_model)),
            "q10": str(resolve_path(project_dir, args.q10_model)),
            "q50": str(resolve_path(project_dir, args.q50_model)),
            "q90": str(resolve_path(project_dir, args.q90_model)),
        },
        "n_source_rows": int(len(data)),
        "n_support_reference_rows": int(len(support_reference)),
        "support_reference_definition": "L1+L2 rows with Composite component count >= 2",
        "n_catalyst_states": int(len(catalyst_states)),
        "n_route_states": int(len(route_states)),
        "n_candidate_rows": int(len(predictions)),
        "quantile_crossings": int(predictions["quantile_fixed"].sum()),
        "candidate_definition": (
            "Literature-observed route-independent catalyst/carbon base states recombined "
            "with literature-observed F+P routes; route-dependent N/C descriptors are "
            "rebuilt from the selected route before prediction."
        ),
        "score_definition": {
            "balanced": "raw q50 model prediction",
            "potential": "raw q90 model prediction",
            "conservative": "raw q10 model prediction",
            "interval_width": "raw q90 - raw q10; descriptive only",
        },
        "support_definition": {
            "physical": (
                "Training-median imputation after restoring miss_* values to NaN; "
                "training-IQR scaling with standard-deviation fallback; Euclidean nearest neighbour."
            ),
            "missingness": "60D Hamming nearest-neighbour distance, kept separate.",
            "thresholds": (
                "Supported <= training leave-one-out NN p95; near-boundary p95-p99; "
                "out-of-domain > p99."
            ),
            "interpretation": (
                "Empirical training-domain support only; not a probability of correctness."
            ),
        },
        "domain_counts": {status: int(domain_counts.get(status, 0)) for status in DOMAIN_ORDER},
        "domain_fractions": {
            status: float(domain_counts.get(status, 0) / max(1, len(predictions)))
            for status in DOMAIN_ORDER
        },
        "outputs": {
            "candidates": str(candidate_csv),
            "supported_candidates": str(supported_csv),
            "catalyst_states": str(catalyst_csv),
            "route_states": str(route_csv),
            "catalyst_aggregated": str(catalyst_agg_csv),
            "route_aggregated": str(route_agg_csv),
            "support_thresholds": str(threshold_csv),
            "pca_audit": str(pca_csv),
            "dashboard": str(dashboard),
            "all_figures_pdf": str(pdf_path),
        },
    }

    summary_path = out_dir / "RUN_SUMMARY.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    manifest = [
        f"project_dir={project_dir}",
        f"dataset={dataset_path}",
        f"L1={resolve_path(project_dir, args.l1_model)}",
        f"q10={resolve_path(project_dir, args.q10_model)}",
        f"q50={resolve_path(project_dir, args.q50_model)}",
        f"q90={resolve_path(project_dir, args.q90_model)}",
        f"out_dir={out_dir}",
        "",
        "Generated files:",
        *[path.name for path in sorted(out_dir.iterdir())],
    ]
    (out_dir / "MANIFEST.txt").write_text("\n".join(manifest), encoding="utf-8")

    print("\n[RESULT SUMMARY]")
    print(f"Source L1/L2 rows      : {len(data)}")
    print(f"Support reference rows : {len(support_reference)}")
    print(f"Catalyst states        : {len(catalyst_states)}")
    print(f"Route states           : {len(route_states)}")
    print(f"Candidate predictions  : {len(predictions)}")
    print(f"Quantile crossings     : {int(predictions['quantile_fixed'].sum())}")
    print(
        "Domain status          : "
        + ", ".join(
            f"{status}={int(domain_counts.get(status, 0))}" for status in DOMAIN_ORDER
        )
    )
    print(f"Full-distance p95/p99  : {full_space.p95:.6g} / {full_space.p99:.6g}")
    print(f"Output directory       : {out_dir}")
    print(f"Candidate CSV          : {candidate_csv}")
    print(f"Aggregated catalyst CSV: {catalyst_agg_csv}")
    print(f"Aggregated route CSV   : {route_agg_csv}")
    print(f"Support thresholds CSV : {threshold_csv}")
    print(f"Combined PDF           : {pdf_path}")
    print(f"Dashboard              : {dashboard}")
    print("\n[OK] CNT WorldMap completed.")
    print("=" * 108)


if __name__ == "__main__":
    main()
