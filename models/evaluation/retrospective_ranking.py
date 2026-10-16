# -*- coding: utf-8 -*-
"""
retrospective_ranking.py

CNT 3.0 — Historical replay and retrospective ranking validation

This article-analysis script performs historical replay and ranking validation.
It retrains historical point/quantile models at several year cutoffs and tests
whether later, truly published high-quality CNT conditions are ranked near the top.

It does NOT use the old L123 virtual candidate generator and does NOT claim a
completed wet-lab active-learning loop.

Default input
-------------
<project_dir>/data/11_flattened/ml_dataset_3p0_full60D.csv
<project_dir>/data/11_flattened/feature_schema.json

Default output
--------------
<project_dir>/outputs/external_validation/retrospective_ranking/

Run
---
python retrospective_ranking.py

Quick check
-----------
python retrospective_ranking.py --max_cutoffs 2 --random_repeats 200 --bootstrap_repeats 200

Manual cutoffs
--------------
python retrospective_ranking.py --cutoffs 2018 2020 2022
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors


SEED = 42
DEFAULT_RANDOM_SEED = 20260722
DEFAULT_DPI = 600

MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
INNER_VALID_FRACTION = 0.15

DEFAULT_MAX_CUTOFFS = 6
DEFAULT_MIN_TRAIN_SAMPLES = 250
DEFAULT_MIN_FUTURE_SAMPLES = 80
DEFAULT_MIN_TRAIN_PAPERS = 100
DEFAULT_MIN_FUTURE_PAPERS = 30
DEFAULT_RANDOM_REPEATS = 1000
DEFAULT_BOOTSTRAP_REPEATS = 1000
DEFAULT_POSITIVE_FRACTION = 0.20
DEFAULT_TRAIN_THRESHOLD_QUANTILE = 0.80

PAIRWISE_MAX_PAIRS = 100000
TOP_CONDITIONS_PER_STRATEGY = 20

DATA_LEVELS = ["L1", "L2"]
MIN_COMPONENT_COUNT = 2
TARGET_NAME = "y_composite_final_score"

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

WEIGHTS = {
    "gq": 0.30,
    "igid": 0.20,
    "ori": 0.20,
    "purity": 0.15,
    "bundle": 0.15,
}

POINT_PARAMS = {
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

QUANTILE_BASE_PARAMS = {
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

QUANTILES = {
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
}

STRATEGY_ORDER = [
    "L1_point",
    "L12_q10_lower_quantile",
    "L12_q50_balanced",
    "L12_q90_potential",
]

MAIN_STRATEGY_ORDER = [
    "L1_point",
    "L12_q50_balanced",
    "L12_q90_potential",
]

STRATEGY_LABELS = {
    "L1_point": "L1 point",
    "L12_q10_lower_quantile": "q10 lower-quantile",
    "L12_q50_balanced": "q50 balanced",
    "L12_q90_potential": "q90 potential",
}

POSITIVE_DEFINITIONS = [
    "future_top_fraction",
    "above_train_threshold",
]

POSITIVE_LABELS = {
    "future_top_fraction": "Future-pool Top 20%",
    "above_train_threshold": "Above training-period P80",
}

K_SPECS = [
    ("Top 5", "absolute", 5.0),
    ("Top 10", "absolute", 10.0),
    ("Top 20", "absolute", 20.0),
    ("Top 5%", "fraction", 0.05),
    ("Top 10%", "fraction", 0.10),
    ("Top 20%", "fraction", 0.20),
]


@dataclass(frozen=True)
class Paths:
    project_dir: str
    dataset: str
    schema: str
    output_dir: str
    figures_dir: str
    source_data_dir: str
    tables_dir: str
    manifests_dir: str
    models_dir: str
    predictions: str
    ranking_results: str
    ranking_summary: str
    random_baseline: str
    bootstrap_ci: str
    cutoff_manifest: str
    model_manifest: str
    top_conditions: str
    support_summary: str
    run_summary: str
    checkpoint_config: str


def build_paths(
    project_dir: str,
    output_dir: Optional[str],
) -> Paths:
    project_dir = os.path.abspath(project_dir)

    out = (
        os.path.abspath(output_dir)
        if output_dir
        else os.path.join(
            project_dir,
            "outputs",
            "external_validation",
            "retrospective_ranking",
        )
    )

    return Paths(
        project_dir=project_dir,
        dataset=os.path.join(
            project_dir,
            "data",
            "11_flattened",
            "ml_dataset_3p0_full60D.csv",
        ),
        schema=os.path.join(
            project_dir,
            "data",
            "11_flattened",
            "feature_schema.json",
        ),
        output_dir=out,
        figures_dir=os.path.join(out, "figures"),
        source_data_dir=os.path.join(out, "source_data"),
        tables_dir=os.path.join(out, "tables"),
        manifests_dir=os.path.join(out, "manifests"),
        models_dir=os.path.join(out, "validation_models"),
        predictions=os.path.join(
            out,
            "historical_replay_predictions.csv",
        ),
        ranking_results=os.path.join(
            out,
            "retrospective_ranking_results.csv",
        ),
        ranking_summary=os.path.join(
            out,
            "retrospective_ranking_summary.csv",
        ),
        random_baseline=os.path.join(
            out,
            "random_ranking_baseline.csv",
        ),
        bootstrap_ci=os.path.join(
            out,
            "retrospective_bootstrap_ci.csv",
        ),
        cutoff_manifest=os.path.join(
            out,
            "cutoff_selection_manifest.csv",
        ),
        model_manifest=os.path.join(
            out,
            "historical_model_manifest.csv",
        ),
        top_conditions=os.path.join(
            out,
            "top_discovered_conditions.csv",
        ),
        support_summary=os.path.join(
            out,
            "support_distance_summary.csv",
        ),
        run_summary=os.path.join(
            out,
            "retrospective_analysis_summary.json",
        ),
        checkpoint_config=os.path.join(
            out,
            "retrospective_checkpoint_config.json",
        ),
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(
    frame: pd.DataFrame,
    path: str,
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    fd, temp = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".csv",
        dir=os.path.dirname(path),
    )

    os.close(fd)

    try:
        frame.to_csv(
            temp,
            index=False,
            encoding="utf-8-sig",
        )

        os.replace(
            temp,
            path,
        )

    finally:
        if os.path.exists(temp):
            os.remove(temp)


def atomic_write_json(
    data: Any,
    path: str,
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    fd, temp = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".json",
        dir=os.path.dirname(path),
    )

    os.close(fd)

    try:
        with open(
            temp,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temp,
            path,
        )

    finally:
        if os.path.exists(temp):
            os.remove(temp)


def ensure_inputs(
    paths: Paths,
) -> None:
    missing = [
        path
        for path in [
            paths.dataset,
            paths.schema,
        ]
        if not os.path.isfile(path)
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要输入文件：\n"
            + "\n".join(missing)
        )


def create_dirs(
    paths: Paths,
    save_models: bool,
) -> None:
    directories = [
        paths.output_dir,
        paths.figures_dir,
        paths.source_data_dir,
        paths.tables_dir,
        paths.manifests_dir,
    ]

    if save_models:
        directories.append(
            paths.models_dir
        )

    for directory in directories:
        os.makedirs(
            directory,
            exist_ok=True,
        )


def load_schema(
    path: str,
) -> Tuple[
    List[str],
    List[str],
    List[str],
]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    model_columns = schema.get(
        "model_feature_columns"
    )

    physical_columns = schema.get(
        "physical_feature_columns"
    )

    missingness_columns = schema.get(
        "missingness_columns"
    )

    if (
        not isinstance(model_columns, list)
        or not model_columns
    ):
        raise RuntimeError(
            "feature_schema.json 缺少 "
            "model_feature_columns。"
        )

    if (
        not isinstance(physical_columns, list)
        or not physical_columns
    ):
        physical_columns = [
            column
            for column in model_columns
            if not str(column).startswith(
                "miss_"
            )
        ]

    if (
        not isinstance(missingness_columns, list)
        or not missingness_columns
    ):
        missingness_columns = [
            column
            for column in model_columns
            if str(column).startswith(
                "miss_"
            )
        ]

    model_columns = [
        str(column)
        for column in model_columns
    ]

    physical_columns = [
        str(column)
        for column in physical_columns
    ]

    missingness_columns = [
        str(column)
        for column in missingness_columns
    ]

    column_groups = {
        "model_feature_columns": model_columns,
        "physical_feature_columns": physical_columns,
        "missingness_columns": missingness_columns,
    }

    for name, columns in column_groups.items():
        if len(columns) != len(set(columns)):
            raise RuntimeError(
                "feature_schema.json 的 "
                f"{name} 中存在重复特征。"
            )

    if not set(
        physical_columns
    ).issubset(
        model_columns
    ):
        raise RuntimeError(
            "physical_feature_columns 不是 "
            "model_feature_columns 的子集。"
        )

    if not set(
        missingness_columns
    ).issubset(
        model_columns
    ):
        raise RuntimeError(
            "missingness_columns 不是 "
            "model_feature_columns 的子集。"
        )

    return (
        model_columns,
        physical_columns,
        missingness_columns,
    )


def ensure_igid_unified(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    preferred = (
        pd.to_numeric(
            frame[PREFERRED_IGID_COL],
            errors="coerce",
        )
        if PREFERRED_IGID_COL
        in frame.columns
        else pd.Series(
            np.nan,
            index=frame.index,
        )
    )

    legacy = (
        pd.to_numeric(
            frame[LEGACY_IGID_COL],
            errors="coerce",
        )
        if LEGACY_IGID_COL
        in frame.columns
        else pd.Series(
            np.nan,
            index=frame.index,
        )
    )

    output = frame.copy()

    output[
        UNIFIED_IGID_COL
    ] = preferred.where(
        preferred.notna(),
        legacy,
    )

    return output


def load_dataset(
    path: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "data_level",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"数据集缺少元数据：{missing}"
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_features:
        raise RuntimeError(
            "数据集缺少schema特征，示例："
            f"{missing_features[:10]}"
        )

    frame = frame[
        frame[
            "data_level"
        ].astype(str).isin(
            DATA_LEVELS
        )
    ].copy()

    frame = ensure_igid_unified(
        frame
    )

    defaults = {
        "growth_mode": "",
        "source_file": "",
        "synthesis_id": "",
        "doi": "",
    }

    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame["sample_id"] = (
        frame["sample_id"]
        .fillna("")
        .astype(str)
    )

    frame["paper_id"] = (
        frame["paper_id"]
        .fillna("")
        .astype(str)
    )

    frame["paper_year_int"] = (
        pd.to_numeric(
            frame["paper_year_int"],
            errors="coerce",
        )
    )

    frame["data_level"] = (
        frame["data_level"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    if (
        frame["sample_id"]
        .str.len()
        .eq(0)
        .any()
    ):
        raise RuntimeError(
            "存在空 sample_id。"
        )

    if (
        frame["paper_id"]
        .str.len()
        .eq(0)
        .any()
    ):
        raise RuntimeError(
            "存在空 paper_id。"
        )

    if frame[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            "sample_id 不唯一。"
        )

    for column in feature_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return frame.reset_index(
        drop=True
    )


def prepare_composite_base(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    work = ensure_igid_unified(
        dataset
    ).copy()

    for column in [
        COL_GQ,
        UNIFIED_IGID_COL,
        COL_ORI,
        COL_PURITY,
        COL_BUNDLE,
    ]:
        if column not in work.columns:
            work[column] = np.nan

    work[RAW_GQ] = pd.to_numeric(
        work[COL_GQ],
        errors="coerce",
    )

    work[RAW_IGID] = pd.to_numeric(
        work[UNIFIED_IGID_COL],
        errors="coerce",
    )

    work[RAW_ORI] = pd.to_numeric(
        work[COL_ORI],
        errors="coerce",
    )

    work[RAW_PURITY] = pd.to_numeric(
        work[COL_PURITY],
        errors="coerce",
    )

    work[RAW_BUNDLE] = pd.to_numeric(
        work[COL_BUNDLE],
        errors="coerce",
    )

    bundle = work[
        RAW_BUNDLE
    ].to_numpy(dtype=float)

    work[RAW_BUNDLE_LOG] = np.where(
        np.isfinite(bundle)
        & (bundle >= 0.0),
        np.log1p(bundle),
        np.nan,
    )

    availability = pd.DataFrame(
        {
            "gq": work[RAW_GQ].notna(),
            "igid": work[RAW_IGID].notna(),
            "ori": work[RAW_ORI].notna(),
            "purity": work[RAW_PURITY].notna(),
            "bundle": work[
                RAW_BUNDLE_LOG
            ].notna(),
        }
    )

    work[
        RAW_COMPONENT_COUNT
    ] = (
        availability
        .sum(axis=1)
        .astype(int)
    )

    return (
        work[
            work[
                RAW_COMPONENT_COUNT
            ]
            >= MIN_COMPONENT_COUNT
        ]
        .copy()
        .reset_index(drop=True)
    )


def minmax_parameters(
    series: pd.Series,
) -> Dict[str, Any]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return {
            "minimum": None,
            "maximum": None,
            "usable": False,
        }

    minimum = float(
        np.min(finite)
    )

    maximum = float(
        np.max(finite)
    )

    return {
        "minimum": minimum,
        "maximum": maximum,
        "usable": not math.isclose(
            minimum,
            maximum,
            abs_tol=1e-15,
        ),
    }


def fit_normalizer(
    training: pd.DataFrame,
) -> Dict[str, Any]:
    igid = minmax_parameters(
        training[RAW_IGID]
    )

    bundle = minmax_parameters(
        training[RAW_BUNDLE_LOG]
    )

    return {
        "igid_min": igid["minimum"],
        "igid_max": igid["maximum"],
        "igid_usable": bool(
            igid["usable"]
        ),
        "bundle_log_min": bundle["minimum"],
        "bundle_log_max": bundle["maximum"],
        "bundle_usable": bool(
            bundle["usable"]
        ),
    }


def apply_minmax(
    series: pd.Series,
    minimum: Any,
    maximum: Any,
    usable: bool,
) -> pd.Series:
    if (
        not usable
        or minimum is None
        or maximum is None
        or math.isclose(
            minimum,
            maximum,
            abs_tol=1e-15,
        )
    ):
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    output = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        - minimum
    ) / (
        maximum
        - minimum
    )

    return output.clip(
        0.0,
        1.0,
    )


def transform_composite(
    work: pd.DataFrame,
    normalizer: Mapping[str, Any],
) -> pd.DataFrame:
    output = pd.DataFrame(
        index=work.index
    )

    output["gq"] = (
        pd.to_numeric(
            work[RAW_GQ],
            errors="coerce",
        )
        / 10.0
    ).clip(
        0.0,
        1.0,
    )

    output["igid"] = apply_minmax(
        work[RAW_IGID],
        normalizer.get("igid_min"),
        normalizer.get("igid_max"),
        bool(
            normalizer.get(
                "igid_usable"
            )
        ),
    )

    output["ori"] = pd.to_numeric(
        work[RAW_ORI],
        errors="coerce",
    ).clip(
        0.0,
        1.0,
    )

    output["purity"] = pd.to_numeric(
        work[RAW_PURITY],
        errors="coerce",
    ).clip(
        0.0,
        1.0,
    )

    output["bundle"] = apply_minmax(
        work[RAW_BUNDLE_LOG],
        normalizer.get(
            "bundle_log_min"
        ),
        normalizer.get(
            "bundle_log_max"
        ),
        bool(
            normalizer.get(
                "bundle_usable"
            )
        ),
    )

    components = output[
        [
            "gq",
            "igid",
            "ori",
            "purity",
            "bundle",
        ]
    ]

    weights = pd.Series(
        WEIGHTS,
        dtype=float,
    )

    available = components.notna()

    component_count = (
        available
        .sum(axis=1)
        .astype(int)
    )

    weight_sum = (
        available
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
    )

    numerator = (
        components
        .mul(
            weights,
            axis=1,
        )
        .sum(
            axis=1,
            skipna=True,
        )
    )

    composite = pd.Series(
        np.where(
            weight_sum > 0,
            numerator / weight_sum,
            np.nan,
        ),
        index=work.index,
        dtype=float,
    ).where(
        component_count
        >= MIN_COMPONENT_COUNT,
        np.nan,
    )

    output[
        TARGET_NAME
    ] = composite

    output[
        "component_count"
    ] = component_count

    output[
        "weight_sum_used"
    ] = weight_sum.where(
        composite.notna(),
        np.nan,
    )

    return output


def cutoff_manifest(
    base: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    years = sorted(
        int(year)
        for year in base[
            "paper_year_int"
        ].dropna().unique()
    )

    rows = []

    for year in years[:-1]:
        train = base[
            base[
                "paper_year_int"
            ] <= year
        ]

        future = base[
            base[
                "paper_year_int"
            ] > year
        ]

        rows.append(
            {
                "cutoff_year": year,
                "n_train_raw": len(
                    train
                ),
                "n_future_raw": len(
                    future
                ),
                "n_train_papers": train[
                    "paper_id"
                ].nunique(),
                "n_future_papers": future[
                    "paper_id"
                ].nunique(),
                "eligible": (
                    len(train)
                    >= args.min_train_samples
                    and len(future)
                    >= args.min_future_samples
                    and train[
                        "paper_id"
                    ].nunique()
                    >= args.min_train_papers
                    and future[
                        "paper_id"
                    ].nunique()
                    >= args.min_future_papers
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def choose_cutoffs(
    manifest: pd.DataFrame,
    requested: Optional[
        Sequence[int]
    ],
    max_count: int,
) -> List[int]:
    eligible = (
        manifest.loc[
            manifest[
                "eligible"
            ],
            "cutoff_year",
        ]
        .astype(int)
        .tolist()
    )

    if not eligible:
        raise RuntimeError(
            "没有年份满足当前训练/未来样本和论文数量要求。"
        )

    if requested:
        requested = sorted(
            set(
                int(value)
                for value in requested
            )
        )

        invalid = [
            value
            for value in requested
            if value not in eligible
        ]

        if invalid:
            raise ValueError(
                "这些cutoff不满足条件："
                f"{invalid}；"
                f"可选：{eligible}"
            )

        return requested

    if len(
        eligible
    ) <= max_count:
        return eligible

    indices = sorted(
        set(
            int(
                round(value)
            )
            for value in np.linspace(
                0,
                len(eligible) - 1,
                max_count,
            )
        )
    )

    return [
        eligible[index]
        for index in indices
    ]


def inner_split(
    training: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    str,
]:
    years = pd.to_numeric(
        training[
            "paper_year_int"
        ],
        errors="coerce",
    )

    unique_years = sorted(
        int(year)
        for year in years.dropna().unique()
    )

    target_valid = max(
        20,
        int(
            math.ceil(
                INNER_VALID_FRACTION
                * len(training)
            )
        ),
    )

    selected: List[int] = []
    count = 0

    for year in reversed(
        unique_years
    ):
        selected.append(
            year
        )

        count += int(
            (
                years == year
            ).sum()
        )

        fit_mask = (
            years
            < min(selected)
        )

        valid_mask = years.isin(
            selected
        )

        if (
            count >= target_valid
            and fit_mask.sum() >= 30
            and training.loc[
                fit_mask,
                "paper_id",
            ].nunique() >= 3
            and training.loc[
                valid_mask,
                "paper_id",
            ].nunique() >= 2
        ):
            return (
                np.where(
                    fit_mask
                )[0],
                np.where(
                    valid_mask
                )[0],
                "inner_temporal_year_block",
            )

    groups = training[
        "paper_id"
    ].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=INNER_VALID_FRACTION,
        random_state=SEED,
    )

    fit, valid = next(
        splitter.split(
            np.zeros(
                len(training)
            ),
            groups=groups.to_numpy(),
        )
    )

    return (
        np.asarray(
            fit
        ),
        np.asarray(
            valid
        ),
        "inner_group_shuffle_fallback",
    )


def best_iteration(
    X: pd.DataFrame,
    y: np.ndarray,
    metadata: pd.DataFrame,
    params: Mapping[str, Any],
) -> Tuple[
    int,
    str,
]:
    fit, valid, method = inner_split(
        metadata
    )

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
        dict(params),
        train_set,
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
            lgb.log_evaluation(0),
        ],
    )

    iteration = int(
        model.best_iteration
        if (
            model.best_iteration
            and model.best_iteration > 0
        )
        else MAX_BOOST_ROUNDS
    )

    return (
        iteration,
        method,
    )


def fit_final(
    X: pd.DataFrame,
    y: np.ndarray,
    params: Mapping[str, Any],
    iteration: int,
) -> lgb.Booster:
    data = lgb.Dataset(
        X,
        label=y,
        free_raw_data=False,
    )

    return lgb.train(
        dict(params),
        data,
        num_boost_round=int(
            iteration
        ),
        callbacks=[
            lgb.log_evaluation(0)
        ],
    )


def save_model(
    model: lgb.Booster,
    paths: Paths,
    cutoff: int,
    name: str,
) -> str:
    directory = os.path.join(
        paths.models_dir,
        f"cutoff_{cutoff}",
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    path = os.path.join(
        directory,
        f"{name}.txt",
    )

    model.save_model(
        path
    )

    return path


def support_distance(
    training: pd.DataFrame,
    future: pd.DataFrame,
    physical_features: Sequence[str],
    missingness_features: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    train_physical = (
        training[
            list(
                physical_features
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    test_physical = (
        future[
            list(
                physical_features
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    for feature in physical_features:
        miss = (
            f"miss_{feature}"
        )

        if miss in training.columns:
            train_missing_mask = (
                pd.to_numeric(
                    training[miss],
                    errors="coerce",
                )
                .fillna(1)
                .gt(0)
            )

            train_physical.loc[
                train_missing_mask,
                feature,
            ] = np.nan

        if miss in future.columns:
            test_missing_mask = (
                pd.to_numeric(
                    future[miss],
                    errors="coerce",
                )
                .fillna(1)
                .gt(0)
            )

            test_physical.loc[
                test_missing_mask,
                feature,
            ] = np.nan

    median = (
        train_physical
        .median()
        .fillna(0.0)
    )

    train_physical = (
        train_physical
        .fillna(median)
    )

    test_physical = (
        test_physical
        .fillna(median)
    )

    scale = (
        train_physical
        .quantile(0.75)
        - train_physical
        .quantile(0.25)
    ).abs()

    scale = scale.where(
        scale > 1e-12,
        train_physical.std(
            ddof=0
        ),
    )

    scale = scale.where(
        scale > 1e-12,
        1.0,
    )

    train_scaled = (
        (
            train_physical
            - median
        )
        / scale
    ).to_numpy(
        dtype=float
    )

    test_scaled = (
        (
            test_physical
            - median
        )
        / scale
    ).to_numpy(
        dtype=float
    )

    physical_nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    physical_nn.fit(
        train_scaled
    )

    (
        physical_distance,
        physical_index,
    ) = physical_nn.kneighbors(
        test_scaled
    )

    physical_nearest = (
        training.iloc[
            physical_index[:, 0]
        ]
        .reset_index(
            drop=True
        )
    )

    train_missing = (
        training[
            list(
                missingness_features
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(1.0)
        .gt(0)
        .astype(float)
        .to_numpy()
    )

    test_missing = (
        future[
            list(
                missingness_features
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(1.0)
        .gt(0)
        .astype(float)
        .to_numpy()
    )

    missing_nn = NearestNeighbors(
        n_neighbors=1,
        metric="hamming",
    )

    missing_nn.fit(
        train_missing
    )

    (
        missing_distance,
        missing_index,
    ) = missing_nn.kneighbors(
        test_missing
    )

    missing_nearest = (
        training.iloc[
            missing_index[:, 0]
        ]
        .reset_index(
            drop=True
        )
    )

    return pd.DataFrame(
        {
            f"{prefix}_physical_support_distance":
                physical_distance[:, 0],

            f"{prefix}_physical_nearest_sample_id":
                physical_nearest[
                    "sample_id"
                ]
                .astype(str)
                .to_numpy(),

            f"{prefix}_physical_nearest_paper_id":
                physical_nearest[
                    "paper_id"
                ]
                .astype(str)
                .to_numpy(),

            f"{prefix}_physical_nearest_year":
                physical_nearest[
                    "paper_year_int"
                ].to_numpy(),

            f"{prefix}_missingness_distance":
                missing_distance[:, 0],

            f"{prefix}_missingness_nearest_sample_id":
                missing_nearest[
                    "sample_id"
                ]
                .astype(str)
                .to_numpy(),

            f"{prefix}_missingness_nearest_paper_id":
                missing_nearest[
                    "paper_id"
                ]
                .astype(str)
                .to_numpy(),
        }
    )


def strategy_support_columns(
    strategy: str,
) -> Tuple[
    str,
    str,
]:
    prefix = (
        "L1"
        if strategy == "L1_point"
        else "L12"
    )

    return (
        f"{prefix}_physical_support_distance",
        f"{prefix}_missingness_distance",
    )


def replay_cutoff(
    base: pd.DataFrame,
    features: Sequence[str],
    physical_features: Sequence[str],
    missingness_features: Sequence[str],
    cutoff: int,
    positive_fraction: float,
    train_threshold_quantile: float,
    paths: Paths,
    save_models_flag: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    early_raw = base[
        base[
            "paper_year_int"
        ] <= cutoff
    ].copy()

    normalizer = fit_normalizer(
        early_raw
    )

    transformed = transform_composite(
        base,
        normalizer,
    )

    work = base.copy()

    work[
        TARGET_NAME
    ] = transformed[
        TARGET_NAME
    ].to_numpy(
        dtype=float
    )

    work[
        "component_count_replay"
    ] = transformed[
        "component_count"
    ].to_numpy(
        dtype=float
    )

    work[
        "weight_sum_used_replay"
    ] = transformed[
        "weight_sum_used"
    ].to_numpy(
        dtype=float
    )

    work = work[
        work[
            TARGET_NAME
        ].notna()
        & work[
            "paper_year_int"
        ].notna()
    ].copy()

    early = work[
        work[
            "paper_year_int"
        ] <= cutoff
    ].copy()

    future = work[
        work[
            "paper_year_int"
        ] > cutoff
    ].copy()

    point_train = early[
        early[
            "data_level"
        ] == "L1"
    ].copy()

    quantile_train = early.copy()

    if (
        len(point_train) < 30
        or len(future) == 0
    ):
        raise RuntimeError(
            f"cutoff={cutoff} "
            "缺少足够训练或未来样本。"
        )

    X_point = point_train[
        list(features)
    ]

    y_point = point_train[
        TARGET_NAME
    ].to_numpy(
        dtype=float
    )

    (
        point_iter,
        point_method,
    ) = best_iteration(
        X_point,
        y_point,
        point_train,
        POINT_PARAMS,
    )

    point_model = fit_final(
        X_point,
        y_point,
        POINT_PARAMS,
        point_iter,
    )

    model_rows = []

    point_path = (
        save_model(
            point_model,
            paths,
            cutoff,
            "L1_point",
        )
        if save_models_flag
        else ""
    )

    model_rows.append(
        {
            "cutoff_year": cutoff,
            "model_name": "L1_point",
            "training_levels": "L1",
            "n_train_samples": len(
                point_train
            ),
            "n_train_papers": point_train[
                "paper_id"
            ].nunique(),
            "best_iteration": point_iter,
            "inner_split_method": point_method,
            "model_path": point_path,
            "normalizer_json": json.dumps(
                normalizer,
                ensure_ascii=False,
            ),
        }
    )

    X_quantile = quantile_train[
        list(features)
    ]

    y_quantile = quantile_train[
        TARGET_NAME
    ].to_numpy(
        dtype=float
    )

    X_future = future[
        list(features)
    ]

    point_prediction = (
        point_model.predict(
            X_future,
            num_iteration=point_iter,
        )
    )

    quantile_predictions: Dict[
        str,
        np.ndarray,
    ] = {}

    for name, alpha in QUANTILES.items():
        params = dict(
            QUANTILE_BASE_PARAMS
        )

        params[
            "alpha"
        ] = alpha

        (
            iteration,
            method,
        ) = best_iteration(
            X_quantile,
            y_quantile,
            quantile_train,
            params,
        )

        model = fit_final(
            X_quantile,
            y_quantile,
            params,
            iteration,
        )

        quantile_predictions[
            name
        ] = np.asarray(
            model.predict(
                X_future,
                num_iteration=iteration,
            ),
            dtype=float,
        )

        model_path = (
            save_model(
                model,
                paths,
                cutoff,
                f"L12_{name}",
            )
            if save_models_flag
            else ""
        )

        model_rows.append(
            {
                "cutoff_year": cutoff,
                "model_name": f"L12_{name}",
                "training_levels": "L1+L2",
                "n_train_samples": len(
                    quantile_train
                ),
                "n_train_papers": quantile_train[
                    "paper_id"
                ].nunique(),
                "best_iteration": iteration,
                "inner_split_method": method,
                "model_path": model_path,
                "normalizer_json": json.dumps(
                    normalizer,
                    ensure_ascii=False,
                ),
            }
        )

    ordered = np.sort(
        np.column_stack(
            [
                quantile_predictions[
                    "q10"
                ],
                quantile_predictions[
                    "q50"
                ],
                quantile_predictions[
                    "q90"
                ],
            ]
        ),
        axis=1,
    )

    width = (
        ordered[:, 2]
        - ordered[:, 0]
    )

    support_l1 = support_distance(
        point_train,
        future,
        physical_features,
        missingness_features,
        "L1",
    )

    support_l12 = support_distance(
        quantile_train,
        future,
        physical_features,
        missingness_features,
        "L12",
    )

    keep = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "data_level",
        "growth_mode",
        "source_file",
        "synthesis_id",
        "doi",
        TARGET_NAME,
        "component_count_replay",
        "weight_sum_used_replay",
    ]

    output = (
        future[
            keep
        ]
        .reset_index(
            drop=True
        )
    )

    output = pd.concat(
        [
            output,
            support_l1.reset_index(
                drop=True
            ),
            support_l12.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output[
        "cutoff_year"
    ] = cutoff

    output[
        "y_true_replay"
    ] = output[
        TARGET_NAME
    ]

    output[
        "pred_L1_point"
    ] = np.asarray(
        point_prediction,
        dtype=float,
    )

    output[
        "pred_q10_raw"
    ] = quantile_predictions[
        "q10"
    ]

    output[
        "pred_q50_raw"
    ] = quantile_predictions[
        "q50"
    ]

    output[
        "pred_q90_raw"
    ] = quantile_predictions[
        "q90"
    ]

    output[
        "pred_q10_ordered"
    ] = ordered[:, 0]

    output[
        "pred_q50_ordered"
    ] = ordered[:, 1]

    output[
        "pred_q90_ordered"
    ] = ordered[:, 2]

    output[
        "prediction_interval_width"
    ] = width

    output[
        "quantile_crossing"
    ] = (
        (
            quantile_predictions[
                "q10"
            ]
            > quantile_predictions[
                "q50"
            ]
        )
        | (
            quantile_predictions[
                "q50"
            ]
            > quantile_predictions[
                "q90"
            ]
        )
    )

    future_threshold = float(
        output[
            "y_true_replay"
        ].quantile(
            1.0
            - positive_fraction
        )
    )

    train_threshold = float(
        early[
            TARGET_NAME
        ].quantile(
            train_threshold_quantile
        )
    )

    output[
        "positive_future_top_fraction"
    ] = (
        output[
            "y_true_replay"
        ]
        >= future_threshold
    )

    output[
        "positive_above_train_threshold"
    ] = (
        output[
            "y_true_replay"
        ]
        >= train_threshold
    )

    output[
        "future_top_fraction_threshold"
    ] = future_threshold

    output[
        "training_period_threshold"
    ] = train_threshold

    output[
        "true_rank_desc"
    ] = output[
        "y_true_replay"
    ].rank(
        method="average",
        ascending=False,
    )

    output[
        "true_percentile_from_top"
    ] = (
        output[
            "true_rank_desc"
        ]
        / len(output)
    )

    output[
        "true_relevance_rank_percentile"
    ] = (
        1.0
        - (
            output[
                "true_rank_desc"
            ]
            - 1.0
        )
        / max(
            1.0,
            len(output) - 1.0,
        )
    )

    return (
        output,
        pd.DataFrame(
            model_rows
        ),
    )


def strategy_columns() -> Dict[
    str,
    str,
]:
    return {
        "L1_point":
            "pred_L1_point",

        "L12_q10_lower_quantile":
            "pred_q10_raw",

        "L12_q50_balanced":
            "pred_q50_raw",

        "L12_q90_potential":
            "pred_q90_raw",
    }


def positive_column(
    definition: str,
) -> str:
    return {
        "future_top_fraction":
            "positive_future_top_fraction",

        "above_train_threshold":
            "positive_above_train_threshold",
    }[
        definition
    ]


def resolve_k(
    pool_size: int,
) -> List[
    Tuple[
        str,
        int,
    ]
]:
    result = []

    for (
        label,
        kind,
        value,
    ) in K_SPECS:
        k = (
            int(value)
            if kind == "absolute"
            else int(
                math.ceil(
                    value
                    * pool_size
                )
            )
        )

        result.append(
            (
                label,
                max(
                    1,
                    min(
                        pool_size,
                        k,
                    ),
                ),
            )
        )

    return result


def dcg(
    relevance: np.ndarray,
    k: int,
) -> float:
    relevance = np.asarray(
        relevance,
        dtype=float,
    )[:k]

    if len(relevance) == 0:
        return np.nan

    discounts = np.log2(
        np.arange(
            2,
            len(relevance) + 2,
            dtype=float,
        )
    )

    gains = (
        np.power(
            2.0,
            relevance,
        )
        - 1.0
    )

    return float(
        np.sum(
            gains
            / discounts
        )
    )


def ndcg(
    ranked_relevance: np.ndarray,
    all_relevance: np.ndarray,
    k: int,
) -> float:
    actual = dcg(
        ranked_relevance,
        k,
    )

    ideal = dcg(
        np.sort(
            np.asarray(
                all_relevance,
                dtype=float,
            )
        )[::-1],
        k,
    )

    if (
        np.isfinite(actual)
        and np.isfinite(ideal)
        and ideal > 0
    ):
        return float(
            actual
            / ideal
        )

    return np.nan


def safe_spearman(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    x_array = np.asarray(
        x,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=float,
    )

    finite = (
        np.isfinite(
            x_array
        )
        & np.isfinite(
            y_array
        )
    )

    x_array = x_array[
        finite
    ]

    y_array = y_array[
        finite
    ]

    if len(x_array) < 2:
        return np.nan

    x_rank = pd.Series(
        x_array
    ).rank(
        method="average"
    ).to_numpy(
        dtype=float
    )

    y_rank = pd.Series(
        y_array
    ).rank(
        method="average"
    ).to_numpy(
        dtype=float
    )

    if (
        np.std(x_rank) == 0
        or np.std(y_rank) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            x_rank,
            y_rank,
        )[0, 1]
    )


def pairwise_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
) -> Tuple[
    float,
    int,
]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    n = len(
        y_true
    )

    if n < 2:
        return (
            np.nan,
            0,
        )

    total = (
        n
        * (
            n - 1
        )
        // 2
    )

    rng = np.random.default_rng(
        seed
    )

    if total <= PAIRWISE_MAX_PAIRS:
        (
            left,
            right,
        ) = np.triu_indices(
            n,
            k=1,
        )

    else:
        left = rng.integers(
            0,
            n,
            size=(
                PAIRWISE_MAX_PAIRS
                * 2
            ),
        )

        right = rng.integers(
            0,
            n,
            size=(
                PAIRWISE_MAX_PAIRS
                * 2
            ),
        )

        mask = (
            left
            != right
        )

        left = left[
            mask
        ][
            :PAIRWISE_MAX_PAIRS
        ]

        right = right[
            mask
        ][
            :PAIRWISE_MAX_PAIRS
        ]

    true_diff = (
        y_true[left]
        - y_true[right]
    )

    pred_diff = (
        y_pred[left]
        - y_pred[right]
    )

    valid = (
        np.isfinite(
            true_diff
        )
        & np.isfinite(
            pred_diff
        )
        & (
            true_diff
            != 0
        )
        & (
            pred_diff
            != 0
        )
    )

    if not valid.any():
        return (
            np.nan,
            0,
        )

    accuracy = np.mean(
        np.sign(
            true_diff[
                valid
            ]
        )
        == np.sign(
            pred_diff[
                valid
            ]
        )
    )

    return (
        float(
            accuracy
        ),
        int(
            valid.sum()
        ),
    )


def evaluate_strategy(
    frame: pd.DataFrame,
    cutoff: int,
    strategy: str,
    definition: str,
    seed: int,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    score_column = (
        strategy_columns()[
            strategy
        ]
    )

    ranked = (
        frame
        .sort_values(
            score_column,
            ascending=False,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    positives = (
        ranked[
            positive_column(
                definition
            )
        ]
        .astype(bool)
        .to_numpy()
    )

    y_true = ranked[
        "y_true_replay"
    ].to_numpy(
        dtype=float
    )

    score = ranked[
        score_column
    ].to_numpy(
        dtype=float
    )

    continuous = ranked[
        "true_relevance_rank_percentile"
    ].to_numpy(
        dtype=float
    )

    total_positive = int(
        positives.sum()
    )

    prevalence = (
        total_positive
        / len(ranked)
        if len(ranked)
        else np.nan
    )

    (
        pairwise,
        valid_pairs,
    ) = pairwise_accuracy(
        y_true,
        score,
        seed,
    )

    spearman = safe_spearman(
        y_true,
        score,
    )

    ranked[
        "predicted_rank"
    ] = np.arange(
        1,
        len(ranked) + 1,
    )

    ranked[
        "predicted_percentile_from_top"
    ] = (
        ranked[
            "predicted_rank"
        ]
        / len(ranked)
    )

    true_best_id = (
        ranked
        .sort_values(
            "y_true_replay",
            ascending=False,
        )
        .iloc[0][
            "sample_id"
        ]
    )

    true_top1_percentile = float(
        ranked.loc[
            ranked[
                "sample_id"
            ] == true_best_id,
            "predicted_percentile_from_top",
        ].iloc[0]
    )

    (
        physical_support,
        missingness_support,
    ) = strategy_support_columns(
        strategy
    )

    rows = []

    for (
        k_label,
        k,
    ) in resolve_k(
        len(ranked)
    ):
        top = ranked.iloc[
            :k
        ]

        hits = int(
            top[
                positive_column(
                    definition
                )
            ].sum()
        )

        precision = (
            hits
            / k
        )

        recall = (
            hits
            / total_positive
            if total_positive > 0
            else np.nan
        )

        enrichment = (
            precision
            / prevalence
            if (
                np.isfinite(
                    prevalence
                )
                and prevalence > 0
            )
            else np.nan
        )

        rows.append(
            {
                "cutoff_year": cutoff,
                "strategy": strategy,
                "strategy_label":
                    STRATEGY_LABELS[
                        strategy
                    ],
                "score_column":
                    score_column,
                "positive_definition":
                    definition,
                "positive_label":
                    POSITIVE_LABELS[
                        definition
                    ],
                "k_label":
                    k_label,
                "k":
                    k,
                "pool_size":
                    len(ranked),
                "n_future_papers":
                    ranked[
                        "paper_id"
                    ].nunique(),
                "total_positives":
                    total_positive,
                "positive_prevalence":
                    prevalence,
                "hits_at_k":
                    hits,
                "precision_at_k":
                    precision,
                "recall_at_k":
                    recall,
                "hit_rate_at_k":
                    float(
                        hits > 0
                    ),
                "enrichment_factor_at_k":
                    enrichment,
                "ndcg_binary_at_k":
                    ndcg(
                        positives.astype(
                            float
                        ),
                        positives.astype(
                            float
                        ),
                        k,
                    ),
                "ndcg_continuous_at_k":
                    ndcg(
                        continuous,
                        frame[
                            "true_relevance_rank_percentile"
                        ].to_numpy(
                            dtype=float
                        ),
                        k,
                    ),
                "spearman_all_future":
                    spearman,
                "pairwise_ranking_accuracy":
                    pairwise,
                "pairwise_valid_pairs":
                    valid_pairs,
                "true_top1_predicted_percentile_from_top":
                    true_top1_percentile,
                "mean_physical_support_distance_top_k":
                    float(
                        top[
                            physical_support
                        ].mean()
                    ),
                "median_physical_support_distance_top_k":
                    float(
                        top[
                            physical_support
                        ].median()
                    ),
                "mean_missingness_distance_top_k":
                    float(
                        top[
                            missingness_support
                        ].mean()
                    ),
                "median_missingness_distance_top_k":
                    float(
                        top[
                            missingness_support
                        ].median()
                    ),
            }
        )

    return rows


BOOTSTRAP_METRICS = [
    "ndcg_continuous_at_k",
    "recall_at_k",
    "enrichment_factor_at_k",
]


def refresh_bootstrap_labels(
    frame: pd.DataFrame,
    positive_fraction: float,
) -> pd.DataFrame:
    output = frame.copy()

    future_threshold = float(
        output[
            "y_true_replay"
        ].quantile(
            1.0
            - positive_fraction
        )
    )

    output[
        "positive_future_top_fraction"
    ] = (
        output[
            "y_true_replay"
        ]
        >= future_threshold
    )

    output[
        "positive_above_train_threshold"
    ] = (
        output[
            "y_true_replay"
        ]
        >= output[
            "training_period_threshold"
        ]
    )

    output[
        "true_rank_desc"
    ] = (
        output[
            "y_true_replay"
        ].rank(
            method="average",
            ascending=False,
        )
    )

    output[
        "true_relevance_rank_percentile"
    ] = (
        1.0
        - (
            output[
                "true_rank_desc"
            ]
            - 1.0
        )
        / max(
            1.0,
            len(output) - 1.0,
        )
    )

    return output


def bootstrap_metric_rows(
    frame: pd.DataFrame,
    strategy: str,
    definition: str,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    score_column = (
        strategy_columns()[
            strategy
        ]
    )

    ranked = (
        frame
        .sort_values(
            score_column,
            ascending=False,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    positives = (
        ranked[
            positive_column(
                definition
            )
        ]
        .astype(bool)
        .to_numpy()
    )

    continuous = ranked[
        "true_relevance_rank_percentile"
    ].to_numpy(
        dtype=float
    )

    total_positive = int(
        positives.sum()
    )

    prevalence = (
        total_positive
        / len(ranked)
        if len(ranked)
        else np.nan
    )

    rows = []

    for (
        k_label,
        k,
    ) in resolve_k(
        len(ranked)
    ):
        hits = int(
            positives[:k].sum()
        )

        precision = (
            hits
            / k
        )

        rows.append(
            {
                "k_label":
                    k_label,
                "k":
                    k,
                "ndcg_continuous_at_k":
                    ndcg(
                        continuous,
                        frame[
                            "true_relevance_rank_percentile"
                        ].to_numpy(
                            dtype=float
                        ),
                        k,
                    ),
                "recall_at_k":
                    (
                        hits
                        / total_positive
                        if total_positive > 0
                        else np.nan
                    ),
                "enrichment_factor_at_k":
                    (
                        precision
                        / prevalence
                        if (
                            np.isfinite(
                                prevalence
                            )
                            and prevalence > 0
                        )
                        else np.nan
                    ),
            }
        )

    return rows


def bootstrap_cutoff_by_paper(
    frame: pd.DataFrame,
    cutoff: int,
    definition: str,
    repeats: int,
    positive_fraction: float,
    seed: int,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    paper_groups = {
        str(paper_id):
            group.copy()
        for (
            paper_id,
            group,
        ) in frame.groupby(
            "paper_id",
            sort=False,
        )
    }

    papers = list(
        paper_groups
    )

    if len(papers) < 2:
        return []

    point_rows = {
        strategy:
            bootstrap_metric_rows(
                frame,
                strategy,
                definition,
            )
        for strategy
        in STRATEGY_ORDER
    }

    distributions = {
        (
            strategy,
            row[
                "k_label"
            ],
            metric,
        ): []
        for (
            strategy,
            rows,
        ) in point_rows.items()
        for row in rows
        for metric in BOOTSTRAP_METRICS
    }

    rng = np.random.default_rng(
        seed
    )

    for _ in range(
        repeats
    ):
        sampled_papers = rng.choice(
            papers,
            size=len(papers),
            replace=True,
        )

        sampled = pd.concat(
            [
                paper_groups[
                    str(paper_id)
                ]
                for paper_id
                in sampled_papers
            ],
            ignore_index=True,
        )

        sampled = (
            refresh_bootstrap_labels(
                sampled,
                positive_fraction,
            )
        )

        for strategy in STRATEGY_ORDER:
            sampled_rows = (
                bootstrap_metric_rows(
                    sampled,
                    strategy,
                    definition,
                )
            )

            for row in sampled_rows:
                for metric in BOOTSTRAP_METRICS:
                    value = row[
                        metric
                    ]

                    if np.isfinite(
                        value
                    ):
                        distributions[
                            (
                                strategy,
                                row[
                                    "k_label"
                                ],
                                metric,
                            )
                        ].append(
                            float(
                                value
                            )
                        )

    output = []

    for (
        strategy,
        rows,
    ) in point_rows.items():
        for row in rows:
            for metric in BOOTSTRAP_METRICS:
                values = np.asarray(
                    distributions[
                        (
                            strategy,
                            row[
                                "k_label"
                            ],
                            metric,
                        )
                    ],
                    dtype=float,
                )

                output.append(
                    {
                        "cutoff_year":
                            cutoff,
                        "strategy":
                            strategy,
                        "strategy_label":
                            STRATEGY_LABELS[
                                strategy
                            ],
                        "positive_definition":
                            definition,
                        "k_label":
                            row[
                                "k_label"
                            ],
                        "metric":
                            metric,
                        "estimate":
                            row[
                                metric
                            ],
                        "ci_low":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.025,
                                    )
                                )
                                if len(values)
                                else np.nan
                            ),
                        "ci_high":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.975,
                                    )
                                )
                                if len(values)
                                else np.nan
                            ),
                        "valid_repeats":
                            int(
                                len(values)
                            ),
                        "future_samples":
                            len(frame),
                        "future_papers":
                            len(papers),
                    }
                )

    return output


def random_baseline(
    frame: pd.DataFrame,
    cutoff: int,
    definition: str,
    repeats: int,
    seed: int,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    positive = (
        frame[
            positive_column(
                definition
            )
        ]
        .astype(bool)
        .to_numpy()
    )

    continuous = frame[
        "true_relevance_rank_percentile"
    ].to_numpy(
        dtype=float
    )

    n = len(
        frame
    )

    total_positive = int(
        positive.sum()
    )

    prevalence = (
        total_positive
        / n
        if n
        else np.nan
    )

    k_values = resolve_k(
        n
    )

    metrics = [
        "precision",
        "recall",
        "hit_rate",
        "enrichment",
        "ndcg_binary",
        "ndcg_continuous",
    ]

    store = {
        (
            label,
            k,
        ): {
            metric: []
            for metric
            in metrics
        }
        for (
            label,
            k,
        ) in k_values
    }

    rng = np.random.default_rng(
        seed
    )

    for _ in range(
        repeats
    ):
        order = rng.permutation(
            n
        )

        p = positive[
            order
        ]

        c = continuous[
            order
        ]

        for (
            label,
            k,
        ) in k_values:
            hits = int(
                p[
                    :k
                ].sum()
            )

            precision = (
                hits
                / k
            )

            recall = (
                hits
                / total_positive
                if total_positive > 0
                else np.nan
            )

            enrichment = (
                precision
                / prevalence
                if (
                    np.isfinite(
                        prevalence
                    )
                    and prevalence > 0
                )
                else np.nan
            )

            values = store[
                (
                    label,
                    k,
                )
            ]

            values[
                "precision"
            ].append(
                precision
            )

            values[
                "recall"
            ].append(
                recall
            )

            values[
                "hit_rate"
            ].append(
                float(
                    hits > 0
                )
            )

            values[
                "enrichment"
            ].append(
                enrichment
            )

            values[
                "ndcg_binary"
            ].append(
                ndcg(
                    p.astype(
                        float
                    ),
                    positive.astype(
                        float
                    ),
                    k,
                )
            )

            values[
                "ndcg_continuous"
            ].append(
                ndcg(
                    c,
                    continuous,
                    k,
                )
            )

    rows = []

    for (
        label,
        k,
    ) in k_values:
        row = {
            "cutoff_year":
                cutoff,
            "positive_definition":
                definition,
            "positive_label":
                POSITIVE_LABELS[
                    definition
                ],
            "k_label":
                label,
            "k":
                k,
            "pool_size":
                n,
            "total_positives":
                total_positive,
            "positive_prevalence":
                prevalence,
            "random_repeats":
                repeats,
        }

        output_mapping = [
            (
                "precision",
                "precision_at_k",
            ),
            (
                "recall",
                "recall_at_k",
            ),
            (
                "hit_rate",
                "hit_rate_at_k",
            ),
            (
                "enrichment",
                "enrichment_factor_at_k",
            ),
            (
                "ndcg_binary",
                "ndcg_binary_at_k",
            ),
            (
                "ndcg_continuous",
                "ndcg_continuous_at_k",
            ),
        ]

        for (
            metric,
            output_name,
        ) in output_mapping:
            values = np.asarray(
                store[
                    (
                        label,
                        k,
                    )
                ][
                    metric
                ],
                dtype=float,
            )

            values = values[
                np.isfinite(
                    values
                )
            ]

            row[
                f"random_{output_name}_mean"
            ] = (
                float(
                    np.mean(
                        values
                    )
                )
                if len(values)
                else np.nan
            )

            row[
                f"random_{output_name}_ci_lower"
            ] = (
                float(
                    np.quantile(
                        values,
                        0.025,
                    )
                )
                if len(values)
                else np.nan
            )

            row[
                f"random_{output_name}_ci_upper"
            ] = (
                float(
                    np.quantile(
                        values,
                        0.975,
                    )
                )
                if len(values)
                else np.nan
            )

        rows.append(
            row
        )

    return rows


def merge_baseline(
    results: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "cutoff_year",
        "positive_definition",
        "positive_label",
        "k_label",
        "k",
        "pool_size",
        "total_positives",
        "positive_prevalence",
    ]

    merged = results.merge(
        baseline,
        on=keys,
        how="left",
    )

    metrics = [
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "enrichment_factor_at_k",
        "ndcg_binary_at_k",
        "ndcg_continuous_at_k",
    ]

    for metric in metrics:
        random_mean = (
            f"random_{metric}_mean"
        )

        merged[
            f"{metric}_minus_random_mean"
        ] = (
            merged[
                metric
            ]
            - merged[
                random_mean
            ]
        )

        merged[
            f"{metric}_over_random_mean"
        ] = np.where(
            merged[
                random_mean
            ] != 0,
            (
                merged[
                    metric
                ]
                / merged[
                    random_mean
                ]
            ),
            np.nan,
        )

    return merged


def ranking_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "strategy",
        "strategy_label",
        "positive_definition",
        "positive_label",
        "k_label",
    ]

    metrics = [
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "enrichment_factor_at_k",
        "ndcg_binary_at_k",
        "ndcg_continuous_at_k",
        "spearman_all_future",
        "pairwise_ranking_accuracy",
        "true_top1_predicted_percentile_from_top",
        "mean_physical_support_distance_top_k",
        "mean_missingness_distance_top_k",
    ]

    grouped = results.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for (
        keys
    ), group in grouped:
        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "cutoff_count"
        ] = group[
            "cutoff_year"
        ].nunique()

        row[
            "mean_pool_size"
        ] = group[
            "pool_size"
        ].mean()

        for metric in metrics:
            row[
                f"{metric}_mean"
            ] = group[
                metric
            ].mean()

            row[
                f"{metric}_std"
            ] = (
                group[
                    metric
                ].std(
                    ddof=1
                )
                if len(group) > 1
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def top_conditions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    cutoffs = sorted(
        predictions[
            "cutoff_year"
        ].unique()
    )

    for cutoff in cutoffs:
        subset = predictions[
            predictions[
                "cutoff_year"
            ] == cutoff
        ]

        for (
            strategy,
            score_column,
        ) in strategy_columns().items():
            top = (
                subset
                .sort_values(
                    score_column,
                    ascending=False,
                )
                .head(
                    TOP_CONDITIONS_PER_STRATEGY
                )
                .copy()
            )

            top[
                "strategy"
            ] = strategy

            top[
                "strategy_label"
            ] = (
                STRATEGY_LABELS[
                    strategy
                ]
            )

            top[
                "predicted_rank"
            ] = np.arange(
                1,
                len(top) + 1,
            )

            top[
                "strategy_score"
            ] = top[
                score_column
            ]

            frames.append(
                top
            )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def support_summary(
    top: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = top.groupby(
        [
            "cutoff_year",
            "strategy",
        ],
        sort=False,
    )

    for (
        cutoff,
        strategy,
    ), group in grouped:
        (
            physical_support,
            missingness_support,
        ) = strategy_support_columns(
            strategy
        )

        rows.append(
            {
                "cutoff_year":
                    cutoff,
                "strategy":
                    strategy,
                "strategy_label":
                    STRATEGY_LABELS[
                        strategy
                    ],
                "support_reference":
                    (
                        "L1"
                        if strategy
                        == "L1_point"
                        else "L1+L2"
                    ),
                "top_n":
                    len(group),
                "mean_physical_support_distance":
                    group[
                        physical_support
                    ].mean(),
                "median_physical_support_distance":
                    group[
                        physical_support
                    ].median(),
                "maximum_physical_support_distance":
                    group[
                        physical_support
                    ].max(),
                "mean_missingness_distance":
                    group[
                        missingness_support
                    ].mean(),
                "median_missingness_distance":
                    group[
                        missingness_support
                    ].median(),
                "maximum_missingness_distance":
                    group[
                        missingness_support
                    ].max(),
                "fraction_future_top20":
                    group[
                        "positive_future_top_fraction"
                    ].mean(),
                "fraction_above_train80":
                    group[
                        "positive_above_train_threshold"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def apply_axes(
    axis: plt.Axes,
) -> None:
    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.tick_params(
        direction="out"
    )

    axis.grid(
        alpha=0.25,
        linewidth=0.5,
    )


def save_figure(
    figure: plt.Figure,
    base: str,
    dpi: int,
) -> None:
    figure.savefig(
        base + ".png",
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        base + ".pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main_figure_subset(
    results: pd.DataFrame,
) -> pd.DataFrame:
    return results[
        (
            results[
                "positive_definition"
            ]
            == "future_top_fraction"
        )
        & (
            results[
                "k_label"
            ]
            == "Top 10%"
        )
    ].copy()


def plot_metric_timeline(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    paths: Paths,
    dpi: int,
) -> None:
    source = main_figure_subset(
        results
    )

    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            filename
            + "_source_data.csv",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            9.5,
            5.8,
        )
    )

    for strategy in STRATEGY_ORDER:
        group = source[
            source[
                "strategy"
            ] == strategy
        ].sort_values(
            "cutoff_year"
        )

        if group.empty:
            continue

        axis.plot(
            group[
                "cutoff_year"
            ],
            group[
                metric
            ],
            marker="o",
            linewidth=1.5,
            label=(
                STRATEGY_LABELS[
                    strategy
                ]
            ),
        )

    if (
        metric
        == "pairwise_ranking_accuracy"
    ):
        axis.axhline(
            0.5,
            linestyle="--",
            linewidth=1.2,
            label=(
                "Random pairwise ordering"
            ),
        )

    axis.set_xlabel(
        "Historical cutoff year"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    apply_axes(
        axis
    )

    figure.tight_layout()

    save_figure(
        figure,
        os.path.join(
            paths.figures_dir,
            filename,
        ),
        dpi,
    )


def plot_metric_timeline_ci(
    results: pd.DataFrame,
    bootstrap: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    paths: Paths,
    dpi: int,
    reference: Optional[
        float
    ] = None,
) -> None:
    point = main_figure_subset(
        results
    )

    point = point[
        point[
            "strategy"
        ].isin(
            MAIN_STRATEGY_ORDER
        )
    ].copy()

    confidence = bootstrap[
        (
            bootstrap[
                "positive_definition"
            ]
            == "future_top_fraction"
        )
        & (
            bootstrap[
                "k_label"
            ]
            == "Top 10%"
        )
        & (
            bootstrap[
                "metric"
            ]
            == metric
        )
        & (
            bootstrap[
                "strategy"
            ].isin(
                MAIN_STRATEGY_ORDER
            )
        )
    ].copy()

    source = point.merge(
        confidence[
            [
                "cutoff_year",
                "strategy",
                "estimate",
                "ci_low",
                "ci_high",
                "valid_repeats",
            ]
        ],
        on=[
            "cutoff_year",
            "strategy",
        ],
        how="left",
        validate="one_to_one",
    )

    atomic_write_csv(
        source,
        os.path.join(
            paths.source_data_dir,
            filename
            + "_source_data.csv",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            9.5,
            5.8,
        )
    )

    for strategy in MAIN_STRATEGY_ORDER:
        group = source[
            source[
                "strategy"
            ] == strategy
        ].sort_values(
            "cutoff_year"
        )

        if group.empty:
            continue

        estimate = group[
            "estimate"
        ].to_numpy(
            dtype=float
        )

        low = group[
            "ci_low"
        ].to_numpy(
            dtype=float
        )

        high = group[
            "ci_high"
        ].to_numpy(
            dtype=float
        )

        error = np.vstack(
            [
                np.maximum(
                    0.0,
                    estimate
                    - low,
                ),
                np.maximum(
                    0.0,
                    high
                    - estimate,
                ),
            ]
        )

        axis.errorbar(
            group[
                "cutoff_year"
            ],
            estimate,
            yerr=error,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=(
                STRATEGY_LABELS[
                    strategy
                ]
            ),
        )

    random_mean = (
        f"random_{metric}_mean"
    )

    random_low = (
        f"random_{metric}_ci_lower"
    )

    random_high = (
        f"random_{metric}_ci_upper"
    )

    if random_mean in point.columns:
        random = (
            point[
                [
                    "cutoff_year",
                    random_mean,
                    random_low,
                    random_high,
                ]
            ]
            .drop_duplicates(
                "cutoff_year"
            )
            .sort_values(
                "cutoff_year"
            )
        )

        axis.plot(
            random[
                "cutoff_year"
            ],
            random[
                random_mean
            ],
            linestyle="--",
            linewidth=1.2,
            label="Random baseline",
        )

        axis.fill_between(
            random[
                "cutoff_year"
            ].to_numpy(
                dtype=float
            ),
            random[
                random_low
            ].to_numpy(
                dtype=float
            ),
            random[
                random_high
            ].to_numpy(
                dtype=float
            ),
            alpha=0.12,
        )

    if reference is not None:
        axis.axhline(
            reference,
            linestyle=":",
            linewidth=1.0,
        )

    axis.set_xlabel(
        "Historical cutoff year"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    apply_axes(
        axis
    )

    figure.tight_layout()

    save_figure(
        figure,
        os.path.join(
            paths.figures_dir,
            filename,
        ),
        dpi,
    )


def plot_support(
    support: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    paths: Paths,
    dpi: int,
) -> None:
    atomic_write_csv(
        support,
        os.path.join(
            paths.source_data_dir,
            filename
            + "_source_data.csv",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            9.5,
            5.8,
        )
    )

    for strategy in STRATEGY_ORDER:
        group = support[
            support[
                "strategy"
            ] == strategy
        ].sort_values(
            "cutoff_year"
        )

        if group.empty:
            continue

        axis.plot(
            group[
                "cutoff_year"
            ],
            group[
                metric
            ],
            marker="o",
            linewidth=1.5,
            label=(
                STRATEGY_LABELS[
                    strategy
                ]
            ),
        )

    axis.set_xlabel(
        "Historical cutoff year"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    apply_axes(
        axis
    )

    figure.tight_layout()

    save_figure(
        figure,
        os.path.join(
            paths.figures_dir,
            filename,
        ),
        dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Historical replay and retrospective "
            "CNT ranking validation"
        )
    )

    parser.add_argument(
        "--project_dir",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_cutoffs",
        type=int,
        default=DEFAULT_MAX_CUTOFFS,
    )

    parser.add_argument(
        "--min_train_samples",
        type=int,
        default=DEFAULT_MIN_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--min_future_samples",
        type=int,
        default=DEFAULT_MIN_FUTURE_SAMPLES,
    )

    parser.add_argument(
        "--min_train_papers",
        type=int,
        default=DEFAULT_MIN_TRAIN_PAPERS,
    )

    parser.add_argument(
        "--min_future_papers",
        type=int,
        default=DEFAULT_MIN_FUTURE_PAPERS,
    )

    parser.add_argument(
        "--random_repeats",
        type=int,
        default=DEFAULT_RANDOM_REPEATS,
    )

    parser.add_argument(
        "--bootstrap_repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )

    parser.add_argument(
        "--positive_fraction",
        type=float,
        default=DEFAULT_POSITIVE_FRACTION,
    )

    parser.add_argument(
        "--train_threshold_quantile",
        type=float,
        default=DEFAULT_TRAIN_THRESHOLD_QUANTILE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
    )

    parser.add_argument(
        "--save_models",
        action="store_true",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.positive_fraction
        < 1.0
    ):
        raise ValueError(
            "--positive_fraction "
            "必须位于0和1之间。"
        )

    if not (
        0.0
        < args.train_threshold_quantile
        < 1.0
    ):
        raise ValueError(
            "--train_threshold_quantile "
            "必须位于0和1之间。"
        )

    if (
        args.random_repeats < 1
        or args.bootstrap_repeats < 1
    ):
        raise ValueError(
            "--random_repeats 和 "
            "--bootstrap_repeats 必须大于0。"
        )

    script_dir = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )

    project_dir = (
        os.path.abspath(
            args.project_dir
        )
        if args.project_dir
        else os.path.dirname(
            os.path.dirname(
                script_dir
            )
        )
    )

    paths = build_paths(
        project_dir,
        args.output_dir,
    )

    ensure_inputs(
        paths
    )

    if (
        args.fresh
        and os.path.isdir(
            paths.output_dir
        )
    ):
        shutil.rmtree(
            paths.output_dir
        )

    create_dirs(
        paths,
        args.save_models,
    )

    (
        features,
        physical_features,
        missingness_features,
    ) = load_schema(
        paths.schema
    )

    dataset = load_dataset(
        paths.dataset,
        features,
    )

    base = prepare_composite_base(
        dataset
    )

    base = base[
        base[
            "paper_year_int"
        ].notna()
    ].copy()

    manifest = cutoff_manifest(
        base,
        args,
    )

    cutoffs = choose_cutoffs(
        manifest,
        args.cutoffs,
        args.max_cutoffs,
    )

    manifest[
        "selected"
    ] = manifest[
        "cutoff_year"
    ].isin(
        cutoffs
    )

    atomic_write_csv(
        manifest,
        paths.cutoff_manifest,
    )

    configuration = {
        "dataset_sha256":
            sha256_file(
                paths.dataset
            ),
        "schema_sha256":
            sha256_file(
                paths.schema
            ),
        "cutoffs":
            cutoffs,
        "minimum_component_count":
            MIN_COMPONENT_COUNT,
        "positive_fraction":
            args.positive_fraction,
        "train_threshold_quantile":
            args.train_threshold_quantile,
        "seed":
            args.seed,
    }

    configuration_hash = (
        hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
            ).encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    if os.path.isfile(
        paths.checkpoint_config
    ):
        with open(
            paths.checkpoint_config,
            "r",
            encoding="utf-8",
        ) as f:
            old_configuration = json.load(
                f
            )

        if (
            old_configuration.get(
                "configuration_hash"
            )
            != configuration_hash
        ):
            raise RuntimeError(
                "现有断点配置不同，"
                "请使用 --fresh。"
            )

    else:
        atomic_write_json(
            {
                "configuration_hash":
                    configuration_hash,
                "configuration":
                    configuration,
            },
            paths.checkpoint_config,
        )

    existing_predictions = (
        pd.read_csv(
            paths.predictions,
            low_memory=False,
        )
        if os.path.isfile(
            paths.predictions
        )
        else pd.DataFrame()
    )

    existing_models = (
        pd.read_csv(
            paths.model_manifest,
            low_memory=False,
        )
        if os.path.isfile(
            paths.model_manifest
        )
        else pd.DataFrame()
    )

    completed = (
        set(
            existing_predictions[
                "cutoff_year"
            ]
            .astype(int)
            .unique()
        )
        if not existing_predictions.empty
        else set()
    )

    prediction_frames = (
        [
            existing_predictions
        ]
        if not existing_predictions.empty
        else []
    )

    model_frames = (
        [
            existing_models
        ]
        if not existing_models.empty
        else []
    )

    print(
        "=" * 104
    )

    print(
        "[RETROSPECTIVE HISTORICAL REPLAY START]"
    )

    print(
        "Project directory      : "
        f"{paths.project_dir}"
    )

    print(
        "Dated L1/L2 rows       : "
        f"{len(base)}"
    )

    print(
        "Composite threshold    : "
        f"{MIN_COMPONENT_COUNT}"
    )

    print(
        "Features               : "
        f"{len(physical_features)} physical + "
        f"{len(missingness_features)} missing"
    )

    print(
        "Selected cutoffs       : "
        f"{cutoffs}"
    )

    print(
        "Completed cutoffs      : "
        f"{sorted(completed)}"
    )

    print(
        "Random repeats         : "
        f"{args.random_repeats}"
    )

    print(
        "Paper bootstrap        : "
        f"{args.bootstrap_repeats}"
    )

    print(
        "Output                 : "
        f"{paths.output_dir}"
    )

    print(
        "=" * 104
    )

    for (
        index,
        cutoff,
    ) in enumerate(
        cutoffs,
        start=1,
    ):
        if cutoff in completed:
            print(
                f"[CUTOFF {index}/"
                f"{len(cutoffs)}] "
                f"{cutoff} already completed, skip"
            )

            continue

        print(
            f"[CUTOFF {index}/"
            f"{len(cutoffs)}] "
            f"training models for {cutoff}"
        )

        (
            prediction,
            models,
        ) = replay_cutoff(
            base,
            features,
            physical_features,
            missingness_features,
            cutoff,
            args.positive_fraction,
            args.train_threshold_quantile,
            paths,
            args.save_models,
        )

        prediction_frames.append(
            prediction
        )

        model_frames.append(
            models
        )

        all_predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
            sort=False,
        )

        all_models = pd.concat(
            model_frames,
            ignore_index=True,
            sort=False,
        )

        atomic_write_csv(
            all_predictions,
            paths.predictions,
        )

        atomic_write_csv(
            all_models,
            paths.model_manifest,
        )

        print(
            "    future samples="
            f"{len(prediction)}, "
            "future papers="
            f"{prediction['paper_id'].nunique()}"
        )

    predictions = pd.read_csv(
        paths.predictions,
        low_memory=False,
    )

    ranking_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    random_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    bootstrap_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    unique_cutoffs = sorted(
        predictions[
            "cutoff_year"
        ]
        .astype(int)
        .unique()
    )

    for (
        cutoff_index,
        cutoff,
    ) in enumerate(
        unique_cutoffs,
        start=1,
    ):
        frame = predictions[
            predictions[
                "cutoff_year"
            ].astype(int)
            == cutoff
        ].copy()

        print(
            f"[RANKING {cutoff_index}/"
            f"{len(unique_cutoffs)}] "
            f"cutoff={cutoff}"
        )

        for (
            definition_index,
            definition,
        ) in enumerate(
            POSITIVE_DEFINITIONS
        ):
            random_rows.extend(
                random_baseline(
                    frame,
                    cutoff,
                    definition,
                    args.random_repeats,
                    (
                        args.seed
                        + cutoff * 100
                        + definition_index
                    ),
                )
            )

            for (
                strategy_index,
                strategy,
            ) in enumerate(
                STRATEGY_ORDER
            ):
                ranking_rows.extend(
                    evaluate_strategy(
                        frame,
                        cutoff,
                        strategy,
                        definition,
                        (
                            args.seed
                            + cutoff * 1000
                            + definition_index
                            * 100
                            + strategy_index
                        ),
                    )
                )

            bootstrap_rows.extend(
                bootstrap_cutoff_by_paper(
                    frame,
                    cutoff,
                    definition,
                    args.bootstrap_repeats,
                    args.positive_fraction,
                    (
                        args.seed
                        + cutoff * 10000
                        + definition_index
                    ),
                )
            )

    raw_results = pd.DataFrame(
        ranking_rows
    )

    random_frame = pd.DataFrame(
        random_rows
    )

    bootstrap_frame = pd.DataFrame(
        bootstrap_rows
    )

    results = merge_baseline(
        raw_results,
        random_frame,
    )

    summary = ranking_summary(
        results
    )

    top = top_conditions(
        predictions
    )

    support = support_summary(
        top
    )

    atomic_write_csv(
        results,
        paths.ranking_results,
    )

    atomic_write_csv(
        summary,
        paths.ranking_summary,
    )

    atomic_write_csv(
        random_frame,
        paths.random_baseline,
    )

    atomic_write_csv(
        bootstrap_frame,
        paths.bootstrap_ci,
    )

    atomic_write_csv(
        top,
        paths.top_conditions,
    )

    atomic_write_csv(
        support,
        paths.support_summary,
    )

    main_table = results[
        (
            results[
                "positive_definition"
            ]
            == "future_top_fraction"
        )
        & (
            results[
                "k_label"
            ]
            == "Top 10%"
        )
        & (
            results[
                "strategy"
            ].isin(
                MAIN_STRATEGY_ORDER
            )
        )
    ].copy()

    main_bootstrap = bootstrap_frame[
        (
            bootstrap_frame[
                "positive_definition"
            ]
            == "future_top_fraction"
        )
        & (
            bootstrap_frame[
                "k_label"
            ]
            == "Top 10%"
        )
        & (
            bootstrap_frame[
                "strategy"
            ].isin(
                MAIN_STRATEGY_ORDER
            )
        )
    ].copy()

    atomic_write_csv(
        main_table,
        os.path.join(
            paths.tables_dir,
            "table_main_retrospective_ranking.csv",
        ),
    )

    atomic_write_csv(
        main_bootstrap,
        os.path.join(
            paths.tables_dir,
            "table_main_retrospective_bootstrap_ci.csv",
        ),
    )

    atomic_write_csv(
        summary,
        os.path.join(
            paths.tables_dir,
            "table_si_retrospective_ranking_summary.csv",
        ),
    )

    atomic_write_csv(
        random_frame,
        os.path.join(
            paths.tables_dir,
            "table_si_random_ranking_baseline.csv",
        ),
    )

    atomic_write_csv(
        top,
        os.path.join(
            paths.tables_dir,
            "table_si_top_historical_conditions.csv",
        ),
    )

    plot_metric_timeline_ci(
        results,
        bootstrap_frame,
        "ndcg_continuous_at_k",
        "Continuous NDCG@Top 10%",
        "Historical ranking of later CNT conditions",
        "fig_historical_ndcg_ci",
        paths,
        args.dpi,
    )

    plot_metric_timeline_ci(
        results,
        bootstrap_frame,
        "recall_at_k",
        "Recall@Top 10%",
        "Recovery of later high-quality CNT conditions",
        "fig_historical_recall_ci",
        paths,
        args.dpi,
    )

    plot_metric_timeline_ci(
        results,
        bootstrap_frame,
        "enrichment_factor_at_k",
        "Enrichment factor@Top 10%",
        "Enrichment of later high-quality CNT conditions",
        "fig_historical_enrichment_ci",
        paths,
        args.dpi,
        reference=1.0,
    )

    plot_metric_timeline(
        results,
        "pairwise_ranking_accuracy",
        "Pairwise ranking accuracy",
        "Pairwise ordering of later CNT conditions",
        "figS_pairwise_ranking_accuracy",
        paths,
        args.dpi,
    )

    plot_support(
        support,
        "mean_physical_support_distance",
        "Mean physical nearest-training distance of Top 20",
        "Physical support of historical recommendations",
        "figS_historical_support_physical",
        paths,
        args.dpi,
    )

    plot_support(
        support,
        "mean_missingness_distance",
        "Mean missingness-pattern distance of Top 20",
        "Missingness support of historical recommendations",
        "figS_historical_support_missingness",
        paths,
        args.dpi,
    )

    audit_rows = []

    for cutoff in cutoffs:
        group = predictions[
            predictions[
                "cutoff_year"
            ].astype(int)
            == cutoff
        ]

        audit_rows.append(
            {
                "cutoff_year":
                    cutoff,

                "prediction_rows":
                    len(group),

                "unique_samples":
                    group[
                        "sample_id"
                    ].nunique(),

                "unique_papers":
                    group[
                        "paper_id"
                    ].nunique(),

                "sample_id_unique":
                    (
                        len(group)
                        == group[
                            "sample_id"
                        ].nunique()
                    ),

                "all_years_after_cutoff":
                    bool(
                        (
                            group[
                                "paper_year_int"
                            ]
                            > cutoff
                        ).all()
                    ),

                "all_predictions_finite":
                    bool(
                        np.isfinite(
                            group[
                                [
                                    "pred_L1_point",
                                    "pred_q10_raw",
                                    "pred_q50_raw",
                                    "pred_q90_raw",
                                ]
                            ].to_numpy(
                                dtype=float
                            )
                        ).all()
                    ),

                "minimum_component_count":
                    int(
                        group[
                            "component_count_replay"
                        ].min()
                    ),

                "future_top_positive_count":
                    int(
                        group[
                            "positive_future_top_fraction"
                        ].sum()
                    ),

                "above_train_positive_count":
                    int(
                        group[
                            "positive_above_train_threshold"
                        ].sum()
                    ),
            }
        )

    atomic_write_csv(
        pd.DataFrame(
            audit_rows
        ),
        os.path.join(
            paths.manifests_dir,
            "retrospective_replay_audit.csv",
        ),
    )

    input_manifest = pd.DataFrame(
        [
            {
                "path":
                    paths.dataset,
                "size_bytes":
                    os.path.getsize(
                        paths.dataset
                    ),
                "sha256":
                    sha256_file(
                        paths.dataset
                    ),
            },
            {
                "path":
                    paths.schema,
                "size_bytes":
                    os.path.getsize(
                        paths.schema
                    ),
                "sha256":
                    sha256_file(
                        paths.schema
                    ),
            },
        ]
    )

    atomic_write_csv(
        input_manifest,
        os.path.join(
            paths.manifests_dir,
            "input_file_manifest.csv",
        ),
    )

    run_summary = {
        "analysis":
            (
                "CNT 3.0 retrospective historical "
                "replay and ranking validation"
            ),

        "selected_cutoffs":
            cutoffs,

        "dated_rows":
            len(base),

        "minimum_component_count":
            MIN_COMPONENT_COUNT,

        "feature_count":
            len(features),

        "random_repeats":
            args.random_repeats,

        "paper_bootstrap_repeats":
            args.bootstrap_repeats,

        "ranking_strategies": {
            name:
                STRATEGY_LABELS[
                    name
                ]
            for name
            in STRATEGY_ORDER
        },

        "scientific_boundary":
            (
                "Retrospective recovery of later "
                "published CNT conditions; not a "
                "prospective wet-lab discovery loop."
            ),

        "outputs": {
            "predictions":
                paths.predictions,

            "ranking_results":
                paths.ranking_results,

            "ranking_summary":
                paths.ranking_summary,

            "random_baseline":
                paths.random_baseline,

            "bootstrap_ci":
                paths.bootstrap_ci,

            "cutoff_manifest":
                paths.cutoff_manifest,

            "model_manifest":
                paths.model_manifest,

            "top_conditions":
                paths.top_conditions,

            "support_summary":
                paths.support_summary,

            "figures_dir":
                paths.figures_dir,

            "source_data_dir":
                paths.source_data_dir,

            "tables_dir":
                paths.tables_dir,
        },
    }

    atomic_write_json(
        run_summary,
        paths.run_summary,
    )

    print(
        "\n"
        + "=" * 104
    )

    print(
        "[RETROSPECTIVE HISTORICAL REPLAY FINISHED]"
    )

    print(
        "Predictions        : "
        f"{paths.predictions}"
    )

    print(
        "Ranking results    : "
        f"{paths.ranking_results}"
    )

    print(
        "Ranking summary    : "
        f"{paths.ranking_summary}"
    )

    print(
        "Random baseline    : "
        f"{paths.random_baseline}"
    )

    print(
        "Bootstrap CI       : "
        f"{paths.bootstrap_ci}"
    )

    print(
        "Cutoff manifest    : "
        f"{paths.cutoff_manifest}"
    )

    print(
        "Model manifest     : "
        f"{paths.model_manifest}"
    )

    print(
        "Top conditions     : "
        f"{paths.top_conditions}"
    )

    print(
        "Support summary    : "
        f"{paths.support_summary}"
    )

    print(
        "Figures            : "
        f"{paths.figures_dir}"
    )

    print(
        "Source data        : "
        f"{paths.source_data_dir}"
    )

    print(
        "Tables             : "
        f"{paths.tables_dir}"
    )

    print(
        "Run summary        : "
        f"{paths.run_summary}"
    )

    print(
        "=" * 104
    )


if __name__ == "__main__":
    main()