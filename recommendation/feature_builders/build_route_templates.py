# build_route_templates.py
# =============================================================================
# CNT 3.0 — build z_flow + z_proc route templates
#
# Input:
#   data/11_flattened/ml_dataset_3p0_full60D.csv
#
# Outputs:
#   outputs/recommendation/route_templates/route_templates_zpf.csv
#   outputs/recommendation/route_templates/route_templates_zpf_meta.csv
#   outputs/recommendation/route_templates/route_templates_zpf_winsor_limits.csv
#
# Strategy:
#   - build templates separately for growth modes A–E;
#   - use L2 first, then fill the remaining templates with stable L1 clusters;
#   - cluster standardized winsorized F1–F18/P1–P12 plus missing masks;
#   - save centroids in the original winsorized physical scale;
#   - record cluster size, missingness, and the nearest real sample.
#
# This script only uses the current CNT project.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "11_flattened"
    / "ml_dataset_3p0_full60D.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "recommendation"
    / "route_templates"
)

OUT_TEMPLATES = (
    OUT_DIR
    / "route_templates_zpf.csv"
)

OUT_META = (
    OUT_DIR
    / "route_templates_zpf_meta.csv"
)

OUT_WINSOR = (
    OUT_DIR
    / "route_templates_zpf_winsor_limits.csv"
)

GROWTH_MODES = [
    "A",
    "B",
    "C",
    "D",
    "E",
]

K_MODE = 20
MIN_CLUSTER_SIZE = 5
INCLUDE_MISS_IN_CLUSTER = True

SEED = 42
BATCH_SIZE = 1024
MAX_ITER = 300
N_INIT = 10

WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99
STANDARDIZE_PF = True

ID_COLUMN_CANDIDATES = [
    "sample_id",
    "sample_uid",
    "record_id",
    "row_id",
    "id",
    "paper_id",
]


# =============================================================================
# HELPERS
# =============================================================================

def pf_feature_cols() -> tuple[
    list[str],
    list[str],
]:
    return (
        [
            f"F{i}"
            for i in range(1, 19)
        ],
        [
            f"P{i}"
            for i in range(1, 13)
        ],
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    lookup = {
        str(column).strip().lower():
            column
        for column in df.columns
    }

    for candidate in candidates:

        found = lookup.get(
            candidate.strip().lower()
        )

        if found is not None:
            return found

    return None


def normalize_growth_mode(
    value: Any,
) -> str | None:

    if (
        value is None
        or pd.isna(value)
    ):
        return None

    token = (
        str(value)
        .strip()
        .upper()
    )

    if (
        not token
        or token == "NAN"
    ):
        return None

    numeric_map = {
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "5": "E",
    }

    if token in numeric_map:
        return numeric_map[token]

    if token in GROWTH_MODES:
        return token

    for mode in GROWTH_MODES:

        if token.startswith(
            (
                f"{mode} ",
                f"{mode}_",
                f"{mode}-",
                f"{mode}:",
                f"{mode}(",
                f"{mode}（",
            )
        ):
            return mode

    return token


def add_source_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:

    work = df.copy()

    id_col = find_column(
        work,
        ID_COLUMN_CANDIDATES,
    )

    work[
        "_source_row_index"
    ] = (
        work.index
        .astype(str)
    )

    if id_col is None:

        work[
            "_source_sample_id"
        ] = work[
            "_source_row_index"
        ]

        work[
            "_source_id_column"
        ] = "dataframe_index"

    else:

        ids = (
            work[id_col]
            .astype(str)
        )

        invalid = (
            work[id_col].isna()
            | ids.str.strip().eq("")
            | ids.str.upper().eq("NAN")
        )

        ids.loc[
            invalid
        ] = work.loc[
            invalid,
            "_source_row_index",
        ]

        work[
            "_source_sample_id"
        ] = ids

        work[
            "_source_id_column"
        ] = id_col

    return work


# =============================================================================
# PREPROCESSING
# =============================================================================

def ensure_pf_and_miss(
    df: pd.DataFrame,
    pf_cols: list[str],
) -> pd.DataFrame:
    """
    Missing physical values are represented as:
        feature = 0.0
        miss_feature = 1

    A real observed zero remains:
        feature = 0.0
        miss_feature = 0
    """

    work = df.copy()

    for feature in pf_cols:

        if feature not in work.columns:
            work[feature] = np.nan

        work[feature] = (
            pd.to_numeric(
                work[feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        miss_col = (
            f"miss_{feature}"
        )

        if miss_col in work.columns:

            old_miss = (
                pd.to_numeric(
                    work[miss_col],
                    errors="coerce",
                )
                .fillna(0)
                .gt(0)
            )

        else:

            old_miss = pd.Series(
                False,
                index=work.index,
            )

        work[
            miss_col
        ] = (
            old_miss
            | work[feature].isna()
        ).astype(int)

        work.loc[
            work[miss_col].eq(1),
            feature,
        ] = np.nan

    return work


def winsorize_pf(
    df: pd.DataFrame,
    pf_cols: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Winsor limits are calculated only from observed values.
    Missing entries do not affect q01/q99.
    """

    work = df.copy()

    rows: list[
        dict[str, Any]
    ] = []

    for feature in pf_cols:

        miss_col = (
            f"miss_{feature}"
        )

        observed = work.loc[
            work[miss_col].eq(0),
            feature,
        ].dropna()

        if observed.empty:

            lower = np.nan
            upper = np.nan
            clipped_n = 0

        else:

            lower = float(
                observed.quantile(
                    WINSOR_Q_LOW
                )
            )

            upper = float(
                observed.quantile(
                    WINSOR_Q_HIGH
                )
            )

            if lower > upper:
                lower, upper = (
                    upper,
                    lower,
                )

            original = (
                work[feature]
                .copy()
            )

            mask = (
                work[
                    miss_col
                ].eq(0)
            )

            work.loc[
                mask,
                feature,
            ] = (
                work.loc[
                    mask,
                    feature,
                ]
                .clip(
                    lower=lower,
                    upper=upper,
                )
            )

            clipped_n = int(
                (
                    original.ne(
                        work[feature]
                    )
                    & mask
                ).sum()
            )

        work.loc[
            work[
                miss_col
            ].eq(1),
            feature,
        ] = 0.0

        work[
            feature
        ] = (
            pd.to_numeric(
                work[feature],
                errors="coerce",
            )
            .fillna(0.0)
        )

        rows.append(
            {
                "feature":
                    feature,

                "observed_n":
                    int(
                        len(
                            observed
                        )
                    ),

                "missing_n":
                    int(
                        work[
                            miss_col
                        ].sum()
                    ),

                "q_low":
                    WINSOR_Q_LOW,

                "q_high":
                    WINSOR_Q_HIGH,

                "lower_limit":
                    lower,

                "upper_limit":
                    upper,

                "clipped_n":
                    clipped_n,
            }
        )

    return (
        work,
        pd.DataFrame(
            rows
        ),
    )


def print_diagnostics(
    df: pd.DataFrame,
    pf_cols: list[str],
) -> None:

    rows = []

    for feature in pf_cols:

        miss_col = (
            f"miss_{feature}"
        )

        observed = df.loc[
            df[miss_col].eq(0),
            feature,
        ]

        rows.append(
            {
                "feature":
                    feature,

                "observed_n":
                    int(
                        len(
                            observed
                        )
                    ),

                "observed_rate":
                    float(
                        len(
                            observed
                        )
                        / max(
                            len(df),
                            1,
                        )
                    ),

                "median":
                    (
                        float(
                            observed.median()
                        )
                        if len(observed)
                        else np.nan
                    ),

                "max":
                    (
                        float(
                            observed.max()
                        )
                        if len(observed)
                        else np.nan
                    ),
            }
        )

    diagnostics = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "observed_rate",
                "feature",
            ]
        )
    )

    print(
        "\n"
        "[DIAGNOSTIC: "
        "10 MOST MISSING F/P FEATURES]"
    )

    print(
        diagnostics
        .head(10)
        .to_string(
            index=False
        )
    )


# =============================================================================
# CLUSTERING
# =============================================================================

def build_cluster_matrix(
    df: pd.DataFrame,
    pf_cols: list[str],
    miss_cols: list[str],
) -> np.ndarray:

    X_pf = (
        df[
            pf_cols
        ]
        .to_numpy(
            dtype=float
        )
    )

    if STANDARDIZE_PF:

        X_pf = (
            StandardScaler()
            .fit_transform(
                X_pf
            )
        )

    if INCLUDE_MISS_IN_CLUSTER:

        X_miss = (
            df[
                miss_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

        return np.hstack(
            [
                X_pf,
                X_miss,
            ]
        )

    return X_pf


def physical_centroid(
    cluster_df: pd.DataFrame,
    pf_cols: list[str],
    miss_cols: list[str],
) -> tuple[
    dict[str, float],
    dict[str, float],
]:
    """
    The stored centroid uses only observed values for each feature.
    Missing zeros used during clustering are not averaged into the centroid.
    """

    values: dict[
        str,
        float,
    ] = {}

    miss_ratios: dict[
        str,
        float,
    ] = {}

    for (
        feature,
        miss_col,
    ) in zip(
        pf_cols,
        miss_cols,
    ):

        miss_ratio = float(
            cluster_df[
                miss_col
            ].mean()
        )

        observed = (
            cluster_df.loc[
                cluster_df[
                    miss_col
                ].eq(0),
                feature,
            ]
        )

        values[
            feature
        ] = (
            float(
                observed.mean()
            )
            if len(observed)
            else 0.0
        )

        values[
            miss_col
        ] = int(
            miss_ratio >= 0.5
        )

        miss_ratios[
            f"{miss_col}_ratio"
        ] = miss_ratio

    return (
        values,
        miss_ratios,
    )


def nearest_real_member(
    member_positions: np.ndarray,
    X: np.ndarray,
    center: np.ndarray,
    data: pd.DataFrame,
) -> tuple[
    str,
    str,
    float,
]:

    distances = (
        np.linalg.norm(
            X[
                member_positions
            ]
            - center,
            axis=1,
        )
    )

    nearest_position = int(
        member_positions[
            int(
                np.argmin(
                    distances
                )
            )
        ]
    )

    row = data.iloc[
        nearest_position
    ]

    return (
        str(
            row[
                "_source_sample_id"
            ]
        ),
        str(
            row[
                "_source_row_index"
            ]
        ),
        float(
            distances.min()
        ),
    )


def cluster_one_set(
    df_in: pd.DataFrame,
    mode: str,
    level: str,
    k_target: int,
    pf_cols: list[str],
    miss_cols: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if df_in.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    data = (
        df_in
        .reset_index(
            drop=True
        )
        .copy()
    )

    n_samples = len(
        data
    )

    max_k = max(
        1,
        n_samples
        // max(
            MIN_CLUSTER_SIZE,
            1,
        ),
    )

    n_clusters = min(
        k_target,
        max_k,
    )

    X = build_cluster_matrix(
        data,
        pf_cols,
        miss_cols,
    )

    if (
        n_samples
        < max(
            2 * MIN_CLUSTER_SIZE,
            10,
        )
        or n_clusters <= 1
    ):

        labels = np.zeros(
            n_samples,
            dtype=int,
        )

        centers = np.asarray(
            [
                X.mean(
                    axis=0
                )
            ]
        )

    else:

        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            batch_size=min(
                BATCH_SIZE,
                max(
                    n_samples,
                    n_clusters,
                ),
            ),
            n_init=N_INIT,
            max_iter=MAX_ITER,
            reassignment_ratio=0.01,
        )

        labels = (
            model.fit_predict(
                X
            )
        )

        centers = (
            model.cluster_centers_
        )

    data[
        "_cluster_id"
    ] = labels

    template_rows = []
    meta_rows = []

    unique_clusters = sorted(
        data[
            "_cluster_id"
        ].unique()
    )

    for cluster_id in unique_clusters:

        member_positions = (
            np.flatnonzero(
                labels
                == cluster_id
            )
        )

        cluster_size = int(
            len(
                member_positions
            )
        )

        if (
            len(
                unique_clusters
            ) > 1
            and cluster_size
            < MIN_CLUSTER_SIZE
        ):
            continue

        cluster_df = data.iloc[
            member_positions
        ]

        (
            centroid,
            miss_ratios,
        ) = physical_centroid(
            cluster_df,
            pf_cols,
            miss_cols,
        )

        (
            nearest_id,
            nearest_index,
            nearest_distance,
        ) = nearest_real_member(
            member_positions,
            X,
            centers[
                int(
                    cluster_id
                )
            ],
            data,
        )

        missing_count = int(
            sum(
                centroid[
                    miss_col
                ]
                for miss_col
                in miss_cols
            )
        )

        mean_missing_ratio = float(
            np.mean(
                list(
                    miss_ratios.values()
                )
            )
        )

        common = {
            "growth_mode":
                mode,

            "source_level":
                level,

            "cluster_id_original":
                int(
                    cluster_id
                ),

            "cluster_size":
                cluster_size,

            "template_missing_feature_count":
                missing_count,

            "template_missing_fraction":
                (
                    missing_count
                    / len(
                        miss_cols
                    )
                ),

            "cluster_mean_missing_ratio":
                mean_missing_ratio,

            "nearest_real_sample_id":
                nearest_id,

            "nearest_real_row_index":
                nearest_index,

            "nearest_real_distance":
                nearest_distance,
        }

        template_rows.append(
            {
                **common,
                **centroid,
                **miss_ratios,
            }
        )

        meta_rows.append(
            {
                **common,
                **miss_ratios,
            }
        )

    if not template_rows:

        raise RuntimeError(
            "No cluster survived for "
            f"growth_mode={mode}, "
            f"data_level={level}."
        )

    templates = pd.DataFrame(
        template_rows
    )

    meta = pd.DataFrame(
        meta_rows
    )

    sort_cols = [
        "cluster_size",
        "cluster_mean_missing_ratio",
        "nearest_real_distance",
        "cluster_id_original",
    ]

    ascending = [
        False,
        True,
        True,
        True,
    ]

    templates = (
        templates
        .sort_values(
            sort_cols,
            ascending=ascending,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    meta = (
        meta
        .sort_values(
            sort_cols,
            ascending=ascending,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    for index in range(
        len(
            templates
        )
    ):

        template_id = (
            f"{mode}_"
            f"{level}_"
            f"{index + 1:03d}"
        )

        templates.loc[
            index,
            "template_id",
        ] = template_id

        meta.loc[
            index,
            "template_id",
        ] = template_id

        templates.loc[
            index,
            "cluster_rank_within_level",
        ] = (
            index + 1
        )

        meta.loc[
            index,
            "cluster_rank_within_level",
        ] = (
            index + 1
        )

    template_front = [
        "growth_mode",
        "source_level",
        "template_id",
        "cluster_rank_within_level",
        "cluster_size",
    ]

    ratio_cols = [
        f"{miss_col}_ratio"
        for miss_col
        in miss_cols
    ]

    templates = templates[
        template_front
        + pf_cols
        + miss_cols
        + ratio_cols
        + [
            "template_missing_feature_count",
            "template_missing_fraction",
            "cluster_mean_missing_ratio",
            "nearest_real_sample_id",
            "nearest_real_row_index",
            "nearest_real_distance",
            "cluster_id_original",
        ]
    ]

    meta = meta[
        template_front
        + [
            "template_missing_feature_count",
            "template_missing_fraction",
            "cluster_mean_missing_ratio",
            "nearest_real_sample_id",
            "nearest_real_row_index",
            "nearest_real_distance",
            "cluster_id_original",
        ]
        + ratio_cols
    ]

    return (
        templates,
        meta,
    )


def select_top_clusters(
    templates: pd.DataFrame,
    meta: pd.DataFrame,
    k: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if (
        k <= 0
        or templates.empty
        or meta.empty
    ):

        return (
            templates
            .head(0)
            .copy(),

            meta
            .head(0)
            .copy(),
        )

    selected_meta = (
        meta
        .sort_values(
            [
                "cluster_size",
                "cluster_mean_missing_ratio",
                "nearest_real_distance",
                "template_id",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            kind="stable",
        )
        .head(
            k
        )
        .reset_index(
            drop=True
        )
    )

    order = {
        template_id:
            index

        for index, template_id
        in enumerate(
            selected_meta[
                "template_id"
            ]
        )
    }

    selected_templates = (
        templates[
            templates[
                "template_id"
            ].isin(
                order
            )
        ]
        .copy()
    )

    selected_templates[
        "_order"
    ] = (
        selected_templates[
            "template_id"
        ]
        .map(
            order
        )
    )

    selected_templates = (
        selected_templates
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        selected_templates,
        selected_meta,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 112
    )

    print(
        "[ROUTE TEMPLATE BUILD START]"
    )

    print(
        f"Project directory    : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Input dataset        : "
        f"{CSV_PATH}"
    )

    print(
        f"Output directory     : "
        f"{OUT_DIR}"
    )

    print(
        f"Target per mode      : "
        f"{K_MODE}"
    )

    print(
        f"Minimum cluster size : "
        f"{MIN_CLUSTER_SIZE}"
    )

    print(
        "=" * 112
    )

    if not CSV_PATH.is_file():

        raise FileNotFoundError(
            "Dataset not found:\n"
            f"  {CSV_PATH}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    level_col = find_column(
        df,
        [
            "data_level"
        ],
    )

    mode_col = find_column(
        df,
        [
            "growth_mode"
        ],
    )

    if level_col is None:

        raise RuntimeError(
            "Dataset is missing "
            "data_level."
        )

    if mode_col is None:

        raise RuntimeError(
            "Dataset is missing "
            "growth_mode."
        )

    df[
        "_data_level"
    ] = (
        df[
            level_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df[
        "_growth_mode"
    ] = (
        df[
            mode_col
        ]
        .map(
            normalize_growth_mode
        )
    )

    df_use = (
        df[
            df[
                "_data_level"
            ].isin(
                [
                    "L1",
                    "L2",
                ]
            )
        ]
        .copy()
    )

    if df_use.empty:

        raise RuntimeError(
            "No L1/L2 rows were found."
        )

    (
        f_cols,
        p_cols,
    ) = pf_feature_cols()

    pf_cols = (
        f_cols
        + p_cols
    )

    miss_cols = [
        f"miss_{feature}"

        for feature
        in pf_cols
    ]

    df_use = add_source_ids(
        df_use
    )

    df_use = ensure_pf_and_miss(
        df_use,
        pf_cols,
    )

    print_diagnostics(
        df_use,
        pf_cols,
    )

    (
        df_use,
        winsor_limits,
    ) = winsorize_pf(
        df_use,
        pf_cols,
    )

    all_templates = []
    all_meta = []

    print(
        "\n"
        "[BUILD BY GROWTH MODE: "
        "L2 FIRST, THEN L1 FILL]"
    )

    for mode in GROWTH_MODES:

        mode_df = df_use[
            df_use[
                "_growth_mode"
            ].eq(
                mode
            )
        ]

        l2_df = mode_df[
            mode_df[
                "_data_level"
            ].eq(
                "L2"
            )
        ]

        l1_df = mode_df[
            mode_df[
                "_data_level"
            ].eq(
                "L1"
            )
        ]

        print(
            f"  - Mode {mode}: "
            f"L2 samples={len(l2_df)}, "
            f"L1 samples={len(l1_df)}"
        )

        (
            l2_templates,
            l2_meta,
        ) = cluster_one_set(
            l2_df,
            mode,
            "L2",
            K_MODE,
            pf_cols,
            miss_cols,
        )

        remaining = max(
            0,
            K_MODE
            - len(
                l2_meta
            ),
        )

        l1_selected_templates = (
            pd.DataFrame()
        )

        l1_selected_meta = (
            pd.DataFrame()
        )

        if (
            remaining > 0
            and not l1_df.empty
        ):

            l1_try_k = min(
                K_MODE,
                max(
                    remaining,
                    len(
                        l1_df
                    )
                    // max(
                        MIN_CLUSTER_SIZE,
                        1,
                    ),
                ),
            )

            (
                l1_templates,
                l1_meta,
            ) = cluster_one_set(
                l1_df,
                mode,
                "L1",
                l1_try_k,
                pf_cols,
                miss_cols,
            )

            (
                l1_selected_templates,
                l1_selected_meta,
            ) = select_top_clusters(
                l1_templates,
                l1_meta,
                remaining,
            )

        mode_templates = (
            pd.concat(
                [
                    l2_templates,
                    l1_selected_templates,
                ],
                ignore_index=True,
            )
        )

        mode_meta = (
            pd.concat(
                [
                    l2_meta,
                    l1_selected_meta,
                ],
                ignore_index=True,
            )
        )

        if mode_templates.empty:

            print(
                "    -> no templates produced"
            )

            continue

        l2_order = (
            mode_meta.loc[
                mode_meta[
                    "source_level"
                ].eq(
                    "L2"
                )
            ]
            .sort_values(
                [
                    "cluster_size",
                    "cluster_mean_missing_ratio",
                    "template_id",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )[
                "template_id"
            ]
            .tolist()
        )

        l1_order = (
            mode_meta.loc[
                mode_meta[
                    "source_level"
                ].eq(
                    "L1"
                )
            ]
            .sort_values(
                [
                    "cluster_size",
                    "cluster_mean_missing_ratio",
                    "template_id",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )[
                "template_id"
            ]
            .tolist()
        )

        rank_map = {
            template_id:
                index + 1

            for index, template_id
            in enumerate(
                l2_order
                + l1_order
            )
        }

        mode_templates[
            "template_rank_in_mode"
        ] = (
            mode_templates[
                "template_id"
            ]
            .map(
                rank_map
            )
        )

        mode_meta[
            "template_rank_in_mode"
        ] = (
            mode_meta[
                "template_id"
            ]
            .map(
                rank_map
            )
        )

        mode_templates = (
            mode_templates
            .sort_values(
                "template_rank_in_mode"
            )
            .reset_index(
                drop=True
            )
        )

        mode_meta = (
            mode_meta
            .sort_values(
                "template_rank_in_mode"
            )
            .reset_index(
                drop=True
            )
        )

        kept_l2 = int(
            mode_meta[
                "source_level"
            ].eq(
                "L2"
            ).sum()
        )

        kept_l1 = int(
            mode_meta[
                "source_level"
            ].eq(
                "L1"
            ).sum()
        )

        print(
            f"    -> templates="
            f"{len(mode_meta)} "
            f"(L2={kept_l2}, "
            f"L1 fill={kept_l1}), "
            f"target={K_MODE}"
        )

        all_templates.append(
            mode_templates
        )

        all_meta.append(
            mode_meta
        )

    if not all_templates:

        raise RuntimeError(
            "No route templates "
            "were produced."
        )

    templates = pd.concat(
        all_templates,
        ignore_index=True,
    )

    meta = pd.concat(
        all_meta,
        ignore_index=True,
    )

    if (
        templates[
            "template_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate template_id "
            "values were generated."
        )

    if (
        set(
            templates[
                "template_id"
            ]
        )
        != set(
            meta[
                "template_id"
            ]
        )
    ):

        raise RuntimeError(
            "Template and metadata files "
            "contain different template IDs."
        )

    templates.to_csv(
        OUT_TEMPLATES,
        index=False,
        encoding="utf-8-sig",
    )

    meta.to_csv(
        OUT_META,
        index=False,
        encoding="utf-8-sig",
    )

    winsor_limits.to_csv(
        OUT_WINSOR,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        meta
        .groupby(
            [
                "growth_mode",
                "source_level",
            ],
            as_index=False,
        )
        .agg(
            template_count=(
                "template_id",
                "nunique",
            ),

            represented_samples=(
                "cluster_size",
                "sum",
            ),

            median_cluster_size=(
                "cluster_size",
                "median",
            ),

            mean_template_missing_fraction=(
                "template_missing_fraction",
                "mean",
            ),
        )
    )

    print(
        "\n[TEMPLATE SUMMARY]"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n[OUTPUTS]"
    )

    print(
        f"Templates      : "
        f"{OUT_TEMPLATES}"
    )

    print(
        f"Metadata       : "
        f"{OUT_META}"
    )

    print(
        f"Winsor limits  : "
        f"{OUT_WINSOR}"
    )

    print(
        "\n[OK] "
        "Route-template build completed."
    )

    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()