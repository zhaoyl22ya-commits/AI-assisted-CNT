# optimize_physical_state.py
# =============================================================================
# CNT 3.0 — physical-state search for each route package
#
# Search variables:
#   np_size_nm and raw C_over_M
#
# Fixed route variables:
#   S_over_M from C4
#   p_C / p_H2 / p_C_over_p_H2 from observed P8 / P9 / P10
#
# Each grid point rebuilds N1–N18 and C1–C12 with the unified z_nuc/z_c
# modules, then predicts q10/q50/q90.
#
# Balanced = max q50; Potential = max q90.
# =============================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "promptsV2"
FEATURE_BUILDERS_DIR = SCRIPT_DIR / "feature_builders"

INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "rank_catalysts"
    / "process_packages_union_q50_q90.csv"
)
OUTDIR = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "optimize_physical_state"
)

ZNUC_FILE = FEATURE_BUILDERS_DIR / "build_znuc.py"
ZC_FILE = FEATURE_BUILDERS_DIR / "build_zc.py"
ELEMENTS_FILE = PROMPTS_DIR / "elements_properties.txt"

MODEL_DIR = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "l12_quantile"
    / "final_models"
)
Q10_MODEL = MODEL_DIR / "L12_composite_q10.txt"
Q50_MODEL = MODEL_DIR / "L12_composite_q50.txt"
Q90_MODEL = MODEL_DIR / "L12_composite_q90.txt"

NP_MIN, NP_MAX, NP_STEP = 1.0, 30.0, 1.0
CM_MIN, CM_MAX, CM_STEP = 2.5, 28.0, 0.5
TOPK_POINTS = 10
PRED_CHUNK = 200000
NEAR_OPTIMAL_TOL = 0.01
EXACT_MAX_ATOL = 1e-12
SURFACE_PLOT_COUNT = 8
FIGURE_DPI = 600


N_COLS = [f"N{i}" for i in range(1, 19)]
C_COLS = [f"C{i}" for i in range(1, 13)]
F_COLS = [f"F{i}" for i in range(1, 19)]
P_COLS = [f"P{i}" for i in range(1, 13)]
PHYSICAL_COLS = N_COLS + C_COLS + F_COLS + P_COLS
MODEL_COLS = PHYSICAL_COLS + [f"miss_{c}" for c in PHYSICAL_COLS]


# =============================================================================
# HELPERS
# =============================================================================

def to_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found:\n  {path}")

    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def resolve_file(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()

    if path.is_file():
        return path

    matches = [p.resolve() for p in PROJECT_DIR.rglob(path.name) if p.is_file()]
    if len(matches) == 1:
        print(f"[WARN] Using project match: {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files named {path.name!r} found.")
    raise FileNotFoundError(f"File not found:\n  {path}")


def load_module(name: str, path: str | Path) -> ModuleType:
    path = resolve_file(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_grid(low: float, high: float, step: float) -> np.ndarray:
    if step <= 0 or high < low:
        raise ValueError("Invalid grid range.")
    return np.arange(low, high + step * 0.5, step, dtype=float)


# =============================================================================
# INPUT STATE
# =============================================================================

def prepare_routes(df: pd.DataFrame) -> pd.DataFrame:
    required = ["catalyst_id", "template_id", "metal1", "frac1", "C4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Input is missing required columns: {missing}")

    df = df.copy()
    df["catalyst_id"] = df["catalyst_id"].astype(str).str.strip()
    df["template_id"] = df["template_id"].astype(str).str.strip()
    df["C4"] = pd.to_numeric(df["C4"], errors="coerce")

    df = df.loc[
        df["catalyst_id"].ne("")
        & df["template_id"].ne("")
    ].copy()

    if "combination_id" not in df.columns:
        df["combination_id"] = df["catalyst_id"] + "||" + df["template_id"]

    df = df.drop_duplicates("combination_id", keep="first").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid route package remains.")
    return df


def recipe_from_row(row: pd.Series) -> list[tuple[str, float]]:
    recipe = []

    for i in (1, 2, 3):
        metal = row.get(f"metal{i}")
        fraction = to_float(row.get(f"frac{i}", np.nan))

        if (
            metal is not None
            and not pd.isna(metal)
            and str(metal).strip()
            and np.isfinite(fraction)
            and fraction > 0
        ):
            recipe.append((str(metal).strip(), fraction))

    if not recipe:
        return []

    total = sum(fraction for _, fraction in recipe)
    return [(metal, fraction / total) for metal, fraction in recipe]


def route_temperature(row: pd.Series) -> float:
    for column in ("temperature_K", "C1", "P1"):
        value = to_float(row.get(column, np.nan))
        if np.isfinite(value) and value > 0:
            return value
    return np.nan


def get_observed_route_value(
    row: pd.Series,
    feature: str,
) -> float:
    """
    Return a route feature only when its miss_* flag says it was observed.

    Missing physical features are stored as 0.0 in model space, so P8/P9/P10
    must be restored to NaN before they are passed back into build_zc.py.
    """
    miss_value = to_float(
        row.get(
            f"miss_{feature}",
            1,
        )
    )

    if (
        not np.isfinite(miss_value)
        or miss_value > 0
    ):
        return np.nan

    return to_float(
        row.get(
            feature,
            np.nan,
        )
    )


# =============================================================================
# FEATURE BUILD
# =============================================================================

def enforce_model_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for feature in PHYSICAL_COLS:
        miss = f"miss_{feature}"

        if feature not in df.columns:
            df[feature] = np.nan
        if miss not in df.columns:
            df[miss] = df[feature].isna().astype(int)

        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[miss] = (
            pd.to_numeric(df[miss], errors="coerce")
            .fillna(1)
            .gt(0)
            .astype(int)
        )

        df.loc[df[feature].isna(), miss] = 1
        df.loc[df[miss].eq(1), feature] = np.nan

    return df


def build_route_grid(
    route: pd.Series,
    np_values: np.ndarray,
    cm_values: np.ndarray,
    znuc_module: ModuleType,
    zc_module: ModuleType,
    elements_df: pd.DataFrame,
) -> pd.DataFrame:
    recipe = recipe_from_row(route)
    temperature = route_temperature(route)
    fixed_s_m = get_observed_route_value(
        route,
        "C4",
    )

    p_C = get_observed_route_value(
        route,
        "P8",
    )
    p_H2 = get_observed_route_value(
        route,
        "P9",
    )
    p_C_over_p_H2 = get_observed_route_value(
        route,
        "P10",
    )

    if not recipe:
        raise RuntimeError(f"Invalid recipe: {route.get('catalyst_id', '')}")
    if not np.isfinite(temperature):
        raise RuntimeError(f"Missing temperature: {route.get('template_id', '')}")

    base = route.to_dict()
    rows = []

    for np_size in np_values:
        znuc = znuc_module.compute_znuc(
            recipe=recipe,
            T_value=temperature,
            T_unit="K",
            np_diameter_nm=float(np_size),
            elements_df=elements_df,
        )

        for c_m in cm_values:
            zc = zc_module.compute_zc(
                znuc=znuc,
                T_K=temperature,
                C_over_M=float(c_m),
                S_over_M=fixed_s_m,
                np_size_nm=float(np_size),
                p_C=p_C,
                p_H2=p_H2,
                p_C_over_p_H2=p_C_over_p_H2,
                elements_df=elements_df,
            )

            point = base.copy()

            for feature in N_COLS:
                point[feature] = zc.get(
                    f"znuc_{feature}",
                    znuc.get(feature, np.nan),
                )
                point[f"miss_{feature}"] = zc.get(
                    f"znuc_miss_{feature}",
                    znuc.get(f"miss_{feature}", 1),
                )

            for feature in C_COLS:
                point[feature] = zc.get(feature, np.nan)
                point[f"miss_{feature}"] = zc.get(f"miss_{feature}", 1)

            point["grid_np_size_nm"] = float(np_size)
            point["grid_C_over_M"] = float(c_m)
            point["fixed_S_over_M"] = fixed_s_m
            rows.append(point)

    return enforce_model_missing(pd.DataFrame(rows))


# =============================================================================
# MODEL PREDICTION
# =============================================================================

def load_booster(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(resolve_file(path)))


def model_order(model: lgb.Booster, label: str) -> list[str]:
    names = model.feature_name()

    if model.num_feature() != len(MODEL_COLS):
        raise RuntimeError(
            f"{label} expects {model.num_feature()} features; "
            f"current schema has {len(MODEL_COLS)}."
        )

    if set(names) == set(MODEL_COLS):
        return list(names)
    if all(str(name).startswith("Column_") for name in names):
        return list(MODEL_COLS)

    raise RuntimeError(f"{label} feature names do not match the 120D schema.")


def predict_chunks(
    model: lgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:
    predictions = []

    for start in range(0, len(X), PRED_CHUNK):
        predictions.append(
            np.asarray(
                model.predict(X.iloc[start : start + PRED_CHUNK]),
                dtype=float,
            )
        )

    return np.concatenate(predictions)


def predict_grid(
    grid: pd.DataFrame,
    models: dict[str, lgb.Booster],
    orders: dict[str, list[str]],
) -> pd.DataFrame:
    grid = grid.copy()

    q10 = predict_chunks(models["q10"], grid[orders["q10"]])
    q50 = predict_chunks(models["q50"], grid[orders["q50"]])
    q90 = predict_chunks(models["q90"], grid[orders["q90"]])

    raw = np.column_stack([q10, q50, q90])
    ordered = np.sort(raw, axis=1)
    crossing = (q10 > q50) | (q50 > q90) | (q10 > q90)

    # Keep model identity: Balanced uses raw q50 and Potential uses raw q90.
    grid["pred_q10_grid"] = q10
    grid["pred_q50_grid"] = q50
    grid["pred_q90_grid"] = q90

    # Ordered values are retained only for interval description and diagnostics.
    grid["pred_q10_ordered_grid"] = ordered[:, 0]
    grid["pred_q50_ordered_grid"] = ordered[:, 1]
    grid["pred_q90_ordered_grid"] = ordered[:, 2]
    grid["interval_width_grid"] = ordered[:, 2] - ordered[:, 0]
    grid["quantile_crossing_detected"] = crossing

    # Legacy alias retained so downstream readers using the old column do not break.
    grid["quantile_crossing_corrected"] = crossing
    return grid


# =============================================================================
# RECOMMENDATION AND SURFACE AUDIT
# =============================================================================

def normalized_center_distance(grid: pd.DataFrame) -> pd.Series:
    np_min, np_max = grid["grid_np_size_nm"].min(), grid["grid_np_size_nm"].max()
    cm_min, cm_max = grid["grid_C_over_M"].min(), grid["grid_C_over_M"].max()
    np_mid, cm_mid = (np_min + np_max) / 2.0, (cm_min + cm_max) / 2.0
    np_scale = max(float(np_max - np_min), 1e-12)
    cm_scale = max(float(cm_max - cm_min), 1e-12)
    return np.sqrt(
        ((grid["grid_np_size_nm"] - np_mid) / np_scale) ** 2
        + ((grid["grid_C_over_M"] - cm_mid) / cm_scale) ** 2
    )


def representative_max_point(
    grid: pd.DataFrame,
    primary: str,
    secondary: str,
) -> pd.Series:
    maximum = float(grid[primary].max())
    exact = grid.loc[
        np.isclose(grid[primary], maximum, rtol=0.0, atol=EXACT_MAX_ATOL)
    ].copy()

    np_center = float(exact["grid_np_size_nm"].mean())
    cm_center = float(exact["grid_C_over_M"].mean())
    np_scale = max(
        float(
            grid["grid_np_size_nm"].max()
            - grid["grid_np_size_nm"].min()
        ),
        1e-12,
    )
    cm_scale = max(
        float(
            grid["grid_C_over_M"].max()
            - grid["grid_C_over_M"].min()
        ),
        1e-12,
    )
    exact["_plateau_center_distance"] = np.sqrt(
        ((exact["grid_np_size_nm"] - np_center) / np_scale) ** 2
        + ((exact["grid_C_over_M"] - cm_center) / cm_scale) ** 2
    )

    return exact.sort_values(
        [
            secondary,
            "_plateau_center_distance",
            "grid_np_size_nm",
            "grid_C_over_M",
        ],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]


def surface_audit(
    grid: pd.DataFrame,
    score_column: str,
    prefix: str,
    near_tolerance: float,
) -> dict[str, Any]:
    maximum = float(grid[score_column].max())
    exact = grid.loc[
        np.isclose(
            grid[score_column],
            maximum,
            rtol=0.0,
            atol=EXACT_MAX_ATOL,
        )
    ].copy()
    near = grid.loc[
        grid[score_column] >= maximum - near_tolerance
    ].copy()

    np_min = float(grid["grid_np_size_nm"].min())
    np_max = float(grid["grid_np_size_nm"].max())
    cm_min = float(grid["grid_C_over_M"].min())
    cm_max = float(grid["grid_C_over_M"].max())

    np_lower = bool(
        np.isclose(
            exact["grid_np_size_nm"],
            np_min,
        ).any()
    )
    np_upper = bool(
        np.isclose(
            exact["grid_np_size_nm"],
            np_max,
        ).any()
    )
    cm_lower = bool(
        np.isclose(
            exact["grid_C_over_M"],
            cm_min,
        ).any()
    )
    cm_upper = bool(
        np.isclose(
            exact["grid_C_over_M"],
            cm_max,
        ).any()
    )

    np_edge_mask = (
        np.isclose(
            exact["grid_np_size_nm"],
            np_min,
        )
        | np.isclose(
            exact["grid_np_size_nm"],
            np_max,
        )
    )
    cm_edge_mask = (
        np.isclose(
            exact["grid_C_over_M"],
            cm_min,
        )
        | np.isclose(
            exact["grid_C_over_M"],
            cm_max,
        )
    )

    has_corner_maximum = bool(
        (
            np_edge_mask
            & cm_edge_mask
        ).any()
    )
    has_edge_maximum = bool(
        np_edge_mask.any()
        or cm_edge_mask.any()
    )

    if has_corner_maximum:
        status = "corner_boundary_limited"
    elif has_edge_maximum:
        status = "edge_boundary_limited"
    else:
        status = "interior_grid_maximum"

    return {
        f"{prefix}_maximum_score":
            maximum,

        f"{prefix}_exact_max_count":
            int(
                len(
                    exact
                )
            ),

        f"{prefix}_near_optimal_count":
            int(
                len(
                    near
                )
            ),

        f"{prefix}_near_optimal_np_min":
            float(
                near[
                    "grid_np_size_nm"
                ].min()
            ),

        f"{prefix}_near_optimal_np_max":
            float(
                near[
                    "grid_np_size_nm"
                ].max()
            ),

        f"{prefix}_near_optimal_cm_min":
            float(
                near[
                    "grid_C_over_M"
                ].min()
            ),

        f"{prefix}_near_optimal_cm_max":
            float(
                near[
                    "grid_C_over_M"
                ].max()
            ),

        f"{prefix}_near_optimal_grid_fraction":
            float(
                len(
                    near
                )
                / len(
                    grid
                )
            ),

        f"{prefix}_np_lower_boundary":
            np_lower,

        f"{prefix}_np_upper_boundary":
            np_upper,

        f"{prefix}_cm_lower_boundary":
            cm_lower,

        f"{prefix}_cm_upper_boundary":
            cm_upper,

        f"{prefix}_search_status":
            status,
    }


def annotate_surface(
    grid: pd.DataFrame,
    near_tolerance: float,
) -> pd.DataFrame:
    grid = grid.copy()

    q50_max = float(
        grid[
            "pred_q50_grid"
        ].max()
    )
    q90_max = float(
        grid[
            "pred_q90_grid"
        ].max()
    )

    grid[
        "is_q50_exact_max"
    ] = np.isclose(
        grid[
            "pred_q50_grid"
        ],
        q50_max,
        rtol=0.0,
        atol=EXACT_MAX_ATOL,
    )

    grid[
        "is_q90_exact_max"
    ] = np.isclose(
        grid[
            "pred_q90_grid"
        ],
        q90_max,
        rtol=0.0,
        atol=EXACT_MAX_ATOL,
    )

    grid[
        "is_q50_near_optimal"
    ] = (
        grid[
            "pred_q50_grid"
        ]
        >= q50_max
        - near_tolerance
    )

    grid[
        "is_q90_near_optimal"
    ] = (
        grid[
            "pred_q90_grid"
        ]
        >= q90_max
        - near_tolerance
    )

    return grid


def select_top_points(
    grid: pd.DataFrame,
    top_k: int,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    ranked_base = grid.copy()

    ranked_base[
        "_search_center_distance"
    ] = normalized_center_distance(
        ranked_base
    )

    balanced = (
        ranked_base
        .sort_values(
            [
                "pred_q50_grid",
                "pred_q90_grid",
                "_search_center_distance",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    potential = (
        ranked_base
        .sort_values(
            [
                "pred_q90_grid",
                "pred_q50_grid",
                "_search_center_distance",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .copy()
    )

    balanced[
        "rank_balanced_grid"
    ] = np.arange(
        1,
        len(
            balanced
        )
        + 1,
    )

    potential[
        "rank_potential_grid"
    ] = np.arange(
        1,
        len(
            potential
        )
        + 1,
    )

    balanced_top = (
        balanced
        .head(
            top_k
        )
    )

    potential_top = (
        potential
        .head(
            top_k
        )
    )

    def point_key(
        row: pd.Series,
    ) -> tuple[
        float,
        float,
    ]:
        return (
            float(
                row[
                    "grid_np_size_nm"
                ]
            ),
            float(
                row[
                    "grid_C_over_M"
                ]
            ),
        )

    balanced_ranks = {
        point_key(
            row
        ):
            int(
                row[
                    "rank_balanced_grid"
                ]
            )

        for (
            _,
            row,
        )
        in balanced_top.iterrows()
    }

    potential_ranks = {
        point_key(
            row
        ):
            int(
                row[
                    "rank_potential_grid"
                ]
            )

        for (
            _,
            row,
        )
        in potential_top.iterrows()
    }

    union = grid.copy()

    keys = [
        point_key(
            row
        )

        for (
            _,
            row,
        )
        in union.iterrows()
    ]

    union[
        "rank_balanced_grid"
    ] = [
        balanced_ranks.get(
            key,
            np.nan,
        )

        for key
        in keys
    ]

    union[
        "rank_potential_grid"
    ] = [
        potential_ranks.get(
            key,
            np.nan,
        )

        for key
        in keys
    ]

    union[
        "selected_by_balanced"
    ] = union[
        "rank_balanced_grid"
    ].notna()

    union[
        "selected_by_potential"
    ] = union[
        "rank_potential_grid"
    ].notna()

    union = (
        union.loc[
            union[
                "selected_by_balanced"
            ]
            | union[
                "selected_by_potential"
            ]
        ]
        .copy()
    )

    union[
        "selection_reason"
    ] = np.select(
        [
            union[
                "selected_by_balanced"
            ]
            & union[
                "selected_by_potential"
            ],

            union[
                "selected_by_balanced"
            ],

            union[
                "selected_by_potential"
            ],
        ],
        [
            "balanced+potential",
            "balanced",
            "potential",
        ],
        default="",
    )

    union[
        "best_available_grid_rank"
    ] = (
        union[
            [
                "rank_balanced_grid",
                "rank_potential_grid",
            ]
        ]
        .min(
            axis=1,
            skipna=True,
        )
    )

    keep = [
        "catalyst_id",
        "template_id",
        "combination_id",
        "growth_mode",
        "source_level",
        "grid_np_size_nm",
        "grid_C_over_M",
        "fixed_S_over_M",
        "pred_q10_grid",
        "pred_q50_grid",
        "pred_q90_grid",
        "pred_q10_ordered_grid",
        "pred_q50_ordered_grid",
        "pred_q90_ordered_grid",
        "interval_width_grid",
        "quantile_crossing_detected",
        "quantile_crossing_corrected",
        "is_q50_exact_max",
        "is_q90_exact_max",
        "is_q50_near_optimal",
        "is_q90_near_optimal",
        "rank_balanced_grid",
        "rank_potential_grid",
        "selected_by_balanced",
        "selected_by_potential",
        "selection_reason",
        "best_available_grid_rank",
    ]

    union = union[
        [
            column

            for column
            in keep

            if column
            in union.columns
        ]
    ]

    balanced_representative = (
        representative_max_point(
            grid,
            "pred_q50_grid",
            "pred_q90_grid",
        )
    )

    potential_representative = (
        representative_max_point(
            grid,
            "pred_q90_grid",
            "pred_q50_grid",
        )
    )

    return (
        balanced_representative,
        potential_representative,
        union.sort_values(
            [
                "best_available_grid_rank",
                "pred_q50_grid",
                "pred_q90_grid",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
    )


def build_recommendation(
    route: pd.Series,
    balanced: pd.Series,
    potential: pd.Series,
    audit: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    output = route.to_dict()

    output.update(
        {
            "recommended_np_size_nm_balanced":
                balanced[
                    "grid_np_size_nm"
                ],

            "recommended_C_over_M_balanced":
                balanced[
                    "grid_C_over_M"
                ],

            "recommended_S_over_M_fixed":
                balanced[
                    "fixed_S_over_M"
                ],

            "recommended_pred_q10_balanced":
                balanced[
                    "pred_q10_grid"
                ],

            "recommended_pred_q50_balanced":
                balanced[
                    "pred_q50_grid"
                ],

            "recommended_pred_q90_balanced":
                balanced[
                    "pred_q90_grid"
                ],

            "recommended_interval_width_balanced":
                balanced[
                    "interval_width_grid"
                ],

            "recommended_np_size_nm_potential":
                potential[
                    "grid_np_size_nm"
                ],

            "recommended_C_over_M_potential":
                potential[
                    "grid_C_over_M"
                ],

            "recommended_pred_q10_potential":
                potential[
                    "pred_q10_grid"
                ],

            "recommended_pred_q50_potential":
                potential[
                    "pred_q50_grid"
                ],

            "recommended_pred_q90_potential":
                potential[
                    "pred_q90_grid"
                ],

            "recommended_interval_width_potential":
                potential[
                    "interval_width_grid"
                ],
        }
    )

    output.update(
        audit
    )

    old_q50 = (
        to_float(
            route.get(
                "pred_q50",
                np.nan,
            )
        )
    )

    old_q90 = (
        to_float(
            route.get(
                "pred_q90",
                np.nan,
            )
        )
    )

    output[
        "delta_q50_balanced_vs_input"
    ] = (
        balanced[
            "pred_q50_grid"
        ]
        - old_q50

        if np.isfinite(
            old_q50
        )

        else np.nan
    )

    output[
        "delta_q90_potential_vs_input"
    ] = (
        potential[
            "pred_q90_grid"
        ]
        - old_q90

        if np.isfinite(
            old_q90
        )

        else np.nan
    )

    return output


def surface_output_columns(
    grid: pd.DataFrame,
) -> list[
    str,
]:
    keep = [
        "catalyst_id",
        "template_id",
        "combination_id",
        "growth_mode",
        "source_level",
        "grid_np_size_nm",
        "grid_C_over_M",
        "fixed_S_over_M",
        "pred_q10_grid",
        "pred_q50_grid",
        "pred_q90_grid",
        "pred_q10_ordered_grid",
        "pred_q50_ordered_grid",
        "pred_q90_ordered_grid",
        "interval_width_grid",
        "quantile_crossing_detected",
        "quantile_crossing_corrected",
        "is_q50_exact_max",
        "is_q90_exact_max",
        "is_q50_near_optimal",
        "is_q90_near_optimal",
    ]

    return [
        column

        for column
        in keep

        if column
        in grid.columns
    ]


def plot_surface_panels(
    surface: pd.DataFrame,
    recommendations: pd.DataFrame,
    score_column: str,
    recommendation_np: str,
    recommendation_cm: str,
    recommendation_score: str,
    title: str,
    output_path: Path,
    max_routes: int,
    dpi: int,
) -> None:
    selected_routes = (
        recommendations
        .sort_values(
            recommendation_score,
            ascending=False,
        )
        .head(
            max_routes
        )[
            "combination_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    if not selected_routes:
        return

    ncols = min(
        3,
        len(
            selected_routes
        ),
    )

    nrows = int(
        np.ceil(
            len(
                selected_routes
            )
            / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            5.0
            * ncols,
            4.1
            * nrows,
        ),
        squeeze=False,
    )

    for (
        axis,
        combination_id,
    ) in zip(
        axes.flat,
        selected_routes,
    ):

        group = surface[
            surface[
                "combination_id"
            ].astype(
                str
            )
            == combination_id
        ]

        pivot = (
            group.pivot(
                index="grid_np_size_nm",
                columns="grid_C_over_M",
                values=score_column,
            )
            .sort_index()
            .sort_index(
                axis=1
            )
        )

        mesh = axis.pcolormesh(
            pivot.columns.to_numpy(
                float
            ),
            pivot.index.to_numpy(
                float
            ),
            pivot.to_numpy(
                float
            ),
            shading="auto",
        )

        recommendation = (
            recommendations.loc[
                recommendations[
                    "combination_id"
                ].astype(
                    str
                )
                == combination_id
            ]
            .iloc[
                0
            ]
        )

        axis.scatter(
            [
                recommendation[
                    recommendation_cm
                ]
            ],
            [
                recommendation[
                    recommendation_np
                ]
            ],
            marker="*",
            s=90,
            label="Representative maximum",
        )

        axis.set_title(
            str(
                combination_id
            ),
            fontsize=9,
        )

        axis.set_xlabel(
            "Raw C/M"
        )

        axis.set_ylabel(
            "NP size (nm)"
        )

        fig.colorbar(
            mesh,
            ax=axis,
            shrink=0.82,
        )

    for axis in axes.flat[
        len(
            selected_routes
        ):
    ]:
        axis.set_visible(
            False
        )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_boundary_summary(
    recommendations: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    statuses = [
        "interior_grid_maximum",
        "edge_boundary_limited",
        "corner_boundary_limited",
    ]

    q50_counts = (
        recommendations[
            "q50_search_status"
        ]
        .value_counts()
        .reindex(
            statuses,
            fill_value=0,
        )
    )

    q90_counts = (
        recommendations[
            "q90_search_status"
        ]
        .value_counts()
        .reindex(
            statuses,
            fill_value=0,
        )
    )

    x = np.arange(
        len(
            statuses
        )
    )

    width = 0.36

    fig, axis = plt.subplots(
        figsize=(
            9.0,
            5.5,
        )
    )

    axis.bar(
        x
        - width
        / 2,
        q50_counts.to_numpy(),
        width,
        label="Balanced q50",
    )

    axis.bar(
        x
        + width
        / 2,
        q90_counts.to_numpy(),
        width,
        label="Potential q90",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        statuses,
        rotation=20,
        ha="right",
    )

    axis.set_ylabel(
        "Number of route packages"
    )

    axis.set_title(
        "Boundary status of physical-state grid maxima"
    )

    axis.legend(
        frameon=False
    )

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

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CNT 3.0 physical-state optimization"
        )
    )

    parser.add_argument(
        "--input_csv",
        default=str(
            INPUT_CSV
        ),
    )

    parser.add_argument(
        "--outdir",
        default=str(
            OUTDIR
        ),
    )

    parser.add_argument(
        "--q10_model",
        default=str(
            Q10_MODEL
        ),
    )

    parser.add_argument(
        "--q50_model",
        default=str(
            Q50_MODEL
        ),
    )

    parser.add_argument(
        "--q90_model",
        default=str(
            Q90_MODEL
        ),
    )

    parser.add_argument(
        "--np_min",
        type=float,
        default=NP_MIN,
    )

    parser.add_argument(
        "--np_max",
        type=float,
        default=NP_MAX,
    )

    parser.add_argument(
        "--np_step",
        type=float,
        default=NP_STEP,
    )

    parser.add_argument(
        "--cm_min",
        type=float,
        default=CM_MIN,
    )

    parser.add_argument(
        "--cm_max",
        type=float,
        default=CM_MAX,
    )

    parser.add_argument(
        "--cm_step",
        type=float,
        default=CM_STEP,
    )

    parser.add_argument(
        "--topk_points",
        type=int,
        default=TOPK_POINTS,
    )

    parser.add_argument(
        "--near_optimal_tol",
        type=float,
        default=NEAR_OPTIMAL_TOL,
    )

    parser.add_argument(
        "--surface_plot_count",
        type=int,
        default=SURFACE_PLOT_COUNT,
    )

    parser.add_argument(
        "--figure_dpi",
        type=int,
        default=FIGURE_DPI,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.topk_points <= 0:
        raise ValueError(
            "topk_points must be positive."
        )

    if args.near_optimal_tol < 0:
        raise ValueError(
            "near_optimal_tol must be non-negative."
        )

    if args.surface_plot_count <= 0:
        raise ValueError(
            "surface_plot_count must be positive."
        )

    input_path = (
        Path(
            args.input_csv
        )
        .expanduser()
        .resolve()
    )

    outdir = (
        Path(
            args.outdir
        )
        .expanduser()
        .resolve()
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np_values = make_grid(
        args.np_min,
        args.np_max,
        args.np_step,
    )

    cm_values = make_grid(
        args.cm_min,
        args.cm_max,
        args.cm_step,
    )

    print(
        "=" * 108
    )

    print(
        "[PHYSICAL-STATE OPTIMIZATION START]"
    )

    print(
        f"Project directory   : "
        f"{PROJECT_DIR}"
    )

    print(
        f"Input route packages: "
        f"{input_path}"
    )

    print(
        f"Output directory    : "
        f"{outdir}"
    )

    print(
        f"NP grid             : "
        f"{args.np_min}–{args.np_max}, "
        f"step={args.np_step}, "
        f"n={len(np_values)}"
    )

    print(
        f"Raw C/M grid        : "
        f"{args.cm_min}–{args.cm_max}, "
        f"step={args.cm_step}, "
        f"n={len(cm_values)}"
    )

    print(
        f"Points per route    : "
        f"{len(np_values) * len(cm_values)}"
    )

    print(
        f"Near-optimal tol    : "
        f"{args.near_optimal_tol}"
    )

    print(
        "=" * 108
    )

    routes = prepare_routes(
        read_csv(
            input_path
        )
    )

    znuc = load_module(
        "cnt_znuc_builder",
        ZNUC_FILE,
    )

    zc = load_module(
        "cnt_zc_builder",
        ZC_FILE,
    )

    for (
        module,
        functions,
    ) in [
        (
            znuc,
            [
                "compute_znuc",
                "load_elements_properties",
            ],
        ),
        (
            zc,
            [
                "compute_zc",
            ],
        ),
    ]:

        for function in functions:

            if not hasattr(
                module,
                function,
            ):

                raise RuntimeError(
                    f"Required function missing: "
                    f"{function}"
                )

    elements_df = (
        znuc
        .load_elements_properties(
            ELEMENTS_FILE
        )
    )

    models = {
        "q10":
            load_booster(
                args.q10_model
            ),

        "q50":
            load_booster(
                args.q50_model
            ),

        "q90":
            load_booster(
                args.q90_model
            ),
    }

    orders = {
        label:
            model_order(
                model,
                label,
            )

        for (
            label,
            model,
        )
        in models.items()
    }

    recommendations = []
    top_points = []
    full_surfaces = []

    crossing_count = 0

    for (
        index,
        (
            _,
            route,
        ),
    ) in enumerate(
        routes.iterrows(),
        start=1,
    ):

        grid = build_route_grid(
            route,
            np_values,
            cm_values,
            znuc,
            zc,
            elements_df,
        )

        grid = predict_grid(
            grid,
            models,
            orders,
        )

        grid = annotate_surface(
            grid,
            args.near_optimal_tol,
        )

        crossing_count += int(
            grid[
                "quantile_crossing_detected"
            ].sum()
        )

        (
            balanced,
            potential,
            selected,
        ) = select_top_points(
            grid,
            args.topk_points,
        )

        audit = {}

        audit.update(
            surface_audit(
                grid,
                "pred_q50_grid",
                "q50",
                args.near_optimal_tol,
            )
        )

        audit.update(
            surface_audit(
                grid,
                "pred_q90_grid",
                "q90",
                args.near_optimal_tol,
            )
        )

        recommendations.append(
            build_recommendation(
                route,
                balanced,
                potential,
                audit,
            )
        )

        top_points.append(
            selected
        )

        full_surfaces.append(
            grid[
                surface_output_columns(
                    grid
                )
            ].copy()
        )

        print(
            f"[ROUTE {index}/{len(routes)}] "
            f"{route['catalyst_id']} | "
            f"{route['template_id']} | "
            f"balanced=("
            f"{balanced['grid_np_size_nm']:.3g} nm, "
            f"C/M="
            f"{balanced['grid_C_over_M']:.3g}, "
            f"{audit['q50_search_status']}) | "
            f"potential=("
            f"{potential['grid_np_size_nm']:.3g} nm, "
            f"C/M="
            f"{potential['grid_C_over_M']:.3g}, "
            f"{audit['q90_search_status']})"
        )

    recommendations = pd.DataFrame(
        recommendations
    )

    top_points = pd.concat(
        top_points,
        ignore_index=True,
    )

    full_surface = pd.concat(
        full_surfaces,
        ignore_index=True,
    )

    out_best = (
        outdir
        / "process_packages_with_physical_state.csv"
    )

    out_points = (
        outdir
        / "physical_state_top_points.csv"
    )

    out_surface = (
        outdir
        / "physical_state_full_surface.csv"
    )

    out_summary = (
        outdir
        / "run_summary_L123_003.json"
    )

    out_q50_figure = (
        outdir
        / "figS_l123_q50_physical_surfaces.png"
    )

    out_q90_figure = (
        outdir
        / "figS_l123_q90_physical_surfaces.png"
    )

    out_boundary_figure = (
        outdir
        / "fig_l123_boundary_summary.png"
    )

    recommendations.to_csv(
        out_best,
        index=False,
        encoding="utf-8-sig",
    )

    top_points.to_csv(
        out_points,
        index=False,
        encoding="utf-8-sig",
    )

    full_surface.to_csv(
        out_surface,
        index=False,
        encoding="utf-8-sig",
    )

    plot_surface_panels(
        full_surface,
        recommendations,
        "pred_q50_grid",
        "recommended_np_size_nm_balanced",
        "recommended_C_over_M_balanced",
        "recommended_pred_q50_balanced",
        "Representative q50 physical-state response surfaces",
        out_q50_figure,
        args.surface_plot_count,
        args.figure_dpi,
    )

    plot_surface_panels(
        full_surface,
        recommendations,
        "pred_q90_grid",
        "recommended_np_size_nm_potential",
        "recommended_C_over_M_potential",
        "recommended_pred_q90_potential",
        "Representative q90 physical-state response surfaces",
        out_q90_figure,
        args.surface_plot_count,
        args.figure_dpi,
    )

    plot_boundary_summary(
        recommendations,
        out_boundary_figure,
        args.figure_dpi,
    )

    total_predictions = (
        len(
            routes
        )
        * len(
            np_values
        )
        * len(
            cm_values
        )
    )

    summary = {
        "input_csv":
            str(
                input_path
            ),

        "n_route_packages":
            int(
                len(
                    routes
                )
            ),

        "grid": {
            "np_size_nm": [
                args.np_min,
                args.np_max,
                args.np_step,
            ],

            "C_over_M_raw": [
                args.cm_min,
                args.cm_max,
                args.cm_step,
            ],

            "points_per_route":
                int(
                    len(
                        np_values
                    )
                    * len(
                        cm_values
                    )
                ),

            "total_predictions":
                int(
                    total_predictions
                ),
        },

        "ranking": {
            "balanced":
                "maximum raw q50",

            "potential":
                "maximum raw q90",

            "q10_and_interval_width":
                "descriptive only",

            "representative_tie_break":
                "exact-maximum plateau center after secondary score",
        },

        "surface_audit": {
            "near_optimal_definition":
                f"score >= maximum - "
                f"{args.near_optimal_tol}",

            "exact_maximum_tolerance":
                EXACT_MAX_ATOL,

            "status_values": [
                "interior_grid_maximum",
                "edge_boundary_limited",
                "corner_boundary_limited",
            ],
        },

        "physical_consistency": {
            "searched": [
                "np_size_nm",
                "raw C_over_M",
            ],

            "fixed":
                "S_over_M from C4; "
                "p_C / p_H2 / p_C_over_p_H2 "
                "from observed P8 / P9 / P10",

            "recomputed": [
                "N1-N18",
                "C1-C12",
                "N/C missing masks",
            ],
        },

        "quantile_crossing_points_detected":
            int(
                crossing_count
            ),

        "outputs": {
            "recommendations":
                str(
                    out_best
                ),

            "top_points":
                str(
                    out_points
                ),

            "full_surface":
                str(
                    out_surface
                ),

            "q50_surface_figure":
                str(
                    out_q50_figure
                ),

            "q90_surface_figure":
                str(
                    out_q90_figure
                ),

            "boundary_summary_figure":
                str(
                    out_boundary_figure
                ),
        },
    }

    with out_summary.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n[RESULT SUMMARY]"
    )

    print(
        f"Route packages               : "
        f"{len(routes)}"
    )

    print(
        f"Total grid predictions       : "
        f"{total_predictions}"
    )

    print(
        f"Quantile crossings detected  : "
        f"{crossing_count}"
    )

    print(
        f"Recommended states           : "
        f"{out_best}"
    )

    print(
        f"Top grid points              : "
        f"{out_points}"
    )

    print(
        f"Full response surface        : "
        f"{out_surface}"
    )

    print(
        f"q50 response surfaces        : "
        f"{out_q50_figure}"
    )

    print(
        f"q90 response surfaces        : "
        f"{out_q90_figure}"
    )

    print(
        f"Boundary summary             : "
        f"{out_boundary_figure}"
    )

    print(
        f"Summary                      : "
        f"{out_summary}"
    )

    print(
        "\n[OK] "
        "Physical-state optimization completed."
    )

    print(
        "=" * 108
    )


if __name__ == "__main__":
    main()