# build_candidate_priors.py
# =============================================================================
# CNT 3.0 — build GLOBAL and growth-mode-resolved priors for L123
#
# Input:
#   data/11_flattened/ml_dataset_3p0_full60D.csv
#
# Outputs:
#   outputs/recommendation/candidate_priors/priors_by_growth_mode_3p0.csv
#   outputs/recommendation/candidate_priors/priors_global_3p0.csv
#
# Rules:
#   - use L1 + L2 rows;
#   - prefer raw columns and fill remaining gaps from N11 / C3 / C4;
#   - calculate q05, q10, q50, q90 and q95;
#   - build GLOBAL and A–E priors;
#   - when a mode has fewer than MIN_N valid values, use GLOBAL quantiles and
#     record the fallback;
#   - do not create or use log_C_NP_ratio;
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

OUTDIR = (
    PROJECT_ROOT
    / "outputs"
    / "recommendation"
    / "candidate_priors"
)

LEVELS = ["L1", "L2"]
EXPECTED_MODES = ["A", "B", "C", "D", "E"]
MIN_N = 30

QUANTILES = {
    "q05": 0.05,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q95": 0.95,
}

MODE_COLUMN_CANDIDATES = [
    "growth_mode",
    "growth_mode_code",
    "cvd_mode",
    "mode",
    "growth_type",
    "process_type",
]

RAW_CANDIDATES = {
    "np_size_nm": [
        "np_size_nm",
        "np_diameter_nm",
        "particle_size_nm",
        "nanoparticle_size_nm",
    ],
    "C_over_M": [
        "C_over_M",
        "C_M_ratio",
        "carbon_to_metal_ratio",
    ],
    "S_over_M": [
        "S_over_M",
        "S_M_ratio",
        "sulfur_to_metal_ratio",
    ],
}

LATENT_FALLBACK = {
    "np_size_nm": "N11",
    "C_over_M": "C3",
    "S_over_M": "C4",
}


# =============================================================================
# HELPERS
# =============================================================================

def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in names:
        found = lookup.get(
            name.strip().lower()
        )

        if found is not None:
            return found

    return None


def normalize_growth_mode(
    value: Any,
) -> str | None:

    if value is None or pd.isna(value):
        return None

    token = str(value).strip().upper()

    if not token or token == "NAN":
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

    if token in EXPECTED_MODES:
        return token

    for mode in EXPECTED_MODES:
        prefixes = (
            f"{mode} ",
            f"{mode}_",
            f"{mode}-",
            f"{mode}:",
            f"{mode}(",
            f"{mode}（",
        )

        if token.startswith(prefixes):
            return mode

    return token


def clean_series(
    series: pd.Series,
    key: str,
) -> pd.Series:

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if key == "np_size_nm":
        values = values.where(
            values > 0
        )
    else:
        values = values.where(
            values >= 0
        )

    return values


def build_value_series(
    df: pd.DataFrame,
    key: str,
) -> tuple[
    pd.Series,
    dict[str, Any],
]:

    raw_col = find_column(
        df,
        RAW_CANDIDATES[key],
    )

    latent_col = find_column(
        df,
        [LATENT_FALLBACK[key]],
    )

    raw = (
        clean_series(
            df[raw_col],
            key,
        )
        if raw_col is not None
        else pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )
    )

    latent = (
        clean_series(
            df[latent_col],
            key,
        )
        if latent_col is not None
        else pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )
    )

    combined = raw.copy()

    fill_mask = (
        combined.isna()
        & latent.notna()
    )

    combined.loc[
        fill_mask
    ] = latent.loc[
        fill_mask
    ]

    if raw_col and latent_col:
        strategy = (
            "raw_then_latent_fill"
        )

        source_col = (
            f"{raw_col}|{latent_col}"
        )

    elif raw_col:
        strategy = "raw_only"
        source_col = raw_col

    elif latent_col:
        strategy = "latent_only"
        source_col = latent_col

    else:
        strategy = "missing"
        source_col = ""

    metadata = {
        "source_col":
            source_col,

        "source_strategy":
            strategy,

        "raw_col":
            raw_col or "",

        "latent_col":
            latent_col or "",

        "raw_n":
            int(
                raw.notna().sum()
            ),

        "latent_fill_n":
            int(
                fill_mask.sum()
            ),

        "combined_n":
            int(
                combined.notna().sum()
            ),
    }

    return (
        combined,
        metadata,
    )


def quantile_stats(
    series: pd.Series,
) -> dict[str, Any]:

    clean = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    result: dict[
        str,
        Any,
    ] = {
        "n":
            int(
                len(clean)
            )
    }

    for label, q in QUANTILES.items():

        result[label] = (
            float(
                clean.quantile(q)
            )
            if len(clean)
            else np.nan
        )

    return result


def validate_quantiles(
    row: dict[str, Any],
    key: str,
) -> None:

    values = [
        row.get(
            f"{key}_{label}",
            np.nan,
        )
        for label in QUANTILES
    ]

    values = [
        float(value)
        for value in values
        if (
            pd.notna(value)
            and np.isfinite(
                float(value)
            )
        )
    ]

    if any(
        right < left
        for left, right
        in zip(
            values,
            values[1:],
        )
    ):
        raise RuntimeError(
            f"Invalid quantile order for "
            f"{key}: {values}"
        )


# =============================================================================
# CORE
# =============================================================================

def build_priors(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    data = df.copy()

    # -------------------------------------------------------------------------
    # L1 + L2 only
    # -------------------------------------------------------------------------

    level_col = find_column(
        data,
        ["data_level"],
    )

    if level_col is not None:

        allowed = {
            level.upper()
            for level in LEVELS
        }

        data = data[
            data[level_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(
                allowed
            )
        ].copy()

    # -------------------------------------------------------------------------
    # growth mode
    # -------------------------------------------------------------------------

    mode_col = find_column(
        data,
        MODE_COLUMN_CANDIDATES,
    )

    if mode_col is not None:

        data[
            "_growth_mode"
        ] = data[
            mode_col
        ].map(
            normalize_growth_mode
        )

    else:

        data[
            "_growth_mode"
        ] = None

    # -------------------------------------------------------------------------
    # build prior variables
    # -------------------------------------------------------------------------

    source_meta: dict[
        str,
        dict[str, Any],
    ] = {}

    global_stats: dict[
        str,
        dict[str, Any],
    ] = {}

    for key in RAW_CANDIDATES:

        values, metadata = (
            build_value_series(
                data,
                key,
            )
        )

        data[
            f"_prior_{key}"
        ] = values

        source_meta[
            key
        ] = metadata

        global_stats[
            key
        ] = quantile_stats(
            values
        )

    # -------------------------------------------------------------------------
    # A-E + any observed extra label
    # -------------------------------------------------------------------------

    observed_modes = sorted(
        {
            str(mode)

            for mode
            in data[
                "_growth_mode"
            ].dropna()

            if str(mode).strip()
        }
    )

    modes = list(
        dict.fromkeys(
            EXPECTED_MODES
            + observed_modes
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    # -------------------------------------------------------------------------
    # GLOBAL + growth-mode-specific priors
    # -------------------------------------------------------------------------

    for scope in [
        "GLOBAL",
        *modes,
    ]:

        mask = (
            pd.Series(
                True,
                index=data.index,
            )

            if scope == "GLOBAL"

            else data[
                "_growth_mode"
            ].eq(
                scope
            )
        )

        row: dict[
            str,
            Any,
        ] = {
            "scope":
                scope,

            "growth_mode":
                scope,

            "levels_used":
                ",".join(
                    LEVELS
                ),

            "rows_used":
                int(
                    mask.sum()
                ),

            "mode_column":
                mode_col or "",

            "min_n":
                MIN_N,
        }

        any_fallback = False

        for key in RAW_CANDIDATES:

            local = quantile_stats(
                data.loc[
                    mask,
                    f"_prior_{key}",
                ]
            )

            use_global = (
                scope != "GLOBAL"
                and local["n"] < MIN_N
            )

            chosen = (
                global_stats[key]
                if use_global
                else local
            )

            meta = source_meta[
                key
            ]

            row[
                f"{key}_source_col"
            ] = meta[
                "source_col"
            ]

            row[
                f"{key}_source_strategy"
            ] = meta[
                "source_strategy"
            ]

            row[
                f"{key}_raw_col"
            ] = meta[
                "raw_col"
            ]

            row[
                f"{key}_latent_col"
            ] = meta[
                "latent_col"
            ]

            row[
                f"{key}_raw_n"
            ] = meta[
                "raw_n"
            ]

            row[
                f"{key}_latent_fill_n"
            ] = meta[
                "latent_fill_n"
            ]

            row[
                f"{key}_local_n"
            ] = int(
                local["n"]
            )

            row[
                f"{key}_n"
            ] = int(
                chosen["n"]
            )

            row[
                f"{key}_fallback_to_global"
            ] = bool(
                use_global
            )

            for label in QUANTILES:

                row[
                    f"{key}_{label}"
                ] = chosen[
                    label
                ]

            any_fallback = (
                any_fallback
                or use_global
            )

        row[
            "fallback_to_global"
        ] = bool(
            any_fallback
        )

        for key in RAW_CANDIDATES:

            validate_quantiles(
                row,
                key,
            )

        rows.append(
            row
        )

    by_mode = pd.DataFrame(
        rows
    )

    global_only = (
        by_mode.loc[
            by_mode[
                "scope"
            ].eq(
                "GLOBAL"
            )
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        by_mode,
        global_only,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 110
    )

    print(
        "[L123 PRIORS BUILD START]"
    )

    print(
        f"Project directory : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Input dataset     : "
        f"{CSV_PATH}"
    )

    print(
        f"Output directory  : "
        f"{OUTDIR}"
    )

    print(
        f"Levels            : "
        f"{LEVELS}"
    )

    print(
        f"Minimum local n   : "
        f"{MIN_N}"
    )

    print(
        "=" * 110
    )

    if not CSV_PATH.is_file():

        raise FileNotFoundError(
            "Dataset not found:\n"
            f"  {CSV_PATH}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    (
        by_mode,
        global_only,
    ) = build_priors(
        df
    )

    by_mode_path = (
        OUTDIR
        / "priors_by_growth_mode_3p0.csv"
    )

    global_path = (
        OUTDIR
        / "priors_global_3p0.csv"
    )

    by_mode.to_csv(
        by_mode_path,
        index=False,
        encoding="utf-8-sig",
    )

    global_only.to_csv(
        global_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_cols = [
        "scope",
        "rows_used",
    ]

    for key in RAW_CANDIDATES:

        summary_cols += [
            f"{key}_local_n",
            f"{key}_q05",
            f"{key}_q50",
            f"{key}_q95",
            f"{key}_fallback_to_global",
        ]

    print(
        "\n[PRIORS SUMMARY]"
    )

    print(
        by_mode[
            summary_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\n[OUTPUTS]"
    )

    print(
        f"Mode-resolved priors : "
        f"{by_mode_path}"
    )

    print(
        f"Global priors        : "
        f"{global_path}"
    )

    print(
        "\n[OK] "
        "L123 priors build completed."
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()