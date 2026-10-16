# build_zc.py
# =============================================================================
# CNT 3.0 — deterministic z_c calculator
#
# Project structure:
#   <project_root>/recommendation/feature_builders/build_zc.py
#
# Shared resources:
#   <project_root>/promptsV2/elements_properties.txt
#   <project_root>/promptsV2/A_prompt_zC.txt
#
# Upstream physical-state module:
#   <project_root>/recommendation/feature_builders/build_znuc.py
#
# Input:
#   z_nuc + route temperature + C/M + S/M + route p_C/p_H2 state
#   + optional NP-size override
#
# Output:
#   C1–C12 + miss_C1–miss_C12
#
# Main rules:
#   1. Explicit candidate values have the highest priority.
#   2. Priors are used only to fill missing values.
#   3. C3 is raw C/M, not log_C_NP_ratio.
#   4. C5/C6/C7 are route carbon/hydrogen partial-pressure descriptors.
#   5. C8/C9 come from the carbide-energy state in z_nuc.
#   6. C10 and C11 are log1p-transformed model features, matching flattening.
#   7. When NP size or route temperature changes, z_nuc is rebuilt through
#      the single upstream compute_znuc() implementation so that N8/N9/N11/
#      N12/N13/N16 and the dependent C features remain physically consistent.
#   8. This file reads only resources from the current project.
# =============================================================================

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROMPTS_DIR = PROJECT_ROOT / "promptsV2"

ELEMENTS_PROPERTIES_PATH = PROMPTS_DIR / "elements_properties.txt"
ZC_PROMPT_PATH = PROMPTS_DIR / "A_prompt_zC.txt"

ZNUC_MODULE_PATH = SCRIPT_DIR / "build_znuc.py"

PRIORS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "recommendation"
    / "candidate_priors"
)

PRIORS_BY_MODE_PATH = (
    PRIORS_DIR
    / "priors_by_growth_mode_3p0.csv"
)

PRIORS_GLOBAL_PATH = (
    PRIORS_DIR
    / "priors_global_3p0.csv"
)

# q10 / q50 / q90
PRIOR_Q = "q50"

# Direct-run self-test does not require a priors CSV because it supplies
# explicit C/M and S/M values.
RUN_SELF_TEST = True

SELF_TEST_RECIPE = [
    ("Fe", 0.60),
    ("Co", 0.30),
    ("Mo", 0.10),
]

SELF_TEST_T_K = 1173.15
SELF_TEST_NP_DIAMETER_NM = 10.0
SELF_TEST_C_OVER_M = 8.0
SELF_TEST_S_OVER_M = 0.20
SELF_TEST_P_C = 20000.0
SELF_TEST_P_H2 = 40000.0
SELF_TEST_P_C_OVER_P_H2 = 0.50


# =============================================================================
# MODULE CACHE
# =============================================================================

_ZNUC_MODULE_CACHE: ModuleType | None = None
_ELEMENTS_DF_CACHE: pd.DataFrame | None = None


# =============================================================================
# BASIC HELPERS
# =============================================================================

def _is_missing(
    value: Any,
) -> bool:

    if value is None:
        return True

    try:
        return bool(
            pd.isna(
                value
            )
        )

    except Exception:
        return False


def _to_float(
    value: Any,
) -> float:

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return np.nan

    return (
        number

        if np.isfinite(
            number
        )

        else np.nan
    )


def _first_finite(
    values: list[
        tuple[
            str,
            Any,
        ]
    ],
) -> tuple[
    float,
    str,
]:
    """
    Return the first finite numeric value
    and its source label.
    """

    for (
        source,
        value,
    ) in values:

        number = _to_float(
            value
        )

        if np.isfinite(
            number
        ):
            return (
                number,
                source,
            )

    return (
        np.nan,
        "missing",
    )


def _normalize_growth_mode(
    value: Any,
) -> str | None:

    if _is_missing(
        value
    ):
        return None

    token = str(
        value
    ).strip()

    if not token:
        return None

    return token.upper()


def _ordered_quantile_name(
    q: str,
) -> str:

    q = (
        str(q)
        .strip()
        .lower()
    )

    if q not in {
        "q10",
        "q50",
        "q90",
    }:

        raise ValueError(
            f"Invalid prior quantile "
            f"{q!r}; use q10, q50, or q90."
        )

    return q


# =============================================================================
# LOAD THE UNIQUE Z_NUC MODULE
# =============================================================================

def load_znuc_module() -> ModuleType:
    """
    Dynamically load the upstream z_nuc file
    from the same feature-builders directory.
    """

    global _ZNUC_MODULE_CACHE

    if (
        _ZNUC_MODULE_CACHE
        is not None
    ):
        return (
            _ZNUC_MODULE_CACHE
        )

    if not ZNUC_MODULE_PATH.is_file():

        raise FileNotFoundError(
            "The upstream z_nuc module "
            "was not found:\n"
            f"  {ZNUC_MODULE_PATH}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "cnt_znuc_builder",
            ZNUC_MODULE_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            "Could not load z_nuc module: "
            f"{ZNUC_MODULE_PATH}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    required_functions = {
        "compute_znuc",
        "load_elements_properties",
    }

    missing_functions = [
        name

        for name
        in required_functions

        if not hasattr(
            module,
            name,
        )
    ]

    if missing_functions:

        raise RuntimeError(
            "The z_nuc module is missing "
            "required functions:\n  "
            + "\n  ".join(
                missing_functions
            )
        )

    _ZNUC_MODULE_CACHE = module

    return module


def load_elements_properties_once() -> pd.DataFrame:

    global _ELEMENTS_DF_CACHE

    if (
        _ELEMENTS_DF_CACHE
        is None
    ):

        znuc_module = (
            load_znuc_module()
        )

        _ELEMENTS_DF_CACHE = (
            znuc_module
            .load_elements_properties(
                ELEMENTS_PROPERTIES_PATH
            )
        )

    return _ELEMENTS_DF_CACHE


# =============================================================================
# PRIORS
# =============================================================================

def _select_prior_row(
    df: pd.DataFrame,
    growth_mode: str | None,
) -> tuple[
    pd.Series,
    str,
    bool,
]:
    """
    Select a mode-specific prior when available.

    Returns:
        selected row
        selected scope
        whether a fallback was used
    """

    mode = (
        _normalize_growth_mode(
            growth_mode
        )
    )

    scope_column = None

    for candidate in (
        "growth_mode",
        "scope",
        "mode",
    ):

        if candidate in df.columns:

            scope_column = candidate

            break

    if scope_column is None:

        return (
            df.iloc[0],
            "GLOBAL",
            mode is not None,
        )

    normalized_scope = (
        df[
            scope_column
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if mode is not None:

        mode_rows = df.loc[
            normalized_scope
            == mode
        ]

        if not mode_rows.empty:

            return (
                mode_rows.iloc[0],
                mode,
                False,
            )

    global_rows = df.loc[
        normalized_scope.isin(
            {
                "GLOBAL",
                "ALL",
                "OVERALL",
            }
        )
    ]

    if not global_rows.empty:

        return (
            global_rows.iloc[0],
            "GLOBAL",
            mode is not None,
        )

    return (
        df.iloc[0],
        str(
            df.iloc[0][
                scope_column
            ]
        ),
        True,
    )


def load_priors(
    path: str | Path | None = None,
    q: str = PRIOR_Q,
    growth_mode: str | None = None,
) -> dict[
    str,
    Any,
]:
    """
    Load NP-size, C/M, and S/M priors.

    Supported files:
    1. outputs/recommendation/candidate_priors/
       priors_by_growth_mode_3p0.csv
    2. outputs/recommendation/candidate_priors/
       priors_global_3p0.csv

    The mode-resolved schema and the global
    schema are both supported.
    Only paths inside the current project are used.
    """

    q = (
        _ordered_quantile_name(
            q
        )
    )

    if path is not None:

        resolved_path = (
            Path(path)
            .expanduser()
            .resolve()
        )

    elif PRIORS_BY_MODE_PATH.is_file():

        resolved_path = (
            PRIORS_BY_MODE_PATH
            .resolve()
        )

    else:

        resolved_path = (
            PRIORS_GLOBAL_PATH
            .resolve()
        )

    if not resolved_path.is_file():

        raise FileNotFoundError(
            "No priors file was found.\n"
            "Checked:\n"
            f"  {PRIORS_BY_MODE_PATH}\n"
            f"  {PRIORS_GLOBAL_PATH}"
        )

    df = pd.read_csv(
        resolved_path,
        low_memory=False,
    )

    if df.empty:

        raise RuntimeError(
            "The priors file is empty: "
            f"{resolved_path}"
        )

    (
        row,
        selected_scope,
        fallback_used,
    ) = _select_prior_row(
        df,
        growth_mode,
    )

    values = {
        "np_size_nm":
            _to_float(
                row.get(
                    f"np_size_nm_{q}",
                    row.get(
                        "np_size_nm",
                        np.nan,
                    ),
                )
            ),

        "C_over_M":
            _to_float(
                row.get(
                    f"C_over_M_{q}",
                    row.get(
                        "C_over_M",
                        np.nan,
                    ),
                )
            ),

        "S_over_M":
            _to_float(
                row.get(
                    f"S_over_M_{q}",
                    row.get(
                        "S_over_M",
                        np.nan,
                    ),
                )
            ),

        "prior_q":
            q,

        "prior_scope":
            selected_scope,

        "prior_file":
            str(
                resolved_path
            ),

        "prior_fallback_used":
            bool(
                fallback_used
            ),
    }

    return values


# =============================================================================
# Z_NUC RECONSTRUCTION
# =============================================================================

def _recipe_from_znuc_metadata(
    znuc: Mapping[
        str,
        Any,
    ],
) -> list[
    tuple[
        Any,
        Any,
    ]
]:
    """
    Recover the catalyst recipe from metadata
    produced by the z_nuc module.

    Long-name aliases are also accepted.
    """

    recipe: list[
        tuple[
            Any,
            Any,
        ]
    ] = []

    for index in (
        1,
        2,
        3,
    ):

        metal = znuc.get(
            f"metal{index}",
            znuc.get(
                f"N{index}_element",
                None,
            ),
        )

        fraction = znuc.get(
            f"frac{index}",
            znuc.get(
                f"N{index}_atomic_fraction",
                np.nan,
            ),
        )

        if (
            not _is_missing(
                metal
            )
            and np.isfinite(
                _to_float(
                    fraction
                )
            )
            and _to_float(
                fraction
            ) > 0
        ):

            recipe.append(
                (
                    metal,
                    fraction,
                )
            )

    return recipe


def _znuc_value(
    znuc: Mapping[
        str,
        Any,
    ],
    short_name: str,
    long_name: str,
) -> float:

    return _to_float(
        znuc.get(
            short_name,
            znuc.get(
                long_name,
                np.nan,
            ),
        )
    )


def rebuild_znuc_for_state(
    znuc: Mapping[
        str,
        Any,
    ],
    T_K: float,
    np_size_nm: float,
    elements_df: pd.DataFrame | None = None,
) -> tuple[
    dict[
        str,
        Any,
    ],
    bool,
]:
    """
    Rebuild the complete z_nuc state through
    the unique upstream implementation.

    When recipe metadata is available, rebuilding
    is always performed so that route temperature
    and NP-size overrides propagate consistently.

    When recipe metadata is unavailable:
    - unchanged NP size is accepted;
    - changing NP size is rejected because it
      would make N11/N12/N13 inconsistent.
    """

    recipe = (
        _recipe_from_znuc_metadata(
            znuc
        )
    )

    if recipe:

        znuc_module = (
            load_znuc_module()
        )

        if elements_df is None:

            elements_df = (
                load_elements_properties_once()
            )

        rebuilt = (
            znuc_module
            .compute_znuc(
                recipe=recipe,
                T_value=T_K,
                T_unit="K",
                np_diameter_nm=np_size_nm,
                elements_df=elements_df,
            )
        )

        return (
            rebuilt,
            True,
        )

    current_np_size = (
        _znuc_value(
            znuc,
            "N11",
            "N11_np_diameter_nm",
        )
    )

    if (
        np.isfinite(
            np_size_nm
        )
        and np.isfinite(
            current_np_size
        )
        and not np.isclose(
            np_size_nm,
            current_np_size,
            rtol=1e-9,
            atol=1e-12,
        )
    ):

        raise ValueError(
            "NP-size override requested, "
            "but catalyst recipe metadata "
            "is missing. The complete z_nuc "
            "state cannot be rebuilt safely."
        )

    return (
        dict(
            znuc
        ),
        False,
    )


# =============================================================================
# CORE Z_C CALCULATION
# =============================================================================

def compute_zc(
    znuc: Mapping[
        str,
        Any,
    ],
    T_K: Any,
    priors: Mapping[
        str,
        Any,
    ] | None = None,
    C_over_M: Any = np.nan,
    S_over_M: Any = np.nan,
    np_size_nm: Any = np.nan,
    p_C: Any = np.nan,
    p_H2: Any = np.nan,
    p_C_over_p_H2: Any = np.nan,
    growth_mode: str | None = None,
    prior_q: str = PRIOR_Q,
    elements_df: pd.DataFrame | None = None,
) -> dict[
    str,
    Any,
]:
    """
    Calculate C1–C12 and miss_C1–miss_C12.

    Value priority
    --------------
    NP size:
        explicit np_size_nm
        -> z_nuc N11
        -> prior np_size_nm

    C/M:
        explicit C_over_M
        -> z_nuc metadata C_over_M/C3
        -> prior C_over_M

    S/M:
        explicit S_over_M
        -> z_nuc metadata S_over_M/C4
        -> prior S_over_M

    Route gas-state descriptors:
        p_C, p_H2 and p_C_over_p_H2 are supplied by
        the selected route/template. They are not
        filled from catalyst priors.

    Priors are loaded only when at least
    one required value remains missing.
    """

    T_K_value = (
        _to_float(
            T_K
        )
    )

    existing_np_size = (
        _znuc_value(
            znuc,
            "N11",
            "N11_np_diameter_nm",
        )
    )

    existing_C_over_M = (
        _to_float(
            znuc.get(
                "C_over_M",
                znuc.get(
                    "C3",
                    znuc.get(
                        "C3_C_over_M",
                        np.nan,
                    ),
                ),
            )
        )
    )

    existing_S_over_M = (
        _to_float(
            znuc.get(
                "S_over_M",
                znuc.get(
                    "C4",
                    znuc.get(
                        "C4_S_over_M",
                        np.nan,
                    ),
                ),
            )
        )
    )

    explicit_np = (
        _to_float(
            np_size_nm
        )
    )

    explicit_C_M = (
        _to_float(
            C_over_M
        )
    )

    explicit_S_M = (
        _to_float(
            S_over_M
        )
    )

    explicit_p_C = (
        _to_float(
            p_C
        )
    )

    explicit_p_H2 = (
        _to_float(
            p_H2
        )
    )

    explicit_p_C_over_p_H2 = (
        _to_float(
            p_C_over_p_H2
        )
    )

    need_priors = (
        (
            not np.isfinite(
                explicit_np
            )
            and not np.isfinite(
                existing_np_size
            )
        )
        or (
            not np.isfinite(
                explicit_C_M
            )
            and not np.isfinite(
                existing_C_over_M
            )
        )
        or (
            not np.isfinite(
                explicit_S_M
            )
            and not np.isfinite(
                existing_S_over_M
            )
        )
    )

    prior_values: Mapping[
        str,
        Any,
    ]

    if priors is not None:

        prior_values = priors

    elif need_priors:

        prior_values = (
            load_priors(
                q=prior_q,
                growth_mode=growth_mode,
            )
        )

    else:

        prior_values = {}

    (
        selected_np_size,
        np_size_source,
    ) = _first_finite(
        [
            (
                "explicit",
                explicit_np,
            ),
            (
                "znuc",
                existing_np_size,
            ),
            (
                "prior",
                prior_values.get(
                    "np_size_nm",
                    np.nan,
                ),
            ),
        ]
    )

    (
        selected_C_M,
        C_M_source,
    ) = _first_finite(
        [
            (
                "explicit",
                explicit_C_M,
            ),
            (
                "znuc",
                existing_C_over_M,
            ),
            (
                "prior",
                prior_values.get(
                    "C_over_M",
                    np.nan,
                ),
            ),
        ]
    )

    (
        selected_S_M,
        S_M_source,
    ) = _first_finite(
        [
            (
                "explicit",
                explicit_S_M,
            ),
            (
                "znuc",
                existing_S_over_M,
            ),
            (
                "prior",
                prior_values.get(
                    "S_over_M",
                    np.nan,
                ),
            ),
        ]
    )

    (
        consistent_znuc,
        znuc_rebuilt,
    ) = rebuild_znuc_for_state(
        znuc=znuc,
        T_K=T_K_value,
        np_size_nm=selected_np_size,
        elements_df=elements_df,
    )

    N8 = (
        _znuc_value(
            consistent_znuc,
            "N8",
            "N8_weighted_melting_point_K",
        )
    )

    N11 = (
        _znuc_value(
            consistent_znuc,
            "N11",
            "N11_np_diameter_nm",
        )
    )

    N12_raw = (
        _to_float(
            consistent_znuc.get(
                "N12_raw_atoms",
                np.nan,
            )
        )
    )

    N13 = (
        _znuc_value(
            consistent_znuc,
            "N13",
            "N13_surface_atom_fraction_proxy",
        )
    )

    N16 = (
        _znuc_value(
            consistent_znuc,
            "N16",
            "N16_weighted_carbide_formation_energy_eV_per_C",
        )
    )

    # C1: route temperature in K
    C1 = T_K_value

    # C2: T / effective melting temperature
    C2 = (
        C1 / N8

        if (
            np.isfinite(
                C1
            )
            and np.isfinite(
                N8
            )
            and N8 != 0
        )

        else np.nan
    )

    # C3: raw C/M, not a logarithmic variable
    C3 = selected_C_M

    # C4: raw S/M
    C4 = selected_S_M

    # C5: carbon partial pressure p_C (Pa)
    C5 = explicit_p_C

    # C6: hydrogen partial pressure p_H2 (Pa)
    C6 = explicit_p_H2

    # C7: p_C / p_H2
    C7 = (
        explicit_p_C_over_p_H2

        if np.isfinite(
            explicit_p_C_over_p_H2
        )

        else (
            C5 / C6

            if (
                np.isfinite(
                    C5
                )
                and np.isfinite(
                    C6
                )
                and C6 != 0
            )

            else np.nan
        )
    )

    # C8: weighted carbide formation energy from z_nuc
    C8 = N16

    # C9: carbide affinity × T/Tm_eff
    C9 = (
        C8 * C2

        if (
            np.isfinite(
                C8
            )
            and np.isfinite(
                C2
            )
        )

        else np.nan
    )

    # C10 raw physical definition: NP total atoms N_np.
    # Model-space C10 matches 11_flatten_dataset.py: log1p(raw C10).
    C10_raw = N12_raw

    C10 = (
        math.log1p(
            C10_raw
        )

        if (
            np.isfinite(
                C10_raw
            )
            and C10_raw >= 0
        )

        else np.nan
    )

    # C11 raw physical definition: C/NP = (C/M) × N_np.
    # Model-space C11 matches 11_flatten_dataset.py: log1p(raw C11).
    C11_raw = (
        C3 * N12_raw

        if (
            np.isfinite(
                C3
            )
            and np.isfinite(
                N12_raw
            )
        )

        else np.nan
    )

    C11 = (
        math.log1p(
            C11_raw
        )

        if (
            np.isfinite(
                C11_raw
            )
            and C11_raw >= 0
        )

        else np.nan
    )

    # C12: surface carbon loading ≈ (C/NP) × N_np^(-1/3)
    C12 = (
        C11_raw
        * (
            N12_raw
            ** (-1.0 / 3.0)
        )

        if (
            np.isfinite(
                C11_raw
            )
            and np.isfinite(
                N12_raw
            )
            and N12_raw > 0
        )

        else np.nan
    )

    output: dict[
        str,
        Any,
    ] = {
        "C1":
            C1,

        "C2":
            C2,

        "C3":
            C3,

        "C4":
            C4,

        "C5":
            C5,

        "C6":
            C6,

        "C7":
            C7,

        "C8":
            C8,

        "C9":
            C9,

        "C10":
            C10,

        "C11":
            C11,

        "C12":
            C12,
    }

    for index in range(
        1,
        13,
    ):

        feature_name = (
            f"C{index}"
        )

        output[
            f"miss_{feature_name}"
        ] = int(
            not np.isfinite(
                _to_float(
                    output[
                        feature_name
                    ]
                )
            )
        )

    # Debug and provenance fields
    output.update(
        {
            "C10_raw_np_total_atoms":
                C10_raw,

            "C11_raw_C_atoms_per_NP":
                C11_raw,

            "C12_raw_surface_carbon_loading":
                C12,

            "np_size_source":
                np_size_source,

            "C_over_M_source":
                C_M_source,

            "S_over_M_source":
                S_M_source,

            "growth_mode":
                _normalize_growth_mode(
                    growth_mode
                ),

            "prior_q":
                prior_values.get(
                    "prior_q",
                    _ordered_quantile_name(
                        prior_q
                    ),
                ),

            "prior_scope":
                prior_values.get(
                    "prior_scope",
                    None,
                ),

            "prior_file":
                prior_values.get(
                    "prior_file",
                    None,
                ),

            "prior_fallback_used":
                prior_values.get(
                    "prior_fallback_used",
                    False,
                ),

            "znuc_rebuilt":
                bool(
                    znuc_rebuilt
                ),
        }
    )

    # Compatibility aliases for existing downstream code.
    output.update(
        {
            "C1_temperature_K":
                C1,

            "C2_T_over_Tm_eff":
                C2,

            "C3_C_over_M":
                C3,

            "C4_S_over_M":
                C4,

            "C5_carbon_partial_pressure_pC_Pa":
                C5,

            "C6_hydrogen_partial_pressure_pH2_Pa":
                C6,

            "C7_pC_over_pH2":
                C7,

            "C8_carbide_affinity_eV_per_C":
                C8,

            "C9_carbide_affinity_times_T_over_Tm":
                C9,

            "C10_log1p_np_total_atoms":
                C10,

            "C11_log1p_C_atoms_per_NP":
                C11,

            "C12_surface_carbon_loading_proxy":
                C12,
        }
    )

    # Include the consistent upstream N state used to create C2/C8–C12.
    # This allows L123 to assemble one physically consistent candidate row.
    for (
        key,
        value,
    ) in consistent_znuc.items():

        if (
            key.startswith(
                "N"
            )
            or key.startswith(
                "miss_N"
            )
            or key in {
                "metal1",
                "metal2",
                "metal3",
                "frac1",
                "frac2",
                "frac3",
                "temperature_K",
                "np_size_nm",
                "N12_raw_atoms",
                "recipe_key",
            }
        ):

            output[
                f"znuc_{key}"
            ] = value

    return output


def compute_zc_row(
    znuc_row: Mapping[
        str,
        Any,
    ],
    T_K: Any,
    priors: Mapping[
        str,
        Any,
    ] | None = None,
    C_over_M: Any = np.nan,
    S_over_M: Any = np.nan,
    np_size_nm: Any = np.nan,
    p_C: Any = np.nan,
    p_H2: Any = np.nan,
    p_C_over_p_H2: Any = np.nan,
    growth_mode: str | None = None,
    prior_q: str = PRIOR_Q,
    elements_df: pd.DataFrame | None = None,
) -> dict[
    str,
    Any,
]:
    """
    L123-compatible wrapper.
    """

    return compute_zc(
        znuc=znuc_row,
        T_K=T_K,
        priors=priors,
        C_over_M=C_over_M,
        S_over_M=S_over_M,
        np_size_nm=np_size_nm,
        p_C=p_C,
        p_H2=p_H2,
        p_C_over_p_H2=p_C_over_p_H2,
        growth_mode=growth_mode,
        prior_q=prior_q,
        elements_df=elements_df,
    )


# =============================================================================
# DIRECT SELF-TEST
# =============================================================================

def run_self_test() -> None:

    print(
        "=" * 108
    )

    print(
        "[Z_C SELF-TEST START]"
    )

    print(
        f"Project directory   : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Prompts directory   : "
        f"{PROMPTS_DIR}"
    )

    print(
        f"Element properties  : "
        f"{ELEMENTS_PROPERTIES_PATH}"
    )

    print(
        f"z_c prompt          : "
        f"{ZC_PROMPT_PATH}"
    )

    print(
        f"z_nuc module        : "
        f"{ZNUC_MODULE_PATH}"
    )

    print(
        "=" * 108
    )

    znuc_module = (
        load_znuc_module()
    )

    elements_df = (
        load_elements_properties_once()
    )

    znuc_10nm = (
        znuc_module
        .compute_znuc(
            recipe=SELF_TEST_RECIPE,
            T_value=SELF_TEST_T_K,
            T_unit="K",
            np_diameter_nm=SELF_TEST_NP_DIAMETER_NM,
            elements_df=elements_df,
        )
    )

    zc_10nm = compute_zc(
        znuc=znuc_10nm,
        T_K=SELF_TEST_T_K,
        C_over_M=SELF_TEST_C_OVER_M,
        S_over_M=SELF_TEST_S_OVER_M,
        np_size_nm=SELF_TEST_NP_DIAMETER_NM,
        p_C=SELF_TEST_P_C,
        p_H2=SELF_TEST_P_H2,
        p_C_over_p_H2=SELF_TEST_P_C_OVER_P_H2,
        elements_df=elements_df,
    )

    zc_5nm = compute_zc(
        znuc=znuc_10nm,
        T_K=SELF_TEST_T_K,
        C_over_M=SELF_TEST_C_OVER_M,
        S_over_M=SELF_TEST_S_OVER_M,
        np_size_nm=5.0,
        p_C=SELF_TEST_P_C,
        p_H2=SELF_TEST_P_H2,
        p_C_over_p_H2=SELF_TEST_P_C_OVER_P_H2,
        elements_df=elements_df,
    )

    print(
        "\n[SELF-TEST INPUT]"
    )

    print(
        "Recipe       : "
        "Fe 0.60 / Co 0.30 / Mo 0.10"
    )

    print(
        f"Temperature  : "
        f"{SELF_TEST_T_K} K"
    )

    print(
        f"C/M          : "
        f"{SELF_TEST_C_OVER_M}"
    )

    print(
        f"S/M          : "
        f"{SELF_TEST_S_OVER_M}"
    )

    print(
        f"p_C          : "
        f"{SELF_TEST_P_C} Pa"
    )

    print(
        f"p_H2         : "
        f"{SELF_TEST_P_H2} Pa"
    )

    display_order = (
        [
            f"C{index}"

            for index
            in range(
                1,
                13,
            )
        ]

        + [
            f"miss_C{index}"

            for index
            in range(
                1,
                13,
            )
        ]

        + [
            "C10_raw_np_total_atoms",
            "C11_raw_C_atoms_per_NP",
            "np_size_source",
            "C_over_M_source",
            "S_over_M_source",
            "znuc_rebuilt",
        ]
    )

    print(
        "\n[10 NM OUTPUT]"
    )

    print(
        pd.Series(
            {
                key:
                    zc_10nm.get(
                        key
                    )

                for key
                in display_order
            }
        ).to_string()
    )

    comparison = (
        pd.DataFrame(
            {
                "feature": [
                    "C5_carbon_partial_pressure",
                    "C6_hydrogen_partial_pressure",
                    "C7_pC_over_pH2",
                    "C8_carbide_affinity",
                    "C9_carbide_affinity_times_T_over_Tm",
                    "C10_log1p_np_total_atoms",
                    "C11_log1p_C_atoms_per_NP",
                    "C12_surface_carbon_loading",
                ],

                "10_nm": [
                    zc_10nm[
                        "C5"
                    ],
                    zc_10nm[
                        "C6"
                    ],
                    zc_10nm[
                        "C7"
                    ],
                    zc_10nm[
                        "C8"
                    ],
                    zc_10nm[
                        "C9"
                    ],
                    zc_10nm[
                        "C10"
                    ],
                    zc_10nm[
                        "C11"
                    ],
                    zc_10nm[
                        "C12"
                    ],
                ],

                "5_nm": [
                    zc_5nm[
                        "C5"
                    ],
                    zc_5nm[
                        "C6"
                    ],
                    zc_5nm[
                        "C7"
                    ],
                    zc_5nm[
                        "C8"
                    ],
                    zc_5nm[
                        "C9"
                    ],
                    zc_5nm[
                        "C10"
                    ],
                    zc_5nm[
                        "C11"
                    ],
                    zc_5nm[
                        "C12"
                    ],
                ],
            }
        )
    )

    print(
        "\n"
        "[NP-SIZE CONSISTENCY CHECK: "
        "10 NM -> 5 NM]"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    checks = {
        "C3_is_raw_C_over_M":
            np.isclose(
                zc_10nm[
                    "C3"
                ],
                SELF_TEST_C_OVER_M,
            ),

        "C4_is_raw_S_over_M":
            np.isclose(
                zc_10nm[
                    "C4"
                ],
                SELF_TEST_S_OVER_M,
            ),

        "C5_is_route_p_C":
            np.isclose(
                zc_10nm[
                    "C5"
                ],
                SELF_TEST_P_C,
            ),

        "C6_is_route_p_H2":
            np.isclose(
                zc_10nm[
                    "C6"
                ],
                SELF_TEST_P_H2,
            ),

        "C7_is_route_ratio":
            np.isclose(
                zc_10nm[
                    "C7"
                ],
                SELF_TEST_P_C_OVER_P_H2,
            ),

        "C5_unchanged_with_size":
            np.isclose(
                zc_10nm[
                    "C5"
                ],
                zc_5nm[
                    "C5"
                ],
            ),

        "C6_unchanged_with_size":
            np.isclose(
                zc_10nm[
                    "C6"
                ],
                zc_5nm[
                    "C6"
                ],
            ),

        "C7_unchanged_with_size":
            np.isclose(
                zc_10nm[
                    "C7"
                ],
                zc_5nm[
                    "C7"
                ],
            ),

        "C8_unchanged_with_size":
            np.isclose(
                zc_10nm[
                    "C8"
                ],
                zc_5nm[
                    "C8"
                ],
            ),

        "C9_unchanged_with_size":
            np.isclose(
                zc_10nm[
                    "C9"
                ],
                zc_5nm[
                    "C9"
                ],
            ),

        "C10_changes_with_size":
            not np.isclose(
                zc_10nm[
                    "C10"
                ],
                zc_5nm[
                    "C10"
                ],
            ),

        "C11_changes_with_size":
            not np.isclose(
                zc_10nm[
                    "C11"
                ],
                zc_5nm[
                    "C11"
                ],
            ),

        "C12_changes_with_size":
            not np.isclose(
                zc_10nm[
                    "C12"
                ],
                zc_5nm[
                    "C12"
                ],
            ),

        "all_C_features_available":
            all(
                zc_10nm[
                    f"miss_C{index}"
                ]
                == 0

                for index
                in range(
                    1,
                    13,
                )
            ),
    }

    print(
        "\n[SELF-TEST CHECKS]"
    )

    print(
        pd.Series(
            checks
        ).to_string()
    )

    failed_checks = [
        name

        for (
            name,
            passed,
        ) in checks.items()

        if not passed
    ]

    if failed_checks:

        raise RuntimeError(
            "z_c self-test failed:\n  "
            + "\n  ".join(
                failed_checks
            )
        )

    print(
        "\n[OK] "
        "z_c self-test completed."
    )

    print(
        "=" * 108
    )


if __name__ == "__main__":

    if RUN_SELF_TEST:

        run_self_test()

    else:

        print(
            "[OK] "
            "z_c module is ready."
        )