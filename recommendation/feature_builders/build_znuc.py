# build_znuc.py
# =============================================================================
# CNT 3.0 — deterministic z_nuc calculator
#
# Project structure:
#   <project_root>/recommendation/feature_builders/build_znuc.py
#   <project_root>/promptsV2/
#
# Input:
#   catalyst recipe + temperature + nanoparticle diameter
# Output:
#   N1–N18 + miss_N1–miss_N18
# =============================================================================

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROMPTS_DIR = PROJECT_ROOT / "promptsV2"

ELEMENTS_PROPERTIES_PATH = PROMPTS_DIR / "elements_properties.txt"
ZNUC_PROMPT_PATH = PROMPTS_DIR / "A_prompt_znuc.txt"

PACKING_FRACTION = 0.74

RUN_SELF_TEST = True
SELF_TEST_RECIPE = [("Fe", 0.60), ("Co", 0.30), ("Mo", 0.10)]
SELF_TEST_T_K = 1173.15
SELF_TEST_NP_DIAMETER_NM = 10.0


# =============================================================================
# HELPERS
# =============================================================================

def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan

    return number if np.isfinite(number) else np.nan


def canonical_element_symbol(value: Any) -> str | None:
    if _is_missing(value):
        return None

    token = str(value).strip()

    if not token or token.lower() == "nan":
        return None

    return (
        token.upper()
        if len(token) == 1
        else token[0].upper() + token[1:].lower()
    )


def _first_present(
    row: Mapping[str, Any] | pd.Series,
    candidates: list[str],
) -> Any:
    for name in candidates:
        if (
            name in row
            and not _is_missing(row[name])
        ):
            return row[name]

    return np.nan


# =============================================================================
# ELEMENT TABLE
# =============================================================================

def load_elements_properties(
    path: str | Path | None = None,
) -> pd.DataFrame:

    resolved_path = (
        Path(path)
        .expanduser()
        .resolve()

        if path is not None

        else ELEMENTS_PROPERTIES_PATH.resolve()
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "elements_properties.txt not found:\n"
            f"  {resolved_path}\n\n"
            "Expected location:\n"
            f"  {ELEMENTS_PROPERTIES_PATH}"
        )

    df = pd.read_csv(
        resolved_path,
        sep=None,
        engine="python",
        na_values=[
            "-",
            "—",
            "–",
            "NA",
            "N/A",
            "",
        ],
        keep_default_na=True,
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if "Element" not in df.columns:

        lower_map = {
            str(column).strip().lower():
                column

            for column in df.columns
        }

        for alias in (
            "element",
            "elem",
            "symbol",
        ):
            if alias in lower_map:

                df.rename(
                    columns={
                        lower_map[alias]:
                            "Element"
                    },
                    inplace=True,
                )

                break

    if "Element" not in df.columns:
        raise RuntimeError(
            "Element column not found "
            "in elements_properties.txt"
        )

    if "atomic_radius" not in df.columns:

        for column in df.columns:

            normalized = (
                str(column)
                .lower()
                .replace(" ", "")
            )

            if normalized in {
                "atomic_radius/å",
                "atomic_radius/angstrom",
                "atomic_radius(å)",
            }:

                df.rename(
                    columns={
                        column:
                            "atomic_radius"
                    },
                    inplace=True,
                )

                break

    required = [
        "atomic_radius",
        "electronegativity",
        "Tm",
        "density",
        "d_band_center",
        "work_function_eV",
        "surface_energy_Jm2",
        "carbide_form_energy_eV_per_C",
        "oxide_DG_kJ_per_molO2",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "elements_properties.txt "
            "is missing required columns:\n  "
            + "\n  ".join(missing)
        )

    df["Element"] = (
        df["Element"]
        .map(
            canonical_element_symbol
        )
    )

    df = df[
        df["Element"].notna()
    ].copy()

    for column in df.columns:

        if column != "Element":

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return (
        df
        .drop_duplicates(
            subset=["Element"],
            keep="first",
        )
        .set_index(
            "Element",
            drop=True,
        )
    )


# =============================================================================
# RECIPE
# =============================================================================

def normalize_fractions(
    recipe: Iterable[
        tuple[Any, Any]
    ],
) -> list[
    tuple[str, float]
]:

    merged: dict[
        str,
        float,
    ] = {}

    for (
        raw_element,
        raw_fraction,
    ) in recipe:

        element = (
            canonical_element_symbol(
                raw_element
            )
        )

        fraction = _to_float(
            raw_fraction
        )

        if (
            element is None
            or not np.isfinite(
                fraction
            )
            or fraction <= 0
        ):
            continue

        merged[element] = (
            merged.get(
                element,
                0.0,
            )
            + fraction
        )

    if not merged:
        return []

    total = float(
        sum(
            merged.values()
        )
    )

    if total <= 0:
        return []

    return [
        (
            element,
            value / total,
        )
        for element, value
        in merged.items()
    ]


def top3_metals(
    recipe_norm: list[
        tuple[str, float]
    ],
) -> list[
    tuple[str | None, float]
]:

    ordered = sorted(
        recipe_norm,
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return (
        ordered[:3]
        + [(None, 0.0)]
        * (
            3
            - len(
                ordered[:3]
            )
        )
    )


def recipe_from_row(
    row: Mapping[str, Any]
    | pd.Series,
) -> list[
    tuple[str, float]
]:

    metal_aliases = {
        1: [
            "metal1",
            "metal_1",
            "m1",
            "element1",
            "element_1",
        ],

        2: [
            "metal2",
            "metal_2",
            "m2",
            "element2",
            "element_2",
        ],

        3: [
            "metal3",
            "metal_3",
            "m3",
            "element3",
            "element_3",
        ],
    }

    fraction_aliases = {
        1: [
            "frac1",
            "frac_1",
            "pct1",
            "pct_1",
            "percent1",
            "percent_1",
        ],

        2: [
            "frac2",
            "frac_2",
            "pct2",
            "pct_2",
            "percent2",
            "percent_2",
        ],

        3: [
            "frac3",
            "frac_3",
            "pct3",
            "pct_3",
            "percent3",
            "percent_3",
        ],
    }

    recipe: list[
        tuple[Any, Any]
    ] = []

    for index in (
        1,
        2,
        3,
    ):

        metal = _first_present(
            row,
            metal_aliases[index],
        )

        fraction = _first_present(
            row,
            fraction_aliases[index],
        )

        if (
            not _is_missing(
                metal
            )
            and not _is_missing(
                fraction
            )
        ):

            recipe.append(
                (
                    metal,
                    fraction,
                )
            )

    return normalize_fractions(
        recipe
    )


# =============================================================================
# WEIGHTED PROPERTIES
# =============================================================================

def weighted_mean_strict(
    recipe_norm: list[
        tuple[str, float]
    ],
    series: pd.Series,
) -> float:

    values: list[
        tuple[
            float,
            float,
        ]
    ] = []

    for (
        element,
        fraction,
    ) in recipe_norm:

        value = _to_float(
            series.get(
                element,
                np.nan,
            )
        )

        if not np.isfinite(
            value
        ):
            return np.nan

        values.append(
            (
                fraction,
                value,
            )
        )

    return float(
        sum(
            fraction * value
            for fraction, value
            in values
        )
    )


def weighted_std_strict(
    recipe_norm: list[
        tuple[str, float]
    ],
    series: pd.Series,
) -> float:

    mean_value = (
        weighted_mean_strict(
            recipe_norm,
            series,
        )
    )

    if not np.isfinite(
        mean_value
    ):
        return np.nan

    variance = 0.0

    for (
        element,
        fraction,
    ) in recipe_norm:

        value = _to_float(
            series.get(
                element,
                np.nan,
            )
        )

        if not np.isfinite(
            value
        ):
            return np.nan

        variance += (
            fraction
            * (
                value
                - mean_value
            ) ** 2
        )

    return float(
        math.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


def weighted_mean_renorm(
    recipe_norm: list[
        tuple[str, float]
    ],
    series: pd.Series,
) -> tuple[
    float,
    list[
        tuple[
            float,
            float,
        ]
    ],
]:

    valid: list[
        tuple[
            float,
            float,
        ]
    ] = []

    for (
        element,
        fraction,
    ) in recipe_norm:

        value = _to_float(
            series.get(
                element,
                np.nan,
            )
        )

        if np.isfinite(
            value
        ):

            valid.append(
                (
                    fraction,
                    value,
                )
            )

    total = float(
        sum(
            fraction
            for fraction, _
            in valid
        )
    )

    if (
        not valid
        or total <= 0
    ):
        return (
            np.nan,
            [],
        )

    normalized = [
        (
            fraction / total,
            value,
        )
        for fraction, value
        in valid
    ]

    mean_value = float(
        sum(
            fraction * value

            for fraction, value
            in normalized
        )
    )

    return (
        mean_value,
        normalized,
    )


# =============================================================================
# PHYSICAL HELPERS
# =============================================================================

def to_kelvin(
    temperature_value: Any,
    temperature_unit: str = "K",
) -> float:

    temperature = _to_float(
        temperature_value
    )

    if not np.isfinite(
        temperature
    ):
        return np.nan

    unit = (
        str(
            temperature_unit
        )
        .strip()
        .lower()
    )

    if unit in {
        "k",
        "kelvin",
    }:
        return temperature

    if unit in {
        "c",
        "°c",
        "celsius",
        "degc",
    }:
        return (
            temperature
            + 273.15
        )

    raise ValueError(
        "temperature_unit must be "
        "'K' or 'C'"
    )


def compute_np_atom_count(
    np_diameter_nm: Any,
    weighted_atomic_radius_nm: Any,
) -> float:

    diameter_nm = _to_float(
        np_diameter_nm
    )

    radius_nm = _to_float(
        weighted_atomic_radius_nm
    )

    if (
        not np.isfinite(
            diameter_nm
        )
        or diameter_nm <= 0
        or not np.isfinite(
            radius_nm
        )
        or radius_nm <= 0
    ):
        return np.nan

    diameter_m = (
        diameter_nm
        * 1e-9
    )

    radius_m = (
        radius_nm
        * 1e-9
    )

    atom_volume = (
        (4.0 / 3.0)
        * math.pi
        * radius_m**3
        / PACKING_FRACTION
    )

    particle_volume = (
        (4.0 / 3.0)
        * math.pi
        * (
            diameter_m
            / 2.0
        ) ** 3
    )

    return (
        float(
            particle_volume
            / atom_volume
        )

        if atom_volume > 0

        else np.nan
    )


# =============================================================================
# CORE Z_NUC
# =============================================================================

def compute_znuc(
    recipe: Iterable[
        tuple[Any, Any]
    ],
    T_value: Any = np.nan,
    T_unit: str = "K",
    np_diameter_nm: Any = np.nan,
    elements_df: pd.DataFrame | None = None,
    elements_properties_path: str | Path | None = None,
) -> dict[
    str,
    Any,
]:

    if elements_df is None:

        elements_df = (
            load_elements_properties(
                elements_properties_path
            )
        )

    recipe_norm = (
        normalize_fractions(
            recipe
        )
    )

    if not recipe_norm:
        raise ValueError(
            "The catalyst recipe contains "
            "no valid positive fractions."
        )

    unknown = [
        element

        for element, _
        in recipe_norm

        if element
        not in elements_df.index
    ]

    if unknown:
        raise ValueError(
            "Elements absent from "
            "elements_properties.txt:\n  "
            + "\n  ".join(
                sorted(
                    set(
                        unknown
                    )
                )
            )
        )

    (
        (metal1, frac1),
        (metal2, frac2),
        (metal3, frac3),
    ) = top3_metals(
        recipe_norm
    )

    radius_A = (
        weighted_mean_strict(
            recipe_norm,
            elements_df[
                "atomic_radius"
            ],
        )
    )

    radius_std_A = (
        weighted_std_strict(
            recipe_norm,
            elements_df[
                "atomic_radius"
            ],
        )
    )

    N4 = (
        radius_A * 0.1

        if np.isfinite(
            radius_A
        )

        else np.nan
    )

    N5 = (
        radius_std_A * 0.1

        if np.isfinite(
            radius_std_A
        )

        else np.nan
    )

    N6 = (
        weighted_mean_strict(
            recipe_norm,
            elements_df[
                "electronegativity"
            ],
        )
    )

    N7, _ = (
        weighted_mean_renorm(
            recipe_norm,
            elements_df[
                "d_band_center"
            ],
        )
    )

    N8 = (
        weighted_mean_strict(
            recipe_norm,
            elements_df[
                "Tm"
            ],
        )
    )

    T_K = to_kelvin(
        T_value,
        T_unit,
    )

    N9 = (
        T_K / N8

        if (
            np.isfinite(
                T_K
            )
            and np.isfinite(
                N8
            )
            and N8 != 0
        )

        else np.nan
    )

    N10 = (
        weighted_mean_strict(
            recipe_norm,
            elements_df[
                "density"
            ],
        )
    )

    N11 = _to_float(
        np_diameter_nm
    )

    if (
        np.isfinite(
            N11
        )
        and N11 <= 0
    ):
        N11 = np.nan

    N12_raw = (
        compute_np_atom_count(
            N11,
            N4,
        )
    )

    N12 = (
        math.log1p(
            N12_raw
        )

        if (
            np.isfinite(
                N12_raw
            )
            and N12_raw >= 0
        )

        else np.nan
    )

    N13 = (
        N12_raw
        ** (-1.0 / 3.0)

        if (
            np.isfinite(
                N12_raw
            )
            and N12_raw > 0
        )

        else np.nan
    )

    N14 = (
        weighted_mean_strict(
            recipe_norm,
            elements_df[
                "work_function_eV"
            ],
        )
    )

    N15 = (
        weighted_mean_strict(
            recipe_norm,
            elements_df[
                "surface_energy_Jm2"
            ],
        )
    )

    (
        N16,
        carbide_pairs,
    ) = weighted_mean_renorm(
        recipe_norm,
        elements_df[
            "carbide_form_energy_eV_per_C"
        ],
    )

    if (
        carbide_pairs
        and np.isfinite(
            N16
        )
    ):

        variance = sum(
            fraction
            * (
                value
                - N16
            ) ** 2

            for fraction, value
            in carbide_pairs
        )

        N17 = float(
            math.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

    else:
        N17 = np.nan

    N18, _ = (
        weighted_mean_renorm(
            recipe_norm,
            elements_df[
                "oxide_DG_kJ_per_molO2"
            ],
        )
    )

    output: dict[
        str,
        Any,
    ] = {
        "N1":
            float(
                frac1
            ),

        "N2":
            float(
                frac2
            ),

        "N3":
            float(
                frac3
            ),

        "N4":
            N4,

        "N5":
            N5,

        "N6":
            N6,

        "N7":
            N7,

        "N8":
            N8,

        "N9":
            N9,

        "N10":
            N10,

        "N11":
            N11,

        "N12":
            N12,

        "N13":
            N13,

        "N14":
            N14,

        "N15":
            N15,

        "N16":
            N16,

        "N17":
            N17,

        "N18":
            N18,
    }

    for index in range(
        1,
        19,
    ):

        feature = (
            f"N{index}"
        )

        output[
            f"miss_{feature}"
        ] = (
            0

            if index <= 3

            else int(
                not np.isfinite(
                    _to_float(
                        output[
                            feature
                        ]
                    )
                )
            )
        )

    output.update(
        {
            "metal1":
                metal1,

            "metal2":
                metal2,

            "metal3":
                metal3,

            "frac1":
                float(
                    frac1
                ),

            "frac2":
                float(
                    frac2
                ),

            "frac3":
                float(
                    frac3
                ),

            "temperature_K":
                T_K,

            "np_size_nm":
                N11,

            "N12_raw_atoms":
                N12_raw,

            "recipe_key":
                ";".join(
                    f"{element}:{fraction:.8f}"

                    for element, fraction
                    in sorted(
                        recipe_norm
                    )
                ),
        }
    )

    # Compatibility aliases required by the current z_c script.
    output.update(
        {
            "N1_element":
                metal1,

            "N1_atomic_fraction":
                float(
                    frac1
                ),

            "N2_element":
                metal2,

            "N2_atomic_fraction":
                float(
                    frac2
                ),

            "N3_element":
                metal3,

            "N3_atomic_fraction":
                float(
                    frac3
                ),

            "N4_weighted_atomic_radius_nm":
                N4,

            "N5_std_atomic_radius_nm":
                N5,

            "N6_weighted_electronegativity":
                N6,

            "N7_weighted_d_band_center_eV":
                N7,

            "N8_weighted_melting_point_K":
                N8,

            "N9_T_over_Tm_eff":
                N9,

            "N10_weighted_density_g_cm3":
                N10,

            "N11_np_diameter_nm":
                N11,

            "N12_np_total_metal_atoms_log1p":
                N12,

            "N13_surface_atom_fraction_proxy":
                N13,

            "N14_weighted_work_function_eV":
                N14,

            "N15_weighted_surface_energy_J_m2":
                N15,

            "N16_weighted_carbide_formation_energy_eV_per_C":
                N16,

            "N17_std_carbide_formation_energy_eV_per_C":
                N17,

            "N18_weighted_oxide_DG_kJ_per_molO2":
                N18,
        }
    )

    return output


def compute_znuc_row(
    row: Mapping[str, Any]
    | pd.Series,
    T_K: Any | None = None,
    np_d_nm: Any | None = None,
    elements_df: pd.DataFrame | None = None,
    elements_properties_path: str | Path | None = None,
) -> dict[
    str,
    Any,
]:

    recipe = (
        recipe_from_row(
            row
        )
    )

    if not recipe:
        raise ValueError(
            "No valid metal/fraction columns "
            "were found in this row."
        )

    if T_K is None:

        T_K = _first_present(
            row,
            [
                "temperature_K",
                "growth_temperature_K",
                "T_K",
                "P1",
            ],
        )

    if np_d_nm is None:

        np_d_nm = _first_present(
            row,
            [
                "np_size_nm",
                "np_diameter_nm",
                "particle_size_nm",
                "N11",
            ],
        )

    return compute_znuc(
        recipe=recipe,
        T_value=T_K,
        T_unit="K",
        np_diameter_nm=np_d_nm,
        elements_df=elements_df,
        elements_properties_path=elements_properties_path,
    )


# =============================================================================
# DIRECT SELF-TEST
# =============================================================================

def run_self_test() -> None:

    print(
        "=" * 100
    )

    print(
        "[Z_NUC SELF-TEST START]"
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
        f"z_nuc prompt        : "
        f"{ZNUC_PROMPT_PATH}"
    )

    print(
        "=" * 100
    )

    elements_df = (
        load_elements_properties(
            ELEMENTS_PROPERTIES_PATH
        )
    )

    result = compute_znuc(
        recipe=SELF_TEST_RECIPE,
        T_value=SELF_TEST_T_K,
        T_unit="K",
        np_diameter_nm=SELF_TEST_NP_DIAMETER_NM,
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
        f"NP diameter  : "
        f"{SELF_TEST_NP_DIAMETER_NM} nm"
    )

    display_order = (
        [
            f"N{index}"
            for index
            in range(
                1,
                19,
            )
        ]
        + [
            f"miss_N{index}"
            for index
            in range(
                1,
                19,
            )
        ]
        + [
            "metal1",
            "metal2",
            "metal3",
            "N12_raw_atoms",
            "recipe_key",
        ]
    )

    print(
        "\n[SELF-TEST OUTPUT]"
    )

    print(
        pd.Series(
            {
                key:
                    result.get(
                        key
                    )

                for key
                in display_order
            }
        ).to_string()
    )

    print(
        "\n[OK] "
        "z_nuc self-test completed."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":

    if RUN_SELF_TEST:
        run_self_test()

    else:
        print(
            "[OK] "
            "z_nuc module is ready."
        )