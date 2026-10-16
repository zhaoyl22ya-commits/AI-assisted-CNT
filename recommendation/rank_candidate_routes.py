# rank_candidate_routes.py
# =============================================================================
# CNT 3.0 — L123-001 route-level screening
#
# Core logic
# ----------
# 1. Build all valid catalyst × route combinations in one batch.
# 2. Rebuild physically consistent N/C states using the revised z_nuc and z_c.
# 3. Predict raw L12 q10/q50/q90; ordered quantiles are diagnostic only.
# 4. Balanced = raw q50; Potential = raw q90; Conservative reference = raw q10.
# 5. Ordered q90-q10 is descriptive only and is never used as a ranking penalty.
# 6. Select q50 Top-R and q90 Top-R per catalyst, then take their union.
# 7. L1 is retained only as a baseline prediction on the union shortlist.
#
# Project structure:
#   <project_root>/recommendation/rank_candidate_routes.py
# =============================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "promptsV2"
FEATURE_BUILDERS_DIR = SCRIPT_DIR / "feature_builders"

CANDIDATES_CSV = PROJECT_DIR / "candidates_recipes.csv"
TEMPLATES_CSV = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "route_templates"
    / "route_templates_zpf.csv"
)

PRIORS_BY_MODE_CSV = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "candidate_priors"
    / "priors_by_growth_mode_3p0.csv"
)
PRIORS_GLOBAL_CSV = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "candidate_priors"
    / "priors_global_3p0.csv"
)

ELEMENTS_PROPERTIES_TXT = PROMPTS_DIR / "elements_properties.txt"
ZNUC_MODULE_PATH = FEATURE_BUILDERS_DIR / "build_znuc.py"
ZC_MODULE_PATH = FEATURE_BUILDERS_DIR / "build_zc.py"

L12_MODEL_DIR = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "l12_quantile"
    / "final_models"
)
L12_Q10_MODEL = L12_MODEL_DIR / "L12_composite_q10.txt"
L12_Q50_MODEL = L12_MODEL_DIR / "L12_composite_q50.txt"
L12_Q90_MODEL = L12_MODEL_DIR / "L12_composite_q90.txt"

L1_MODEL = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "l1_composite"
    / "final_models"
    / "L1_composite_final_score.txt"
)

OUTDIR = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "rank_candidate_routes"
)

PRIOR_Q = "q50"
CANDIDATE_ROUTE_MODE = "auto"  # auto / attached / expand
DEFAULT_GROWTH_MODE = ""

TOPR_BALANCED_PER_CATALYST = 10
TOPR_POTENTIAL_PER_CATALYST = 10
TOPK_BALANCED_PER_CATALYST = 5
TOPK_POTENTIAL_PER_CATALYST = 5

PREDICTION_CHUNK = 200000


# =============================================================================
# FEATURE SCHEMA
# =============================================================================

N_COLS = [f"N{i}" for i in range(1, 19)]
C_COLS = [f"C{i}" for i in range(1, 13)]
F_COLS = [f"F{i}" for i in range(1, 19)]
P_COLS = [f"P{i}" for i in range(1, 13)]

PHYSICAL_COLS = N_COLS + C_COLS + F_COLS + P_COLS
MISS_COLS = [f"miss_{column}" for column in PHYSICAL_COLS]
EXPECTED_FEATURES = PHYSICAL_COLS + MISS_COLS

MODE_MAP_NUM_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
MODE_MAP_LETTER_TO_NUM = {v: k for k, v in MODE_MAP_NUM_TO_LETTER.items()}

METAL_ALIASES = {
    1: ["metal1", "metal_1", "m1", "elem1", "element1", "element_1"],
    2: ["metal2", "metal_2", "m2", "elem2", "element2", "element_2"],
    3: ["metal3", "metal_3", "m3", "elem3", "element3", "element_3"],
}
FRACTION_ALIASES = {
    1: ["frac1", "frac_1", "pct1", "pct_1", "percent1", "percent_1"],
    2: ["frac2", "frac_2", "pct2", "pct_2", "percent2", "percent_2"],
    3: ["frac3", "frac_3", "pct3", "pct_3", "percent3", "percent_3"],
}

_ZNUC_MODULE_CACHE: ModuleType | None = None
_ZC_MODULE_CACHE: ModuleType | None = None
_ELEMENTS_DF_CACHE: pd.DataFrame | None = None


# =============================================================================
# BASIC HELPERS
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


def _first_present(
    row: Mapping[str, Any] | pd.Series,
    candidates: Iterable[str],
) -> Any:
    for name in candidates:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return np.nan


def read_csv_smart(path: str | Path, low_memory: bool = False) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV not found:\n  {resolved}")

    last_error: Exception | None = None
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(
                resolved,
                encoding=encoding,
                low_memory=low_memory,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not read CSV: {resolved}")


def normalize_route_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"auto", "attached", "expand"}:
        raise ValueError("candidate_route_mode must be auto, attached, or expand")
    return mode


def normalize_prior_q(value: str) -> str:
    q = str(value).strip().lower()
    if q not in {"q10", "q50", "q90"}:
        raise ValueError("prior_q must be q10, q50, or q90")
    return q


def resolve_project_file(
    primary_path: str | Path,
    expected_name: str | None = None,
) -> Path:
    """
    Resolve only inside the current project.
    """
    primary = Path(primary_path).expanduser()
    if not primary.is_absolute():
        primary = PROJECT_DIR / primary
    primary = primary.resolve()

    if primary.is_file():
        return primary

    filename = expected_name or primary.name
    matches = sorted(
        path.resolve()
        for path in PROJECT_DIR.rglob(filename)
        if path.is_file()
    )

    if len(matches) == 1:
        print(f"[WARN] Primary path missing; using project match: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        listed = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Multiple files named {filename!r} were found inside the current project.\n"
            f"{listed}\nSet the exact path in CONFIG."
        )

    raise FileNotFoundError(
        f"File not found inside the current project:\n  {primary}"
    )


# =============================================================================
# DYNAMIC IMPORTS
# =============================================================================

def load_module_from_path(module_name: str, path: str | Path) -> ModuleType:
    resolved = resolve_project_file(path)
    spec = importlib.util.spec_from_file_location(module_name, resolved)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {resolved}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_znuc_module() -> ModuleType:
    global _ZNUC_MODULE_CACHE

    if _ZNUC_MODULE_CACHE is None:
        module = load_module_from_path("cnt_znuc_builder", ZNUC_MODULE_PATH)
        for required in ("compute_znuc", "load_elements_properties"):
            if not hasattr(module, required):
                raise RuntimeError(f"z_nuc module is missing {required}()")
        _ZNUC_MODULE_CACHE = module

    return _ZNUC_MODULE_CACHE


def load_zc_module() -> ModuleType:
    global _ZC_MODULE_CACHE

    if _ZC_MODULE_CACHE is None:
        module = load_module_from_path("cnt_zc_builder", ZC_MODULE_PATH)
        for required in ("compute_zc", "load_priors"):
            if not hasattr(module, required):
                raise RuntimeError(f"z_c module is missing {required}()")
        _ZC_MODULE_CACHE = module

    return _ZC_MODULE_CACHE


def load_elements_properties_once() -> pd.DataFrame:
    global _ELEMENTS_DF_CACHE

    if _ELEMENTS_DF_CACHE is None:
        znuc_module = load_znuc_module()
        _ELEMENTS_DF_CACHE = znuc_module.load_elements_properties(
            ELEMENTS_PROPERTIES_TXT
        )

    return _ELEMENTS_DF_CACHE


# =============================================================================
# GROWTH MODE
# =============================================================================

def normalize_growth_mode(value: Any) -> str | None:
    if _is_missing(value):
        return None

    token = str(value).strip().upper()
    if not token:
        return None

    if token in MODE_MAP_LETTER_TO_NUM:
        return token

    try:
        numeric = int(float(token))
        if numeric in MODE_MAP_NUM_TO_LETTER:
            return MODE_MAP_NUM_TO_LETTER[numeric]
    except Exception:
        pass

    for letter in "ABCDE":
        if token.startswith(
            (
                f"{letter}_",
                f"{letter}-",
                f"{letter} ",
                f"{letter}:",
                f"MODE_{letter}",
                f"GROWTH_MODE_{letter}",
            )
        ):
            return letter

    return None


def infer_template_mode(row: Mapping[str, Any] | pd.Series) -> str | None:
    explicit = normalize_growth_mode(row.get("growth_mode"))
    if explicit is not None:
        return explicit

    template_id = str(row.get("template_id", "")).strip().upper()
    if (
        len(template_id) >= 2
        and template_id[0] in MODE_MAP_LETTER_TO_NUM
        and template_id[1] == "_"
    ):
        return template_id[0]

    return None


# =============================================================================
# CANDIDATES
# =============================================================================

def canonical_element_symbol(value: Any) -> str | None:
    if _is_missing(value):
        return None

    token = str(value).strip()
    if not token:
        return None

    return token.upper() if len(token) == 1 else token[0].upper() + token[1:].lower()


def recipe_key_from_row(row: Mapping[str, Any] | pd.Series) -> str:
    pairs = []

    for index in (1, 2, 3):
        metal = row.get(f"metal{index}")
        fraction = _to_float(row.get(f"frac{index}", np.nan))

        if metal is not None and np.isfinite(fraction) and fraction > 0:
            pairs.append((str(metal), fraction))

    if not pairs:
        return "UNKNOWN"

    return ";".join(
        f"{metal}:{fraction:.8f}"
        for metal, fraction in sorted(pairs, key=lambda item: item[0])
    )


def metal_set_from_row(row: Mapping[str, Any] | pd.Series) -> str:
    metals = sorted(
        {
            str(row[f"metal{index}"])
            for index in (1, 2, 3)
            if f"metal{index}" in row and not _is_missing(row[f"metal{index}"])
        }
    )
    return "+".join(metals) if metals else "UNKNOWN"


def recipe_list_from_row(
    row: Mapping[str, Any] | pd.Series,
) -> list[tuple[str, float]]:
    recipe = []

    for index in (1, 2, 3):
        metal = row.get(f"metal{index}")
        fraction = _to_float(row.get(f"frac{index}", np.nan))

        if metal is not None and np.isfinite(fraction) and fraction > 0:
            recipe.append((str(metal), fraction))

    return recipe


def normalize_recipe_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    output = candidates.copy()

    for index in (1, 2, 3):
        output[f"metal{index}"] = output.apply(
            lambda row, i=index: _first_present(row, METAL_ALIASES[i]),
            axis=1,
        )
        output[f"frac{index}"] = output.apply(
            lambda row, i=index: _first_present(row, FRACTION_ALIASES[i]),
            axis=1,
        )

        output[f"metal{index}"] = output[f"metal{index}"].map(
            canonical_element_symbol
        )
        output[f"frac{index}"] = pd.to_numeric(
            output[f"frac{index}"],
            errors="coerce",
        )

    for row_index, row in output.iterrows():
        merged: dict[str, float] = {}

        for index in (1, 2, 3):
            metal = row[f"metal{index}"]
            fraction = _to_float(row[f"frac{index}"])

            if metal is None or not np.isfinite(fraction) or fraction <= 0:
                continue

            merged[metal] = merged.get(metal, 0.0) + fraction

        if not merged:
            continue

        total = float(sum(merged.values()))
        recipe = sorted(
            [(metal, fraction / total) for metal, fraction in merged.items()],
            key=lambda item: (-item[1], item[0]),
        )[:3]

        padded = recipe + [(None, 0.0)] * (3 - len(recipe))

        for position, (metal, fraction) in enumerate(padded, start=1):
            output.loc[row_index, f"metal{position}"] = metal
            output.loc[row_index, f"frac{position}"] = float(fraction)

    output["recipe_key"] = output.apply(recipe_key_from_row, axis=1)
    output["metal_set"] = output.apply(metal_set_from_row, axis=1)

    candidate_id_column = next(
        (
            column
            for column in ("catalyst_id", "candidate_id", "recipe_id")
            if column in output.columns
        ),
        None,
    )

    if candidate_id_column is None:
        output["catalyst_id"] = output["recipe_key"]
    else:
        raw_ids = output[candidate_id_column].astype(str)
        invalid = (
            output[candidate_id_column].isna()
            | raw_ids.str.strip().eq("")
            | raw_ids.str.upper().eq("NAN")
        )
        raw_ids.loc[invalid] = output.loc[invalid, "recipe_key"]
        output["catalyst_id"] = raw_ids

    output["growth_mode_normalized"] = (
        output["growth_mode"].map(normalize_growth_mode)
        if "growth_mode" in output.columns
        else None
    )
    output["_candidate_input_row"] = output.index.astype(str)

    return output


# =============================================================================
# TEMPLATES
# =============================================================================

def prepare_templates(templates: pd.DataFrame) -> pd.DataFrame:
    work = templates.copy()

    if "template_id" not in work.columns:
        raise RuntimeError("route_templates_zpf.csv is missing template_id.")

    work["template_id"] = work["template_id"].astype(str).str.strip()

    if work["template_id"].eq("").any():
        raise RuntimeError("Blank template_id found.")

    if work["template_id"].duplicated().any():
        duplicates = work.loc[
            work["template_id"].duplicated(keep=False),
            "template_id",
        ].tolist()
        raise RuntimeError(
            "Duplicate template_id values found:\n  "
            + "\n  ".join(map(str, duplicates))
        )

    work["growth_mode_normalized"] = work.apply(
        infer_template_mode,
        axis=1,
    )

    for feature in F_COLS + P_COLS:
        if feature not in work.columns:
            raise RuntimeError(f"Template file is missing {feature}.")

        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        miss_column = f"miss_{feature}"

        if miss_column not in work.columns:
            work[miss_column] = work[feature].isna().astype(int)
        else:
            work[miss_column] = (
                pd.to_numeric(work[miss_column], errors="coerce")
                .fillna(0)
                .gt(0)
                .astype(int)
            )

        work.loc[work[miss_column].eq(1), feature] = 0.0
        work[feature] = work[feature].fillna(0.0)

    return work

def get_temperature_K_from_template(
    template_row: Mapping[str, Any] | pd.Series,
) -> float:
    if (
        "P1" in template_row
        and _to_float(template_row.get("miss_P1", 1)) <= 0
    ):
        value = _to_float(template_row["P1"])
        if np.isfinite(value) and value > 0:
            return value

    return np.nan


def get_observed_template_value(
    template_row: Mapping[str, Any] | pd.Series,
    feature: str,
) -> float:
    """
    Return a template feature only when its miss_* flag says it was observed.

    Route templates use 0.0 as the model-space placeholder for missing values,
    so the miss_* flag must be checked before passing P8/P9/P10 into z_c.
    """
    miss_value = _to_float(
        template_row.get(
            f"miss_{feature}",
            1,
        )
    )

    if (
        not np.isfinite(miss_value)
        or miss_value > 0
    ):
        return np.nan

    return _to_float(
        template_row.get(
            feature,
            np.nan,
        )
    )


# =============================================================================
# CANDIDATE × TEMPLATE COMBINATIONS
# =============================================================================

def choose_route_mode(candidates: pd.DataFrame, requested_mode: str) -> str:
    mode = normalize_route_mode(requested_mode)

    if mode != "auto":
        return mode

    if "template_id" not in candidates.columns:
        return "expand"

    template_ids = candidates["template_id"].astype(str).str.strip()
    has_attached = (
        candidates["template_id"].notna()
        & template_ids.ne("")
        & template_ids.str.upper().ne("NAN")
    )

    return "attached" if has_attached.any() else "expand"


def merge_candidate_and_template(
    candidate: Mapping[str, Any] | pd.Series,
    template: Mapping[str, Any] | pd.Series,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "catalyst_id": candidate.get("catalyst_id", candidate.get("recipe_key", "")),
        "recipe_key": candidate.get("recipe_key", ""),
        "metal_set": candidate.get("metal_set", ""),
        "metal1": candidate.get("metal1"),
        "metal2": candidate.get("metal2"),
        "metal3": candidate.get("metal3"),
        "frac1": candidate.get("frac1", 0.0),
        "frac2": candidate.get("frac2", 0.0),
        "frac3": candidate.get("frac3", 0.0),
        "candidate_growth_mode": candidate.get("growth_mode_normalized"),
        "candidate_input_row": candidate.get("_candidate_input_row"),
        "candidate_state_id": (
            str(candidate.get("catalyst_id", candidate.get("recipe_key", "")))
            + "::row_"
            + str(candidate.get("_candidate_input_row", ""))
        ),
        "template_id": template.get("template_id", ""),
        "growth_mode": template.get(
            "growth_mode_normalized",
            template.get("growth_mode"),
        ),
        "source_level": template.get("source_level"),
        "cluster_size": template.get("cluster_size", np.nan),
        "template_rank_in_mode": template.get("template_rank_in_mode", np.nan),
        "template_missing_fraction": template.get(
            "template_missing_fraction",
            np.nan,
        ),
        "cluster_mean_missing_ratio": template.get(
            "cluster_mean_missing_ratio",
            np.nan,
        ),
        "nearest_real_sample_id": template.get("nearest_real_sample_id"),
        "nearest_real_distance": template.get("nearest_real_distance", np.nan),
    }

    for name in (
        "N11",
        "np_size_nm",
        "np_diameter_nm",
        "particle_size_nm",
        "C3",
        "C_over_M",
        "C_M_ratio",
        "C4",
        "S_over_M",
        "S_M_ratio",
    ):
        if name in candidate:
            row[f"candidate_{name}"] = candidate[name]

    for feature in F_COLS + P_COLS:
        row[feature] = template.get(feature, 0.0)
        row[f"miss_{feature}"] = template.get(f"miss_{feature}", 1)

    return row


def build_candidate_template_pairs(
    candidates: pd.DataFrame,
    templates: pd.DataFrame,
    route_mode: str,
    global_growth_mode: str | None,
) -> pd.DataFrame:
    mode = choose_route_mode(candidates, route_mode)
    print(f"[INFO] Candidate-route mode: {mode}")

    global_mode = normalize_growth_mode(global_growth_mode)
    pair_rows = []

    if mode == "attached":
        if "template_id" not in candidates.columns:
            raise RuntimeError("attached mode requires template_id in candidates.")

        template_lookup = templates.set_index("template_id", drop=False)

        for _, candidate in candidates.iterrows():
            template_id = str(candidate.get("template_id", "")).strip()

            if not template_id or template_id.upper() == "NAN":
                continue
            if template_id not in template_lookup.index:
                continue

            template = template_lookup.loc[template_id]
            candidate_mode = (
                global_mode or candidate.get("growth_mode_normalized")
            )
            template_mode = template.get("growth_mode_normalized")

            if (
                candidate_mode is not None
                and template_mode is not None
                and candidate_mode != template_mode
            ):
                continue

            pair_rows.append(
                merge_candidate_and_template(candidate, template)
            )

    else:
        dedup_columns = [
            "catalyst_id",
            "recipe_key",
            "growth_mode_normalized",
        ]

        for optional in (
            "N11",
            "np_size_nm",
            "np_diameter_nm",
            "C3",
            "C_over_M",
            "C4",
            "S_over_M",
        ):
            if optional in candidates.columns:
                dedup_columns.append(optional)

        unique_candidates = candidates.drop_duplicates(
            subset=[
                column
                for column in dedup_columns
                if column in candidates.columns
            ],
            keep="first",
        )

        for _, candidate in unique_candidates.iterrows():
            candidate_mode = (
                global_mode or candidate.get("growth_mode_normalized")
            )

            candidate_templates = (
                templates
                if candidate_mode is None
                else templates.loc[
                    templates["growth_mode_normalized"].eq(candidate_mode)
                ]
            )

            for _, template in candidate_templates.iterrows():
                pair_rows.append(
                    merge_candidate_and_template(candidate, template)
                )

    if not pair_rows:
        raise RuntimeError(
            "No valid catalyst-template combinations were generated."
        )

    pairs = pd.DataFrame(pair_rows)
    pairs["combination_id"] = (
        pairs["candidate_state_id"].astype(str)
        + "||"
        + pairs["template_id"].astype(str)
    )

    return pairs.drop_duplicates(
        subset=["combination_id"],
        keep="first",
    ).reset_index(drop=True)


# =============================================================================
# PRIORS AND PHYSICAL FEATURE BUILD
# =============================================================================

def load_mode_priors(
    mode: str | None,
    q: str,
    zc_module: ModuleType,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    mode_key = mode if mode is not None else "GLOBAL"
    cache_key = (mode_key, q)

    if cache_key not in cache:
        prior_path = (
            PRIORS_BY_MODE_CSV
            if PRIORS_BY_MODE_CSV.is_file()
            else PRIORS_GLOBAL_CSV
        )
        cache[cache_key] = zc_module.load_priors(
            path=prior_path,
            q=q,
            growth_mode=mode,
        )

    return cache[cache_key]


def resolve_candidate_scalar(
    row: Mapping[str, Any] | pd.Series,
    names: Iterable[str],
    prior_value: Any,
) -> tuple[float, str]:
    for name in names:
        candidate_name = f"candidate_{name}"

        if candidate_name in row:
            value = _to_float(row[candidate_name])
            if np.isfinite(value):
                return value, "candidate"

    prior_number = _to_float(prior_value)

    if np.isfinite(prior_number):
        return prior_number, "prior"

    return np.nan, "missing"


def enforce_model_missing_convention(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()

    for feature in PHYSICAL_COLS:
        miss_column = f"miss_{feature}"

        if feature not in work.columns:
            work[feature] = np.nan

        if miss_column not in work.columns:
            work[miss_column] = work[feature].isna().astype(int)

        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work[miss_column] = (
            pd.to_numeric(work[miss_column], errors="coerce")
            .fillna(1)
            .gt(0)
            .astype(int)
        )

        work.loc[work[feature].isna(), miss_column] = 1
        work.loc[work[miss_column].eq(1), feature] = np.nan

    return work


def build_physical_feature_rows(
    combinations: pd.DataFrame,
    prior_q: str,
) -> pd.DataFrame:
    znuc_module = load_znuc_module()
    zc_module = load_zc_module()
    elements_df = load_elements_properties_once()

    prior_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    total = len(combinations)

    for position, (_, pair) in enumerate(combinations.iterrows(), start=1):
        recipe = recipe_list_from_row(pair)
        if not recipe:
            continue

        mode = normalize_growth_mode(pair.get("growth_mode"))
        priors = load_mode_priors(
            mode,
            prior_q,
            zc_module,
            prior_cache,
        )

        np_size_nm, np_size_source = resolve_candidate_scalar(
            pair,
            ("N11", "np_size_nm", "np_diameter_nm", "particle_size_nm"),
            priors.get("np_size_nm", np.nan),
        )
        C_over_M, C_over_M_source = resolve_candidate_scalar(
            pair,
            ("C3", "C_over_M", "C_M_ratio"),
            priors.get("C_over_M", np.nan),
        )
        S_over_M, S_over_M_source = resolve_candidate_scalar(
            pair,
            ("C4", "S_over_M", "S_M_ratio"),
            priors.get("S_over_M", np.nan),
        )

        T_K = get_temperature_K_from_template(pair)

        p_C = get_observed_template_value(
            pair,
            "P8",
        )
        p_H2 = get_observed_template_value(
            pair,
            "P9",
        )
        p_C_over_p_H2 = get_observed_template_value(
            pair,
            "P10",
        )

        znuc = znuc_module.compute_znuc(
            recipe=recipe,
            T_value=T_K,
            T_unit="K",
            np_diameter_nm=np_size_nm,
            elements_df=elements_df,
        )

        zc = zc_module.compute_zc(
            znuc=znuc,
            T_K=T_K,
            priors=priors,
            C_over_M=C_over_M,
            S_over_M=S_over_M,
            np_size_nm=np_size_nm,
            p_C=p_C,
            p_H2=p_H2,
            p_C_over_p_H2=p_C_over_p_H2,
            growth_mode=mode,
            prior_q=prior_q,
            elements_df=elements_df,
        )

        feature_row = pair.to_dict()

        for feature in N_COLS:
            feature_row[feature] = zc.get(
                f"znuc_{feature}",
                znuc.get(feature, np.nan),
            )
            feature_row[f"miss_{feature}"] = zc.get(
                f"znuc_miss_{feature}",
                znuc.get(f"miss_{feature}", 1),
            )

        for feature in C_COLS:
            feature_row[feature] = zc.get(feature, np.nan)
            feature_row[f"miss_{feature}"] = zc.get(
                f"miss_{feature}",
                1,
            )

        feature_row.update(
            {
                "temperature_K": T_K,
                "np_size_source": np_size_source,
                "C_over_M_source": C_over_M_source,
                "S_over_M_source": S_over_M_source,
                "prior_q": prior_q,
                "prior_scope": priors.get("prior_scope"),
                "prior_fallback_used": priors.get(
                    "prior_fallback_used",
                    False,
                ),
                "C10_raw_np_total_atoms": zc.get(
                    "C10_raw_np_total_atoms",
                    np.nan,
                ),
                "C11_raw_C_atoms_per_NP": zc.get(
                    "C11_raw_C_atoms_per_NP",
                    np.nan,
                ),
                "N12_raw_atoms": znuc.get("N12_raw_atoms", np.nan),
            }
        )

        rows.append(feature_row)

        if position % 1000 == 0 or position == total:
            print(f"[FEATURE BUILD] {position}/{total}")

    if not rows:
        raise RuntimeError("No physical feature rows were built.")

    return enforce_model_missing_convention(pd.DataFrame(rows))


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def load_booster(path: str | Path) -> tuple[lgb.Booster, Path]:
    resolved = resolve_project_file(path)
    return lgb.Booster(model_file=str(resolved)), resolved


def booster_feature_order(
    booster: lgb.Booster,
    model_label: str,
) -> list[str]:
    model_features = booster.feature_name()

    if booster.num_feature() != len(EXPECTED_FEATURES):
        raise RuntimeError(
            f"{model_label} expects {booster.num_feature()} features, "
            f"but the current schema contains {len(EXPECTED_FEATURES)}."
        )

    expected_set = set(EXPECTED_FEATURES)
    model_set = set(model_features)

    if model_set == expected_set:
        return list(model_features)

    if all(str(name).startswith("Column_") for name in model_features):
        return list(EXPECTED_FEATURES)

    missing = sorted(expected_set - model_set)
    extra = sorted(model_set - expected_set)

    raise RuntimeError(
        f"{model_label} feature names do not match the 120D schema.\n"
        f"Missing from model: {missing[:20]}\n"
        f"Unexpected in model: {extra[:20]}"
    )


def validate_model_schemas(
    boosters: dict[str, lgb.Booster],
) -> dict[str, list[str]]:
    orders = {
        label: booster_feature_order(booster, label)
        for label, booster in boosters.items()
    }

    reference_set = set(orders["L12_q50"])

    for label, order in orders.items():
        if set(order) != reference_set:
            raise RuntimeError(
                f"{label} does not use the same feature set as L12_q50."
            )

    return orders


def predict_in_chunks(
    booster: lgb.Booster,
    X: pd.DataFrame,
    chunk: int,
) -> np.ndarray:
    predictions = []

    for start in range(0, len(X), chunk):
        part = X.iloc[start : start + chunk]
        predictions.append(
            np.asarray(booster.predict(part), dtype=float)
        )

    return np.concatenate(predictions)


def predict_quantiles(
    feature_rows: pd.DataFrame,
    boosters: dict[str, lgb.Booster],
    feature_orders: dict[str, list[str]],
    chunk: int,
) -> pd.DataFrame:
    output = feature_rows.copy()

    raw_q10 = predict_in_chunks(
        boosters["L12_q10"],
        output[feature_orders["L12_q10"]],
        chunk,
    )
    raw_q50 = predict_in_chunks(
        boosters["L12_q50"],
        output[feature_orders["L12_q50"]],
        chunk,
    )
    raw_q90 = predict_in_chunks(
        boosters["L12_q90"],
        output[feature_orders["L12_q90"]],
        chunk,
    )

    raw_matrix = np.column_stack([raw_q10, raw_q50, raw_q90])
    ordered = np.sort(raw_matrix, axis=1)

    output["pred_q10_raw"] = raw_q10
    output["pred_q50_raw"] = raw_q50
    output["pred_q90_raw"] = raw_q90

    # Raw model outputs define the three recommendation strategies.
    output["pred_q10"] = raw_q10
    output["pred_q50"] = raw_q50
    output["pred_q90"] = raw_q90

    # Ordered quantiles are diagnostic only.
    output["pred_q10_ordered"] = ordered[:, 0]
    output["pred_q50_ordered"] = ordered[:, 1]
    output["pred_q90_ordered"] = ordered[:, 2]

    output["quantile_crossing_detected"] = (
        (raw_q10 > raw_q50)
        | (raw_q50 > raw_q90)
        | (raw_q10 > raw_q90)
    )

    output["interval_width"] = (
        output["pred_q90_ordered"] - output["pred_q10_ordered"]
    )

    # No interval-width penalty.
    output["score_balanced"] = output["pred_q50_raw"]
    output["score_potential"] = output["pred_q90_raw"]
    output["score_conservative"] = output["pred_q10_raw"]

    # Backward-compatible aliases.
    output["screen_score"] = output["score_balanced"]
    output["final_score"] = output["score_balanced"]

    return output


# =============================================================================
# RANKING
# =============================================================================

def top_rows_per_catalyst(
    frame: pd.DataFrame,
    score_column: str,
    top_n: int,
    rank_column: str,
) -> pd.DataFrame:
    ranked = frame.sort_values(
        [
            "catalyst_id",
            score_column,
            "template_missing_fraction",
            "template_id",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).copy()

    ranked[rank_column] = ranked.groupby("catalyst_id").cumcount() + 1
    return ranked.loc[ranked[rank_column] <= int(top_n)].copy()


def build_union_shortlist(
    predictions: pd.DataFrame,
    topR_balanced: int,
    topR_potential: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balanced = top_rows_per_catalyst(
        predictions,
        "score_balanced",
        topR_balanced,
        "rank_balanced",
    )
    potential = top_rows_per_catalyst(
        predictions,
        "score_potential",
        topR_potential,
        "rank_potential",
    )

    balanced_ids = set(balanced["combination_id"])
    potential_ids = set(potential["combination_id"])

    union = predictions.loc[
        predictions["combination_id"].isin(balanced_ids | potential_ids)
    ].copy()

    union["rank_balanced"] = union["combination_id"].map(
        balanced.set_index("combination_id")["rank_balanced"]
    )
    union["rank_potential"] = union["combination_id"].map(
        potential.set_index("combination_id")["rank_potential"]
    )

    union["selected_by_balanced"] = union["combination_id"].isin(balanced_ids)
    union["selected_by_potential"] = union["combination_id"].isin(potential_ids)

    union["selection_reason"] = np.select(
        [
            union["selected_by_balanced"] & union["selected_by_potential"],
            union["selected_by_balanced"],
            union["selected_by_potential"],
        ],
        ["balanced+potential", "balanced", "potential"],
        default="",
    )

    union = union.sort_values(
        ["catalyst_id", "score_balanced", "score_potential"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)

    return balanced, potential, union


def add_l1_baseline(
    shortlist: pd.DataFrame,
    booster: lgb.Booster,
    feature_order: list[str],
    chunk: int,
) -> pd.DataFrame:
    output = shortlist.copy()
    output["pred_L1"] = predict_in_chunks(
        booster,
        output[feature_order],
        chunk,
    )
    return output


def compact_topk_union(
    shortlist: pd.DataFrame,
    topK_balanced: int,
    topK_potential: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balanced = top_rows_per_catalyst(
        shortlist,
        "score_balanced",
        topK_balanced,
        "final_rank_balanced",
    )
    potential = top_rows_per_catalyst(
        shortlist,
        "score_potential",
        topK_potential,
        "final_rank_potential",
    )

    balanced_ids = set(balanced["combination_id"])
    potential_ids = set(potential["combination_id"])

    union = shortlist.loc[
        shortlist["combination_id"].isin(balanced_ids | potential_ids)
    ].copy()

    union["final_rank_balanced"] = union["combination_id"].map(
        balanced.set_index("combination_id")["final_rank_balanced"]
    )
    union["final_rank_potential"] = union["combination_id"].map(
        potential.set_index("combination_id")["final_rank_potential"]
    )

    union["final_selected_balanced"] = union["combination_id"].isin(
        balanced_ids
    )
    union["final_selected_potential"] = union["combination_id"].isin(
        potential_ids
    )

    union["final_selection_reason"] = np.select(
        [
            union["final_selected_balanced"]
            & union["final_selected_potential"],
            union["final_selected_balanced"],
            union["final_selected_potential"],
        ],
        ["balanced+potential", "balanced", "potential"],
        default="",
    )

    return balanced, potential, union


# =============================================================================
# AUDIT
# =============================================================================

def build_feature_audit(
    feature_rows: pd.DataFrame,
    boosters: dict[str, lgb.Booster],
    model_paths: dict[str, Path],
) -> pd.DataFrame:
    rows = []

    for feature in EXPECTED_FEATURES:
        values = pd.to_numeric(feature_rows[feature], errors="coerce")

        row = {
            "feature": feature,
            "n_rows": int(len(feature_rows)),
            "n_nan_after_assembly": int(values.isna().sum()),
            "n_zero": int(values.eq(0).sum()),
            "min": float(values.min()) if len(values) else np.nan,
            "median": float(values.median()) if len(values) else np.nan,
            "max": float(values.max()) if len(values) else np.nan,
        }

        for label, booster in boosters.items():
            generic = all(
                str(name).startswith("Column_")
                for name in booster.feature_name()
            )
            row[f"in_{label}"] = (
                feature in set(booster.feature_name()) or generic
            )
            row[f"{label}_model"] = str(model_paths[label])

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNT 3.0 route-level candidate screening"
    )

    parser.add_argument("--candidates_csv", default=str(CANDIDATES_CSV))
    parser.add_argument("--templates_csv", default=str(TEMPLATES_CSV))
    parser.add_argument(
        "--prior_q",
        default=PRIOR_Q,
        choices=["q10", "q50", "q90"],
    )
    parser.add_argument(
        "--candidate_route_mode",
        default=CANDIDATE_ROUTE_MODE,
        choices=["auto", "attached", "expand"],
    )
    parser.add_argument("--growth_mode", default=DEFAULT_GROWTH_MODE)

    parser.add_argument("--L12_q10", default=str(L12_Q10_MODEL))
    parser.add_argument("--L12_q50", default=str(L12_Q50_MODEL))
    parser.add_argument("--L12_q90", default=str(L12_Q90_MODEL))
    parser.add_argument("--L1_model", default=str(L1_MODEL))

    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument(
        "--topR_balanced",
        type=int,
        default=TOPR_BALANCED_PER_CATALYST,
    )
    parser.add_argument(
        "--topR_potential",
        type=int,
        default=TOPR_POTENTIAL_PER_CATALYST,
    )
    parser.add_argument(
        "--topK_balanced",
        type=int,
        default=TOPK_BALANCED_PER_CATALYST,
    )
    parser.add_argument(
        "--topK_potential",
        type=int,
        default=TOPK_POTENTIAL_PER_CATALYST,
    )
    parser.add_argument(
        "--prediction_chunk",
        type=int,
        default=PREDICTION_CHUNK,
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    prior_q = normalize_prior_q(args.prior_q)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    candidates_path = resolve_project_file(args.candidates_csv)
    templates_path = resolve_project_file(args.templates_csv)

    print("=" * 116)
    print("[RANK CANDIDATE ROUTES START]")
    print(f"Project directory       : {PROJECT_DIR}")
    print(f"Candidates              : {candidates_path}")
    print(f"Route templates         : {templates_path}")
    print(f"Prior quantile          : {prior_q}")
    print(f"Candidate-route mode    : {args.candidate_route_mode}")
    print(f"Global growth-mode gate : {args.growth_mode or 'NONE'}")
    print(f"Output directory        : {outdir}")
    print("=" * 116)

    candidates = normalize_recipe_columns(
        read_csv_smart(candidates_path, low_memory=False)
    )
    candidates = candidates.loc[
        candidates["recipe_key"].ne("UNKNOWN")
    ].copy()

    templates = prepare_templates(
        read_csv_smart(templates_path, low_memory=False)
    )
    invalid_temperature = (
            templates["miss_P1"].gt(0)
            | pd.to_numeric(templates["P1"], errors="coerce").isna()
            | pd.to_numeric(templates["P1"], errors="coerce").le(0)
    )

    if invalid_temperature.any():
        print(
            f"[WARN] Dropping {int(invalid_temperature.sum())} route templates "
            f"without an observed valid P1 temperature."
        )

    templates = templates.loc[
        ~invalid_temperature
    ].copy().reset_index(drop=True)

    if templates.empty:
        raise RuntimeError(
            "No route template with an observed valid P1 temperature remains."
        )

    combinations = build_candidate_template_pairs(
        candidates=candidates,
        templates=templates,
        route_mode=args.candidate_route_mode,
        global_growth_mode=args.growth_mode,
    )

    print(
        f"[INFO] Unique catalysts     : "
        f"{combinations['catalyst_id'].nunique()}"
    )
    print(
        f"[INFO] Unique templates     : "
        f"{combinations['template_id'].nunique()}"
    )
    print(f"[INFO] Catalyst-route pairs : {len(combinations)}")

    feature_rows = build_physical_feature_rows(
        combinations,
        prior_q=prior_q,
    )

    q10_booster, q10_path = load_booster(args.L12_q10)
    q50_booster, q50_path = load_booster(args.L12_q50)
    q90_booster, q90_path = load_booster(args.L12_q90)
    l1_booster, l1_path = load_booster(args.L1_model)

    boosters = {
        "L12_q10": q10_booster,
        "L12_q50": q50_booster,
        "L12_q90": q90_booster,
        "L1": l1_booster,
    }
    model_paths = {
        "L12_q10": q10_path,
        "L12_q50": q50_path,
        "L12_q90": q90_path,
        "L1": l1_path,
    }

    feature_orders = validate_model_schemas(boosters)
    chunk = max(1, int(args.prediction_chunk))

    predicted = predict_quantiles(
        feature_rows,
        boosters,
        feature_orders,
        chunk,
    )

    balanced_topR, potential_topR, shortlist = build_union_shortlist(
        predicted,
        topR_balanced=max(1, int(args.topR_balanced)),
        topR_potential=max(1, int(args.topR_potential)),
    )

    shortlist = add_l1_baseline(
        shortlist,
        l1_booster,
        feature_orders["L1"],
        chunk,
    )

    top_balanced, top_potential, top_union = compact_topk_union(
        shortlist,
        topK_balanced=max(1, int(args.topK_balanced)),
        topK_potential=max(1, int(args.topK_potential)),
    )

    feature_audit = build_feature_audit(
        feature_rows,
        boosters,
        model_paths,
    )

    output_paths = {
        "all_predictions": outdir / "all_route_predictions.csv",
        "shortlist": outdir / "shortlisted_route_predictions.csv",
        "balanced_topR": outdir / "topR_routes_balanced.csv",
        "potential_topR": outdir / "topR_routes_potential.csv",
        "balanced_topK": outdir / "top_routes_balanced.csv",
        "potential_topK": outdir / "top_routes_potential.csv",
        "top_union": outdir / "top_routes_union.csv",
        "feature_audit": outdir / "feature_consistency_audit.csv",
        "legacy_ranked": outdir / "ranked_process_packages_L123_001.csv",
        "legacy_topk": outdir / "top_routes_per_catalyst_L123_001.csv",
        "summary": outdir / "run_summary_L123_001.json",
    }

    predicted.to_csv(
        output_paths["all_predictions"],
        index=False,
        encoding="utf-8-sig",
    )
    shortlist.to_csv(
        output_paths["shortlist"],
        index=False,
        encoding="utf-8-sig",
    )
    balanced_topR.to_csv(
        output_paths["balanced_topR"],
        index=False,
        encoding="utf-8-sig",
    )
    potential_topR.to_csv(
        output_paths["potential_topR"],
        index=False,
        encoding="utf-8-sig",
    )
    top_balanced.to_csv(
        output_paths["balanced_topK"],
        index=False,
        encoding="utf-8-sig",
    )
    top_potential.to_csv(
        output_paths["potential_topK"],
        index=False,
        encoding="utf-8-sig",
    )
    top_union.to_csv(
        output_paths["top_union"],
        index=False,
        encoding="utf-8-sig",
    )
    feature_audit.to_csv(
        output_paths["feature_audit"],
        index=False,
        encoding="utf-8-sig",
    )

    # Backward-compatible filenames for the current L123-002.
    shortlist.to_csv(
        output_paths["legacy_ranked"],
        index=False,
        encoding="utf-8-sig",
    )
    top_union.to_csv(
        output_paths["legacy_topk"],
        index=False,
        encoding="utf-8-sig",
    )

    route_counts = combinations.groupby("catalyst_id").size()
    crossing_count = int(
        predicted["quantile_crossing_detected"].sum()
    )

    summary = {
        "project_directory": str(PROJECT_DIR),
        "candidates_csv": str(candidates_path),
        "templates_csv": str(templates_path),
        "priors_by_mode_csv": str(PRIORS_BY_MODE_CSV),
        "priors_global_csv": str(PRIORS_GLOBAL_CSV),
        "elements_properties_txt": str(ELEMENTS_PROPERTIES_TXT),
        "znuc_module": str(ZNUC_MODULE_PATH),
        "zc_module": str(ZC_MODULE_PATH),
        "models": {
            label: str(path)
            for label, path in model_paths.items()
        },
        "candidate_route_mode_requested": args.candidate_route_mode,
        "global_growth_mode_gate": args.growth_mode,
        "prior_q": prior_q,
        "n_input_candidate_rows": int(len(candidates)),
        "n_unique_catalysts": int(
            combinations["catalyst_id"].nunique()
        ),
        "n_unique_templates_used": int(
            combinations["template_id"].nunique()
        ),
        "n_all_route_predictions": int(len(predicted)),
        "routes_per_catalyst_min": int(route_counts.min()),
        "routes_per_catalyst_median": float(route_counts.median()),
        "routes_per_catalyst_max": int(route_counts.max()),
        "topR_balanced_per_catalyst": int(args.topR_balanced),
        "topR_potential_per_catalyst": int(args.topR_potential),
        "n_balanced_topR_rows": int(len(balanced_topR)),
        "n_potential_topR_rows": int(len(potential_topR)),
        "n_union_shortlist_rows": int(len(shortlist)),
        "topK_balanced_per_catalyst": int(args.topK_balanced),
        "topK_potential_per_catalyst": int(args.topK_potential),
        "n_final_union_rows": int(len(top_union)),
        "quantile_crossing_rows_detected": crossing_count,
        "quantile_crossing_rate": float(
            crossing_count / max(len(predicted), 1)
        ),
        "ranking_definition": {
            "balanced": "raw q50",
            "potential": "raw q90",
            "conservative_reference": "raw q10",
            "interval_width": "ordered q90 - ordered q10; descriptive only; not used as a penalty",
            "L1": "baseline prediction on union shortlist",
        },
        "outputs": {
            key: str(value)
            for key, value in output_paths.items()
        },
    }

    with output_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[RESULT SUMMARY]")
    print(f"All route predictions        : {len(predicted)}")
    print(f"Balanced Top-R rows          : {len(balanced_topR)}")
    print(f"Potential Top-R rows         : {len(potential_topR)}")
    print(f"Union shortlist rows         : {len(shortlist)}")
    print(f"Final balanced Top-K rows    : {len(top_balanced)}")
    print(f"Final potential Top-K rows   : {len(top_potential)}")
    print(f"Final union rows             : {len(top_union)}")
    print(f"Quantile crossings detected  : {crossing_count}")

    print("\n[OUTPUTS]")
    for label, path in output_paths.items():
        print(f"{label:20s}: {path}")

    print("\n[OK] Candidate-route ranking completed.")
    print("=" * 116)


if __name__ == "__main__":
    main()