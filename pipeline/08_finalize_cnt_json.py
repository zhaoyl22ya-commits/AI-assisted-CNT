# -*- coding: utf-8 -*-
"""
08_finalize_cnt_json.py

07_image_merged -> 08_finalized

CNT 数据在 N/C/F/P 潜在向量提取前的最后一个确定性整理步骤。

本脚本只做确定性清洗，不调用任何 LLM，不覆盖 07_image_merged。

与历史流程保持一致的核心规则：
1) NP size 统一到 calc_block.np_size_nm；
2) 历史 alias 统一到后续 latent prompt 识别的 canonical 字段；
3) growth_quality_score* 尽量转为 float；
4) C_M_ratio / C_NP_ratio 只按历史固定 0.1 s 规则从 per_second 补齐；
   不使用 effective_time_s，也不使用 *_effective 作为 canonical ratio 来源；
5) L1 颗粒物理严格保留旧 cleanforL1.py 的 need_calc gate：
   仅当 C/NP 完全缺失，或 C/NP 明显等于 C/M（含 per-second）时，
   才计算 metal_atoms_per_np 并补/纠正 C/NP；
6) L1/L2 图像字段中 bundle_density_per_um == cnt_density_per_um 时
   删除 cnt_density_per_um；
7) 做 mode/level、synthesis_id、legacy 字段、数值合法性、幂等性和
   全局文件数量检查。

输入：
    data/07_image_merged/{A-E}/{L1-L3}/*.json

输出：
    data/08_finalized/{A-E}/{L1-L3}/*.json

日志：
    data/08_logs/finalize_cnt_json_file_log.csv
    data/08_logs/finalize_cnt_json_audit.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


# =============================================================================
# 1. PATHS / SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

INPUT_ROOT = DATA_ROOT / "07_image_merged"
OUTPUT_ROOT = DATA_ROOT / "08_finalized"
LOG_ROOT = DATA_ROOT / "08_logs"

MODES = ("A", "B", "C", "D", "E")
LEVELS = ("L1", "L2", "L3")

# 历史 cleanforscoreandteff.py 的固定有效时间定义。
T_EFFECTIVE_S = 0.1

# 图像补全后的确定性去重：只有数值完全相同时才删除 cnt_density。
DENSITY_EQUAL_TOL = 0.0

# 08_finalized 是可重复生成层，每次从 07_image_merged 全新重建。
REBUILD_OUTPUT = True

FILE_LOG_CSV = LOG_ROOT / "finalize_cnt_json_file_log.csv"
AUDIT_LOG_CSV = LOG_ROOT / "finalize_cnt_json_audit.csv"


# =============================================================================
# 2. HISTORICAL ELEMENT TABLE
#    density: kg/m^3
#    molar_mass: kg/mol
# =============================================================================

AVOGADRO = 6.022e23

ELEMENTS: Mapping[str, Mapping[str, float]] = {
    "Fe": {"density": 7874.0, "molar_mass": 0.055845},
    "Co": {"density": 8900.0, "molar_mass": 0.058933},
    "Ni": {"density": 8908.0, "molar_mass": 0.058693},
    "Mo": {"density": 10280.0, "molar_mass": 0.09595},
    "W":  {"density": 19250.0, "molar_mass": 0.18384},
    "Cr": {"density": 7190.0, "molar_mass": 0.051996},
    "Mn": {"density": 7470.0, "molar_mass": 0.054938},
    "V":  {"density": 6110.0, "molar_mass": 0.050942},
    "Ti": {"density": 4506.0, "molar_mass": 0.047867},
    "Cu": {"density": 8960.0, "molar_mass": 0.063546},
    "Au": {"density": 19320.0, "molar_mass": 0.196967},
    "Ag": {"density": 10490.0, "molar_mass": 0.1078682},
    "Pd": {"density": 12023.0, "molar_mass": 0.10642},
    "Pt": {"density": 21450.0, "molar_mass": 0.195084},
    "Al": {"density": 2700.0, "molar_mass": 0.0269815},
    "Mg": {"density": 1738.0, "molar_mass": 0.024305},
    "Zn": {"density": 7140.0, "molar_mass": 0.06538},
    "Sn": {"density": 7310.0, "molar_mass": 0.11871},
    "Pb": {"density": 11340.0, "molar_mass": 0.2072},
    "In": {"density": 7310.0, "molar_mass": 0.114818},
    "Y":  {"density": 4470.0, "molar_mass": 0.0889059},
    "La": {"density": 6146.0, "molar_mass": 0.138905},
    "Ce": {"density": 6770.0, "molar_mass": 0.140116},
    "Ta": {"density": 16690.0, "molar_mass": 0.180947},
    "Nb": {"density": 8570.0, "molar_mass": 0.092906},
    "Zr": {"density": 6520.0, "molar_mass": 0.091224},
    "Hf": {"density": 13310.0, "molar_mass": 0.17849},
}


# =============================================================================
# 3. AUDIT STRUCTURES
# =============================================================================

@dataclass
class SynthesisAudit:
    file: str
    mode: str
    level: str
    synthesis_id: str

    np_size_source: str = ""
    np_size_final_nm: Any = ""
    np_old_fields_removed: int = 0

    atom_alias_removed: int = 0
    c_np_alias_removed: int = 0

    score_fields_converted: int = 0
    c_m_static_from_rate: bool = False
    c_np_static_from_rate: bool = False

    l1_need_calc: bool = False
    metal_atoms_action: str = ""
    metal_atoms_final: Any = ""
    composition_assumption: str = ""
    c_np_action: str = ""
    c_np_final: Any = ""
    c_np_rate_action: str = ""
    c_np_rate_final: Any = ""

    cnt_density_removed_as_duplicate: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["warnings"] = " | ".join(self.warnings)
        return row


@dataclass
class FileSummary:
    status: str
    mode: str
    level: str
    source: str
    output: str

    syntheses_total: int = 0
    score_fields_converted: int = 0
    np_sizes_present: int = 0
    np_old_fields_removed: int = 0
    atom_aliases_removed: int = 0
    c_np_aliases_removed: int = 0
    c_m_static_from_rate: int = 0
    c_np_static_from_rate: int = 0
    l1_need_calc: int = 0
    metal_atoms_calculated: int = 0
    c_np_written_or_corrected: int = 0
    c_np_rate_written_or_corrected: int = 0
    duplicate_cnt_density_removed: int = 0
    warnings: int = 0
    idempotent: bool = False
    error: str = ""


# =============================================================================
# 4. BASIC HELPERS
# =============================================================================

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RANGE_RE = re.compile(
    rf"^\s*({_NUMBER})\s*(?:-|–|—|~|to)\s*({_NUMBER})\s*$",
    flags=re.IGNORECASE,
)


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def safe_float(value: Any, allow_range_midpoint: bool = False) -> float | None:
    """Convert a scalar numeric value to a finite float."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if allow_range_midpoint:
        match = _RANGE_RE.fullmatch(text)
        if match:
            midpoint = (float(match.group(1)) + float(match.group(2))) / 2.0
            return midpoint if math.isfinite(midpoint) else None

    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def numbers_equal(left: float | None, right: float | None, tol: float = 0.0) -> bool:
    return left is not None and right is not None and abs(left - right) <= tol


def ensure_dict(parent: MutableMapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    parent[key] = {}
    return parent[key]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be object.")
    if not isinstance(data.get("syntheses"), list):
        raise ValueError('JSON must contain a "syntheses" list.')
    return data


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    os.replace(temp_path, path)


def put_first(mapping: Mapping[str, Any], key: str, value: Any) -> dict[str, Any]:
    output: dict[str, Any] = OrderedDict()
    output[key] = value
    for old_key, old_value in mapping.items():
        if old_key != key:
            output[old_key] = old_value
    return output


# =============================================================================
# 5. NP SIZE CANONICALIZATION
# =============================================================================

NP_CATALYST_FIELDS = ("nanoparticle_size_nm", "catalyst_nanoparticle_size_nm")
NP_CALC_FIELDS = ("np_size_nm", "Average_size_M_nanoparticle")


def select_np_size(synthesis: Mapping[str, Any]) -> tuple[float | None, str]:
    """
    Historical priority:
      1) catalysts[].nanoparticle_size_nm
      2) catalysts[].catalyst_nanoparticle_size_nm
      3) calc_block.np_size_nm
      4) calc_block.Average_size_M_nanoparticle
    """
    catalysts = synthesis.get("catalysts")
    if isinstance(catalysts, list):
        for index, catalyst in enumerate(catalysts):
            if not isinstance(catalyst, dict):
                continue
            for field_name in NP_CATALYST_FIELDS:
                value = safe_float(catalyst.get(field_name), True)
                if value is not None and value > 0:
                    return value, f"catalysts[{index}].{field_name}"

    calc = synthesis.get("calc_block")
    if isinstance(calc, dict):
        for field_name in NP_CALC_FIELDS:
            value = safe_float(calc.get(field_name), True)
            if value is not None and value > 0:
                return value, f"calc_block.{field_name}"

    return None, ""


def normalize_np_size(synthesis: MutableMapping[str, Any], audit: SynthesisAudit) -> None:
    np_size, source = select_np_size(synthesis)
    removed = 0

    catalysts = synthesis.get("catalysts")
    if isinstance(catalysts, list):
        for catalyst in catalysts:
            if not isinstance(catalyst, dict):
                continue
            for field_name in NP_CATALYST_FIELDS:
                if field_name in catalyst:
                    catalyst.pop(field_name, None)
                    removed += 1

    calc = ensure_dict(synthesis, "calc_block")
    for field_name in NP_CALC_FIELDS:
        if field_name in calc:
            calc.pop(field_name, None)
            removed += 1

    final_value: Any = np_size if np_size is not None else ""
    synthesis["calc_block"] = put_first(calc, "np_size_nm", final_value)

    audit.np_size_source = source
    audit.np_size_final_nm = final_value
    audit.np_old_fields_removed = removed


# =============================================================================
# 6. LEGACY ALIAS CANONICALIZATION
# =============================================================================

ATOM_ALIASES = ("Metal_atoms_per_nanoparticle", "metal_atoms_per_nanoparticle")
C_NP_ALIASES = ("C_M_nanoparticle_ratio",)


def canonicalize_aliases(synthesis: MutableMapping[str, Any], audit: SynthesisAudit) -> None:
    """
    只统一名称，不重新估算数值。
    后续 z_C prompt 直接识别 metal_atoms_per_np / C_NP_ratio。
    """
    calc = ensure_dict(synthesis, "calc_block")

    if safe_float(calc.get("metal_atoms_per_np")) is None:
        for field_name in ATOM_ALIASES:
            value = safe_float(calc.get(field_name))
            if value is not None and value > 0:
                calc["metal_atoms_per_np"] = value
                break

    for field_name in ATOM_ALIASES:
        if field_name in calc:
            calc.pop(field_name, None)
            audit.atom_alias_removed += 1

    if safe_float(calc.get("C_NP_ratio")) is None:
        for field_name in C_NP_ALIASES:
            value = safe_float(calc.get(field_name))
            if value is not None:
                calc["C_NP_ratio"] = value
                break

    for field_name in C_NP_ALIASES:
        if field_name in calc:
            calc.pop(field_name, None)
            audit.c_np_alias_removed += 1


# =============================================================================
# 7. SCORE TYPE + HISTORICAL FIXED 0.1 s RATIO RULE
# =============================================================================

def normalize_scores_and_fixed_ratios(
    synthesis: MutableMapping[str, Any],
    audit: SynthesisAudit,
) -> None:
    """
    Strict historical behavior from 20251223-cleanforscoreandteff.py.

    Important:
    - Do NOT use effective_time_s.
    - Do NOT use C_M_ratio_effective / C_NP_ratio_effective as canonical sources.
    - Only if canonical key is absent:
        C_M_ratio = C_M_ratio_per_second * 0.1
        C_NP_ratio = C_M_nanoparticle_ratio_per_second * 0.1
    """
    calc = ensure_dict(synthesis, "calc_block")

    for key, old_value in list(calc.items()):
        if not key.startswith("growth_quality_score"):
            continue
        new_value = safe_float(old_value)
        if new_value is not None and (
            not isinstance(old_value, float) or old_value != new_value
        ):
            calc[key] = new_value
            audit.score_fields_converted += 1

    if "C_M_ratio" not in calc:
        rate = safe_float(calc.get("C_M_ratio_per_second"))
        if rate is not None:
            calc["C_M_ratio"] = rate * T_EFFECTIVE_S
            audit.c_m_static_from_rate = True

    if "C_NP_ratio" not in calc:
        rate = safe_float(calc.get("C_M_nanoparticle_ratio_per_second"))
        if rate is not None:
            calc["C_NP_ratio"] = rate * T_EFFECTIVE_S
            audit.c_np_static_from_rate = True


# =============================================================================
# 8. L1 PARTICLE PHYSICS — HISTORICAL need_calc GATE
# =============================================================================

def normalized_metals(
    metals: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    copied = [
        deepcopy(dict(metal))
        for metal in metals
        if isinstance(metal, dict)
    ]
    if not copied:
        return [], False

    values: list[float] = []
    valid = True
    for metal in copied:
        value = safe_float(metal.get("atomic_percentage"))
        if value is None or value < 0:
            valid = False
            break
        values.append(value)

    if valid and sum(values) > 0:
        for metal, value in zip(copied, values):
            metal["atomic_percentage"] = value
        return copied, False

    equal_value = 100.0 / len(copied)
    for metal in copied:
        metal["atomic_percentage"] = equal_value
    return copied, True


def compute_alloy_properties(
    metals: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, bool, str]:
    normalized, used_equal = normalized_metals(metals)
    if not normalized:
        return None, None, used_equal, "catalyst metals are empty"

    unknown = [
        str(metal.get("element", ""))
        for metal in normalized
        if str(metal.get("element", "")) not in ELEMENTS
    ]
    if unknown:
        return None, None, used_equal, "unknown element properties: " + ", ".join(unknown)

    total_atomic = sum(float(metal["atomic_percentage"]) for metal in normalized)
    if total_atomic <= 0:
        return None, None, used_equal, "total atomic percentage is not positive"

    atomic_fraction = {
        str(metal["element"]): float(metal["atomic_percentage"]) / total_atomic
        for metal in normalized
    }

    mean_molar_mass = sum(
        fraction * ELEMENTS[element]["molar_mass"]
        for element, fraction in atomic_fraction.items()
    )
    if mean_molar_mass <= 0:
        return None, None, used_equal, "mean molar mass is not positive"

    denominator = sum(
        atomic_fraction[element] * ELEMENTS[element]["molar_mass"]
        for element in atomic_fraction
    )
    mass_fraction = {
        element: atomic_fraction[element] * ELEMENTS[element]["molar_mass"] / denominator
        for element in atomic_fraction
    }
    mixed_density = sum(
        mass_fraction[element] * ELEMENTS[element]["density"]
        for element in mass_fraction
    )
    if mixed_density <= 0:
        return None, None, used_equal, "mixed density is not positive"

    return mixed_density, mean_molar_mass, used_equal, ""


def compute_metal_atoms_per_np(
    diameter_nm: float,
    metals: Sequence[Mapping[str, Any]],
) -> tuple[float | None, bool, str]:
    density, molar_mass, used_equal, error = compute_alloy_properties(metals)
    if density is None or molar_mass is None:
        return None, used_equal, error

    radius_m = diameter_nm * 1e-9 / 2.0
    volume = 4.0 / 3.0 * math.pi * radius_m**3
    atoms = density * volume / molar_mass * AVOGADRO

    if not math.isfinite(atoms) or atoms <= 0:
        return None, used_equal, "computed metal_atoms_per_np is invalid"

    return atoms, used_equal, ""


def first_catalyst_with_metals(
    synthesis: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, int]:
    catalysts = synthesis.get("catalysts")
    if not isinstance(catalysts, list):
        return None, -1

    for index, catalyst in enumerate(catalysts):
        if (
            isinstance(catalyst, dict)
            and isinstance(catalyst.get("metals"), list)
            and catalyst["metals"]
        ):
            return catalyst, index

    return None, -1


def reorder_calc_after_l1(calc: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Preserve the historical field placement convention after L1 correction."""
    np_size = calc.get("np_size_nm", "")
    remainder = OrderedDict(
        (key, value) for key, value in calc.items() if key != "np_size_nm"
    )
    output: dict[str, Any] = OrderedDict()
    output["np_size_nm"] = np_size

    if mode == "A":
        inserted = False
        for key, value in remainder.items():
            if key in ("metal_atoms_per_np", "C_NP_ratio"):
                continue
            output[key] = value
            if key == "C_M_ratio":
                if "metal_atoms_per_np" in remainder:
                    output["metal_atoms_per_np"] = remainder["metal_atoms_per_np"]
                if "C_NP_ratio" in remainder:
                    output["C_NP_ratio"] = remainder["C_NP_ratio"]
                inserted = True
        if not inserted:
            if "metal_atoms_per_np" in remainder:
                output["metal_atoms_per_np"] = remainder["metal_atoms_per_np"]
            if "C_NP_ratio" in remainder:
                output["C_NP_ratio"] = remainder["C_NP_ratio"]
        return output

    if mode in ("B", "C", "E"):
        inserted = False
        for key, value in remainder.items():
            if key in ("metal_atoms_per_np", "C_M_nanoparticle_ratio_per_second"):
                continue
            output[key] = value
            if key == "C_M_ratio_per_second":
                if "metal_atoms_per_np" in remainder:
                    output["metal_atoms_per_np"] = remainder["metal_atoms_per_np"]
                if "C_M_nanoparticle_ratio_per_second" in remainder:
                    output["C_M_nanoparticle_ratio_per_second"] = remainder[
                        "C_M_nanoparticle_ratio_per_second"
                    ]
                inserted = True
        if not inserted:
            if "metal_atoms_per_np" in remainder:
                output["metal_atoms_per_np"] = remainder["metal_atoms_per_np"]
            if "C_M_nanoparticle_ratio_per_second" in remainder:
                output["C_M_nanoparticle_ratio_per_second"] = remainder[
                    "C_M_nanoparticle_ratio_per_second"
                ]
        return output

    # D: apart from keeping np_size first, retain the remaining order.
    output.update(remainder)
    return output


def apply_l1_particle_physics(
    synthesis: MutableMapping[str, Any],
    mode: str,
    audit: SynthesisAudit,
) -> None:
    """
    Historical cleanforL1.py gate, adapted only for the new schema where
    NP size has already been moved to calc_block.np_size_nm.

    need_calc = True only when:
      1) both C_NP_ratio and C_M_nanoparticle_ratio_per_second are missing; or
      2) C_NP_ratio == C_M_ratio; or
      3) C_M_nanoparticle_ratio_per_second == C_M_ratio_per_second.

    If need_calc is False, this function does NOT create metal_atoms_per_np.
    """
    calc = ensure_dict(synthesis, "calc_block")

    c_m = safe_float(calc.get("C_M_ratio"))
    c_m_rate = safe_float(calc.get("C_M_ratio_per_second"))

    # Historical Gate 1: no numerical C/M input -> do nothing.
    if c_m is None and c_m_rate is None:
        audit.metal_atoms_action = "not_applicable_no_C_M"
        return

    c_np = safe_float(calc.get("C_NP_ratio"))
    c_np_rate = safe_float(calc.get("C_M_nanoparticle_ratio_per_second"))

    need_calc = False
    if c_np is None and c_np_rate is None:
        need_calc = True
    elif c_np is not None and c_m is not None and numbers_equal(c_np, c_m, 1e-12):
        need_calc = True
    elif (
        c_np_rate is not None
        and c_m_rate is not None
        and numbers_equal(c_np_rate, c_m_rate, 1e-12)
    ):
        need_calc = True

    audit.l1_need_calc = need_calc

    # This is the key historical-compatibility gate.
    if not need_calc:
        existing_atoms = safe_float(calc.get("metal_atoms_per_np"))
        audit.metal_atoms_action = (
            "preserved_existing" if existing_atoms is not None else "not_needed"
        )
        audit.metal_atoms_final = existing_atoms if existing_atoms is not None else ""
        audit.c_np_action = "preserved_existing" if c_np is not None else ""
        audit.c_np_final = c_np if c_np is not None else ""
        audit.c_np_rate_action = "preserved_existing" if c_np_rate is not None else ""
        audit.c_np_rate_final = c_np_rate if c_np_rate is not None else ""
        return

    np_size = safe_float(calc.get("np_size_nm"), allow_range_midpoint=True)
    if np_size is None or np_size <= 0:
        audit.metal_atoms_action = "skipped_missing_np_size"
        audit.warnings.append(
            "historical need_calc=True but particle physics skipped: missing valid np_size_nm"
        )
        return

    catalyst, catalyst_index = first_catalyst_with_metals(synthesis)
    if catalyst is None:
        audit.metal_atoms_action = "skipped_missing_metals"
        audit.warnings.append(
            "historical need_calc=True but particle physics skipped: no catalyst with metals"
        )
        return

    metals = catalyst.get("metals")
    assert isinstance(metals, list)

    atoms, used_equal, error = compute_metal_atoms_per_np(np_size, metals)
    if atoms is None:
        audit.metal_atoms_action = "skipped_calculation_error"
        audit.warnings.append(
            f"particle physics failed for catalysts[{catalyst_index}]: {error}"
        )
        return

    # Historical behavior: once need_calc=True, metal_atoms_per_np is recalculated.
    calc["metal_atoms_per_np"] = atoms
    calc["metal_atoms_per_np_method"] = "sphere_alloy_density"
    calc["derived"] = True
    audit.metal_atoms_action = "calculated_under_historical_gate"
    audit.metal_atoms_final = atoms

    if used_equal:
        calc["composition_assumption"] = "equal_atomic_fraction"
        audit.composition_assumption = "equal_atomic_fraction"

    if c_m is not None:
        new_c_np = c_m * atoms
        calc["C_NP_ratio"] = new_c_np
        audit.c_np_action = (
            "calculated_missing"
            if c_np is None
            else "corrected_equal_to_C_M"
        )
        audit.c_np_final = new_c_np

    if c_m_rate is not None:
        old_rate = safe_float(calc.get("C_M_nanoparticle_ratio_per_second"))

        # Historical behavior:
        # missing / zero -> fill;
        # equal to C/M/s -> correct;
        # otherwise preserve.
        if old_rate is None or abs(old_rate) < 1e-20:
            new_rate = c_m_rate * atoms
            calc["C_M_nanoparticle_ratio_per_second"] = new_rate
            audit.c_np_rate_action = (
                "calculated_missing" if old_rate is None else "corrected_zero"
            )
            audit.c_np_rate_final = new_rate
        elif numbers_equal(old_rate, c_m_rate, 1e-12):
            new_rate = c_m_rate * atoms
            calc["C_M_nanoparticle_ratio_per_second"] = new_rate
            audit.c_np_rate_action = "corrected_equal_to_C_M_per_second"
            audit.c_np_rate_final = new_rate
        else:
            audit.c_np_rate_action = "preserved_existing"
            audit.c_np_rate_final = old_rate

    synthesis["calc_block"] = reorder_calc_after_l1(calc, mode)


# =============================================================================
# 9. IMAGE DENSITY DUPLICATE
# =============================================================================

def remove_duplicate_density(
    synthesis: MutableMapping[str, Any],
    audit: SynthesisAudit,
) -> None:
    product = synthesis.get("product")
    if not isinstance(product, dict):
        return

    bundle_density = safe_float(product.get("bundle_density_per_um"))
    cnt_density = safe_float(product.get("cnt_density_per_um"))

    if (
        numbers_equal(bundle_density, cnt_density, DENSITY_EQUAL_TOL)
        and "cnt_density_per_um" in product
    ):
        product.pop("cnt_density_per_um", None)
        audit.cnt_density_removed_as_duplicate = True


# =============================================================================
# 10. PROCESS ONE SYNTHESIS / DOCUMENT
# =============================================================================

def process_synthesis(
    synthesis: MutableMapping[str, Any],
    file_name: str,
    mode: str,
    level: str,
    synthesis_index: int,
) -> SynthesisAudit:
    synthesis_id = synthesis.get("synthesis_id")
    audit_id = (
        synthesis_id
        if isinstance(synthesis_id, str) and synthesis_id.strip()
        else f"__index_{synthesis_index}"
    )

    audit = SynthesisAudit(
        file=file_name,
        mode=mode,
        level=level,
        synthesis_id=audit_id,
    )

    # Fixed order:
    # 1) NP size
    # 2) legacy aliases
    # 3) scores + historical fixed 0.1 s ratio completion
    # 4) historical L1 particle-physics gate
    # 5) image-density duplicate cleanup
    normalize_np_size(synthesis, audit)
    canonicalize_aliases(synthesis, audit)
    normalize_scores_and_fixed_ratios(synthesis, audit)

    if level == "L1":
        apply_l1_particle_physics(synthesis, mode, audit)

    if level in ("L1", "L2"):
        remove_duplicate_density(synthesis, audit)

    calc = ensure_dict(synthesis, "calc_block")
    synthesis["calc_block"] = put_first(
        calc,
        "np_size_nm",
        calc.get("np_size_nm", ""),
    )
    return audit


def process_document(
    data: dict[str, Any],
    file_name: str,
    mode: str,
    level: str,
) -> list[SynthesisAudit]:
    syntheses = data.get("syntheses")
    if not isinstance(syntheses, list):
        raise ValueError('JSON must contain a "syntheses" list.')

    audits: list[SynthesisAudit] = []
    for index, synthesis in enumerate(syntheses):
        if not isinstance(synthesis, dict):
            raise ValueError(f"syntheses[{index}] is not object.")
        audits.append(
            process_synthesis(
                synthesis,
                file_name,
                mode,
                level,
                index,
            )
        )
    return audits


# =============================================================================
# 11. FINAL GATE
# =============================================================================

FORBIDDEN_CALC_FIELDS = {
    "Average_size_M_nanoparticle",
    "Metal_atoms_per_nanoparticle",
    "metal_atoms_per_nanoparticle",
    "C_M_nanoparticle_ratio",
}

KNOWN_RATIO_FIELDS = (
    "C_M_ratio",
    "C_NP_ratio",
    "C_M_ratio_per_second",
    "C_M_nanoparticle_ratio_per_second",
)

QWEN_PRODUCT_FIELDS = (
    "length_mm",
    "diameter_nm",
    "orientation_score",
    "ig_id_ratio",
    "purity_score",
    "bundle_density_per_um",
    "cnt_density_per_um",
)


def validate_document(
    data: Mapping[str, Any],
    mode: str,
    level: str,
) -> list[str]:
    warnings: list[str] = []

    json_mode = data.get("growth_mode")
    if (
        isinstance(json_mode, str)
        and json_mode.strip()
        and json_mode.strip() != mode
    ):
        raise ValueError(
            f"growth_mode mismatch: JSON={json_mode!r}, folder={mode!r}"
        )

    json_level = data.get("data_level")
    if (
        isinstance(json_level, str)
        and json_level.strip()
        and json_level.strip() != level
    ):
        raise ValueError(
            f"data_level mismatch: JSON={json_level!r}, folder={level!r}"
        )

    syntheses = data.get("syntheses")
    if not isinstance(syntheses, list):
        raise ValueError('JSON must contain a "syntheses" list.')

    seen_ids: set[str] = set()

    for index, synthesis in enumerate(syntheses):
        if not isinstance(synthesis, dict):
            raise ValueError(f"syntheses[{index}] is not object.")

        synthesis_id = synthesis.get("synthesis_id")
        if not isinstance(synthesis_id, str) or not synthesis_id.strip():
            raise ValueError(f"syntheses[{index}] missing synthesis_id.")

        synthesis_id = synthesis_id.strip()
        if synthesis_id in seen_ids:
            raise ValueError(f"duplicate synthesis_id: {synthesis_id}")
        seen_ids.add(synthesis_id)

        catalysts = synthesis.get("catalysts")
        if isinstance(catalysts, list):
            for catalyst_index, catalyst in enumerate(catalysts):
                if not isinstance(catalyst, dict):
                    continue
                leftovers = set(NP_CATALYST_FIELDS).intersection(catalyst)
                if leftovers:
                    raise ValueError(
                        f"{synthesis_id}: legacy catalyst NP fields remain in "
                        f"catalysts[{catalyst_index}]: {sorted(leftovers)}"
                    )

        calc = synthesis.get("calc_block")
        if not isinstance(calc, dict):
            raise ValueError(f"{synthesis_id}: calc_block is not object.")

        if "np_size_nm" not in calc:
            raise ValueError(f"{synthesis_id}: calc_block.np_size_nm missing.")

        leftovers = FORBIDDEN_CALC_FIELDS.intersection(calc)
        if leftovers:
            raise ValueError(
                f"{synthesis_id}: legacy calc fields remain: {sorted(leftovers)}"
            )

        np_size = calc.get("np_size_nm")
        if not is_blank(np_size):
            number = safe_float(np_size)
            if number is None or number <= 0:
                raise ValueError(
                    f"{synthesis_id}: invalid np_size_nm={np_size!r}"
                )

        atoms = calc.get("metal_atoms_per_np")
        if not is_blank(atoms):
            number = safe_float(atoms)
            if number is None or number <= 0:
                raise ValueError(
                    f"{synthesis_id}: invalid metal_atoms_per_np={atoms!r}"
                )

        # Preserve unusual historical values, but surface them in the audit.
        for field_name in KNOWN_RATIO_FIELDS:
            if (
                field_name in calc
                and not is_blank(calc.get(field_name))
                and safe_float(calc.get(field_name)) is None
            ):
                warnings.append(
                    f"{synthesis_id}: non-numeric {field_name} preserved: "
                    f"{calc.get(field_name)!r}"
                )

        product = synthesis.get("product")
        if isinstance(product, dict):
            for field_name in QWEN_PRODUCT_FIELDS:
                if (
                    field_name in product
                    and not is_blank(product.get(field_name))
                    and safe_float(product.get(field_name)) is None
                ):
                    warnings.append(
                        f"{synthesis_id}: non-numeric product.{field_name} preserved: "
                        f"{product.get(field_name)!r}"
                    )

    return warnings


def assert_idempotent(
    data: dict[str, Any],
    file_name: str,
    mode: str,
    level: str,
) -> None:
    second_pass = deepcopy(data)
    process_document(second_pass, file_name, mode, level)
    if second_pass != data:
        raise ValueError("finalization is not idempotent.")


# =============================================================================
# 12. ONE FILE
# =============================================================================

def process_one_file(
    source_path: Path,
    output_path: Path,
    mode: str,
    level: str,
) -> tuple[FileSummary, list[SynthesisAudit]]:
    summary = FileSummary(
        status="OK",
        mode=mode,
        level=level,
        source=str(source_path),
        output=str(output_path),
    )
    audits: list[SynthesisAudit] = []

    try:
        data = load_json(source_path)
        synthesis_count_before = len(data["syntheses"])

        audits = process_document(data, source_path.name, mode, level)

        synthesis_count_after = len(data["syntheses"])
        if synthesis_count_after != synthesis_count_before:
            raise ValueError("synthesis count changed.")

        gate_warnings = validate_document(data, mode, level)
        if gate_warnings and audits:
            audits[0].warnings.extend(gate_warnings)

        assert_idempotent(data, source_path.name, mode, level)
        summary.idempotent = True

        atomic_write_json(output_path, data)

        summary.syntheses_total = synthesis_count_after
        summary.score_fields_converted = sum(a.score_fields_converted for a in audits)
        summary.np_sizes_present = sum(a.np_size_final_nm != "" for a in audits)
        summary.np_old_fields_removed = sum(a.np_old_fields_removed for a in audits)
        summary.atom_aliases_removed = sum(a.atom_alias_removed for a in audits)
        summary.c_np_aliases_removed = sum(a.c_np_alias_removed for a in audits)
        summary.c_m_static_from_rate = sum(a.c_m_static_from_rate for a in audits)
        summary.c_np_static_from_rate = sum(a.c_np_static_from_rate for a in audits)
        summary.l1_need_calc = sum(a.l1_need_calc for a in audits)
        summary.metal_atoms_calculated = sum(
            a.metal_atoms_action == "calculated_under_historical_gate"
            for a in audits
        )
        summary.c_np_written_or_corrected = sum(
            a.c_np_action in ("calculated_missing", "corrected_equal_to_C_M")
            for a in audits
        )
        summary.c_np_rate_written_or_corrected = sum(
            a.c_np_rate_action
            in (
                "calculated_missing",
                "corrected_zero",
                "corrected_equal_to_C_M_per_second",
            )
            for a in audits
        )
        summary.duplicate_cnt_density_removed = sum(
            a.cnt_density_removed_as_duplicate for a in audits
        )
        summary.warnings = sum(len(a.warnings) for a in audits)

    except Exception as exc:
        summary.status = "FAIL"
        summary.error = f"{type(exc).__name__}: {exc}"

    return summary, audits


# =============================================================================
# 13. LOGGING / MAIN
# =============================================================================

FILE_LOG_FIELDS = list(FileSummary.__dataclass_fields__)
AUDIT_LOG_FIELDS = [
    name for name in SynthesisAudit.__dataclass_fields__ if name != "warnings"
] + ["warnings"]


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def count_json(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.json")) if root.exists() else 0


def main() -> None:
    print("=== CNT JSON Finalization: 07_image_merged -> 08_finalized ===")
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"INPUT_ROOT   : {INPUT_ROOT}")
    print(f"OUTPUT_ROOT  : {OUTPUT_ROOT}")
    print(f"LOG_ROOT     : {LOG_ROOT}")

    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {INPUT_ROOT}")

    if REBUILD_OUTPUT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    input_file_count = count_json(INPUT_ROOT)
    if input_file_count == 0:
        raise RuntimeError(f"No input JSON found under {INPUT_ROOT}")

    summaries: list[FileSummary] = []
    audits: list[SynthesisAudit] = []
    discovered_files = 0
    successful_files = 0
    failed_files = 0

    for mode in MODES:
        for level in LEVELS:
            source_dir = INPUT_ROOT / mode / level
            output_dir = OUTPUT_ROOT / mode / level

            source_files = (
                sorted(source_dir.glob("*.json"), key=lambda p: p.name.casefold())
                if source_dir.is_dir()
                else []
            )

            print(f"\n[{mode}/{level}] FILES={len(source_files)}")

            for index, source_path in enumerate(source_files, start=1):
                discovered_files += 1
                output_path = output_dir / source_path.name

                summary, one_file_audits = process_one_file(
                    source_path,
                    output_path,
                    mode,
                    level,
                )
                summaries.append(summary)
                audits.extend(one_file_audits)

                if summary.status == "OK":
                    successful_files += 1
                    print(
                        f"[OK {index}/{len(source_files)}] {source_path.name} | "
                        f"syn={summary.syntheses_total} "
                        f"np={summary.np_sizes_present} "
                        f"CM0.1={summary.c_m_static_from_rate} "
                        f"CNP0.1={summary.c_np_static_from_rate} "
                        f"need_calc={summary.l1_need_calc} "
                        f"atoms={summary.metal_atoms_calculated} "
                        f"density_removed={summary.duplicate_cnt_density_removed} "
                        f"warn={summary.warnings}"
                    )
                else:
                    failed_files += 1
                    print(
                        f"[FAIL {index}/{len(source_files)}] "
                        f"{source_path.name}: {summary.error}"
                    )

    write_csv(
        FILE_LOG_CSV,
        FILE_LOG_FIELDS,
        (asdict(summary) for summary in summaries),
    )
    write_csv(
        AUDIT_LOG_CSV,
        AUDIT_LOG_FIELDS,
        (audit.to_row() for audit in audits),
    )

    output_file_count = count_json(OUTPUT_ROOT)

    print("\n" + "=" * 100)
    print("FINAL GLOBAL GATE")
    print(f"INPUT FILES      : {input_file_count}")
    print(f"DISCOVERED FILES : {discovered_files}")
    print(f"SUCCESSFUL FILES : {successful_files}")
    print(f"FAILED FILES     : {failed_files}")
    print(f"OUTPUT FILES     : {output_file_count}")
    print(f"FILE LOG         : {FILE_LOG_CSV}")
    print(f"AUDIT LOG        : {AUDIT_LOG_CSV}")
    print("=" * 100)

    errors: list[str] = []

    if discovered_files != input_file_count:
        errors.append(
            f"discovery mismatch: {discovered_files} != {input_file_count}"
        )
    if failed_files:
        errors.append(f"{failed_files} file(s) failed")
    if successful_files != input_file_count:
        errors.append(
            f"success mismatch: {successful_files} != {input_file_count}"
        )
    if output_file_count != input_file_count:
        errors.append(
            f"output mismatch: {output_file_count} != {input_file_count}"
        )

    if errors:
        print("[FINAL STATUS] FAIL")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError(
            "Finalization gate failed. Check data/08_logs before latent extraction."
        )

    print("[FINAL STATUS] PASS")
    print("08_finalized is ready for N/C/F/P latent-vector extraction.")


if __name__ == "__main__":
    main()
