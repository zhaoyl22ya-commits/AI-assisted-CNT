# -*- coding: utf-8 -*-
# 11_flatten_dataset.py
#
# 扫描 data/10_andjson 下 A-E × L1-L3/*.andjson，生成：
# 1) ml_dataset_3p0_full60D.csv
# 2) build_report.csv
# 3) dataset_manifest.csv
# 4) feature_schema.json
#
# 保留原逻辑：
# - N1-N18、C1-C12、F1-F18、P1-P12
# - miss_* 与 y_*
# - log1p: N12、C10、C11、F6、F11、F12、F18
#
# 新增：
# - paper_year_int、paper_id、sample_id
# - 数据清单与基础审计
#
# 注意：新增元数据不能作为模型特征。

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 1. 配置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

INPUT_ROOT = DATA_ROOT / "10_andjson"
OUTPUT_ROOT = DATA_ROOT / "11_flattened"

GROWTH_MODES = ["A", "B", "C", "D", "E"]
LEVELS = ["L1", "L2", "L3"]

OUT_CSV = str(OUTPUT_ROOT / "ml_dataset_3p0_full60D.csv")
REPORT_CSV = str(OUTPUT_ROOT / "build_report.csv")
MANIFEST_CSV = str(OUTPUT_ROOT / "dataset_manifest.csv")
SCHEMA_JSON = str(OUTPUT_ROOT / "feature_schema.json")

DIM_SPECS = {"N": 18, "C": 12, "F": 18, "P": 12}

DIM_PATTERNS = {
    "N": re.compile(r"^N(\d+)\b|^N(\d+)_", re.IGNORECASE),
    "C": re.compile(r"^C(\d+)\b|^C(\d+)_", re.IGNORECASE),
    "F": re.compile(r"^F(\d+)\b|^F(\d+)_", re.IGNORECASE),
    "P": re.compile(r"^P(\d+)\b|^P(\d+)_", re.IGNORECASE),
}

ZBLOCK_KEYS = {
    "N": ["z_nuc"],
    "C": ["z_C", "z_c", "zc", "zC"],
    "F": ["z_flow"],
    "P": ["z_proc"],
}

LOG1P_TARGETS = {
    "N": {"N12"},
    "C": {"C10", "C11"},
    "F": {"F6", "F11", "F12", "F18"},
}


# =============================================================================
# 2. 基础工具
# =============================================================================

def safe_load_json(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "json_root_not_object"
        return data, None
    except FileNotFoundError:
        return None, "not_found"
    except json.JSONDecodeError as e:
        return None, f"decode_error: {e}"
    except Exception as e:
        return None, f"read_error: {type(e).__name__}: {e}"


def to_float_maybe(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x.strip())
            return v if math.isfinite(v) else None
        except Exception:
            return None
    return None


def extract_numeric_value(obj: Any) -> Optional[float]:
    v = to_float_maybe(obj)
    if v is not None:
        return v

    if isinstance(obj, dict):
        if "value" in obj:
            v = extract_numeric_value(obj.get("value"))
            if v is not None:
                return v

        if "atomic_fraction" in obj:
            v = to_float_maybe(obj.get("atomic_fraction"))
            if v is not None:
                return v

        for key in ("score", "mean", "median"):
            if key in obj:
                v = to_float_maybe(obj.get(key))
                if v is not None:
                    return v

        for item in obj.values():
            v = extract_numeric_value(item)
            if v is not None:
                return v

    if isinstance(obj, list):
        for item in obj:
            v = extract_numeric_value(item)
            if v is not None:
                return v

    return None


def get_first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in d:
            return d.get(key)

    lower_map = {str(key).lower(): key for key in d.keys()}
    for key in keys:
        actual = lower_map.get(str(key).lower())
        if actual is not None:
            return d.get(actual)

    return None


def normalize_text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).strip().split())


def normalize_doi(value: Any) -> str:
    text = normalize_text(value).lower()
    if not text:
        return ""

    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    return text.replace(" ", "").strip(" .;,/")


def normalize_year(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        year = value
    elif isinstance(value, float) and value.is_integer():
        year = int(value)
    else:
        match = re.search(r"(?<!\d)((?:18|19|20|21)\d{2})(?!\d)", normalize_text(value))
        if not match:
            return None
        year = int(match.group(1))

    return year if 1800 <= year <= 2199 else None


def sanitize_id(text: str, max_length: int = 100) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:max_length]


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def make_paper_id(paper: Dict[str, Any], source_file: str) -> Tuple[str, str]:
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return f"doi_{sanitize_id(doi)}", "doi"

    title = normalize_text(paper.get("title"))
    year = normalize_year(paper.get("year"))
    if title:
        key = f"{title.lower()}|{year or ''}"
        title_id = sanitize_id(title, 60)
        if year is not None:
            return f"title_{year}_{title_id}_{short_hash(key)}", "title_year"
        return f"title_{title_id}_{short_hash(key)}", "title"

    stem = os.path.splitext(source_file)[0]
    return f"source_{sanitize_id(stem)}", "source_file"


def make_sample_id(paper_id: str, synthesis_id: Any, row_index: int) -> str:
    text = normalize_text(synthesis_id)
    token = sanitize_id(text, 60) if text else f"missing_synthesis_id_{row_index + 1}"
    return f"{paper_id}__{token}"


def has_numeric_target(row: Dict[str, Any]) -> bool:
    return any(
        key.startswith("y_") and to_float_maybe(value) is not None
        for key, value in row.items()
        if isinstance(key, str)
    )


# =============================================================================
# 3. 数据展开
# =============================================================================

def flatten_zblock(
    zblock: Any,
    dim_prefix: str,
    max_dim: int,
) -> Dict[str, Optional[float]]:
    out = {f"{dim_prefix}{i}": None for i in range(1, max_dim + 1)}
    if not isinstance(zblock, dict):
        return out

    pattern = DIM_PATTERNS[dim_prefix]

    for key, value in zblock.items():
        if not isinstance(key, str):
            continue

        match = pattern.match(key.strip())
        if not match:
            continue

        idx_text = match.group(1) or match.group(2)
        if not idx_text:
            continue

        try:
            idx = int(idx_text)
        except Exception:
            continue

        if 1 <= idx <= max_dim:
            out[f"{dim_prefix}{idx}"] = extract_numeric_value(value)

    return out


def flatten_product(product: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(product, dict):
        return out

    for key, value in product.items():
        if not isinstance(key, str):
            continue

        column = f"y_{key}"
        numeric = extract_numeric_value(value)

        if numeric is not None:
            out[column] = numeric
        elif isinstance(value, (str, int, float)) or value is None:
            out[column] = value
        else:
            try:
                out[column] = json.dumps(value, ensure_ascii=False)
            except Exception:
                out[column] = str(value)

    return out


def iter_andjson_files() -> List[str]:
    files: List[str] = []

    for mode in GROWTH_MODES:
        for level in LEVELS:
            folder = INPUT_ROOT / mode / level
            if not folder.is_dir():
                continue

            for path in folder.iterdir():
                if path.is_file() and path.name.lower().endswith(".andjson"):
                    files.append(str(path))

    return sorted(files)


def safe_log1p(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return value if value < 0 else math.log1p(value)


def apply_log1p_targets(
    values: Dict[str, Optional[float]],
    targets: set,
) -> Dict[str, Optional[float]]:
    out = dict(values)
    for key in targets:
        if key in out:
            out[key] = safe_log1p(out[key])
    return out


def write_csv(path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


# =============================================================================
# 4. 主程序
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    and_files = iter_andjson_files()

    x_cols = [
        f"{prefix}{i}"
        for prefix in ("N", "C", "F", "P")
        for i in range(1, DIM_SPECS[prefix] + 1)
    ]
    miss_cols = [f"miss_{column}" for column in x_cols]

    meta_cols = [
        "paper_doi",
        "paper_title",
        "paper_year",
        "paper_year_int",
        "paper_journal",
        "growth_mode",
        "data_level",
        "synthesis_id",
        "source_file",
        "paper_id",
        "paper_id_method",
        "sample_id",
    ]

    rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Dict[str, Any]] = {}
    global_sample_counter: Counter = Counter()

    for path in and_files:
        document, error = safe_load_json(path)

        report = {
            "file": os.path.basename(path),
            "status": "ok" if document is not None else "bad",
            "error": "" if error is None else error,
            "paper_id": "",
            "paper_year_missing": 0,
            "syntheses_total": 0,
            "rows_emitted": 0,
            "duplicate_synthesis_ids": 0,
            "duplicate_sample_ids": 0,
            "missing_synthesis_ids": 0,
            "missing_z_nuc": 0,
            "missing_z_C": 0,
            "missing_z_flow": 0,
            "missing_z_proc": 0,
            "total_missing_features": 0,
            "mean_missing_features": "",
            "rows_without_numeric_targets": 0,
        }

        if document is None:
            report_rows.append(report)
            continue

        paper = document.get("paper", {})
        if not isinstance(paper, dict):
            paper = {}

        growth_mode = document.get("growth_mode")
        data_level = document.get("data_level")
        source_file = os.path.basename(path)

        paper_id, paper_id_method = make_paper_id(paper, source_file)
        paper_year_int = normalize_year(paper.get("year"))

        report["paper_id"] = paper_id
        report["paper_year_missing"] = int(paper_year_int is None)

        if paper_id not in manifest:
            manifest[paper_id] = {
                "paper_id": paper_id,
                "paper_id_method": paper_id_method,
                "paper_doi": normalize_doi(paper.get("doi")),
                "paper_title": paper.get("title"),
                "paper_year": paper.get("year"),
                "paper_year_int": paper_year_int,
                "paper_journal": paper.get("journal"),
                "source_files": set(),
                "growth_modes": set(),
                "data_levels": set(),
                "synthesis_count": 0,
                "rows_with_numeric_targets": 0,
                "rows_without_numeric_targets": 0,
                "total_missing_features": 0,
            }

        manifest_entry = manifest[paper_id]
        manifest_entry["source_files"].add(source_file)
        manifest_entry["growth_modes"].add(str(growth_mode))
        manifest_entry["data_levels"].add(str(data_level))

        syntheses = document.get("syntheses", [])
        if not isinstance(syntheses, list):
            report["status"] = "bad"
            report["error"] = "syntheses_not_list"
            report_rows.append(report)
            continue

        report["syntheses_total"] = len(syntheses)
        synthesis_counter: Counter = Counter()

        for row_index, synthesis in enumerate(syntheses):
            if not isinstance(synthesis, dict):
                continue

            synthesis_id = synthesis.get("synthesis_id")
            synthesis_text = normalize_text(synthesis_id)

            if synthesis_text:
                synthesis_counter[synthesis_text] += 1
            else:
                report["missing_synthesis_ids"] += 1

            sample_id_base = make_sample_id(paper_id, synthesis_id, row_index)
            global_sample_counter[sample_id_base] += 1
            duplicate_number = global_sample_counter[sample_id_base]

            sample_id = (
                sample_id_base
                if duplicate_number == 1
                else f"{sample_id_base}__duplicate_{duplicate_number}"
            )
            if duplicate_number > 1:
                report["duplicate_sample_ids"] += 1

            z_nuc = get_first_present(synthesis, ZBLOCK_KEYS["N"])
            z_c = get_first_present(synthesis, ZBLOCK_KEYS["C"])
            z_flow = get_first_present(synthesis, ZBLOCK_KEYS["F"])
            z_proc = get_first_present(synthesis, ZBLOCK_KEYS["P"])

            report["missing_z_nuc"] += int(not isinstance(z_nuc, dict))
            report["missing_z_C"] += int(not isinstance(z_c, dict))
            report["missing_z_flow"] += int(not isinstance(z_flow, dict))
            report["missing_z_proc"] += int(not isinstance(z_proc, dict))

            n_values = apply_log1p_targets(
                flatten_zblock(z_nuc, "N", DIM_SPECS["N"]),
                LOG1P_TARGETS["N"],
            )
            c_values = apply_log1p_targets(
                flatten_zblock(z_c, "C", DIM_SPECS["C"]),
                LOG1P_TARGETS["C"],
            )
            f_values = apply_log1p_targets(
                flatten_zblock(z_flow, "F", DIM_SPECS["F"]),
                LOG1P_TARGETS["F"],
            )
            p_values = flatten_zblock(z_proc, "P", DIM_SPECS["P"])

            row: Dict[str, Any] = {
                "paper_doi": paper.get("doi"),
                "paper_title": paper.get("title"),
                "paper_year": paper.get("year"),
                "paper_year_int": paper_year_int,
                "paper_journal": paper.get("journal"),
                "growth_mode": growth_mode,
                "data_level": data_level,
                "synthesis_id": synthesis_id,
                "source_file": source_file,
                "paper_id": paper_id,
                "paper_id_method": paper_id_method,
                "sample_id": sample_id,
            }

            row.update(n_values)
            row.update(c_values)
            row.update(f_values)
            row.update(p_values)

            missing_count = 0
            for column in x_cols:
                missing = int(row.get(column) is None)
                row[f"miss_{column}"] = missing
                missing_count += missing

            report["total_missing_features"] += missing_count
            row.update(flatten_product(synthesis.get("product")))

            if has_numeric_target(row):
                manifest_entry["rows_with_numeric_targets"] += 1
            else:
                report["rows_without_numeric_targets"] += 1
                manifest_entry["rows_without_numeric_targets"] += 1

            rows.append(row)
            report["rows_emitted"] += 1
            manifest_entry["synthesis_count"] += 1
            manifest_entry["total_missing_features"] += missing_count

        report["duplicate_synthesis_ids"] = sum(
            count - 1
            for count in synthesis_counter.values()
            if count > 1
        )

        if report["rows_emitted"] > 0:
            report["mean_missing_features"] = (
                report["total_missing_features"] / report["rows_emitted"]
            )

        report_rows.append(report)

    y_cols = sorted({
        key
        for row in rows
        for key in row
        if isinstance(key, str) and key.startswith("y_")
    })

    write_csv(
        OUT_CSV,
        meta_cols + x_cols + miss_cols + y_cols,
        rows,
    )

    report_header = [
        "file",
        "status",
        "error",
        "paper_id",
        "paper_year_missing",
        "syntheses_total",
        "rows_emitted",
        "duplicate_synthesis_ids",
        "duplicate_sample_ids",
        "missing_synthesis_ids",
        "missing_z_nuc",
        "missing_z_C",
        "missing_z_flow",
        "missing_z_proc",
        "total_missing_features",
        "mean_missing_features",
        "rows_without_numeric_targets",
    ]
    write_csv(REPORT_CSV, report_header, report_rows)

    manifest_rows: List[Dict[str, Any]] = []
    for paper_id in sorted(manifest):
        item = manifest[paper_id]
        count = item["synthesis_count"]

        manifest_rows.append({
            "paper_id": item["paper_id"],
            "paper_id_method": item["paper_id_method"],
            "paper_doi": item["paper_doi"],
            "paper_title": item["paper_title"],
            "paper_year": item["paper_year"],
            "paper_year_int": item["paper_year_int"],
            "paper_journal": item["paper_journal"],
            "source_files": " | ".join(sorted(item["source_files"])),
            "growth_modes": " | ".join(sorted(item["growth_modes"])),
            "data_levels": " | ".join(sorted(item["data_levels"])),
            "synthesis_count": count,
            "rows_with_numeric_targets": item["rows_with_numeric_targets"],
            "rows_without_numeric_targets": item["rows_without_numeric_targets"],
            "mean_missing_features": (
                item["total_missing_features"] / count if count > 0 else ""
            ),
        })

    manifest_header = [
        "paper_id",
        "paper_id_method",
        "paper_doi",
        "paper_title",
        "paper_year",
        "paper_year_int",
        "paper_journal",
        "source_files",
        "growth_modes",
        "data_levels",
        "synthesis_count",
        "rows_with_numeric_targets",
        "rows_without_numeric_targets",
        "mean_missing_features",
    ]
    write_csv(MANIFEST_CSV, manifest_header, manifest_rows)

    feature_schema = {
        "metadata_columns": meta_cols,
        "physical_feature_columns": x_cols,
        "missingness_columns": miss_cols,
        "model_feature_columns": x_cols + miss_cols,
        "target_columns": y_cols,
        "log1p_transformed_dimensions": {
            key: sorted(value)
            for key, value in LOG1P_TARGETS.items()
        },
        "physical_feature_count": len(x_cols),
        "missingness_feature_count": len(miss_cols),
        "model_feature_count": len(x_cols) + len(miss_cols),
    }

    with open(SCHEMA_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print(f"[OK] scanned andjson files : {len(and_files)}")
    print(f"[OK] dataset rows          : {len(rows)}")
    print(f"[OK] unique papers         : {len(manifest_rows)}")
    print(f"[OK] target columns        : {len(y_cols)}")
    print(f"[OK] dataset               : {OUT_CSV}")
    print(f"[OK] build report          : {REPORT_CSV}")
    print(f"[OK] dataset manifest      : {MANIFEST_CSV}")
    print(f"[OK] feature schema        : {SCHEMA_JSON}")
    print("[OK] log1p                 : N12, C10, C11, F6, F11, F12, F18")
    print("=" * 78)


if __name__ == "__main__":
    main()
