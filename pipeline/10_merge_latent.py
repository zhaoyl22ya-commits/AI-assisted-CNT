# 10_merge_latent.py
# ============================================================
# Batch merge for ALL growth modes (A-E) and levels (L1-L3).
#
# Current pipeline interfaces:
#   base JSON:
#     data/08_finalized/{mode}/{level}/*.json
#   latent JSON:
#     data/09_latent/{mode}/{level}/zcjson/*.zcjson
#     data/09_latent/{mode}/{level}/znucjson/*.znucjson
#     data/09_latent/{mode}/{level}/zflowjson/*.zflowjson
#     data/09_latent/{mode}/{level}/zprocjson/*.zprocjson
#   output:
#     data/10_andjson/{mode}/{level}/*.andjson
#
# Output .andjson has minimal header and per-synthesis:
#   synthesis_id + product(with growth_quality_score extracted)
#   + z_C + z_nuc + z_flow + z_proc
#
# Rules:
# - DO NOT modify any source file.
# - Use base .json syntheses list/order as authoritative.
# - Match z blocks by synthesis_id (never by array order).
# - Remove catalysts/process/calc_block; keep only product + z blocks.
# - Extract growth_quality_score* from calc_block into product.
# - Keep paper header: title/doi/year/journal.
# ============================================================

import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# ---------------------------
# Global Config
# ---------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

FINALIZED_ROOT = DATA_ROOT / "08_finalized"
LATENT_ROOT = DATA_ROOT / "09_latent"
OUTPUT_ROOT = DATA_ROOT / "10_andjson"

GROWTH_MODES = ["A", "B", "C", "D", "E"]
LEVELS = ["L1", "L2", "L3"]

PRETTY_JSON = True
ALLOW_PARTIAL = True  # if False, missing any z file => skip that base json


# ---------------------------
# Utilities
# ---------------------------

def safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON decode error: {path}\n{e}") from e


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def filename_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def minimal_paper_header(paper: Any) -> Dict[str, Any]:
    if not isinstance(paper, dict):
        return {"title": None, "doi": None, "year": None, "journal": None}
    return {
        "title": paper.get("title"),
        "doi": paper.get("doi"),
        "year": paper.get("year"),
        "journal": paper.get("journal"),
    }


def build_sid_to_zmap(z_doc: Optional[Dict[str, Any]], z_key: str) -> Dict[str, Any]:
    sid_map: Dict[str, Any] = {}
    if not z_doc:
        return sid_map

    syntheses = z_doc.get("syntheses", [])
    if not isinstance(syntheses, list):
        return sid_map

    for s in syntheses:
        if not isinstance(s, dict):
            continue
        sid = s.get("synthesis_id")
        if not sid:
            continue

        if z_key in s:
            sid_map[sid] = s.get(z_key)
        else:
            # fallback: case-insensitive match
            z_like = None
            for k, v in s.items():
                if isinstance(k, str) and k.lower() == z_key.lower():
                    z_like = v
                    break
            # fallback: firstversioncode non-id key starting with "z"
            if z_like is None:
                for k, v in s.items():
                    if isinstance(k, str) and k.lower().startswith("z") and k != "synthesis_id":
                        z_like = v
                        break
            sid_map[sid] = z_like

    return sid_map


def extract_growth_quality_from_calc(calc_block: Any, growth_mode: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(calc_block, dict):
        return None, None

    keys = [k for k in calc_block.keys() if isinstance(k, str) and k.startswith("growth_quality_score")]
    if not keys:
        return None, None

    preferred = None
    gm = (growth_mode or "").strip()
    if gm:
        exact = f"growth_quality_score_{gm}"
        if exact in calc_block:
            preferred = exact

    if preferred is None:
        preferred = sorted(keys)[0]

    val = calc_block.get(preferred)
    gq_obj = {"value": val, "unit": "score", "source_key": preferred}
    return gq_obj, preferred


def ordered_product_with_gq(product: Any, gq_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if gq_obj is not None:
        out["growth_quality_score"] = gq_obj

    if isinstance(product, dict):
        for k, v in product.items():
            if k == "growth_quality_score":
                continue
            out[k] = v
    else:
        if product is not None:
            out["_raw_product"] = product
    return out


# ---------------------------
# Merge logic for one base json
# ---------------------------

def merge_one(base_json_path: str, zc_dir: str, znuc_dir: str, zflow_dir: str, zproc_dir: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    stats = {
        "base_file": os.path.basename(base_json_path),
        "syntheses_total": 0,
        "gq_extracted": 0,
        "gq_missing": 0,
        "zc_missing": 0,
        "znuc_missing": 0,
        "zflow_missing": 0,
        "zproc_missing": 0,
        "zc_file_found": 0,
        "znuc_file_found": 0,
        "zflow_file_found": 0,
        "zproc_file_found": 0,
        "status": "ok",
        "note": "",
    }

    base_doc = safe_load_json(base_json_path)
    if base_doc is None:
        stats["status"] = "fatal"
        stats["note"] = "base json not found"
        return None, stats

    stem = filename_stem(base_json_path)

    zc_path = os.path.join(zc_dir, stem + ".zcjson")
    znuc_path = os.path.join(znuc_dir, stem + ".znucjson")
    zflow_path = os.path.join(zflow_dir, stem + ".zflowjson")
    zproc_path = os.path.join(zproc_dir, stem + ".zprocjson")

    zc_doc = safe_load_json(zc_path)
    znuc_doc = safe_load_json(znuc_path)
    zflow_doc = safe_load_json(zflow_path)
    zproc_doc = safe_load_json(zproc_path)

    stats["zc_file_found"] = 1 if zc_doc is not None else 0
    stats["znuc_file_found"] = 1 if znuc_doc is not None else 0
    stats["zflow_file_found"] = 1 if zflow_doc is not None else 0
    stats["zproc_file_found"] = 1 if zproc_doc is not None else 0

    if not ALLOW_PARTIAL:
        if any(x == 0 for x in [stats["zc_file_found"], stats["znuc_file_found"], stats["zflow_file_found"], stats["zproc_file_found"]]):
            stats["status"] = "skip"
            stats["note"] = "missing one or more z files (ALLOW_PARTIAL=False)"
            return None, stats

    zc_map = build_sid_to_zmap(zc_doc, "z_C")
    znuc_map = build_sid_to_zmap(znuc_doc, "z_nuc")
    zflow_map = build_sid_to_zmap(zflow_doc, "z_flow")
    zproc_map = build_sid_to_zmap(zproc_doc, "z_proc")

    merged: Dict[str, Any] = {
        "growth_mode": base_doc.get("growth_mode"),
        "data_level": base_doc.get("data_level"),
        "paper": minimal_paper_header(base_doc.get("paper")),
        "syntheses": []
    }

    base_syntheses = base_doc.get("syntheses", [])
    if not isinstance(base_syntheses, list):
        stats["status"] = "fatal"
        stats["note"] = "base_doc.syntheses is not a list"
        return None, stats

    stats["syntheses_total"] = len(base_syntheses)
    growth_mode = base_doc.get("growth_mode")

    for s in base_syntheses:
        if not isinstance(s, dict):
            continue
        sid = s.get("synthesis_id")

        gq_obj, _ = extract_growth_quality_from_calc(s.get("calc_block"), growth_mode)
        if gq_obj is not None:
            stats["gq_extracted"] += 1
        else:
            stats["gq_missing"] += 1

        product_out = ordered_product_with_gq(s.get("product"), gq_obj)

        zc = zc_map.get(sid) if sid is not None else None
        znuc = znuc_map.get(sid) if sid is not None else None
        zflow = zflow_map.get(sid) if sid is not None else None
        zproc = zproc_map.get(sid) if sid is not None else None

        if zc is None: stats["zc_missing"] += 1
        if znuc is None: stats["znuc_missing"] += 1
        if zflow is None: stats["zflow_missing"] += 1
        if zproc is None: stats["zproc_missing"] += 1

        merged["syntheses"].append({
            "synthesis_id": sid,
            "product": product_out,
            "z_C": zc,
            "z_nuc": znuc,
            "z_flow": zflow,
            "z_proc": zproc,
        })

    return merged, stats


# ---------------------------
# Process one (mode, level) directory
# ---------------------------

def process_one_dir(mode: str, level: str) -> None:
    base_json_dir = str(FINALIZED_ROOT / mode / level)

    latent_dir = LATENT_ROOT / mode / level
    zc_dir = str(latent_dir / "zcjson")
    znuc_dir = str(latent_dir / "znucjson")
    zflow_dir = str(latent_dir / "zflowjson")
    zproc_dir = str(latent_dir / "zprocjson")

    out_dir = str(OUTPUT_ROOT / mode / level)

    # If the finalized base directory does not exist, skip gracefully
    if not os.path.isdir(base_json_dir):
        print(f"[SKIP] {mode}/{level} :: missing 08_finalized directory")
        return

    ensure_dir(out_dir)
    report_csv = os.path.join(out_dir, "merge_report.csv")

    base_files = [
        os.path.join(base_json_dir, fn)
        for fn in os.listdir(base_json_dir)
        if fn.lower().endswith(".json")
    ]
    base_files.sort()

    report_rows: List[Dict[str, Any]] = []

    for base_path in base_files:
        merged_doc, stats = merge_one(base_path, zc_dir, znuc_dir, zflow_dir, zproc_dir)
        report_rows.append(stats)

        if merged_doc is None:
            print(f"[{stats['status']}] {mode}/{level} :: {stats['base_file']} :: {stats['note']}")
            continue

        stem = filename_stem(base_path)
        out_path = os.path.join(out_dir, stem + ".andjson")

        with open(out_path, "w", encoding="utf-8") as f:
            if PRETTY_JSON:
                json.dump(merged_doc, f, ensure_ascii=False, indent=2)
            else:
                json.dump(merged_doc, f, ensure_ascii=False, separators=(",", ":"))

    # Write per-dir report
    fieldnames = [
        "base_file", "status", "note",
        "syntheses_total",
        "zc_file_found", "znuc_file_found", "zflow_file_found", "zproc_file_found",
        "gq_extracted", "gq_missing",
        "zc_missing", "znuc_missing", "zflow_missing", "zproc_missing",
    ]
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[DONE] {mode}/{level} -> {out_dir} | report: {report_csv}")


# ---------------------------
# Entry
# ---------------------------

def main() -> None:
    for mode in GROWTH_MODES:
        for level in LEVELS:
            process_one_dir(mode, level)


if __name__ == "__main__":
    main()