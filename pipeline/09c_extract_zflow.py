    # 09c_extract_zflow.py
# ============================================================
# Historical z_flow extractor migrated to the current CNT pipeline.
#
# Scientific behavior is unchanged:
# - one finalized JSON + same-stem Markdown = one DeepSeek call
# - prompt: promptsV2/A_prompt_zflow.txt
# - reference table: promptsV2/gas_properteis.txt
# - model: deepseek-reasoner
# - A-E × L1-L3
#
# Current interfaces:
# - input : data/08_finalized/{MODE}/{LEVEL}/*.json
# - MD    : data/01_markdown/<same stem>.md
# - output: data/09_latent/{MODE}/{LEVEL}/zflowjson/*.zflowjson
# - failed raw response: data/09_logs/09c_zflow_failed_raw/
#
# Requirements:
#   pip install requests
# ============================================================

import os
import json
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# -----------------------------
# Current project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

FINALIZED_ROOT = DATA_ROOT / "08_finalized"
MD_DIR = DATA_ROOT / "01_markdown"
LATENT_ROOT = DATA_ROOT / "09_latent"
LOG_ROOT = DATA_ROOT / "09_logs"
FAILED_RAW_ROOT = LOG_ROOT / "09c_zflow_failed_raw"

PROMPT_FILE = PROJECT_ROOT / "promptsV2" / "A_prompt_zflow.txt"

# Keep the historical filename spelling exactly as used by the old extractor.
GAS_TABLE_TXT = PROJECT_ROOT / "promptsV2" / "gas_properteis.txt"

ENV_FILE = PROJECT_ROOT / ".env"

# A-E × L1-L3
LETTERS = ["A", "B", "C", "D", "E"]
LEVELS = ["L1", "L2", "L3"]

# -----------------------------
# DeepSeek API config
# -----------------------------
def _load_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


_ENV = _load_env_file(ENV_FILE)

DEEPSEEK_BASE_URL = (
    os.getenv("DEEPSEEK_BASE_URL")
    or _ENV.get("DEEPSEEK_BASE_URL")
    or "https://api.deepseek.com"
).rstrip("/")

DEEPSEEK_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or _ENV.get("DEEPSEEK_API_KEY")
    or ""
).strip()

MODEL = "deepseek-flash"

# -----------------------------
# Runtime config
# -----------------------------
TIMEOUT_SEC = 600
MAX_RETRIES = 5
BACKOFF_BASE_SEC = 1.5

# 防止 gas_properteis.txt 太大撑爆上下文：可按需调小/调大
MAX_TABLE_CHARS = 200_000  # 超出会截断

# -----------------------------
# Parallel config
# -----------------------------
# 你如果想“一次并行 4 个 .json”，把这里改成 4
MAX_WORKERS = 20

# 每线程一个 Session：更稳定（避免多线程共享同一个 session 的隐患）
_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_json_text(p: Path) -> str:
    raw = p.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return raw


def safe_parse_json(raw: str) -> Tuple[Optional[dict], str]:
    raw_strip = raw.strip()
    try:
        return json.loads(raw_strip), raw_strip
    except Exception:
        pass

    if "{" in raw_strip and "}" in raw_strip:
        start = raw_strip.find("{")
        end = raw_strip.rfind("}")
        candidate = raw_strip[start:end + 1].strip()
        try:
            return json.loads(candidate), candidate
        except Exception:
            return None, candidate

    return None, raw_strip


def read_gas_table_text(p: Path) -> str:
    text = read_text(p)
    header = (
        "[gas_properteis.txt]\n"
        "format: plain text (often TSV-like) gas properties reference\n"
        "----BEGIN TABLE----\n"
    )
    footer = "\n----END TABLE----\n"
    merged = header + text + footer

    if len(merged) > MAX_TABLE_CHARS:
        merged = merged[:MAX_TABLE_CHARS] + "\n[TRUNCATED: gas_properteis.txt content clipped due to size]\n"

    return merged


def build_messages(prompt_text: str, gas_table_text: str, json_text: str, md_text: str) -> list:
    system_msg = (
        "You are a careful information extraction engine.\n"
        "Follow the user's instructions strictly.\n"
        "You MUST read and use the provided gas properties table when needed.\n"
        "Output MUST be valid JSON only. No markdown. No extra text."
    )

    user_msg = (
        "INSTRUCTIONS (from A_prompt_zflow.txt):\n"
        f"{prompt_text}\n\n"
        "REFERENCE TABLE (from gas_properteis.txt):\n"
        f"{gas_table_text}\n\n"
        "INPUT JSON (source):\n"
        f"{json_text}\n\n"
        "INPUT MD (source):\n"
        f"{md_text}\n\n"
        "RESPONSE REQUIREMENT:\n"
        "Return ONLY valid JSON (a single JSON object). Do not include any other text."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

def call_deepseek(session: requests.Session, messages: list) -> str:
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 32000,
        "thinking": {
            "type": "enabled"
        },
        "reasoning_effort": "high",
        "response_format": {
            "type": "json_object"
        },
    }

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SEC,
            )

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}"
                )
                time.sleep(
                    BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                )
                continue

            resp.raise_for_status()

            data = resp.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_err = e

            time.sleep(
                BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            )

    raise RuntimeError(
        f"DeepSeek call failed after "
        f"{MAX_RETRIES} retries: {last_err}"
    )

def ensure_static_inputs():
    if not FINALIZED_ROOT.exists():
        raise FileNotFoundError(f"08_finalized not found: {FINALIZED_ROOT}")
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    if not GAS_TABLE_TXT.exists():
        raise FileNotFoundError(f"Gas properties txt not found: {GAS_TABLE_TXT}")
    if not MD_DIR.exists():
        raise FileNotFoundError(f"MD_DIR not found: {MD_DIR}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FAILED_RAW_ROOT.mkdir(parents=True, exist_ok=True)

    if not DEEPSEEK_API_KEY:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is empty. Set it in project .env "
            "or the system environment."
        )


def list_batches() -> List[tuple]:
    """
    Return list of (letter, level, json_dir, out_dir)
    """
    batches = []
    for letter in LETTERS:
        for level in LEVELS:
            json_dir = FINALIZED_ROOT / letter / level
            out_dir = LATENT_ROOT / letter / level / "zflowjson"
            batches.append((letter, level, json_dir, out_dir))
    return batches


# -----------------------------
# Parallel task: one json + one md -> one DeepSeek call -> one output
# -----------------------------
def process_one_json_task(
    prompt_text: str,
    gas_table_text: str,
    letter: str,
    level: str,
    jp: Path,
    md_path: Path,
    out_path: Path,
) -> Tuple[str, str]:
    """
    Return (status, info)
    status:
      - "ok"    : wrote valid JSON to out_path
      - "skip"  : missing md
      - "fail"  : request/parse error; invalid raw goes to data/09_logs
    """
    if not md_path.exists():
        return "skip", f"MD not found: {md_path}"

    json_text = read_json_text(jp)
    md_text = read_text(md_path)
    messages = build_messages(prompt_text, gas_table_text, json_text, md_text)

    session = _get_thread_session()

    try:
        raw = call_deepseek(session, messages)
        obj, cleaned = safe_parse_json(raw)

        if obj is None:
            failed_dir = FAILED_RAW_ROOT / letter / level
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / f"{jp.stem}.txt"
            failed_path.write_text(cleaned, encoding="utf-8")
            return "fail", f"Output not valid JSON; raw saved: {failed_path}"

        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return "ok", "Saved."

    except Exception as e:
        return "fail", f"Exception: {e}"


def process_one_batch(
    session: requests.Session,   # kept for signature compatibility (original), not used in tasks
    prompt_text: str,
    gas_table_text: str,
    letter: str,
    level: str,
    json_dir: Path,
    out_dir: Path,
) -> Tuple[int, int, int, int]:
    """
    Returns (total_json, ok, skipped, failed)
    skipped: missing md

    Parallel guarantee (no duplicate/missing):
    - Build task list once from json_files
    - Submit exactly one future per file
    - Collect exactly len(tasks) results via as_completed
    - End verify ok+skipped+failed == total
    """
    if not json_dir.exists():
        print(f"\n[BATCH {letter}/{level}] JSON folder not found, skip: {json_dir}")
        return (0, 0, 0, 0)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"\n[BATCH {letter}/{level}] No .json files: {json_dir}")
        return (0, 0, 0, 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== BATCH {letter}/{level} ==========")
    print(f"JSON : {json_dir}")
    print(f"OUT : {out_dir}")
    print(f"FILES: {len(json_files)}")
    print(f"PARALLEL_WORKERS: {MAX_WORKERS}")

    ok = 0
    skipped = 0
    failed = 0

    # Build deterministic task list: one json -> one md -> one output
    tasks = []
    seen_stems = set()
    for jp in json_files:
        stem = jp.stem
        if stem in seen_stems:
            print(f"  [WARN] Duplicate stem in file list (unexpected): {stem}")
            continue
        seen_stems.add(stem)

        md_path = MD_DIR / f"{stem}.md"
        out_path = out_dir / f"{stem}.zflowjson"
        tasks.append((stem, jp, md_path, out_path))

    # Submit tasks in parallel
    future_to_stem: Dict[object, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for stem, jp, md_path, out_path in tasks:
            fut = executor.submit(
                process_one_json_task,
                prompt_text, gas_table_text, letter, level,
                jp, md_path, out_path
            )
            future_to_stem[fut] = stem

        # Collect results (exactly once per task)
        done_count = 0
        for fut in as_completed(future_to_stem):
            stem = future_to_stem[fut]
            done_count += 1
            try:
                status, info = fut.result()
            except Exception as e:
                status, info = "fail", f"Future exception: {e}"

            # Per-file audit log
            print(f"[{letter}/{level}] DONE ({done_count}/{len(tasks)}) {stem} -> {status} | {info}")

            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1

    total = len(tasks)

    # Accounting check
    accounted = ok + skipped + failed
    if accounted != total:
        print(f"[ALERT] Accounting mismatch! TOTAL={total} but OK+SKIP+FAIL={accounted}. Something is wrong.")

    print(f"--- BATCH {letter}/{level} SUMMARY ---")
    print(f"TOTAL  : {total}")
    print(f"OK     : {ok}")
    print(f"SKIPPED: {skipped} (missing md)")
    print(f"FAILED : {failed}")

    return (total, ok, skipped, failed)



def preflight_markdown_pairs():
    total_json = 0
    matched_md = 0
    missing_md = []

    for letter, level, json_dir, _ in list_batches():
        if not json_dir.exists():
            continue

        json_files = sorted(json_dir.glob("*.json"))
        total_json += len(json_files)

        for jp in json_files:
            md_path = MD_DIR / f"{jp.stem}.md"

            if md_path.exists():
                matched_md += 1
            else:
                missing_md.append(
                    (letter, level, jp.name, md_path)
                )

    print("\n" + "=" * 80)
    print("09c z_flow PREFLIGHT")
    print(f"FINALIZED JSON : {total_json}")
    print(f"MATCHED MD     : {matched_md}")
    print(f"MISSING MD     : {len(missing_md)}")
    print(f"MODEL          : {MODEL}")
    print(f"WORKERS        : {MAX_WORKERS}")
    print("=" * 80)

    if total_json == 0:
        raise RuntimeError("No finalized JSON found.")

    if missing_md:
        for letter, level, name, md_path in missing_md[:20]:
            print(
                f"[MISSING MD] {letter}/{level} | "
                f"{name} -> {md_path}"
            )

        raise RuntimeError(
            "Preflight failed: finalized JSON and Markdown "
            "are not fully paired."
        )

    return total_json


def main():
    ensure_static_inputs()
    preflight_markdown_pairs()

    # Read static inputs ONCE
    prompt_text = read_text(PROMPT_FILE)
    gas_table_text = read_gas_table_text(GAS_TABLE_TXT)

    # Kept to preserve original structure/signature (tasks use thread-local sessions)
    session = requests.Session()

    grand_total = 0
    grand_ok = 0
    grand_skipped = 0
    grand_failed = 0

    for letter, level, json_dir, out_dir in list_batches():
        total, ok, skipped, failed = process_one_batch(
            session=session,
            prompt_text=prompt_text,
            gas_table_text=gas_table_text,
            letter=letter,
            level=level,
            json_dir=json_dir,
            out_dir=out_dir,
        )
        grand_total += total
        grand_ok += ok
        grand_skipped += skipped
        grand_failed += failed

    # Global accounting check
    accounted = grand_ok + grand_skipped + grand_failed
    if accounted != grand_total:
        print(f"[ALERT] GLOBAL mismatch! TOTAL={grand_total} but OK+SKIP+FAIL={accounted}.")

    print("\n========== ALL BATCHES SUMMARY ==========")
    print(f"TOTAL  : {grand_total}")
    print(f"OK     : {grand_ok}")
    print(f"SKIPPED: {grand_skipped} (missing md)")
    print(f"FAILED : {grand_failed}")
    print("DONE.")


if __name__ == "__main__":
    main()