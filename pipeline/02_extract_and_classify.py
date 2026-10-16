import os
import re
import csv
import json
from dotenv import load_dotenv
import time
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ==========================================================
# PROJECT PATHS
# ==========================================================

# 当前文件应位于：
# project_root/pipeline/02_extract_and_classify.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

INPUT_FOLDER = PROJECT_ROOT / "data" / "01_markdown"

# 普通模式最终 JSON
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "02_extracted_json"

# B/C/E 的物理计算增强文本
# A/F/G/H/I 保存原始文本副本
# D 后续 run_extract_forD 再补充
ENRICHED_FOLDER = PROJECT_ROOT / "data" / "02_enriched_markdown"

# D 模式第一阶段计算结果
D_GROWTH_FOLDER = PROJECT_ROOT / "data" / "02_dgrowth"

# 日志
LOG_FOLDER = PROJECT_ROOT / "data" / "02_logs"

for folder in [
    OUTPUT_FOLDER,
    ENRICHED_FOLDER,
    D_GROWTH_FOLDER,
    LOG_FOLDER,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# RUN SETTINGS
# ==========================================================

# 第一次先跑50篇。
# 确认结果、费用后改成 None，即全部运行。
MAX_FILES = None

# 并行线程数
# 先用8比较稳，后续可根据实际速度调整。
MAX_WORKERS = 10

# API失败自动重试次数
API_RETRIES = 3

# 每次请求最长等待时间
API_TIMEOUT = 600


# ==========================================================
# DEEPSEEK API SETTINGS
# ==========================================================

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "\n未检测到环境变量 DEEPSEEK_API_KEY。\n"
        "请在 PyCharm Run Configuration 中设置：\n"
        "DEEPSEEK_API_KEY=你的API Key\n"
        "不要把 API Key 直接写进准备上传 GitHub 的代码。\n"
    )

API_URL = "https://api.deepseek.com/chat/completions"

# 新版 DeepSeek API
MODEL = "deepseek-flash"


# ==========================================================
# PROMPT PATHS
# ==========================================================

PROMPT_DIR = PROJECT_ROOT / "prompts"

MODE_CLASSIFY_PROMPT = (
    PROMPT_DIR / "prompt_classify_mode.txt"
)

CALC_PROMPT_MAP = {
    "B": PROMPT_DIR / "B_prompt_calc_cnt.txt",
    "C": PROMPT_DIR / "C_prompt_calc_cnt.txt",
    "E": PROMPT_DIR / "E_prompt_calc_cnt.txt",
}

EXTRACT_PROMPT_MAP = {
    "A": PROMPT_DIR / "A_prompt_extract_cnt.txt",
    "B": PROMPT_DIR / "B_prompt_extract_cnt.txt",
    "C": PROMPT_DIR / "C_prompt_extract_cnt.txt",
    "D": PROMPT_DIR / "D_prompt_extract_cnt.txt",
    "E": PROMPT_DIR / "E_prompt_extract_cnt.txt",
    "F": PROMPT_DIR / "F_prompt_extract_cnt.txt",
    "G": PROMPT_DIR / "G_prompt_extract_cnt.txt",
    "H": PROMPT_DIR / "H_prompt_extract_cnt.txt",
    "I": PROMPT_DIR / "I_prompt_extract_cnt.txt",
}

D_CALC_PROMPT = (
    PROMPT_DIR / "D_prompt_calc_cnt.txt"
)


# ==========================================================
# LOG FILES
# ==========================================================

USAGE_LOG = LOG_FOLDER / "api_usage.csv"
CLASSIFICATION_LOG = LOG_FOLDER / "classification.csv"
FAILURE_LOG = LOG_FOLDER / "failures.csv"

log_lock = threading.Lock()
print_lock = threading.Lock()

total_cost_rmb = 0.0
cost_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ==========================================================
# CURRENT DEEPSEEK FLASH PRICING
#
# 官方价格：2026-09-10 起
# 单位：人民币 / 1M tokens
#
# 后续若 DeepSeek 调价，只需要修改这里。
# ==========================================================

FLASH_PRICING = {
    "offpeak": {
        "cache_hit": 0.02,
        "cache_miss": 1.0,
        "output": 4.0,
    },
    "peak": {
        "cache_hit": 0.04,
        "cache_miss": 2.0,
        "output": 8.0,
    },
}


def current_price_tier():
    """
    DeepSeek 北京时间：
    周一至周五
    09:00-12:00
    14:00-18:00
    为高峰。

    其他全部为空闲时段。
    """

    now = datetime.now(
        ZoneInfo("Asia/Shanghai")
    )

    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour

    is_weekday = weekday < 5

    is_peak_hour = (
        9 <= hour < 12
        or 14 <= hour < 18
    )

    if is_weekday and is_peak_hour:
        return "peak"

    return "offpeak"


# ==========================================================
# CSV LOG HELPERS
# ==========================================================

def append_csv(path, fieldnames, row):

    with log_lock:

        exists = path.exists()

        with open(
            path,
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if not exists:
                writer.writeheader()

            writer.writerow(row)


def log_failure(filename, stage, error):

    append_csv(
        FAILURE_LOG,
        [
            "filename",
            "stage",
            "error",
        ],
        {
            "filename": filename,
            "stage": stage,
            "error": str(error),
        },
    )


def log_classification(filename, mode):

    append_csv(
        CLASSIFICATION_LOG,
        [
            "filename",
            "growth_mode",
        ],
        {
            "filename": filename,
            "growth_mode": mode,
        },
    )


# ==========================================================
# PROMPT HELPER
# ==========================================================

def load_prompt(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


# ==========================================================
# DEEPSEEK API
# ==========================================================

def call_deepseek(
    system_prompt,
    user_prompt,
    source_file,
    stage,
    max_tokens,
    thinking,
    reasoning_effort="high",
    json_output=False,
):

    global total_cost_rmb

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "max_tokens": max_tokens,
    }

    # ----------------------------
    # 新版 thinking API
    # ----------------------------

    if thinking:

        payload["thinking"] = {
            "type": "enabled"
        }

        payload["reasoning_effort"] = (
            reasoning_effort
        )

    else:

        payload["thinking"] = {
            "type": "disabled"
        }

        payload["reasoning_effort"] = "none"

    # ----------------------------
    # JSON Output
    # ----------------------------

    if json_output:

        payload["response_format"] = {
            "type": "json_object"
        }

    # ----------------------------
    # Retry
    # ----------------------------

    last_error = None

    for attempt in range(
        1,
        API_RETRIES + 1,
    ):

        try:

            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT,
            )

            if response.status_code != 200:

                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )

            data = response.json()

            if "error" in data:

                raise RuntimeError(
                    str(data["error"])
                )

            # ----------------------------
            # Usage
            # ----------------------------

            usage = data.get(
                "usage",
                {}
            )

            hit_tokens = int(
                usage.get(
                    "prompt_cache_hit_tokens",
                    0,
                )
                or 0
            )

            miss_tokens = int(
                usage.get(
                    "prompt_cache_miss_tokens",
                    0,
                )
                or 0
            )

            completion_tokens = int(
                usage.get(
                    "completion_tokens",
                    0,
                )
                or 0
            )

            reasoning_tokens = int(
                usage.get(
                    "completion_tokens_details",
                    {},
                ).get(
                    "reasoning_tokens",
                    0,
                )
                or 0
            )

            tier = current_price_tier()

            price = FLASH_PRICING[tier]

            estimated_cost = (
                hit_tokens
                / 1_000_000
                * price["cache_hit"]
                +
                miss_tokens
                / 1_000_000
                * price["cache_miss"]
                +
                completion_tokens
                / 1_000_000
                * price["output"]
            )

            with cost_lock:

                total_cost_rmb += (
                    estimated_cost
                )

                running_cost = (
                    total_cost_rmb
                )

            # ----------------------------
            # Usage log
            # ----------------------------

            append_csv(
                USAGE_LOG,
                [
                    "filename",
                    "stage",
                    "price_tier",
                    "cache_hit_tokens",
                    "cache_miss_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "estimated_cost_rmb",
                ],
                {
                    "filename": source_file,
                    "stage": stage,
                    "price_tier": tier,
                    "cache_hit_tokens": hit_tokens,
                    "cache_miss_tokens": miss_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "estimated_cost_rmb": (
                        f"{estimated_cost:.6f}"
                    ),
                },
            )

            safe_print(
                f"[API] {source_file} | "
                f"{stage} | "
                f"hit={hit_tokens} | "
                f"miss={miss_tokens} | "
                f"out={completion_tokens} | "
                f"¥{estimated_cost:.4f} | "
                f"累计≈¥{running_cost:.2f}"
            )

            return (
                data["choices"][0]
                ["message"]
                ["content"]
            )

        except Exception as e:

            last_error = e

            safe_print(
                f"[API ERROR] "
                f"{source_file} | "
                f"{stage} | "
                f"attempt "
                f"{attempt}/{API_RETRIES} | "
                f"{e}"
            )

            if attempt < API_RETRIES:

                time.sleep(
                    3 * attempt
                )

    raise RuntimeError(
        f"DeepSeek API failed after "
        f"{API_RETRIES} attempts: "
        f"{last_error}"
    )


# ==========================================================
# STEP 1 — GROWTH MODE CLASSIFICATION
# ==========================================================

def classify_mode(
    md_text,
    source_file,
):

    system_prompt = load_prompt(
        MODE_CLASSIFY_PROMPT
    )

    user_prompt = (
        "Determine the CNT growth mode "
        "(A/B/C/D/E/F/G/H/I) "
        "for the following paper.\n"
        "Return ONLY one character.\n\n"
        + md_text
    )

    raw = call_deepseek(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_file=source_file,
        stage="classification",
        max_tokens=20,
        thinking=False,
    )

    raw = raw.strip().upper()

    if not raw:
        return ""

    # 优先接受第一个字符
    mode = raw[0]

    if mode in list("ABCDEFGHI"):
        return mode

    # 备用解析
    m = re.search(
        r"\b([A-I])\b",
        raw
    )

    if m:
        return m.group(1)

    return ""


# ==========================================================
# STEP 2 — B / C / E PHYSICS CALCULATION
# ==========================================================

def run_calc_stage(
    mode,
    md_text,
    source_file,
):

    if mode not in [
        "B",
        "C",
        "E",
    ]:
        return md_text

    system_prompt = load_prompt(
        CALC_PROMPT_MAP[mode]
    )

    user_prompt = (
        "Below is the CNT growth article. "
        "Perform full physical reasoning "
        "and laminar-flow calculations "
        "as specified.\n\n"
        + md_text
    )

    calc_result = call_deepseek(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_file=source_file,
        stage=f"{mode}_physics_calc",
        max_tokens=32000,
        thinking=True,
        reasoning_effort="high",
    )

    extended_text = (
        md_text
        + "\n\n\n"
        + "============================\n"
        + "STAGE 1 PHYSICAL COMPUTATION RESULTS:\n"
        + "============================\n\n"
        + calc_result
    )

    return extended_text


# ==========================================================
# STEP 3 — FINAL JSON EXTRACTION
# ==========================================================

def extract_with_reasoner(
    mode,
    final_text,
    source_file,
):

    prompt_file = (
        EXTRACT_PROMPT_MAP.get(mode)
    )

    if prompt_file is None:
        raise ValueError(
            f"No extraction prompt "
            f"for mode {mode}"
        )

    system_prompt = load_prompt(
        prompt_file
    )

    user_prompt = (
        "Extract CNT synthesis data "
        "and return ONLY JSON.\n\n"
        "Below is the full article text "
        "combined with the "
        "physics-calculation results:\n\n"
        + final_text
    )

    raw = call_deepseek(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_file=source_file,
        stage=f"{mode}_json_extract",
        max_tokens=32000,
        thinking=True,
        reasoning_effort="high",
        json_output=True,
    )

    # JSON Output 理论上已经保证合法 JSON，
    # 这里仍保留原来的双保险。
    try:

        return json.loads(raw)

    except Exception:

        try:

            start = raw.find("{")
            end = raw.rfind("}") + 1

            if start < 0 or end <= start:
                raise ValueError(
                    "No JSON object found"
                )

            trimmed = raw[start:end]

            return json.loads(
                trimmed
            )

        except Exception as e:

            raise RuntimeError(
                f"JSON decode failed: {e}"
            )


# ==========================================================
# D MODE CALCULATION
# ==========================================================

def run_d_calculation(
    md_text,
    source_file,
):

    d_prompt = load_prompt(
        D_CALC_PROMPT
    )

    user_prompt = (
        d_prompt
        + "\n\n"
        + "======================\n"
        + "INPUT TEXT:\n"
        + md_text
    )

    result = call_deepseek(
        system_prompt=(
            "You are a CNT D-mode "
            "physics calculator."
        ),
        user_prompt=user_prompt,
        source_file=source_file,
        stage="D_physics_calc",
        max_tokens=32000,
        thinking=True,
        reasoning_effort="high",
    )

    return result


# ==========================================================
# SINGLE FILE PROCESS
# ==========================================================

def process_one_file(filename):

    input_path = (
        INPUT_FOLDER / filename
    )

    stem = Path(filename).stem

    json_path = (
        OUTPUT_FOLDER
        / f"{stem}.json"
    )

    enriched_path = (
        ENRICHED_FOLDER
        / f"{stem}.md"
    )

    dgrowth_path = (
        D_GROWTH_FOLDER
        / f"{stem}.dgrowth"
    )

    # ------------------------------------------
    # 已经完成 JSON 的文件直接跳过
    # ------------------------------------------

    if json_path.exists():

        safe_print(
            f"[SKIP] 已存在 JSON："
            f"{filename}"
        )

        return (
            filename,
            "existing_json",
        )

    try:

        md_text = input_path.read_text(
            encoding="utf-8"
        )

        if not md_text.strip():

            raise ValueError(
                "Markdown is empty"
            )

        # ======================================
        # STEP 1
        # ======================================

        mode = classify_mode(
            md_text,
            filename,
        )

        if not mode:

            raise RuntimeError(
                "Growth mode "
                "classification failed"
            )

        log_classification(
            filename,
            mode,
        )

        safe_print(
            f"[MODE] "
            f"{filename} → {mode}"
        )

        # ======================================
        # D MODE
        # ======================================

        if mode == "D":

            # 保留原始 MD 副本
            if not enriched_path.exists():

                enriched_path.write_text(
                    md_text,
                    encoding="utf-8",
                )

            # 断点续跑
            if dgrowth_path.exists():

                safe_print(
                    f"[SKIP D] "
                    f"已有 dgrowth："
                    f"{filename}"
                )

                return (
                    filename,
                    "D_existing",
                )

            d_result = (
                run_d_calculation(
                    md_text,
                    filename,
                )
            )

            dgrowth_path.write_text(
                d_result,
                encoding="utf-8",
            )

            safe_print(
                f"[OK D] "
                f"{filename} "
                f"→ {dgrowth_path.name}"
            )

            return (
                filename,
                "D_done",
            )

        # ======================================
        # B / C / E
        # ======================================

        if mode in [
            "B",
            "C",
            "E",
        ]:

            # 若以前已经完成物理计算，
            # 直接读取，避免重复花钱。
            if enriched_path.exists():

                old_text = (
                    enriched_path.read_text(
                        encoding="utf-8"
                    )
                )

                marker = (
                    "STAGE 1 PHYSICAL "
                    "COMPUTATION RESULTS:"
                )

                if marker in old_text:

                    final_text = old_text

                    safe_print(
                        f"[REUSE] "
                        f"已有物理计算："
                        f"{filename}"
                    )

                else:

                    final_text = (
                        run_calc_stage(
                            mode,
                            md_text,
                            filename,
                        )
                    )

                    enriched_path.write_text(
                        final_text,
                        encoding="utf-8",
                    )

            else:

                final_text = (
                    run_calc_stage(
                        mode,
                        md_text,
                        filename,
                    )
                )

                enriched_path.write_text(
                    final_text,
                    encoding="utf-8",
                )

        # ======================================
        # A / F / G / H / I
        # ======================================

        else:

            final_text = md_text

            if not enriched_path.exists():

                enriched_path.write_text(
                    final_text,
                    encoding="utf-8",
                )

        # ======================================
        # FINAL JSON
        # ======================================

        data = extract_with_reasoner(
            mode,
            final_text,
            filename,
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        safe_print(
            f"[OK] "
            f"{filename} "
            f"→ {mode} "
            f"→ {json_path.name}"
        )

        return (
            filename,
            f"{mode}_done",
        )

    except Exception as e:

        log_failure(
            filename,
            "processing",
            e,
        )

        safe_print(
            f"[FAILED] "
            f"{filename} | {e}"
        )

        return (
            filename,
            "failed",
        )


# ==========================================================
# BATCH PROCESS
# ==========================================================

def batch_process():

    global total_cost_rmb

    files = sorted(
        f.name
        for f in INPUT_FOLDER.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".md"
    )

    total_available = len(files)

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    safe_print(
        "=" * 80
    )

    safe_print(
        "CNT Literature "
        "Classification & Extraction"
    )

    safe_print(
        "=" * 80
    )

    safe_print(
        f"Markdown 总数："
        f"{total_available}"
    )

    safe_print(
        f"本次计划处理："
        f"{len(files)}"
    )

    safe_print(
        f"并行线程："
        f"{MAX_WORKERS}"
    )

    safe_print(
        f"当前价格时段："
        f"{current_price_tier()}"
    )

    safe_print(
        "=" * 80
    )

    counts = {}

    start_time = time.time()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_one_file,
                filename,
            ): filename
            for filename in files
        }

        finished = 0

        for future in as_completed(
            futures
        ):

            filename = futures[future]

            try:

                _, status = (
                    future.result()
                )

            except Exception as e:

                status = "failed"

                safe_print(
                    f"[THREAD ERROR] "
                    f"{filename}: {e}"
                )

            counts[status] = (
                counts.get(
                    status,
                    0
                )
                + 1
            )

            finished += 1

            safe_print(
                f"[PROGRESS] "
                f"{finished}/"
                f"{len(files)}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    safe_print(
        "\n"
        + "=" * 80
    )

    safe_print(
        "本批次处理完成"
    )

    safe_print(
        "=" * 80
    )

    safe_print(
        f"耗时："
        f"{elapsed / 60:.2f} min"
    )

    safe_print(
        f"估算 API 费用："
        f"¥{total_cost_rmb:.2f}"
    )

    safe_print(
        "\n结果统计："
    )

    for status, count in sorted(
        counts.items()
    ):

        safe_print(
            f"{status}: {count}"
        )

    safe_print(
        "\nUsage log："
        f"{USAGE_LOG}"
    )

    safe_print(
        "Classification log："
        f"{CLASSIFICATION_LOG}"
    )

    safe_print(
        "Failure log："
        f"{FAILURE_LOG}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    batch_process()