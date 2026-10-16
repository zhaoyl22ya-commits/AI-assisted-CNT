# ============================================================
# 03_run_extract_forD.py
#
# 功能：
#   仅处理 Mode D：
#   01_markdown 原文
#       +
#   02_dgrowth 中的 D-mode physics calculation
#       ↓
#   D_prompt_extract_cnt.txt
#       ↓
#   02_extracted_json 中的最终 D JSON
#
# 重要：
#   1. 不修改 data/01_markdown 中的原始 Markdown
#   2. 已存在 JSON 自动跳过，可断点续跑
#   3. 支持并行调用 DeepSeek
# ============================================================

from pathlib import Path
import os
import json
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv


# ============================================================
# 0. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DGROWTH_FOLDER = PROJECT_ROOT / "data" / "02_dgrowth"
MD_FOLDER = PROJECT_ROOT / "data" / "01_markdown"
JSON_FOLDER = PROJECT_ROOT / "data" / "02_extracted_json"
LOG_FOLDER = PROJECT_ROOT / "data" / "02_logs"

D_EXTRACT_PROMPT = (
    PROJECT_ROOT
    / "prompts"
    / "D_prompt_extract_cnt.txt"
)

JSON_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DeepSeek API
# ============================================================

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "未检测到 DEEPSEEK_API_KEY。\n"
        "请确认项目根目录存在 .env，并包含：\n"
        "DEEPSEEK_API_KEY=你的API_KEY"
    )

API_URL = "https://api.deepseek.com/chat/completions"

MODEL = "deepseek-flash"

# D 模式最终结构化抽取使用高推理
THINKING = True
REASONING_EFFORT = "high"

REQUEST_TIMEOUT = 300
MAX_RETRIES = 4

# 并行线程
MAX_WORKERS = 8

# None = 处理全部
# 如果第一次想测试，可以临时改成 5
MAX_FILES = None


# ============================================================
# 2. 日志
# ============================================================

USAGE_LOG = LOG_FOLDER / "api_usage_D.csv"
ERROR_LOG = LOG_FOLDER / "D_extraction_errors.txt"

log_lock = threading.Lock()
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def init_usage_log():
    if USAGE_LOG.exists():
        return

    with open(
        USAGE_LOG,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "stage",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ])


def log_usage(filename, usage):
    if not isinstance(usage, dict):
        return

    with log_lock:
        with open(
            USAGE_LOG,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                filename,
                "D_json_extract",
                MODEL,
                usage.get("prompt_tokens", ""),
                usage.get("completion_tokens", ""),
                usage.get("total_tokens", ""),
            ])


def log_error(filename, message):
    with log_lock:
        with open(
            ERROR_LOG,
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                f"{filename}\n"
                f"{message}\n"
                f"{'=' * 80}\n"
            )


# ============================================================
# 3. Prompt
# ============================================================

def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt not found:\n{path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


# ============================================================
# 4. DeepSeek API
# ============================================================

def call_deepseek(
    system_prompt: str,
    user_prompt: str,
    filename: str,
    max_tokens: int = 32000,
):
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
        "temperature": 0,
        "max_tokens": max_tokens,

        # 当前 DeepSeek thinking 接口
        "thinking": {
            "type": "enabled"
        },
        "reasoning_effort": REASONING_EFFORT,

        # 最终阶段必须返回 JSON
        "response_format": {
            "type": "json_object"
        },
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # 429 / server error
            if (
                response.status_code == 429
                or response.status_code >= 500
            ):
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            response.raise_for_status()

            data = response.json()

            if "error" in data:
                raise RuntimeError(
                    str(data["error"])
                )

            choices = data.get(
                "choices",
                []
            )

            if not choices:
                raise RuntimeError(
                    "DeepSeek response contains no choices."
                )

            content = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )

            if not content:
                raise RuntimeError(
                    "DeepSeek returned empty content."
                )

            usage = data.get(
                "usage",
                {}
            )

            log_usage(
                filename,
                usage
            )

            return content

        except Exception as e:
            last_error = e

            if attempt >= MAX_RETRIES:
                break

            wait_s = min(
                2 ** attempt,
                20
            )

            safe_print(
                f"[RETRY] {filename} "
                f"第 {attempt}/{MAX_RETRIES} 次失败："
                f"{e}"
            )

            safe_print(
                f"        {wait_s}s 后重试..."
            )

            time.sleep(wait_s)

    raise RuntimeError(
        f"DeepSeek 调用最终失败：{last_error}"
    )


# ============================================================
# 5. 原始 MD + dgrowth
#
# 注意：
# 不再像旧代码那样写回原始 Markdown。
# 只在内存里合并。
# ============================================================

def merge_dgrowth_in_memory(
    dg_path: Path,
):
    md_name = (
        dg_path.stem
        + ".md"
    )

    md_path = (
        MD_FOLDER
        / md_name
    )

    if not md_path.exists():
        raise FileNotFoundError(
            f"找不到对应 Markdown：{md_path}"
        )

    md_text = md_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    dg_text = dg_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    merged_text = (
        md_text
        + "\n\n"
        + "=" * 80
        + "\n"
        + "D-MODE PHYSICAL CALCULATION RESULTS\n"
        + "=" * 80
        + "\n\n"
        + dg_text
    )

    return md_name, merged_text


# ============================================================
# 6. JSON 解析
# ============================================================

def parse_json_response(raw: str):
    raw = raw.strip()

    # 正常情况
    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        pass

    # 兼容偶尔出现的 Markdown fence
    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith(
            "```"
        ):
            lines = lines[:-1]

        raw2 = "\n".join(
            lines
        ).strip()

        try:
            return json.loads(raw2)

        except json.JSONDecodeError:
            pass

    # 最后 fallback：
    # 截取最外层 JSON object
    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):
        candidate = raw[
            start:end + 1
        ]

        return json.loads(
            candidate
        )

    raise json.JSONDecodeError(
        "无法定位 JSON object",
        raw,
        0,
    )


# ============================================================
# 7. 单篇 D 文献处理
# ============================================================

def process_one_dgrowth(
    dg_path: Path,
    system_prompt: str,
):
    filename = dg_path.name

    md_name = (
        dg_path.stem
        + ".md"
    )

    output_json = (
        JSON_FOLDER
        / (
            dg_path.stem
            + ".json"
        )
    )

    # --------------------------------
    # 断点续跑
    # --------------------------------
    if output_json.exists():
        return {
            "status": "skip",
            "filename": filename,
            "message": (
                "JSON 已存在"
            ),
        }

    try:
        # --------------------------------
        # STEP 1
        # 原始 MD + D growth
        # 仅内存合并
        # --------------------------------
        _, merged_text = (
            merge_dgrowth_in_memory(
                dg_path
            )
        )

        safe_print(
            f"[D JSON] {filename}"
        )

        # --------------------------------
        # STEP 2
        # 最终 D JSON extraction
        # --------------------------------
        user_prompt = (
            "The following content contains "
            "the original CNT paper Markdown "
            "followed by previously calculated "
            "Mode-D physical quantities.\n\n"
            "Extract the complete Mode-D CNT "
            "synthesis data according to the "
            "system prompt.\n"
            "Return ONLY one valid JSON object.\n\n"
            + merged_text
        )

        raw = call_deepseek(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            filename=filename,
            max_tokens=32000,
        )

        data = parse_json_response(
            raw
        )

        # --------------------------------
        # 基本结构检查
        # 不篡改模型结果
        # --------------------------------
        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "返回结果不是 JSON object"
            )

        growth_mode = data.get(
            "growth_mode",
            ""
        )

        if growth_mode not in (
            "",
            "D",
        ):
            safe_print(
                f"[WARN] {filename} "
                f"返回 growth_mode={growth_mode}"
            )

        # --------------------------------
        # 保存最终 JSON
        # --------------------------------
        with open(
            output_json,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "status": "done",
            "filename": filename,
            "message": str(
                output_json
            ),
        }

    except Exception as e:
        message = str(e)

        log_error(
            filename,
            message,
        )

        return {
            "status": "failed",
            "filename": filename,
            "message": message,
        }


# ============================================================
# 8. MAIN
# ============================================================

def run():
    print(
        "=" * 80
    )
    print(
        "CNT Mode-D Final JSON Extraction"
    )
    print(
        "=" * 80
    )

    # --------------------------------
    # 基础检查
    # --------------------------------
    if not DGROWTH_FOLDER.exists():
        raise FileNotFoundError(
            f"D-growth 文件夹不存在："
            f"{DGROWTH_FOLDER}"
        )

    if not MD_FOLDER.exists():
        raise FileNotFoundError(
            f"Markdown 文件夹不存在："
            f"{MD_FOLDER}"
        )

    system_prompt = load_prompt(
        D_EXTRACT_PROMPT
    )

    # --------------------------------
    # 按文件名排序
    # --------------------------------
    files = sorted(
        DGROWTH_FOLDER.glob(
            "*.dgrowth"
        ),
        key=lambda p: p.name.lower(),
    )

    total_found = len(files)

    if MAX_FILES is not None:
        files = files[
            :MAX_FILES
        ]

    print(
        f".dgrowth 总数：{total_found}"
    )
    print(
        f"本次计划检查：{len(files)}"
    )
    print(
        f"并行线程：{MAX_WORKERS}"
    )
    print(
        f"模型：{MODEL}"
    )
    print(
        f"Thinking："
        f"{'enabled' if THINKING else 'disabled'}"
    )
    print(
        f"输入 Markdown：{MD_FOLDER}"
    )
    print(
        f"输入 dgrowth：{DGROWTH_FOLDER}"
    )
    print(
        f"输出 JSON：{JSON_FOLDER}"
    )
    print(
        "=" * 80
    )

    if not files:
        print(
            "没有发现任何 .dgrowth 文件。"
        )
        return

    init_usage_log()

    counters = {
        "done": 0,
        "skip": 0,
        "failed": 0,
    }

    # --------------------------------
    # 并行处理
    # --------------------------------
    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                process_one_dgrowth,
                dg_path,
                system_prompt,
            ): dg_path
            for dg_path in files
        }

        for future in as_completed(
            future_map
        ):
            dg_path = future_map[
                future
            ]

            try:
                result = future.result()

            except Exception as e:
                result = {
                    "status": "failed",
                    "filename": (
                        dg_path.name
                    ),
                    "message": str(e),
                }

            status = result[
                "status"
            ]

            counters[
                status
            ] += 1

            if status == "done":
                safe_print(
                    f"[OK]   "
                    f"{result['filename']}"
                )

            elif status == "skip":
                safe_print(
                    f"[SKIP] "
                    f"{result['filename']} "
                    f"→ {result['message']}"
                )

            else:
                safe_print(
                    f"[FAIL] "
                    f"{result['filename']} "
                    f"→ {result['message']}"
                )

    # --------------------------------
    # 汇总
    # --------------------------------
    print()
    print(
        "=" * 80
    )
    print(
        "Mode-D extraction finished"
    )
    print(
        "=" * 80
    )
    print(
        f"done   : {counters['done']}"
    )
    print(
        f"skip   : {counters['skip']}"
    )
    print(
        f"failed : {counters['failed']}"
    )
    print(
        f"JSON folder : "
        f"{JSON_FOLDER}"
    )
    print(
        f"usage log   : "
        f"{USAGE_LOG}"
    )

    if counters[
        "failed"
    ] > 0:
        print(
            f"error log   : "
            f"{ERROR_LOG}"
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    run()