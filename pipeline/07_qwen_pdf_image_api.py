# -*- coding: utf-8 -*-
"""
07_qwen_pdf_image_api.py

A-E × L1/L2 自动遍历；6 并发。

输入：
    data/05_cleaned/{MODE}/{LEVEL}/CLEANED/*.json
    data/00_raw_papers 下同名 PDF
    prompts/gpt_auto_prompt.txt

输出：
    data/07_image_patches/{MODE}/{LEVEL}/*.rejson

原则：
    1. 不修改 cleaned JSON
    2. 只保存图像补全 patch
    3. 可无歧义修复的格式错误自动修复
    4. 字段语义完全丢失时才重试
    5. 原 JSON 已有值不覆盖，直接从 patch 删除
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import re
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests


# ============================================================
# 1. 配置
# ============================================================

MODES = (
    "A",
    "B",
    "C",
    "D",
    "E",
)

LEVELS = (
    "L1",
    "L2",
)

MODEL_NAME = "qwen3.8-max"

# 同时并发 6 篇
MAX_WORKERS = 6

# 单篇最多调用 3 次 API
MAX_ATTEMPTS = 3

RETRY_WAIT_S = 8

TIMEOUT_S = 900

TEMPERATURE = 0

SKIP_EXISTING = True


# ============================================================
# 2. 路径
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CLEAN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "05_cleaned"
)

PDF_ROOT = (
    PROJECT_ROOT
    / "data"
    / "00_raw_papers"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "07_image_patches"
)

LOG_DIR = (
    PROJECT_ROOT
    / "data"
    / "07_logs"
)

RAW_FAIL_DIR = (
    LOG_DIR
    / "raw_failures"
)

PROMPT_FILE = (
    PROJECT_ROOT
    / "prompts"
    / "gpt_auto_prompt.txt"
)

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)

USAGE_CSV = (
    LOG_DIR
    / "api_usage_all.csv"
)

RUN_CSV = (
    LOG_DIR
    / "run_log_all.csv"
)

for folder in (
    OUT_ROOT,
    LOG_DIR,
    RAW_FAIL_DIR,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

LOG_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


# ============================================================
# 3. 环境变量
# ============================================================

def load_env(
    path: Path,
) -> dict[str, str]:

    out = {}

    if not path.exists():
        return out

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        out[
            key.strip()
        ] = (
            value.strip()
            .strip('"')
            .strip("'")
        )

    return out


ENV = load_env(
    ENV_FILE
)

API_KEY = (
    os.getenv(
        "DASHSCOPE_API_KEY"
    )
    or os.getenv(
        "API_KEY"
    )
    or ENV.get(
        "DASHSCOPE_API_KEY"
    )
    or ENV.get(
        "API_KEY"
    )
)

BASE_URL = (
    os.getenv(
        "DASHSCOPE_BASE_URL"
    )
    or os.getenv(
        "QWEN_BASE_URL"
    )
    or ENV.get(
        "DASHSCOPE_BASE_URL"
    )
    or ENV.get(
        "QWEN_BASE_URL"
    )
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

CHAT_URL = (
    BASE_URL.rstrip("/")
    + "/chat/completions"
)


# ============================================================
# 4. 基础工具
# ============================================================

def safe_print(
    *args,
) -> None:

    with PRINT_LOCK:
        print(
            *args,
            flush=True,
        )


def read_text(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8"
    )


def stem_key(
    path: Path,
) -> str:

    return (
        path.stem
        .strip()
        .lower()
    )


def build_pdf_index() -> dict[str, Path]:
    """
    PDF_ROOT 只扫描一次。

    如果有多个同名 PDF，
    优先路径层级较浅的那个。
    """

    index: dict[str, Path] = {}

    for pdf in PDF_ROOT.rglob(
        "*.pdf"
    ):

        key = stem_key(
            pdf
        )

        if key not in index:

            index[
                key
            ] = pdf

            continue

        old = index[
            key
        ]

        old_depth = len(
            old.relative_to(
                PDF_ROOT
            ).parts
        )

        new_depth = len(
            pdf.relative_to(
                PDF_ROOT
            ).parts
        )

        if new_depth < old_depth:

            index[
                key
            ] = pdf

    return index


def discover_jobs() -> list[
    tuple[
        str,
        str,
        Path,
    ]
]:
    """
    自动遍历：

    A-E
      ×
    L1/L2
    """

    jobs = []

    for mode in MODES:

        for level in LEVELS:

            folder = (
                CLEAN_ROOT
                / mode
                / level
                / "CLEANED"
            )

            if not folder.exists():
                continue

            files = sorted(
                folder.glob(
                    "*.json"
                ),
                key=lambda x:
                    x.name.lower(),
            )

            for path in files:

                jobs.append(
                    (
                        mode,
                        level,
                        path,
                    )
                )

    return jobs


def encode_pdf(
    path: Path,
) -> str:

    data = (
        path.read_bytes()
    )

    encoded = (
        base64.b64encode(
            data
        ).decode(
            "utf-8"
        )
    )

    return (
        "data:application/pdf;base64,"
        + encoded
    )


# ============================================================
# 5. Prompt
# ============================================================

def build_system_prompt(
    task_prompt: str,
) -> str:

    return (
        "You are a careful scientific information "
        "extraction assistant.\n"

        "Use the PDF only for image/spectroscopic "
        "supplementation.\n"

        "Do not modify existing non-empty values.\n"

        "Return only one JSON object and no prose.\n\n"

        "=== TASK SPECIFICATION BEGIN ===\n"

        f"{task_prompt}\n"

        "=== TASK SPECIFICATION END ===\n"
    )


def build_user_text(
    cleaned_json: str,
    pdf_name: str,
) -> str:

    return (
        f"Attached PDF: {pdf_name}\n\n"

        "Inspect the entire PDF and return only newly "
        "obtained image-derived fields for the existing "
        "synthesis records.\n"

        "Preserve synthesis_id exactly.\n"

        "Omit unsupported fields.\n\n"

        "=== CURRENT CLEANED JSON BEGIN ===\n"

        f"{cleaned_json}\n"

        "=== CURRENT CLEANED JSON END ===\n"
    )


# ============================================================
# 6. API 返回解析
# ============================================================

def message_text(
    content: Any,
) -> str:

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if (
                isinstance(
                    item,
                    dict,
                )
                and "text" in item
            ):

                parts.append(
                    str(
                        item[
                            "text"
                        ]
                    )
                )

        return "\n".join(
            parts
        )

    return str(
        content
    )


def parse_json_text(
    text: str,
) -> dict[str, Any]:

    text = text.strip()

    if text.startswith(
        "```"
    ):

        lines = (
            text.splitlines()
        )

        if (
            lines
            and lines[0].startswith(
                "```"
            )
        ):

            lines = lines[
                1:
            ]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):

            lines = lines[
                :-1
            ]

        text = (
            "\n".join(
                lines
            ).strip()
        )

    left = text.find(
        "{"
    )

    right = text.rfind(
        "}"
    )

    if (
        left >= 0
        and right > left
    ):

        text = text[
            left:
            right + 1
        ]

    obj = json.loads(
        text
    )

    if not isinstance(
        obj,
        dict,
    ):

        raise ValueError(
            "API output is not a JSON object."
        )

    return obj


# ============================================================
# 7. 字段定义
# ============================================================

PRODUCT_FIELDS = {
    "length_mm",
    "diameter_nm",
    "orientation_score",
    "ig_id_ratio",
    "purity_score",
    "bundle_density_per_um",
    "cnt_density_per_um",
}

CALC_FIELDS = {
    "np_size_nm",
}


def norm_key(
    key: Any,
) -> str:

    text = (
        str(
            key
        )
        .strip()
        .casefold()
        .replace(
            "µ",
            "u",
        )
        .replace(
            "μ",
            "u",
        )
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


PRODUCT_MAP = {
    norm_key(
        field
    ): field
    for field
    in PRODUCT_FIELDS
}

PRODUCT_MAP.update(
    {
        "cntlengthmm":
            "length_mm",

        "cntdiameternm":
            "diameter_nm",

        "cntorientationscore":
            "orientation_score",

        "igid":
            "ig_id_ratio",

        "gdratio":
            "ig_id_ratio",

        "cntpurityscore":
            "purity_score",

        "bundledensity":
            "bundle_density_per_um",

        "cntdensity":
            "cnt_density_per_um",
    }
)

CALC_MAP = {
    norm_key(
        field
    ): field
    for field
    in CALC_FIELDS
}

CALC_MAP.update(
    {
        "npsize":
            "np_size_nm",

        "nanoparticlesizenm":
            "np_size_nm",

        "catalystparticlesizenm":
            "np_size_nm",

        "catalystnanoparticlesizenm":
            "np_size_nm",
    }
)

TOP_MAP = {
    "syntheses":
        "syntheses",

    "synthesis":
        "syntheses",

    "product":
        "product",

    "calcblock":
        "calc_block",
}

SYNTHESIS_MAP = {
    "synthesisid":
        "synthesis_id",

    "product":
        "product",

    "calcblock":
        "calc_block",
}


def canonical_leaf(
    key: Any,
) -> tuple[
    str,
    str,
] | None:

    token = norm_key(
        key
    )

    # Purity_score_placeholder
    # -> purity_score
    if token.endswith(
        "placeholder"
    ):

        token = token[
            :-len(
                "placeholder"
            )
        ]

    if token in PRODUCT_MAP:

        return (
            "product",
            PRODUCT_MAP[
                token
            ],
        )

    if token in CALC_MAP:

        return (
            "calc_block",
            CALC_MAP[
                token
            ],
        )

    return None


# ============================================================
# 8. 数值
# ============================================================

NUMBER_RE = re.compile(
    r"^\s*[+-]?"
    r"(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:[eE][+-]?\d+)?\s*$"
)


def as_number(
    value: Any,
) -> float:

    if isinstance(
        value,
        bool,
    ):

        raise ValueError(
            f"Not numeric: {value!r}"
        )

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):

        number = float(
            value
        )

    elif (
        isinstance(
            value,
            str,
        )
        and NUMBER_RE.fullmatch(
            value
        )
    ):

        number = float(
            value
        )

    else:

        raise ValueError(
            f"Not a pure numeric scalar: "
            f"{value!r}"
        )

    if not math.isfinite(
        number
    ):

        raise ValueError(
            f"Non-finite number: "
            f"{value!r}"
        )

    return number


def checked_number(
    field: str,
    value: Any,
) -> float:

    number = as_number(
        value
    )

    if field in {
        "orientation_score",
        "purity_score",
    }:

        if not (
            0 <= number <= 1
        ):

            raise ValueError(
                f"{field} must be "
                f"within [0,1], "
                f"got {number}"
            )

    elif field in {
        "length_mm",
        "diameter_nm",
        "ig_id_ratio",
        "np_size_nm",
    }:

        if number <= 0:

            raise ValueError(
                f"{field} must be >0, "
                f"got {number}"
            )

    elif field in {
        "bundle_density_per_um",
        "cnt_density_per_um",
    }:

        if number < 0:

            raise ValueError(
                f"{field} must be >=0, "
                f"got {number}"
            )

    return number


def has_value(
    value: Any,
) -> bool:

    if value is None:
        return False

    if isinstance(
        value,
        str,
    ):

        return bool(
            value.strip()
        )

    return True


# ============================================================
# 9. 原 JSON
# ============================================================

def build_original_map(
    original: dict[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:

    out = {}

    for synthesis in (
        original.get(
            "syntheses"
        )
        or []
    ):

        if (
            isinstance(
                synthesis,
                dict,
            )
            and synthesis.get(
                "synthesis_id"
            )
        ):

            sid = str(
                synthesis[
                    "synthesis_id"
                ]
            ).strip()

            out[
                sid
            ] = synthesis

    return out


def original_np_exists(
    synthesis: dict[str, Any],
) -> bool:

    calc = (
        synthesis.get(
            "calc_block"
        )
        or {}
    )

    if (
        has_value(
            calc.get(
                "np_size_nm"
            )
        )
        or has_value(
            calc.get(
                "Average_size_M_nanoparticle"
            )
        )
    ):

        return True

    for catalyst in (
        synthesis.get(
            "catalysts"
        )
        or []
    ):

        if not isinstance(
            catalyst,
            dict,
        ):

            continue

        if (
            has_value(
                catalyst.get(
                    "nanoparticle_size_nm"
                )
            )
            or has_value(
                catalyst.get(
                    "catalyst_nanoparticle_size_nm"
                )
            )
        ):

            return True

    return False


def put_number(
    target: dict[
        str,
        float,
    ],
    field: str,
    value: Any,
    context: str,
) -> None:

    number = checked_number(
        field,
        value,
    )

    if field in target:

        if not math.isclose(
            target[
                field
            ],
            number,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):

            raise ValueError(
                f"Conflicting values for "
                f"{context}.{field}: "
                f"{target[field]} vs {number}"
            )

    target[
        field
    ] = number


# ============================================================
# 10. block 自动修复
# ============================================================

def normalize_block(
    raw: Any,
    synthesis_id: str,
    source_block: str,
    product: dict[
        str,
        float,
    ],
    calc_block: dict[
        str,
        float,
    ],
    actions: list[str],
) -> None:

    if raw is None:
        return

    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(
            f"{synthesis_id}."
            f"{source_block} "
            f"must be object."
        )

    for raw_key, value in (
        raw.items()
    ):

        found = canonical_leaf(
            raw_key
        )

        if found is None:

            # 例如：
            #
            # ": 0.85,": 0.85
            #
            # 已经不知道数字是什么意思，
            # 这种坚决不能猜。
            raise ValueError(
                f"{synthesis_id}: "
                f"ambiguous "
                f"{source_block} "
                f"field: {raw_key!r}"
            )

        target_block, field = (
            found
        )

        target = (
            product
            if target_block
            == "product"
            else calc_block
        )

        put_number(
            target,
            field,
            value,
            f"{synthesis_id}."
            f"{target_block}",
        )

        if (
            raw_key != field
            or source_block
            != target_block
        ):

            actions.append(
                f"{synthesis_id}: "
                f"{source_block}."
                f"{raw_key!r}"
                f" -> "
                f"{target_block}."
                f"{field}"
            )


# ============================================================
# 11. 自动 repair
# ============================================================

def repair_patch(
    raw_patch: dict[str, Any],
    original: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[str],
]:

    original_map = (
        build_original_map(
            original
        )
    )

    if not original_map:

        raise ValueError(
            "Original JSON has no valid "
            "synthesis_id."
        )

    actions = []

    # --------------------------------------------------------
    # 顶层
    # --------------------------------------------------------

    top = {}

    for raw_key, value in (
        raw_patch.items()
    ):

        key = TOP_MAP.get(
            norm_key(
                raw_key
            )
        )

        if key is not None:

            if key in top:

                raise ValueError(
                    f"Duplicate top-level "
                    f"field: {key}"
                )

            top[
                key
            ] = value

            continue

        # orientation_score / purity_score
        # 如果直接跑到顶层，
        # 也可以安全归位。
        leaf = canonical_leaf(
            raw_key
        )

        if leaf is None:

            raise ValueError(
                f"Unknown top-level "
                f"field: {raw_key!r}"
            )

        block, field = leaf

        target = (
            top.setdefault(
                block,
                {},
            )
        )

        if not isinstance(
            target,
            dict,
        ):

            raise ValueError(
                f"Top-level "
                f"{block} conflict."
            )

        if (
            field in target
            and target[
                field
            ] != value
        ):

            raise ValueError(
                f"Conflicting "
                f"top-level "
                f"{block}.{field}"
            )

        target[
            field
        ] = value

        actions.append(
            f"move top-level "
            f"{raw_key!r}"
            f" -> "
            f"{block}.{field}"
        )

    syntheses = (
        top.get(
            "syntheses",
            [],
        )
    )

    if isinstance(
        syntheses,
        dict,
    ):

        syntheses = [
            syntheses
        ]

        actions.append(
            "wrap synthesis object "
            "into list"
        )

    if not isinstance(
        syntheses,
        list,
    ):

        raise ValueError(
            "syntheses must be "
            "list/object."
        )

    # --------------------------------------------------------
    # product / calc_block 错放顶层
    # --------------------------------------------------------

    if (
        "product" in top
        or "calc_block" in top
    ):

        if len(
            original_map
        ) != 1:

            raise ValueError(
                "Top-level product/calc_block "
                "ambiguous for multiple syntheses."
            )

        synthesis_id = next(
            iter(
                original_map
            )
        )

        target = None

        for synthesis in syntheses:

            if not isinstance(
                synthesis,
                dict,
            ):

                continue

            sid = synthesis.get(
                "synthesis_id"
            )

            if (
                sid is None
                or str(
                    sid
                ).strip()
                == synthesis_id
            ):

                target = synthesis
                break

        if target is None:

            target = {
                "synthesis_id":
                    synthesis_id
            }

            syntheses.append(
                target
            )

        elif not target.get(
            "synthesis_id"
        ):

            target[
                "synthesis_id"
            ] = synthesis_id

        for block in (
            "product",
            "calc_block",
        ):

            if block not in top:
                continue

            if not isinstance(
                top[
                    block
                ],
                dict,
            ):

                raise ValueError(
                    f"Top-level "
                    f"{block} "
                    f"must be object."
                )

            destination = (
                target.setdefault(
                    block,
                    {},
                )
            )

            if not isinstance(
                destination,
                dict,
            ):

                raise ValueError(
                    f"Nested {block} "
                    f"must be object."
                )

            for key, value in (
                top[
                    block
                ].items()
            ):

                if (
                    key
                    in destination
                    and destination[
                        key
                    ] != value
                ):

                    raise ValueError(
                        f"Conflicting duplicate "
                        f"{block}.{key}"
                    )

                destination[
                    key
                ] = value

            actions.append(
                f"move top-level "
                f"{block}"
                f" -> "
                f"{synthesis_id}."
                f"{block}"
            )

    # --------------------------------------------------------
    # synthesis
    # --------------------------------------------------------

    merged = {}

    for index, raw_synthesis in (
        enumerate(
            syntheses,
            1,
        )
    ):

        if not isinstance(
            raw_synthesis,
            dict,
        ):

            raise ValueError(
                f"syntheses[{index}] "
                f"must be object."
            )

        synthesis = {}

        for raw_key, value in (
            raw_synthesis.items()
        ):

            key = SYNTHESIS_MAP.get(
                norm_key(
                    raw_key
                )
            )

            if key is not None:

                if key in synthesis:

                    raise ValueError(
                        f"Duplicate synthesis "
                        f"field: {key}"
                    )

                synthesis[
                    key
                ] = value

                continue

            # orientation_score 等如果直接跑到 synthesis 层，
            # 自动归到 product/calc_block。
            leaf = canonical_leaf(
                raw_key
            )

            if leaf is None:

                raise ValueError(
                    f"Unknown synthesis "
                    f"field: {raw_key!r}"
                )

            block, field = leaf

            target = (
                synthesis.setdefault(
                    block,
                    {},
                )
            )

            if not isinstance(
                target,
                dict,
            ):

                raise ValueError(
                    f"Synthesis "
                    f"{block} conflict."
                )

            if (
                field in target
                and target[
                    field
                ] != value
            ):

                raise ValueError(
                    f"Conflicting synthesis "
                    f"{block}.{field}"
                )

            target[
                field
            ] = value

            actions.append(
                f"move synthesis-level "
                f"{raw_key!r}"
                f" -> "
                f"{block}.{field}"
            )

        synthesis_id = (
            synthesis.get(
                "synthesis_id"
            )
        )

        if (
            synthesis_id is None
            or not str(
                synthesis_id
            ).strip()
        ):

            if len(
                original_map
            ) != 1:

                raise ValueError(
                    "Missing synthesis_id "
                    "with multiple syntheses."
                )

            synthesis_id = next(
                iter(
                    original_map
                )
            )

            actions.append(
                f"fill missing "
                f"synthesis_id"
                f" -> "
                f"{synthesis_id}"
            )

        synthesis_id = str(
            synthesis_id
        ).strip()

        if synthesis_id not in (
            original_map
        ):

            raise ValueError(
                f"Unknown synthesis_id: "
                f"{synthesis_id}"
            )

        product = {}
        calc_block = {}

        normalize_block(
            synthesis.get(
                "product"
            ),
            synthesis_id,
            "product",
            product,
            calc_block,
            actions,
        )

        normalize_block(
            synthesis.get(
                "calc_block"
            ),
            synthesis_id,
            "calc_block",
            product,
            calc_block,
            actions,
        )

        # ----------------------------------------------------
        # 已有字段不覆盖
        # ----------------------------------------------------

        original_synthesis = (
            original_map[
                synthesis_id
            ]
        )

        original_product = (
            original_synthesis.get(
                "product"
            )
            or {}
        )

        for field in list(
            product
        ):

            if has_value(
                original_product.get(
                    field
                )
            ):

                del product[
                    field
                ]

                actions.append(
                    f"{synthesis_id}: "
                    f"drop existing "
                    f"product.{field}"
                )

        if (
            "np_size_nm"
            in calc_block
            and original_np_exists(
                original_synthesis
            )
        ):

            del calc_block[
                "np_size_nm"
            ]

            actions.append(
                f"{synthesis_id}: "
                f"drop existing "
                f"catalyst NP size"
            )

        # 没有新增信息则不输出该 synthesis
        if (
            not product
            and not calc_block
        ):

            continue

        fixed = {
            "synthesis_id":
                synthesis_id
        }

        if product:

            fixed[
                "product"
            ] = product

        if calc_block:

            fixed[
                "calc_block"
            ] = calc_block

        # 同一 synthesis_id 重复出现
        if synthesis_id not in merged:

            merged[
                synthesis_id
            ] = fixed

        else:

            target = merged[
                synthesis_id
            ]

            for block in (
                "product",
                "calc_block",
            ):

                values = fixed.get(
                    block,
                    {},
                )

                for field, value in (
                    values.items()
                ):

                    destination = (
                        target.setdefault(
                            block,
                            {},
                        )
                    )

                    put_number(
                        destination,
                        field,
                        value,
                        f"{synthesis_id}."
                        f"{block}",
                    )

            actions.append(
                f"{synthesis_id}: "
                f"merge duplicate "
                f"synthesis entries"
            )

    return (
        {
            "syntheses":
                list(
                    merged.values()
                )
        },
        actions,
    )


# ============================================================
# 12. 最终 validator
# ============================================================

def validate_patch(
    patch: dict[str, Any],
    original: dict[str, Any],
) -> None:

    if (
        set(
            patch
        )
        != {
            "syntheses"
        }
    ):

        raise ValueError(
            "Invalid patch root."
        )

    if not isinstance(
        patch[
            "syntheses"
        ],
        list,
    ):

        raise ValueError(
            "syntheses must be list."
        )

    original_map = (
        build_original_map(
            original
        )
    )

    seen = set()

    for synthesis in (
        patch[
            "syntheses"
        ]
    ):

        if not isinstance(
            synthesis,
            dict,
        ):

            raise ValueError(
                "Invalid synthesis object."
            )

        invalid_keys = (
            set(
                synthesis
            )
            - {
                "synthesis_id",
                "product",
                "calc_block",
            }
        )

        if invalid_keys:

            raise ValueError(
                f"Invalid synthesis fields: "
                f"{invalid_keys}"
            )

        synthesis_id = (
            synthesis.get(
                "synthesis_id"
            )
        )

        if (
            not isinstance(
                synthesis_id,
                str,
            )
            or synthesis_id
            not in original_map
            or synthesis_id
            in seen
        ):

            raise ValueError(
                f"Invalid/duplicate "
                f"synthesis_id: "
                f"{synthesis_id!r}"
            )

        seen.add(
            synthesis_id
        )

        product = (
            synthesis.get(
                "product"
            )
        )

        if product is not None:

            if (
                not isinstance(
                    product,
                    dict,
                )
                or not product
            ):

                raise ValueError(
                    f"{synthesis_id}."
                    f"product invalid."
                )

            if (
                set(
                    product
                )
                - PRODUCT_FIELDS
            ):

                raise ValueError(
                    f"{synthesis_id}."
                    f"product contains "
                    f"invalid fields."
                )

            original_product = (
                original_map[
                    synthesis_id
                ].get(
                    "product"
                )
                or {}
            )

            for field, value in (
                product.items()
            ):

                if has_value(
                    original_product.get(
                        field
                    )
                ):

                    raise ValueError(
                        f"{synthesis_id}."
                        f"product.{field} "
                        f"already exists."
                    )

                checked_number(
                    field,
                    value,
                )

        calc_block = (
            synthesis.get(
                "calc_block"
            )
        )

        if calc_block is not None:

            if (
                not isinstance(
                    calc_block,
                    dict,
                )
                or not calc_block
            ):

                raise ValueError(
                    f"{synthesis_id}."
                    f"calc_block invalid."
                )

            if (
                set(
                    calc_block
                )
                - CALC_FIELDS
            ):

                raise ValueError(
                    f"{synthesis_id}."
                    f"calc_block contains "
                    f"invalid fields."
                )

            for field, value in (
                calc_block.items()
            ):

                if (
                    field
                    == "np_size_nm"
                    and original_np_exists(
                        original_map[
                            synthesis_id
                        ]
                    )
                ):

                    raise ValueError(
                        f"{synthesis_id}."
                        f"np_size_nm "
                        f"already exists."
                    )

                checked_number(
                    field,
                    value,
                )


# ============================================================
# 13. API
# ============================================================

def call_qwen(
    pdf_path: Path,
    cleaned_json: str,
    task_prompt: str,
) -> tuple[
    dict[str, Any],
    str,
    str,
]:

    payload = {

        "model":
            MODEL_NAME,

        "temperature":
            TEMPERATURE,


        "enable_thinking":
            True,

        "stream":
            False,

        "response_format": {
            "type":
                "json_object"
        },

        "messages": [

            {
                "role":
                    "system",

                "content":
                    build_system_prompt(
                        task_prompt
                    ),
            },

            {
                "role":
                    "user",

                "content": [

                    {
                        "type":
                            "text",

                        "text":
                            build_user_text(
                                cleaned_json,
                                pdf_path.name,
                            ),
                    },

                    {
                        "type":
                            "file",

                        "file": {

                            "filename":
                                pdf_path.name,

                            "file_data":
                                encode_pdf(
                                    pdf_path
                                ),
                        },
                    },
                ],
            },
        ],
    }

    response = requests.post(

        CHAT_URL,

        headers={
            "Authorization":
                f"Bearer {API_KEY}",

            "Content-Type":
                "application/json",
        },

        json=payload,

        timeout=TIMEOUT_S,
    )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            f"HTTP "
            f"{response.status_code}, "
            f"non-JSON response: "
            f"{response.text[:1000]}"
        )

    if response.status_code >= 400:

        raise RuntimeError(
            f"HTTP "
            f"{response.status_code}: "
            f"{json.dumps(data, ensure_ascii=False)[:2000]}"
        )

    if "error" in data:

        raise RuntimeError(
            f"API error: "
            f"{json.dumps(data['error'], ensure_ascii=False)[:2000]}"
        )

    if not data.get(
        "choices"
    ):

        raise RuntimeError(
            "API response has no choices."
        )

    message = (
        data[
            "choices"
        ][0]
        .get(
            "message",
            {},
        )
    )

    raw_text = (
        message_text(
            message.get(
                "content",
                "",
            )
        )
    )

    reasoning_text = (
            message.get(
                "reasoning_content",
                "",
            )
            or ""
    )

    return (
        data,
        raw_text,
        reasoning_text,
    )


# ============================================================
# 14. 日志
# ============================================================

USAGE_FIELDS = [
    "time",
    "mode",
    "level",
    "model",
    "file",
    "attempt",
    "elapsed_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "usage_raw",
]

RUN_FIELDS = [
    "time",
    "mode",
    "level",
    "file",
    "status",
    "attempts",
    "elapsed_s",
    "pdf",
    "output",
    "note",
]


def append_csv(
    path: Path,
    fields: list[str],
    row: dict[str, Any],
) -> None:

    with LOG_LOCK:

        new_file = (
            not path.exists()
        )

        with path.open(
            "a",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            if new_file:
                writer.writeheader()

            writer.writerow(
                row
            )


def log_usage(
    mode: str,
    level: str,
    file_name: str,
    attempt: int,
    elapsed_s: float,
    response: dict[str, Any],
) -> None:

    usage = (
        response.get(
            "usage",
            {},
        )
    )

    append_csv(
        USAGE_CSV,
        USAGE_FIELDS,
        {
            "time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "mode":
                mode,

            "level":
                level,

            "model":
                MODEL_NAME,

            "file":
                file_name,

            "attempt":
                attempt,

            "elapsed_s":
                round(
                    elapsed_s,
                    2,
                ),

            "prompt_tokens":
                usage.get(
                    "prompt_tokens",
                    "",
                ),

            "completion_tokens":
                usage.get(
                    "completion_tokens",
                    "",
                ),

            "total_tokens":
                usage.get(
                    "total_tokens",
                    "",
                ),

            "usage_raw":
                json.dumps(
                    usage,
                    ensure_ascii=False,
                ),
        },
    )


def log_run(
    mode: str,
    level: str,
    file_name: str,
    status: str,
    attempts: int,
    elapsed_s: float,
    pdf_path: str,
    out_path: str,
    note: str = "",
) -> None:

    append_csv(
        RUN_CSV,
        RUN_FIELDS,
        {
            "time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "mode":
                mode,

            "level":
                level,

            "file":
                file_name,

            "status":
                status,

            "attempts":
                attempts,

            "elapsed_s":
                round(
                    elapsed_s,
                    2,
                ),

            "pdf":
                pdf_path,

            "output":
                out_path,

            "note":
                note,
        },
    )


def save_failure(
    mode: str,
    level: str,
    stem: str,
    attempt: int,
    error: str,
    raw_text: str,
) -> None:

    folder = (
        RAW_FAIL_DIR
        / mode
        / level
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        folder
        / (
            f"{stem}"
            f"__attempt{attempt}.txt"
        )
    )

    path.write_text(
        (
            "ERROR:\n"
            f"{error}\n\n"
            "RAW RESPONSE:\n"
            f"{raw_text}"
        ),
        encoding="utf-8",
    )


# ============================================================
# 15. 单篇
# ============================================================

def run_job(
    job: tuple[
        str,
        str,
        Path,
    ],
    pdf_index: dict[
        str,
        Path,
    ],
    task_prompt: str,
) -> str:

    mode, level, json_path = (
        job
    )

    started = time.time()

    out_dir = (
        OUT_ROOT
        / mode
        / level
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / (
            f"{json_path.stem}"
            f".rejson"
        )
    )

    # --------------------------------------------------------
    # 原始 JSON
    # --------------------------------------------------------

    try:

        cleaned_json = read_text(
            json_path
        )

        original = json.loads(
            cleaned_json
        )

    except Exception as error:

        log_run(
            mode,
            level,
            json_path.name,
            "FAIL_JSON",
            0,
            time.time() - started,
            "",
            str(
                out_path
            ),
            str(
                error
            ),
        )

        return "FAIL"

    # --------------------------------------------------------
    # 已有结果
    # --------------------------------------------------------

    if (
        SKIP_EXISTING
        and out_path.exists()
    ):

        try:

            existing = json.loads(
                read_text(
                    out_path
                )
            )

            fixed, actions = (
                repair_patch(
                    existing,
                    original,
                )
            )

            validate_patch(
                fixed,
                original,
            )

            if actions:

                out_path.write_text(
                    json.dumps(
                        fixed,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            log_run(
                mode,
                level,
                json_path.name,
                "SKIP_EXISTING",
                0,
                time.time()
                - started,
                "",
                str(
                    out_path
                ),
                "; ".join(
                    actions
                ),
            )

            safe_print(
                f"[SKIP] "
                f"{mode}/{level} | "
                f"{json_path.name}"
            )

            return "SKIP"

        except Exception:
            pass

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_path = (
        pdf_index.get(
            stem_key(
                json_path
            )
        )
    )

    if pdf_path is None:

        log_run(
            mode,
            level,
            json_path.name,
            "FAIL_NO_PDF",
            0,
            time.time()
            - started,
            "",
            str(
                out_path
            ),
        )

        safe_print(
            f"[NO PDF] "
            f"{mode}/{level} | "
            f"{json_path.name}"
        )

        return "FAIL"

    last_error = ""

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        raw_text = ""
        reasoning_text = ""

        try:

            safe_print(
                f"[API "
                f"{attempt}/"
                f"{MAX_ATTEMPTS}] "
                f"{mode}/{level} | "
                f"{json_path.name}"
            )

            api_started = (
                time.time()
            )

            response, raw_text, reasoning_text = (
                call_qwen(
                    pdf_path,
                    cleaned_json,
                    task_prompt,
                )
            )

            reasoning_dir = (
                    LOG_DIR
                    / "reasoning"
                    / mode
                    / level
            )

            reasoning_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            reasoning_path = (
                    reasoning_dir
                    / f"{json_path.stem}__reasoning.txt"
            )

            reasoning_path.write_text(
                reasoning_text,
                encoding="utf-8",
            )

            api_elapsed = (
                time.time()
                - api_started
            )

            log_usage(
                mode,
                level,
                json_path.name,
                attempt,
                api_elapsed,
                response,
            )

            raw_patch = (
                parse_json_text(
                    raw_text
                )
            )

            fixed, actions = (
                repair_patch(
                    raw_patch,
                    original,
                )
            )

            validate_patch(
                fixed,
                original,
            )

            out_path.write_text(
                json.dumps(
                    fixed,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            status = (
                "OK_REPAIRED"
                if actions
                else "OK"
            )

            log_run(
                mode,
                level,
                json_path.name,
                status,
                attempt,
                time.time()
                - started,
                str(
                    pdf_path
                ),
                str(
                    out_path
                ),
                "; ".join(
                    actions
                ),
            )

            safe_print(
                f"[{status}] "
                f"{mode}/{level} | "
                f"{json_path.name}"
            )

            return "OK"

        except Exception as error:

            last_error = str(
                error
            )

            save_failure(
                mode,
                level,
                json_path.stem,
                attempt,
                last_error,
                raw_text,
            )

            safe_print(
                f"[RETRY-FAIL] "
                f"{mode}/{level} | "
                f"attempt={attempt} | "
                f"{last_error}"
            )

            if attempt < MAX_ATTEMPTS:

                time.sleep(
                    RETRY_WAIT_S
                )

    log_run(
        mode,
        level,
        json_path.name,
        "FAIL",
        MAX_ATTEMPTS,
        time.time()
        - started,
        str(
            pdf_path
        ),
        str(
            out_path
        ),
        last_error,
    )

    return "FAIL"


# ============================================================
# 16. 主程序
# ============================================================

def main() -> None:

    if not API_KEY:

        raise RuntimeError(
            "No DASHSCOPE_API_KEY "
            "found in project .env"
        )

    if not PDF_ROOT.exists():

        raise FileNotFoundError(
            PDF_ROOT
        )

    if not PROMPT_FILE.exists():

        raise FileNotFoundError(
            PROMPT_FILE
        )

    task_prompt = read_text(
        PROMPT_FILE
    )

    jobs = discover_jobs()

    pdf_index = build_pdf_index()

    if not jobs:

        raise RuntimeError(
            "No A-E / L1-L2 "
            "cleaned JSON jobs found."
        )

    # --------------------------------------------------------
    # 数量统计
    # --------------------------------------------------------

    counts = {}

    for mode, level, _ in jobs:

        key = (
            mode,
            level,
        )

        counts[
            key
        ] = (
            counts.get(
                key,
                0,
            )
            + 1
        )

    safe_print(
        "=" * 80
    )

    safe_print(
        "Qwen PDF Image Supplementation"
    )

    safe_print(
        f"MODEL       : "
        f"{MODEL_NAME}"
    )

    safe_print(
        f"CONCURRENCY : "
        f"{MAX_WORKERS}"
    )

    safe_print(
        f"TOTAL JOBS  : "
        f"{len(jobs)}"
    )

    safe_print(
        f"PDF INDEX   : "
        f"{len(pdf_index)}"
    )

    for mode in MODES:

        for level in LEVELS:

            safe_print(
                f"  {mode}/{level}: "
                f"{counts.get((mode, level), 0)}"
            )

    safe_print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 并发
    # --------------------------------------------------------

    started = time.time()

    done = 0
    ok = 0
    skip = 0
    fail = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                run_job,
                job,
                pdf_index,
                task_prompt,
            ):
                job

            for job in jobs
        }

        for future in as_completed(
            futures
        ):

            done += 1

            try:

                status = (
                    future.result()
                )

            except Exception as error:

                status = "FAIL"

                mode, level, path = (
                    futures[
                        future
                    ]
                )

                safe_print(
                    f"[UNCAUGHT] "
                    f"{mode}/{level} | "
                    f"{path.name} | "
                    f"{error}"
                )

            if status == "OK":

                ok += 1

            elif status == "SKIP":

                skip += 1

            else:

                fail += 1

            elapsed = (
                time.time()
                - started
            )

            rate = (
                done / elapsed
                if elapsed > 0
                else 0
            )

            eta = (
                (
                    len(jobs)
                    - done
                )
                / rate
                if rate > 0
                else 0
            )

            safe_print(
                f"[PROGRESS] "
                f"{done}/"
                f"{len(jobs)} | "
                f"OK={ok} "
                f"SKIP={skip} "
                f"FAIL={fail} | "
                f"elapsed="
                f"{timedelta(seconds=int(elapsed))} | "
                f"ETA="
                f"{timedelta(seconds=int(eta))}"
            )

    elapsed = (
        time.time()
        - started
    )

    safe_print(
        "=" * 80
    )

    safe_print(
        "BATCH FINISHED"
    )

    safe_print(
        f"TOTAL   : "
        f"{len(jobs)}"
    )

    safe_print(
        f"OK      : "
        f"{ok}"
    )

    safe_print(
        f"SKIP    : "
        f"{skip}"
    )

    safe_print(
        f"FAIL    : "
        f"{fail}"
    )

    safe_print(
        f"ELAPSED : "
        f"{timedelta(seconds=int(elapsed))}"
    )

    safe_print(
        f"OUTPUT  : "
        f"{OUT_ROOT}"
    )

    safe_print(
        f"USAGE   : "
        f"{USAGE_CSV}"
    )

    safe_print(
        f"RUN LOG : "
        f"{RUN_CSV}"
    )

    safe_print(
        "=" * 80
    )


if __name__ == "__main__":

    main()