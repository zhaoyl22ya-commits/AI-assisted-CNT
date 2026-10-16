# -*- coding: utf-8 -*-
"""
representation_baseline.py

目的
----
在完全相同的 L1 样本、目标、外层划分、LightGBM 参数和 early-stopping
规则下，对比三套输入表征：

1. raw_plus_missing
   从 data/08_finalized 直接读取常规实验变量与常规计算描述符：
   催化剂组成、前驱体/促进剂、温度、压力、时间、气体组成、碳源、
   NP 尺寸、C/M、C/NP、S/M、浓度和每颗粒金属原子数，以及缺失标记。

2. physical_60D
   N1-N18 + C1-C12 + F1-F18 + P1-P12。

3. physical_plus_missing_120D
   60 个物理特征 + 60 个 miss_* 缺失标记。

评价目标
--------
- Growth quality
- IG/ID
- Orientation
- Composite（至少 2 项性能分量）

严格规则
--------
- 不重新生成 N/C/F/P；
- 不调用 API；
- 不修改 L1、L12、L123 或 WorldMap；
- 直接复用 train_l1_targets.py 和 train_l1_composite.py 的 split manifest；
- 三套输入在每个 fold 中使用完全相同的训练样本和测试样本；
- Composite 的归一化仍只由该外层训练集拟合；
- Raw 输入采用白名单，绝不读取 product、quality_evidence、calc_steps 或 notes，
  避免目标泄漏；
- 本代码只用于文章中的 representation baseline，不导出部署模型。

放置位置
--------
<project_dir>/models/external_validation/representation_baseline.py

运行
----
快速检查：
python representation_baseline.py --bootstrap 200 --dpi 200

正式运行：
python representation_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "08_finalized"
ONLY_LEVEL = "L1"

DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "11_flattened"
    / "ml_dataset_3p0_full60D.csv"
)

SCHEMA_JSON = (
    PROJECT_ROOT
    / "data"
    / "11_flattened"
    / "feature_schema.json"
)

SINGLE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "l1_targets"
    / "split_manifest.csv"
)

COMPOSITE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "l1_composite"
    / "composite_split_manifest.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "external_validation"
    / "representation_baseline"
)

FIGURE_DIR = OUT_DIR / "figures"
SOURCE_DATA_DIR = OUT_DIR / "source_data"

SEED = 42
INNER_VALID_SIZE = 0.15
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200

MIN_TRAIN_SAMPLES = 40
MIN_TEST_SAMPLES = 10
MIN_TRAIN_PAPERS = 5
MIN_TEST_PAPERS = 2

MIN_COMPOSITE_COMPONENTS = 2
MIN_CATEGORY_FREQUENCY = 3

DEFAULT_BOOTSTRAP = 2000
DEFAULT_DPI = 600

SPLITS = (
    "paper_grouped_cv",
    "temporal_holdout",
)

SPLIT_LABELS = {
    "paper_grouped_cv": "Paper-grouped CV",
    "temporal_holdout": "Temporal holdout",
}

TARGETS = {
    "growth_quality": "y_growth_quality_score",
    "igid": "y_ig_id_ratio_unified",
    "orientation": "y_orientation_score",
    "composite": "y_composite_final_score",
}

TARGET_LABELS = {
    "growth_quality": "Growth quality",
    "igid": "IG/ID",
    "orientation": "Orientation",
    "composite": "Composite≥2",
}

REPRESENTATIONS = (
    "raw_plus_missing",
    "physical_60D",
    "physical_plus_missing_120D",
)

REPRESENTATION_LABELS = {
    "raw_plus_missing": "Raw + missing",
    "physical_60D": "Physical 60D",
    "physical_plus_missing_120D": "Physical + missing 120D",
}

PAIRWISE_COMPARISONS = (
    (
        "physical_60D",
        "raw_plus_missing",
        "Physical 60D − Raw",
    ),
    (
        "physical_plus_missing_120D",
        "raw_plus_missing",
        "Physical 120D − Raw",
    ),
    (
        "physical_plus_missing_120D",
        "physical_60D",
        "Physical 120D − Physical 60D",
    ),
)

LGB_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": ["l2", "l1"],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
}

COMPOSITE_WEIGHTS = {
    "gq": 0.30,
    "igid": 0.20,
    "ori": 0.20,
    "purity": 0.15,
    "bundle": 0.15,
}

COL_GQ = "y_growth_quality_score"
COL_IGID = "y_ig_id_ratio_unified"
COL_IGID_PREF = "y_ig_id_ratio"
COL_IGID_LEGACY = "y_IG_ID"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

RAW_GQ = "_raw_gq"
RAW_IGID = "_raw_igid"
RAW_ORI = "_raw_ori"
RAW_PURITY = "_raw_purity"
RAW_BUNDLE_LOG = "_raw_bundle_log1p"
RAW_COMPONENT_COUNT = "_raw_component_count"

RAW_NUMERIC_COLUMNS = [
    "raw_catalyst_count",
    "raw_metal_count",
    "raw_main_metal_fraction",
    "raw_second_metal_fraction",
    "raw_promoter_feed_rate_log1p",
    "raw_temperature_C",
    "raw_pressure_log1p",
    "raw_growth_time_log1p",
    "raw_np_size_log1p",
    "raw_C_M_ratio_log1p",
    "raw_C_NP_ratio_log1p",
    "raw_S_M_ratio_log1p",
    "raw_C_concentration_log1p",
    "raw_M_concentration_log1p",
    "raw_S_concentration_log1p",
    "raw_metal_atoms_per_np_log1p",
]

RAW_GROUP_MISSING_COLUMNS = [
    "miss_raw_catalysts",
    "miss_raw_metals",
    "miss_raw_precursor",
    "miss_raw_catalyst_type",
    "miss_raw_promoter",
    "miss_raw_gas_composition",
    "miss_raw_carbon_source",
]

RAW_EXTRA_COLUMNS = [
    "raw_metal_fraction_assumed_equal",
]

NUMBER_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)

GAS_ALIASES = {
    "hydrogen": "H2",
    "h2": "H2",
    "argon": "Ar",
    "ar": "Ar",
    "nitrogen": "N2",
    "n2": "N2",
    "helium": "He",
    "he": "He",
    "methane": "CH4",
    "ch4": "CH4",
    "ethylene": "C2H4",
    "c2h4": "C2H4",
    "acetylene": "C2H2",
    "c2h2": "C2H2",
    "carbonmonoxide": "CO",
    "co": "CO",
    "carbondioxide": "CO2",
    "co2": "CO2",
    "ammonia": "NH3",
    "nh3": "NH3",
}

CARBON_SOURCE_ALIASES = {
    "methane": "CH4",
    "ch4": "CH4",
    "ethylene": "C2H4",
    "c2h4": "C2H4",
    "acetylene": "C2H2",
    "c2h2": "C2H2",
    "carbonmonoxide": "CO",
    "co": "CO",
    "carbondioxide": "CO2",
    "co2": "CO2",
    "ethylalcohol": "ethanol",
    "ethanol": "ethanol",
    "methylalcohol": "methanol",
    "methanol": "methanol",
    "toluene": "toluene",
    "benzene": "benzene",
    "xylene": "xylene",
}


def text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalized_text(value: Any) -> str:
    value = html.unescape(text(value)).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalized_compact(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalized_text(value),
    )


def normalized_id(value: Any) -> str:
    return normalized_text(value)


def safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None

    if isinstance(value, dict):
        for key in (
            "value",
            "mean",
            "average",
            "median",
        ):
            if key in value:
                result = safe_float(value.get(key))
                if result is not None:
                    return result

        return None

    raw = text(value)

    if not raw:
        return None

    numbers = [
        float(item)
        for item in NUMBER_RE.findall(raw)
    ]

    if not numbers:
        return None

    result = (
        numbers[0]
        if len(numbers) == 1
        else (
            numbers[0]
            + numbers[1]
        )
        / 2.0
    )

    return (
        result
        if math.isfinite(result)
        else None
    )


def log1p_nonnegative(value: Any) -> float:
    result = safe_float(value)

    if result is None or result < 0:
        return np.nan

    return float(
        np.log1p(result)
    )


def safe_name(value: Any) -> str:
    token = normalized_text(value)

    token = re.sub(
        r"[^a-z0-9]+",
        "_",
        token,
    ).strip("_")

    return token or "unknown"


def short_hash(value: str) -> str:
    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:6]


def make_unique_category_columns(
    prefix: str,
    categories: Sequence[str],
) -> Dict[str, str]:
    output: Dict[str, str] = {}
    used: Dict[str, str] = {}

    for category in sorted(
        set(categories)
    ):
        base = (
            f"{prefix}_{safe_name(category)}"
        )

        column = base

        if (
            column in used
            and used[column]
            != category
        ):
            column = (
                f"{base}_{short_hash(category)}"
            )

        used[column] = category
        output[category] = column

    return output


def normalize_gas(value: Any) -> str:
    compact = normalized_compact(value)

    return GAS_ALIASES.get(
        compact,
        text(value),
    )


def normalize_carbon_source(value: Any) -> str:
    compact = normalized_compact(value)

    if not compact:
        return ""

    return CARBON_SOURCE_ALIASES.get(
        compact,
        normalized_text(value),
    )


def normalize_category(value: Any) -> str:
    return normalized_text(value)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "JSON root is not an object"
        )

    return data


def write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def write_json(
    data: Mapping[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def safe_r2(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    left = np.asarray(
        y_true,
        dtype=float,
    )

    right = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.isfinite(left)
        & np.isfinite(right)
    )

    left = left[valid]
    right = right[valid]

    if (
        len(left) < 2
        or np.allclose(
            left,
            left[0],
        )
    ):
        return float("nan")

    return float(
        r2_score(
            left,
            right,
        )
    )


def safe_spearman(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    left = np.asarray(
        y_true,
        dtype=float,
    )

    right = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.isfinite(left)
        & np.isfinite(right)
    )

    left = left[valid]
    right = right[valid]

    if len(left) < 2:
        return float("nan")

    left_rank = (
        pd.Series(left)
        .rank(method="average")
        .to_numpy(float)
    )

    right_rank = (
        pd.Series(right)
        .rank(method="average")
        .to_numpy(float)
    )

    if (
        np.std(left_rank) == 0
        or np.std(right_rank) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            left_rank,
            right_rank,
        )[0, 1]
    )


def evaluate_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    left = np.asarray(
        y_true,
        dtype=float,
    )

    right = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.isfinite(left)
        & np.isfinite(right)
    )

    left = left[valid]
    right = right[valid]

    if len(left) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }

    return {
        "n": int(len(left)),

        "mae": float(
            mean_absolute_error(
                left,
                right,
            )
        ),

        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    left,
                    right,
                )
            )
        ),

        "r2": safe_r2(
            left,
            right,
        ),

        "spearman": safe_spearman(
            left,
            right,
        ),
    }


def load_schema() -> Tuple[
    List[str],
    List[str],
]:
    with SCHEMA_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(handle)

    physical = schema.get(
        "physical_feature_columns"
    )

    full = schema.get(
        "model_feature_columns"
    )

    if (
        not isinstance(
            physical,
            list,
        )
        or not physical
    ):
        raise RuntimeError(
            "feature_schema.json 缺少 "
            "physical_feature_columns。"
        )

    if (
        not isinstance(
            full,
            list,
        )
        or not full
    ):
        raise RuntimeError(
            "feature_schema.json 缺少 "
            "model_feature_columns。"
        )

    physical = [
        str(column)
        for column in physical
    ]

    full = [
        str(column)
        for column in full
    ]

    if (
        len(physical) != 60
        or len(full) != 120
    ):
        raise RuntimeError(
            "当前 schema 应为 60 个物理特征"
            "和 120 个完整特征，"
            f"实际为 "
            f"{len(physical)} / {len(full)}。"
        )

    return (
        physical,
        full,
    )


def ensure_igid(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    preferred = (
        pd.to_numeric(
            frame[
                COL_IGID_PREF
            ],
            errors="coerce",
        )
        if COL_IGID_PREF
        in frame.columns
        else pd.Series(
            np.nan,
            index=frame.index,
        )
    )

    legacy = (
        pd.to_numeric(
            frame[
                COL_IGID_LEGACY
            ],
            errors="coerce",
        )
        if COL_IGID_LEGACY
        in frame.columns
        else pd.Series(
            np.nan,
            index=frame.index,
        )
    )

    output = frame.copy()

    output[
        COL_IGID
    ] = preferred.where(
        preferred.notna(),
        legacy,
    )

    return output


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(
        DATASET_CSV,
        low_memory=False,
    )

    required = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "growth_mode",
        "data_level",
        "source_file",
        "synthesis_id",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "ml_dataset_3p0_full60D.csv 缺少："
            f"{missing}"
        )

    frame = frame[
        frame[
            "data_level"
        ]
        .astype(str)
        .str.upper()
        == ONLY_LEVEL
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "筛选 data_level == L1 后没有数据。"
        )

    frame = ensure_igid(
        frame
    )

    frame[
        "sample_id"
    ] = (
        frame[
            "sample_id"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "paper_id"
    ] = (
        frame[
            "paper_id"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "growth_mode"
    ] = (
        frame[
            "growth_mode"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "data_level"
    ] = (
        frame[
            "data_level"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "source_file"
    ] = (
        frame[
            "source_file"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "synthesis_id"
    ] = (
        frame[
            "synthesis_id"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "paper_year_int"
    ] = pd.to_numeric(
        frame[
            "paper_year_int"
        ],
        errors="coerce",
    )

    if (
        frame[
            "sample_id"
        ].duplicated().any()
    ):
        examples = (
            frame.loc[
                frame[
                    "sample_id"
                ].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "sample_id 不唯一，示例："
            f"{examples}"
        )

    occurrence_keys = [
        "growth_mode",
        "data_level",
        "source_file",
        "synthesis_id",
    ]

    frame[
        "_source_occurrence"
    ] = (
        frame.groupby(
            occurrence_keys,
            dropna=False,
        )
        .cumcount()
        .astype(int)
    )

    return frame.reset_index(
        drop=True
    )


def source_stem(
    source_file: Any,
) -> str:
    name = os.path.basename(
        text(source_file)
    )

    lower = name.lower()

    if lower.endswith(
        ".andjson"
    ):
        return name[:-8]

    if lower.endswith(
        ".json"
    ):
        return name[:-5]

    return os.path.splitext(
        name
    )[0]


def build_folder_file_map(
    folder: Path,
) -> Dict[str, Path]:
    if not folder.is_dir():
        return {}

    output: Dict[
        str,
        Path,
    ] = {}

    for path in folder.iterdir():
        if (
            path.is_file()
            and path.suffix.lower()
            == ".json"
        ):
            output[
                path.stem.lower()
            ] = path

    return output


def resolve_source_json(
    mode: str,
    level: str,
    source_file: str,
    folder_maps: Dict[
        Tuple[str, str],
        Dict[str, Path],
    ],
) -> Tuple[
    Optional[Path],
    Path,
]:
    stem = source_stem(
        source_file
    )

    folder = (
        RAW_ROOT
        / mode
        / level
    )

    expected = (
        folder
        / f"{stem}.json"
    )

    if expected.is_file():
        return (
            expected,
            expected,
        )

    key = (
        mode,
        level,
    )

    if key not in folder_maps:
        folder_maps[
            key
        ] = build_folder_file_map(
            folder
        )

    found = (
        folder_maps[
            key
        ].get(
            stem.lower()
        )
    )

    return (
        found,
        expected,
    )


def synthesis_lists_by_id(
    document: Mapping[
        str,
        Any,
    ],
) -> Dict[
    str,
    List[
        Dict[
            str,
            Any,
        ]
    ],
]:
    output: Dict[
        str,
        List[
            Dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    syntheses = document.get(
        "syntheses"
    )

    if not isinstance(
        syntheses,
        list,
    ):
        return output

    for synthesis in syntheses:
        if not isinstance(
            synthesis,
            dict,
        ):
            continue

        sid = normalized_id(
            synthesis.get(
                "synthesis_id"
            )
        )

        output[
            sid
        ].append(
            synthesis
        )

    return output


def attach_raw_syntheses(
    frame: pd.DataFrame,
) -> Tuple[
    List[
        Optional[
            Dict[
                str,
                Any,
            ]
        ]
    ],
    pd.DataFrame,
]:
    document_cache: Dict[
        Path,
        Dict[str, Any],
    ] = {}

    synthesis_cache: Dict[
        Path,
        Dict[
            str,
            List[
                Dict[
                    str,
                    Any,
                ]
            ],
        ],
    ] = {}

    folder_maps: Dict[
        Tuple[str, str],
        Dict[str, Path],
    ] = {}

    records: List[
        Optional[
            Dict[
                str,
                Any,
            ]
        ]
    ] = []

    audit_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for _, row in frame.iterrows():
        mode = text(
            row[
                "growth_mode"
            ]
        )

        level = text(
            row[
                "data_level"
            ]
        )

        source_file = text(
            row[
                "source_file"
            ]
        )

        sid = normalized_id(
            row[
                "synthesis_id"
            ]
        )

        occurrence = int(
            row[
                "_source_occurrence"
            ]
        )

        (
            path,
            expected,
        ) = resolve_source_json(
            mode,
            level,
            source_file,
            folder_maps,
        )

        status = "ok"
        error = ""

        synthesis: Optional[
            Dict[
                str,
                Any,
            ]
        ] = None

        duplicate_count = 0

        if path is None:
            status = (
                "json_not_found"
            )

        else:
            try:
                if (
                    path
                    not in document_cache
                ):
                    document_cache[
                        path
                    ] = read_json(
                        path
                    )

                    synthesis_cache[
                        path
                    ] = synthesis_lists_by_id(
                        document_cache[
                            path
                        ]
                    )

                matches = (
                    synthesis_cache[
                        path
                    ].get(
                        sid,
                        [],
                    )
                )

                duplicate_count = len(
                    matches
                )

                if not matches:
                    status = (
                        "synthesis_id_not_found"
                    )

                elif (
                    occurrence
                    >= len(matches)
                ):
                    status = (
                        "synthesis_occurrence_not_found"
                    )

                else:
                    synthesis = matches[
                        occurrence
                    ]

            except Exception as exc:
                status = (
                    "json_read_failed"
                )

                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        records.append(
            synthesis
        )

        audit_rows.append(
            {
                "sample_id":
                    row[
                        "sample_id"
                    ],

                "paper_id":
                    row[
                        "paper_id"
                    ],

                "growth_mode":
                    mode,

                "data_level":
                    level,

                "source_file":
                    source_file,

                "synthesis_id":
                    row[
                        "synthesis_id"
                    ],

                "synthesis_occurrence":
                    occurrence,

                "expected_raw_json_path":
                    str(
                        expected
                    ),

                "resolved_raw_json_path":
                    (
                        str(path)
                        if path
                        is not None
                        else ""
                    ),

                "json_file_found":
                    int(
                        path
                        is not None
                    ),

                "synthesis_found":
                    int(
                        synthesis
                        is not None
                    ),

                "same_id_count_in_source":
                    duplicate_count,

                "status":
                    status,

                "error":
                    error,
            }
        )

    return (
        records,
        pd.DataFrame(
            audit_rows
        ),
    )


def catalyst_summary(
    synthesis: Mapping[
        str,
        Any,
    ],
) -> Dict[
    str,
    Any,
]:
    catalysts = synthesis.get(
        "catalysts"
    )

    catalyst_list = (
        [
            item
            for item in catalysts
            if isinstance(
                item,
                dict,
            )
        ]
        if isinstance(
            catalysts,
            list,
        )
        else []
    )

    precursor_values: List[str] = []
    catalyst_type_values: List[str] = []
    promoter_values: List[str] = []
    promoter_rates: List[float] = []

    catalyst_compositions: List[
        Dict[
            str,
            float,
        ]
    ] = []

    assumed_equal = False

    for catalyst in catalyst_list:
        precursor = normalize_category(
            catalyst.get(
                "precursor"
            )
        )

        if precursor:
            precursor_values.append(
                precursor
            )

        catalyst_type = normalize_category(
            catalyst.get(
                "catalyst_type"
            )
        )

        if catalyst_type:
            catalyst_type_values.append(
                catalyst_type
            )

        promoter = catalyst.get(
            "promoter"
        )

        if isinstance(
            promoter,
            dict,
        ):
            promoter_name = normalize_category(
                promoter.get(
                    "name"
                )
            )

            if promoter_name:
                promoter_values.append(
                    promoter_name
                )

            promoter_rate = safe_float(
                promoter.get(
                    "feed_rate_mol_s"
                )
            )

            if (
                promoter_rate
                is not None
                and promoter_rate >= 0
            ):
                promoter_rates.append(
                    promoter_rate
                )

        metals = catalyst.get(
            "metals"
        )

        metal_rows = (
            [
                item
                for item in metals
                if isinstance(
                    item,
                    dict,
                )
            ]
            if isinstance(
                metals,
                list,
            )
            else []
        )

        valid_elements = [
            text(
                item.get(
                    "element"
                )
            )
            for item
            in metal_rows
            if text(
                item.get(
                    "element"
                )
            )
        ]

        if not valid_elements:
            continue

        percentages = [
            safe_float(
                item.get(
                    "atomic_percentage"
                )
            )
            for item
            in metal_rows
            if text(
                item.get(
                    "element"
                )
            )
        ]

        use_reported = (
            len(percentages)
            == len(
                valid_elements
            )
            and all(
                value
                is not None
                and value >= 0
                for value
                in percentages
            )
            and sum(
                float(value)
                for value
                in percentages
                if value
                is not None
            )
            > 0
        )

        if use_reported:
            total = sum(
                float(value)
                for value
                in percentages
                if value
                is not None
            )

            fractions = [
                float(value)
                / total
                for value
                in percentages
                if value
                is not None
            ]

        else:
            fractions = [
                1.0
                / len(
                    valid_elements
                )
            ] * len(
                valid_elements
            )

            assumed_equal = True

        composition: Dict[
            str,
            float,
        ] = defaultdict(
            float
        )

        for (
            element,
            fraction,
        ) in zip(
            valid_elements,
            fractions,
        ):
            composition[
                element
            ] += float(
                fraction
            )

        catalyst_compositions.append(
            dict(
                composition
            )
        )

    aggregate: Dict[
        str,
        float,
    ] = defaultdict(
        float
    )

    if catalyst_compositions:
        weight = (
            1.0
            / len(
                catalyst_compositions
            )
        )

        for composition in catalyst_compositions:
            for (
                element,
                fraction,
            ) in composition.items():
                aggregate[
                    element
                ] += (
                    weight
                    * fraction
                )

    ordered = sorted(
        aggregate.values(),
        reverse=True,
    )

    return {
        "catalyst_count":
            (
                len(
                    catalyst_list
                )
                if catalyst_list
                else None
            ),

        "metal_count":
            (
                len(
                    aggregate
                )
                if aggregate
                else None
            ),

        "main_metal_fraction":
            (
                ordered[0]
                if ordered
                else None
            ),

        "second_metal_fraction":
            (
                ordered[1]
                if len(
                    ordered
                ) > 1
                else (
                    0.0
                    if len(
                        ordered
                    ) == 1
                    else None
                )
            ),

        "metal_fractions":
            dict(
                aggregate
            ),

        "metal_fraction_assumed_equal":
            float(
                assumed_equal
            ),

        "precursors":
            sorted(
                set(
                    precursor_values
                )
            ),

        "catalyst_types":
            sorted(
                set(
                    catalyst_type_values
                )
            ),

        "promoters":
            sorted(
                set(
                    promoter_values
                )
            ),

        "promoter_feed_rate":
            (
                float(
                    sum(
                        promoter_rates
                    )
                )
                if promoter_rates
                else None
            ),

        "catalysts_reported":
            (
                isinstance(
                    catalysts,
                    list,
                )
                and len(
                    catalyst_list
                ) > 0
            ),

        "metals_reported":
            bool(
                aggregate
            ),
    }


def gas_summary(
    process: Mapping[
        str,
        Any,
    ],
) -> Tuple[
    Dict[
        str,
        float,
    ],
    bool,
]:
    gas_composition = process.get(
        "gas_composition"
    )

    if (
        not isinstance(
            gas_composition,
            dict,
        )
        or not gas_composition
    ):
        return (
            {},
            False,
        )

    output: Dict[
        str,
        float,
    ] = defaultdict(
        float
    )

    for (
        gas,
        fraction_raw,
    ) in gas_composition.items():
        gas_name = normalize_gas(
            gas
        )

        fraction = safe_float(
            fraction_raw
        )

        if (
            not gas_name
            or fraction
            is None
            or fraction < 0
        ):
            continue

        output[
            gas_name
        ] += fraction

    total = sum(
        output.values()
    )

    if total > 0:
        output = defaultdict(
            float,
            {
                gas:
                    value
                    / total
                for (
                    gas,
                    value,
                )
                in output.items()
            },
        )

    return (
        dict(
            output
        ),
        bool(
            output
        ),
    )


def first_numeric(
    calc: Mapping[
        str,
        Any,
    ],
    keys: Sequence[str],
) -> Optional[float]:
    for key in keys:
        if key not in calc:
            continue

        value = safe_float(
            calc.get(key)
        )

        if value is not None:
            return value

    return None


def extract_raw_record(
    synthesis: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
) -> Dict[
    str,
    Any,
]:
    if not isinstance(
        synthesis,
        Mapping,
    ):
        return {
            "matched": False,
            "observed_field_count": 0,
        }

    catalyst = catalyst_summary(
        synthesis
    )

    process = synthesis.get(
        "process"
    )

    process = (
        process
        if isinstance(
            process,
            dict,
        )
        else {}
    )

    calc = synthesis.get(
        "calc_block"
    )

    calc = (
        calc
        if isinstance(
            calc,
            dict,
        )
        else {}
    )

    (
        gases,
        gases_reported,
    ) = gas_summary(
        process
    )

    carbon_source = normalize_carbon_source(
        process.get(
            "carbon_source"
        )
    )

    numeric_raw = {
        "raw_catalyst_count":
            catalyst[
                "catalyst_count"
            ],

        "raw_metal_count":
            catalyst[
                "metal_count"
            ],

        "raw_main_metal_fraction":
            catalyst[
                "main_metal_fraction"
            ],

        "raw_second_metal_fraction":
            catalyst[
                "second_metal_fraction"
            ],

        "raw_promoter_feed_rate_log1p":
            log1p_nonnegative(
                catalyst[
                    "promoter_feed_rate"
                ]
            ),

        "raw_temperature_C":
            safe_float(
                process.get(
                    "temperature_C"
                )
            ),

        "raw_pressure_log1p":
            log1p_nonnegative(
                process.get(
                    "pressure_Pa"
                )
            ),

        "raw_growth_time_log1p":
            log1p_nonnegative(
                process.get(
                    "growth_time_min"
                )
            ),

        "raw_np_size_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "np_size_nm",
                        "Average_size_M_nanoparticle",
                    ),
                )
            ),

        "raw_C_M_ratio_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "C_M_ratio",
                    ),
                )
            ),

        "raw_C_NP_ratio_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "C_NP_ratio",
                        "C_M_nanoparticle_ratio",
                    ),
                )
            ),

        "raw_S_M_ratio_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "S_M_ratio",
                    ),
                )
            ),

        "raw_C_concentration_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "C_concentration",
                    ),
                )
            ),

        "raw_M_concentration_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "M_concentration",
                        "metal_concentration",
                    ),
                )
            ),

        "raw_S_concentration_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "S_concentration",
                    ),
                )
            ),

        "raw_metal_atoms_per_np_log1p":
            log1p_nonnegative(
                first_numeric(
                    calc,
                    (
                        "metal_atoms_per_np",
                        "Metal_atoms_per_nanoparticle",
                    ),
                )
            ),
    }

    observed_count = sum(
        value is not None
        and np.isfinite(value)
        for value
        in numeric_raw.values()
    )

    observed_count += int(
        bool(
            catalyst[
                "metal_fractions"
            ]
        )
    )

    observed_count += int(
        bool(
            catalyst[
                "precursors"
            ]
        )
    )

    observed_count += int(
        bool(
            catalyst[
                "catalyst_types"
            ]
        )
    )

    observed_count += int(
        bool(
            catalyst[
                "promoters"
            ]
        )
    )

    observed_count += int(
        gases_reported
    )

    observed_count += int(
        bool(
            carbon_source
        )
    )

    return {
        "matched": True,

        "observed_field_count":
            int(
                observed_count
            ),

        "numeric":
            numeric_raw,

        "metal_fractions":
            catalyst[
                "metal_fractions"
            ],

        "metal_fraction_assumed_equal":
            catalyst[
                "metal_fraction_assumed_equal"
            ],

        "precursors":
            catalyst[
                "precursors"
            ],

        "catalyst_types":
            catalyst[
                "catalyst_types"
            ],

        "promoters":
            catalyst[
                "promoters"
            ],

        "gases":
            gases,

        "carbon_source":
            carbon_source,

        "flags": {
            "miss_raw_catalysts":
                int(
                    not catalyst[
                        "catalysts_reported"
                    ]
                ),

            "miss_raw_metals":
                int(
                    not catalyst[
                        "metals_reported"
                    ]
                ),

            "miss_raw_precursor":
                int(
                    not catalyst[
                        "precursors"
                    ]
                ),

            "miss_raw_catalyst_type":
                int(
                    not catalyst[
                        "catalyst_types"
                    ]
                ),

            "miss_raw_promoter":
                int(
                    not catalyst[
                        "promoters"
                    ]
                ),

            "miss_raw_gas_composition":
                int(
                    not gases_reported
                ),

            "miss_raw_carbon_source":
                int(
                    not carbon_source
                ),
        },
    }


def frequent_categories(
    values: Iterable[str],
) -> List[str]:
    counts = Counter(
        value
        for value in values
        if value
    )

    return sorted(
        category
        for (
            category,
            count,
        ) in counts.items()
        if count
        >= MIN_CATEGORY_FREQUENCY
    )


def build_vocabulary(
    raw_records: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
) -> Dict[
    str,
    Any,
]:
    metals: set[str] = set()
    gases: set[str] = set()

    precursors: List[str] = []
    catalyst_types: List[str] = []
    promoters: List[str] = []
    carbon_sources: List[str] = []

    for record in raw_records:
        metals.update(
            record.get(
                "metal_fractions",
                {},
            ).keys()
        )

        gases.update(
            record.get(
                "gases",
                {},
            ).keys()
        )

        precursors.extend(
            record.get(
                "precursors",
                [],
            )
        )

        catalyst_types.extend(
            record.get(
                "catalyst_types",
                [],
            )
        )

        promoters.extend(
            record.get(
                "promoters",
                [],
            )
        )

        carbon_source = record.get(
            "carbon_source",
            "",
        )

        if carbon_source:
            carbon_sources.append(
                carbon_source
            )

    vocabulary = {
        "metals":
            sorted(
                metals
            ),

        "gases":
            sorted(
                gases
            ),

        "precursors":
            frequent_categories(
                precursors
            ),

        "catalyst_types":
            frequent_categories(
                catalyst_types
            ),

        "promoters":
            frequent_categories(
                promoters
            ),

        "carbon_sources":
            frequent_categories(
                carbon_sources
            ),
    }

    vocabulary[
        "columns"
    ] = {
        "metals":
            make_unique_category_columns(
                "raw_metal",
                vocabulary[
                    "metals"
                ],
            ),

        "gases":
            make_unique_category_columns(
                "raw_gas",
                vocabulary[
                    "gases"
                ],
            ),

        "precursors":
            make_unique_category_columns(
                "raw_precursor",
                vocabulary[
                    "precursors"
                ],
            ),

        "catalyst_types":
            make_unique_category_columns(
                "raw_catalyst_type",
                vocabulary[
                    "catalyst_types"
                ],
            ),

        "promoters":
            make_unique_category_columns(
                "raw_promoter",
                vocabulary[
                    "promoters"
                ],
            ),

        "carbon_sources":
            make_unique_category_columns(
                "raw_carbon_source",
                vocabulary[
                    "carbon_sources"
                ],
            ),
    }

    return vocabulary


def encode_categories(
    output: Dict[
        str,
        float,
    ],
    values: Sequence[str],
    vocabulary: Sequence[str],
    column_map: Mapping[
        str,
        str,
    ],
    other_column: str,
) -> None:
    value_set = set(
        values
    )

    vocabulary_set = set(
        vocabulary
    )

    for category in vocabulary:
        output[
            column_map[
                category
            ]
        ] = float(
            category
            in value_set
        )

    output[
        other_column
    ] = float(
        any(
            value
            and value
            not in vocabulary_set
            for value
            in value_set
        )
    )


def build_raw_feature_table(
    frame: pd.DataFrame,
    raw_syntheses: Sequence[
        Optional[
            Dict[
                str,
                Any,
            ]
        ]
    ],
    alignment_audit: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    Dict[
        str,
        Any,
    ],
    List[str],
    pd.DataFrame,
]:
    raw_records = [
        extract_raw_record(
            item
        )
        for item
        in raw_syntheses
    ]

    vocabulary = build_vocabulary(
        raw_records
    )

    metal_columns = vocabulary[
        "columns"
    ][
        "metals"
    ]

    gas_columns = vocabulary[
        "columns"
    ][
        "gases"
    ]

    precursor_columns = vocabulary[
        "columns"
    ][
        "precursors"
    ]

    type_columns = vocabulary[
        "columns"
    ][
        "catalyst_types"
    ]

    promoter_columns = vocabulary[
        "columns"
    ][
        "promoters"
    ]

    carbon_columns = vocabulary[
        "columns"
    ][
        "carbon_sources"
    ]

    feature_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for (
        index,
        record,
    ) in enumerate(
        raw_records
    ):
        row: Dict[
            str,
            Any,
        ] = {
            "sample_id":
                frame.iloc[
                    index
                ][
                    "sample_id"
                ],

            "raw_matched":
                int(
                    record.get(
                        "matched",
                        False,
                    )
                ),

            "raw_observed_field_count":
                int(
                    record.get(
                        "observed_field_count",
                        0,
                    )
                ),

            "raw_eligible":
                int(
                    bool(
                        record.get(
                            "matched",
                            False,
                        )
                    )
                    and int(
                        record.get(
                            "observed_field_count",
                            0,
                        )
                    )
                    >= 1
                ),
        }

        numeric = record.get(
            "numeric",
            {},
        )

        for column in RAW_NUMERIC_COLUMNS:
            value = numeric.get(
                column,
                np.nan,
            )

            row[
                column
            ] = value

            row[
                f"miss_{column}"
            ] = int(
                value is None
                or not np.isfinite(
                    value
                )
            )

        row[
            "raw_metal_fraction_assumed_equal"
        ] = float(
            record.get(
                "metal_fraction_assumed_equal",
                0.0,
            )
        )

        for column in RAW_GROUP_MISSING_COLUMNS:
            row[
                column
            ] = int(
                record.get(
                    "flags",
                    {},
                ).get(
                    column,
                    1,
                )
            )

        for mode in (
            "A",
            "B",
            "C",
            "D",
            "E",
        ):
            row[
                f"raw_mode_{mode}"
            ] = float(
                text(
                    frame.iloc[
                        index
                    ][
                        "growth_mode"
                    ]
                ).upper()
                == mode
            )

        metal_fractions = record.get(
            "metal_fractions",
            {},
        )

        metals_reported = bool(
            metal_fractions
        )

        for metal in vocabulary[
            "metals"
        ]:
            row[
                metal_columns[
                    metal
                ]
            ] = (
                float(
                    metal_fractions.get(
                        metal,
                        0.0,
                    )
                )
                if metals_reported
                else np.nan
            )

        gases = record.get(
            "gases",
            {},
        )

        gases_reported = bool(
            gases
        )

        for gas in vocabulary[
            "gases"
        ]:
            row[
                gas_columns[
                    gas
                ]
            ] = (
                float(
                    gases.get(
                        gas,
                        0.0,
                    )
                )
                if gases_reported
                else np.nan
            )

        encode_categories(
            row,
            record.get(
                "precursors",
                [],
            ),
            vocabulary[
                "precursors"
            ],
            precursor_columns,
            "raw_precursor_other",
        )

        encode_categories(
            row,
            record.get(
                "catalyst_types",
                [],
            ),
            vocabulary[
                "catalyst_types"
            ],
            type_columns,
            "raw_catalyst_type_other",
        )

        encode_categories(
            row,
            record.get(
                "promoters",
                [],
            ),
            vocabulary[
                "promoters"
            ],
            promoter_columns,
            "raw_promoter_other",
        )

        carbon_source = record.get(
            "carbon_source",
            "",
        )

        encode_categories(
            row,
            (
                [
                    carbon_source
                ]
                if carbon_source
                else []
            ),
            vocabulary[
                "carbon_sources"
            ],
            carbon_columns,
            "raw_carbon_source_other",
        )

        feature_rows.append(
            row
        )

    raw_frame = pd.DataFrame(
        feature_rows
    )

    candidate_columns = [
        *RAW_NUMERIC_COLUMNS,

        *[
            f"miss_{column}"
            for column
            in RAW_NUMERIC_COLUMNS
        ],

        *RAW_GROUP_MISSING_COLUMNS,

        *RAW_EXTRA_COLUMNS,

        *[
            f"raw_mode_{mode}"
            for mode
            in (
                "A",
                "B",
                "C",
                "D",
                "E",
            )
        ],

        *metal_columns.values(),

        *gas_columns.values(),

        *precursor_columns.values(),

        "raw_precursor_other",

        *type_columns.values(),

        "raw_catalyst_type_other",

        *promoter_columns.values(),

        "raw_promoter_other",

        *carbon_columns.values(),

        "raw_carbon_source_other",
    ]

    candidate_columns = list(
        dict.fromkeys(
            candidate_columns
        )
    )

    coverage_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    usable_columns: List[
        str
    ] = []

    for column in candidate_columns:
        series = pd.to_numeric(
            raw_frame[
                column
            ],
            errors="coerce",
        )

        finite = series[
            np.isfinite(
                series
            )
        ]

        unique_count = int(
            finite.nunique(
                dropna=True
            )
        )

        keep = (
            len(finite) > 0
            and unique_count > 1
        )

        if keep:
            usable_columns.append(
                column
            )

        coverage_rows.append(
            {
                "feature":
                    column,

                "nonmissing_count":
                    int(
                        series.notna().sum()
                    ),

                "missing_fraction":
                    float(
                        series.isna().mean()
                    ),

                "unique_finite_values":
                    unique_count,

                "minimum":
                    (
                        float(
                            finite.min()
                        )
                        if len(finite)
                        else np.nan
                    ),

                "median":
                    (
                        float(
                            finite.median()
                        )
                        if len(finite)
                        else np.nan
                    ),

                "maximum":
                    (
                        float(
                            finite.max()
                        )
                        if len(finite)
                        else np.nan
                    ),

                "used_in_model":
                    int(
                        keep
                    ),

                "drop_reason":
                    (
                        ""
                        if keep
                        else (
                            "all_missing"
                            if len(
                                finite
                            ) == 0
                            else "constant"
                        )
                    ),
            }
        )

    if not usable_columns:
        raise RuntimeError(
            "Raw baseline 没有任何可用特征。"
        )

    return (
        raw_frame,
        vocabulary,
        usable_columns,
        pd.DataFrame(
            coverage_rows
        ),
    )


def prepare_composite_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in (
        COL_GQ,
        COL_IGID,
        COL_ORI,
        COL_PURITY,
        COL_BUNDLE,
    ):
        if column not in output.columns:
            output[
                column
            ] = np.nan

    output[
        RAW_GQ
    ] = pd.to_numeric(
        output[
            COL_GQ
        ],
        errors="coerce",
    )

    output[
        RAW_IGID
    ] = pd.to_numeric(
        output[
            COL_IGID
        ],
        errors="coerce",
    )

    output[
        RAW_ORI
    ] = pd.to_numeric(
        output[
            COL_ORI
        ],
        errors="coerce",
    )

    output[
        RAW_PURITY
    ] = pd.to_numeric(
        output[
            COL_PURITY
        ],
        errors="coerce",
    )

    bundle = pd.to_numeric(
        output[
            COL_BUNDLE
        ],
        errors="coerce",
    ).to_numpy(
        float
    )

    output[
        RAW_BUNDLE_LOG
    ] = np.where(
        np.isfinite(
            bundle
        )
        & (
            bundle >= 0
        ),
        np.log1p(
            bundle
        ),
        np.nan,
    )

    availability = pd.DataFrame(
        {
            "gq":
                output[
                    RAW_GQ
                ].notna(),

            "igid":
                output[
                    RAW_IGID
                ].notna(),

            "ori":
                output[
                    RAW_ORI
                ].notna(),

            "purity":
                output[
                    RAW_PURITY
                ].notna(),

            "bundle":
                output[
                    RAW_BUNDLE_LOG
                ].notna(),
        }
    )

    output[
        RAW_COMPONENT_COUNT
    ] = (
        availability
        .sum(
            axis=1
        )
        .astype(
            int
        )
    )

    return output


def minmax_parameters(
    series: pd.Series,
) -> Dict[str, Any]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        float
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return {
            "minimum": None,
            "maximum": None,
            "usable": False,
        }

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    return {
        "minimum":
            minimum,

        "maximum":
            maximum,

        "usable":
            not math.isclose(
                minimum,
                maximum,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
    }


def apply_minmax(
    series: pd.Series,
    parameters: Mapping[
        str,
        Any,
    ],
) -> pd.Series:
    if not parameters[
        "usable"
    ]:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    normalized = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        - parameters[
            "minimum"
        ]
    ) / (
        parameters[
            "maximum"
        ]
        - parameters[
            "minimum"
        ]
    )

    return normalized.clip(
        0.0,
        1.0,
    )


def composite_for_fold(
    frame: pd.DataFrame,
    original_outer_train_ids: Sequence[str],
) -> pd.Series:
    train_id_set = set(
        map(
            str,
            original_outer_train_ids,
        )
    )

    train = frame[
        frame[
            "sample_id"
        ]
        .astype(str)
        .isin(
            train_id_set
        )
        & (
            frame[
                RAW_COMPONENT_COUNT
            ]
            >= MIN_COMPOSITE_COMPONENTS
        )
    ]

    igid_parameters = (
        minmax_parameters(
            train[
                RAW_IGID
            ]
        )
    )

    bundle_parameters = (
        minmax_parameters(
            train[
                RAW_BUNDLE_LOG
            ]
        )
    )

    components = pd.DataFrame(
        index=frame.index
    )

    components[
        "gq"
    ] = (
        pd.to_numeric(
            frame[
                RAW_GQ
            ],
            errors="coerce",
        )
        / 10.0
    ).clip(
        0.0,
        1.0,
    )

    components[
        "igid"
    ] = apply_minmax(
        frame[
            RAW_IGID
        ],
        igid_parameters,
    )

    components[
        "ori"
    ] = pd.to_numeric(
        frame[
            RAW_ORI
        ],
        errors="coerce",
    ).clip(
        0.0,
        1.0,
    )

    components[
        "purity"
    ] = pd.to_numeric(
        frame[
            RAW_PURITY
        ],
        errors="coerce",
    ).clip(
        0.0,
        1.0,
    )

    components[
        "bundle"
    ] = apply_minmax(
        frame[
            RAW_BUNDLE_LOG
        ],
        bundle_parameters,
    )

    weights = pd.Series(
        COMPOSITE_WEIGHTS,
        dtype=float,
    )

    available = (
        components
        .notna()
    )

    component_count = (
        available
        .sum(
            axis=1
        )
    )

    numerator = (
        components
        .mul(
            weights,
            axis=1,
        )
        .sum(
            axis=1,
            skipna=True,
        )
    )

    denominator = (
        available
        .mul(
            weights,
            axis=1,
        )
        .sum(
            axis=1
        )
        .replace(
            0.0,
            np.nan,
        )
    )

    score = (
        numerator
        / denominator
    )

    return score.where(
        component_count
        >= MIN_COMPOSITE_COMPONENTS,
        np.nan,
    )


def load_manifest(
    path: Path,
    target_name: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "sample_id",
        "split_type",
        "fold",
        "role",
    ]

    missing = [
        column
        for column
        in required
        if column
        not in manifest.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path.name} 缺少："
            f"{missing}"
        )

    if "target" in manifest.columns:
        aliases = {
            "growth_quality": {
                "y_growth_quality_score",
            },

            "igid": {
                "y_ig_id_ratio",
                "y_IG_ID",
                "y_ig_id_ratio_unified",
            },

            "orientation": {
                "y_orientation_score",
            },

            "composite": {
                "y_composite_final_score",
            },
        }[
            target_name
        ]

        manifest = manifest[
            manifest[
                "target"
            ]
            .astype(str)
            .isin(
                aliases
            )
        ].copy()

    manifest[
        "sample_id"
    ] = (
        manifest[
            "sample_id"
        ]
        .fillna("")
        .astype(str)
    )

    manifest[
        "split_type"
    ] = (
        manifest[
            "split_type"
        ]
        .astype(str)
    )

    manifest[
        "fold"
    ] = pd.to_numeric(
        manifest[
            "fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    manifest[
        "role"
    ] = (
        manifest[
            "role"
        ]
        .astype(str)
        .str.lower()
    )

    manifest = manifest[
        manifest[
            "split_type"
        ].isin(
            SPLITS
        )
        & manifest[
            "role"
        ].isin(
            (
                "train",
                "test",
            )
        )
    ].copy()

    if manifest.empty:
        raise RuntimeError(
            f"{path.name} 中没有目标 "
            f"{target_name} 的有效划分。"
        )

    duplicates = (
        manifest.duplicated(
            subset=[
                "sample_id",
                "split_type",
                "fold",
                "role",
            ]
        )
    )

    if duplicates.any():
        raise RuntimeError(
            f"{path.name} 中存在重复划分记录。"
        )

    return manifest.reset_index(
        drop=True
    )


def split_tasks(
    manifest: pd.DataFrame,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    rank = {
        name:
            index
        for (
            index,
            name,
        ) in enumerate(
            SPLITS
        )
    }

    grouped = manifest.groupby(
        [
            "split_type",
            "fold",
        ],
        sort=False,
    )

    keys = sorted(
        grouped.groups,
        key=lambda item: (
            rank[
                item[0]
            ],
            item[1],
        ),
    )

    tasks: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for (
        split_type,
        fold,
    ) in keys:
        group = grouped.get_group(
            (
                split_type,
                fold,
            )
        )

        train_ids = (
            group.loc[
                group[
                    "role"
                ] == "train",
                "sample_id",
            ]
            .tolist()
        )

        test_ids = (
            group.loc[
                group[
                    "role"
                ] == "test",
                "sample_id",
            ]
            .tolist()
        )

        if (
            set(
                train_ids
            )
            & set(
                test_ids
            )
        ):
            raise RuntimeError(
                f"{split_type}/fold={fold} "
                "的训练集与测试集存在重叠。"
            )

        cutoff = np.nan

        if (
            "temporal_cutoff_year"
            in group.columns
        ):
            values = pd.to_numeric(
                group[
                    "temporal_cutoff_year"
                ],
                errors="coerce",
            ).dropna()

            if not values.empty:
                cutoff = float(
                    values.iloc[
                        0
                    ]
                )

        tasks.append(
            {
                "split_type":
                    split_type,

                "fold":
                    int(
                        fold
                    ),

                "train_ids":
                    train_ids,

                "test_ids":
                    test_ids,

                "temporal_cutoff_year":
                    cutoff,
            }
        )

    return tasks


def make_inner_split(
    outer_train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    str,
]:
    outer_train_indices = np.asarray(
        outer_train_indices,
        dtype=int,
    )

    groups = (
        groups_all
        .iloc[
            outer_train_indices
        ]
        .fillna("")
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    if (
        groups.nunique()
        >= 2
    ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=INNER_VALID_SIZE,
            random_state=SEED,
        )

        (
            fit_relative,
            valid_relative,
        ) = next(
            splitter.split(
                np.zeros(
                    len(
                        outer_train_indices
                    )
                ),
                groups=groups.to_numpy(),
            )
        )

        return (
            outer_train_indices[
                fit_relative
            ],

            outer_train_indices[
                valid_relative
            ],

            "group_shuffle",
        )

    relative = np.arange(
        len(
            outer_train_indices
        )
    )

    (
        fit_relative,
        valid_relative,
    ) = train_test_split(
        relative,
        test_size=INNER_VALID_SIZE,
        random_state=SEED,
    )

    return (
        outer_train_indices[
            fit_relative
        ],

        outer_train_indices[
            valid_relative
        ],

        "random_fallback",
    )


def fit_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple[
    np.ndarray,
    int,
]:
    train_set = lgb.Dataset(
        X.iloc[
            fit_indices
        ],
        label=y[
            fit_indices
        ],
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X.iloc[
            valid_indices
        ],
        label=y[
            valid_indices
        ],
        reference=train_set,
        free_raw_data=False,
    )

    model = lgb.train(
        params=dict(
            LGB_PARAMS
        ),
        train_set=train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[
            train_set,
            valid_set,
        ],
        valid_names=[
            "inner_fit",
            "inner_valid",
        ],
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                verbose=False,
            ),
            lgb.log_evaluation(
                period=0
            ),
        ],
    )

    best_iteration = int(
        model.best_iteration
        if (
            model.best_iteration
            and model.best_iteration > 0
        )
        else MAX_BOOST_ROUNDS
    )

    prediction = model.predict(
        X.iloc[
            test_indices
        ],
        num_iteration=best_iteration,
    )

    return (
        np.asarray(
            prediction,
            dtype=float,
        ),
        best_iteration,
    )


def run_models(
    frame: pd.DataFrame,
    feature_sets: Mapping[
        str,
        Sequence[str],
    ],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    prediction_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    audit_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    id_to_index = {
        str(sample_id):
            index
        for (
            index,
            sample_id,
        ) in enumerate(
            frame[
                "sample_id"
            ].astype(str)
        )
    }

    X_cache = {
        name:
            frame[
                list(
                    columns
                )
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )
        for (
            name,
            columns,
        ) in feature_sets.items()
    }

    for (
        target_name,
        target_column,
    ) in TARGETS.items():
        manifest_path = (
            COMPOSITE_MANIFEST_CSV
            if target_name
            == "composite"
            else SINGLE_MANIFEST_CSV
        )

        manifest = load_manifest(
            manifest_path,
            target_name,
        )

        for task in split_tasks(
            manifest
        ):
            if (
                target_name
                == "composite"
            ):
                y_series = composite_for_fold(
                    frame,
                    task[
                        "train_ids"
                    ],
                )

            else:
                if (
                    target_column
                    not in frame.columns
                ):
                    raise RuntimeError(
                        "数据集中缺少目标："
                        f"{target_column}"
                    )

                y_series = pd.to_numeric(
                    frame[
                        target_column
                    ],
                    errors="coerce",
                )

            raw_eligible = (
                frame[
                    "raw_eligible"
                ]
                .fillna(0)
                .astype(int)
                == 1
            )

            eligible = (
                y_series.notna()
                & raw_eligible
            )

            eligible_ids = set(
                frame.loc[
                    eligible,
                    "sample_id",
                ].astype(str)
            )

            train_ids = [
                str(item)
                for item
                in task[
                    "train_ids"
                ]
                if str(item)
                in eligible_ids
            ]

            test_ids = [
                str(item)
                for item
                in task[
                    "test_ids"
                ]
                if str(item)
                in eligible_ids
            ]

            train_indices = np.asarray(
                [
                    id_to_index[
                        item
                    ]
                    for item
                    in train_ids
                ],
                dtype=int,
            )

            test_indices = np.asarray(
                [
                    id_to_index[
                        item
                    ]
                    for item
                    in test_ids
                ],
                dtype=int,
            )

            train_papers = (
                int(
                    frame.iloc[
                        train_indices
                    ][
                        "paper_id"
                    ]
                    .nunique()
                )
                if len(
                    train_indices
                )
                else 0
            )

            test_papers = (
                int(
                    frame.iloc[
                        test_indices
                    ][
                        "paper_id"
                    ]
                    .nunique()
                )
                if len(
                    test_indices
                )
                else 0
            )

            usable = (
                len(
                    train_indices
                )
                >= MIN_TRAIN_SAMPLES

                and len(
                    test_indices
                )
                >= MIN_TEST_SAMPLES

                and train_papers
                >= MIN_TRAIN_PAPERS

                and test_papers
                >= MIN_TEST_PAPERS
            )

            audit_rows.append(
                {
                    "target":
                        target_name,

                    "split_type":
                        task[
                            "split_type"
                        ],

                    "fold":
                        task[
                            "fold"
                        ],

                    "manifest_train_rows":
                        len(
                            task[
                                "train_ids"
                            ]
                        ),

                    "manifest_test_rows":
                        len(
                            task[
                                "test_ids"
                            ]
                        ),

                    "common_train_rows":
                        len(
                            train_indices
                        ),

                    "common_test_rows":
                        len(
                            test_indices
                        ),

                    "common_train_papers":
                        train_papers,

                    "common_test_papers":
                        test_papers,

                    "train_retention":
                        (
                            len(
                                train_indices
                            )
                            / len(
                                task[
                                    "train_ids"
                                ]
                            )
                            if task[
                                "train_ids"
                            ]
                            else np.nan
                        ),

                    "test_retention":
                        (
                            len(
                                test_indices
                            )
                            / len(
                                task[
                                    "test_ids"
                                ]
                            )
                            if task[
                                "test_ids"
                            ]
                            else np.nan
                        ),

                    "status":
                        (
                            "ok"
                            if usable
                            else (
                                "skipped_too_few_"
                                "common_samples_or_papers"
                            )
                        ),
                }
            )

            if not usable:
                continue

            y = pd.to_numeric(
                y_series,
                errors="coerce",
            ).to_numpy(
                float
            )

            truth = y[
                test_indices
            ]

            (
                fit_indices,
                valid_indices,
                inner_method,
            ) = make_inner_split(
                train_indices,
                frame[
                    "paper_id"
                ],
            )

            for representation in REPRESENTATIONS:
                columns = feature_sets[
                    representation
                ]

                (
                    prediction,
                    best_iteration,
                ) = fit_predict(
                    X_cache[
                        representation
                    ],
                    y,
                    fit_indices,
                    valid_indices,
                    test_indices,
                )

                result = evaluate_predictions(
                    truth,
                    prediction,
                )

                metric_rows.append(
                    {
                        "target":
                            target_name,

                        "target_label":
                            TARGET_LABELS[
                                target_name
                            ],

                        "representation":
                            representation,

                        "representation_label":
                            REPRESENTATION_LABELS[
                                representation
                            ],

                        "split_type":
                            task[
                                "split_type"
                            ],

                        "split_label":
                            SPLIT_LABELS[
                                task[
                                    "split_type"
                                ]
                            ],

                        "fold":
                            task[
                                "fold"
                            ],

                        "n_train":
                            len(
                                train_indices
                            ),

                        "n_inner_fit":
                            len(
                                fit_indices
                            ),

                        "n_inner_valid":
                            len(
                                valid_indices
                            ),

                        "n_test":
                            len(
                                test_indices
                            ),

                        "n_train_papers":
                            train_papers,

                        "n_test_papers":
                            test_papers,

                        "feature_count":
                            len(
                                columns
                            ),

                        "best_iteration":
                            best_iteration,

                        "inner_split_method":
                            inner_method,

                        "temporal_cutoff_year":
                            task[
                                "temporal_cutoff_year"
                            ],

                        **result,
                    }
                )

                for (
                    position,
                    index,
                ) in enumerate(
                    test_indices
                ):
                    item = frame.iloc[
                        index
                    ]

                    true_value = float(
                        truth[
                            position
                        ]
                    )

                    predicted_value = float(
                        prediction[
                            position
                        ]
                    )

                    prediction_rows.append(
                        {
                            "sample_id":
                                item[
                                    "sample_id"
                                ],

                            "paper_id":
                                item[
                                    "paper_id"
                                ],

                            "paper_year_int":
                                item[
                                    "paper_year_int"
                                ],

                            "growth_mode":
                                item[
                                    "growth_mode"
                                ],

                            "source_file":
                                item[
                                    "source_file"
                                ],

                            "synthesis_id":
                                item[
                                    "synthesis_id"
                                ],

                            "target":
                                target_name,

                            "representation":
                                representation,

                            "split_type":
                                task[
                                    "split_type"
                                ],

                            "fold":
                                task[
                                    "fold"
                                ],

                            "y_true":
                                true_value,

                            "y_pred":
                                predicted_value,

                            "residual":
                                (
                                    predicted_value
                                    - true_value
                                ),

                            "absolute_error":
                                abs(
                                    predicted_value
                                    - true_value
                                ),
                        }
                    )

    return (
        pd.DataFrame(
            metric_rows
        ),
        pd.DataFrame(
            prediction_rows
        ),
        pd.DataFrame(
            audit_rows
        ),
    )


def pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for (
        target,
        representation,
        split_type,
    ), group in predictions.groupby(
        [
            "target",
            "representation",
            "split_type",
        ],
        sort=False,
    ):
        rows.append(
            {
                "target":
                    target,

                "target_label":
                    TARGET_LABELS[
                        target
                    ],

                "representation":
                    representation,

                "representation_label":
                    REPRESENTATION_LABELS[
                        representation
                    ],

                "split_type":
                    split_type,

                "split_label":
                    SPLIT_LABELS[
                        split_type
                    ],

                "papers":
                    int(
                        group[
                            "paper_id"
                        ].nunique()
                    ),

                "folds":
                    int(
                        group[
                            "fold"
                        ].nunique()
                    ),

                **evaluate_predictions(
                    group[
                        "y_true"
                    ],
                    group[
                        "y_pred"
                    ],
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def paired_prediction_table(
    group: pd.DataFrame,
) -> pd.DataFrame:
    metadata = (
        group[
            [
                "sample_id",
                "paper_id",
                "fold",
                "y_true",
            ]
        ]
        .drop_duplicates()
    )

    wide = (
        group
        .pivot_table(
            index=[
                "sample_id",
                "paper_id",
                "fold",
            ],
            columns="representation",
            values="y_pred",
            aggfunc="first",
        )
        .reset_index()
    )

    truth = (
        metadata
        .drop_duplicates(
            subset=[
                "sample_id",
                "paper_id",
                "fold",
            ]
        )
    )

    wide = wide.merge(
        truth,
        on=[
            "sample_id",
            "paper_id",
            "fold",
        ],
        how="left",
        validate="one_to_one",
    )

    required = list(
        REPRESENTATIONS
    )

    return (
        wide
        .dropna(
            subset=(
                required
                + [
                    "y_true"
                ]
            )
        )
        .reset_index(
            drop=True
        )
    )


def bootstrap_analysis(
    predictions: pd.DataFrame,
    repeats: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    difference_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for (
        group_index,
        (
            (
                target,
                split_type,
            ),
            group,
        ),
    ) in enumerate(
        predictions.groupby(
            [
                "target",
                "split_type",
            ],
            sort=False,
        ),
        start=1,
    ):
        paired = paired_prediction_table(
            group
        )

        paper_rows = {
            paper_id:
                paper_group
                .index
                .to_numpy(
                    int
                )
            for (
                paper_id,
                paper_group,
            ) in paired.groupby(
                "paper_id",
                sort=False,
            )
        }

        papers = list(
            paper_rows
        )

        if (
            len(
                paired
            ) < 5
            or len(
                papers
            ) < 2
        ):
            continue

        point_estimates = {
            representation:
                evaluate_predictions(
                    paired[
                        "y_true"
                    ],
                    paired[
                        representation
                    ],
                )
            for representation
            in REPRESENTATIONS
        }

        metric_distributions = {
            representation: {
                metric: []
                for metric
                in (
                    "mae",
                    "rmse",
                    "r2",
                    "spearman",
                )
            }
            for representation
            in REPRESENTATIONS
        }

        difference_distributions = {
            (
                left,
                right,
                metric,
            ): []
            for (
                left,
                right,
                _,
            ) in PAIRWISE_COMPARISONS
            for metric
            in (
                "spearman",
                "mae",
            )
        }

        rng = np.random.default_rng(
            SEED
            + group_index
            * 1009
        )

        for _ in range(
            repeats
        ):
            sampled_papers = rng.choice(
                papers,
                size=len(
                    papers
                ),
                replace=True,
            )

            sampled_indices = np.concatenate(
                [
                    paper_rows[
                        paper
                    ]
                    for paper
                    in sampled_papers
                ]
            )

            sample = paired.iloc[
                sampled_indices
            ]

            sample_metrics = {
                representation:
                    evaluate_predictions(
                        sample[
                            "y_true"
                        ],
                        sample[
                            representation
                        ],
                    )
                for representation
                in REPRESENTATIONS
            }

            for representation in REPRESENTATIONS:
                for metric in (
                    "mae",
                    "rmse",
                    "r2",
                    "spearman",
                ):
                    value = (
                        sample_metrics[
                            representation
                        ][
                            metric
                        ]
                    )

                    if np.isfinite(
                        value
                    ):
                        metric_distributions[
                            representation
                        ][
                            metric
                        ].append(
                            float(
                                value
                            )
                        )

            for (
                left,
                right,
                _,
            ) in PAIRWISE_COMPARISONS:
                for metric in (
                    "spearman",
                    "mae",
                ):
                    left_value = (
                        sample_metrics[
                            left
                        ][
                            metric
                        ]
                    )

                    right_value = (
                        sample_metrics[
                            right
                        ][
                            metric
                        ]
                    )

                    difference = (
                        left_value
                        - right_value
                    )

                    if np.isfinite(
                        difference
                    ):
                        difference_distributions[
                            (
                                left,
                                right,
                                metric,
                            )
                        ].append(
                            float(
                                difference
                            )
                        )

        for representation in REPRESENTATIONS:
            for metric in (
                "mae",
                "rmse",
                "r2",
                "spearman",
            ):
                values = np.asarray(
                    metric_distributions[
                        representation
                    ][
                        metric
                    ],
                    dtype=float,
                )

                metric_rows.append(
                    {
                        "target":
                            target,

                        "target_label":
                            TARGET_LABELS[
                                target
                            ],

                        "representation":
                            representation,

                        "representation_label":
                            REPRESENTATION_LABELS[
                                representation
                            ],

                        "split_type":
                            split_type,

                        "split_label":
                            SPLIT_LABELS[
                                split_type
                            ],

                        "metric":
                            metric,

                        "estimate":
                            point_estimates[
                                representation
                            ][
                                metric
                            ],

                        "ci_lower":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.025,
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),

                        "ci_upper":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.975,
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),

                        "valid_repeats":
                            int(
                                len(
                                    values
                                )
                            ),

                        "paired_samples":
                            int(
                                len(
                                    paired
                                )
                            ),

                        "paired_papers":
                            int(
                                len(
                                    papers
                                )
                            ),
                    }
                )

        for (
            left,
            right,
            label,
        ) in PAIRWISE_COMPARISONS:
            for metric in (
                "spearman",
                "mae",
            ):
                values = np.asarray(
                    difference_distributions[
                        (
                            left,
                            right,
                            metric,
                        )
                    ],
                    dtype=float,
                )

                estimate = (
                    point_estimates[
                        left
                    ][
                        metric
                    ]
                    - point_estimates[
                        right
                    ][
                        metric
                    ]
                )

                difference_rows.append(
                    {
                        "target":
                            target,

                        "target_label":
                            TARGET_LABELS[
                                target
                            ],

                        "split_type":
                            split_type,

                        "split_label":
                            SPLIT_LABELS[
                                split_type
                            ],

                        "comparison":
                            label,

                        "left_representation":
                            left,

                        "right_representation":
                            right,

                        "metric":
                            metric,

                        "difference_definition":
                            f"{left} - {right}",

                        "estimate":
                            estimate,

                        "ci_lower":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.025,
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),

                        "ci_upper":
                            (
                                float(
                                    np.quantile(
                                        values,
                                        0.975,
                                    )
                                )
                                if len(
                                    values
                                )
                                else np.nan
                            ),

                        "valid_repeats":
                            int(
                                len(
                                    values
                                )
                            ),

                        "paired_samples":
                            int(
                                len(
                                    paired
                                )
                            ),

                        "paired_papers":
                            int(
                                len(
                                    papers
                                )
                            ),

                        "interpretation":
                            (
                                "positive_favors_left"
                                if metric
                                == "spearman"
                                else (
                                    "negative_favors_left"
                                )
                            ),
                    }
                )

    return (
        pd.DataFrame(
            metric_rows
        ),
        pd.DataFrame(
            difference_rows
        ),
    )


def plot_result(
    bootstrap: pd.DataFrame,
    dpi: int,
) -> None:
    source = bootstrap[
        (
            bootstrap[
                "metric"
            ]
            == "spearman"
        )
        & bootstrap[
            "split_type"
        ].isin(
            SPLITS
        )
    ].copy()

    write_csv(
        source,
        SOURCE_DATA_DIR
        / "fig_representation_baseline_source_data.csv",
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.5,
            4.8,
        ),
        sharey=True,
    )

    target_names = list(
        TARGETS
    )

    target_positions = np.arange(
        len(
            target_names
        ),
        dtype=float,
    )

    offsets = {
        "raw_plus_missing":
            -0.22,

        "physical_60D":
            0.0,

        "physical_plus_missing_120D":
            0.22,
    }

    markers = {
        "raw_plus_missing":
            "o",

        "physical_60D":
            "s",

        "physical_plus_missing_120D":
            "^",
    }

    for (
        axis,
        split_type,
    ) in zip(
        axes,
        SPLITS,
    ):
        for representation in REPRESENTATIONS:
            part = source[
                (
                    source[
                        "split_type"
                    ]
                    == split_type
                )
                & (
                    source[
                        "representation"
                    ]
                    == representation
                )
            ].set_index(
                "target"
            )

            x_values: List[
                float
            ] = []

            y_values: List[
                float
            ] = []

            lower_errors: List[
                float
            ] = []

            upper_errors: List[
                float
            ] = []

            for (
                index,
                target,
            ) in enumerate(
                target_names
            ):
                if (
                    target
                    not in part.index
                ):
                    continue

                row = part.loc[
                    target
                ]

                estimate = float(
                    row[
                        "estimate"
                    ]
                )

                lower = float(
                    row[
                        "ci_lower"
                    ]
                )

                upper = float(
                    row[
                        "ci_upper"
                    ]
                )

                if not np.isfinite(
                    estimate
                ):
                    continue

                x_values.append(
                    target_positions[
                        index
                    ]
                    + offsets[
                        representation
                    ]
                )

                y_values.append(
                    estimate
                )

                lower_errors.append(
                    max(
                        0.0,
                        estimate
                        - lower,
                    )
                )

                upper_errors.append(
                    max(
                        0.0,
                        upper
                        - estimate,
                    )
                )

            if x_values:
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=np.vstack(
                        [
                            lower_errors,
                            upper_errors,
                        ]
                    ),
                    marker=markers[
                        representation
                    ],
                    linestyle="none",
                    capsize=3,
                    markersize=6,
                    label=(
                        REPRESENTATION_LABELS[
                            representation
                        ]
                    ),
                )

        axis.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
        )

        axis.set_title(
            SPLIT_LABELS[
                split_type
            ]
        )

        axis.set_xticks(
            target_positions
        )

        axis.set_xticklabels(
            [
                TARGET_LABELS[
                    target
                ]
                for target
                in target_names
            ],
            rotation=15,
            ha="right",
        )

        axis.set_xlabel(
            "Prediction target"
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

    axes[
        0
    ].set_ylabel(
        "Spearman rank correlation"
    )

    axes[
        1
    ].legend(
        frameon=False,
        loc="best",
    )

    figure.suptitle(
        "Raw experimental variables versus "
        "physics-derived CNT representations"
    )

    figure.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig_representation_baseline.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig_representation_baseline.pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
    )

    parser.add_argument(
        "--keep_output",
        action="store_true",
        help=(
            "保留现有输出目录；"
            "默认运行前删除旧输出。"
        ),
    )

    return parser.parse_args()


def check_inputs() -> None:
    required = (
        DATASET_CSV,
        SCHEMA_JSON,
        SINGLE_MANIFEST_CSV,
        COMPOSITE_MANIFEST_CSV,
    )

    missing = [
        str(path)
        for path
        in required
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要输入文件：\n"
            + "\n".join(
                missing
            )
        )

    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(
            "RAW_ROOT 不存在："
            f"{RAW_ROOT}"
        )


def main() -> None:
    args = parse_args()

    if args.bootstrap < 1:
        raise ValueError(
            "--bootstrap 必须大于 0。"
        )

    check_inputs()

    if (
        OUT_DIR.exists()
        and not args.keep_output
    ):
        shutil.rmtree(
            OUT_DIR
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 96
    )

    print(
        "[REPRESENTATION BASELINE START]"
    )

    print(
        f"Dataset             : "
        f"{DATASET_CSV}"
    )

    print(
        f"Raw root            : "
        f"{RAW_ROOT}"
    )

    print(
        "Composite threshold : "
        f"{MIN_COMPOSITE_COMPONENTS}"
    )

    print(
        f"Splits              : "
        f"{SPLITS}"
    )

    print(
        "Bootstrap repeats   : "
        f"{args.bootstrap}"
    )

    print(
        "=" * 96
    )

    (
        physical_60d,
        physical_120d,
    ) = load_schema()

    frame = load_dataset()

    (
        raw_syntheses,
        alignment_audit,
    ) = attach_raw_syntheses(
        frame
    )

    (
        raw_feature_frame,
        vocabulary,
        raw_columns,
        raw_coverage,
    ) = build_raw_feature_table(
        frame,
        raw_syntheses,
        alignment_audit,
    )

    frame = pd.concat(
        [
            frame.reset_index(
                drop=True
            ),

            raw_feature_frame.drop(
                columns=[
                    "sample_id"
                ]
            ).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    frame = prepare_composite_columns(
        frame
    )

    alignment_audit = (
        alignment_audit.copy()
    )

    alignment_audit[
        "raw_observed_field_count"
    ] = frame[
        "raw_observed_field_count"
    ].to_numpy(
        int
    )

    alignment_audit[
        "raw_eligible"
    ] = frame[
        "raw_eligible"
    ].to_numpy(
        int
    )

    matched_rows = int(
        (
            alignment_audit[
                "synthesis_found"
            ]
            == 1
        ).sum()
    )

    eligible_rows = int(
        frame[
            "raw_eligible"
        ].sum()
    )

    write_csv(
        alignment_audit,
        OUT_DIR
        / "raw_source_alignment_audit.csv",
    )

    write_csv(
        raw_feature_frame,
        OUT_DIR
        / "raw_feature_table.csv",
    )

    write_csv(
        raw_coverage,
        OUT_DIR
        / "raw_feature_coverage.csv",
    )

    write_json(
        vocabulary,
        OUT_DIR
        / "raw_category_vocabulary.json",
    )

    if eligible_rows == 0:
        examples = alignment_audit[
            alignment_audit[
                "status"
            ]
            != "ok"
        ][
            [
                "sample_id",
                "status",
                "expected_raw_json_path",
                "error",
            ]
        ].head(
            10
        )

        raise RuntimeError(
            "没有任何样本成功获得 Raw 特征。"
            "请查看 "
            "outputs/external_validation/representation_baseline/"
            "raw_source_alignment_audit.csv。\n"
            "前几条失败记录：\n"
            f"{examples.to_string(index=False)}"
        )

    feature_sets = {
        "raw_plus_missing":
            raw_columns,

        "physical_60D":
            physical_60d,

        "physical_plus_missing_120D":
            physical_120d,
    }

    missing_features = [
        column
        for columns
        in feature_sets.values()
        for column
        in columns
        if column
        not in frame.columns
    ]

    if missing_features:
        raise RuntimeError(
            "合并数据缺少特征列，示例："
            f"{missing_features[:20]}"
        )

    feature_set_table = pd.DataFrame(
        [
            {
                "representation":
                    name,

                "representation_label":
                    REPRESENTATION_LABELS[
                        name
                    ],

                "feature_count":
                    len(
                        columns
                    ),

                "features":
                    " | ".join(
                        columns
                    ),
            }
            for (
                name,
                columns,
            ) in feature_sets.items()
        ]
    )

    write_csv(
        feature_set_table,
        OUT_DIR
        / "representation_feature_sets.csv",
    )

    (
        metrics_fold,
        predictions,
        sample_audit,
    ) = run_models(
        frame,
        feature_sets,
    )

    write_csv(
        sample_audit,
        OUT_DIR
        / "representation_sample_audit.csv",
    )

    if predictions.empty:
        raise RuntimeError(
            "没有产生有效预测。"
            "请查看 representation_sample_audit.csv。"
        )

    metrics_pooled = pooled_metrics(
        predictions
    )

    (
        bootstrap,
        differences,
    ) = bootstrap_analysis(
        predictions,
        args.bootstrap,
    )

    strict_summary = (
        metrics_pooled
        .sort_values(
            [
                "target",
                "split_type",
                "representation",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    write_csv(
        metrics_fold,
        OUT_DIR
        / "representation_metrics_fold.csv",
    )

    write_csv(
        metrics_pooled,
        OUT_DIR
        / "representation_metrics_pooled.csv",
    )

    write_csv(
        bootstrap,
        OUT_DIR
        / "representation_bootstrap_ci.csv",
    )

    write_csv(
        differences,
        OUT_DIR
        / "representation_difference_bootstrap.csv",
    )

    write_csv(
        predictions,
        OUT_DIR
        / "representation_predictions.csv",
    )

    write_csv(
        strict_summary,
        OUT_DIR
        / "representation_strict_summary.csv",
    )

    plot_result(
        bootstrap,
        args.dpi,
    )

    status_counts = (
        alignment_audit[
            "status"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "dataset":
            str(
                DATASET_CSV
            ),

        "raw_root":
            str(
                RAW_ROOT
            ),

        "data_level":
            ONLY_LEVEL,

        "targets":
            list(
                TARGETS
            ),

        "representations": {
            name:
                len(
                    columns
                )
            for (
                name,
                columns,
            ) in feature_sets.items()
        },

        "splits":
            list(
                SPLITS
            ),

        "composite_minimum_component_count":
            MIN_COMPOSITE_COMPONENTS,

        "category_minimum_frequency":
            MIN_CATEGORY_FREQUENCY,

        "bootstrap_repeats":
            args.bootstrap,

        "dataset_L1_rows":
            int(
                len(
                    frame
                )
            ),

        "raw_matched_rows":
            matched_rows,

        "raw_matched_fraction":
            float(
                matched_rows
                / len(
                    frame
                )
            ),

        "raw_eligible_rows":
            eligible_rows,

        "raw_eligible_fraction":
            float(
                eligible_rows
                / len(
                    frame
                )
            ),

        "alignment_status_counts":
            status_counts,

        "successful_model_tasks":
            int(
                len(
                    metrics_fold
                )
            ),

        "prediction_rows":
            int(
                len(
                    predictions
                )
            ),

        "important_outputs": {
            "alignment_audit":
                str(
                    OUT_DIR
                    / "raw_source_alignment_audit.csv"
                ),

            "strict_summary":
                str(
                    OUT_DIR
                    / "representation_strict_summary.csv"
                ),

            "paired_differences":
                str(
                    OUT_DIR
                    / "representation_difference_bootstrap.csv"
                ),

            "main_figure":
                str(
                    FIGURE_DIR
                    / "fig_representation_baseline.png"
                ),
        },
    }

    write_json(
        summary,
        OUT_DIR
        / "representation_summary.json",
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        "[REPRESENTATION BASELINE FINISHED]"
    )

    print(
        "L1 dataset rows     : "
        f"{len(frame)}"
    )

    print(
        "Raw matched rows    : "
        f"{matched_rows}"
    )

    print(
        "Raw eligible rows   : "
        f"{eligible_rows}"
    )

    print(
        "Raw feature count   : "
        f"{len(raw_columns)}"
    )

    print(
        f"Output              : "
        f"{OUT_DIR}"
    )

    print(
        "Main table          : "
        "representation_strict_summary.csv"
    )

    print(
        "Paired differences  : "
        "representation_difference_bootstrap.csv"
    )

    print(
        "Main figure         : "
        "figures/fig_representation_baseline.png"
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()