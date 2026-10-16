# -*- coding: utf-8 -*-
"""
catalyst_family_holdout.py

用途
----
在现有 L1 + L2 文献数据上执行三层催化剂迁移验证：

1. exact
   精确元素组合留出，例如 Fe、Co、Fe-Co、Fe-Ni。

2. backbone
   按确定性“锚定元素 + 金属数”合并稀疏组合，例如：
   Fe_anchored_binary
   Co_anchored_binary
   Fe_anchored_ternary_plus

   锚定元素只用于建立互斥分组，不表示真实主金属，
   也不表示该元素含量最高。

3. element_presence
   留出全部含某一元素的样本。例如测试全部含 Co 的样本，
   训练集完全不含 Co。

每个任务均：
- 在 A-E 全部生长模式中全局留出目标催化剂化学集合；
- 使用其他催化剂样本重新训练 Composite>=2 的 q50 与 q90；
- 运行 family_only 与 family_and_paper 两种协议；
- 输出总体指标，并按 growth_mode 与 data_level 分层审计；
- 同时输出 Top 10% 与 Top 20% 排序/富集指标；
- 按测试论文进行 cluster bootstrap，输出 95% CI。

重要边界
--------
- 这是回顾性的伪未见验证，不生成新实验数据；
- 被留出的目标集合只用于测试，绝不参与该轮训练；
- q90 是条件 90% 分位数，不是置信上界、误差上界或风险；
- exact/backbone 评价未见元素组合或粗粒度化学骨架；
- element_presence 评价训练中完全未见某一元素的迁移；
- 生长模式不参与催化剂家族定义，只用于保留 F/P 工艺差异
  并进行分层审计；
- 本代码只生成验证证据，不替换正式 L12、L123 或 WorldMap 模型。

输入
----
<project_dir>/data/11_flattened/
    ml_dataset_3p0_full60D.csv
    feature_schema.json

原始催化剂来源
--------------
<project_dir>/data/08_finalized/{A..E}/{L1,L2}/*.json

输出
----
<project_dir>/outputs/external_validation/catalyst_family_holdout/
    sample_catalyst_family_map.csv
    catalyst_family_counts.csv
    family_holdout_split_manifest.csv
    family_holdout_normalization.csv
    family_holdout_predictions.csv
    family_holdout_metrics.csv
    family_holdout_metrics_by_mode.csv
    family_holdout_metrics_by_level.csv
    family_holdout_bootstrap_ci.csv
    family_holdout_skipped_families.csv
    family_holdout_summary.json
    source_data/
        fig_catalyst_family_holdout_source_data.csv
    figures/
        fig_catalyst_family_holdout.png
        fig_catalyst_family_holdout.pdf

快速检查
--------
python catalyst_family_holdout.py ^
    --bootstrap 200 ^
    --dpi 200 ^
    --protocols family_and_paper

正式运行
--------
python catalyst_family_holdout.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# =============================================================================
# 1. 路径与固定设置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "08_finalized"
)

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

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "external_validation"
    / "catalyst_family_holdout"
)

FIGURE_DIR = OUT_DIR / "figures"
SOURCE_DATA_DIR = OUT_DIR / "source_data"

TRAIN_LEVELS = ("L1", "L2")

SEED = 42


# -----------------------------------------------------------------------------
# 测试家族门槛
# -----------------------------------------------------------------------------

# 严格层：可以用于正文中的正式结论。
STRICT_MIN_TEST_SAMPLES = 30
STRICT_MIN_TEST_PAPERS = 5

# 探索层：用于扩大覆盖范围，建议放在 SI 或边界分析中。
EXPLORATORY_MIN_TEST_SAMPLES = 15
EXPLORATORY_MIN_TEST_PAPERS = 3

# 训练集最低要求。
MIN_TRAIN_SAMPLES = 100
MIN_TRAIN_PAPERS = 10

# 生长模式或数据等级分层评价门槛。
MIN_STRATUM_SAMPLES = 10
MIN_STRATUM_PAPERS = 3

# 某个模式占测试集比例超过该值时，标记为模式主导。
MODE_DOMINANCE_THRESHOLD = 0.80


# -----------------------------------------------------------------------------
# LightGBM训练设置
# -----------------------------------------------------------------------------

INNER_VALID_SIZE = 0.15

MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200


# -----------------------------------------------------------------------------
# Composite设置
# -----------------------------------------------------------------------------

MIN_COMPONENT_COUNT = 2

TOP_FRACTIONS = (
    0.10,
    0.20,
)

DEFAULT_BOOTSTRAP = 2000
DEFAULT_DPI = 600


# -----------------------------------------------------------------------------
# 验证协议
# -----------------------------------------------------------------------------

PROTOCOLS = (
    "family_only",
    "family_and_paper",
)

PRIMARY_PROTOCOL = "family_and_paper"


# -----------------------------------------------------------------------------
# 三种催化剂验证层级
# -----------------------------------------------------------------------------

FAMILY_SCHEMES = (
    "exact",
    "backbone",
    "element_presence",
)

ANALYSIS_TIERS = (
    "strict",
    "exploratory",
)


QUANTILES = {
    "q50": 0.50,
    "q90": 0.90,
}


LGB_BASE_PARAMS: Dict[str, Any] = {
    "objective": "quantile",
    "metric": "quantile",
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


# =============================================================================
# 2. Composite定义
# =============================================================================

TARGET_COMPOSITE = "y_composite_final_score"


WEIGHTS: Dict[str, float] = {
    "gq": 0.30,
    "igid": 0.20,
    "ori": 0.20,
    "purity": 0.15,
    "bundle": 0.15,
}


COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

COL_IGID_PREF = "y_ig_id_ratio"
COL_IGID_LEGACY = "y_IG_ID"
COL_IGID_UNIFIED = "y_ig_id_ratio_unified"


RAW_GQ = "_raw_gq"
RAW_IGID = "_raw_igid"
RAW_ORI = "_raw_ori"
RAW_PURITY = "_raw_purity"
RAW_BUNDLE = "_raw_bundle"
RAW_BUNDLE_LOG = "_raw_bundle_log1p"
RAW_COMPONENT_COUNT = "_raw_component_count"


COMPONENT_COLUMNS = {
    "gq": "gq_01",
    "igid": "igid_norm01",
    "ori": "ori_01",
    "purity": "purity_01",
    "bundle": "bundle_norm01",
}


# =============================================================================
# 3. 催化剂元素与骨架定义
# =============================================================================

# 用于统一元素顺序，同时用于backbone的确定性锚定分组。
#
# 注意：
# 这里的顺序只用于生成一致且互斥的类别名称，
# 不代表含量、催化活性或真实主金属。
ELEMENT_ORDER = (
    "Fe",
    "Co",
    "Ni",
    "Mn",
    "Cu",
    "Cr",
    "Mo",
    "W",
    "V",
    "Ti",
    "Al",
    "Mg",
    "Zn",
    "Pt",
    "Pd",
    "Au",
    "Ag",
    "Ru",
    "Rh",
    "Ir",
    "Re",
    "B",
    "Si",
    "Sn",
    "Ce",
    "La",
    "Y",
    "Zr",
    "Hf",
)

ELEMENT_RANK = {
    element: index
    for index, element in enumerate(ELEMENT_ORDER)
}


# =============================================================================
# 4. 通用工具
# =============================================================================

def text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value).strip().split()
    )


def normalized_id(value: Any) -> str:
    return text(value).lower()


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


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


# =============================================================================
# 5. 基础评价指标
# =============================================================================

def safe_r2(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) < 2:
        return float("nan")

    if np.allclose(
        y_true,
        y_true[0],
    ):
        return float("nan")

    return float(
        r2_score(
            y_true,
            y_pred,
        )
    )


def safe_spearman(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) < 2:
        return float("nan")

    true_rank = (
        pd.Series(y_true)
        .rank(method="average")
        .to_numpy(float)
    )

    pred_rank = (
        pd.Series(y_pred)
        .rank(method="average")
        .to_numpy(float)
    )

    if np.std(true_rank) == 0:
        return float("nan")

    if np.std(pred_rank) == 0:
        return float("nan")

    return float(
        np.corrcoef(
            true_rank,
            pred_rank,
        )[0, 1]
    )


def pinball_loss(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    alpha: float,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    error = (
        y_true[finite]
        - y_pred[finite]
    )

    if len(error) == 0:
        return float("nan")

    loss = np.maximum(
        alpha * error,
        (alpha - 1.0) * error,
    )

    return float(
        np.mean(loss)
    )


def top_fraction_metrics(
    y_true: Sequence[float],
    y_score: Sequence[float],
    fraction: float,
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_score)
    )

    y_true = y_true[finite]
    y_score = y_score[finite]

    n = len(y_true)

    if n == 0:
        return {
            "k": 0,
            "ndcg": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "enrichment": np.nan,
            "hits": 0,
        }

    k = max(
        1,
        int(
            math.ceil(
                fraction * n
            )
        ),
    )

    predicted_order = np.argsort(
        -y_score,
        kind="mergesort",
    )

    true_order = np.argsort(
        -y_true,
        kind="mergesort",
    )

    discounts = (
        1.0
        / np.log2(
            np.arange(
                2,
                k + 2,
                dtype=float,
            )
        )
    )

    dcg = float(
        np.sum(
            y_true[
                predicted_order[:k]
            ]
            * discounts
        )
    )

    idcg = float(
        np.sum(
            y_true[
                true_order[:k]
            ]
            * discounts
        )
    )

    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = np.nan

    predicted_top = set(
        predicted_order[:k].tolist()
    )

    true_top = set(
        true_order[:k].tolist()
    )

    hits = len(
        predicted_top
        & true_top
    )

    precision = hits / k
    recall = hits / k

    random_rate = k / n

    if random_rate > 0:
        enrichment = precision / random_rate
    else:
        enrichment = np.nan

    return {
        "k": int(k),
        "ndcg": float(ndcg),
        "precision": float(precision),
        "recall": float(recall),
        "enrichment": float(enrichment),
        "hits": int(hits),
    }


def evaluate_holdout(
    y_true: Sequence[float],
    q50: Sequence[float],
    q90: Sequence[float],
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    q50 = np.asarray(
        q50,
        dtype=float,
    )

    q90 = np.asarray(
        q90,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(q50)
        & np.isfinite(q90)
    )

    y_true = y_true[finite]
    q50 = q50[finite]
    q90 = q90[finite]

    if len(y_true) == 0:
        return {
            "n": 0,
        }

    q90_coverage = float(
        np.mean(
            y_true <= q90
        )
    )

    result: Dict[str, float] = {
        "n": int(
            len(y_true)
        ),
        "q50_mae": float(
            mean_absolute_error(
                y_true,
                q50,
            )
        ),
        "q50_rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    q50,
                )
            )
        ),
        "q50_r2": safe_r2(
            y_true,
            q50,
        ),
        "q50_spearman": safe_spearman(
            y_true,
            q50,
        ),
        "q50_pinball": pinball_loss(
            y_true,
            q50,
            0.50,
        ),
        "q90_spearman": safe_spearman(
            y_true,
            q90,
        ),
        "q90_pinball": pinball_loss(
            y_true,
            q90,
            0.90,
        ),
        "q90_empirical_coverage": q90_coverage,
        "q90_calibration_error": abs(
            q90_coverage - 0.90
        ),
        "crossing_q50_q90_rate": float(
            np.mean(
                q50 > q90
            )
        ),
    }

    for fraction in TOP_FRACTIONS:
        label = int(
            round(
                fraction * 100
            )
        )

        q50_top = top_fraction_metrics(
            y_true,
            q50,
            fraction,
        )

        q90_top = top_fraction_metrics(
            y_true,
            q90,
            fraction,
        )

        result.update({
            f"top{label}pct_k":
                q50_top["k"],

            f"q50_ndcg_top{label}pct":
                q50_top["ndcg"],

            f"q50_precision_top{label}pct":
                q50_top["precision"],

            f"q50_enrichment_top{label}pct":
                q50_top["enrichment"],

            f"q50_top{label}pct_hits":
                q50_top["hits"],

            f"q90_ndcg_top{label}pct":
                q90_top["ndcg"],

            f"q90_precision_top{label}pct":
                q90_top["precision"],

            f"q90_enrichment_top{label}pct":
                q90_top["enrichment"],

            f"q90_top{label}pct_hits":
                q90_top["hits"],
        })

    return result


# =============================================================================
# 6. 读取特征schema
# =============================================================================

def load_feature_columns() -> List[str]:
    with SCHEMA_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(handle)

    columns = schema.get(
        "model_feature_columns"
    )

    if not isinstance(columns, list):
        raise RuntimeError(
            "feature_schema.json 缺少 model_feature_columns。"
        )

    if not columns:
        raise RuntimeError(
            "feature_schema.json 中 model_feature_columns 为空。"
        )

    columns = [
        str(column)
        for column in columns
    ]

    if len(columns) != 120:
        raise RuntimeError(
            "正式 L12 输入应为 120 维，"
            f"当前读取到 {len(columns)} 维。"
        )

    return columns


def ensure_igid_unified(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if COL_IGID_PREF in frame.columns:
        preferred = to_num(
            frame[COL_IGID_PREF]
        )
    else:
        preferred = pd.Series(
            np.nan,
            index=frame.index,
        )

    if COL_IGID_LEGACY in frame.columns:
        legacy = to_num(
            frame[COL_IGID_LEGACY]
        )
    else:
        legacy = pd.Series(
            np.nan,
            index=frame.index,
        )

    output = frame.copy()

    output[COL_IGID_UNIFIED] = preferred.where(
        preferred.notna(),
        legacy,
    )

    return output


# =============================================================================
# 7. 读取L1+L2数据并统计Composite数量
# =============================================================================

def load_base_data(
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    frame = pd.read_csv(
        DATASET_CSV,
        low_memory=False,
    )

    required = [
        "sample_id",
        "paper_id",
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
        frame["data_level"]
        .astype(str)
        .isin(TRAIN_LEVELS)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "筛选 L1 + L2 后没有数据。"
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_features:
        raise RuntimeError(
            "缺少模型特征列，示例："
            f"{missing_features[:10]}"
        )

    frame = ensure_igid_unified(
        frame
    )

    for column in (
        COL_GQ,
        COL_ORI,
        COL_PURITY,
        COL_BUNDLE,
        COL_IGID_UNIFIED,
    ):
        if column not in frame.columns:
            frame[column] = np.nan

    frame[RAW_GQ] = to_num(
        frame[COL_GQ]
    )

    frame[RAW_IGID] = to_num(
        frame[COL_IGID_UNIFIED]
    )

    frame[RAW_ORI] = to_num(
        frame[COL_ORI]
    )

    frame[RAW_PURITY] = to_num(
        frame[COL_PURITY]
    )

    frame[RAW_BUNDLE] = to_num(
        frame[COL_BUNDLE]
    )

    bundle = frame[
        RAW_BUNDLE
    ].to_numpy(float)

    frame[RAW_BUNDLE_LOG] = np.where(
        np.isfinite(bundle)
        & (bundle >= 0.0),
        np.log1p(bundle),
        np.nan,
    )

    availability = pd.DataFrame({
        "gq":
            frame[RAW_GQ].notna(),

        "igid":
            frame[RAW_IGID].notna(),

        "ori":
            frame[RAW_ORI].notna(),

        "purity":
            frame[RAW_PURITY].notna(),

        "bundle":
            frame[RAW_BUNDLE_LOG].notna(),
    })

    frame[RAW_COMPONENT_COUNT] = (
        availability
        .sum(axis=1)
        .astype(int)
    )

    # 这里记录筛选前后的精确数量。
    count_summary = {
        "l1_l2_rows_before_component_filter":
            int(
                len(frame)
            ),

        "composite_ge1_rows":
            int(
                (
                    frame[RAW_COMPONENT_COUNT]
                    >= 1
                ).sum()
            ),

        "composite_ge2_rows":
            int(
                (
                    frame[RAW_COMPONENT_COUNT]
                    >= 2
                ).sum()
            ),

        "single_component_rows_removed_by_ge2":
            int(
                (
                    frame[RAW_COMPONENT_COUNT]
                    == 1
                ).sum()
            ),
    }

    frame = frame[
        frame[RAW_COMPONENT_COUNT]
        >= MIN_COMPONENT_COUNT
    ].copy()

    string_columns = (
        "sample_id",
        "paper_id",
        "growth_mode",
        "data_level",
        "source_file",
        "synthesis_id",
    )

    for column in string_columns:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
        )

    if frame["sample_id"].duplicated().any():
        examples = frame.loc[
            frame["sample_id"].duplicated(
                keep=False
            ),
            "sample_id",
        ].head(10).tolist()

        raise RuntimeError(
            "L1/L2 中 sample_id 不唯一，示例："
            f"{examples}"
        )

    if (
        frame["paper_id"]
        .str.len()
        .eq(0)
        .any()
    ):
        raise RuntimeError(
            "存在空 paper_id，无法执行按论文划分与bootstrap。"
        )

    occurrence_keys = [
        "growth_mode",
        "data_level",
        "source_file",
        "synthesis_id",
    ]

    frame["_source_occurrence"] = (
        frame
        .groupby(
            occurrence_keys,
            dropna=False,
        )
        .cumcount()
        .astype(int)
    )

    for column in feature_columns:
        frame[column] = to_num(
            frame[column]
        )

    return (
        frame.reset_index(drop=True),
        count_summary,
    )


# =============================================================================
# 8. 从08_finalized提取催化剂元素
# =============================================================================

def source_stem(
    source_file: Any,
) -> str:
    name = os.path.basename(
        text(source_file)
    )

    lower = name.lower()

    if lower.endswith(".andjson"):
        return name[:-8]

    if lower.endswith(".json"):
        return name[:-5]

    return os.path.splitext(
        name
    )[0]


def build_case_insensitive_file_map(
    folder: Path,
) -> Dict[str, Path]:
    if not folder.is_dir():
        return {}

    return {
        path.stem.lower(): path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".json"
        )
    }


def resolve_source_path(
    mode: str,
    level: str,
    source_file: str,
    folder_maps: Dict[
        Tuple[str, str],
        Dict[str, Path],
    ],
) -> Tuple[Optional[Path], Path]:
    stem = source_stem(
        source_file
    )

    folder = (
        RAW_ROOT
        / mode
        / level
    )

    expected = folder / f"{stem}.json"

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
        folder_maps[key] = (
            build_case_insensitive_file_map(
                folder
            )
        )

    return (
        folder_maps[key].get(
            stem.lower()
        ),
        expected,
    )


def build_synthesis_map(
    document: Mapping[str, Any],
) -> Dict[
    str,
    List[Dict[str, Any]],
]:
    output: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    syntheses = document.get(
        "syntheses"
    )

    if not isinstance(syntheses, list):
        return output

    for synthesis in syntheses:
        if not isinstance(synthesis, dict):
            continue

        synthesis_id = normalized_id(
            synthesis.get(
                "synthesis_id"
            )
        )

        output[synthesis_id].append(
            synthesis
        )

    return output


def normalize_element(
    value: Any,
) -> str:
    raw = re.sub(
        r"[^A-Za-z]",
        "",
        text(value),
    )

    if not raw:
        return ""

    return (
        raw[0].upper()
        + raw[1:].lower()
    )


def sort_elements(
    elements: Sequence[str],
) -> List[str]:
    return sorted(
        set(elements),
        key=lambda item: (
            ELEMENT_RANK.get(
                item,
                999,
            ),
            item,
        ),
    )


def extract_elements(
    synthesis: Mapping[str, Any],
) -> List[str]:
    elements: set[str] = set()

    catalysts = synthesis.get(
        "catalysts"
    )

    if not isinstance(catalysts, list):
        return []

    for catalyst in catalysts:
        if not isinstance(catalyst, dict):
            continue

        metals = catalyst.get(
            "metals"
        )

        if not isinstance(metals, list):
            continue

        for item in metals:
            if not isinstance(item, dict):
                continue

            element = normalize_element(
                item.get(
                    "element"
                )
            )

            if element:
                elements.add(
                    element
                )

    return sort_elements(
        elements
    )


def exact_family(
    elements: Sequence[str],
) -> Tuple[str, str, int]:
    unique = sort_elements(
        elements
    )

    if not unique:
        return (
            "Unknown",
            "unknown",
            0,
        )

    size = len(unique)

    if size == 1:
        group = "single"

    elif size == 2:
        group = "binary"

    else:
        group = "ternary_or_more"

    family = "-".join(
        unique
    )

    return (
        family,
        group,
        size,
    )


def backbone_family(
    elements: Sequence[str],
) -> str:
    unique = sort_elements(
        elements
    )

    if not unique:
        return "Unknown"

    anchor = unique[0]
    size = len(unique)

    if size == 1:
        return (
            f"{anchor}_monometal"
        )

    if size == 2:
        return (
            f"{anchor}_anchored_binary"
        )

    return (
        f"{anchor}_anchored_ternary_plus"
    )


def attach_catalyst_families(
    frame: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
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
            List[Dict[str, Any]],
        ],
    ] = {}

    folder_maps: Dict[
        Tuple[str, str],
        Dict[str, Path],
    ] = {}

    family_rows: List[
        Dict[str, Any]
    ] = []

    audit_rows: List[
        Dict[str, Any]
    ] = []

    for _, row in frame.iterrows():
        mode = text(
            row["growth_mode"]
        )

        level = text(
            row["data_level"]
        )

        synthesis_id = normalized_id(
            row["synthesis_id"]
        )

        occurrence = int(
            row["_source_occurrence"]
        )

        path, expected = resolve_source_path(
            mode=mode,
            level=level,
            source_file=row["source_file"],
            folder_maps=folder_maps,
        )

        status = "ok"
        error = ""

        synthesis: Optional[
            Dict[str, Any]
        ] = None

        same_id_count = 0

        if path is None:
            status = "json_not_found"

        else:
            try:
                if path not in document_cache:
                    document_cache[path] = (
                        read_json(path)
                    )

                    synthesis_cache[path] = (
                        build_synthesis_map(
                            document_cache[path]
                        )
                    )

                matches = synthesis_cache[
                    path
                ].get(
                    synthesis_id,
                    [],
                )

                same_id_count = len(
                    matches
                )

                if not matches:
                    status = (
                        "synthesis_id_not_found"
                    )

                elif occurrence >= len(matches):
                    status = (
                        "synthesis_occurrence_not_found"
                    )

                else:
                    synthesis = matches[
                        occurrence
                    ]

            except Exception as exc:
                status = "json_read_failed"

                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        if synthesis is not None:
            elements = extract_elements(
                synthesis
            )
        else:
            elements = []

        (
            family,
            family_group,
            family_size,
        ) = exact_family(
            elements
        )

        backbone = backbone_family(
            elements
        )

        if (
            synthesis is not None
            and family == "Unknown"
        ):
            status = (
                "catalyst_elements_missing"
            )

        element_string = "-".join(
            elements
        )

        family_rows.append({
            "catalyst_elements":
                element_string,

            "catalyst_element_list":
                tuple(elements),

            "catalyst_family_exact":
                family,

            "catalyst_family_group":
                family_group,

            "catalyst_family_size":
                family_size,

            "catalyst_family_backbone":
                backbone,

            "catalyst_mapping_status":
                status,
        })

        audit_rows.append({
            "sample_id":
                row["sample_id"],

            "paper_id":
                row["paper_id"],

            "growth_mode":
                mode,

            "data_level":
                level,

            "source_file":
                row["source_file"],

            "synthesis_id":
                row["synthesis_id"],

            "synthesis_occurrence":
                occurrence,

            "expected_raw_json_path":
                str(expected),

            "resolved_raw_json_path":
                (
                    str(path)
                    if path is not None
                    else ""
                ),

            "json_file_found":
                int(path is not None),

            "synthesis_found":
                int(
                    synthesis is not None
                ),

            "same_id_count_in_source":
                same_id_count,

            "catalyst_elements":
                element_string,

            "catalyst_family_exact":
                family,

            "catalyst_family_group":
                family_group,

            "catalyst_family_size":
                family_size,

            "catalyst_family_backbone":
                backbone,

            "status":
                status,

            "error":
                error,
        })

    output = pd.concat(
        [
            frame.reset_index(
                drop=True
            ),
            pd.DataFrame(
                family_rows
            ),
        ],
        axis=1,
    )

    audit = pd.DataFrame(
        audit_rows
    )

    return (
        output,
        audit,
    )


# =============================================================================
# 9. 三种留出任务定义
# =============================================================================

def classify_analysis_tier(
    n_samples: int,
    n_papers: int,
) -> str:
    if (
        n_samples
        >= STRICT_MIN_TEST_SAMPLES
        and n_papers
        >= STRICT_MIN_TEST_PAPERS
    ):
        return "strict"

    if (
        n_samples
        >= EXPLORATORY_MIN_TEST_SAMPLES
        and n_papers
        >= EXPLORATORY_MIN_TEST_PAPERS
    ):
        return "exploratory"

    return "ineligible"


def task_mask(
    frame: pd.DataFrame,
    scheme: str,
    target: str,
) -> pd.Series:
    if scheme == "exact":
        return (
            frame[
                "catalyst_family_exact"
            ].astype(str)
            == target
        )

    if scheme == "backbone":
        return (
            frame[
                "catalyst_family_backbone"
            ].astype(str)
            == target
        )

    if scheme == "element_presence":
        return frame[
            "catalyst_element_list"
        ].map(
            lambda items: (
                target in items
            )
        )

    raise ValueError(
        "未知 family scheme："
        f"{scheme}"
    )


def build_task_catalog(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    mapped = frame[
        frame["catalyst_family_exact"]
        != "Unknown"
    ].copy()

    tasks: List[
        Dict[str, Any]
    ] = []

    exact_targets = sorted(
        mapped[
            "catalyst_family_exact"
        ].unique()
    )

    backbone_targets = sorted(
        mapped[
            "catalyst_family_backbone"
        ].unique()
    )

    elements = sorted(
        {
            element
            for items in mapped[
                "catalyst_element_list"
            ]
            for element in items
        },
        key=lambda item: (
            ELEMENT_RANK.get(
                item,
                999,
            ),
            item,
        ),
    )

    target_sets = {
        "exact":
            exact_targets,

        "backbone":
            backbone_targets,

        "element_presence":
            elements,
    }

    all_modes = sorted(
        mapped[
            "growth_mode"
        ].unique()
    )

    for scheme, targets in target_sets.items():
        for target in targets:
            mask = task_mask(
                mapped,
                scheme,
                target,
            )

            subset = mapped[
                mask
            ]

            n_samples = int(
                len(subset)
            )

            n_papers = int(
                subset[
                    "paper_id"
                ].nunique()
            )

            tier = classify_analysis_tier(
                n_samples,
                n_papers,
            )

            dominant_mode = ""
            dominant_mode_fraction = np.nan

            if n_samples > 0:
                mode_counts = (
                    subset[
                        "growth_mode"
                    ]
                    .value_counts(
                        dropna=False
                    )
                )

                dominant_mode = str(
                    mode_counts.index[0]
                )

                dominant_mode_fraction = float(
                    mode_counts.iloc[0]
                    / n_samples
                )

            row: Dict[str, Any] = {
                "family_scheme":
                    scheme,

                "held_out_target":
                    target,

                "sample_count":
                    n_samples,

                "paper_count":
                    n_papers,

                "analysis_tier":
                    tier,

                "eligible_for_holdout":
                    int(
                        tier
                        in ANALYSIS_TIERS
                    ),

                "dominant_mode":
                    dominant_mode,

                "dominant_mode_fraction":
                    dominant_mode_fraction,

                "mode_dominated":
                    int(
                        np.isfinite(
                            dominant_mode_fraction
                        )
                        and dominant_mode_fraction
                        >= MODE_DOMINANCE_THRESHOLD
                    ),
            }

            for level in TRAIN_LEVELS:
                level_mask = (
                    subset[
                        "data_level"
                    ]
                    == level
                )

                row[
                    f"{level}_samples"
                ] = int(
                    level_mask.sum()
                )

                row[
                    f"{level}_papers"
                ] = int(
                    subset.loc[
                        level_mask,
                        "paper_id",
                    ].nunique()
                )

            for mode in all_modes:
                mode_mask = (
                    subset[
                        "growth_mode"
                    ]
                    == mode
                )

                row[
                    f"mode_{mode}_samples"
                ] = int(
                    mode_mask.sum()
                )

                row[
                    f"mode_{mode}_papers"
                ] = int(
                    subset.loc[
                        mode_mask,
                        "paper_id",
                    ].nunique()
                )

            tasks.append(
                row
            )

    catalog = pd.DataFrame(
        tasks
    )

    catalog = catalog.sort_values(
        [
            "family_scheme",
            "eligible_for_holdout",
            "analysis_tier",
            "sample_count",
            "held_out_target",
        ],
        ascending=[
            True,
            False,
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return catalog


# =============================================================================
# 10. Composite归一化
# =============================================================================

def minmax_parameters(
    series: pd.Series,
) -> Dict[str, Any]:
    values = to_num(
        series
    ).to_numpy(float)

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return {
            "minimum": None,
            "maximum": None,
            "usable": False,
            "finite_count": 0,
            "reason": "no_finite_values",
        }

    minimum = float(
        np.min(finite)
    )

    maximum = float(
        np.max(finite)
    )

    usable = not math.isclose(
        minimum,
        maximum,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    return {
        "minimum":
            minimum,

        "maximum":
            maximum,

        "usable":
            usable,

        "finite_count":
            int(len(finite)),

        "reason":
            (
                "ok"
                if usable
                else "constant_values"
            ),
    }


def apply_minmax(
    series: pd.Series,
    minimum: Optional[float],
    maximum: Optional[float],
    usable: bool,
) -> pd.Series:
    if (
        not usable
        or minimum is None
        or maximum is None
    ):
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    normalized = (
        (
            to_num(series)
            - minimum
        )
        / (
            maximum
            - minimum
        )
    )

    return normalized.clip(
        0.0,
        1.0,
    )


def fit_composite_normalizer(
    work: pd.DataFrame,
    train_indices: np.ndarray,
) -> Dict[str, Any]:
    train = work.iloc[
        np.asarray(
            train_indices,
            dtype=int,
        )
    ]

    igid = minmax_parameters(
        train[
            RAW_IGID
        ]
    )

    bundle = minmax_parameters(
        train[
            RAW_BUNDLE_LOG
        ]
    )

    return {
        "igid_min":
            igid["minimum"],

        "igid_max":
            igid["maximum"],

        "igid_usable":
            igid["usable"],

        "igid_finite_count":
            igid["finite_count"],

        "igid_reason":
            igid["reason"],

        "bundle_log_min":
            bundle["minimum"],

        "bundle_log_max":
            bundle["maximum"],

        "bundle_usable":
            bundle["usable"],

        "bundle_finite_count":
            bundle["finite_count"],

        "bundle_reason":
            bundle["reason"],

        "fit_sample_count":
            int(
                len(train)
            ),

        "fit_paper_count":
            int(
                train[
                    "paper_id"
                ].nunique()
            ),
    }


def transform_composite(
    work: pd.DataFrame,
    normalizer: Mapping[str, Any],
) -> pd.DataFrame:
    output = pd.DataFrame(
        index=work.index
    )

    output["gq_01"] = (
        to_num(
            work[RAW_GQ]
        )
        / 10.0
    ).clip(
        0.0,
        1.0,
    )

    output["igid_norm01"] = apply_minmax(
        series=work[RAW_IGID],
        minimum=normalizer[
            "igid_min"
        ],
        maximum=normalizer[
            "igid_max"
        ],
        usable=bool(
            normalizer[
                "igid_usable"
            ]
        ),
    )

    output["ori_01"] = to_num(
        work[RAW_ORI]
    ).clip(
        0.0,
        1.0,
    )

    output["purity_01"] = to_num(
        work[RAW_PURITY]
    ).clip(
        0.0,
        1.0,
    )

    output["bundle_norm01"] = apply_minmax(
        series=work[RAW_BUNDLE_LOG],
        minimum=normalizer[
            "bundle_log_min"
        ],
        maximum=normalizer[
            "bundle_log_max"
        ],
        usable=bool(
            normalizer[
                "bundle_usable"
            ]
        ),
    )

    components = pd.DataFrame(
        {
            name: output[column]
            for name, column
            in COMPONENT_COLUMNS.items()
        },
        index=work.index,
    )

    weights = pd.Series(
        WEIGHTS,
        dtype=float,
    )

    available = components.notna()

    component_count = (
        available
        .sum(axis=1)
        .astype(int)
    )

    weight_sum = (
        available
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
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

    composite = pd.Series(
        np.where(
            weight_sum > 0,
            numerator / weight_sum,
            np.nan,
        ),
        index=work.index,
        dtype=float,
    )

    composite = composite.where(
        component_count
        >= MIN_COMPONENT_COUNT,
        np.nan,
    )

    output[
        TARGET_COMPOSITE
    ] = composite

    output[
        "composite_component_count"
    ] = component_count

    output[
        "composite_weight_sum_used"
    ] = weight_sum.where(
        composite.notna(),
        np.nan,
    )

    return output


# =============================================================================
# 11. 分位数模型
# =============================================================================

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

    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=INNER_VALID_SIZE,
            random_state=SEED,
        )

        (
            relative_fit,
            relative_valid,
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
                relative_fit
            ],
            outer_train_indices[
                relative_valid
            ],
            "group_shuffle",
        )

    relative = np.arange(
        len(
            outer_train_indices
        )
    )

    (
        relative_fit,
        relative_valid,
    ) = train_test_split(
        relative,
        test_size=INNER_VALID_SIZE,
        random_state=SEED,
    )

    return (
        outer_train_indices[
            relative_fit
        ],
        outer_train_indices[
            relative_valid
        ],
        "random_fallback",
    )


def quantile_params(
    alpha: float,
) -> Dict[str, Any]:
    parameters = dict(
        LGB_BASE_PARAMS
    )

    parameters[
        "alpha"
    ] = alpha

    return parameters


def fit_quantile(
    X: pd.DataFrame,
    y: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    alpha: float,
) -> Tuple[
    lgb.Booster,
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
        params=quantile_params(
            alpha
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

    if (
        model.best_iteration
        and model.best_iteration > 0
    ):
        best_iteration = int(
            model.best_iteration
        )
    else:
        best_iteration = (
            MAX_BOOST_ROUNDS
        )

    return (
        model,
        best_iteration,
    )


# =============================================================================
# 12. 分层指标
# =============================================================================

def stratified_metrics(
    predictions: pd.DataFrame,
    group_column: str,
    family_scheme: str,
    held_target: str,
    analysis_tier: str,
    protocol: str,
    training_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    groups = predictions.groupby(
        group_column,
        dropna=False,
        sort=True,
    )

    for group_value, subset in groups:
        value = str(
            group_value
        )

        n_samples = int(
            len(subset)
        )

        n_papers = int(
            subset[
                "paper_id"
            ].nunique()
        )

        training_same_group = training_frame[
            training_frame[
                group_column
            ].astype(str)
            == value
        ]

        if (
            n_samples
            >= MIN_STRATUM_SAMPLES
            and n_papers
            >= MIN_STRATUM_PAPERS
        ):
            status = "ok"

            metrics = evaluate_holdout(
                subset[
                    "y_true"
                ],
                subset[
                    "q50_raw"
                ],
                subset[
                    "q90_raw"
                ],
            )

        else:
            status = (
                "descriptive_only_"
                "too_few_samples_or_papers"
            )

            metrics = {}

        rows.append({
            "family_scheme":
                family_scheme,

            "held_out_target":
                held_target,

            "analysis_tier":
                analysis_tier,

            "protocol":
                protocol,

            group_column:
                value,

            "n_test":
                n_samples,

            "n_test_papers":
                n_papers,

            "same_group_train_samples":
                int(
                    len(
                        training_same_group
                    )
                ),

            "same_group_train_papers":
                int(
                    training_same_group[
                        "paper_id"
                    ].nunique()
                ),

            "status":
                status,

            **metrics,
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 13. 论文级bootstrap
# =============================================================================

BOOTSTRAP_METRICS = (
    "q50_mae",
    "q50_spearman",
    "q90_spearman",
    "q90_pinball",
    "q90_empirical_coverage",

    "q50_ndcg_top10pct",
    "q90_ndcg_top10pct",
    "q50_enrichment_top10pct",
    "q90_enrichment_top10pct",

    "q50_ndcg_top20pct",
    "q90_ndcg_top20pct",
    "q50_enrichment_top20pct",
    "q90_enrichment_top20pct",
)


def bootstrap_by_paper(
    predictions: pd.DataFrame,
    family_scheme: str,
    held_target: str,
    analysis_tier: str,
    protocol: str,
    repeats: int,
) -> pd.DataFrame:
    paper_groups = {
        paper_id:
            group.copy()

        for paper_id, group
        in predictions.groupby(
            "paper_id",
            sort=False,
        )
    }

    papers = list(
        paper_groups
    )

    if len(papers) < 2:
        return pd.DataFrame()

    point = evaluate_holdout(
        predictions[
            "y_true"
        ],
        predictions[
            "q50_raw"
        ],
        predictions[
            "q90_raw"
        ],
    )

    distributions = {
        metric: []
        for metric
        in BOOTSTRAP_METRICS
    }

    seed_text = (
        f"{family_scheme}|"
        f"{held_target}|"
        f"{analysis_tier}|"
        f"{protocol}"
    )

    seed_offset = sum(
        ord(char)
        for char in seed_text
    )

    rng = np.random.default_rng(
        SEED
        + seed_offset
    )

    for _ in range(repeats):
        sampled_papers = rng.choice(
            papers,
            size=len(papers),
            replace=True,
        )

        sample_parts = []

        for index, paper in enumerate(
            sampled_papers
        ):
            part = (
                paper_groups[
                    paper
                ]
                .copy()
                .assign(
                    _bootstrap_paper_instance=index
                )
            )

            sample_parts.append(
                part
            )

        sample = pd.concat(
            sample_parts,
            ignore_index=True,
        )

        result = evaluate_holdout(
            sample[
                "y_true"
            ],
            sample[
                "q50_raw"
            ],
            sample[
                "q90_raw"
            ],
        )

        for metric in BOOTSTRAP_METRICS:
            value = result.get(
                metric,
                np.nan,
            )

            if np.isfinite(value):
                distributions[
                    metric
                ].append(
                    float(value)
                )

    rows: List[
        Dict[str, Any]
    ] = []

    for metric in BOOTSTRAP_METRICS:
        values = np.asarray(
            distributions[
                metric
            ],
            dtype=float,
        )

        if len(values):
            ci_lower = float(
                np.quantile(
                    values,
                    0.025,
                )
            )

            ci_upper = float(
                np.quantile(
                    values,
                    0.975,
                )
            )

        else:
            ci_lower = np.nan
            ci_upper = np.nan

        rows.append({
            "family_scheme":
                family_scheme,

            "held_out_target":
                held_target,

            "analysis_tier":
                analysis_tier,

            "protocol":
                protocol,

            "metric":
                metric,

            "estimate":
                point.get(
                    metric,
                    np.nan,
                ),

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            "valid_repeats":
                int(
                    len(values)
                ),

            "test_samples":
                int(
                    len(
                        predictions
                    )
                ),

            "test_papers":
                int(
                    len(papers)
                ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 14. 主holdout循环
# =============================================================================

def run_holdouts(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    task_catalog: pd.DataFrame,
    schemes: Sequence[str],
    tiers: Sequence[str],
    protocols: Sequence[str],
    bootstrap_repeats: int,
) -> Dict[str, pd.DataFrame]:
    X = frame[
        list(
            feature_columns
        )
    ].copy()

    mapped_mask = (
        frame[
            "catalyst_family_exact"
        ]
        != "Unknown"
    )

    selected_tasks = task_catalog[
        task_catalog[
            "family_scheme"
        ].isin(schemes)
        & task_catalog[
            "analysis_tier"
        ].isin(tiers)
        & (
            task_catalog[
                "eligible_for_holdout"
            ]
            == 1
        )
    ].copy()

    prediction_frames: List[
        pd.DataFrame
    ] = []

    metric_rows: List[
        Dict[str, Any]
    ] = []

    mode_frames: List[
        pd.DataFrame
    ] = []

    level_frames: List[
        pd.DataFrame
    ] = []

    bootstrap_frames: List[
        pd.DataFrame
    ] = []

    split_rows: List[
        Dict[str, Any]
    ] = []

    normalization_rows: List[
        Dict[str, Any]
    ] = []

    skipped_rows: List[
        Dict[str, Any]
    ] = []

    total_tasks = int(
        len(selected_tasks)
        * len(protocols)
    )

    task_number = 0

    for _, task in selected_tasks.iterrows():
        family_scheme = str(
            task[
                "family_scheme"
            ]
        )

        held_target = str(
            task[
                "held_out_target"
            ]
        )

        analysis_tier = str(
            task[
                "analysis_tier"
            ]
        )

        base_test_mask = (
            task_mask(
                frame,
                family_scheme,
                held_target,
            )
            & mapped_mask
        )

        test_papers = set(
            frame.loc[
                base_test_mask,
                "paper_id",
            ].astype(str)
        )

        for protocol in protocols:
            task_number += 1

            print(
                f"[{task_number:03d}/"
                f"{total_tasks:03d}] "
                f"scheme={family_scheme} | "
                f"target={held_target} | "
                f"tier={analysis_tier} | "
                f"protocol={protocol}"
            )

            # 催化剂在A-E所有模式中全局留出。
            train_mask = (
                mapped_mask
                & ~base_test_mask
            )

            # 严格协议进一步删除所有测试论文中的其他样本。
            if protocol == "family_and_paper":
                train_mask &= ~(
                    frame[
                        "paper_id"
                    ]
                    .astype(str)
                    .isin(
                        test_papers
                    )
                )

            train_indices_initial = np.where(
                train_mask.to_numpy()
            )[0]

            test_indices_initial = np.where(
                base_test_mask.to_numpy()
            )[0]

            train_paper_set = set(
                frame.iloc[
                    train_indices_initial
                ][
                    "paper_id"
                ]
            )

            test_paper_set = set(
                frame.iloc[
                    test_indices_initial
                ][
                    "paper_id"
                ]
            )

            paper_overlap = len(
                train_paper_set
                & test_paper_set
            )

            initial_train_papers = int(
                frame.iloc[
                    train_indices_initial
                ][
                    "paper_id"
                ].nunique()
            )

            initial_test_papers = int(
                frame.iloc[
                    test_indices_initial
                ][
                    "paper_id"
                ].nunique()
            )

            if (
                len(train_indices_initial)
                < MIN_TRAIN_SAMPLES
                or initial_train_papers
                < MIN_TRAIN_PAPERS
            ):
                skipped_rows.append({
                    "family_scheme":
                        family_scheme,

                    "held_out_target":
                        held_target,

                    "analysis_tier":
                        analysis_tier,

                    "protocol":
                        protocol,

                    "reason":
                        (
                            "too_few_training_"
                            "samples_or_papers_"
                            "before_normalization"
                        ),

                    "n_train":
                        int(
                            len(
                                train_indices_initial
                            )
                        ),

                    "n_train_papers":
                        initial_train_papers,

                    "n_test":
                        int(
                            len(
                                test_indices_initial
                            )
                        ),

                    "n_test_papers":
                        initial_test_papers,
                })

                print(
                    "    skipped: "
                    "training set too small"
                )

                continue

            # Composite归一化仅使用当前任务训练集。
            normalizer = fit_composite_normalizer(
                frame,
                train_indices_initial,
            )

            transformed = transform_composite(
                frame,
                normalizer,
            )

            y = to_num(
                transformed[
                    TARGET_COMPOSITE
                ]
            ).to_numpy(float)

            train_indices = train_indices_initial[
                np.isfinite(
                    y[
                        train_indices_initial
                    ]
                )
            ]

            test_indices = test_indices_initial[
                np.isfinite(
                    y[
                        test_indices_initial
                    ]
                )
            ]

            n_train_papers = int(
                frame.iloc[
                    train_indices
                ][
                    "paper_id"
                ].nunique()
            )

            n_test_papers = int(
                frame.iloc[
                    test_indices
                ][
                    "paper_id"
                ].nunique()
            )

            tier_after_normalization = classify_analysis_tier(
                n_samples=int(
                    len(
                        test_indices
                    )
                ),
                n_papers=n_test_papers,
            )

            if (
                len(train_indices)
                < MIN_TRAIN_SAMPLES
                or n_train_papers
                < MIN_TRAIN_PAPERS
                or tier_after_normalization
                == "ineligible"
            ):
                skipped_rows.append({
                    "family_scheme":
                        family_scheme,

                    "held_out_target":
                        held_target,

                    "analysis_tier":
                        analysis_tier,

                    "protocol":
                        protocol,

                    "reason":
                        (
                            "too_few_valid_"
                            "samples_or_papers_"
                            "after_normalization"
                        ),

                    "n_train":
                        int(
                            len(
                                train_indices
                            )
                        ),

                    "n_train_papers":
                        n_train_papers,

                    "n_test":
                        int(
                            len(
                                test_indices
                            )
                        ),

                    "n_test_papers":
                        n_test_papers,
                })

                print(
                    "    skipped: "
                    "valid train/test set too small "
                    "after normalization"
                )

                continue

            effective_tier = (
                tier_after_normalization
            )

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

            models: Dict[
                str,
                lgb.Booster,
            ] = {}

            iterations: Dict[
                str,
                int,
            ] = {}

            for (
                quantile_name,
                alpha,
            ) in QUANTILES.items():
                (
                    model,
                    best_iteration,
                ) = fit_quantile(
                    X=X,
                    y=y,
                    fit_indices=fit_indices,
                    valid_indices=valid_indices,
                    alpha=alpha,
                )

                models[
                    quantile_name
                ] = model

                iterations[
                    quantile_name
                ] = best_iteration

                print(
                    f"    {quantile_name} trained, "
                    f"best_iteration="
                    f"{best_iteration}"
                )

            q50 = models[
                "q50"
            ].predict(
                X.iloc[
                    test_indices
                ],
                num_iteration=iterations[
                    "q50"
                ],
            )

            q90 = models[
                "q90"
            ].predict(
                X.iloc[
                    test_indices
                ],
                num_iteration=iterations[
                    "q90"
                ],
            )

            y_true = y[
                test_indices
            ]

            predictions = frame.iloc[
                test_indices
            ][[
                "sample_id",
                "paper_id",
                "growth_mode",
                "data_level",
                "source_file",
                "synthesis_id",
                "catalyst_elements",
                "catalyst_family_exact",
                "catalyst_family_backbone",
            ]].copy()

            predictions.insert(
                0,
                "protocol",
                protocol,
            )

            predictions.insert(
                0,
                "analysis_tier",
                effective_tier,
            )

            predictions.insert(
                0,
                "held_out_target",
                held_target,
            )

            predictions.insert(
                0,
                "family_scheme",
                family_scheme,
            )

            predictions[
                "y_true"
            ] = y_true

            predictions[
                "q50_raw"
            ] = q50

            predictions[
                "q90_raw"
            ] = q90

            predictions[
                "q50_q90_crossing"
            ] = (
                q50 > q90
            ).astype(int)

            predictions[
                "q50_ordered"
            ] = np.minimum(
                q50,
                q90,
            )

            predictions[
                "q90_ordered"
            ] = np.maximum(
                q50,
                q90,
            )

            predictions[
                "q50_absolute_error"
            ] = np.abs(
                q50
                - y_true
            )

            prediction_frames.append(
                predictions
            )

            metrics = evaluate_holdout(
                y_true,
                q50,
                q90,
            )

            test_mode_counts = (
                predictions[
                    "growth_mode"
                ]
                .value_counts(
                    dropna=False
                )
            )

            dominant_mode = str(
                test_mode_counts.index[0]
            )

            dominant_mode_fraction = float(
                test_mode_counts.iloc[0]
                / len(predictions)
            )

            metric_rows.append({
                "family_scheme":
                    family_scheme,

                "held_out_target":
                    held_target,

                "analysis_tier":
                    effective_tier,

                "protocol":
                    protocol,

                "status":
                    "ok",

                "n_train":
                    int(
                        len(
                            train_indices
                        )
                    ),

                "n_train_papers":
                    n_train_papers,

                "n_inner_fit":
                    int(
                        len(
                            fit_indices
                        )
                    ),

                "n_inner_valid":
                    int(
                        len(
                            valid_indices
                        )
                    ),

                "n_test":
                    int(
                        len(
                            test_indices
                        )
                    ),

                "n_test_papers":
                    n_test_papers,

                "paper_overlap_count":
                    int(
                        paper_overlap
                    ),

                "inner_split_method":
                    inner_method,

                "q50_best_iteration":
                    iterations[
                        "q50"
                    ],

                "q90_best_iteration":
                    iterations[
                        "q90"
                    ],

                "dominant_mode":
                    dominant_mode,

                "dominant_mode_fraction":
                    dominant_mode_fraction,

                "mode_dominated":
                    int(
                        dominant_mode_fraction
                        >= MODE_DOMINANCE_THRESHOLD
                    ),

                **metrics,
            })

            normalization_rows.append({
                "family_scheme":
                    family_scheme,

                "held_out_target":
                    held_target,

                "analysis_tier":
                    effective_tier,

                "protocol":
                    protocol,

                **normalizer,
            })

            for role, indices in (
                (
                    "train",
                    train_indices,
                ),
                (
                    "inner_fit",
                    fit_indices,
                ),
                (
                    "inner_valid",
                    valid_indices,
                ),
                (
                    "test",
                    test_indices,
                ),
            ):
                for index in indices:
                    split_rows.append({
                        "family_scheme":
                            family_scheme,

                        "held_out_target":
                            held_target,

                        "analysis_tier":
                            effective_tier,

                        "protocol":
                            protocol,

                        "role":
                            role,

                        "sample_id":
                            frame.iloc[
                                index
                            ][
                                "sample_id"
                            ],

                        "paper_id":
                            frame.iloc[
                                index
                            ][
                                "paper_id"
                            ],

                        "growth_mode":
                            frame.iloc[
                                index
                            ][
                                "growth_mode"
                            ],

                        "data_level":
                            frame.iloc[
                                index
                            ][
                                "data_level"
                            ],

                        "catalyst_elements":
                            frame.iloc[
                                index
                            ][
                                "catalyst_elements"
                            ],

                        "catalyst_family_exact":
                            frame.iloc[
                                index
                            ][
                                "catalyst_family_exact"
                            ],

                        "catalyst_family_backbone":
                            frame.iloc[
                                index
                            ][
                                "catalyst_family_backbone"
                            ],
                    })

            training_frame = frame.iloc[
                train_indices
            ]

            mode_frames.append(
                stratified_metrics(
                    predictions=predictions,
                    group_column="growth_mode",
                    family_scheme=family_scheme,
                    held_target=held_target,
                    analysis_tier=effective_tier,
                    protocol=protocol,
                    training_frame=training_frame,
                )
            )

            level_frames.append(
                stratified_metrics(
                    predictions=predictions,
                    group_column="data_level",
                    family_scheme=family_scheme,
                    held_target=held_target,
                    analysis_tier=effective_tier,
                    protocol=protocol,
                    training_frame=training_frame,
                )
            )

            print(
                "    bootstrap by paper: "
                f"{bootstrap_repeats} repeats"
            )

            bootstrap_result = bootstrap_by_paper(
                predictions=predictions,
                family_scheme=family_scheme,
                held_target=held_target,
                analysis_tier=effective_tier,
                protocol=protocol,
                repeats=bootstrap_repeats,
            )

            if not bootstrap_result.empty:
                bootstrap_frames.append(
                    bootstrap_result
                )

            print(
                "    completed"
            )

    if prediction_frames:
        predictions_output = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    else:
        predictions_output = pd.DataFrame()

    if mode_frames:
        mode_output = pd.concat(
            mode_frames,
            ignore_index=True,
        )
    else:
        mode_output = pd.DataFrame()

    if level_frames:
        level_output = pd.concat(
            level_frames,
            ignore_index=True,
        )
    else:
        level_output = pd.DataFrame()

    if bootstrap_frames:
        bootstrap_output = pd.concat(
            bootstrap_frames,
            ignore_index=True,
        )
    else:
        bootstrap_output = pd.DataFrame()

    return {
        "predictions":
            predictions_output,

        "metrics":
            pd.DataFrame(
                metric_rows
            ),

        "metrics_by_mode":
            mode_output,

        "metrics_by_level":
            level_output,

        "bootstrap":
            bootstrap_output,

        "splits":
            pd.DataFrame(
                split_rows
            ),

        "normalizations":
            pd.DataFrame(
                normalization_rows
            ),

        "skipped":
            pd.DataFrame(
                skipped_rows
            ),
    }


# =============================================================================
# 15. 作图
# =============================================================================

def plot_results(
    bootstrap: pd.DataFrame,
    task_catalog: pd.DataFrame,
    dpi: int,
) -> str:
    if bootstrap.empty:
        return (
            "not_created_no_bootstrap_results"
        )

    protocol = PRIMARY_PROTOCOL

    if not (
        bootstrap[
            "protocol"
        ]
        == protocol
    ).any():
        protocol = "family_only"

    metrics_to_plot = (
        (
            "q50_spearman",
            "q50 Spearman",
            0.0,
        ),
        (
            "q90_enrichment_top20pct",
            "q90 Enrichment@Top20%",
            1.0,
        ),
    )

    allowed_metrics = [
        item[0]
        for item in metrics_to_plot
    ]

    source = bootstrap[
        (
            bootstrap[
                "protocol"
            ]
            == protocol
        )
        & bootstrap[
            "metric"
        ].isin(
            allowed_metrics
        )
    ].copy()

    if source.empty:
        return (
            "not_created_no_primary_protocol_results"
        )

    write_csv(
        source,
        SOURCE_DATA_DIR
        / "fig_catalyst_family_holdout_source_data.csv",
    )

    figure, axes = plt.subplots(
        nrows=len(FAMILY_SCHEMES),
        ncols=2,
        figsize=(
            14.0,
            4.2
            * len(FAMILY_SCHEMES),
        ),
        squeeze=False,
    )

    count_lookup = (
        task_catalog
        .set_index([
            "family_scheme",
            "held_out_target",
        ])[
            "sample_count"
        ]
        .to_dict()
    )

    for row_index, scheme in enumerate(
        FAMILY_SCHEMES
    ):
        scheme_source = source[
            source[
                "family_scheme"
            ]
            == scheme
        ].copy()

        targets = sorted(
            scheme_source[
                "held_out_target"
            ].unique(),
            key=lambda target: (
                -count_lookup.get(
                    (
                        scheme,
                        target,
                    ),
                    0,
                ),
                target,
            ),
        )

        for (
            col_index,
            (
                metric,
                ylabel,
                reference,
            ),
        ) in enumerate(
            metrics_to_plot
        ):
            axis = axes[
                row_index,
                col_index,
            ]

            part = scheme_source[
                scheme_source[
                    "metric"
                ]
                == metric
            ].copy()

            part = part.set_index(
                "held_out_target"
            )

            strict_x: List[int] = []
            strict_y: List[float] = []
            strict_low: List[float] = []
            strict_high: List[float] = []

            exploratory_x: List[int] = []
            exploratory_y: List[float] = []
            exploratory_low: List[float] = []
            exploratory_high: List[float] = []

            for position, target in enumerate(
                targets
            ):
                if target not in part.index:
                    continue

                row = part.loc[
                    target
                ]

                if isinstance(
                    row,
                    pd.DataFrame,
                ):
                    row = row.iloc[0]

                estimate = float(
                    row[
                        "estimate"
                    ]
                )

                ci_lower = float(
                    row[
                        "ci_lower"
                    ]
                )

                ci_upper = float(
                    row[
                        "ci_upper"
                    ]
                )

                tier = str(
                    row[
                        "analysis_tier"
                    ]
                )

                lower_error = max(
                    0.0,
                    estimate
                    - ci_lower,
                )

                upper_error = max(
                    0.0,
                    ci_upper
                    - estimate,
                )

                if tier == "strict":
                    strict_x.append(
                        position
                    )

                    strict_y.append(
                        estimate
                    )

                    strict_low.append(
                        lower_error
                    )

                    strict_high.append(
                        upper_error
                    )

                else:
                    exploratory_x.append(
                        position
                    )

                    exploratory_y.append(
                        estimate
                    )

                    exploratory_low.append(
                        lower_error
                    )

                    exploratory_high.append(
                        upper_error
                    )

            if strict_x:
                axis.errorbar(
                    strict_x,
                    strict_y,
                    yerr=np.vstack([
                        strict_low,
                        strict_high,
                    ]),
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    label="Strict",
                )

            if exploratory_x:
                axis.errorbar(
                    exploratory_x,
                    exploratory_y,
                    yerr=np.vstack([
                        exploratory_low,
                        exploratory_high,
                    ]),
                    marker="s",
                    linestyle="none",
                    capsize=3,
                    label="Exploratory",
                )

            axis.axhline(
                reference,
                linestyle="--",
                linewidth=1.0,
            )

            axis.set_xticks(
                np.arange(
                    len(targets),
                    dtype=float,
                )
            )

            axis.set_xticklabels(
                targets,
                rotation=35,
                ha="right",
            )

            axis.set_ylabel(
                ylabel
            )

            axis.set_xlabel(
                "Held-out catalyst target"
            )

            axis.grid(
                axis="y",
                alpha=0.25,
            )

            axis.set_title(
                f"{scheme}: {ylabel}"
            )

            if strict_x or exploratory_x:
                axis.legend(
                    frameon=False
                )

    figure.suptitle(
        "Catalyst transferability across "
        "A-E growth modes "
        f"({protocol.replace('_', ' ')})"
    )

    figure.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig_catalyst_family_holdout.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig_catalyst_family_holdout.pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return protocol


# =============================================================================
# 16. 命令行参数
# =============================================================================

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
        "--protocols",
        nargs="+",
        choices=PROTOCOLS,
        default=list(
            PROTOCOLS
        ),
        help=(
            "默认同时运行 family_only "
            "与 family_and_paper。"
        ),
    )

    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=FAMILY_SCHEMES,
        default=list(
            FAMILY_SCHEMES
        ),
        help=(
            "默认运行 exact、backbone "
            "和 element_presence。"
        ),
    )

    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=ANALYSIS_TIERS,
        default=list(
            ANALYSIS_TIERS
        ),
        help=(
            "默认同时运行 strict "
            "与 exploratory。"
        ),
    )

    parser.add_argument(
        "--keep_output",
        action="store_true",
        help=(
            "保留已有输出目录；"
            "默认运行前删除旧输出。"
        ),
    )

    return parser.parse_args()


def check_inputs() -> None:
    missing = [
        str(path)
        for path
        in (
            DATASET_CSV,
            SCHEMA_JSON,
        )
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


# =============================================================================
# 17. 主程序
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.bootstrap < 1:
        raise ValueError(
            "--bootstrap 必须大于0。"
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
        "=" * 104
    )

    print(
        "[CATALYST FAMILY HOLDOUT START]"
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
        f"Training levels     : "
        f"{TRAIN_LEVELS}"
    )

    print(
        f"Composite threshold : "
        f"{MIN_COMPONENT_COUNT}"
    )

    print(
        f"Schemes             : "
        f"{tuple(args.schemes)}"
    )

    print(
        f"Tiers               : "
        f"{tuple(args.tiers)}"
    )

    print(
        f"Protocols           : "
        f"{tuple(args.protocols)}"
    )

    print(
        f"Bootstrap repeats   : "
        f"{args.bootstrap}"
    )

    print(
        "=" * 104
    )

    feature_columns = (
        load_feature_columns()
    )

    (
        frame,
        component_count_summary,
    ) = load_base_data(
        feature_columns
    )

    (
        frame,
        mapping_audit,
    ) = attach_catalyst_families(
        frame
    )

    mapped = (
        frame[
            "catalyst_family_exact"
        ]
        != "Unknown"
    )

    if int(mapped.sum()) == 0:
        write_csv(
            mapping_audit,
            OUT_DIR
            / "sample_catalyst_family_map.csv",
        )

        raise RuntimeError(
            "没有任何样本成功识别催化剂家族，"
            "请检查 sample_catalyst_family_map.csv。"
        )

    task_catalog = build_task_catalog(
        frame
    )

    selected_catalog = task_catalog[
        task_catalog[
            "family_scheme"
        ].isin(
            args.schemes
        )
        & task_catalog[
            "analysis_tier"
        ].isin(
            args.tiers
        )
        & (
            task_catalog[
                "eligible_for_holdout"
            ]
            == 1
        )
    ]

    if selected_catalog.empty:
        write_csv(
            mapping_audit,
            OUT_DIR
            / "sample_catalyst_family_map.csv",
        )

        write_csv(
            task_catalog,
            OUT_DIR
            / "catalyst_family_counts.csv",
        )

        raise RuntimeError(
            "没有任何催化剂任务满足当前 "
            "scheme/tier 门槛。"
        )

    print(
        "\nEligible holdout targets:"
    )

    for scheme in args.schemes:
        subset = selected_catalog[
            selected_catalog[
                "family_scheme"
            ]
            == scheme
        ]

        strict_n = int(
            (
                subset[
                    "analysis_tier"
                ]
                == "strict"
            ).sum()
        )

        exploratory_n = int(
            (
                subset[
                    "analysis_tier"
                ]
                == "exploratory"
            ).sum()
        )

        print(
            f"  {scheme:<18} "
            f"strict={strict_n:<3d} "
            f"exploratory={exploratory_n:<3d}"
        )

    print()

    results = run_holdouts(
        frame=frame,
        feature_columns=feature_columns,
        task_catalog=task_catalog,
        schemes=args.schemes,
        tiers=args.tiers,
        protocols=args.protocols,
        bootstrap_repeats=args.bootstrap,
    )

    write_csv(
        mapping_audit,
        OUT_DIR
        / "sample_catalyst_family_map.csv",
    )

    write_csv(
        task_catalog,
        OUT_DIR
        / "catalyst_family_counts.csv",
    )

    write_csv(
        results["splits"],
        OUT_DIR
        / "family_holdout_split_manifest.csv",
    )

    write_csv(
        results["normalizations"],
        OUT_DIR
        / "family_holdout_normalization.csv",
    )

    write_csv(
        results["predictions"],
        OUT_DIR
        / "family_holdout_predictions.csv",
    )

    write_csv(
        results["metrics"],
        OUT_DIR
        / "family_holdout_metrics.csv",
    )

    write_csv(
        results["metrics_by_mode"],
        OUT_DIR
        / "family_holdout_metrics_by_mode.csv",
    )

    write_csv(
        results["metrics_by_level"],
        OUT_DIR
        / "family_holdout_metrics_by_level.csv",
    )

    write_csv(
        results["bootstrap"],
        OUT_DIR
        / "family_holdout_bootstrap_ci.csv",
    )

    write_csv(
        results["skipped"],
        OUT_DIR
        / "family_holdout_skipped_families.csv",
    )

    figure_protocol = plot_results(
        bootstrap=results[
            "bootstrap"
        ],
        task_catalog=task_catalog,
        dpi=args.dpi,
    )

    if not results["metrics"].empty:
        completed_tasks = (
            results[
                "metrics"
            ][[
                "family_scheme",
                "held_out_target",
                "analysis_tier",
                "protocol",
            ]]
            .drop_duplicates()
            .to_dict(
                "records"
            )
        )
    else:
        completed_tasks = []

    eligible_summary: Dict[
        str,
        Dict[str, int],
    ] = {}

    for scheme in FAMILY_SCHEMES:
        subset = task_catalog[
            (
                task_catalog[
                    "family_scheme"
                ]
                == scheme
            )
            & (
                task_catalog[
                    "eligible_for_holdout"
                ]
                == 1
            )
        ]

        eligible_summary[
            scheme
        ] = {
            "strict":
                int(
                    (
                        subset[
                            "analysis_tier"
                        ]
                        == "strict"
                    ).sum()
                ),

            "exploratory":
                int(
                    (
                        subset[
                            "analysis_tier"
                        ]
                        == "exploratory"
                    ).sum()
                ),
        }

    mapped_elements = {
        element
        for items in frame.loc[
            mapped,
            "catalyst_element_list",
        ]
        for element in items
    }

    summary = {
        "dataset":
            str(
                DATASET_CSV
            ),

        "raw_root":
            str(
                RAW_ROOT
            ),

        "training_levels":
            list(
                TRAIN_LEVELS
            ),

        "feature_count":
            len(
                feature_columns
            ),

        "composite_minimum_component_count":
            MIN_COMPONENT_COUNT,

        "component_count_summary":
            component_count_summary,

        "schemes":
            list(
                args.schemes
            ),

        "tiers":
            list(
                args.tiers
            ),

        "protocols":
            list(
                args.protocols
            ),

        "primary_protocol":
            PRIMARY_PROTOCOL,

        "figure_protocol":
            figure_protocol,

        "strict_threshold": {
            "minimum_test_samples":
                STRICT_MIN_TEST_SAMPLES,

            "minimum_test_papers":
                STRICT_MIN_TEST_PAPERS,
        },

        "exploratory_threshold": {
            "minimum_test_samples":
                EXPLORATORY_MIN_TEST_SAMPLES,

            "minimum_test_papers":
                EXPLORATORY_MIN_TEST_PAPERS,
        },

        "minimum_train_samples":
            MIN_TRAIN_SAMPLES,

        "minimum_train_papers":
            MIN_TRAIN_PAPERS,

        "top_fractions":
            list(
                TOP_FRACTIONS
            ),

        "bootstrap_repeats":
            args.bootstrap,

        "base_rows_after_composite_count_filter":
            int(
                len(frame)
            ),

        "mapped_family_rows":
            int(
                mapped.sum()
            ),

        "mapped_family_fraction":
            float(
                mapped.mean()
            ),

        "unique_exact_families":
            int(
                frame.loc[
                    mapped,
                    "catalyst_family_exact",
                ].nunique()
            ),

        "unique_backbone_families":
            int(
                frame.loc[
                    mapped,
                    "catalyst_family_backbone",
                ].nunique()
            ),

        "unique_elements":
            int(
                len(
                    mapped_elements
                )
            ),

        "eligible_targets_by_scheme":
            eligible_summary,

        "completed_tasks":
            completed_tasks,

        "successful_models":
            int(
                len(
                    results[
                        "metrics"
                    ]
                )
                * len(
                    QUANTILES
                )
            ),

        "prediction_rows":
            int(
                len(
                    results[
                        "predictions"
                    ]
                )
            ),

        "q90_definition":
            (
                "conditional 90th percentile; "
                "not a confidence bound, "
                "error bound, or risk"
            ),

        "backbone_definition":
            (
                "deterministic anchor-element "
                "plus catalyst-size grouping; "
                "anchor does not imply "
                "dominant composition"
            ),

        "holdout_interpretation": {
            "exact":
                (
                    "unseen exact catalyst "
                    "element combinations"
                ),

            "backbone":
                (
                    "unseen coarse catalyst "
                    "chemistry groups"
                ),

            "element_presence":
                (
                    "training excludes every "
                    "sample containing the "
                    "held-out element"
                ),
        },

        "important_outputs": {
            "family_counts":
                str(
                    OUT_DIR
                    / "catalyst_family_counts.csv"
                ),

            "metrics":
                str(
                    OUT_DIR
                    / "family_holdout_metrics.csv"
                ),

            "mode_stratified_metrics":
                str(
                    OUT_DIR
                    / "family_holdout_metrics_by_mode.csv"
                ),

            "bootstrap":
                str(
                    OUT_DIR
                    / "family_holdout_bootstrap_ci.csv"
                ),

            "figure":
                str(
                    FIGURE_DIR
                    / "fig_catalyst_family_holdout.png"
                ),
        },
    }

    write_json(
        summary,
        OUT_DIR
        / "family_holdout_summary.json",
    )

    print(
        "\n"
        + "=" * 104
    )

    print(
        "[CATALYST FAMILY HOLDOUT FINISHED]"
    )

    print(
        "L1+L2 before filter : "
        f"{component_count_summary['l1_l2_rows_before_component_filter']}"
    )

    print(
        "Composite >=1      : "
        f"{component_count_summary['composite_ge1_rows']}"
    )

    print(
        "Composite >=2      : "
        f"{component_count_summary['composite_ge2_rows']}"
    )

    print(
        "Removed count=1    : "
        f"{component_count_summary['single_component_rows_removed_by_ge2']}"
    )

    print(
        f"Mapped family rows  : "
        f"{int(mapped.sum())}"
    )

    print(
        "Exact families      : "
        f"{frame.loc[mapped, 'catalyst_family_exact'].nunique()}"
    )

    print(
        "Backbone families   : "
        f"{frame.loc[mapped, 'catalyst_family_backbone'].nunique()}"
    )

    print(
        f"Selected targets    : "
        f"{len(selected_catalog)}"
    )

    print(
        f"Completed tasks     : "
        f"{len(results['metrics'])}"
    )

    print(
        f"Quantile models     : "
        f"{len(results['metrics']) * len(QUANTILES)}"
    )

    print(
        f"Output              : "
        f"{OUT_DIR}"
    )

    print(
        "Main table          : "
        "family_holdout_metrics.csv"
    )

    print(
        "Mode analysis       : "
        "family_holdout_metrics_by_mode.csv"
    )

    print(
        "Bootstrap           : "
        "family_holdout_bootstrap_ci.csv"
    )

    print(
        "Main figure         : "
        "figures/fig_catalyst_family_holdout.png"
    )

    print(
        "=" * 104
    )


if __name__ == "__main__":
    main()