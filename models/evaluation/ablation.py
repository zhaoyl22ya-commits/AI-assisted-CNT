# -*- coding: utf-8 -*-
"""
ablation.py

CNT 3.0 — Article-ready retraining ablation and interpretation analysis

本代码执行真正的“重新训练式消融”，不是把特征替换为均值的预测扰动。

核心规则
--------
1. 只使用 paper_grouped_cv，作为文章中物理模块消融的主验证；
2. 严格复用 train_l1_targets.py 与 train_l1_composite.py 已保存的 split manifest；
3. 所有特征组合使用完全相同的外层训练论文与测试论文；
4. early stopping 仅使用外层训练集内部的 GroupShuffleSplit；
5. LightGBM 参数、目标变换和综合评分构建方式与现有 L1 代码保持一致；
6. 对 N/C/F/P、逐步组合、drop-one、physical-only、missing-only 重新训练；
7. 输出 fold 级、pooled OOF、ΔR²/ΔSpearman、逐样本预测和运行审计；
8. 整合已有 OOF TreeSHAP 结果，生成可直接用于 Figure 4 的图和 Source Data；
9. 支持断点续跑。长时间运行中断后，直接重新运行即可继续；
10. 不修改 L123、WorldMap、网页或主动学习模块。

默认输入
--------
<project_dir>/
├─ data/11_flattened/
│  ├─ ml_dataset_3p0_full60D.csv
│  └─ feature_schema.json
├─ outputs/models/l1_targets/
│  ├─ split_manifest.csv
│  └─ L1_predictions.csv
├─ outputs/models/l1_composite/
│  ├─ composite_split_manifest.csv
│  ├─ composite_normalization_manifest.csv
│  └─ L1_composite_predictions.csv
└─ outputs/explain/l1_shap/                    # 可选但建议存在
   ├─ oof_block_shap_summary.csv
   ├─ missingness_reliance.csv
   └─ oof_feature_shap_summary.csv

默认输出
--------
<project_dir>/outputs/external_validation/ablation/
├─ figures/
├─ source_data/
├─ tables/
├─ manifests/
├─ validation_models/                         # 仅 --save_models 时生成
├─ ablation_metrics_fold.csv
├─ ablation_metrics_summary.csv
├─ ablation_predictions.csv
├─ ablation_feature_sets.csv
├─ ablation_baseline_reproduction_audit.csv
├─ ablation_split_audit.csv
└─ ablation_run_summary.json

运行
----
正式完整运行：
    python ablation.py

中断后继续：
    python ablation.py

重新从头运行：
    python ablation.py --fresh

只检查少量目标：
    python ablation.py --targets y_growth_quality_score y_composite_final_score

保存全部消融模型：
    python ablation.py --save_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# =============================================================================
# 1. 固定配置
# =============================================================================

SEED = 42
INNER_VALID_SIZE = 0.15
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
DEFAULT_DPI = 600

ONLY_TRAIN_LEVEL = "L1"
GROUPED_SPLIT_NAME = "paper_grouped_cv"
EXPECTED_GROUP_FOLDS = 5

MIN_TRAIN_SAMPLES = 30
MIN_TEST_SAMPLES = 5
MIN_TRAIN_PAPERS = 3
MIN_TEST_PAPERS = 2

BASELINE_REPRODUCTION_TOLERANCE = 1e-6

TARGET_COMPOSITE = "y_composite_final_score"

MAIN_TARGET = "y_growth_quality_score"
PREFERRED_IGID_COL = "y_ig_id_ratio"
LEGACY_IGID_COL = "y_IG_ID"
UNIFIED_IGID_COL = "y_ig_id_ratio_unified"

LOG_TARGETS = {
    "y_length_mm",
    "y_bundle_density_per_um",
    "y_cnt_density_per_um",
}

TARGET_ORDER = [
    "y_growth_quality_score",
    "y_diameter_nm",
    "y_length_mm",
    "y_bundle_density_per_um",
    "y_cnt_density_per_um",
    "y_ig_id_ratio_unified",
    "y_orientation_score",
    "y_purity_score",
    "y_composite_final_score",
]

TARGET_LABELS = {
    "y_growth_quality_score": "Growth quality",
    "y_diameter_nm": "Diameter",
    "y_length_mm": "Length",
    "y_bundle_density_per_um": "Bundle density",
    "y_cnt_density_per_um": "CNT density",
    "y_ig_id_ratio_unified": r"$I_G/I_D$",
    "y_orientation_score": "Orientation",
    "y_purity_score": "Purity",
    "y_composite_final_score": "Composite",
}

SHORT_TARGET_MAP = {
    "y_growth_quality_score": "gq",
    "y_ig_id_ratio_unified": "igid",
    "y_orientation_score": "ori",
    "y_purity_score": "purity",
    "y_bundle_density_per_um": "bundle",
    "y_composite_final_score": "composite",
}

MODULES = ["N", "C", "F", "P"]

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

# 综合评分
WEIGHTS: Dict[str, float] = {
    "gq": 0.30,
    "igid": 0.20,
    "ori": 0.20,
    "purity": 0.15,
    "bundle": 0.15,
}

MISSING_POLICY = "renorm"

# 正式 Composite 定义：至少两个有效组成项
MIN_COMPONENT_COUNT = 2

COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"

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
# 2. 路径
# =============================================================================

@dataclass(frozen=True)
class Paths:
    project_dir: str
    dataset: str
    schema: str

    ordinary_dir: str
    ordinary_manifest: str
    ordinary_predictions: str

    composite_dir: str
    composite_manifest: str
    composite_normalization: str
    composite_predictions: str

    shap_dir: str
    shap_block_summary: str
    shap_missingness: str
    shap_feature_summary: str

    output_dir: str
    figures_dir: str
    source_data_dir: str
    tables_dir: str
    manifests_dir: str
    models_dir: str

    metrics_fold: str
    metrics_summary: str
    predictions: str
    feature_sets: str
    baseline_audit: str
    split_audit: str
    run_summary: str
    checkpoint_config: str


def build_paths(
    project_dir: str,
    output_dir: Optional[str],
) -> Paths:
    project_dir = os.path.abspath(project_dir)

    output_dir = (
        os.path.abspath(output_dir)
        if output_dir
        else os.path.join(
            project_dir,
            "outputs",
            "external_validation",
            "ablation",
        )
    )

    ordinary_dir = os.path.join(
        project_dir,
        "outputs",
        "models",
        "l1_targets",
    )

    composite_dir = os.path.join(
        project_dir,
        "outputs",
        "models",
        "l1_composite",
    )

    shap_dir = os.path.join(
        project_dir,
        "outputs",
        "explain",
        "l1_shap",
    )

    return Paths(
        project_dir=project_dir,

        dataset=os.path.join(
            project_dir,
            "data",
            "11_flattened",
            "ml_dataset_3p0_full60D.csv",
        ),

        schema=os.path.join(
            project_dir,
            "data",
            "11_flattened",
            "feature_schema.json",
        ),

        ordinary_dir=ordinary_dir,

        ordinary_manifest=os.path.join(
            ordinary_dir,
            "split_manifest.csv",
        ),

        ordinary_predictions=os.path.join(
            ordinary_dir,
            "L1_predictions.csv",
        ),

        composite_dir=composite_dir,

        composite_manifest=os.path.join(
            composite_dir,
            "composite_split_manifest.csv",
        ),

        composite_normalization=os.path.join(
            composite_dir,
            "composite_normalization_manifest.csv",
        ),

        composite_predictions=os.path.join(
            composite_dir,
            "L1_composite_predictions.csv",
        ),

        shap_dir=shap_dir,

        shap_block_summary=os.path.join(
            shap_dir,
            "oof_block_shap_summary.csv",
        ),

        shap_missingness=os.path.join(
            shap_dir,
            "missingness_reliance.csv",
        ),

        shap_feature_summary=os.path.join(
            shap_dir,
            "oof_feature_shap_summary.csv",
        ),

        output_dir=output_dir,

        figures_dir=os.path.join(
            output_dir,
            "figures",
        ),

        source_data_dir=os.path.join(
            output_dir,
            "source_data",
        ),

        tables_dir=os.path.join(
            output_dir,
            "tables",
        ),

        manifests_dir=os.path.join(
            output_dir,
            "manifests",
        ),

        models_dir=os.path.join(
            output_dir,
            "validation_models",
        ),

        metrics_fold=os.path.join(
            output_dir,
            "ablation_metrics_fold.csv",
        ),

        metrics_summary=os.path.join(
            output_dir,
            "ablation_metrics_summary.csv",
        ),

        predictions=os.path.join(
            output_dir,
            "ablation_predictions.csv",
        ),

        feature_sets=os.path.join(
            output_dir,
            "ablation_feature_sets.csv",
        ),

        baseline_audit=os.path.join(
            output_dir,
            "ablation_baseline_reproduction_audit.csv",
        ),

        split_audit=os.path.join(
            output_dir,
            "ablation_split_audit.csv",
        ),

        run_summary=os.path.join(
            output_dir,
            "ablation_run_summary.json",
        ),

        checkpoint_config=os.path.join(
            output_dir,
            "ablation_checkpoint_config.json",
        ),
    )


def ensure_required_inputs(
    paths: Paths,
) -> None:
    required = [
        paths.dataset,
        paths.schema,
        paths.ordinary_manifest,
        paths.ordinary_predictions,
        paths.composite_manifest,
        paths.composite_normalization,
        paths.composite_predictions,
    ]

    missing = [
        path
        for path in required
        if not os.path.isfile(path)
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要输入文件：\n"
            + "\n".join(missing)
        )


def create_output_dirs(
    paths: Paths,
    save_models: bool,
) -> None:
    directories = [
        paths.output_dir,
        paths.figures_dir,
        paths.source_data_dir,
        paths.tables_dir,
        paths.manifests_dir,
    ]

    if save_models:
        directories.append(
            paths.models_dir
        )

    for directory in directories:
        os.makedirs(
            directory,
            exist_ok=True,
        )


# =============================================================================
# 3. 文件与统计工具
# =============================================================================

def sha256_file(
    path: str,
) -> str:
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def atomic_write_csv(
    frame: pd.DataFrame,
    path: str,
) -> None:
    directory = os.path.dirname(
        path
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    file_descriptor, temp_path = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".csv",
        dir=directory,
    )

    os.close(
        file_descriptor
    )

    try:
        frame.to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
        )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if os.path.exists(
            temp_path
        ):
            os.remove(
                temp_path
            )


def atomic_write_json(
    data: Any,
    path: str,
) -> None:
    directory = os.path.dirname(
        path
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    file_descriptor, temp_path = tempfile.mkstemp(
        prefix="._tmp_",
        suffix=".json",
        dir=directory,
    )

    os.close(
        file_descriptor
    )

    try:
        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if os.path.exists(
            temp_path
        ):
            os.remove(
                temp_path
            )


def to_num(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if (
        len(y_true) < 2
        or np.allclose(
            y_true,
            y_true[0],
        )
    ):
        return float(
            "nan"
        )

    return float(
        r2_score(
            y_true,
            y_pred,
        )
    )


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
        np.isfinite(
            y_true
        )
        & np.isfinite(
            y_pred
        )
    )

    y_true = y_true[
        finite
    ]

    y_pred = y_pred[
        finite
    ]

    if len(y_true) < 2:
        return float(
            "nan"
        )

    true_rank = (
        pd.Series(
            y_true
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )

    pred_rank = (
        pd.Series(
            y_pred
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.std(
            true_rank
        ) == 0
        or np.std(
            pred_rank
        ) == 0
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            true_rank,
            pred_rank,
        )[0, 1]
    )


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    finite = (
        np.isfinite(
            y_true
        )
        & np.isfinite(
            y_pred
        )
    )

    y_true = y_true[
        finite
    ]

    y_pred = y_pred[
        finite
    ]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }

    return {
        "n": int(
            len(y_true)
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": safe_r2(
            y_true,
            y_pred,
        ),
        "spearman": safe_spearman(
            y_true,
            y_pred,
        ),
    }


def inverse_target_scale(
    values: np.ndarray,
    use_log: bool,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        np.expm1(
            values
        )
        if use_log
        else values
    )


def ordered_targets_present(
    values: Iterable[str],
) -> List[str]:
    present = set(
        str(value)
        for value in values
    )

    ordered = [
        target
        for target in TARGET_ORDER
        if target in present
    ]

    return (
        ordered
        + sorted(
            present.difference(
                ordered
            )
        )
    )


# =============================================================================
# 4. 特征定义
# =============================================================================

def load_feature_schema(
    path: str,
) -> List[str]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        schema = json.load(
            file
        )

    columns = schema.get(
        "model_feature_columns"
    )

    if (
        not isinstance(
            columns,
            list,
        )
        or not columns
        or not all(
            isinstance(
                column,
                str,
            )
            for column in columns
        )
    ):
        raise RuntimeError(
            "feature_schema.json 缺少有效 model_feature_columns。"
        )

    if len(columns) != len(
        set(columns)
    ):
        raise RuntimeError(
            "feature_schema.json 中存在重复特征名。"
        )

    return columns


def feature_module(
    feature_name: str,
) -> str:
    name = str(
        feature_name
    )

    if name.startswith(
        "miss_"
    ):
        name = name[5:]

    if (
        len(name) >= 2
        and name[0] in MODULES
        and name[1:].isdigit()
    ):
        return name[0]

    return "other"


def is_missing_feature(
    feature_name: str,
) -> bool:
    return str(
        feature_name
    ).startswith(
        "miss_"
    )


def build_feature_sets(
    all_features: Sequence[str],
) -> Dict[str, List[str]]:
    all_features = list(
        all_features
    )

    physical_by_module: Dict[
        str,
        List[str],
    ] = {
        module: [
            feature
            for feature in all_features
            if (
                feature_module(
                    feature
                ) == module
                and not is_missing_feature(
                    feature
                )
            )
        ]
        for module in MODULES
    }

    missing_by_module: Dict[
        str,
        List[str],
    ] = {
        module: [
            feature
            for feature in all_features
            if (
                feature_module(
                    feature
                ) == module
                and is_missing_feature(
                    feature
                )
            )
        ]
        for module in MODULES
    }

    def modules_with_missing(
        modules: Sequence[str],
    ) -> List[str]:
        selected: List[str] = []

        for feature in all_features:
            module = feature_module(
                feature
            )

            if module in modules:
                selected.append(
                    feature
                )

        return selected

    physical_all = [
        feature
        for feature in all_features
        if not is_missing_feature(
            feature
        )
    ]

    missing_all = [
        feature
        for feature in all_features
        if is_missing_feature(
            feature
        )
    ]

    feature_sets: Dict[
        str,
        List[str],
    ] = {
        "full":
            list(
                all_features
            ),

        "physical_only":
            physical_all,

        "missing_only":
            missing_all,

        "N_only":
            modules_with_missing(
                ["N"]
            ),

        "C_only":
            modules_with_missing(
                ["C"]
            ),

        "F_only":
            modules_with_missing(
                ["F"]
            ),

        "P_only":
            modules_with_missing(
                ["P"]
            ),

        "N_C":
            modules_with_missing(
                [
                    "N",
                    "C",
                ]
            ),

        "N_C_F":
            modules_with_missing(
                [
                    "N",
                    "C",
                    "F",
                ]
            ),

        "drop_N":
            modules_with_missing(
                [
                    "C",
                    "F",
                    "P",
                ]
            ),

        "drop_C":
            modules_with_missing(
                [
                    "N",
                    "F",
                    "P",
                ]
            ),

        "drop_F":
            modules_with_missing(
                [
                    "N",
                    "C",
                    "P",
                ]
            ),

        "drop_P":
            modules_with_missing(
                [
                    "N",
                    "C",
                    "F",
                ]
            ),
    }

    for (
        name,
        features,
    ) in feature_sets.items():
        if not features:
            raise RuntimeError(
                f"特征组合 {name} 为空。"
            )

        if len(
            features
        ) != len(
            set(
                features
            )
        ):
            raise RuntimeError(
                f"特征组合 {name} 存在重复列。"
            )

    # 确保四个物理模块均被 schema 识别
    for module in MODULES:
        if not physical_by_module[
            module
        ]:
            raise RuntimeError(
                f"schema中没有识别到 {module} 物理特征。"
            )

        if not missing_by_module[
            module
        ]:
            raise RuntimeError(
                f"schema中没有识别到 miss_{module} 特征。"
            )

    return feature_sets


def feature_set_manifest(
    feature_sets: Mapping[
        str,
        Sequence[str],
    ],
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    descriptions = {
        "full":
            "N+C+F+P physical descriptors and all miss_* indicators",

        "physical_only":
            "All 60 physical descriptors without miss_* indicators",

        "missing_only":
            "All miss_* indicators without physical descriptors",

        "N_only":
            "N physical descriptors and corresponding miss_N* indicators",

        "C_only":
            "C physical descriptors and corresponding miss_C* indicators",

        "F_only":
            "F physical descriptors and corresponding miss_F* indicators",

        "P_only":
            "P physical descriptors and corresponding miss_P* indicators",

        "N_C":
            "N+C descriptors and corresponding missingness indicators",

        "N_C_F":
            "N+C+F descriptors and corresponding missingness indicators",

        "drop_N":
            "C+F+P descriptors and corresponding missingness indicators",

        "drop_C":
            "N+F+P descriptors and corresponding missingness indicators",

        "drop_F":
            "N+C+P descriptors and corresponding missingness indicators",

        "drop_P":
            "N+C+F descriptors and corresponding missingness indicators",
    }

    for (
        feature_set,
        features,
    ) in feature_sets.items():
        modules = sorted(
            {
                feature_module(
                    feature
                )
                for feature in features
                if feature_module(
                    feature
                ) in MODULES
            }
        )

        rows.append(
            {
                "feature_set":
                    feature_set,

                "description":
                    descriptions.get(
                        feature_set,
                        "",
                    ),

                "feature_count":
                    int(
                        len(features)
                    ),

                "physical_feature_count":
                    int(
                        sum(
                            not is_missing_feature(
                                feature
                            )
                            for feature in features
                        )
                    ),

                "missing_indicator_count":
                    int(
                        sum(
                            is_missing_feature(
                                feature
                            )
                            for feature in features
                        )
                    ),

                "modules":
                    "+".join(
                        modules
                    ),

                "feature_columns_json":
                    json.dumps(
                        list(
                            features
                        ),
                        ensure_ascii=False,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 5. 数据读取
# =============================================================================

def ensure_igid_unified(
    df: pd.DataFrame,
) -> pd.DataFrame:
    preferred = (
        to_num(
            df[
                PREFERRED_IGID_COL
            ]
        )
        if PREFERRED_IGID_COL
        in df.columns
        else pd.Series(
            np.nan,
            index=df.index,
        )
    )

    legacy = (
        to_num(
            df[
                LEGACY_IGID_COL
            ]
        )
        if LEGACY_IGID_COL
        in df.columns
        else pd.Series(
            np.nan,
            index=df.index,
        )
    )

    out = df.copy()

    out[
        UNIFIED_IGID_COL
    ] = preferred.where(
        preferred.notna(),
        legacy,
    )

    return out


def load_dataset(
    path: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if "data_level" not in df.columns:
        raise RuntimeError(
            "数据集缺少 data_level。"
        )

    df = df[
        df[
            "data_level"
        ].astype(
            str
        ) == ONLY_TRAIN_LEVEL
    ].copy()

    df = ensure_igid_unified(
        df
    )

    required_metadata = [
        "sample_id",
        "paper_id",
        "paper_year_int",
        "data_level",
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in df.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            f"数据集缺少必要元数据：{missing_metadata}"
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "数据集缺少schema特征，示例："
            f"{missing_features[:10]}"
        )

    defaults = {
        "growth_mode": "",
        "source_file": "",
        "synthesis_id": "",
    }

    for (
        column,
        default,
    ) in defaults.items():
        if column not in df.columns:
            df[
                column
            ] = default

    df[
        "sample_id"
    ] = (
        df[
            "sample_id"
        ]
        .fillna("")
        .astype(str)
    )

    df[
        "paper_id"
    ] = (
        df[
            "paper_id"
        ]
        .fillna("")
        .astype(str)
    )

    df[
        "paper_year_int"
    ] = pd.to_numeric(
        df[
            "paper_year_int"
        ],
        errors="coerce",
    )

    blank_ids = (
        df[
            "sample_id"
        ].str.len()
        == 0
    )

    if blank_ids.any():
        raise RuntimeError(
            "L1数据存在 "
            f"{int(blank_ids.sum())} 条空 sample_id。"
        )

    duplicates = df[
        "sample_id"
    ].duplicated(
        keep=False
    )

    if duplicates.any():
        examples = (
            df.loc[
                duplicates,
                "sample_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "L1数据 sample_id 不唯一，"
            f"示例：{examples}"
        )

    for column in feature_columns:
        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    return df.reset_index(
        drop=True
    )


def load_manifest(
    path: str,
    source_model: str,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "sample_id",
        "paper_id",
        "target",
        "split_type",
        "fold",
        "role",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source_model} manifest缺少：{missing}"
        )

    frame = frame.copy()

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
        "target"
    ] = frame[
        "target"
    ].astype(
        str
    )

    frame[
        "split_type"
    ] = frame[
        "split_type"
    ].astype(
        str
    )

    frame[
        "role"
    ] = frame[
        "role"
    ].astype(
        str
    )

    frame[
        "fold"
    ] = pd.to_numeric(
        frame[
            "fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    frame[
        "source_model"
    ] = source_model

    return frame[
        frame[
            "split_type"
        ] == GROUPED_SPLIT_NAME
    ].copy()


def load_saved_predictions(
    ordinary_path: str,
    composite_path: str,
) -> pd.DataFrame:
    frames: List[
        pd.DataFrame
    ] = []

    for (
        path,
        source_model,
    ) in [
        (
            ordinary_path,
            "L1_multi_target",
        ),
        (
            composite_path,
            "L1_composite",
        ),
    ]:
        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        required = [
            "sample_id",
            "paper_id",
            "target",
            "split_type",
            "fold",
            "y_true",
            "y_pred",
        ]

        missing = [
            column
            for column in required
            if column not in frame.columns
        ]

        if missing:
            raise RuntimeError(
                f"{source_model} predictions缺少：{missing}"
            )

        frame = frame[
            frame[
                "split_type"
            ] == GROUPED_SPLIT_NAME
        ].copy()

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
            "target"
        ] = frame[
            "target"
        ].astype(
            str
        )

        frame[
            "fold"
        ] = pd.to_numeric(
            frame[
                "fold"
            ],
            errors="raise",
        ).astype(
            int
        )

        frame[
            "y_true"
        ] = pd.to_numeric(
            frame[
                "y_true"
            ],
            errors="coerce",
        )

        frame[
            "y_pred"
        ] = pd.to_numeric(
            frame[
                "y_pred"
            ],
            errors="coerce",
        )

        frame[
            "source_model"
        ] = source_model

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


# =============================================================================
# 6. 普通目标准备
# =============================================================================

@dataclass
class TargetData:
    target: str
    source_model: str
    work: pd.DataFrame
    X_all: pd.DataFrame
    y_model: np.ndarray
    y_original: np.ndarray
    use_log: bool


def prepare_ordinary_target(
    dataset: pd.DataFrame,
    all_features: Sequence[str],
    target: str,
) -> TargetData:
    if target not in dataset.columns:
        raise RuntimeError(
            f"数据集不存在目标：{target}"
        )

    y_raw = pd.to_numeric(
        dataset[
            target
        ],
        errors="coerce",
    )

    valid = y_raw.notna()

    work = dataset.loc[
        valid
    ].copy()

    y_original = y_raw.loc[
        valid
    ].astype(
        float
    )

    use_log = (
        target
        in LOG_TARGETS
    )

    if use_log:
        positive = (
            y_original
            > 0
        )

        work = work.loc[
            positive
        ].copy()

        y_original = y_original.loc[
            positive
        ]

        y_model = np.log1p(
            y_original.to_numpy(
                dtype=float
            )
        )

    else:
        y_model = y_original.to_numpy(
            dtype=float
        )

    work = work.reset_index(
        drop=True
    )

    y_original_array = (
        y_original
        .reset_index(
            drop=True
        )
        .to_numpy(
            dtype=float
        )
    )

    X_all = (
        work[
            list(
                all_features
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return TargetData(
        target=target,
        source_model="L1_multi_target",
        work=work,
        X_all=X_all,
        y_model=np.asarray(
            y_model,
            dtype=float,
        ),
        y_original=y_original_array,
        use_log=use_log,
    )


# =============================================================================
# 7. 综合目标准备
# =============================================================================

def minmax_parameters(
    series: pd.Series,
) -> Dict[str, Any]:
    values = to_num(
        series
    ).to_numpy(
        dtype=float
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if len(finite) == 0:
        return {
            "minimum": None,
            "maximum": None,
            "usable": False,
        }

    minimum = float(
        np.min(
            finite
        )
    )

    maximum = float(
        np.max(
            finite
        )
    )

    if math.isclose(
        minimum,
        maximum,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        return {
            "minimum": minimum,
            "maximum": maximum,
            "usable": False,
        }

    return {
        "minimum": minimum,
        "maximum": maximum,
        "usable": True,
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
        or math.isclose(
            minimum,
            maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    result = (
        to_num(
            series
        )
        - minimum
    ) / (
        maximum
        - minimum
    )

    return result.clip(
        lower=0.0,
        upper=1.0,
    )


def prepare_composite_base(
    dataset: pd.DataFrame,
    all_features: Sequence[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    work = ensure_igid_unified(
        dataset
    ).copy()

    for column in [
        COL_GQ,
        UNIFIED_IGID_COL,
        COL_ORI,
        COL_PURITY,
        COL_BUNDLE,
    ]:
        if column not in work.columns:
            work[
                column
            ] = np.nan

    work[
        RAW_GQ
    ] = to_num(
        work[
            COL_GQ
        ]
    )

    work[
        RAW_IGID
    ] = to_num(
        work[
            UNIFIED_IGID_COL
        ]
    )

    work[
        RAW_ORI
    ] = to_num(
        work[
            COL_ORI
        ]
    )

    work[
        RAW_PURITY
    ] = to_num(
        work[
            COL_PURITY
        ]
    )

    work[
        RAW_BUNDLE
    ] = to_num(
        work[
            COL_BUNDLE
        ]
    )

    bundle = work[
        RAW_BUNDLE
    ].to_numpy(
        dtype=float
    )

    work[
        RAW_BUNDLE_LOG
    ] = np.where(
        np.isfinite(
            bundle
        )
        & (
            bundle >= 0.0
        ),
        np.log1p(
            bundle
        ),
        np.nan,
    )

    availability = pd.DataFrame(
        {
            "gq":
                work[
                    RAW_GQ
                ].notna(),

            "igid":
                work[
                    RAW_IGID
                ].notna(),

            "ori":
                work[
                    RAW_ORI
                ].notna(),

            "purity":
                work[
                    RAW_PURITY
                ].notna(),

            "bundle":
                work[
                    RAW_BUNDLE_LOG
                ].notna(),
        }
    )

    work[
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

    work = work[
        work[
            RAW_COMPONENT_COUNT
        ]
        >= MIN_COMPONENT_COUNT
    ].copy()

    work = work.reset_index(
        drop=True
    )

    X_all = (
        work[
            list(
                all_features
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        work,
        X_all,
    )


def parse_bool(
    value: Any,
    default: bool = True,
) -> bool:
    if pd.isna(
        value
    ):
        return default

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return bool(
            int(
                value
            )
        )

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    return default


def normalizer_from_manifest_row(
    row: pd.Series,
) -> Dict[str, Any]:
    return {
        "igid_min":
            (
                float(
                    row[
                        "igid_min"
                    ]
                )
                if pd.notna(
                    row.get(
                        "igid_min"
                    )
                )
                else None
            ),

        "igid_max":
            (
                float(
                    row[
                        "igid_max"
                    ]
                )
                if pd.notna(
                    row.get(
                        "igid_max"
                    )
                )
                else None
            ),

        "igid_usable":
            parse_bool(
                row.get(
                    "igid_usable",
                    True,
                )
            ),

        "bundle_log_min":
            (
                float(
                    row[
                        "bundle_log_min"
                    ]
                )
                if pd.notna(
                    row.get(
                        "bundle_log_min"
                    )
                )
                else None
            ),

        "bundle_log_max":
            (
                float(
                    row[
                        "bundle_log_max"
                    ]
                )
                if pd.notna(
                    row.get(
                        "bundle_log_max"
                    )
                )
                else None
            ),

        "bundle_usable":
            parse_bool(
                row.get(
                    "bundle_usable",
                    True,
                )
            ),
    }


def transform_composite(
    work: pd.DataFrame,
    normalizer: Mapping[str, Any],
) -> pd.DataFrame:
    out = pd.DataFrame(
        index=work.index
    )

    out[
        "gq_01"
    ] = (
        to_num(
            work[
                RAW_GQ
            ]
        )
        / 10.0
    ).clip(
        0.0,
        1.0,
    )

    out[
        "igid_norm01"
    ] = apply_minmax(
        work[
            RAW_IGID
        ],
        normalizer.get(
            "igid_min"
        ),
        normalizer.get(
            "igid_max"
        ),
        bool(
            normalizer.get(
                "igid_usable"
            )
        ),
    )

    out[
        "ori_01"
    ] = to_num(
        work[
            RAW_ORI
        ]
    ).clip(
        0.0,
        1.0,
    )

    out[
        "purity_01"
    ] = to_num(
        work[
            RAW_PURITY
        ]
    ).clip(
        0.0,
        1.0,
    )

    out[
        "bundle_norm01"
    ] = apply_minmax(
        work[
            RAW_BUNDLE_LOG
        ],
        normalizer.get(
            "bundle_log_min"
        ),
        normalizer.get(
            "bundle_log_max"
        ),
        bool(
            normalizer.get(
                "bundle_usable"
            )
        ),
    )

    components = pd.DataFrame(
        {
            component:
                out[
                    column
                ]
            for (
                component,
                column,
            ) in COMPONENT_COLUMNS.items()
        },
        index=work.index,
    )

    weights = pd.Series(
        WEIGHTS,
        dtype=float,
    )

    component_count = (
        components
        .notna()
        .sum(
            axis=1
        )
        .astype(
            int
        )
    )

    if MISSING_POLICY == "renorm":
        availability = (
            components
            .notna()
        )

        weight_sum = (
            availability
            .mul(
                weights,
                axis=1,
            )
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

        composite = pd.Series(
            np.where(
                weight_sum > 0,
                numerator
                / weight_sum,
                np.nan,
            ),
            index=work.index,
            dtype=float,
        )

    elif MISSING_POLICY == "strict":
        complete = (
            components
            .notna()
            .all(
                axis=1
            )
        )

        composite = pd.Series(
            np.nan,
            index=work.index,
            dtype=float,
        )

        composite.loc[
            complete
        ] = (
            components.loc[
                complete
            ]
            .mul(
                weights,
                axis=1,
            )
            .sum(
                axis=1
            )
        )

        weight_sum = pd.Series(
            np.nan,
            index=work.index,
            dtype=float,
        )

        weight_sum.loc[
            complete
        ] = float(
            weights.sum()
        )

    else:
        raise ValueError(
            "MISSING_POLICY必须为renorm或strict。"
        )

    composite = composite.where(
        component_count
        >= MIN_COMPONENT_COUNT,
        np.nan,
    )

    out[
        TARGET_COMPOSITE
    ] = composite

    out[
        "composite_component_count"
    ] = component_count

    out[
        "composite_weight_sum_used"
    ] = weight_sum.where(
        composite.notna(),
        np.nan,
    )

    return out


# =============================================================================
# 8. 固定外层 fold 审计
# =============================================================================

def target_manifest(
    ordinary_manifest: pd.DataFrame,
    composite_manifest: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    source = (
        composite_manifest
        if target
        == TARGET_COMPOSITE
        else ordinary_manifest
    )

    result = source[
        (
            source[
                "target"
            ] == target
        )
        & (
            source[
                "split_type"
            ] == GROUPED_SPLIT_NAME
        )
    ].copy()

    if result.empty:
        raise RuntimeError(
            f"manifest中找不到目标：{target}"
        )

    return result


def audit_target_folds(
    manifest: pd.DataFrame,
    dataset_sample_ids: Set[str],
    target: str,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    rows: List[
        Dict[str, Any]
    ] = []

    folds = sorted(
        manifest[
            "fold"
        ].unique().tolist()
    )

    if len(
        folds
    ) != EXPECTED_GROUP_FOLDS:
        raise RuntimeError(
            f"{target} fold数量为{len(folds)}，"
            f"预期{EXPECTED_GROUP_FOLDS}。"
        )

    all_test_ids: List[str] = []

    for fold in folds:
        fold_frame = manifest[
            manifest[
                "fold"
            ] == fold
        ]

        train_frame = fold_frame[
            fold_frame[
                "role"
            ] == "train"
        ]

        test_frame = fold_frame[
            fold_frame[
                "role"
            ] == "test"
        ]

        train_ids = (
            train_frame[
                "sample_id"
            ]
            .astype(str)
            .tolist()
        )

        test_ids = (
            test_frame[
                "sample_id"
            ]
            .astype(str)
            .tolist()
        )

        missing_ids = [
            sample_id
            for sample_id
            in (
                train_ids
                + test_ids
            )
            if sample_id
            not in dataset_sample_ids
        ]

        if missing_ids:
            raise RuntimeError(
                f"{target}/fold={fold} "
                "有sample_id不在数据集中，"
                f"示例：{missing_ids[:10]}"
            )

        train_papers = set(
            train_frame[
                "paper_id"
            ].astype(str)
        )

        test_papers = set(
            test_frame[
                "paper_id"
            ].astype(str)
        )

        overlap = (
            train_papers
            .intersection(
                test_papers
            )
        )

        if overlap:
            raise RuntimeError(
                f"{target}/fold={fold} "
                f"存在论文泄漏：{len(overlap)}篇。"
            )

        if set(
            train_ids
        ).intersection(
            test_ids
        ):
            raise RuntimeError(
                f"{target}/fold={fold} "
                "存在训练—测试sample_id重叠。"
            )

        all_test_ids.extend(
            test_ids
        )

        rows.append(
            {
                "target":
                    target,

                "fold":
                    int(
                        fold
                    ),

                "n_train":
                    int(
                        len(
                            train_ids
                        )
                    ),

                "n_test":
                    int(
                        len(
                            test_ids
                        )
                    ),

                "n_train_papers":
                    int(
                        len(
                            train_papers
                        )
                    ),

                "n_test_papers":
                    int(
                        len(
                            test_papers
                        )
                    ),

                "paper_overlap":
                    0,

                "sample_overlap":
                    0,

                "status":
                    "passed",
            }
        )

    if len(
        all_test_ids
    ) != len(
        set(
            all_test_ids
        )
    ):
        raise RuntimeError(
            f"{target} OOF测试样本重复出现。"
        )

    return rows


# =============================================================================
# 9. 内部验证与训练
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

    relative_indices = np.arange(
        len(
            outer_train_indices
        )
    )

    (
        relative_fit,
        relative_valid,
    ) = train_test_split(
        relative_indices,
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


def train_model(
    X: pd.DataFrame,
    y_model: np.ndarray,
    train_indices: np.ndarray,
    groups_all: pd.Series,
) -> Tuple[
    lgb.Booster,
    int,
    str,
]:
    (
        fit_indices,
        valid_indices,
        method,
    ) = make_inner_split(
        train_indices,
        groups_all,
    )

    train_set = lgb.Dataset(
        X.iloc[
            fit_indices
        ],
        label=y_model[
            fit_indices
        ],
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X.iloc[
            valid_indices
        ],
        label=y_model[
            valid_indices
        ],
        reference=train_set,
        free_raw_data=False,
    )

    booster = lgb.train(
        params=LGB_PARAMS,
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
        booster.best_iteration
        if (
            booster.best_iteration
            and booster.best_iteration > 0
        )
        else MAX_BOOST_ROUNDS
    )

    return (
        booster,
        best_iteration,
        method,
    )


def save_model(
    booster: lgb.Booster,
    paths: Paths,
    target: str,
    feature_set: str,
    fold: int,
) -> str:
    target_dir = os.path.join(
        paths.models_dir,
        target,
        feature_set,
    )

    os.makedirs(
        target_dir,
        exist_ok=True,
    )

    model_path = os.path.join(
        target_dir,
        f"fold{fold}.txt",
    )

    booster.save_model(
        model_path
    )

    return model_path


# =============================================================================
# 10. Fold执行
# =============================================================================

def indices_from_sample_ids(
    work: pd.DataFrame,
    sample_ids: Sequence[str],
) -> np.ndarray:
    requested = set(
        str(
            sample_id
        )
        for sample_id
        in sample_ids
    )

    indices = np.where(
        work[
            "sample_id"
        ]
        .astype(str)
        .isin(
            requested
        )
        .to_numpy()
    )[0]

    found = set(
        work.iloc[
            indices
        ][
            "sample_id"
        ].astype(str)
    )

    if found != requested:
        missing = sorted(
            requested.difference(
                found
            )
        )

        raise RuntimeError(
            "无法在目标数据中定位全部sample_id，"
            f"示例：{missing[:10]}"
        )

    return indices


def normalization_for_fold(
    normalization_manifest: pd.DataFrame,
    fold: int,
) -> Dict[str, Any]:
    required = normalization_manifest[
        (
            normalization_manifest[
                "split_type"
            ] == GROUPED_SPLIT_NAME
        )
        & (
            normalization_manifest[
                "fold"
            ] == fold
        )
    ]

    if len(
        required
    ) != 1:
        raise RuntimeError(
            f"综合评分fold={fold}"
            "归一化记录数量为"
            f"{len(required)}，预期1。"
        )

    return normalizer_from_manifest_row(
        required.iloc[
            0
        ]
    )


def existing_prediction_subset(
    saved_predictions: pd.DataFrame,
    target: str,
    fold: int,
) -> pd.DataFrame:
    return saved_predictions[
        (
            saved_predictions[
                "target"
            ] == target
        )
        & (
            saved_predictions[
                "split_type"
            ] == GROUPED_SPLIT_NAME
        )
        & (
            saved_predictions[
                "fold"
            ] == fold
        )
    ].copy()


def prediction_rows(
    target: str,
    feature_set: str,
    fold: int,
    work: pd.DataFrame,
    test_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_iteration: int,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    rows: List[
        Dict[str, Any]
    ] = []

    for (
        position,
        work_index,
    ) in enumerate(
        test_indices
    ):
        item = work.iloc[
            int(
                work_index
            )
        ]

        true_value = float(
            y_true[
                position
            ]
        )

        pred_value = float(
            y_pred[
                position
            ]
        )

        rows.append(
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

                "data_level":
                    item[
                        "data_level"
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
                    target,

                "split_type":
                    GROUPED_SPLIT_NAME,

                "fold":
                    int(
                        fold
                    ),

                "feature_set":
                    feature_set,

                "best_iteration":
                    int(
                        best_iteration
                    ),

                "y_true":
                    true_value,

                "y_pred":
                    pred_value,

                "residual":
                    pred_value
                    - true_value,

                "absolute_error":
                    abs(
                        pred_value
                        - true_value
                    ),
            }
        )

    return rows


def run_fold_model(
    target_data: TargetData,
    feature_set_name: str,
    feature_columns: Sequence[str],
    fold_manifest: pd.DataFrame,
    fold: int,
    paths: Paths,
    save_models: bool,
    normalizer: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> Tuple[
    Dict[
        str,
        Any,
    ],
    List[
        Dict[
            str,
            Any,
        ]
    ],
    Dict[
        str,
        Any,
    ],
]:
    work = target_data.work

    X = target_data.X_all[
        list(
            feature_columns
        )
    ].copy()

    train_ids = (
        fold_manifest.loc[
            fold_manifest[
                "role"
            ] == "train",
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    test_ids = (
        fold_manifest.loc[
            fold_manifest[
                "role"
            ] == "test",
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    train_indices = indices_from_sample_ids(
        work,
        train_ids,
    )

    test_indices = indices_from_sample_ids(
        work,
        test_ids,
    )

    if target_data.target == TARGET_COMPOSITE:
        if normalizer is None:
            raise RuntimeError(
                "综合目标缺少fold归一化参数。"
            )

        transformed = transform_composite(
            work,
            normalizer,
        )

        y_model = pd.to_numeric(
            transformed[
                TARGET_COMPOSITE
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        y_original = y_model.copy()

        train_indices = train_indices[
            np.isfinite(
                y_model[
                    train_indices
                ]
            )
        ]

        test_indices = test_indices[
            np.isfinite(
                y_model[
                    test_indices
                ]
            )
        ]

    else:
        y_model = (
            target_data
            .y_model
        )

        y_original = (
            target_data
            .y_original
        )

    if len(
        train_indices
    ) < MIN_TRAIN_SAMPLES:
        raise RuntimeError(
            f"{target_data.target}/"
            f"{feature_set_name}/"
            f"fold={fold} "
            "训练样本过少："
            f"{len(train_indices)}"
        )

    if len(
        test_indices
    ) < MIN_TEST_SAMPLES:
        raise RuntimeError(
            f"{target_data.target}/"
            f"{feature_set_name}/"
            f"fold={fold} "
            "测试样本过少："
            f"{len(test_indices)}"
        )

    n_train_papers = int(
        work.iloc[
            train_indices
        ][
            "paper_id"
        ].nunique()
    )

    n_test_papers = int(
        work.iloc[
            test_indices
        ][
            "paper_id"
        ].nunique()
    )

    if (
        n_train_papers
        < MIN_TRAIN_PAPERS
    ):
        raise RuntimeError(
            "训练论文数过少。"
        )

    if (
        n_test_papers
        < MIN_TEST_PAPERS
    ):
        raise RuntimeError(
            "测试论文数过少。"
        )

    (
        booster,
        best_iteration,
        inner_method,
    ) = train_model(
        X=X,
        y_model=y_model,
        train_indices=train_indices,
        groups_all=work[
            "paper_id"
        ],
    )

    pred_model = np.asarray(
        booster.predict(
            X.iloc[
                test_indices
            ],
            num_iteration=best_iteration,
        ),
        dtype=float,
    )

    y_true = inverse_target_scale(
        y_model[
            test_indices
        ],
        target_data.use_log,
    )

    y_pred = inverse_target_scale(
        pred_model,
        target_data.use_log,
    )

    metrics = evaluate_regression(
        y_true,
        y_pred,
    )

    model_path = ""

    if save_models:
        model_path = save_model(
            booster=booster,
            paths=paths,
            target=target_data.target,
            feature_set=feature_set_name,
            fold=fold,
        )

    metric_row = {
        "target":
            target_data.target,

        "target_label":
            TARGET_LABELS.get(
                target_data.target,
                target_data.target,
            ),

        "split_type":
            GROUPED_SPLIT_NAME,

        "fold":
            int(
                fold
            ),

        "feature_set":
            feature_set_name,

        "feature_count":
            int(
                len(
                    feature_columns
                )
            ),

        "status":
            "ok",

        "n_train":
            int(
                len(
                    train_indices
                )
            ),

        "n_test":
            int(
                len(
                    test_indices
                )
            ),

        "n_train_papers":
            n_train_papers,

        "n_test_papers":
            n_test_papers,

        "inner_split_method":
            inner_method,

        "best_iteration":
            int(
                best_iteration
            ),

        "mae":
            metrics[
                "mae"
            ],

        "rmse":
            metrics[
                "rmse"
            ],

        "r2":
            metrics[
                "r2"
            ],

        "spearman":
            metrics[
                "spearman"
            ],

        "model_path":
            model_path,
    }

    predictions = prediction_rows(
        target=target_data.target,
        feature_set=feature_set_name,
        fold=fold,
        work=work,
        test_indices=test_indices,
        y_true=y_true,
        y_pred=y_pred,
        best_iteration=best_iteration,
    )

    auxiliary = {
        "test_sample_ids":
            work.iloc[
                test_indices
            ][
                "sample_id"
            ]
            .astype(str)
            .tolist(),

        "y_true":
            y_true,

        "y_pred":
            y_pred,
    }

    return (
        metric_row,
        predictions,
        auxiliary,
    )


# =============================================================================
# 11. Full模型复现审计
# =============================================================================

def baseline_reproduction_row(
    saved_predictions: pd.DataFrame,
    target: str,
    fold: int,
    current_sample_ids: Sequence[str],
    current_y_true: np.ndarray,
    current_y_pred: np.ndarray,
) -> Dict[str, Any]:
    saved = existing_prediction_subset(
        saved_predictions=saved_predictions,
        target=target,
        fold=fold,
    )

    saved = saved.set_index(
        "sample_id",
        drop=False,
    )

    current = pd.DataFrame(
        {
            "sample_id":
                list(
                    current_sample_ids
                ),

            "y_true_current":
                np.asarray(
                    current_y_true,
                    dtype=float,
                ),

            "y_pred_current":
                np.asarray(
                    current_y_pred,
                    dtype=float,
                ),
        }
    ).set_index(
        "sample_id",
        drop=False,
    )

    saved_ids = set(
        saved.index.astype(
            str
        )
    )

    current_ids = set(
        current.index.astype(
            str
        )
    )

    ids_match = (
        saved_ids
        == current_ids
    )

    if ids_match:
        aligned_saved = saved.loc[
            current.index
        ]

        true_difference = np.abs(
            current[
                "y_true_current"
            ].to_numpy(
                dtype=float
            )
            - aligned_saved[
                "y_true"
            ].to_numpy(
                dtype=float
            )
        )

        prediction_difference = np.abs(
            current[
                "y_pred_current"
            ].to_numpy(
                dtype=float
            )
            - aligned_saved[
                "y_pred"
            ].to_numpy(
                dtype=float
            )
        )

        max_true_difference = float(
            np.nanmax(
                true_difference
            )
        )

        max_prediction_difference = float(
            np.nanmax(
                prediction_difference
            )
        )

        mean_prediction_difference = float(
            np.nanmean(
                prediction_difference
            )
        )

    else:
        max_true_difference = np.nan
        max_prediction_difference = np.nan
        mean_prediction_difference = np.nan

    status = "exact"

    if not ids_match:
        status = (
            "sample_ids_mismatch"
        )

    elif (
        max_true_difference
        > BASELINE_REPRODUCTION_TOLERANCE
        or max_prediction_difference
        > BASELINE_REPRODUCTION_TOLERANCE
    ):
        status = (
            "not_exact_check_versions_or_order"
        )

    return {
        "target":
            target,

        "fold":
            int(
                fold
            ),

        "saved_n":
            int(
                len(
                    saved
                )
            ),

        "current_n":
            int(
                len(
                    current
                )
            ),

        "sample_ids_match":
            bool(
                ids_match
            ),

        "max_y_true_difference":
            max_true_difference,

        "max_y_pred_difference":
            max_prediction_difference,

        "mean_y_pred_difference":
            mean_prediction_difference,

        "tolerance":
            BASELINE_REPRODUCTION_TOLERANCE,

        "status":
            status,
    }


# =============================================================================
# 12. 断点续跑
# =============================================================================

def configuration_payload(
    paths: Paths,
    selected_targets: Sequence[str],
    selected_feature_sets: Sequence[str],
    feature_sets: Mapping[
        str,
        Sequence[str],
    ],
) -> Dict[str, Any]:
    return {
        "dataset_path":
            paths.dataset,

        "dataset_sha256":
            sha256_file(
                paths.dataset
            ),

        "schema_path":
            paths.schema,

        "schema_sha256":
            sha256_file(
                paths.schema
            ),

        "ordinary_manifest_sha256":
            sha256_file(
                paths.ordinary_manifest
            ),

        "composite_manifest_sha256":
            sha256_file(
                paths.composite_manifest
            ),

        "composite_normalization_sha256":
            sha256_file(
                paths.composite_normalization
            ),

        "targets":
            list(
                selected_targets
            ),

        "feature_sets": {
            name:
                list(
                    feature_sets[
                        name
                    ]
                )
            for name
            in selected_feature_sets
        },

        "seed":
            SEED,

        "inner_valid_size":
            INNER_VALID_SIZE,

        "max_boost_rounds":
            MAX_BOOST_ROUNDS,

        "early_stopping_rounds":
            EARLY_STOPPING_ROUNDS,

        "lgb_params":
            LGB_PARAMS,

        "grouped_split":
            GROUPED_SPLIT_NAME,

        "minimum_component_count":
            MIN_COMPONENT_COUNT,
    }


def payload_hash(
    payload: Mapping[str, Any],
) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        serialized
    ).hexdigest()


def prepare_checkpoint(
    paths: Paths,
    payload: Mapping[str, Any],
    fresh: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if (
        fresh
        and os.path.isdir(
            paths.output_dir
        )
    ):
        shutil.rmtree(
            paths.output_dir
        )

    create_output_dirs(
        paths,
        save_models=False,
    )

    current_hash = payload_hash(
        payload
    )

    if os.path.isfile(
        paths.checkpoint_config
    ):
        with open(
            paths.checkpoint_config,
            "r",
            encoding="utf-8",
        ) as file:
            previous = json.load(
                file
            )

        if (
            previous.get(
                "configuration_hash"
            )
            != current_hash
        ):
            raise RuntimeError(
                "现有断点结果与当前数据/参数/"
                "目标/特征组合不一致。"
                "请使用 --fresh 重新运行。"
            )

    else:
        atomic_write_json(
            {
                "configuration_hash":
                    current_hash,

                "configuration":
                    payload,
            },
            paths.checkpoint_config,
        )

    metrics = (
        pd.read_csv(
            paths.metrics_fold,
            low_memory=False,
        )
        if os.path.isfile(
            paths.metrics_fold
        )
        else pd.DataFrame()
    )

    predictions = (
        pd.read_csv(
            paths.predictions,
            low_memory=False,
        )
        if os.path.isfile(
            paths.predictions
        )
        else pd.DataFrame()
    )

    baseline_audit = (
        pd.read_csv(
            paths.baseline_audit,
            low_memory=False,
        )
        if os.path.isfile(
            paths.baseline_audit
        )
        else pd.DataFrame()
    )

    return (
        metrics,
        predictions,
        baseline_audit,
    )


def completed_keys(
    metrics: pd.DataFrame,
) -> Set[
    Tuple[
        str,
        str,
        int,
    ]
]:
    if metrics.empty:
        return set()

    required = [
        "target",
        "feature_set",
        "fold",
        "status",
    ]

    if any(
        column
        not in metrics.columns
        for column in required
    ):
        return set()

    completed = metrics[
        metrics[
            "status"
        ] == "ok"
    ]

    return {
        (
            str(
                row[
                    "target"
                ]
            ),
            str(
                row[
                    "feature_set"
                ]
            ),
            int(
                row[
                    "fold"
                ]
            ),
        )
        for (
            _,
            row,
        ) in completed.iterrows()
    }


# =============================================================================
# 13. 汇总
# =============================================================================

def pooled_metrics_from_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    grouped = predictions.groupby(
        [
            "target",
            "feature_set",
        ],
        sort=False,
        dropna=False,
    )

    for (
        target,
        feature_set,
    ), frame in grouped:
        metrics = evaluate_regression(
            frame[
                "y_true"
            ].to_numpy(
                dtype=float
            ),
            frame[
                "y_pred"
            ].to_numpy(
                dtype=float
            ),
        )

        rows.append(
            {
                "target":
                    target,

                "target_label":
                    TARGET_LABELS.get(
                        target,
                        target,
                    ),

                "feature_set":
                    feature_set,

                "total_oof_samples":
                    metrics[
                        "n"
                    ],

                "total_oof_papers":
                    int(
                        frame[
                            "paper_id"
                        ].nunique()
                    ),

                "fold_count":
                    int(
                        frame[
                            "fold"
                        ].nunique()
                    ),

                "pooled_mae":
                    metrics[
                        "mae"
                    ],

                "pooled_rmse":
                    metrics[
                        "rmse"
                    ],

                "pooled_r2":
                    metrics[
                        "r2"
                    ],

                "pooled_spearman":
                    metrics[
                        "spearman"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def build_metrics_summary(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    fold_summary_rows: List[
        Dict[str, Any]
    ] = []

    for (
        target,
        feature_set,
    ), frame in fold_metrics.groupby(
        [
            "target",
            "feature_set",
        ],
        sort=False,
        dropna=False,
    ):
        row: Dict[
            str,
            Any,
        ] = {
            "target":
                target,

            "target_label":
                TARGET_LABELS.get(
                    target,
                    target,
                ),

            "feature_set":
                feature_set,

            "fold_count":
                int(
                    frame[
                        "fold"
                    ].nunique()
                ),

            "feature_count":
                int(
                    frame[
                        "feature_count"
                    ].iloc[
                        0
                    ]
                ),

            "best_iteration_median":
                float(
                    frame[
                        "best_iteration"
                    ].median()
                ),
        }

        for metric in [
            "mae",
            "rmse",
            "r2",
            "spearman",
        ]:
            row[
                f"{metric}_fold_mean"
            ] = frame[
                metric
            ].mean()

            row[
                f"{metric}_fold_std"
            ] = (
                frame[
                    metric
                ].std(
                    ddof=1
                )
                if len(
                    frame
                ) > 1
                else np.nan
            )

        fold_summary_rows.append(
            row
        )

    fold_summary = pd.DataFrame(
        fold_summary_rows
    )

    pooled = pooled_metrics_from_predictions(
        predictions
    )

    summary = fold_summary.merge(
        pooled,
        on=[
            "target",
            "target_label",
            "feature_set",
        ],
        how="outer",
    )

    full = (
        summary[
            summary[
                "feature_set"
            ] == "full"
        ][
            [
                "target",
                "pooled_mae",
                "pooled_rmse",
                "pooled_r2",
                "pooled_spearman",
            ]
        ]
        .rename(
            columns={
                "pooled_mae":
                    "full_pooled_mae",

                "pooled_rmse":
                    "full_pooled_rmse",

                "pooled_r2":
                    "full_pooled_r2",

                "pooled_spearman":
                    "full_pooled_spearman",
            }
        )
    )

    summary = summary.merge(
        full,
        on="target",
        how="left",
    )

    summary[
        "delta_r2_full_minus_variant"
    ] = (
        summary[
            "full_pooled_r2"
        ]
        - summary[
            "pooled_r2"
        ]
    )

    summary[
        "delta_spearman_full_minus_variant"
    ] = (
        summary[
            "full_pooled_spearman"
        ]
        - summary[
            "pooled_spearman"
        ]
    )

    summary[
        "delta_mae_variant_minus_full"
    ] = (
        summary[
            "pooled_mae"
        ]
        - summary[
            "full_pooled_mae"
        ]
    )

    summary[
        "delta_rmse_variant_minus_full"
    ] = (
        summary[
            "pooled_rmse"
        ]
        - summary[
            "full_pooled_rmse"
        ]
    )

    target_rank = {
        target:
            index
        for (
            index,
            target,
        ) in enumerate(
            TARGET_ORDER
        )
    }

    feature_order = [
        "N_only",
        "C_only",
        "F_only",
        "P_only",
        "N_C",
        "N_C_F",
        "full",
        "physical_only",
        "missing_only",
        "drop_N",
        "drop_C",
        "drop_F",
        "drop_P",
    ]

    feature_rank = {
        name:
            index
        for (
            index,
            name,
        ) in enumerate(
            feature_order
        )
    }

    summary[
        "_target_order"
    ] = summary[
        "target"
    ].map(
        lambda value:
            target_rank.get(
                value,
                len(
                    target_rank
                ),
            )
    )

    summary[
        "_feature_order"
    ] = summary[
        "feature_set"
    ].map(
        lambda value:
            feature_rank.get(
                value,
                len(
                    feature_rank
                ),
            )
    )

    return (
        summary
        .sort_values(
            [
                "_target_order",
                "_feature_order",
            ]
        )
        .drop(
            columns=[
                "_target_order",
                "_feature_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 14. 绘图
# =============================================================================

def apply_axes(
    ax: plt.Axes,
) -> None:
    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.tick_params(
        direction="out"
    )


def save_figure(
    fig: plt.Figure,
    base_path: str,
    dpi: int,
) -> None:
    fig.savefig(
        base_path
        + ".png",
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        base_path
        + ".pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def build_heatmap_matrix(
    summary: pd.DataFrame,
    feature_sets: Sequence[str],
    metric: str,
) -> Tuple[
    pd.DataFrame,
    List[str],
]:
    targets = ordered_targets_present(
        summary[
            "target"
        ]
    )

    matrix = pd.DataFrame(
        np.nan,
        index=targets,
        columns=list(
            feature_sets
        ),
        dtype=float,
    )

    for (
        _,
        row,
    ) in summary.iterrows():
        target = str(
            row[
                "target"
            ]
        )

        feature_set = str(
            row[
                "feature_set"
            ]
        )

        if (
            target
            in matrix.index
            and feature_set
            in matrix.columns
        ):
            matrix.loc[
                target,
                feature_set,
            ] = row[
                metric
            ]

    return (
        matrix,
        targets,
    )


def plot_ablation_heatmap(
    summary: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    displayed_feature_sets = [
        "N_only",
        "C_only",
        "F_only",
        "P_only",
        "N_C",
        "N_C_F",
        "full",
        "physical_only",
        "missing_only",
    ]

    (
        r2_matrix,
        targets,
    ) = build_heatmap_matrix(
        summary,
        displayed_feature_sets,
        "pooled_r2",
    )

    (
        spearman_matrix,
        _,
    ) = build_heatmap_matrix(
        summary,
        displayed_feature_sets,
        "pooled_spearman",
    )

    source = []

    for target in targets:
        for feature_set in displayed_feature_sets:
            source.append(
                {
                    "target":
                        target,

                    "target_label":
                        TARGET_LABELS.get(
                            target,
                            target,
                        ),

                    "feature_set":
                        feature_set,

                    "pooled_r2":
                        r2_matrix.loc[
                            target,
                            feature_set,
                        ],

                    "pooled_spearman":
                        spearman_matrix.loc[
                            target,
                            feature_set,
                        ],
                }
            )

    source_path = os.path.join(
        paths.source_data_dir,
        "fig4a_retraining_ablation_heatmap_source_data.csv",
    )

    pd.DataFrame(
        source
    ).to_csv(
        source_path,
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            16.0,
            0.72
            * len(
                targets
            )
            + 3.2,
        ),
    )

    for (
        ax,
        matrix,
        title,
    ) in [
        (
            axes[0],
            r2_matrix,
            r"Paper-grouped pooled OOF $R^2$",
        ),
        (
            axes[1],
            spearman_matrix,
            "Paper-grouped pooled OOF Spearman",
        ),
    ]:
        image = ax.imshow(
            matrix.to_numpy(
                dtype=float
            ),
            aspect="auto",
        )

        ax.set_xticks(
            np.arange(
                len(
                    displayed_feature_sets
                )
            )
        )

        ax.set_xticklabels(
            displayed_feature_sets,
            rotation=35,
            ha="right",
        )

        ax.set_yticks(
            np.arange(
                len(
                    targets
                )
            )
        )

        ax.set_yticklabels(
            [
                TARGET_LABELS.get(
                    target,
                    target,
                )
                for target
                in targets
            ]
        )

        ax.set_title(
            title
        )

        for row_index in range(
            len(
                targets
            )
        ):
            for column_index in range(
                len(
                    displayed_feature_sets
                )
            ):
                value = matrix.iloc[
                    row_index,
                    column_index,
                ]

                if np.isfinite(
                    value
                ):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                    )

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    fig.suptitle(
        "Retraining ablation of the N/C/F/P physical-state representation",
        y=1.01,
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig4a_retraining_ablation_heatmap",
        ),
        dpi,
    )


def plot_drop_one_delta(
    summary: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> None:
    drop_sets = [
        "drop_N",
        "drop_C",
        "drop_F",
        "drop_P",
    ]

    targets = ordered_targets_present(
        summary[
            "target"
        ]
    )

    source = summary[
        summary[
            "feature_set"
        ].isin(
            drop_sets
        )
    ][
        [
            "target",
            "target_label",
            "feature_set",
            "pooled_r2",
            "pooled_spearman",
            "full_pooled_r2",
            "full_pooled_spearman",
            "delta_r2_full_minus_variant",
            "delta_spearman_full_minus_variant",
        ]
    ].copy()

    source_path = os.path.join(
        paths.source_data_dir,
        "fig4b_drop_one_delta_source_data.csv",
    )

    source.to_csv(
        source_path,
        index=False,
        encoding="utf-8-sig",
    )

    x = np.arange(
        len(
            targets
        ),
        dtype=float,
    )

    width = 0.19

    fig, ax = plt.subplots(
        figsize=(
            13.0,
            6.0,
        )
    )

    for (
        index,
        feature_set,
    ) in enumerate(
        drop_sets
    ):
        values = []

        for target in targets:
            row = source[
                (
                    source[
                        "target"
                    ] == target
                )
                & (
                    source[
                        "feature_set"
                    ] == feature_set
                )
            ]

            values.append(
                float(
                    row[
                        "delta_r2_full_minus_variant"
                    ].iloc[
                        0
                    ]
                )
                if not row.empty
                else np.nan
            )

        offset = (
            index
            - (
                len(
                    drop_sets
                )
                - 1
            )
            / 2.0
        ) * width

        ax.bar(
            x
            + offset,
            np.asarray(
                values,
                dtype=float,
            ),
            width=width,
            label=feature_set.replace(
                "drop_",
                "Remove ",
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            TARGET_LABELS.get(
                target,
                target,
            )
            for target
            in targets
        ],
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        r"$\Delta R^2$ = Full $R^2$ − Reduced-model $R^2$"
    )

    ax.set_title(
        "Loss of cross-paper predictive performance after removing one block"
    )

    ax.legend(
        frameon=False,
        ncol=4,
    )

    apply_axes(
        ax
    )

    fig.tight_layout()

    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig4b_drop_one_delta_r2",
        ),
        dpi,
    )


def plot_block_shap(
    paths: Paths,
    dpi: int,
) -> bool:
    if not os.path.isfile(
        paths.shap_block_summary
    ):
        print(
            "[WARN] 缺少 oof_block_shap_summary.csv，"
            "跳过Figure 4c。"
        )

        return False

    frame = pd.read_csv(
        paths.shap_block_summary,
        low_memory=False,
    )

    required = [
        "target",
        "N_share",
        "C_share",
        "F_share",
        "P_share",
        "missingness_share",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        print(
            f"[WARN] SHAP block表缺少{missing}，"
            "跳过Figure 4c。"
        )

        return False

    short_order = [
        "gq",
        "igid",
        "ori",
        "purity",
        "bundle",
        "composite",
    ]

    present = set(
        frame[
            "target"
        ].astype(str)
    )

    order = [
        name
        for name in short_order
        if name in present
    ]

    frame = (
        frame
        .set_index(
            "target"
        )
        .loc[
            order
        ]
        .reset_index()
    )

    source_path = os.path.join(
        paths.source_data_dir,
        "fig4c_oof_block_shap_source_data.csv",
    )

    frame.to_csv(
        source_path,
        index=False,
        encoding="utf-8-sig",
    )

    x = np.arange(
        len(
            frame
        )
    )

    bottom = np.zeros(
        len(
            frame
        ),
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            6.0,
        )
    )

    for module in MODULES:
        values = frame[
            f"{module}_share"
        ].to_numpy(
            dtype=float
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=module,
        )

        bottom += values

    ax.plot(
        x,
        frame[
            "missingness_share"
        ].to_numpy(
            dtype=float
        ),
        marker="o",
        linewidth=1.5,
        label="Missingness share",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        frame[
            "target"
        ].astype(str)
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_ylabel(
        "Share of mean |OOF TreeSHAP|"
    )

    ax.set_title(
        "Cross-paper OOF TreeSHAP attribution across physical-state blocks"
    )

    ax.legend(
        frameon=False,
        ncol=5,
    )

    apply_axes(
        ax
    )

    fig.tight_layout()

    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig4c_oof_block_shap",
        ),
        dpi,
    )

    return True


def plot_missingness_audit(
    summary: pd.DataFrame,
    paths: Paths,
    dpi: int,
) -> bool:
    if not os.path.isfile(
        paths.shap_block_summary
    ):
        print(
            "[WARN] 缺少SHAP block表，跳过Figure 4d。"
        )

        return False

    shap = pd.read_csv(
        paths.shap_block_summary,
        low_memory=False,
    )

    if not {
        "target",
        "missingness_share",
    }.issubset(
        shap.columns
    ):
        print(
            "[WARN] SHAP block表缺少missingness_share。"
        )

        return False

    missing_model = summary[
        summary[
            "feature_set"
        ] == "missing_only"
    ][
        [
            "target",
            "target_label",
            "pooled_r2",
            "pooled_spearman",
        ]
    ].copy()

    missing_model[
        "short_target"
    ] = missing_model[
        "target"
    ].map(
        SHORT_TARGET_MAP
    )

    missing_model = missing_model[
        missing_model[
            "short_target"
        ].notna()
    ]

    merged = missing_model.merge(
        shap[
            [
                "target",
                "missingness_share",
            ]
        ].rename(
            columns={
                "target":
                    "short_target"
            }
        ),
        on="short_target",
        how="inner",
    )

    if merged.empty:
        print(
            "[WARN] SHAP目标与missing-only模型无法对应。"
        )

        return False

    source_path = os.path.join(
        paths.source_data_dir,
        "fig4d_missingness_audit_source_data.csv",
    )

    merged.to_csv(
        source_path,
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            6.0,
        )
    )

    for (
        _,
        row,
    ) in merged.iterrows():
        x = float(
            row[
                "missingness_share"
            ]
        )

        y = float(
            row[
                "pooled_r2"
            ]
        )

        ax.scatter(
            x,
            y,
        )

        ax.text(
            x
            + 0.004,
            y,
            str(
                row[
                    "short_target"
                ]
            ),
            fontsize=9,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Missing-indicator share of mean |OOF TreeSHAP|"
    )

    ax.set_ylabel(
        r"Missing-only pooled OOF $R^2$"
    )

    ax.set_title(
        "Missingness reliance assessed by SHAP and retraining"
    )

    apply_axes(
        ax
    )

    fig.tight_layout()

    save_figure(
        fig,
        os.path.join(
            paths.figures_dir,
            "fig4d_missingness_audit",
        ),
        dpi,
    )

    return True


# =============================================================================
# 15. 主程序
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用固定paper-grouped folds重新训练"
            "N/C/F/P消融模型，生成文章Figure 4与Source Data。"
        )
    )

    parser.add_argument(
        "--project_dir",
        default=None,
        help=(
            "项目目录；默认从 models/external_validation "
            "脚本位置自动识别项目根目录。"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "输出目录；默认 outputs/external_validation/ablation。"
        ),
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "只运行指定目标；默认运行全部8个目标和composite。"
        ),
    )

    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=None,
        help=(
            "只运行指定特征组合；默认运行全部组合。"
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "删除现有 outputs/external_validation/ablation "
            "并从头运行。"
        ),
    )

    parser.add_argument(
        "--save_models",
        action="store_true",
        help=(
            "保存全部fold消融模型；默认不保存以节省磁盘。"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=(
            "PNG分辨率，默认600 dpi。"
        ),
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )

    project_dir = (
        os.path.abspath(
            args.project_dir
        )
        if args.project_dir
        else os.path.dirname(
            os.path.dirname(
                script_dir
            )
        )
    )

    paths = build_paths(
        project_dir,
        args.output_dir,
    )

    ensure_required_inputs(
        paths
    )

    if (
        args.fresh
        and os.path.isdir(
            paths.output_dir
        )
    ):
        shutil.rmtree(
            paths.output_dir
        )

    create_output_dirs(
        paths,
        save_models=args.save_models,
    )

    all_features = load_feature_schema(
        paths.schema
    )

    feature_sets = build_feature_sets(
        all_features
    )

    feature_manifest = feature_set_manifest(
        feature_sets
    )

    selected_feature_sets = (
        list(
            args.feature_sets
        )
        if args.feature_sets
        else list(
            feature_sets.keys()
        )
    )

    unknown_feature_sets = [
        name
        for name in selected_feature_sets
        if name not in feature_sets
    ]

    if unknown_feature_sets:
        raise ValueError(
            f"未知特征组合：{unknown_feature_sets}；"
            f"可选：{list(feature_sets)}"
        )

    dataset = load_dataset(
        paths.dataset,
        all_features,
    )

    ordinary_manifest = load_manifest(
        paths.ordinary_manifest,
        "L1_multi_target",
    )

    composite_manifest = load_manifest(
        paths.composite_manifest,
        "L1_composite",
    )

    composite_normalization = pd.read_csv(
        paths.composite_normalization,
        low_memory=False,
    )

    composite_normalization[
        "fold"
    ] = pd.to_numeric(
        composite_normalization[
            "fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    saved_predictions = load_saved_predictions(
        paths.ordinary_predictions,
        paths.composite_predictions,
    )

    available_targets = ordered_targets_present(
        set(
            ordinary_manifest[
                "target"
            ]
        ).union(
            set(
                composite_manifest[
                    "target"
                ]
            )
        )
    )

    selected_targets = (
        list(
            args.targets
        )
        if args.targets
        else available_targets
    )

    unknown_targets = [
        target
        for target in selected_targets
        if target not in available_targets
    ]

    if unknown_targets:
        raise ValueError(
            f"manifest中不存在目标：{unknown_targets}；"
            f"可选：{available_targets}"
        )

    payload = configuration_payload(
        paths=paths,
        selected_targets=selected_targets,
        selected_feature_sets=selected_feature_sets,
        feature_sets=feature_sets,
    )

    (
        metrics_checkpoint,
        predictions_checkpoint,
        baseline_checkpoint,
    ) = prepare_checkpoint(
        paths=paths,
        payload=payload,
        fresh=False,
    )

    # --fresh 已在上面完成删除，因此此处只负责加载/建立当前断点
    create_output_dirs(
        paths,
        save_models=args.save_models,
    )

    feature_manifest.to_csv(
        paths.feature_sets,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "=" * 100
    )

    print(
        "[ARTICLE RETRAINING ABLATION START]"
    )

    print(
        f"Project directory : {paths.project_dir}"
    )

    print(
        f"L1 rows           : {len(dataset)}"
    )

    print(
        f"Composite threshold: {MIN_COMPONENT_COUNT}"
    )

    print(
        f"Feature count     : {len(all_features)}"
    )

    print(
        f"Targets           : {selected_targets}"
    )

    print(
        f"Feature sets      : {selected_feature_sets}"
    )

    print(
        "Expected models   : "
        f"{len(selected_targets) * len(selected_feature_sets) * EXPECTED_GROUP_FOLDS}"
    )

    print(
        "Resume completed  : "
        f"{len(completed_keys(metrics_checkpoint))}"
    )

    print(
        f"Save models       : {args.save_models}"
    )

    print(
        f"Output            : {paths.output_dir}"
    )

    print(
        "=" * 100
    )

    split_audit_rows: List[
        Dict[str, Any]
    ] = []

    for target in selected_targets:
        manifest = target_manifest(
            ordinary_manifest,
            composite_manifest,
            target,
        )

        split_audit_rows.extend(
            audit_target_folds(
                manifest=manifest,
                dataset_sample_ids=set(
                    dataset[
                        "sample_id"
                    ]
                ),
                target=target,
            )
        )

    split_audit_df = pd.DataFrame(
        split_audit_rows
    )

    split_audit_df.to_csv(
        paths.split_audit,
        index=False,
        encoding="utf-8-sig",
    )

    # 预备普通目标数据
    ordinary_target_data: Dict[
        str,
        TargetData,
    ] = {}

    for target in selected_targets:
        if target != TARGET_COMPOSITE:
            ordinary_target_data[
                target
            ] = prepare_ordinary_target(
                dataset=dataset,
                all_features=all_features,
                target=target,
            )

    (
        composite_work,
        composite_X,
    ) = prepare_composite_base(
        dataset=dataset,
        all_features=all_features,
    )

    completed = completed_keys(
        metrics_checkpoint
    )

    metric_rows: List[
        Dict[str, Any]
    ] = (
        metrics_checkpoint.to_dict(
            "records"
        )
        if not metrics_checkpoint.empty
        else []
    )

    prediction_records: List[
        Dict[str, Any]
    ] = (
        predictions_checkpoint.to_dict(
            "records"
        )
        if not predictions_checkpoint.empty
        else []
    )

    baseline_rows: List[
        Dict[str, Any]
    ] = (
        baseline_checkpoint.to_dict(
            "records"
        )
        if not baseline_checkpoint.empty
        else []
    )

    total_models = (
        len(
            selected_targets
        )
        * len(
            selected_feature_sets
        )
        * EXPECTED_GROUP_FOLDS
    )

    done_before = len(
        completed
    )

    run_counter = 0

    for target in selected_targets:
        manifest = target_manifest(
            ordinary_manifest,
            composite_manifest,
            target,
        )

        if target == TARGET_COMPOSITE:
            target_data = TargetData(
                target=TARGET_COMPOSITE,
                source_model="L1_composite",
                work=composite_work,
                X_all=composite_X,
                y_model=np.full(
                    len(
                        composite_work
                    ),
                    np.nan,
                ),
                y_original=np.full(
                    len(
                        composite_work
                    ),
                    np.nan,
                ),
                use_log=False,
            )

        else:
            target_data = ordinary_target_data[
                target
            ]

        folds = sorted(
            manifest[
                "fold"
            ].unique().tolist()
        )

        for feature_set_name in selected_feature_sets:
            feature_columns = feature_sets[
                feature_set_name
            ]

            for fold in folds:
                key = (
                    target,
                    feature_set_name,
                    int(
                        fold
                    ),
                )

                if key in completed:
                    continue

                run_counter += 1

                absolute_position = (
                    done_before
                    + run_counter
                )

                print(
                    f"[MODEL {absolute_position}/{total_models}] "
                    f"{target} / "
                    f"{feature_set_name} / "
                    f"fold {fold}"
                )

                fold_manifest = manifest[
                    manifest[
                        "fold"
                    ] == fold
                ]

                normalizer = None

                if (
                    target
                    == TARGET_COMPOSITE
                ):
                    normalizer = normalization_for_fold(
                        composite_normalization,
                        int(
                            fold
                        ),
                    )

                (
                    metric_row,
                    prediction_rows_current,
                    auxiliary,
                ) = run_fold_model(
                    target_data=target_data,
                    feature_set_name=feature_set_name,
                    feature_columns=feature_columns,
                    fold_manifest=fold_manifest,
                    fold=int(
                        fold
                    ),
                    paths=paths,
                    save_models=args.save_models,
                    normalizer=normalizer,
                )

                metric_rows.append(
                    metric_row
                )

                prediction_records.extend(
                    prediction_rows_current
                )

                completed.add(
                    key
                )

                if (
                    feature_set_name
                    == "full"
                ):
                    baseline_rows = [
                        row
                        for row in baseline_rows
                        if not (
                            str(
                                row.get(
                                    "target"
                                )
                            )
                            == target
                            and int(
                                row.get(
                                    "fold"
                                )
                            )
                            == int(
                                fold
                            )
                        )
                    ]

                    baseline_rows.append(
                        baseline_reproduction_row(
                            saved_predictions=saved_predictions,
                            target=target,
                            fold=int(
                                fold
                            ),
                            current_sample_ids=auxiliary[
                                "test_sample_ids"
                            ],
                            current_y_true=auxiliary[
                                "y_true"
                            ],
                            current_y_pred=auxiliary[
                                "y_pred"
                            ],
                        )
                    )

                # 每完成一个模型立即保存断点
                metric_frame = pd.DataFrame(
                    metric_rows
                )

                prediction_frame = pd.DataFrame(
                    prediction_records
                )

                baseline_frame = pd.DataFrame(
                    baseline_rows
                )

                atomic_write_csv(
                    metric_frame,
                    paths.metrics_fold,
                )

                atomic_write_csv(
                    prediction_frame,
                    paths.predictions,
                )

                atomic_write_csv(
                    baseline_frame,
                    paths.baseline_audit,
                )

    fold_metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.DataFrame(
        prediction_records
    )

    baseline_audit = pd.DataFrame(
        baseline_rows
    )

    if (
        fold_metrics.empty
        or predictions.empty
    ):
        raise RuntimeError(
            "没有生成任何消融结果。"
        )

    summary = build_metrics_summary(
        fold_metrics=fold_metrics,
        predictions=predictions,
    )

    summary.to_csv(
        paths.metrics_summary,
        index=False,
        encoding="utf-8-sig",
    )

    # 文章表格
    table_main = summary[
        summary[
            "feature_set"
        ].isin(
            [
                "N_only",
                "C_only",
                "F_only",
                "P_only",
                "N_C",
                "N_C_F",
                "full",
                "physical_only",
                "missing_only",
            ]
        )
    ].copy()

    table_drop = summary[
        summary[
            "feature_set"
        ].isin(
            [
                "drop_N",
                "drop_C",
                "drop_F",
                "drop_P",
            ]
        )
    ].copy()

    table_main.to_csv(
        os.path.join(
            paths.tables_dir,
            "table_main_retraining_ablation.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    table_drop.to_csv(
        os.path.join(
            paths.tables_dir,
            "table_si_drop_one_ablation.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics.to_csv(
        os.path.join(
            paths.tables_dir,
            "table_si_ablation_fold_metrics.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # Figure 4
    plot_ablation_heatmap(
        summary,
        paths,
        args.dpi,
    )

    plot_drop_one_delta(
        summary,
        paths,
        args.dpi,
    )

    shap_figure_created = plot_block_shap(
        paths,
        args.dpi,
    )

    missing_figure_created = plot_missingness_audit(
        summary,
        paths,
        args.dpi,
    )

    run_summary = {
        "analysis":
            "CNT 3.0 article-ready retraining ablation",

        "project_dir":
            paths.project_dir,

        "output_dir":
            paths.output_dir,

        "dataset": {
            "path":
                paths.dataset,

            "sha256":
                sha256_file(
                    paths.dataset
                ),

            "L1_rows":
                int(
                    len(
                        dataset
                    )
                ),
        },

        "composite": {
            "weights":
                WEIGHTS,

            "missing_policy":
                MISSING_POLICY,

            "minimum_component_count":
                MIN_COMPONENT_COUNT,

            "bundle_transform":
                "log1p before fold-train min-max normalization",
        },

        "validation": {
            "split_type":
                GROUPED_SPLIT_NAME,

            "expected_folds":
                EXPECTED_GROUP_FOLDS,

            "outer_folds_reused_from_existing_manifest":
                True,

            "inner_validation":
                "GroupShuffleSplit within outer training set",

            "inner_valid_size":
                INNER_VALID_SIZE,

            "test_set_used_for_early_stopping":
                False,
        },

        "model": {
            "algorithm":
                "LightGBM regression",

            "parameters":
                LGB_PARAMS,

            "max_boost_rounds":
                MAX_BOOST_ROUNDS,

            "early_stopping_rounds":
                EARLY_STOPPING_ROUNDS,
        },

        "targets":
            selected_targets,

        "feature_sets":
            selected_feature_sets,

        "model_count_completed":
            int(
                len(
                    fold_metrics
                )
            ),

        "prediction_rows":
            int(
                len(
                    predictions
                )
            ),

        "baseline_reproduction_status_counts":
            (
                baseline_audit[
                    "status"
                ]
                .value_counts()
                .to_dict()
                if not baseline_audit.empty
                else {}
            ),

        "outputs": {
            "metrics_fold":
                paths.metrics_fold,

            "metrics_summary":
                paths.metrics_summary,

            "predictions":
                paths.predictions,

            "feature_sets":
                paths.feature_sets,

            "baseline_audit":
                paths.baseline_audit,

            "split_audit":
                paths.split_audit,

            "figures_dir":
                paths.figures_dir,

            "source_data_dir":
                paths.source_data_dir,

            "tables_dir":
                paths.tables_dir,
        },

        "figures": {
            "fig4a_retraining_ablation_heatmap":
                True,

            "fig4b_drop_one_delta_r2":
                True,

            "fig4c_oof_block_shap":
                shap_figure_created,

            "fig4d_missingness_audit":
                missing_figure_created,
        },

        "interpretation": {
            "single_module_models":
                (
                    "Each block is retrained with its own physical descriptors "
                    "and corresponding missingness indicators."
                ),

            "drop_one_models":
                (
                    "Each reduced model is retrained after removing one complete "
                    "physical-state block and its missingness indicators."
                ),

            "physical_only":
                (
                    "All 60 physical descriptors without missingness indicators."
                ),

            "missing_only":
                (
                    "All 60 missingness indicators without physical descriptors."
                ),

            "delta_definition":
                (
                    "delta_r2_full_minus_variant = pooled OOF R2 of the retrained "
                    "full model minus pooled OOF R2 of the retrained variant."
                ),

            "not_occlusion":
                True,
        },
    }

    atomic_write_json(
        run_summary,
        paths.run_summary,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "[ARTICLE RETRAINING ABLATION FINISHED]"
    )

    print(
        "Fold metrics       : "
        f"{paths.metrics_fold}"
    )

    print(
        "Summary metrics    : "
        f"{paths.metrics_summary}"
    )

    print(
        "Predictions        : "
        f"{paths.predictions}"
    )

    print(
        "Feature sets       : "
        f"{paths.feature_sets}"
    )

    print(
        "Baseline audit     : "
        f"{paths.baseline_audit}"
    )

    print(
        "Split audit        : "
        f"{paths.split_audit}"
    )

    print(
        "Figures            : "
        f"{paths.figures_dir}"
    )

    print(
        "Source data        : "
        f"{paths.source_data_dir}"
    )

    print(
        "Tables             : "
        f"{paths.tables_dir}"
    )

    print(
        "Run summary        : "
        f"{paths.run_summary}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()