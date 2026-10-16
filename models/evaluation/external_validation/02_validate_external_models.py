# -*- coding: utf-8 -*-
"""
02_validate_external_models.py

使用已经冻结的正式模型，对 data/12_external_validation/external_dataset_full60D.csv
进行独立外部验证。

原则：
1. 不重新训练模型；
2. 不重新拟合任何归一化参数；
3. 模型特征顺序严格读取 data/11_flattened/feature_schema.json；
4. L1 单目标仅在 external L1 样本上评价；
5. L1 composite 仅在 external L1 样本上评价；
6. L12 quantile 仅在 external L1+L2 样本上评价；
7. Composite 真值严格使用训练阶段保存的 final normalization；
8. 先检查 external 与正式训练数据库的 paper_id / DOI 重叠。

默认输出：
outputs/evaluation/external_validation/
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. 路径
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_EXTERNAL_CSV = (
    PROJECT_ROOT / "data" / "12_external_validation" / "external_dataset_full60D.csv"
)
DEFAULT_TRAIN_CSV = (
    PROJECT_ROOT / "data" / "11_flattened" / "ml_dataset_3p0_full60D.csv"
)
FROZEN_SCHEMA_JSON = (
    PROJECT_ROOT / "data" / "11_flattened" / "feature_schema.json"
)

L1_TARGET_DIR = PROJECT_ROOT / "outputs" / "models" / "l1_targets"
L1_TARGET_FINAL_DIR = L1_TARGET_DIR / "final_models"
L1_TARGET_MANIFEST = L1_TARGET_DIR / "final_model_manifest.csv"

L1_COMPOSITE_DIR = PROJECT_ROOT / "outputs" / "models" / "l1_composite"
L1_COMPOSITE_MODEL = (
    L1_COMPOSITE_DIR / "final_models" / "L1_composite_final_score.txt"
)
L1_COMPOSITE_NORMALIZATION = (
    L1_COMPOSITE_DIR / "final_composite_normalization.json"
)

L12_DIR = PROJECT_ROOT / "outputs" / "models" / "l12_quantile"
L12_Q10_MODEL = L12_DIR / "final_models" / "L12_composite_q10.txt"
L12_Q50_MODEL = L12_DIR / "final_models" / "L12_composite_q50.txt"
L12_Q90_MODEL = L12_DIR / "final_models" / "L12_composite_q90.txt"
L12_NORMALIZATION = L12_DIR / "final_L12_composite_normalization.json"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "evaluation" / "external_validation"
)


# =============================================================================
# 2. 固定定义
# =============================================================================

TARGET_COMPOSITE = "y_composite_final_score"

COL_GQ = "y_growth_quality_score"
COL_ORI = "y_orientation_score"
COL_PURITY = "y_purity_score"
COL_BUNDLE = "y_bundle_density_per_um"
COL_IGID_PREF = "y_ig_id_ratio"
COL_IGID_LEGACY = "y_IG_ID"
COL_IGID_UNIFIED = "y_ig_id_ratio_unified"

META_COLUMNS = [
    "sample_id",
    "paper_id",
    "paper_doi",
    "paper_title",
    "paper_year_int",
    "growth_mode",
    "data_level",
    "source_file",
    "synthesis_id",
]


# =============================================================================
# 3. 基础工具
# =============================================================================

def resolve_project_path(value: Optional[str], default: Path) -> Path:
    if value is None or not str(value).strip():
        return default.resolve()

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")


def load_json(path: Path) -> Dict[str, Any]:
    require_file(path, "JSON")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON 根对象不是 dict：{path}")
    return data


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_doi_series(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = " ".join(str(value).strip().split()).lower()
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

    return series.map(normalize)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) < 2:
        return float("nan")

    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)

    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        return float("nan")

    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": safe_r2(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
    }


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    loss = np.maximum(alpha * error, (alpha - 1.0) * error)
    return float(np.mean(loss))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower
    score = width.copy()

    below = y_true < lower
    above = y_true > upper

    score[below] += 2.0 / alpha * (lower[below] - y_true[below])
    score[above] += 2.0 / alpha * (y_true[above] - upper[above])

    return score


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def ensure_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults: Dict[str, Any] = {
        "sample_id": "",
        "paper_id": "",
        "paper_doi": "",
        "paper_title": "",
        "paper_year_int": np.nan,
        "growth_mode": "",
        "data_level": "",
        "source_file": "",
        "synthesis_id": "",
    }

    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    for column in (
        "sample_id",
        "paper_id",
        "paper_doi",
        "paper_title",
        "growth_mode",
        "data_level",
        "source_file",
        "synthesis_id",
    ):
        out[column] = out[column].fillna("").astype(str)

    out["paper_year_int"] = pd.to_numeric(
        out["paper_year_int"],
        errors="coerce",
    )

    return out


def ensure_igid_unified(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    preferred = (
        to_num(out[COL_IGID_PREF])
        if COL_IGID_PREF in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    legacy = (
        to_num(out[COL_IGID_LEGACY])
        if COL_IGID_LEGACY in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    out[COL_IGID_UNIFIED] = preferred.where(preferred.notna(), legacy)
    return out


# =============================================================================
# 4. 特征与数据审计
# =============================================================================

def load_frozen_feature_columns() -> List[str]:
    schema = load_json(FROZEN_SCHEMA_JSON)
    columns = schema.get("model_feature_columns")

    if not isinstance(columns, list) or not columns:
        raise RuntimeError(
            f"冻结 feature_schema 缺少 model_feature_columns：{FROZEN_SCHEMA_JSON}"
        )

    if not all(isinstance(column, str) for column in columns):
        raise RuntimeError("model_feature_columns 中存在非字符串字段。")

    if len(columns) != 120:
        raise RuntimeError(
            f"冻结模型特征数应为120，当前为 {len(columns)}。"
        )

    return list(columns)


def prepare_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "external dataset 缺少冻结模型所需特征："
            + ", ".join(missing[:20])
        )

    X = df[list(feature_columns)].copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    return X


def audit_feature_schema(
    external_schema_path: Path,
    frozen_feature_columns: Sequence[str],
) -> Dict[str, Any]:
    result = {
        "external_schema_exists": external_schema_path.is_file(),
        "external_model_feature_count": np.nan,
        "frozen_model_feature_count": len(frozen_feature_columns),
        "feature_order_exact_match": False,
    }

    if not external_schema_path.is_file():
        return result

    external_schema = load_json(external_schema_path)
    external_columns = external_schema.get("model_feature_columns", [])

    result["external_model_feature_count"] = (
        len(external_columns)
        if isinstance(external_columns, list)
        else np.nan
    )
    result["feature_order_exact_match"] = (
        isinstance(external_columns, list)
        and list(external_columns) == list(frozen_feature_columns)
    )

    return result


def build_overlap_audit(
    external_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    external = ensure_metadata(external_df)
    train = ensure_metadata(train_df)

    external["_doi_norm"] = normalize_doi_series(external["paper_doi"])
    train["_doi_norm"] = normalize_doi_series(train["paper_doi"])

    external_papers = (
        external[
            ["paper_id", "paper_doi", "paper_title", "paper_year_int", "_doi_norm"]
        ]
        .drop_duplicates()
        .copy()
    )
    train_papers = (
        train[
            ["paper_id", "paper_doi", "paper_title", "paper_year_int", "_doi_norm"]
        ]
        .drop_duplicates()
        .copy()
    )

    train_paper_ids = set(
        value
        for value in train_papers["paper_id"].astype(str)
        if value
    )
    train_dois = set(
        value
        for value in train_papers["_doi_norm"].astype(str)
        if value
    )

    rows: List[Dict[str, Any]] = []

    for _, item in external_papers.iterrows():
        paper_id = str(item["paper_id"])
        doi_norm = str(item["_doi_norm"])

        match_paper_id = bool(paper_id) and paper_id in train_paper_ids
        match_doi = bool(doi_norm) and doi_norm in train_dois

        if match_paper_id or match_doi:
            rows.append({
                "paper_id": paper_id,
                "paper_doi": item["paper_doi"],
                "paper_title": item["paper_title"],
                "paper_year_int": item["paper_year_int"],
                "match_paper_id": match_paper_id,
                "match_doi": match_doi,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "paper_id",
            "paper_doi",
            "paper_title",
            "paper_year_int",
            "match_paper_id",
            "match_doi",
        ],
    )


def build_dataset_audit(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    schema_audit: Dict[str, Any],
    overlap_count: int,
) -> pd.DataFrame:
    work = ensure_metadata(df)

    physical_columns = [
        column
        for column in feature_columns
        if not column.startswith("miss_")
    ]
    missing_columns = [
        column
        for column in feature_columns
        if column.startswith("miss_")
    ]

    level_counts = work["data_level"].value_counts(dropna=False).to_dict()
    mode_counts = work["growth_mode"].value_counts(dropna=False).to_dict()

    years = pd.to_numeric(work["paper_year_int"], errors="coerce")

    row = {
        "rows": int(len(work)),
        "unique_papers": int(work["paper_id"].nunique()),
        "year_min": int(years.min()) if years.notna().any() else np.nan,
        "year_max": int(years.max()) if years.notna().any() else np.nan,
        "rows_missing_year": int(years.isna().sum()),
        "feature_count": int(len(feature_columns)),
        "physical_feature_count": int(len(physical_columns)),
        "missingness_feature_count": int(len(missing_columns)),
        "external_schema_exists": schema_audit["external_schema_exists"],
        "external_model_feature_count": schema_audit["external_model_feature_count"],
        "frozen_model_feature_count": schema_audit["frozen_model_feature_count"],
        "feature_order_exact_match": schema_audit["feature_order_exact_match"],
        "overlap_papers_with_training": int(overlap_count),
    }

    for level in ("L1", "L2", "L3"):
        row[f"rows_{level}"] = int(level_counts.get(level, 0))

    for mode in ("A", "B", "C", "D", "E"):
        row[f"rows_mode_{mode}"] = int(mode_counts.get(mode, 0))

    return pd.DataFrame([row])


def build_target_coverage(df: pd.DataFrame) -> pd.DataFrame:
    work = ensure_igid_unified(ensure_metadata(df))

    candidate_targets = [
        COL_GQ,
        "y_diameter_nm",
        "y_length_mm",
        "y_bundle_density_per_um",
        "y_cnt_density_per_um",
        COL_IGID_UNIFIED,
        COL_ORI,
        COL_PURITY,
    ]

    rows: List[Dict[str, Any]] = []

    for target in candidate_targets:
        if target not in work.columns:
            continue

        values = to_num(work[target])

        for scope_name, mask in (
            ("all", pd.Series(True, index=work.index)),
            ("L1", work["data_level"].eq("L1")),
            ("L2", work["data_level"].eq("L2")),
        ):
            valid = mask & values.notna()

            rows.append({
                "target": target,
                "scope": scope_name,
                "n_samples": int(valid.sum()),
                "n_papers": int(work.loc[valid, "paper_id"].nunique()),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 5. 模型一致性
# =============================================================================

def load_booster(
    model_path: Path,
    feature_columns: Sequence[str],
) -> lgb.Booster:
    require_file(model_path, "LightGBM model")

    model = lgb.Booster(model_file=str(model_path))
    model_features = list(model.feature_name())

    if len(model_features) != len(feature_columns):
        raise RuntimeError(
            f"模型特征数不一致：{model_path.name} "
            f"model={len(model_features)}, frozen={len(feature_columns)}"
        )

    if model_features != list(feature_columns):
        raise RuntimeError(
            f"模型特征顺序与冻结 feature_schema 不一致：{model_path.name}"
        )

    return model


# =============================================================================
# 6. L1 单目标外部验证
# =============================================================================

def validate_l1_targets(
    external_df: pd.DataFrame,
    X_all: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_file(L1_TARGET_MANIFEST, "L1 final_model_manifest.csv")

    manifest = pd.read_csv(L1_TARGET_MANIFEST, low_memory=False)

    required_manifest = ["target", "status", "log_target"]
    missing_manifest = [
        column
        for column in required_manifest
        if column not in manifest.columns
    ]
    if missing_manifest:
        raise RuntimeError(
            f"L1 final_model_manifest.csv 缺少字段：{missing_manifest}"
        )

    work = ensure_igid_unified(ensure_metadata(external_df))
    l1_mask = work["data_level"].eq("L1")

    prediction_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    for _, model_row in manifest.iterrows():
        if str(model_row.get("status", "")).lower() != "ok":
            continue

        target = str(model_row["target"])
        if target not in work.columns:
            metric_rows.append({
                "target": target,
                "status": "target_missing_in_external_dataset",
                "n_samples": 0,
                "n_papers": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
            })
            continue

        target_values = to_num(work[target])
        valid = l1_mask & target_values.notna()

        use_log = str(model_row["log_target"]).strip().lower() in {
            "1", "true", "yes"
        }

        if use_log:
            valid = valid & (target_values > 0)

        indices = np.where(valid.to_numpy())[0]

        if len(indices) == 0:
            metric_rows.append({
                "target": target,
                "status": "no_valid_external_samples",
                "n_samples": 0,
                "n_papers": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
            })
            continue

        model_path = (
            L1_TARGET_FINAL_DIR
            / f"L1_{safe_name(target)}.txt"
        )
        model = load_booster(model_path, feature_columns)

        pred_model = np.asarray(
            model.predict(X_all.iloc[indices]),
            dtype=float,
        )

        y_true = target_values.iloc[indices].to_numpy(dtype=float)
        y_pred = np.expm1(pred_model) if use_log else pred_model

        metrics = evaluate_regression(y_true, y_pred)

        metric_rows.append({
            "target": target,
            "status": "ok",
            "n_samples": metrics["n"],
            "n_papers": int(work.iloc[indices]["paper_id"].nunique()),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "spearman": metrics["spearman"],
            "log_target": use_log,
            "model_path": str(model_path),
        })

        for position, index in enumerate(indices):
            item = work.iloc[index]
            true_value = float(y_true[position])
            pred_value = float(y_pred[position])

            prediction_rows.append({
                **{
                    column: item[column]
                    for column in META_COLUMNS
                    if column in item.index
                },
                "target": target,
                "y_true": true_value,
                "y_pred": pred_value,
                "residual": pred_value - true_value,
                "absolute_error": abs(pred_value - true_value),
                "log_target": use_log,
            })

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
    )


# =============================================================================
# 7. Composite 真值
# =============================================================================

def minmax_with_frozen(
    series: pd.Series,
    minimum: Optional[float],
    maximum: Optional[float],
    usable: bool,
) -> pd.Series:
    if (
        not usable
        or minimum is None
        or maximum is None
        or not np.isfinite(float(minimum))
        or not np.isfinite(float(maximum))
        or math.isclose(float(minimum), float(maximum), rel_tol=0.0, abs_tol=1e-15)
    ):
        return pd.Series(np.nan, index=series.index, dtype=float)

    normalized = (
        to_num(series) - float(minimum)
    ) / (
        float(maximum) - float(minimum)
    )

    return normalized.clip(lower=0.0, upper=1.0)


def build_composite_truth(
    df: pd.DataFrame,
    normalization_json: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = load_json(normalization_json)
    normalizer = config.get("normalizer")

    if not isinstance(normalizer, dict):
        raise RuntimeError(
            f"Composite normalization 缺少 normalizer：{normalization_json}"
        )

    weights = config.get(
        "weights",
        {
            "gq": 0.30,
            "igid": 0.20,
            "ori": 0.20,
            "purity": 0.15,
            "bundle": 0.15,
        },
    )

    if not isinstance(weights, dict):
        raise RuntimeError("Composite weights 格式错误。")

    missing_policy = str(config.get("missing_policy", "renorm"))
    minimum_component_count = int(
        config.get("minimum_component_count", 2)
    )

    work = ensure_igid_unified(ensure_metadata(df))

    for column in [
        COL_GQ,
        COL_IGID_UNIFIED,
        COL_ORI,
        COL_PURITY,
        COL_BUNDLE,
    ]:
        if column not in work.columns:
            work[column] = np.nan

    bundle = to_num(work[COL_BUNDLE]).to_numpy(dtype=float)
    bundle_log = np.where(
        np.isfinite(bundle) & (bundle >= 0.0),
        np.log1p(bundle),
        np.nan,
    )

    transformed = pd.DataFrame(index=work.index)
    transformed["gq_01"] = (
        to_num(work[COL_GQ]) / 10.0
    ).clip(0.0, 1.0)

    transformed["igid_norm01"] = minmax_with_frozen(
        work[COL_IGID_UNIFIED],
        normalizer.get("igid_min"),
        normalizer.get("igid_max"),
        bool(normalizer.get("igid_usable", False)),
    )

    transformed["ori_01"] = (
        to_num(work[COL_ORI])
        .clip(0.0, 1.0)
    )
    transformed["purity_01"] = (
        to_num(work[COL_PURITY])
        .clip(0.0, 1.0)
    )

    transformed["bundle_norm01"] = minmax_with_frozen(
        pd.Series(bundle_log, index=work.index),
        normalizer.get("bundle_log_min"),
        normalizer.get("bundle_log_max"),
        bool(normalizer.get("bundle_usable", False)),
    )

    component_columns = {
        "gq": "gq_01",
        "igid": "igid_norm01",
        "ori": "ori_01",
        "purity": "purity_01",
        "bundle": "bundle_norm01",
    }

    components = pd.DataFrame(
        {
            name: transformed[column]
            for name, column in component_columns.items()
        },
        index=work.index,
    )

    weight_series = pd.Series(
        {
            name: float(weights[name])
            for name in component_columns
        },
        dtype=float,
    )

    component_count = components.notna().sum(axis=1).astype(int)

    if missing_policy == "strict":
        available = components.notna().all(axis=1)
        composite = pd.Series(
            np.nan,
            index=work.index,
            dtype=float,
        )
        composite.loc[available] = (
            components.loc[available]
            .mul(weight_series, axis=1)
            .sum(axis=1)
        )
        weight_sum = pd.Series(
            np.nan,
            index=work.index,
            dtype=float,
        )
        weight_sum.loc[available] = float(weight_series.sum())

    elif missing_policy == "renorm":
        availability = components.notna()
        weight_sum = availability.mul(
            weight_series,
            axis=1,
        ).sum(axis=1)
        numerator = components.mul(
            weight_series,
            axis=1,
        ).sum(axis=1, skipna=True)

        composite = pd.Series(
            np.where(
                weight_sum > 0,
                numerator / weight_sum,
                np.nan,
            ),
            index=work.index,
            dtype=float,
        )

    else:
        raise RuntimeError(
            f"未知 Composite missing_policy：{missing_policy}"
        )

    composite = composite.where(
        component_count >= minimum_component_count,
        np.nan,
    )

    transformed[TARGET_COMPOSITE] = composite
    transformed["composite_component_count"] = component_count
    transformed["composite_weight_sum_used"] = weight_sum.where(
        composite.notna(),
        np.nan,
    )

    return transformed, config


# =============================================================================
# 8. L1 Composite 外部验证
# =============================================================================

def validate_l1_composite(
    external_df: pd.DataFrame,
    X_all: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    transformed, _ = build_composite_truth(
        external_df,
        L1_COMPOSITE_NORMALIZATION,
    )

    work = ensure_metadata(external_df)
    y = to_num(transformed[TARGET_COMPOSITE])

    valid = (
        work["data_level"].eq("L1")
        & y.notna()
    )
    indices = np.where(valid.to_numpy())[0]

    if len(indices) == 0:
        metrics = pd.DataFrame([{
            "target": TARGET_COMPOSITE,
            "status": "no_valid_external_samples",
            "n_samples": 0,
            "n_papers": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }])
        return pd.DataFrame(), metrics

    model = load_booster(
        L1_COMPOSITE_MODEL,
        feature_columns,
    )

    y_true = y.iloc[indices].to_numpy(dtype=float)
    y_pred = np.asarray(
        model.predict(X_all.iloc[indices]),
        dtype=float,
    )

    metrics_value = evaluate_regression(y_true, y_pred)

    metrics = pd.DataFrame([{
        "target": TARGET_COMPOSITE,
        "status": "ok",
        "n_samples": metrics_value["n"],
        "n_papers": int(work.iloc[indices]["paper_id"].nunique()),
        "mae": metrics_value["mae"],
        "rmse": metrics_value["rmse"],
        "r2": metrics_value["r2"],
        "spearman": metrics_value["spearman"],
        "model_path": str(L1_COMPOSITE_MODEL),
        "normalization_path": str(L1_COMPOSITE_NORMALIZATION),
    }])

    rows: List[Dict[str, Any]] = []

    for position, index in enumerate(indices):
        item = work.iloc[index]
        component = transformed.iloc[index]

        true_value = float(y_true[position])
        pred_value = float(y_pred[position])

        rows.append({
            **{
                column: item[column]
                for column in META_COLUMNS
                if column in item.index
            },
            "target": TARGET_COMPOSITE,
            "y_true": true_value,
            "y_pred": pred_value,
            "residual": pred_value - true_value,
            "absolute_error": abs(pred_value - true_value),
            "component_count": int(
                component["composite_component_count"]
            ),
            "weight_sum_used": component[
                "composite_weight_sum_used"
            ],
            "gq_01": component["gq_01"],
            "igid_norm01": component["igid_norm01"],
            "ori_01": component["ori_01"],
            "purity_01": component["purity_01"],
            "bundle_norm01": component["bundle_norm01"],
        })

    return pd.DataFrame(rows), metrics


# =============================================================================
# 9. L12 Quantile 外部验证
# =============================================================================

def evaluate_quantiles(
    y_true: np.ndarray,
    q10_raw: np.ndarray,
    q50_raw: np.ndarray,
    q90_raw: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    q10_raw = np.asarray(q10_raw, dtype=float)
    q50_raw = np.asarray(q50_raw, dtype=float)
    q90_raw = np.asarray(q90_raw, dtype=float)

    finite = (
        np.isfinite(y_true)
        & np.isfinite(q10_raw)
        & np.isfinite(q50_raw)
        & np.isfinite(q90_raw)
    )

    y_true = y_true[finite]
    q10_raw = q10_raw[finite]
    q50_raw = q50_raw[finite]
    q90_raw = q90_raw[finite]

    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "pinball_q10": np.nan,
            "pinball_q50": np.nan,
            "pinball_q90": np.nan,
            "empirical_coverage_q10": np.nan,
            "empirical_coverage_q50": np.nan,
            "empirical_coverage_q90": np.nan,
            "q50_mae": np.nan,
            "q50_rmse": np.nan,
            "q50_r2": np.nan,
            "q50_spearman": np.nan,
            "picp_80_ordered": np.nan,
            "mpiw_80_ordered": np.nan,
            "winkler_80_ordered": np.nan,
            "crossing_q10_q50_rate": np.nan,
            "crossing_q50_q90_rate": np.nan,
            "any_crossing_rate": np.nan,
        }

    raw_matrix = np.column_stack(
        [q10_raw, q50_raw, q90_raw]
    )
    ordered = np.sort(raw_matrix, axis=1)

    q10_ordered = ordered[:, 0]
    q90_ordered = ordered[:, 2]

    coverage_q10 = float(np.mean(y_true <= q10_raw))
    coverage_q50 = float(np.mean(y_true <= q50_raw))
    coverage_q90 = float(np.mean(y_true <= q90_raw))

    crossing_10_50 = q10_raw > q50_raw
    crossing_50_90 = q50_raw > q90_raw
    any_crossing = crossing_10_50 | crossing_50_90

    covered = (
        (y_true >= q10_ordered)
        & (y_true <= q90_ordered)
    )
    width = q90_ordered - q10_ordered
    winkler = winkler_score(
        y_true,
        q10_ordered,
        q90_ordered,
    )

    return {
        "n_samples": int(len(y_true)),
        "pinball_q10": pinball_loss(y_true, q10_raw, 0.1),
        "pinball_q50": pinball_loss(y_true, q50_raw, 0.5),
        "pinball_q90": pinball_loss(y_true, q90_raw, 0.9),
        "empirical_coverage_q10": coverage_q10,
        "empirical_coverage_q50": coverage_q50,
        "empirical_coverage_q90": coverage_q90,
        "calibration_error_q10": abs(coverage_q10 - 0.1),
        "calibration_error_q50": abs(coverage_q50 - 0.5),
        "calibration_error_q90": abs(coverage_q90 - 0.9),
        "q50_mae": float(
            mean_absolute_error(y_true, q50_raw)
        ),
        "q50_rmse": float(
            math.sqrt(mean_squared_error(y_true, q50_raw))
        ),
        "q50_r2": safe_r2(y_true, q50_raw),
        "q50_spearman": safe_spearman(y_true, q50_raw),
        "picp_80_ordered": float(np.mean(covered)),
        "picp_80_calibration_error": abs(
            float(np.mean(covered)) - 0.8
        ),
        "mpiw_80_ordered": float(np.mean(width)),
        "winkler_80_ordered": float(np.mean(winkler)),
        "crossing_q10_q50_rate": float(
            np.mean(crossing_10_50)
        ),
        "crossing_q50_q90_rate": float(
            np.mean(crossing_50_90)
        ),
        "any_crossing_rate": float(
            np.mean(any_crossing)
        ),
    }


def validate_l12_quantiles(
    external_df: pd.DataFrame,
    X_all: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    transformed, _ = build_composite_truth(
        external_df,
        L12_NORMALIZATION,
    )

    work = ensure_metadata(external_df)
    y = to_num(transformed[TARGET_COMPOSITE])

    valid = (
        work["data_level"].isin(["L1", "L2"])
        & y.notna()
    )
    indices = np.where(valid.to_numpy())[0]

    if len(indices) == 0:
        return pd.DataFrame(), pd.DataFrame([{
            "scope": "overall",
            "status": "no_valid_external_samples",
            "n_samples": 0,
            "n_papers": 0,
        }])

    q10_model = load_booster(
        L12_Q10_MODEL,
        feature_columns,
    )
    q50_model = load_booster(
        L12_Q50_MODEL,
        feature_columns,
    )
    q90_model = load_booster(
        L12_Q90_MODEL,
        feature_columns,
    )

    X = X_all.iloc[indices]

    q10_raw = np.asarray(
        q10_model.predict(X),
        dtype=float,
    )
    q50_raw = np.asarray(
        q50_model.predict(X),
        dtype=float,
    )
    q90_raw = np.asarray(
        q90_model.predict(X),
        dtype=float,
    )
    y_true = y.iloc[indices].to_numpy(dtype=float)

    ordered = np.sort(
        np.column_stack(
            [q10_raw, q50_raw, q90_raw]
        ),
        axis=1,
    )
    q10_ordered = ordered[:, 0]
    q50_ordered = ordered[:, 1]
    q90_ordered = ordered[:, 2]

    crossing_10_50 = q10_raw > q50_raw
    crossing_50_90 = q50_raw > q90_raw
    any_crossing = crossing_10_50 | crossing_50_90

    covered = (
        (y_true >= q10_ordered)
        & (y_true <= q90_ordered)
    )
    interval_width = q90_ordered - q10_ordered
    winkler = winkler_score(
        y_true,
        q10_ordered,
        q90_ordered,
    )

    prediction_rows: List[Dict[str, Any]] = []

    for position, index in enumerate(indices):
        item = work.iloc[index]
        component = transformed.iloc[index]

        prediction_rows.append({
            **{
                column: item[column]
                for column in META_COLUMNS
                if column in item.index
            },
            "target": TARGET_COMPOSITE,
            "y_true": float(y_true[position]),
            "q10_raw": float(q10_raw[position]),
            "q50_raw": float(q50_raw[position]),
            "q90_raw": float(q90_raw[position]),
            "q10_ordered": float(q10_ordered[position]),
            "q50_ordered": float(q50_ordered[position]),
            "q90_ordered": float(q90_ordered[position]),
            "covered_by_ordered_q10_q90": bool(
                covered[position]
            ),
            "interval_width_ordered": float(
                interval_width[position]
            ),
            "winkler_score_ordered": float(
                winkler[position]
            ),
            "crossing_q10_q50": bool(
                crossing_10_50[position]
            ),
            "crossing_q50_q90": bool(
                crossing_50_90[position]
            ),
            "any_crossing": bool(
                any_crossing[position]
            ),
            "component_count": int(
                component["composite_component_count"]
            ),
            "weight_sum_used": component[
                "composite_weight_sum_used"
            ],
            "gq_01": component["gq_01"],
            "igid_norm01": component["igid_norm01"],
            "ori_01": component["ori_01"],
            "purity_01": component["purity_01"],
            "bundle_norm01": component["bundle_norm01"],
        })

    predictions = pd.DataFrame(prediction_rows)

    metric_rows: List[Dict[str, Any]] = []

    for scope, frame in [
        ("overall", predictions),
        ("L1", predictions[predictions["data_level"] == "L1"]),
        ("L2", predictions[predictions["data_level"] == "L2"]),
    ]:
        if frame.empty:
            metric_rows.append({
                "scope": scope,
                "status": "no_valid_external_samples",
                "n_samples": 0,
                "n_papers": 0,
            })
            continue

        metrics = evaluate_quantiles(
            to_num(frame["y_true"]).to_numpy(dtype=float),
            to_num(frame["q10_raw"]).to_numpy(dtype=float),
            to_num(frame["q50_raw"]).to_numpy(dtype=float),
            to_num(frame["q90_raw"]).to_numpy(dtype=float),
        )

        metric_rows.append({
            "scope": scope,
            "status": "ok",
            "n_papers": int(frame["paper_id"].nunique()),
            **metrics,
        })

    return predictions, pd.DataFrame(metric_rows)


# =============================================================================
# 10. 主程序
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen CNT models on an independent external dataset "
            "without retraining or refitting normalization."
        )
    )

    parser.add_argument(
        "--external_csv",
        default=None,
        help=(
            "External flattened dataset; default: "
            "data/12_external_validation/external_dataset_full60D.csv"
        ),
    )
    parser.add_argument(
        "--train_csv",
        default=None,
        help=(
            "Frozen training dataset for overlap audit; default: "
            "data/11_flattened/ml_dataset_3p0_full60D.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output directory; default: "
            "outputs/evaluation/external_validation"
        ),
    )

    args = parser.parse_args()

    external_csv = resolve_project_path(
        args.external_csv,
        DEFAULT_EXTERNAL_CSV,
    )
    train_csv = resolve_project_path(
        args.train_csv,
        DEFAULT_TRAIN_CSV,
    )
    output_dir = resolve_project_path(
        args.output_dir,
        DEFAULT_OUTPUT_DIR,
    )

    require_file(external_csv, "External dataset")
    require_file(train_csv, "Frozen training dataset")
    require_file(FROZEN_SCHEMA_JSON, "Frozen feature_schema")
    require_file(L1_TARGET_MANIFEST, "L1 model manifest")
    require_file(L1_COMPOSITE_MODEL, "L1 composite model")
    require_file(L1_COMPOSITE_NORMALIZATION, "L1 composite normalization")
    require_file(L12_Q10_MODEL, "L12 q10 model")
    require_file(L12_Q50_MODEL, "L12 q50 model")
    require_file(L12_Q90_MODEL, "L12 q90 model")
    require_file(L12_NORMALIZATION, "L12 normalization")

    output_dir.mkdir(parents=True, exist_ok=True)

    external_df = pd.read_csv(
        external_csv,
        low_memory=False,
    )
    train_df = pd.read_csv(
        train_csv,
        low_memory=False,
    )

    external_df = ensure_metadata(external_df)
    train_df = ensure_metadata(train_df)

    feature_columns = load_frozen_feature_columns()
    X_all = prepare_features(
        external_df,
        feature_columns,
    )

    external_schema_path = (
        external_csv.parent
        / "external_feature_schema.json"
    )
    schema_audit = audit_feature_schema(
        external_schema_path,
        feature_columns,
    )

    if (
        schema_audit["external_schema_exists"]
        and not schema_audit["feature_order_exact_match"]
    ):
        raise RuntimeError(
            "external_feature_schema.json 与冻结训练 feature_schema 的 "
            "model_feature_columns 不完全一致。"
        )

    overlap_df = build_overlap_audit(
        external_df,
        train_df,
    )
    overlap_path = output_dir / "overlap_audit.csv"
    overlap_df.to_csv(
        overlap_path,
        index=False,
        encoding="utf-8-sig",
    )

    dataset_audit = build_dataset_audit(
        external_df,
        feature_columns,
        schema_audit,
        len(overlap_df),
    )
    dataset_audit_path = output_dir / "dataset_audit.csv"
    dataset_audit.to_csv(
        dataset_audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    target_coverage = build_target_coverage(external_df)
    target_coverage_path = output_dir / "target_coverage.csv"
    target_coverage.to_csv(
        target_coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    if len(overlap_df) > 0:
        raise RuntimeError(
            f"发现 {len(overlap_df)} 篇 external paper 与正式训练数据库重叠。"
            f"请先检查：{overlap_path}"
        )

    l1_predictions, l1_metrics = validate_l1_targets(
        external_df,
        X_all,
        feature_columns,
    )
    l1_predictions_path = (
        output_dir / "l1_target_predictions.csv"
    )
    l1_metrics_path = (
        output_dir / "l1_target_metrics.csv"
    )

    l1_predictions.to_csv(
        l1_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    l1_metrics.to_csv(
        l1_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    (
        l1_composite_predictions,
        l1_composite_metrics,
    ) = validate_l1_composite(
        external_df,
        X_all,
        feature_columns,
    )

    l1_composite_predictions_path = (
        output_dir / "l1_composite_predictions.csv"
    )
    l1_composite_metrics_path = (
        output_dir / "l1_composite_metrics.csv"
    )

    l1_composite_predictions.to_csv(
        l1_composite_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    l1_composite_metrics.to_csv(
        l1_composite_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    (
        l12_predictions,
        l12_metrics,
    ) = validate_l12_quantiles(
        external_df,
        X_all,
        feature_columns,
    )

    l12_predictions_path = (
        output_dir / "l12_quantile_predictions.csv"
    )
    l12_metrics_path = (
        output_dir / "l12_quantile_metrics.csv"
    )

    l12_predictions.to_csv(
        l12_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    l12_metrics.to_csv(
        l12_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "external_csv": str(external_csv),
        "frozen_training_csv": str(train_csv),
        "frozen_feature_schema": str(FROZEN_SCHEMA_JSON),
        "output_dir": str(output_dir),
        "external_rows": int(len(external_df)),
        "external_papers": int(external_df["paper_id"].nunique()),
        "external_year_min": (
            int(external_df["paper_year_int"].min())
            if external_df["paper_year_int"].notna().any()
            else None
        ),
        "external_year_max": (
            int(external_df["paper_year_int"].max())
            if external_df["paper_year_int"].notna().any()
            else None
        ),
        "training_overlap_papers": int(len(overlap_df)),
        "feature_count": int(len(feature_columns)),
        "external_feature_schema_exact_match": bool(
            schema_audit["feature_order_exact_match"]
        ),
        "l1_target_prediction_rows": int(len(l1_predictions)),
        "l1_composite_prediction_rows": int(
            len(l1_composite_predictions)
        ),
        "l12_quantile_prediction_rows": int(
            len(l12_predictions)
        ),
        "retraining_performed": False,
        "normalization_refit_on_external_data": False,
        "l1_target_models": str(L1_TARGET_FINAL_DIR),
        "l1_composite_model": str(L1_COMPOSITE_MODEL),
        "l1_composite_normalization": str(
            L1_COMPOSITE_NORMALIZATION
        ),
        "l12_models": {
            "q10": str(L12_Q10_MODEL),
            "q50": str(L12_Q50_MODEL),
            "q90": str(L12_Q90_MODEL),
        },
        "l12_normalization": str(L12_NORMALIZATION),
    }

    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 92)
    print("External Validation with Frozen Models")
    print("=" * 92)
    print(f"[OK] external dataset      : {external_csv}")
    print(f"[OK] external rows         : {len(external_df)}")
    print(
        f"[OK] external papers       : "
        f"{external_df['paper_id'].nunique()}"
    )
    print(
        f"[OK] frozen features       : "
        f"{len(feature_columns)}"
    )
    print(
        f"[OK] schema exact match    : "
        f"{schema_audit['feature_order_exact_match']}"
    )
    print(f"[OK] training overlap      : {len(overlap_df)}")
    print(
        f"[OK] L1 target predictions : "
        f"{len(l1_predictions)}"
    )
    print(
        f"[OK] L1 composite rows     : "
        f"{len(l1_composite_predictions)}"
    )
    print(
        f"[OK] L12 quantile rows     : "
        f"{len(l12_predictions)}"
    )
    print("[OK] retraining            : False")
    print("[OK] normalization refit   : False")
    print(f"[OK] output                : {output_dir}")
    print("=" * 92)


if __name__ == "__main__":
    main()
