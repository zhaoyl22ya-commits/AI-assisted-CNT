# -*- coding: utf-8 -*-
"""
train_l12_quantile.py

CNT 3.0 — L12 composite quantile model

目标
----
联合使用 L1 + L2 数据，训练综合评分的三个条件分位数模型：
    q10：保守低位估计
    q50：中位估计
    q90：高位估计

综合评分
--------
    0.30 × growth_quality
  + 0.20 × IG/ID
  + 0.20 × orientation
  + 0.15 × purity
  + 0.15 × bundle_density

验证
----
1. random_holdout
2. paper_grouped_cv
3. temporal_holdout

关键原则
--------
1. 每个外层 split 只使用外层训练集拟合 IG/ID 与 bundle 的归一化参数；
2. q10/q50/q90 共用完全相同的外层与内部数据划分；
3. 外层测试集不参与 early stopping；
4. q10/q50/q90 使用 pinball loss 与经验覆盖率评价；
5. q50 额外使用 MAE、RMSE、R²、Spearman；
6. q10-q90 区间使用 PICP、MPIW、Winkler score；
7. 保存原始分位数和逐样本排序后的有序分位数，并报告 quantile crossing；
8. 分别评价 L1 与 L2 测试样本；
9. 验证完成后，使用全部有效 L1+L2 数据重新训练三个最终模型。

输入
----
data/11_flattened/ml_dataset_3p0_full60D.csv
data/11_flattened/feature_schema.json

输出
----
outputs/models/l12_quantile/
    final_models/
    validation_models/
    L12_quantile_metrics_fold.csv
    L12_quantile_metrics_summary.csv
    L12_quantile_metrics_by_level.csv
    L12_quantile_predictions.csv
    L12_split_manifest.csv
    L12_normalization_manifest.csv
    L12_component_coverage.csv
    final_L12_composite_normalization.json
    final_model_manifest.csv
    training_summary_L12.json
"""
import json
import math
import os
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', '11_flattened')

CSV_PATH = os.path.join(DATA_DIR, 'ml_dataset_3p0_full60D.csv')
SCHEMA_PATH = os.path.join(DATA_DIR, 'feature_schema.json')
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'l12_quantile')
FINAL_MODEL_DIR = os.path.join(OUT_DIR, 'final_models')
VALIDATION_MODEL_DIR = os.path.join(OUT_DIR, 'validation_models')
METRICS_FOLD_CSV = os.path.join(OUT_DIR, 'L12_quantile_metrics_fold.csv')
METRICS_SUMMARY_CSV = os.path.join(OUT_DIR, 'L12_quantile_metrics_summary.csv')
METRICS_BY_LEVEL_CSV = os.path.join(OUT_DIR, 'L12_quantile_metrics_by_level.csv')
PREDICTIONS_CSV = os.path.join(OUT_DIR, 'L12_quantile_predictions.csv')
SPLIT_MANIFEST_CSV = os.path.join(OUT_DIR, 'L12_split_manifest.csv')
NORMALIZATION_CSV = os.path.join(OUT_DIR, 'L12_normalization_manifest.csv')
COMPONENT_COVERAGE_CSV = os.path.join(OUT_DIR, 'L12_component_coverage.csv')
FINAL_NORMALIZATION_JSON = os.path.join(OUT_DIR, 'final_L12_composite_normalization.json')
FINAL_MODEL_MANIFEST_CSV = os.path.join(OUT_DIR, 'final_model_manifest.csv')
TRAINING_SUMMARY_JSON = os.path.join(OUT_DIR, 'training_summary_L12.json')
TRAIN_LEVELS = ['L1', 'L2']
TARGET_COMPOSITE = 'y_composite_final_score'
SEED = 42
RANDOM_TEST_SIZE = 0.2
INNER_VALID_SIZE = 0.15
GROUP_CV_SPLITS = 5
MIN_TOTAL_SAMPLES = 100
MIN_TOTAL_PAPERS = 10
MIN_OUTER_TRAIN_SAMPLES = 60
MIN_OUTER_TEST_SAMPLES = 20
MIN_TEMPORAL_TRAIN_SAMPLES = 80
MIN_TEMPORAL_TEST_SAMPLES = 25
MIN_TEMPORAL_TRAIN_PAPERS = 8
MIN_TEMPORAL_TEST_PAPERS = 3
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
DEFAULT_FINAL_ITERATIONS = 500
SAVE_VALIDATION_MODELS = True
QUANTILES: Dict[str, float] = {'q10': 0.1, 'q50': 0.5, 'q90': 0.9}
INTERVAL_MIS_COVERAGE = 0.2
WEIGHTS: Dict[str, float] = {'gq': 0.3, 'igid': 0.2, 'ori': 0.2, 'purity': 0.15, 'bundle': 0.15}
MISSING_POLICY = 'renorm'
MIN_COMPONENT_COUNT = 2
COL_GQ = 'y_growth_quality_score'
COL_ORI = 'y_orientation_score'
COL_PURITY = 'y_purity_score'
COL_BUNDLE = 'y_bundle_density_per_um'
COL_IGID_PREF = 'y_ig_id_ratio'
COL_IGID_LEGACY = 'y_IG_ID'
COL_IGID_UNIFIED = 'y_ig_id_ratio_unified'
RAW_GQ = '_raw_gq'
RAW_IGID = '_raw_igid'
RAW_ORI = '_raw_ori'
RAW_PURITY = '_raw_purity'
RAW_BUNDLE = '_raw_bundle'
RAW_BUNDLE_LOG = '_raw_bundle_log1p'
RAW_COMPONENT_COUNT = '_raw_component_count'
COMPONENT_COLUMNS = {'gq': 'gq_01', 'igid': 'igid_norm01', 'ori': 'ori_01', 'purity': 'purity_01', 'bundle': 'bundle_norm01'}

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

def safe_name(text: str) -> str:
    return str(text).replace(':', '_').replace('/', '_').replace('\\', '_').replace(' ', '_')

def build_feature_cols() -> List[str]:
    base = [f'N{i}' for i in range(1, 19)] + [f'C{i}' for i in range(1, 13)] + [f'F{i}' for i in range(1, 19)] + [f'P{i}' for i in range(1, 13)]
    return base + [f'miss_{column}' for column in base]

def load_feature_cols() -> Tuple[List[str], str]:
    if os.path.exists(SCHEMA_PATH):
        try:
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as file:
                schema = json.load(file)
            columns = schema.get('model_feature_columns')
            if isinstance(columns, list) and columns and all((isinstance(column, str) for column in columns)):
                return (columns, 'feature_schema.json')
        except Exception as error:
            print(f'[WARN] feature_schema.json 读取失败，使用代码内定义：{error}')
    return (build_feature_cols(), 'code_fallback')

def ensure_igid_unified(df: pd.DataFrame) -> pd.DataFrame:
    preferred = to_num(df[COL_IGID_PREF]) if COL_IGID_PREF in df.columns else pd.Series(np.nan, index=df.index)
    legacy = to_num(df[COL_IGID_LEGACY]) if COL_IGID_LEGACY in df.columns else pd.Series(np.nan, index=df.index)
    out = df.copy()
    out[COL_IGID_UNIFIED] = preferred.where(preferred.notna(), legacy)
    return out

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return float('nan')
    return float(r2_score(y_true, y_pred))

def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if len(y_true) < 2:
        return float('nan')
    true_rank = pd.Series(y_true).rank(method='average').to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method='average').to_numpy(dtype=float)
    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        return float('nan')
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    loss = np.maximum(alpha * error, (alpha - 1.0) * error)
    return float(np.mean(loss))

def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float=INTERVAL_MIS_COVERAGE) -> np.ndarray:
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

def evaluate_quantiles(y_true: np.ndarray, q10_raw: np.ndarray, q50_raw: np.ndarray, q90_raw: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    q10_raw = np.asarray(q10_raw, dtype=float)
    q50_raw = np.asarray(q50_raw, dtype=float)
    q90_raw = np.asarray(q90_raw, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(q10_raw) & np.isfinite(q50_raw) & np.isfinite(q90_raw)
    y_true = y_true[finite]
    q10_raw = q10_raw[finite]
    q50_raw = q50_raw[finite]
    q90_raw = q90_raw[finite]
    if len(y_true) == 0:
        return {'n': 0, 'pinball_q10': np.nan, 'pinball_q50': np.nan, 'pinball_q90': np.nan, 'empirical_coverage_q10': np.nan, 'empirical_coverage_q50': np.nan, 'empirical_coverage_q90': np.nan, 'calibration_error_q10': np.nan, 'calibration_error_q50': np.nan, 'calibration_error_q90': np.nan, 'q50_mae': np.nan, 'q50_rmse': np.nan, 'q50_r2': np.nan, 'q50_spearman': np.nan, 'picp_80_ordered': np.nan, 'mpiw_80_ordered': np.nan, 'winkler_80_ordered': np.nan, 'crossing_q10_q50_rate': np.nan, 'crossing_q50_q90_rate': np.nan, 'any_crossing_rate': np.nan}
    raw_matrix = np.column_stack([q10_raw, q50_raw, q90_raw])
    ordered = np.sort(raw_matrix, axis=1)
    q10_ordered = ordered[:, 0]
    q50_ordered = ordered[:, 1]
    q90_ordered = ordered[:, 2]
    coverage_q10 = float(np.mean(y_true <= q10_raw))
    coverage_q50 = float(np.mean(y_true <= q50_raw))
    coverage_q90 = float(np.mean(y_true <= q90_raw))
    crossing_10_50 = q10_raw > q50_raw
    crossing_50_90 = q50_raw > q90_raw
    any_crossing = crossing_10_50 | crossing_50_90
    covered = (y_true >= q10_ordered) & (y_true <= q90_ordered)
    width = q90_ordered - q10_ordered
    winkler = winkler_score(y_true, q10_ordered, q90_ordered)
    return {'n': int(len(y_true)), 'pinball_q10': pinball_loss(y_true, q10_raw, 0.1), 'pinball_q50': pinball_loss(y_true, q50_raw, 0.5), 'pinball_q90': pinball_loss(y_true, q90_raw, 0.9), 'empirical_coverage_q10': coverage_q10, 'empirical_coverage_q50': coverage_q50, 'empirical_coverage_q90': coverage_q90, 'calibration_error_q10': abs(coverage_q10 - 0.1), 'calibration_error_q50': abs(coverage_q50 - 0.5), 'calibration_error_q90': abs(coverage_q90 - 0.9), 'q50_mae': float(mean_absolute_error(y_true, q50_raw)), 'q50_rmse': float(math.sqrt(mean_squared_error(y_true, q50_raw))), 'q50_r2': safe_r2(y_true, q50_raw), 'q50_spearman': safe_spearman(y_true, q50_raw), 'q50_ordered_mae': float(mean_absolute_error(y_true, q50_ordered)), 'picp_80_ordered': float(np.mean(covered)), 'picp_80_calibration_error': abs(float(np.mean(covered)) - 0.8), 'mpiw_80_ordered': float(np.mean(width)), 'winkler_80_ordered': float(np.mean(winkler)), 'crossing_q10_q50_rate': float(np.mean(crossing_10_50)), 'crossing_q50_q90_rate': float(np.mean(crossing_50_90)), 'any_crossing_rate': float(np.mean(any_crossing))}

def prepare_base_data(df_all: pd.DataFrame, x_cols: Sequence[str]) -> Dict[str, Any]:
    work = ensure_igid_unified(df_all).copy()
    for column in [COL_GQ, COL_ORI, COL_PURITY, COL_BUNDLE, COL_IGID_UNIFIED]:
        if column not in work.columns:
            work[column] = np.nan
    work[RAW_GQ] = to_num(work[COL_GQ])
    work[RAW_IGID] = to_num(work[COL_IGID_UNIFIED])
    work[RAW_ORI] = to_num(work[COL_ORI])
    work[RAW_PURITY] = to_num(work[COL_PURITY])
    work[RAW_BUNDLE] = to_num(work[COL_BUNDLE])
    bundle_values = work[RAW_BUNDLE].to_numpy(dtype=float)
    work[RAW_BUNDLE_LOG] = np.where(np.isfinite(bundle_values) & (bundle_values >= 0.0), np.log1p(bundle_values), np.nan)
    availability = pd.DataFrame({'gq': work[RAW_GQ].notna(), 'igid': work[RAW_IGID].notna(), 'ori': work[RAW_ORI].notna(), 'purity': work[RAW_PURITY].notna(), 'bundle': work[RAW_BUNDLE_LOG].notna()})
    work[RAW_COMPONENT_COUNT] = availability.sum(axis=1).astype(int)
    work = work[work[RAW_COMPONENT_COUNT] >= MIN_COMPONENT_COUNT].copy()
    work = work.reset_index(drop=True)
    required_metadata = {'sample_id': '', 'paper_id': '', 'paper_year_int': np.nan, 'growth_mode': '', 'data_level': '', 'source_file': '', 'synthesis_id': ''}
    for column, default in required_metadata.items():
        if column not in work.columns:
            work[column] = default
    work['sample_id'] = work['sample_id'].fillna('').astype(str)
    work['paper_id'] = work['paper_id'].fillna('').astype(str)
    work['paper_year_int'] = pd.to_numeric(work['paper_year_int'], errors='coerce')
    if work['sample_id'].duplicated().any():
        examples = work.loc[work['sample_id'].duplicated(keep=False), 'sample_id'].head(10).tolist()
        raise RuntimeError(f'L1/L2 数据中 sample_id 不唯一，示例：{examples}')
    if (work['paper_id'].str.len() == 0).any():
        raise RuntimeError('存在空 paper_id，无法进行 paper-grouped validation。')
    X = work[list(x_cols)].copy()
    for column in x_cols:
        X[column] = pd.to_numeric(X[column], errors='coerce')
    return {'work': work, 'X': X.reset_index(drop=True), 'samples': int(len(work)), 'papers': int(work['paper_id'].nunique()), 'dated_samples': int(work['paper_year_int'].notna().sum())}

def minmax_parameters(series: pd.Series) -> Dict[str, Any]:
    values = to_num(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {'minimum': None, 'maximum': None, 'usable': False, 'finite_count': 0, 'reason': 'no_finite_values'}
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        return {'minimum': minimum, 'maximum': maximum, 'usable': False, 'finite_count': int(len(finite)), 'reason': 'constant_values'}
    return {'minimum': minimum, 'maximum': maximum, 'usable': True, 'finite_count': int(len(finite)), 'reason': 'ok'}

def apply_minmax(series: pd.Series, minimum: Optional[float], maximum: Optional[float], usable: bool) -> pd.Series:
    if not usable or minimum is None or maximum is None or math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        return pd.Series(np.nan, index=series.index, dtype=float)
    normalized = (to_num(series) - minimum) / (maximum - minimum)
    return normalized.clip(lower=0.0, upper=1.0)

def fit_composite_normalizer(work: pd.DataFrame, train_indices: np.ndarray) -> Dict[str, Any]:
    train = work.iloc[np.asarray(train_indices, dtype=int)]
    igid = minmax_parameters(train[RAW_IGID])
    bundle = minmax_parameters(train[RAW_BUNDLE_LOG])
    return {'igid_min': igid['minimum'], 'igid_max': igid['maximum'], 'igid_usable': igid['usable'], 'igid_finite_count': igid['finite_count'], 'igid_reason': igid['reason'], 'bundle_log_min': bundle['minimum'], 'bundle_log_max': bundle['maximum'], 'bundle_usable': bundle['usable'], 'bundle_finite_count': bundle['finite_count'], 'bundle_reason': bundle['reason'], 'fit_sample_count': int(len(train)), 'fit_paper_count': int(train['paper_id'].nunique())}

def transform_composite(work: pd.DataFrame, normalizer: Dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=work.index)
    out['gq_01'] = (to_num(work[RAW_GQ]) / 10.0).clip(0.0, 1.0)
    out['igid_norm01'] = apply_minmax(work[RAW_IGID], normalizer['igid_min'], normalizer['igid_max'], bool(normalizer['igid_usable']))
    out['ori_01'] = to_num(work[RAW_ORI]).clip(0.0, 1.0)
    out['purity_01'] = to_num(work[RAW_PURITY]).clip(0.0, 1.0)
    out['bundle_norm01'] = apply_minmax(work[RAW_BUNDLE_LOG], normalizer['bundle_log_min'], normalizer['bundle_log_max'], bool(normalizer['bundle_usable']))
    components = pd.DataFrame({name: out[column] for name, column in COMPONENT_COLUMNS.items()}, index=work.index)
    weights = pd.Series(WEIGHTS, dtype=float)
    if not np.isfinite(weights.values).all() or weights.sum() <= 0:
        raise ValueError('综合评分权重必须为有限数值，且总和必须大于0。')
    component_count = components.notna().sum(axis=1).astype(int)
    if MISSING_POLICY == 'strict':
        available = components.notna().all(axis=1)
        composite = pd.Series(np.nan, index=work.index, dtype=float)
        composite.loc[available] = components.loc[available].mul(weights, axis=1).sum(axis=1)
        weight_sum = pd.Series(np.nan, index=work.index, dtype=float)
        weight_sum.loc[available] = float(weights.sum())
    elif MISSING_POLICY == 'renorm':
        availability = components.notna()
        weight_sum = availability.mul(weights, axis=1).sum(axis=1)
        numerator = components.mul(weights, axis=1).sum(axis=1, skipna=True)
        composite = pd.Series(np.where(weight_sum > 0, numerator / weight_sum, np.nan), index=work.index, dtype=float)
    else:
        raise ValueError("MISSING_POLICY 必须为 'renorm' 或 'strict'。")
    composite = composite.where(component_count >= MIN_COMPONENT_COUNT, np.nan)
    out[TARGET_COMPOSITE] = composite
    out['composite_component_count'] = component_count
    out['composite_weight_sum_used'] = weight_sum.where(composite.notna(), np.nan)
    return out

def make_inner_validation_split(outer_train_indices: np.ndarray, groups_all: pd.Series) -> Tuple[np.ndarray, np.ndarray, str]:
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    groups = groups_all.iloc[outer_train_indices].fillna('').astype(str).reset_index(drop=True)
    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=INNER_VALID_SIZE, random_state=SEED)
        relative_fit, relative_valid = next(splitter.split(np.zeros(len(outer_train_indices)), groups=groups.to_numpy()))
        return (outer_train_indices[relative_fit], outer_train_indices[relative_valid], 'group_shuffle')
    relative_indices = np.arange(len(outer_train_indices))
    relative_fit, relative_valid = train_test_split(relative_indices, test_size=INNER_VALID_SIZE, random_state=SEED)
    return (outer_train_indices[relative_fit], outer_train_indices[relative_valid], 'random_fallback')

def quantile_params(alpha: float) -> Dict[str, Any]:
    return {'objective': 'quantile', 'metric': 'quantile', 'alpha': alpha, 'learning_rate': 0.03, 'num_leaves': 63, 'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbosity': -1, 'seed': SEED, 'feature_fraction_seed': SEED, 'bagging_seed': SEED, 'data_random_seed': SEED}

def fit_one_quantile(X: pd.DataFrame, y: np.ndarray, fit_indices: np.ndarray, valid_indices: np.ndarray, alpha: float) -> Tuple[lgb.Booster, int]:
    train_set = lgb.Dataset(X.iloc[fit_indices], label=y[fit_indices], free_raw_data=False)
    valid_set = lgb.Dataset(X.iloc[valid_indices], label=y[valid_indices], reference=train_set, free_raw_data=False)
    model = lgb.train(params=quantile_params(alpha), train_set=train_set, num_boost_round=MAX_BOOST_ROUNDS, valid_sets=[train_set, valid_set], valid_names=['inner_fit', 'inner_valid'], callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)])
    best_iteration = int(model.best_iteration if model.best_iteration and model.best_iteration > 0 else MAX_BOOST_ROUNDS)
    return (model, best_iteration)

def save_validation_model(model: lgb.Booster, split_type: str, fold: int, quantile_name: str) -> Optional[str]:
    if not SAVE_VALIDATION_MODELS:
        return None
    directory = os.path.join(VALIDATION_MODEL_DIR, split_type)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'L12_composite_{quantile_name}_fold{fold}.txt')
    model.save_model(path)
    return path

def prepare_outer_split(base: Dict[str, Any], train_indices: np.ndarray, test_indices: np.ndarray) -> Dict[str, Any]:
    work = base['work']
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    normalizer = fit_composite_normalizer(work, train_indices)
    transformed = transform_composite(work, normalizer)
    y = pd.to_numeric(transformed[TARGET_COMPOSITE], errors='coerce').to_numpy(dtype=float)
    train_indices = train_indices[np.isfinite(y[train_indices])]
    test_indices = test_indices[np.isfinite(y[test_indices])]
    return {'normalizer': normalizer, 'transformed': transformed, 'target_values': y, 'train_indices': train_indices, 'test_indices': test_indices}

def outer_split_is_usable(prepared: Dict[str, Any]) -> bool:
    return len(prepared['train_indices']) >= MIN_OUTER_TRAIN_SAMPLES and len(prepared['test_indices']) >= MIN_OUTER_TEST_SAMPLES

def build_prediction_rows(split_type: str, fold: int, test_indices: np.ndarray, work: pd.DataFrame, transformed: pd.DataFrame, y_true: np.ndarray, q10_raw: np.ndarray, q50_raw: np.ndarray, q90_raw: np.ndarray, cutoff: Optional[int]=None) -> List[Dict[str, Any]]:
    raw_matrix = np.column_stack([q10_raw, q50_raw, q90_raw])
    ordered = np.sort(raw_matrix, axis=1)
    q10_ordered = ordered[:, 0]
    q50_ordered = ordered[:, 1]
    q90_ordered = ordered[:, 2]
    crossing_10_50 = q10_raw > q50_raw
    crossing_50_90 = q50_raw > q90_raw
    any_crossing = crossing_10_50 | crossing_50_90
    covered = (y_true >= q10_ordered) & (y_true <= q90_ordered)
    width = q90_ordered - q10_ordered
    winkler = winkler_score(y_true, q10_ordered, q90_ordered)
    rows: List[Dict[str, Any]] = []
    for position, index in enumerate(np.asarray(test_indices, dtype=int)):
        item = work.iloc[index]
        components = transformed.iloc[index]
        rows.append({'sample_id': item['sample_id'], 'paper_id': item['paper_id'], 'paper_year_int': item['paper_year_int'], 'data_level': item['data_level'], 'growth_mode': item['growth_mode'], 'source_file': item['source_file'], 'synthesis_id': item['synthesis_id'], 'target': TARGET_COMPOSITE, 'split_type': split_type, 'fold': fold, 'temporal_cutoff_year': cutoff, 'y_true': float(y_true[position]), 'q10_raw': float(q10_raw[position]), 'q50_raw': float(q50_raw[position]), 'q90_raw': float(q90_raw[position]), 'q10_ordered': float(q10_ordered[position]), 'q50_ordered': float(q50_ordered[position]), 'q90_ordered': float(q90_ordered[position]), 'covered_by_ordered_q10_q90': bool(covered[position]), 'interval_width_ordered': float(width[position]), 'winkler_score_ordered': float(winkler[position]), 'crossing_q10_q50': bool(crossing_10_50[position]), 'crossing_q50_q90': bool(crossing_50_90[position]), 'any_crossing': bool(any_crossing[position]), 'component_count': int(components['composite_component_count']), 'weight_sum_used': float(components['composite_weight_sum_used']), 'gq_01': components['gq_01'], 'igid_norm01': components['igid_norm01'], 'ori_01': components['ori_01'], 'purity_01': components['purity_01'], 'bundle_norm01': components['bundle_norm01']})
    return rows

def build_split_rows(split_type: str, fold: int, train_indices: np.ndarray, test_indices: np.ndarray, work: pd.DataFrame, cutoff: Optional[int]=None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for role, indices in (('train', train_indices), ('test', test_indices)):
        for index in np.asarray(indices, dtype=int):
            item = work.iloc[index]
            rows.append({'sample_id': item['sample_id'], 'paper_id': item['paper_id'], 'paper_year_int': item['paper_year_int'], 'data_level': item['data_level'], 'growth_mode': item['growth_mode'], 'target': TARGET_COMPOSITE, 'split_type': split_type, 'fold': fold, 'role': role, 'temporal_cutoff_year': cutoff})
    return rows

def build_normalization_row(split_type: str, fold: int, normalizer: Dict[str, Any], n_train: int, n_test: int, cutoff: Optional[int]=None) -> Dict[str, Any]:
    return {'split_type': split_type, 'fold': fold, 'temporal_cutoff_year': cutoff, 'n_train': int(n_train), 'n_test': int(n_test), **normalizer}

def run_one_outer_split(base: Dict[str, Any], split_type: str, fold: int, train_indices: np.ndarray, test_indices: np.ndarray, cutoff: Optional[int]=None) -> Dict[str, Any]:
    work = base['work']
    X = base['X']
    prepared = prepare_outer_split(base, train_indices, test_indices)
    if not outer_split_is_usable(prepared):
        return {'status': 'skipped_too_few_valid_composite_targets', 'metrics': [], 'predictions': [], 'splits': [], 'normalizations': [], 'best_iterations': {}, 'model_paths': {}}
    train_indices = prepared['train_indices']
    test_indices = prepared['test_indices']
    y = prepared['target_values']
    fit_indices, valid_indices, inner_method = make_inner_validation_split(train_indices, work['paper_id'])
    models: Dict[str, lgb.Booster] = {}
    best_iterations: Dict[str, int] = {}
    model_paths: Dict[str, Optional[str]] = {}
    for quantile_name, alpha in QUANTILES.items():
        model, best_iteration = fit_one_quantile(X, y, fit_indices, valid_indices, alpha)
        models[quantile_name] = model
        best_iterations[quantile_name] = best_iteration
        model_paths[quantile_name] = save_validation_model(model, split_type, fold, quantile_name)
    q10_raw = np.asarray(models['q10'].predict(X.iloc[test_indices], num_iteration=best_iterations['q10']), dtype=float)
    q50_raw = np.asarray(models['q50'].predict(X.iloc[test_indices], num_iteration=best_iterations['q50']), dtype=float)
    q90_raw = np.asarray(models['q90'].predict(X.iloc[test_indices], num_iteration=best_iterations['q90']), dtype=float)
    y_true = y[test_indices]
    metrics = evaluate_quantiles(y_true, q10_raw, q50_raw, q90_raw)
    metric_row = {'target': TARGET_COMPOSITE, 'split_type': split_type, 'fold': fold, 'status': 'ok', 'n_train': int(len(train_indices)), 'n_test': int(len(test_indices)), 'n_train_papers': int(work.iloc[train_indices]['paper_id'].nunique()), 'n_test_papers': int(work.iloc[test_indices]['paper_id'].nunique()), 'inner_split_method': inner_method, 'best_iteration_q10': best_iterations['q10'], 'best_iteration_q50': best_iterations['q50'], 'best_iteration_q90': best_iterations['q90'], 'model_path_q10': model_paths['q10'], 'model_path_q50': model_paths['q50'], 'model_path_q90': model_paths['q90'], 'temporal_cutoff_year': cutoff, **metrics}
    return {'status': 'ok', 'metrics': [metric_row], 'predictions': build_prediction_rows(split_type, fold, test_indices, work, prepared['transformed'], y_true, q10_raw, q50_raw, q90_raw, cutoff), 'splits': build_split_rows(split_type, fold, train_indices, test_indices, work, cutoff), 'normalizations': [build_normalization_row(split_type, fold, prepared['normalizer'], len(train_indices), len(test_indices), cutoff)], 'best_iterations': best_iterations, 'model_paths': model_paths}

def run_random_validation(base: Dict[str, Any]) -> Dict[str, Any]:
    all_indices = np.arange(len(base['work']))
    train_indices, test_indices = train_test_split(all_indices, test_size=RANDOM_TEST_SIZE, random_state=SEED)
    return run_one_outer_split(base, 'random_holdout', 1, train_indices, test_indices)

def run_grouped_validation(base: Dict[str, Any]) -> Dict[str, Any]:
    work = base['work']
    X = base['X']
    groups = work['paper_id'].astype(str)
    if groups.nunique() < GROUP_CV_SPLITS:
        return {'status': 'skipped_too_few_papers', 'metrics': [], 'predictions': [], 'splits': [], 'normalizations': [], 'best_iterations': {name: [] for name in QUANTILES}, 'fold_statuses': []}
    splitter = GroupKFold(n_splits=GROUP_CV_SPLITS)
    metrics: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    splits: List[Dict[str, Any]] = []
    normalizations: List[Dict[str, Any]] = []
    iterations: Dict[str, List[int]] = {name: [] for name in QUANTILES}
    statuses: List[str] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(X, groups=groups), start=1):
        result = run_one_outer_split(base, 'paper_grouped_cv', fold, train_indices, test_indices)
        statuses.append(result['status'])
        metrics.extend(result.get('metrics', []))
        predictions.extend(result.get('predictions', []))
        splits.extend(result.get('splits', []))
        normalizations.extend(result.get('normalizations', []))
        for name in QUANTILES:
            if name in result.get('best_iterations', {}):
                iterations[name].append(result['best_iterations'][name])
    return {'status': 'ok' if all((status == 'ok' for status in statuses)) else 'partial', 'metrics': metrics, 'predictions': predictions, 'splits': splits, 'normalizations': normalizations, 'best_iterations': iterations, 'fold_statuses': statuses}

def temporal_candidates(work: pd.DataFrame) -> List[int]:
    dated = work[work['paper_year_int'].notna()]
    years = sorted((int(year) for year in dated['paper_year_int'].unique()))
    candidates: List[Tuple[float, int]] = []
    for cutoff in years[:-1]:
        train = dated[dated['paper_year_int'] <= cutoff]
        test = dated[dated['paper_year_int'] > cutoff]
        if len(train) < MIN_TEMPORAL_TRAIN_SAMPLES:
            continue
        if len(test) < MIN_TEMPORAL_TEST_SAMPLES:
            continue
        if train['paper_id'].nunique() < MIN_TEMPORAL_TRAIN_PAPERS:
            continue
        if test['paper_id'].nunique() < MIN_TEMPORAL_TEST_PAPERS:
            continue
        test_fraction = len(test) / (len(train) + len(test))
        candidates.append((abs(test_fraction - RANDOM_TEST_SIZE), cutoff))
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return [int(cutoff) for _, cutoff in candidates]

def run_temporal_validation(base: Dict[str, Any]) -> Dict[str, Any]:
    work = base['work']
    candidates = temporal_candidates(work)
    if not candidates:
        return {'status': 'skipped_too_few_dated_samples', 'metrics': [], 'predictions': [], 'splits': [], 'normalizations': [], 'best_iterations': {}, 'cutoff': None}
    dated = work['paper_year_int'].notna()
    for cutoff in candidates:
        train_indices = np.where((dated & (work['paper_year_int'] <= cutoff)).to_numpy())[0]
        test_indices = np.where((dated & (work['paper_year_int'] > cutoff)).to_numpy())[0]
        result = run_one_outer_split(base, 'temporal_holdout', 1, train_indices, test_indices, cutoff)
        if result['status'] == 'ok':
            result['cutoff'] = cutoff
            result['metrics'][0]['temporal_train_year_max'] = cutoff
            result['metrics'][0]['temporal_test_year_min'] = int(work.iloc[test_indices]['paper_year_int'].min())
            return result
    return {'status': 'skipped_too_few_valid_composite_targets', 'metrics': [], 'predictions': [], 'splits': [], 'normalizations': [], 'best_iterations': {}, 'cutoff': None}

def choose_final_iterations(grouped_iterations: Dict[str, Sequence[int]], random_iterations: Dict[str, int], temporal_iterations: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name in QUANTILES:
        grouped = [int(value) for value in grouped_iterations.get(name, []) if value is not None and value > 0]
        if grouped:
            result[name] = {'iterations': max(1, int(round(median(grouped)))), 'source': 'median_paper_grouped_cv'}
            continue
        fallback = []
        if name in random_iterations and random_iterations[name] > 0:
            fallback.append(int(random_iterations[name]))
        if name in temporal_iterations and temporal_iterations[name] > 0:
            fallback.append(int(temporal_iterations[name]))
        if fallback:
            result[name] = {'iterations': max(1, int(round(median(fallback)))), 'source': 'median_available_validation'}
        else:
            result[name] = {'iterations': DEFAULT_FINAL_ITERATIONS, 'source': 'default'}
    return result

def train_final_models(base: Dict[str, Any], x_cols: Sequence[str], iteration_plan: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    work = base['work']
    X = base['X']
    all_indices = np.arange(len(work))
    normalizer = fit_composite_normalizer(work, all_indices)
    transformed = transform_composite(work, normalizer)
    y = pd.to_numeric(transformed[TARGET_COMPOSITE], errors='coerce').to_numpy(dtype=float)
    valid_indices = np.where(np.isfinite(y))[0]
    if len(valid_indices) < MIN_TOTAL_SAMPLES:
        raise RuntimeError(f'最终综合评分有效样本过少：{len(valid_indices)} < {MIN_TOTAL_SAMPLES}')
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model_rows: List[Dict[str, Any]] = []
    for name, alpha in QUANTILES.items():
        iterations = int(iteration_plan[name]['iterations'])
        train_set = lgb.Dataset(X.iloc[valid_indices], label=y[valid_indices], free_raw_data=False)
        model = lgb.train(params=quantile_params(alpha), train_set=train_set, num_boost_round=iterations, callbacks=[lgb.log_evaluation(period=0)])
        model_path = os.path.join(FINAL_MODEL_DIR, f'L12_composite_{name}.txt')
        model.save_model(model_path)
        model_rows.append({'target': TARGET_COMPOSITE, 'quantile_name': name, 'alpha': alpha, 'status': 'ok', 'samples': int(len(valid_indices)), 'papers': int(work.iloc[valid_indices]['paper_id'].nunique()), 'feature_count': int(len(x_cols)), 'final_iterations': iterations, 'iteration_source': iteration_plan[name]['source'], 'missing_policy': MISSING_POLICY, 'minimum_component_count': MIN_COMPONENT_COUNT, 'model_path': model_path})
    final_normalization = {'target': TARGET_COMPOSITE, 'training_levels': TRAIN_LEVELS, 'normalization_scope': 'all_valid_L1_L2_rows', 'normalizer': normalizer, 'weights': WEIGHTS, 'weight_sum': float(sum(WEIGHTS.values())), 'missing_policy': MISSING_POLICY, 'minimum_component_count': MIN_COMPONENT_COUNT, 'quantiles': QUANTILES, 'component_columns': {'growth_quality': COL_GQ, 'igid_preferred': COL_IGID_PREF, 'igid_legacy': COL_IGID_LEGACY, 'igid_unified': COL_IGID_UNIFIED, 'orientation': COL_ORI, 'purity': COL_PURITY, 'bundle_density': COL_BUNDLE}, 'transformations': {'growth_quality': 'clip(y_growth_quality_score / 10, 0, 1)', 'igid': 'training-scope minmax then clip to [0,1]', 'orientation': 'clip(y_orientation_score, 0, 1)', 'purity': 'clip(y_purity_score, 0, 1)', 'bundle_density': 'negative values -> NaN; nonnegative values -> log1p; training-scope minmax; clip to [0,1]'}, 'final_training_samples': int(len(valid_indices)), 'final_training_papers': int(work.iloc[valid_indices]['paper_id'].nunique())}
    return (model_rows, final_normalization, transformed)

def metrics_from_prediction_frame(frame: pd.DataFrame) -> Dict[str, float]:
    return evaluate_quantiles(pd.to_numeric(frame['y_true'], errors='coerce').to_numpy(dtype=float), pd.to_numeric(frame['q10_raw'], errors='coerce').to_numpy(dtype=float), pd.to_numeric(frame['q50_raw'], errors='coerce').to_numpy(dtype=float), pd.to_numeric(frame['q90_raw'], errors='coerce').to_numpy(dtype=float))

def build_metrics_summary(metrics_fold_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_fold_df.empty:
        return pd.DataFrame()
    metric_columns = ['pinball_q10', 'pinball_q50', 'pinball_q90', 'empirical_coverage_q10', 'empirical_coverage_q50', 'empirical_coverage_q90', 'calibration_error_q10', 'calibration_error_q50', 'calibration_error_q90', 'q50_mae', 'q50_rmse', 'q50_r2', 'q50_spearman', 'picp_80_ordered', 'picp_80_calibration_error', 'mpiw_80_ordered', 'winkler_80_ordered', 'crossing_q10_q50_rate', 'crossing_q50_q90_rate', 'any_crossing_rate']
    rows: List[Dict[str, Any]] = []
    for split_type, group in metrics_fold_df.groupby('split_type', dropna=False):
        pred_group = predictions_df[predictions_df['split_type'] == split_type]
        pooled = metrics_from_prediction_frame(pred_group)
        row: Dict[str, Any] = {'target': TARGET_COMPOSITE, 'split_type': split_type, 'fold_count': int(len(group)), 'total_test_predictions': int(len(pred_group)), 'best_iteration_q10_median': int(round(median(group['best_iteration_q10'].astype(int).tolist()))), 'best_iteration_q50_median': int(round(median(group['best_iteration_q50'].astype(int).tolist()))), 'best_iteration_q90_median': int(round(median(group['best_iteration_q90'].astype(int).tolist()))), 'temporal_cutoff_year': int(group['temporal_cutoff_year'].dropna().iloc[0]) if 'temporal_cutoff_year' in group.columns and (not group['temporal_cutoff_year'].dropna().empty) else ''}
        for column in metric_columns:
            row[f'{column}_mean'] = group[column].mean()
            row[f'{column}_std'] = group[column].std(ddof=1) if len(group) > 1 else np.nan
            row[f'pooled_{column}'] = pooled.get(column, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def build_metrics_by_level(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_type, split_frame in predictions_df.groupby('split_type', sort=False):
        for fold, fold_frame in split_frame.groupby('fold', sort=True):
            for level in TRAIN_LEVELS:
                level_frame = fold_frame[fold_frame['data_level'].astype(str) == level]
                if level_frame.empty:
                    continue
                rows.append({'target': TARGET_COMPOSITE, 'split_type': split_type, 'fold': fold, 'data_level': level, 'aggregation': 'fold', **metrics_from_prediction_frame(level_frame)})
        for level in TRAIN_LEVELS:
            level_frame = split_frame[split_frame['data_level'].astype(str) == level]
            if level_frame.empty:
                continue
            rows.append({'target': TARGET_COMPOSITE, 'split_type': split_type, 'fold': 'pooled', 'data_level': level, 'aggregation': 'pooled', **metrics_from_prediction_frame(level_frame)})
    return pd.DataFrame(rows)

def build_component_coverage(work: pd.DataFrame, final_transformed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for level in ['ALL'] + TRAIN_LEVELS:
        if level == 'ALL':
            mask = pd.Series(True, index=work.index)
        else:
            mask = work['data_level'].astype(str) == level
        subset_total = int(mask.sum())
        for stage, counts in (('raw_components_before_normalization', work.loc[mask, RAW_COMPONENT_COUNT]), ('final_components_after_normalization', final_transformed.loc[mask, 'composite_component_count'])):
            for component_count, count in counts.value_counts(dropna=False).sort_index().items():
                rows.append({'data_level': level, 'stage': stage, 'component_count': component_count, 'sample_count': int(count), 'fraction': float(count) / subset_total if subset_total > 0 else np.nan})
        valid = final_transformed.loc[mask, TARGET_COMPOSITE].notna()
        rows.extend([{'data_level': level, 'stage': 'final_composite_validity', 'component_count': 'valid', 'sample_count': int(valid.sum()), 'fraction': float(valid.mean()) if subset_total > 0 else np.nan}, {'data_level': level, 'stage': 'final_composite_validity', 'component_count': 'invalid', 'sample_count': int((~valid).sum()), 'fraction': float((~valid).mean()) if subset_total > 0 else np.nan}])
    return pd.DataFrame(rows)

def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'CSV 不存在：{CSV_PATH}')
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(VALIDATION_MODEL_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if 'data_level' not in df.columns:
        raise RuntimeError('CSV 缺少 data_level，无法筛选 L1/L2。')
    df = df[df['data_level'].astype(str).isin(TRAIN_LEVELS)].copy()
    if df.empty:
        raise RuntimeError('筛选 data_level ∈ {L1, L2} 后没有数据。')
    x_cols, feature_source = load_feature_cols()
    missing_x = [column for column in x_cols if column not in df.columns]
    if missing_x:
        raise RuntimeError(f'缺少模型特征列 ({len(missing_x)})，示例：{missing_x[:10]}')
    required_metadata = ['sample_id', 'paper_id', 'paper_year_int', 'data_level']
    missing_metadata = [column for column in required_metadata if column not in df.columns]
    if missing_metadata:
        raise RuntimeError(f'当前 CSV 不是新版 flatten 输出，缺少：{missing_metadata}')
    base = prepare_base_data(df, x_cols)
    if base['samples'] < MIN_TOTAL_SAMPLES:
        raise RuntimeError(f"综合评分可用样本过少：{base['samples']} < {MIN_TOTAL_SAMPLES}")
    if base['papers'] < MIN_TOTAL_PAPERS:
        raise RuntimeError(f"综合评分可用论文过少：{base['papers']} < {MIN_TOTAL_PAPERS}")
    level_counts = base['work']['data_level'].astype(str).value_counts().to_dict()
    print('=' * 92)
    print('[L12 QUANTILE VALIDATION START]')
    print(f'CSV                : {CSV_PATH}')
    print(f'Raw L1/L2 rows     : {len(df)}')
    print(f"Composite rows     : {base['samples']}")
    print(f"Composite papers   : {base['papers']}")
    print(f"Dated rows         : {base['dated_samples']}")
    print(f'Level counts       : {level_counts}')
    print(f'Feature source     : {feature_source}')
    print(f'Feature count      : {len(x_cols)}')
    print(f'Quantiles          : {QUANTILES}')
    print(f'Missing policy     : {MISSING_POLICY}')
    print(f'Minimum components : {MIN_COMPONENT_COUNT}')
    print(f'Output             : {OUT_DIR}')
    print('=' * 92)
    random_result = run_random_validation(base)
    grouped_result = run_grouped_validation(base)
    temporal_result = run_temporal_validation(base)
    iteration_plan = choose_final_iterations(grouped_result.get('best_iterations', {}), random_result.get('best_iterations', {}), temporal_result.get('best_iterations', {}))
    final_model_rows, final_normalization, final_transformed = train_final_models(base, x_cols, iteration_plan)
    all_metrics: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    all_splits: List[Dict[str, Any]] = []
    all_normalizations: List[Dict[str, Any]] = []
    for result in (random_result, grouped_result, temporal_result):
        all_metrics.extend(result.get('metrics', []))
        all_predictions.extend(result.get('predictions', []))
        all_splits.extend(result.get('splits', []))
        all_normalizations.extend(result.get('normalizations', []))
    metrics_fold_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    split_df = pd.DataFrame(all_splits)
    normalization_df = pd.DataFrame(all_normalizations)
    metrics_summary_df = build_metrics_summary(metrics_fold_df, predictions_df)
    metrics_by_level_df = build_metrics_by_level(predictions_df)
    coverage_df = build_component_coverage(base['work'], final_transformed)
    metrics_fold_df.to_csv(METRICS_FOLD_CSV, index=False, encoding='utf-8-sig')
    metrics_summary_df.to_csv(METRICS_SUMMARY_CSV, index=False, encoding='utf-8-sig')
    metrics_by_level_df.to_csv(METRICS_BY_LEVEL_CSV, index=False, encoding='utf-8-sig')
    predictions_df.to_csv(PREDICTIONS_CSV, index=False, encoding='utf-8-sig')
    split_df.to_csv(SPLIT_MANIFEST_CSV, index=False, encoding='utf-8-sig')
    normalization_df.to_csv(NORMALIZATION_CSV, index=False, encoding='utf-8-sig')
    coverage_df.to_csv(COMPONENT_COVERAGE_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(final_model_rows).to_csv(FINAL_MODEL_MANIFEST_CSV, index=False, encoding='utf-8-sig')
    with open(FINAL_NORMALIZATION_JSON, 'w', encoding='utf-8') as file:
        json.dump(final_normalization, file, ensure_ascii=False, indent=2)
    training_summary = {'csv_path': CSV_PATH, 'output_dir': OUT_DIR, 'training_levels': TRAIN_LEVELS, 'raw_filtered_rows': int(len(df)), 'base_samples': base['samples'], 'base_papers': base['papers'], 'dated_samples': base['dated_samples'], 'level_counts': level_counts, 'target': TARGET_COMPOSITE, 'feature_source': feature_source, 'feature_count': len(x_cols), 'weights': WEIGHTS, 'missing_policy': MISSING_POLICY, 'minimum_component_count': MIN_COMPONENT_COUNT, 'quantiles': QUANTILES, 'random_seed': SEED, 'random_holdout_status': random_result['status'], 'paper_grouped_cv_status': grouped_result['status'], 'paper_grouped_fold_statuses': grouped_result.get('fold_statuses', []), 'temporal_holdout_status': temporal_result['status'], 'temporal_cutoff_year': temporal_result.get('cutoff'), 'final_iteration_plan': iteration_plan, 'final_models': {row['quantile_name']: row['model_path'] for row in final_model_rows}, 'final_normalization_path': FINAL_NORMALIZATION_JSON, 'interval_definition': {'lower': 'q10', 'center': 'q50', 'upper': 'q90', 'nominal_coverage': 0.8, 'ordered_quantiles_for_interval_evaluation': True, 'raw_quantiles_preserved': True}}
    with open(TRAINING_SUMMARY_JSON, 'w', encoding='utf-8') as file:
        json.dump(training_summary, file, ensure_ascii=False, indent=2)
    print('\n' + '=' * 92)
    print('[L12 QUANTILE VALIDATION FINISHED]')
    print(f"Random validation   : {random_result['status']}")
    print(f"Grouped validation  : {grouped_result['status']}")
    print(f"Temporal validation : {temporal_result['status']}")
    print(f"Final q10 iteration : {iteration_plan['q10']['iterations']}")
    print(f"Final q50 iteration : {iteration_plan['q50']['iterations']}")
    print(f"Final q90 iteration : {iteration_plan['q90']['iterations']}")
    print(f'Fold metrics        : {METRICS_FOLD_CSV}')
    print(f'Summary metrics     : {METRICS_SUMMARY_CSV}')
    print(f'Metrics by level    : {METRICS_BY_LEVEL_CSV}')
    print(f'Predictions         : {PREDICTIONS_CSV}')
    print(f'Split manifest      : {SPLIT_MANIFEST_CSV}')
    print(f'Normalization log   : {NORMALIZATION_CSV}')
    print(f'Component coverage  : {COMPONENT_COVERAGE_CSV}')
    print(f'Final normalization : {FINAL_NORMALIZATION_JSON}')
    print(f'Final models        : {FINAL_MODEL_DIR}')
    print(f'Training summary    : {TRAINING_SUMMARY_JSON}')
    print('=' * 92)
if __name__ == '__main__':
    main()
