# -*- coding: utf-8 -*-
"""
explain_l1_shap.py

CNT 3.0 — formal Composite>=2 paper-grouped out-of-fold TreeSHAP.

Changes relative to the original script
---------------------------------------
1. The five single-target analyses are unchanged.
2. Composite is explicitly selected from outputs/external_validation/composite_sensitivity with:
   minimum_component_count=2, model_family=L1_point and
   split_type=paper_grouped_cv.
3. Existing threshold-specific fold models are reused. If they were not saved,
   only the five required L1 point fold models are rebuilt by importing the
   models/external_validation/composite_sensitivity.py, thereby reusing its frozen split,
   fold-specific normalization, LightGBM settings and early stopping.
4. Main-text block attribution contains Growth quality, IG/ID, Orientation and
   Composite>=2. Purity and bundle-density analyses remain in tables/SI plots.
5. Composite top features are converted through
   promptsV2/CNT_feature_dictionary.txt when available. If the formal dictionary
   is absent or incomplete, all SHAP analyses still run, while the formal
   physical-name top-feature figure is skipped and a mapping template is saved.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_REVISION = '001c_formal_composite_min2_plot_functions_fixed'
GROUPED_SPLIT_NAME = 'paper_grouped_cv'
EXPECTED_GROUP_FOLDS = 5
TARGET_COMPOSITE = 'y_composite_final_score'
POINT_FAMILY = 'L1_point'
COMPOSITE_SHORT = 'composite_min2'
COMPONENT_WEIGHTS: Dict[str, float] = {'gq': 0.30, 'igid': 0.20, 'ori': 0.20, 'purity': 0.15, 'bundle': 0.15}
MODULES = ['N', 'C', 'F', 'P']
PREDICTION_TOLERANCE = 1e-6
SHAP_ADDITIVITY_TOLERANCE = 1e-6
TOP_FEATURES_PER_MODULE = 8
DISPLAY_NAMES = {'gq':'Growth quality','igid':'IG/ID','ori':'Orientation','purity':'Purity','bundle':'Bundle density',COMPOSITE_SHORT:'Composite>=2'}
MAIN_TARGETS = ['gq','igid','ori',COMPOSITE_SHORT]

@dataclass(frozen=True)
class TargetSpec:
    short_name: str
    target_col: str
    model_stem: str
    is_log_target: bool
    source: str

TARGET_SPECS: List[TargetSpec] = [
    TargetSpec('gq','y_growth_quality_score','L1_y_growth_quality_score',False,'single'),
    TargetSpec('igid','y_ig_id_ratio_unified','L1_y_ig_id_ratio_unified',False,'single'),
    TargetSpec('ori','y_orientation_score','L1_y_orientation_score',False,'single'),
    TargetSpec('purity','y_purity_score','L1_y_purity_score',False,'single'),
    TargetSpec('bundle','y_bundle_density_per_um','L1_y_bundle_density_per_um',True,'single'),
    TargetSpec(COMPOSITE_SHORT,TARGET_COMPOSITE,'L1_composite',False,'composite'),
]

def resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (project_dir / path).resolve() if not path.is_absolute() else path.resolve()

def build_paths(project_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(project_dir).resolve(); threshold = int(args.composite_threshold)
    data_dir = root/'data'/'11_flattened'
    single = root/'outputs'/'models'/'l1_targets'
    composite = root/'outputs'/'models'/'l1_composite'
    sensitivity = resolve_path(root, args.sensitivity_dir)
    out = resolve_path(root, args.out_dir or 'outputs/explain/l1_shap')
    saved_models = sensitivity/'validation_models'/f'min_components_{threshold}'/POINT_FAMILY/GROUPED_SPLIT_NAME
    rebuild_root = out/'composite_model_rebuild'
    rebuilt_models = rebuild_root/'validation_models'/f'min_components_{threshold}'/POINT_FAMILY/GROUPED_SPLIT_NAME
    return {
        'project_dir':root,'csv':data_dir/'ml_dataset_3p0_full60D.csv','schema':data_dir/'feature_schema.json',
        'dictionary':resolve_path(root,args.feature_dictionary),'sensitivity_script':resolve_path(root,args.sensitivity_script),
        'single_model_dir':single/'validation_models'/GROUPED_SPLIT_NAME,'single_split':single/'split_manifest.csv',
        'single_predictions':single/'L1_predictions.csv',
        'frozen_composite_split':composite/'composite_split_manifest.csv',
        'sensitivity_dir':sensitivity,'sensitivity_metrics':sensitivity/'composite_sensitivity_metrics_fold.csv',
        'sensitivity_predictions':sensitivity/'composite_sensitivity_predictions.csv','saved_models':saved_models,
        'rebuild_root':rebuild_root,'rebuilt_models':rebuilt_models,'out_dir':out,
        'prepared_split':out/f'composite_min{threshold}_prepared_split_manifest.csv',
        'prepared_predictions':out/f'composite_min{threshold}_prepared_predictions.csv',
        'model_manifest':out/f'composite_min{threshold}_fold_model_manifest.csv',
        'mapping_used':out/'feature_dictionary_mapping_used.csv',
    }

def ensure_file(path: Path, label: str) -> None:
    if not path.is_file(): raise FileNotFoundError(f'{label} not found:\n  {path}')

def read_csv(path: Path) -> pd.DataFrame:
    ensure_file(path,'CSV')
    for encoding in ['utf-8-sig','utf-8','gb18030','gbk','latin1']:
        try: return pd.read_csv(path,encoding=encoding,low_memory=False)
        except UnicodeDecodeError: pass
    raise RuntimeError(f'Cannot decode CSV: {path}')

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False,encoding='utf-8-sig')

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing=[c for c in columns if c not in frame.columns]
    if missing: raise RuntimeError(f'{label} missing columns: {missing}')

def ensure_required_paths(paths: Dict[str,Any]) -> None:
    required={
        'dataset':'csv',
        'schema':'schema',
        'single split':'single_split',
        'single predictions':'single_predictions',
        'frozen Composite split':'frozen_composite_split',
        'sensitivity metrics':'sensitivity_metrics',
        'sensitivity predictions':'sensitivity_predictions',
        'sensitivity script':'sensitivity_script',
    }
    for label,key in required.items():
        ensure_file(paths[key],label)
    if not paths['single_model_dir'].is_dir():
        raise FileNotFoundError(
            f"single model directory not found:\n  {paths['single_model_dir']}"
        )


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float('nan')
    if np.allclose(y_true, y_true[0]):
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

    true_rank = (
        pd.Series(y_true)
        .rank(method='average')
        .to_numpy(dtype=float)
    )
    pred_rank = (
        pd.Series(y_pred)
        .rank(method='average')
        .to_numpy(dtype=float)
    )

    true_std = float(np.std(true_rank))
    pred_std = float(np.std(pred_rank))

    if (
        not np.isfinite(true_std)
        or not np.isfinite(pred_std)
        or np.isclose(true_std, 0.0, rtol=0.0, atol=1e-15)
        or np.isclose(pred_std, 0.0, rtol=0.0, atol=1e-15)
    ):
        return float('nan')

    value = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return value if np.isfinite(value) else float('nan')

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if len(y_true) < 2:
        return {'n': int(len(y_true)), 'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan'), 'spearman': float('nan')}
    return {'n': int(len(y_true)), 'mae': float(mean_absolute_error(y_true, y_pred)), 'rmse': float(math.sqrt(mean_squared_error(y_true, y_pred))), 'r2': safe_r2(y_true, y_pred), 'spearman': safe_spearman(y_true, y_pred)}

def model_output_to_original(raw_prediction: np.ndarray, is_log_target: bool) -> np.ndarray:
    raw_prediction = np.asarray(raw_prediction, dtype=float)
    if not is_log_target:
        return raw_prediction
    original = np.expm1(raw_prediction)
    return np.where(np.isfinite(original) & (original >= 0.0), original, np.nan)

def detect_module(feature_name: str) -> str:
    name = str(feature_name)
    if name.startswith('miss_'):
        base = name[len('miss_'):]
        if len(base) >= 2 and base[0] in MODULES and base[1:].isdigit():
            return base[0]
        return 'other'
    if len(name) >= 2 and name[0] in MODULES and name[1:].isdigit():
        return name[0]
    return 'other'

def is_missing_indicator(feature_name: str) -> bool:
    return str(feature_name).startswith('miss_')

def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('nan')
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float('nan')
    return float(np.dot(a, b) / (norm_a * norm_b))

def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return float('nan')
    if p.sum() <= 0 or q.sum() <= 0:
        return float('nan')
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * (kl(p, m) + kl(q, m))

def load_schema(schema_path: str) -> List[str]:
    with open(schema_path, 'r', encoding='utf-8') as file:
        schema = json.load(file)
    features = schema.get('model_feature_columns')
    if not isinstance(features, list) or len(features) == 0 or (not all((isinstance(feature, str) for feature in features))):
        raise RuntimeError('feature_schema.json 中缺少有效的 model_feature_columns。')
    return features

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    required = ['sample_id', 'paper_id', 'paper_year_int', 'growth_mode', 'data_level']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f'数据集缺少必要元数据列：{missing}')
    df = df[df['data_level'].astype(str) == 'L1'].copy()
    if df.empty:
        raise RuntimeError('筛选 data_level == L1 后没有数据。')
    df['sample_id'] = df['sample_id'].fillna('').astype(str)
    duplicate_ids = df['sample_id'].duplicated(keep=False)
    if duplicate_ids.any():
        examples = df.loc[duplicate_ids, 'sample_id'].head(10).tolist()
        raise RuntimeError(f'L1 数据中 sample_id 不唯一，示例：{examples}')
    return df.set_index('sample_id', drop=False)



def load_validation_tables(paths: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    single_split=read_csv(paths['single_split']); single_predictions=read_csv(paths['single_predictions'])
    for table in [single_split,single_predictions]:
        if 'fold' in table.columns: table['fold']=to_num(table['fold']).astype(int)
        if 'sample_id' in table.columns: table['sample_id']=table['sample_id'].fillna('').astype(str)
        if 'paper_id' in table.columns: table['paper_id']=table['paper_id'].fillna('').astype(str)
    return {'single_split':single_split,'single_predictions':single_predictions}

def load_sensitivity_module(path: Path) -> ModuleType:
    spec=importlib.util.spec_from_file_location('cnt_composite_sensitivity_shap',path)
    if spec is None or spec.loader is None: raise ImportError(f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    needed=['load_dataset','prepare_composite_base','load_split_manifest','split_tasks_from_manifest','ids_to_indices',
            'fit_composite_normalizer','transform_composite','run_validation_task','build_paths','TARGET_NAME','POINT_FAMILY']
    missing=[name for name in needed if not hasattr(module,name)]
    if missing: raise RuntimeError(f'Sensitivity script missing: {missing}')
    return module

def resolve_model_path(value: Any, sensitivity_dir: Path) -> Optional[Path]:
    if value is None or pd.isna(value) or str(value).strip().lower() in {'','nan','none'}: return None
    raw=Path(str(value).strip()).expanduser(); candidates=[raw] if raw.is_absolute() else [raw,sensitivity_dir/raw,sensitivity_dir.parent/raw]
    return next((p.resolve() for p in candidates if p.is_file()),None)

def prepare_composite_inputs(paths: Dict[str,Any], features: Sequence[str], threshold: int, force: bool=False):
    up=load_sensitivity_module(paths['sensitivity_script'])
    if str(up.POINT_FAMILY)!=POINT_FAMILY: raise RuntimeError('Upstream point-family name changed.')
    metrics=read_csv(paths['sensitivity_metrics']); predictions=read_csv(paths['sensitivity_predictions'])
    require_columns(metrics,['minimum_component_count','model_family','split_type','fold','status'],'sensitivity metrics')
    require_columns(predictions,['sample_id','paper_id','minimum_component_count','model_family','split_type','fold','y_true','y_pred','component_count'],'sensitivity predictions')
    mm=to_num(metrics['minimum_component_count']).eq(threshold)&metrics['model_family'].astype(str).eq(POINT_FAMILY)&metrics['split_type'].astype(str).eq(GROUPED_SPLIT_NAME)&metrics['status'].astype(str).eq('ok')
    pm=to_num(predictions['minimum_component_count']).eq(threshold)&predictions['model_family'].astype(str).eq(POINT_FAMILY)&predictions['split_type'].astype(str).eq(GROUPED_SPLIT_NAME)
    metrics=metrics.loc[mm].copy(); predictions=predictions.loc[pm].copy()
    metrics['fold']=to_num(metrics['fold']).astype(int); predictions['fold']=to_num(predictions['fold']).astype(int)
    predictions['sample_id']=predictions['sample_id'].fillna('').astype(str); predictions['paper_id']=predictions['paper_id'].fillna('').astype(str)
    metrics=metrics.sort_values('fold').drop_duplicates('fold',keep='last')
    if len(metrics)!=EXPECTED_GROUP_FOLDS or metrics['fold'].nunique()!=EXPECTED_GROUP_FOLDS: raise RuntimeError('Need exactly five threshold-specific L1 point metric rows.')
    if predictions.duplicated(['fold','sample_id']).any(): raise RuntimeError('Duplicate Composite sensitivity predictions.')
    if not to_num(predictions['component_count']).ge(threshold).all(): raise RuntimeError('Sensitivity predictions violate Composite threshold.')

    dataset=up.load_dataset(str(paths['csv']),list(features)); work,X=up.prepare_composite_base(dataset,['L1'],list(features))
    frozen=up.load_split_manifest(str(paths['frozen_composite_split']),['L1'])
    tasks=sorted([t for t in up.split_tasks_from_manifest(frozen) if str(t['split_type'])==GROUPED_SPLIT_NAME],key=lambda t:int(t['fold']))
    if len(tasks)!=EXPECTED_GROUP_FOLDS: raise RuntimeError('Frozen L1 manifest does not contain five paper-grouped folds.')
    rebuild_paths=up.build_paths(str(paths['project_dir']),str(paths['rebuild_root']))
    metric_by_fold={int(r['fold']):r for _,r in metrics.iterrows()}; split_parts=[]; pred_parts=[]; manifest=[]; model_paths={}
    for pos,task in enumerate(tasks,1):
        fold=int(task['fold']); metric=metric_by_fold[fold]
        train0=up.ids_to_indices(work,task['train_ids'],threshold); test0=up.ids_to_indices(work,task['test_ids'],threshold)
        normalizer=up.fit_composite_normalizer(work,train0); transformed=up.transform_composite(work,normalizer,threshold)
        y=to_num(transformed[up.TARGET_NAME]).to_numpy(float); train=train0[np.isfinite(y[train0])]; test=test0[np.isfinite(y[test0])]
        tr=work.iloc[train]; te=work.iloc[test]; train_ids=tr['sample_id'].astype(str).tolist(); test_ids=te['sample_id'].astype(str).tolist()
        if set(tr['paper_id'].astype(str))&set(te['paper_id'].astype(str)): raise RuntimeError(f'fold={fold}: paper leakage.')
        saved=predictions[predictions['fold'].eq(fold)].copy().set_index('sample_id',drop=False)
        if set(saved.index.astype(str))!=set(test_ids): raise RuntimeError(f'fold={fold}: threshold-specific test IDs differ from saved sensitivity predictions.')
        saved=saved.loc[test_ids].reset_index(drop=True)
        metric_path=resolve_model_path(metric.get('model_path',''),paths['sensitivity_dir'])
        candidates=[metric_path,paths['saved_models']/f'L1_composite_fold{fold}.txt',paths['rebuilt_models']/f'L1_composite_fold{fold}.txt']
        model_path=next((p.resolve() for p in candidates if p is not None and p.is_file()),None)
        source='saved_sensitivity_model'
        if force or model_path is None:
            print(f'[COMPOSITE MODEL {pos}/5] fold={fold} rebuilding L1 point Composite>={threshold}')
            met,pred,_,audit=up.run_validation_task(threshold,POINT_FAMILY,task,work,X,rebuild_paths,True)
            if str(met.get('status'))!='ok': raise RuntimeError(f'fold={fold}: rebuild failed: {audit}')
            model_path=Path(str(met['model_path'])).resolve(); source='auto_rebuilt_from_sensitivity_code'
            pred=pred.copy(); pred['sample_id']=pred['sample_id'].astype(str); pred=pred.set_index('sample_id',drop=False).loc[test_ids]
            diff=float(np.nanmax(np.abs(to_num(pred['y_pred']).to_numpy(float)-to_num(saved['y_pred']).to_numpy(float))))
            if not np.isfinite(diff) or diff>PREDICTION_TOLERANCE: raise RuntimeError(f'fold={fold}: rebuilt predictions differ from sensitivity output; max={diff}')
        else: print(f'[COMPOSITE MODEL {pos}/5] fold={fold} existing model reused')
        model_paths[fold]=str(model_path)
        for role,rows in [('train',tr),('test',te)]:
            part=rows[['sample_id','paper_id','paper_year_int','data_level','growth_mode']].copy(); part['target']=TARGET_COMPOSITE; part['split_type']=GROUPED_SPLIT_NAME; part['fold']=fold; part['role']=role; part['minimum_component_count']=threshold; part['model_family']=POINT_FAMILY; split_parts.append(part)
        saved['target']=TARGET_COMPOSITE; saved['split_type']=GROUPED_SPLIT_NAME; saved['minimum_component_count']=threshold; saved['model_family']=POINT_FAMILY; pred_parts.append(saved)
        manifest.append({'fold':fold,'minimum_component_count':threshold,'model_family':POINT_FAMILY,'split_type':GROUPED_SPLIT_NAME,
                         'n_train':len(train),'n_test':len(test),'n_train_papers':tr['paper_id'].nunique(),'n_test_papers':te['paper_id'].nunique(),
                         'model_path':str(model_path),'model_sha256':sha256_file(model_path),'model_source':source,'train_test_paper_overlap':0})
    split=pd.concat(split_parts,ignore_index=True); pred=pd.concat(pred_parts,ignore_index=True); model_manifest=pd.DataFrame(manifest)
    if split.loc[split['role'].eq('test'),'sample_id'].duplicated().any(): raise RuntimeError('Composite OOF test sample repeated across folds.')
    write_csv(split,paths['prepared_split']); write_csv(pred,paths['prepared_predictions']); write_csv(model_manifest,paths['model_manifest'])
    return split,pred,model_paths,model_manifest


def validate_manifest_structure(split_df: pd.DataFrame, pred_df: pd.DataFrame, spec: TargetSpec) -> List[int]:
    split_required = ['sample_id', 'paper_id', 'target', 'split_type', 'fold', 'role']
    pred_required = ['sample_id', 'paper_id', 'target', 'split_type', 'fold', 'y_true', 'y_pred']
    missing_split = [column for column in split_required if column not in split_df.columns]
    missing_pred = [column for column in pred_required if column not in pred_df.columns]
    if missing_split:
        raise RuntimeError(f'{spec.short_name} split manifest 缺少：{missing_split}')
    if missing_pred:
        raise RuntimeError(f'{spec.short_name} predictions 缺少：{missing_pred}')
    split_target = split_df[(split_df['target'] == spec.target_col) & (split_df['split_type'] == GROUPED_SPLIT_NAME)].copy()
    pred_target = pred_df[(pred_df['target'] == spec.target_col) & (pred_df['split_type'] == GROUPED_SPLIT_NAME)].copy()
    if split_target.empty:
        raise RuntimeError(f'{spec.short_name} 没有 grouped split 记录。')
    if pred_target.empty:
        raise RuntimeError(f'{spec.short_name} 没有 grouped prediction 记录。')
    folds = sorted(split_target['fold'].unique().tolist())
    if len(folds) != EXPECTED_GROUP_FOLDS:
        raise RuntimeError(f'{spec.short_name} grouped folds 数量为 {len(folds)}，预期为 {EXPECTED_GROUP_FOLDS}。')
    for fold in folds:
        split_fold = split_target[split_target['fold'] == fold]
        train_fold = split_fold[split_fold['role'] == 'train']
        test_fold = split_fold[split_fold['role'] == 'test']
        pred_fold = pred_target[pred_target['fold'] == fold]
        if train_fold.empty or test_fold.empty:
            raise RuntimeError(f'{spec.short_name} fold {fold} 缺少 train 或 test。')
        train_papers = set(train_fold['paper_id'].astype(str))
        test_papers = set(test_fold['paper_id'].astype(str))
        overlap = train_papers.intersection(test_papers)
        if overlap:
            raise RuntimeError(f'{spec.short_name} fold {fold} 存在论文泄漏，重叠论文数量：{len(overlap)}')
        test_ids = set(test_fold['sample_id'])
        pred_ids = set(pred_fold['sample_id'])
        if test_ids != pred_ids:
            raise RuntimeError(f'{spec.short_name} fold {fold} 的 test manifest 与 predictions sample_id 不一致。')
        if pred_fold['sample_id'].duplicated().any():
            raise RuntimeError(f'{spec.short_name} fold {fold} predictions 中 sample_id 重复。')
    all_test = split_target[split_target['role'] == 'test']
    if all_test['sample_id'].duplicated().any():
        raise RuntimeError(f'{spec.short_name} grouped OOF 测试样本重复出现。')
    return [int(fold) for fold in folds]



def select_tables_for_target(tables: Dict[str, pd.DataFrame], spec: TargetSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (tables['single_split'],tables['single_predictions']) if spec.source=='single' else (tables['composite_split'],tables['composite_predictions'])

def model_path_for(paths: Dict[str, Any], spec: TargetSpec, fold: int) -> str:
    if spec.source=='single': return str(paths['single_model_dir']/f'{spec.model_stem}_fold{fold}.txt')
    if fold not in paths['composite_model_paths']: raise RuntimeError(f'No Composite model for fold {fold}.')
    return str(paths['composite_model_paths'][fold])


def load_model(model_path: str) -> lgb.Booster:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'模型不存在：{model_path}')
    return lgb.Booster(model_file=model_path)

def align_features(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise RuntimeError(f'数据集缺少模型特征，数量={len(missing)}，示例={missing[:10]}')
    X = df[list(feature_names)].copy()
    for feature in feature_names:
        X[feature] = pd.to_numeric(X[feature], errors='coerce')
    return X

def predict_raw(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)

def predict_contributions(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X, pred_contrib=True), dtype=float)

def verify_prediction_consistency(raw_prediction: np.ndarray, original_prediction: np.ndarray, saved_prediction: np.ndarray, contributions: np.ndarray, spec: TargetSpec, fold: int) -> Dict[str, float]:
    saved_prediction = np.asarray(saved_prediction, dtype=float)
    original_prediction = np.asarray(original_prediction, dtype=float)
    raw_prediction = np.asarray(raw_prediction, dtype=float)
    max_saved_difference = float(np.nanmax(np.abs(original_prediction - saved_prediction)))
    shap_reconstructed = np.sum(contributions, axis=1)
    max_additivity_difference = float(np.nanmax(np.abs(shap_reconstructed - raw_prediction)))
    if not np.isfinite(max_saved_difference) or max_saved_difference > PREDICTION_TOLERANCE:
        raise RuntimeError(f'{spec.short_name} fold {fold} 当前模型预测与已保存 OOF 预测不一致，最大差值={max_saved_difference}')
    if not np.isfinite(max_additivity_difference) or max_additivity_difference > SHAP_ADDITIVITY_TOLERANCE:
        raise RuntimeError(f'{spec.short_name} fold {fold} TreeSHAP 加和校验失败，最大差值={max_additivity_difference}')
    return {'max_saved_prediction_difference': max_saved_difference, 'max_shap_additivity_difference': max_additivity_difference}

def replacement_values(X_train: pd.DataFrame) -> pd.Series:
    replacements: Dict[str, float] = {}
    for column in X_train.columns:
        numeric = pd.to_numeric(X_train[column], errors='coerce')
        finite = numeric[numeric.notna()]
        if is_missing_indicator(column):
            if finite.empty:
                replacements[column] = 0.0
            else:
                modes = finite.mode(dropna=True)
                replacements[column] = float(modes.iloc[0] if not modes.empty else 0.0)
        else:
            replacements[column] = float(finite.median()) if not finite.empty else np.nan
    return pd.Series(replacements, dtype=float)

def module_columns(feature_names: Sequence[str], module: str) -> List[str]:
    return [feature for feature in feature_names if detect_module(feature) == module]

def occlude_columns(X: pd.DataFrame, columns: Sequence[str], replacements: pd.Series) -> pd.DataFrame:
    out = X.copy()
    for column in columns:
        out[column] = replacements.get(column, np.nan)
    return out

def only_missingness_input(X: pd.DataFrame, replacements: pd.Series) -> pd.DataFrame:
    physical_columns = [column for column in X.columns if not is_missing_indicator(column)]
    return occlude_columns(X, physical_columns, replacements)


def analyze_target(df: pd.DataFrame, schema_features: Sequence[str], tables: Dict[str, pd.DataFrame], paths: Dict[str, str], spec: TargetSpec, shap_rows_per_fold: int, seed: int) -> Dict[str, Any]:
    split_df, pred_df = select_tables_for_target(tables, spec)
    folds = validate_manifest_structure(split_df, pred_df, spec)
    split_target = split_df[(split_df['target'] == spec.target_col) & (split_df['split_type'] == GROUPED_SPLIT_NAME)].copy()
    pred_target = pred_df[(pred_df['target'] == spec.target_col) & (pred_df['split_type'] == GROUPED_SPLIT_NAME)].copy()
    fold_metric_rows: List[Dict[str, Any]] = []
    perturbation_rows: List[Dict[str, Any]] = []
    sample_block_rows: List[Dict[str, Any]] = []
    shap_frames: List[pd.DataFrame] = []
    all_oof_true: List[float] = []
    all_oof_pred: List[float] = []
    reference_feature_names: Optional[List[str]] = None
    for fold in folds:
        split_fold = split_target[split_target['fold'] == fold]
        train_ids = split_fold[split_fold['role'] == 'train']['sample_id'].astype(str).tolist()
        test_ids = split_fold[split_fold['role'] == 'test']['sample_id'].astype(str).tolist()
        missing_dataset_ids = [sample_id for sample_id in train_ids + test_ids if sample_id not in df.index]
        if missing_dataset_ids:
            raise RuntimeError(f'{spec.short_name} fold {fold} 有 sample_id 不在主数据集中，示例：{missing_dataset_ids[:10]}')
        pred_fold = pred_target[pred_target['fold'] == fold].copy().set_index('sample_id', drop=False)
        pred_fold = pred_fold.loc[test_ids]
        model_path = model_path_for(paths, spec, fold)
        model = load_model(model_path)
        feature_names = list(model.feature_name())
        if feature_names != list(schema_features):
            raise RuntimeError(f'{spec.short_name} fold {fold} 的模型特征顺序与 feature_schema.json 不一致。')
        if reference_feature_names is None:
            reference_feature_names = feature_names
        elif feature_names != reference_feature_names:
            raise RuntimeError(f'{spec.short_name} 不同 fold 的模型特征顺序不一致。')
        X_train = align_features(df.loc[train_ids], feature_names)
        X_test = align_features(df.loc[test_ids], feature_names)
        raw_prediction = predict_raw(model, X_test)
        original_prediction = model_output_to_original(raw_prediction, spec.is_log_target)
        saved_prediction = pd.to_numeric(pred_fold['y_pred'], errors='coerce').to_numpy(dtype=float)
        y_true = pd.to_numeric(pred_fold['y_true'], errors='coerce').to_numpy(dtype=float)
        full_contributions = predict_contributions(model, X_test)
        if full_contributions.shape[1] != len(feature_names) + 1:
            raise RuntimeError(f'{spec.short_name} fold {fold} TreeSHAP 列数异常。')
        checks = verify_prediction_consistency(raw_prediction=raw_prediction, original_prediction=original_prediction, saved_prediction=saved_prediction, contributions=full_contributions, spec=spec, fold=fold)
        metrics = evaluate_predictions(y_true, original_prediction)
        fold_metric_rows.append({'target': spec.short_name, 'target_col': spec.target_col, 'split_type': GROUPED_SPLIT_NAME, 'fold': fold, 'status': 'ok', 'n_test': metrics['n'], 'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2'], 'spearman': metrics['spearman'], 'model_path': model_path, **checks})
        all_oof_true.extend(y_true.tolist())
        all_oof_pred.extend(original_prediction.tolist())
        replacements = replacement_values(X_train)
        perturbation_record: Dict[str, Any] = {'target': spec.short_name, 'target_col': spec.target_col, 'fold': fold, 'n_test': metrics['n'], 'full_r2': metrics['r2'], 'full_mae': metrics['mae'], 'full_rmse': metrics['rmse'], 'full_spearman': metrics['spearman']}
        for module in MODULES:
            X_drop = occlude_columns(X_test, module_columns(feature_names, module), replacements)
            drop_raw = predict_raw(model, X_drop)
            drop_original = model_output_to_original(drop_raw, spec.is_log_target)
            drop_metrics = evaluate_predictions(y_true, drop_original)
            perturbation_record[f'drop_{module}_r2'] = drop_metrics['r2']
            perturbation_record[f'delta_r2_{module}'] = metrics['r2'] - drop_metrics['r2'] if np.isfinite(metrics['r2']) and np.isfinite(drop_metrics['r2']) else np.nan
        X_missing_only = only_missingness_input(X_test, replacements)
        missing_only_raw = predict_raw(model, X_missing_only)
        missing_only_original = model_output_to_original(missing_only_raw, spec.is_log_target)
        missing_only_metrics = evaluate_predictions(y_true, missing_only_original)
        perturbation_record['missingness_only_r2'] = missing_only_metrics['r2']
        perturbation_record['missingness_only_spearman'] = missing_only_metrics['spearman']
        perturbation_rows.append(perturbation_record)
        shap_test_ids = list(test_ids)
        if shap_rows_per_fold > 0 and len(shap_test_ids) > shap_rows_per_fold:
            rng = np.random.default_rng(seed + fold + sum((ord(character) for character in spec.short_name)))
            selected_positions = rng.choice(len(shap_test_ids), size=shap_rows_per_fold, replace=False)
            selected_positions = np.sort(selected_positions)
            shap_test_ids = [shap_test_ids[position] for position in selected_positions]
        X_shap = X_test.loc[shap_test_ids]
        shap_contributions = predict_contributions(model, X_shap)
        shap_values = shap_contributions[:, :-1]
        shap_frame = pd.DataFrame(shap_values, index=shap_test_ids, columns=feature_names)
        shap_frame['__fold'] = fold
        shap_frames.append(shap_frame)
        for sample_position, sample_id in enumerate(shap_test_ids):
            values = shap_values[sample_position, :]
            abs_values = np.abs(values)
            total_abs = float(np.nansum(abs_values))
            missing_abs = float(np.nansum([abs_values[index] for index, feature in enumerate(feature_names) if is_missing_indicator(feature)]))
            row: Dict[str, Any] = {'sample_id': sample_id, 'paper_id': df.loc[sample_id, 'paper_id'], 'paper_year_int': df.loc[sample_id, 'paper_year_int'], 'growth_mode': df.loc[sample_id, 'growth_mode'], 'target': spec.short_name, 'target_col': spec.target_col, 'fold': fold, 'total_abs_shap': total_abs, 'missing_abs_shap': missing_abs, 'missingness_share': missing_abs / total_abs if total_abs > 0 else np.nan}
            for module in MODULES:
                indices = [index for index, feature in enumerate(feature_names) if detect_module(feature) == module]
                module_abs = float(np.nansum(abs_values[indices]))
                module_signed = float(np.nansum(values[indices]))
                row[f'{module}_abs_shap'] = module_abs
                row[f'{module}_signed_shap'] = module_signed
                row[f'{module}_share'] = module_abs / total_abs if total_abs > 0 else np.nan
            sample_block_rows.append(row)
    pooled_metrics = evaluate_predictions(np.asarray(all_oof_true, dtype=float), np.asarray(all_oof_pred, dtype=float))
    fold_metric_rows.append({'target': spec.short_name, 'target_col': spec.target_col, 'split_type': GROUPED_SPLIT_NAME, 'fold': 'pooled_oof', 'status': 'ok', 'n_test': pooled_metrics['n'], 'mae': pooled_metrics['mae'], 'rmse': pooled_metrics['rmse'], 'r2': pooled_metrics['r2'], 'spearman': pooled_metrics['spearman'], 'model_path': '', 'max_saved_prediction_difference': 0.0, 'max_shap_additivity_difference': 0.0})
    if not shap_frames:
        raise RuntimeError(f'{spec.short_name} 没有可汇总的 OOF TreeSHAP。')
    combined_shap = pd.concat(shap_frames, axis=0)
    fold_values = combined_shap.pop('__fold')
    feature_summary_rows: List[Dict[str, Any]] = []
    for feature in reference_feature_names or []:
        values = pd.to_numeric(combined_shap[feature], errors='coerce').to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            mean_abs = np.nan
            median_abs = np.nan
            mean_signed = np.nan
            positive_fraction = np.nan
            negative_fraction = np.nan
        else:
            mean_abs = float(np.mean(np.abs(finite)))
            median_abs = float(np.median(np.abs(finite)))
            mean_signed = float(np.mean(finite))
            positive_fraction = float(np.mean(finite > 0))
            negative_fraction = float(np.mean(finite < 0))
        feature_summary_rows.append({'target': spec.short_name, 'target_col': spec.target_col, 'feature': feature, 'module': detect_module(feature), 'is_missing_indicator': is_missing_indicator(feature), 'sample_count': int(len(finite)), 'mean_abs_shap': mean_abs, 'median_abs_shap': median_abs, 'mean_signed_shap': mean_signed, 'positive_fraction': positive_fraction, 'negative_fraction': negative_fraction, 'shap_scale': 'log1p_target_scale' if spec.is_log_target else 'original_target_scale'})
    return {'metrics': fold_metric_rows, 'perturbation': perturbation_rows, 'sample_blocks': sample_block_rows, 'feature_summary': feature_summary_rows, 'fold_count': len(folds), 'oof_shap_samples': int(len(combined_shap)), 'feature_count': int(len(reference_feature_names or [])), 'used_folds': [int(fold) for fold in folds], 'shap_fold_counts': {str(fold): int((fold_values == fold).sum()) for fold in folds}}


def build_block_summary(feature_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target, group in feature_summary_df.groupby('target', sort=False):
        total_abs = float(group['mean_abs_shap'].sum(skipna=True))
        physical = group[~group['is_missing_indicator'].astype(bool)]
        physical_total = float(physical['mean_abs_shap'].sum(skipna=True))
        missing_total = float(group.loc[group['is_missing_indicator'].astype(bool), 'mean_abs_shap'].sum(skipna=True))
        row: Dict[str, Any] = {'target': target, 'total_abs_shap': total_abs, 'physical_abs_shap': physical_total, 'missing_abs_shap': missing_total, 'missingness_share': missing_total / total_abs if total_abs > 0 else np.nan}
        for module in MODULES:
            module_group = group[group['module'] == module]
            module_abs = float(module_group['mean_abs_shap'].sum(skipna=True))
            module_physical_abs = float(module_group.loc[~module_group['is_missing_indicator'].astype(bool), 'mean_abs_shap'].sum(skipna=True))
            module_missing_abs = float(module_group.loc[module_group['is_missing_indicator'].astype(bool), 'mean_abs_shap'].sum(skipna=True))
            row[f'{module}_abs_shap'] = module_abs
            row[f'{module}_share'] = module_abs / total_abs if total_abs > 0 else np.nan
            row[f'{module}_physical_abs_shap'] = module_physical_abs
            row[f'{module}_physical_share'] = module_physical_abs / physical_total if physical_total > 0 else np.nan
            row[f'{module}_missing_abs_shap'] = module_missing_abs
        rows.append(row)
    return pd.DataFrame(rows)

def build_perturbation_summary(perturbation_fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target, group in perturbation_fold_df.groupby('target', sort=False):
        row: Dict[str, Any] = {'target': target, 'fold_count': int(len(group)), 'full_r2_mean': group['full_r2'].mean(), 'full_r2_std': group['full_r2'].std(ddof=1) if len(group) > 1 else np.nan, 'missingness_only_r2_mean': group['missingness_only_r2'].mean(), 'missingness_only_r2_std': group['missingness_only_r2'].std(ddof=1) if len(group) > 1 else np.nan, 'missingness_only_spearman_mean': group['missingness_only_spearman'].mean()}
        for module in MODULES:
            row[f'delta_r2_{module}_mean'] = group[f'delta_r2_{module}'].mean()
            row[f'delta_r2_{module}_std'] = group[f'delta_r2_{module}'].std(ddof=1) if len(group) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def build_missingness_reliance(block_summary_df: pd.DataFrame, perturbation_summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = ['target', 'missingness_share']
    left = block_summary_df[columns].copy()
    right = perturbation_summary_df[['target', 'full_r2_mean', 'missingness_only_r2_mean', 'missingness_only_spearman_mean']].copy()
    merged = left.merge(right, on='target', how='inner')
    merged['full_minus_missingness_only_r2'] = merged['full_r2_mean'] - merged['missingness_only_r2_mean']
    return merged



def build_consistency_report(block_summary_df: pd.DataFrame) -> pd.DataFrame:
    indexed=block_summary_df.set_index('target'); required=['gq','igid','ori','purity','bundle',COMPOSITE_SHORT]
    missing=[t for t in required if t not in indexed.index]
    if missing: raise RuntimeError(f'Consistency analysis missing targets: {missing}')
    cols=[f'{m}_share' for m in MODULES]; comp=indexed.loc[COMPOSITE_SHORT,cols].to_numpy(float); rows=[]
    for target in ['gq','igid','ori','purity','bundle']:
        vec=indexed.loc[target,cols].to_numpy(float); rows.append({'comparison':f'{target}_vs_{COMPOSITE_SHORT}','component':target,'cosine_similarity':cosine_similarity(vec,comp),'l1_distance':float(np.nansum(np.abs(vec-comp))),'jensen_shannon_divergence':jensen_shannon_divergence(vec,comp)})
    expected=np.zeros(len(MODULES)); wsum=0.0
    for target,weight in COMPONENT_WEIGHTS.items():
        vec=indexed.loc[target,cols].to_numpy(float)
        if np.isfinite(vec).all(): expected+=weight*vec; wsum+=weight
    if wsum>0: expected/=wsum
    if expected.sum()>0: expected/=expected.sum()
    rows.append({'comparison':f'weighted_expected_vs_direct_{COMPOSITE_SHORT}','component':'weighted_expected','cosine_similarity':cosine_similarity(expected,comp),'l1_distance':float(np.nansum(np.abs(expected-comp))),'jensen_shannon_divergence':jensen_shannon_divergence(expected,comp)})
    for i,module in enumerate(MODULES): rows[-1][f'expected_{module}_share']=float(expected[i]); rows[-1][f'direct_{module}_share']=float(comp[i])
    return pd.DataFrame(rows)

FEATURE_RE=re.compile(r'(?<![A-Za-z0-9])([NCFP]\d{1,2})(?!\d)',re.I)
def load_feature_dictionary(path: Path) -> Dict[str,str]:
    text=path.read_text(encoding='utf-8',errors='replace'); mapping={}; pending=None
    try:
        payload=json.loads(text)
        if isinstance(payload,dict):
            for key,value in payload.items():
                match=FEATURE_RE.search(str(key))
                if match and isinstance(value,(str,int,float)): mapping[match.group(1).upper()]=str(value).strip()
    except json.JSONDecodeError: pass
    for raw in text.splitlines():
        line=raw.strip()
        if not line: continue
        if pending and not FEATURE_RE.search(line): mapping.setdefault(pending,re.sub(r'^[\s:=：,，;；\-–—•*#>|]+','',line).strip()); pending=None; continue
        match=FEATURE_RE.search(line)
        if not match: continue
        code=match.group(1).upper(); name=line[match.end():].strip(' \t:：=,，;；-–—|\"\'`*_')
        if name: mapping.setdefault(code,name)
        else: pending=code
    mapping={k:re.sub(r'\s+',' ',v).strip() for k,v in mapping.items() if str(v).strip()}
    if not mapping: raise RuntimeError(f'No feature definitions parsed from {path}')
    return mapping

def feature_dictionary_template() -> pd.DataFrame:
    rows=[]
    for module,count in [('N',18),('C',12),('F',18),('P',12)]:
        for index in range(1,count+1):
            rows.append({
                'feature_code':f'{module}{index}',
                'physical_name':'',
                'unit':'',
                'definition_or_equation':'',
                'module':module,
            })
    return pd.DataFrame(rows)


def build_top_features(
    feature_summary_df: pd.DataFrame,
    out_dir: Path,
    threshold: int,
    dictionary: Optional[Mapping[str,str]],
    dictionary_path: Path,
) -> Tuple[pd.DataFrame,bool,List[str]]:
    composite=feature_summary_df[
        feature_summary_df['target'].eq(COMPOSITE_SHORT)
        & ~feature_summary_df['is_missing_indicator'].astype(bool)
    ].copy()

    all_rows=[]
    for module in MODULES:
        part=(
            composite[composite['module'].eq(module)]
            .sort_values('mean_abs_shap',ascending=False)
            .head(TOP_FEATURES_PER_MODULE)
            .copy()
        )
        part['rank_within_module']=np.arange(1,len(part)+1)
        all_rows.append(part)

    if not all_rows:
        return pd.DataFrame(),False,[]

    result=pd.concat(all_rows,ignore_index=True)
    result['feature_code']=result['feature'].astype(str)

    dictionary_mapping=dict(dictionary or {})
    result['physical_name']=result['feature_code'].map(dictionary_mapping)
    result['formal_name_available']=(
        result['physical_name'].notna()
        & result['physical_name'].astype(str).str.strip().ne('')
    )
    missing=sorted(
        result.loc[~result['formal_name_available'],'feature_code'].unique()
    )
    mapping_complete=bool(dictionary_path.is_file() and len(missing)==0)

    result['display_label']=np.where(
        result['formal_name_available'],
        result['physical_name'].astype(str)
        +' ('+result['feature_code'].astype(str)+')',
        result['feature_code'].astype(str),
    )
    result['dictionary_source']=(
        str(dictionary_path) if dictionary_path.is_file() else ''
    )

    for module in MODULES:
        write_csv(
            result[result['module'].eq(module)],
            out_dir/f'top_features_{module}_composite_min{threshold}.csv',
        )

    if mapping_complete:
        mapping=(
            result[
                ['feature_code','physical_name','display_label','dictionary_source']
            ]
            .drop_duplicates('feature_code')
            .sort_values('feature_code')
        )
        write_csv(
            mapping,
            out_dir/'feature_dictionary_mapping_used.csv',
        )
    else:
        missing_rows=(
            result.loc[
                ~result['formal_name_available'],
                ['feature_code','module','mean_abs_shap','rank_within_module'],
            ]
            .drop_duplicates('feature_code')
            .sort_values(['module','rank_within_module'])
        )
        write_csv(
            missing_rows,
            out_dir/'feature_dictionary_missing_entries.csv',
        )
        template_path=out_dir/'CNT_feature_dictionary_template.csv'
        if not template_path.is_file():
            write_csv(feature_dictionary_template(),template_path)

    return result,mapping_complete,missing


def ordered(frame: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    table = frame.set_index("target")
    missing = [target for target in order if target not in table.index]
    if missing:
        raise RuntimeError(f"Figure missing targets: {missing}")
    return table.loc[list(order)]


def plot_grouped_performance(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    order = [spec.short_name for spec in TARGET_SPECS]
    pooled = metrics_df[
        metrics_df["fold"].astype(str).eq("pooled_oof")
    ].copy()
    table = ordered(pooled, order)

    x = np.arange(len(table))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        x - width / 2,
        table["r2"].to_numpy(dtype=float),
        width=width,
        label="Pooled OOF R²",
    )
    ax.bar(
        x + width / 2,
        table["spearman"].to_numpy(dtype=float),
        width=width,
        label="Pooled OOF Spearman",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[target] for target in order])
    ax.set_ylabel("Metric value")
    ax.set_title("L1 paper-grouped out-of-fold performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_block_attribution(block_summary_df: pd.DataFrame, output_png: str, order: Sequence[str], title: str, output_pdf: Optional[str]=None) -> None:
    table=ordered(block_summary_df,order); x=np.arange(len(table)); bottom=np.zeros(len(table)); fig,ax=plt.subplots(figsize=(10,6))
    for module in MODULES: values=table[f'{module}_share'].to_numpy(float); ax.bar(x,values,bottom=bottom,label=module); bottom+=values
    ax.plot(x,table['missingness_share'].to_numpy(float),marker='o',linewidth=1.5,label='Missingness share'); ax.set_xticks(x); ax.set_xticklabels([DISPLAY_NAMES[t] for t in order]); ax.set_ylim(0,1.05); ax.set_ylabel('Mean |TreeSHAP| share'); ax.set_title(title); ax.legend(ncol=5,frameon=False); fig.tight_layout(); fig.savefig(output_png,dpi=300);
    if output_pdf: fig.savefig(output_pdf)
    plt.close(fig)

def plot_consistency(consistency_df: pd.DataFrame, output_path: str) -> None:
    labels=consistency_df['component'].astype(str).tolist(); values=consistency_df['cosine_similarity'].to_numpy(float); x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar(x,values); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha='right'); ax.set_ylim(0,1.05); ax.set_ylabel('Cosine similarity'); ax.set_title('Component and Composite>=2 attribution consistency'); fig.tight_layout(); fig.savefig(output_path,dpi=300); plt.close(fig)

def plot_composite_top_features(top_features_df: pd.DataFrame, png: str, pdf: str, threshold: int) -> None:
    plot=top_features_df.sort_values(['module','mean_abs_shap'],ascending=[True,True]).copy(); plot['label']=plot['module'].astype(str)+': '+plot['display_label'].astype(str); height=max(7,0.34*len(plot)); fig,ax=plt.subplots(figsize=(12,height)); y=np.arange(len(plot)); ax.barh(y,plot['mean_abs_shap'].to_numpy(float)); ax.set_yticks(y); ax.set_yticklabels(plot['label'].tolist(),fontsize=8); ax.set_xlabel('Mean |OOF TreeSHAP|'); ax.set_title(f'Composite>={threshold}: top physical features within each module'); fig.tight_layout(); fig.savefig(png,dpi=300,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight'); plt.close(fig)


def plot_block_perturbation(perturbation_summary_df: pd.DataFrame, output_path: str) -> None:
    order = [spec.short_name for spec in TARGET_SPECS]
    table = perturbation_summary_df.set_index('target').loc[order]
    x = np.arange(len(table))
    width = 0.19
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    for index, module in enumerate(MODULES):
        offset = (index - (len(MODULES) - 1) / 2) * width
        ax.bar(x + offset, table[f'delta_r2_{module}_mean'].to_numpy(dtype=float), width=width, label=module)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(t,t) for t in order])
    ax.set_ylabel('ΔR² = full R² − perturbed R²')
    ax.set_title('OOF block perturbation sensitivity')
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)

def plot_missingness_reliance(missingness_df: pd.DataFrame, output_path: str) -> None:
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)
    for _, row in missingness_df.iterrows():
        x = float(row['missingness_share'])
        y = float(row['missingness_only_r2_mean'])
        ax.scatter(x, y)
        ax.text(x + 0.008, y, DISPLAY_NAMES.get(str(row['target']),str(row['target'])), fontsize=9)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel('Missing-indicator share of mean |TreeSHAP|')
    ax.set_ylabel('R² after retaining only missingness pattern')
    ax.set_title('Missingness-reliance audit')
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)



def main() -> None:
    parser=argparse.ArgumentParser(description='CNT 3.0 L1 grouped OOF TreeSHAP with formal Composite>=2')
    parser.add_argument('--project_dir',default=None); parser.add_argument('--out_dir',default=None); parser.add_argument('--composite_threshold',type=int,default=2)
    parser.add_argument('--sensitivity_dir',default='outputs/external_validation/composite_sensitivity'); parser.add_argument('--sensitivity_script',default='models/external_validation/composite_sensitivity.py')
    parser.add_argument('--feature_dictionary',default='promptsV2/CNT_laternt_dictionary.txt'); parser.add_argument('--shap_rows_per_fold',type=int,default=0); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--force_rebuild_composite_models',action='store_true'); args=parser.parse_args()
    if args.composite_threshold<1: parser.error('--composite_threshold must be >=1')
    script_dir=os.path.dirname(os.path.abspath(__file__)); default_project_dir=os.path.abspath(os.path.join(script_dir,'..','..')); project_dir=os.path.abspath(args.project_dir) if args.project_dir else default_project_dir
    threshold=int(args.composite_threshold); DISPLAY_NAMES[COMPOSITE_SHORT]=f'Composite>={threshold}'; paths=build_paths(project_dir,args); ensure_required_paths(paths); paths['out_dir'].mkdir(parents=True,exist_ok=True)
    schema_features=load_schema(str(paths['schema'])); df=load_dataset(str(paths['csv'])); missing=[f for f in schema_features if f not in df.columns]
    if missing: raise RuntimeError(f'Dataset missing schema features: {missing[:10]}')
    tables=load_validation_tables(paths); comp_split,comp_pred,model_paths,model_manifest=prepare_composite_inputs(paths,schema_features,threshold,args.force_rebuild_composite_models)
    tables.update(composite_split=comp_split,composite_predictions=comp_pred); paths['composite_model_paths']=model_paths
    print('='*100); print('[L1 GROUPED OOF SHAP START]'); print(f'Script revision     : {SCRIPT_REVISION}'); print(f'Project directory   : {project_dir}'); print(f'L1 rows             : {len(df)}'); print(f'Feature count       : {len(schema_features)}'); print(f'Composite threshold : {threshold}'); print(f'Composite family    : {POINT_FAMILY}'); print(f'Targets             : {[s.short_name for s in TARGET_SPECS]}'); print(f"SHAP rows/fold      : {args.shap_rows_per_fold or 'all'}"); print(f"Output              : {paths['out_dir']}"); print('='*100)
    all_metrics=[]; all_perturb=[]; all_samples=[]; all_features=[]; target_manifest={}
    for spec in TARGET_SPECS:
        print(f'\n========== ANALYZE: {spec.short_name} =========='); result=analyze_target(df,schema_features,tables,paths,spec,args.shap_rows_per_fold,args.seed)
        all_metrics+=result['metrics']; all_perturb+=result['perturbation']; all_samples+=result['sample_blocks']; all_features+=result['feature_summary']; target_manifest[spec.short_name]={k:result[k] for k in ['fold_count','used_folds','oof_shap_samples','feature_count','shap_fold_counts']}; print(f"[OK] folds={result['fold_count']}, OOF SHAP samples={result['oof_shap_samples']}")
    metrics=pd.DataFrame(all_metrics); perturb=pd.DataFrame(all_perturb); samples=pd.DataFrame(all_samples); features=pd.DataFrame(all_features); blocks=build_block_summary(features); perturb_summary=build_perturbation_summary(perturb); missingness=build_missingness_reliance(blocks,perturb_summary); consistency=build_consistency_report(blocks)
    if paths['dictionary'].is_file():
        dictionary=load_feature_dictionary(paths['dictionary'])
        dictionary_status='formal_dictionary_loaded'
    else:
        dictionary=None
        dictionary_status='formal_dictionary_missing'
        print('\n[WARNING] Formal feature dictionary was not found:')
        print(f"  {paths['dictionary']}")
        print(
            'All OOF SHAP analyses will continue. '
            'The formal physical-name top-feature figure will be skipped.'
        )

    top,dictionary_complete,dictionary_missing=build_top_features(
        features,
        paths['out_dir'],
        threshold,
        dictionary,
        paths['dictionary'],
    )
    outputs={'metrics':paths['out_dir']/'grouped_cv_metrics.csv','features':paths['out_dir']/'oof_feature_shap_summary.csv','blocks':paths['out_dir']/'oof_block_shap_summary.csv','samples':paths['out_dir']/'oof_sample_block_contributions.csv','perturb':paths['out_dir']/'perturbation_report.csv','perturb_summary':paths['out_dir']/'perturbation_summary.csv','missingness':paths['out_dir']/'missingness_reliance.csv','consistency':paths['out_dir']/'component_composite_consistency.csv'}
    for frame,path in [(metrics,outputs['metrics']),(features,outputs['features']),(blocks,outputs['blocks']),(samples,outputs['samples']),(perturb,outputs['perturb']),(perturb_summary,outputs['perturb_summary']),(missingness,outputs['missingness']),(consistency,outputs['consistency'])]: write_csv(frame,path)
    figures={'performance':paths['out_dir']/f'fig1_grouped_performance_min{threshold}.png','all_blocks':paths['out_dir']/f'fig2_oof_block_attribution_all_targets_min{threshold}.png','perturb':paths['out_dir']/f'fig3_block_perturbation_min{threshold}.png','missingness':paths['out_dir']/f'fig4_missingness_reliance_min{threshold}.png','consistency':paths['out_dir']/f'fig5_composite_consistency_min{threshold}.png','main_png':paths['out_dir']/f'fig_oof_block_attribution_min{threshold}.png','main_pdf':paths['out_dir']/f'fig_oof_block_attribution_min{threshold}.pdf','top_png':paths['out_dir']/f'fig_composite_min{threshold}_top_features.png','top_pdf':paths['out_dir']/f'fig_composite_min{threshold}_top_features.pdf'}
    plot_grouped_performance(metrics, str(figures['performance']))
    plot_block_attribution(
        blocks,
        str(figures['all_blocks']),
        [spec.short_name for spec in TARGET_SPECS],
        f'OOF TreeSHAP attribution across all targets; Composite>={threshold}',
    )
    plot_block_perturbation(perturb_summary, str(figures['perturb']))
    plot_missingness_reliance(missingness, str(figures['missingness']))
    plot_consistency(consistency, str(figures['consistency']))
    plot_block_attribution(
        blocks,
        str(figures['main_png']),
        MAIN_TARGETS,
        f'OOF TreeSHAP attribution; formal Composite>={threshold}',
        str(figures['main_pdf']),
    )
    if dictionary_complete:
        plot_composite_top_features(
            top,
            str(figures['top_png']),
            str(figures['top_pdf']),
            threshold,
        )
    else:
        print(
            '[WARNING] Composite top-feature figure skipped because '
            'formal physical names are unavailable.'
        )
        print(
            f"Mapping template              : "
            f"{paths['out_dir'] / 'CNT_feature_dictionary_template.csv'}"
        )
        print(f'Missing plotted features      : {dictionary_missing}')
    pooled=metrics[metrics['target'].eq(COMPOSITE_SHORT)&metrics['fold'].astype(str).eq('pooled_oof')].iloc[0]
    manifest={
        'analysis_name':'CNT 3.0 L1 grouped OOF TreeSHAP',
        'script_revision':SCRIPT_REVISION,
        'project_dir':project_dir,
        'dataset':str(paths['csv']),
        'dataset_sha256':sha256_file(paths['csv']),
        'feature_schema':str(paths['schema']),
        'feature_schema_sha256':sha256_file(paths['schema']),
        'feature_dictionary':(
            str(paths['dictionary']) if paths['dictionary'].is_file() else None
        ),
        'feature_dictionary_sha256':(
            sha256_file(paths['dictionary'])
            if paths['dictionary'].is_file()
            else None
        ),
        'feature_dictionary_status':dictionary_status,
        'feature_dictionary_complete_for_plotted_features':bool(
            dictionary_complete
        ),
        'feature_dictionary_missing_plotted_features':dictionary_missing,
        'formal_top_feature_figure_generated':bool(dictionary_complete),
        'composite':{
            'minimum_component_count':threshold,
            'model_family':POINT_FAMILY,
            'split_type':GROUPED_SPLIT_NAME,
            'prepared_split':str(paths['prepared_split']),
            'prepared_predictions':str(paths['prepared_predictions']),
            'model_manifest':str(paths['model_manifest']),
            'pooled_oof_n':int(pooled['n_test']),
            'pooled_oof_r2':float(pooled['r2']),
            'pooled_oof_spearman':float(pooled['spearman']),
        },
        'targets':target_manifest,
        'main_text_targets':MAIN_TARGETS,
        'component_weights':COMPONENT_WEIGHTS,
        'perturbation_is_retraining_ablation':False,
        'outputs':{
            'tables':{k:str(v) for k,v in outputs.items()},
            'figures':{k:str(v) for k,v in figures.items()},
        },
    }
    manifest_path=paths['out_dir']/'analysis_manifest.json'; manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'+'='*100)
    print('[L1 GROUPED OOF SHAP FINISHED]')
    print(f"Composite pooled OOF n       : {int(pooled['n_test'])}")
    print(f"Composite pooled OOF R2      : {float(pooled['r2']):.6f}")
    print(
        f"Composite pooled Spearman    : "
        f"{float(pooled['spearman']):.6f}"
    )
    print(f"Composite fold models        : {paths['model_manifest']}")
    print(f"Main block figure            : {figures['main_png']}")
    if dictionary_complete:
        print(f"Composite top features       : {figures['top_png']}")
    else:
        print(
            'Composite top features       : SKIPPED '
            '(formal feature dictionary missing/incomplete)'
        )
        print(
            f"Dictionary template          : "
            f"{paths['out_dir']/'CNT_feature_dictionary_template.csv'}"
        )
    print(f'Manifest                     : {manifest_path}')
    print('='*100)

if __name__=='__main__': main()

