# oneclick_core.py
# =============================================================================
# CNT 3.0 OneClick core — single growth mode
#
# Pipeline:
#   catalyst recipes -> L123-001 -> L123-002 -> L123-003 -> final table
#
# Fixed definitions:
#   Balanced=q50; Potential=q90; Conservative=q10.
#   q90-q10 is descriptive only and is never used as a ranking penalty.
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODE_TO_NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
NUM_TO_MODE = {v: k for k, v in MODE_TO_NUM.items()}
UI_COLUMNS = [
    "metal_1", "pct_1", "metal_2", "pct_2", "metal_3", "pct_3"
]
METAL_ALIASES = {
    1: ("metal1", "metal_1"),
    2: ("metal2", "metal_2"),
    3: ("metal3", "metal_3"),
}
FRACTION_ALIASES = {
    1: ("frac1", "frac_1", "pct1", "pct_1"),
    2: ("frac2", "frac_2", "pct2", "pct_2"),
    3: ("frac3", "frac_3", "pct3", "pct_3"),
}


@dataclass
class WorldMapConfig:
    # Frozen assets.
    template_csv: str = "outputs/recommendation/route_templates/route_templates_zpf.csv"

    l12_q10: str = (
        "outputs/models/l12_quantile/"
        "final_models/"
        "L12_composite_q10.txt"
    )

    l12_q50: str = (
        "outputs/models/l12_quantile/"
        "final_models/"
        "L12_composite_q50.txt"
    )

    l12_q90: str = (
        "outputs/models/l12_quantile/"
        "final_models/"
        "L12_composite_q90.txt"
    )

    l1_model: str = (
        "outputs/models/l1_composite/"
        "final_models/"
        "L1_composite_final_score.txt"
    )


    # Frozen scripts.
    l123_001: str = "recommendation/rank_candidate_routes.py"
    l123_002: str = "recommendation/rank_catalysts.py"
    l123_003: str = "recommendation/optimize_physical_state.py"

    # L123-001.
    prior_q: str = "q50"
    topR_balanced: int = 10
    topR_potential: int = 10
    topK_balanced_001: int = 5
    topK_potential_001: int = 5
    prediction_chunk: int = 200000

    # L123-002.
    aggregate_top_n: int = 5
    topK_balanced: int = 5
    topK_potential: int = 5
    topK_conservative: int = 5

    # L123-003.
    np_min: float = 1.0
    np_max: float = 30.0
    np_step: float = 1.0
    cm_min: float = 2.5
    cm_max: float = 28.0
    cm_step: float = 0.5
    topk_points: int = 10

    # Isolated output.
    out_root: str = "outputs/app/oneclick_core"
    run_dir: str | None = None
    overwrite_run_dir: bool = False


# =============================================================================
# Public input helpers
# =============================================================================

def ensure_recipe_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in UI_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out


def parse_manual_recipe(text: str) -> pd.DataFrame:
    parts = [part.strip() for part in (text or "").split(",") if part.strip()]
    if not 1 <= len(parts) <= 3:
        raise ValueError("请输入1–3种金属，例如：Fe 60, Co 40。")

    row = {column: np.nan for column in UI_COLUMNS}
    for index, part in enumerate(parts, start=1):
        pieces = part.split()
        if len(pieces) != 2:
            raise ValueError("每段必须是“元素 比例”，例如：Fe 60。")
        try:
            fraction = float(pieces[1])
        except ValueError as exc:
            raise ValueError(f"比例不是数值：{pieces[1]}") from exc
        row[f"metal_{index}"] = pieces[0]
        row[f"pct_{index}"] = fraction
    return pd.DataFrame([row])


def make_template_csv_bytes(n_blank_rows: int = 10) -> bytes:
    rows = max(1, int(n_blank_rows))
    frame = pd.DataFrame([{column: "" for column in UI_COLUMNS} for _ in range(rows)])
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


# =============================================================================
# Internal helpers
# =============================================================================

def normalize_growth_mode(mode: str | int) -> tuple[str, int]:
    token = str(mode).strip().upper()
    if token in MODE_TO_NUM:
        return token, MODE_TO_NUM[token]
    try:
        number = int(float(token))
    except (TypeError, ValueError) as exc:
        raise ValueError("growth_mode必须是A–E或1–5。") from exc
    if number not in NUM_TO_MODE:
        raise ValueError("growth_mode必须是A–E或1–5。")
    return NUM_TO_MODE[number], number


def _path(project: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (project / path).resolve() if not path.is_absolute() else path.resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：\n  {path}")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{label}缺少字段：{missing}")


def _read_csv(path: Path) -> pd.DataFrame:
    _require_file(path, "CSV")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _canonical_symbol(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    token = str(value).strip()
    if not token or token.upper() in {"NA", "NONE", "NAN"}:
        return None
    return token.upper() if len(token) == 1 else token[0].upper() + token[1:].lower()


def _first_value(row: pd.Series, aliases: Iterable[str]) -> Any:
    for column in aliases:
        if column in row.index and row[column] is not None and not pd.isna(row[column]):
            return row[column]
    return np.nan


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name}必须为正整数。")
    return number


def _validate_grid(low: float, high: float, step: float, name: str) -> None:
    if step <= 0 or high < low:
        raise ValueError(f"{name}范围无效：low={low}, high={high}, step={step}。")


def prepare_recipe_input(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("没有有效催化剂输入。")

    work = ensure_recipe_columns(df)
    records: list[dict[str, Any]] = []

    for row_index, row in work.iterrows():
        metals = [_first_value(row, METAL_ALIASES[i]) for i in (1, 2, 3)]
        fractions = [_first_value(row, FRACTION_ALIASES[i]) for i in (1, 2, 3)]

        if all(_canonical_symbol(value) is None for value in metals):
            continue

        merged: dict[str, float] = {}
        for position, (metal_value, fraction_value) in enumerate(
            zip(metals, fractions), start=1
        ):
            metal = _canonical_symbol(metal_value)
            fraction = pd.to_numeric(pd.Series([fraction_value]), errors="coerce").iloc[0]

            if metal is None and pd.notna(fraction):
                raise ValueError(
                    f"第{row_index + 1}行第{position}个比例有值，但金属名称为空。"
                )
            if metal is not None and pd.isna(fraction):
                raise ValueError(f"第{row_index + 1}行金属{metal}缺少比例。")
            if metal is None:
                continue

            fraction = float(fraction)
            if not np.isfinite(fraction) or fraction <= 0:
                raise ValueError(f"第{row_index + 1}行金属{metal}的比例必须大于0。")
            merged[metal] = merged.get(metal, 0.0) + fraction

        if not merged:
            continue
        if len(merged) > 3:
            raise ValueError(f"第{row_index + 1}行包含超过3种不同金属。")

        total = sum(merged.values())
        recipe = sorted(
            [(metal, value / total) for metal, value in merged.items()],
            key=lambda item: (-item[1], item[0]),
        )
        padded = recipe + [(None, 0.0)] * (3 - len(recipe))
        recipe_key = ";".join(
            f"{metal}:{fraction:.8f}" for metal, fraction in recipe
        )
        catalyst_id = "CAT_" + hashlib.sha1(
            recipe_key.encode("utf-8")
        ).hexdigest()[:12]

        record: dict[str, Any] = {
            "catalyst_id": catalyst_id,
            "recipe_key": recipe_key,
            "metal_set": "+".join(sorted(merged)),
            "num_elements": len(recipe),
            "data_level": "USER",
            "list_source": "cnt_oneclick_core",
            "input_row": int(row_index),
        }
        for position, (metal, fraction) in enumerate(padded, start=1):
            record[f"metal{position}"] = metal
            record[f"frac{position}"] = float(fraction)
        records.append(record)

    if not records:
        raise ValueError("没有解析出有效催化剂配方。")

    out = pd.DataFrame(records).drop_duplicates("recipe_key").reset_index(drop=True)
    if out["catalyst_id"].duplicated().any():
        raise RuntimeError("自动生成的catalyst_id发生冲突。")
    return out


def validate_project_assets(project: Path, cfg: WorldMapConfig) -> dict[str, Path]:
    assets = {
        "templates": _path(project, cfg.template_csv),
        "q10_model": _path(project, cfg.l12_q10),
        "q50_model": _path(project, cfg.l12_q50),
        "q90_model": _path(project, cfg.l12_q90),
        "l1_model": _path(project, cfg.l1_model),
        "l123_001_script": _path(project, cfg.l123_001),
        "l123_002_script": _path(project, cfg.l123_002),
        "l123_003_script": _path(project, cfg.l123_003),
        "znuc_script": project / "recommendation/feature_builders/build_znuc.py",
        "zc_script": project / "recommendation/feature_builders/build_zc.py",
        "elements": project / "promptsV2/elements_properties.txt",
        "priors_by_mode": project / "outputs/recommendation/candidate_priors/priors_by_growth_mode_3p0.csv",
        "priors_global": project / "outputs/recommendation/candidate_priors/priors_global_3p0.csv",
    }
    assets = {key: path.resolve() for key, path in assets.items()}
    for label, path in assets.items():
        _require_file(path, label)
    return assets


def create_run_directory(
    project: Path,
    cfg: WorldMapConfig,
    mode_letter: str,
) -> Path:
    if cfg.run_dir:
        run_dir = _path(project, cfg.run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            if not cfg.overwrite_run_dir:
                raise FileExistsError(f"指定run_dir已经包含文件：\n  {run_dir}")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    root = _path(project, cfg.out_root)
    root.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"run_{tag}_mode_{mode_letter}"
    run_dir.mkdir()
    return run_dir


def _run_stage(
    stage: str,
    command: list[str],
    project: Path,
    log_path: Path,
) -> float:
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        command,
        cwd=project,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 108 + "\n")
        handle.write(f"[{stage}]\nCOMMAND:\n{' '.join(command)}\n\n")
        handle.write(f"STDOUT:\n{result.stdout or ''}\n")
        handle.write(f"STDERR:\n{result.stderr or ''}\n")
        handle.write(f"RETURN CODE: {result.returncode}\nELAPSED SEC: {elapsed:.3f}\n")

    if result.returncode != 0:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        tail = "\n".join(combined.splitlines()[-30:])
        raise RuntimeError(
            f"{stage}运行失败，退出代码{result.returncode}。\n"
            f"日志：{log_path}\n\n最后30行：\n{tail}"
        )
    return elapsed


# =============================================================================
# Output checks
# =============================================================================

def validate_l123_001(path: Path) -> dict[str, Any]:
    df = _read_csv(path)
    _require_columns(
        df,
        [
            "catalyst_id", "template_id", "combination_id",
            "pred_q10", "pred_q50", "pred_q90",
        ],
        "L123-001 shortlist",
    )
    for column in ["pred_q10", "pred_q50", "pred_q90"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["pred_q10", "pred_q50", "pred_q90"]].isna().any().any():
        raise RuntimeError("L123-001输出含有缺失的q10/q50/q90。")
    duplicates = int(df["combination_id"].duplicated().sum())
    crossings = int(
        (
            df["pred_q10"].gt(df["pred_q50"])
            | df["pred_q50"].gt(df["pred_q90"])
            | df["pred_q10"].gt(df["pred_q90"])
        ).sum()
    )
    if duplicates:
        raise RuntimeError("L123-001 shortlist存在重复combination_id。")

    return {
        "rows": int(len(df)),
        "catalysts": int(df["catalyst_id"].nunique()),
        "templates": int(df["template_id"].nunique()),
        "duplicate_combination_ids": duplicates,
        "quantile_crossings": crossings,
    }


def validate_l123_002(paths: dict[str, Path]) -> dict[str, Any]:
    balanced = _read_csv(paths["packages_balanced"])
    potential = _read_csv(paths["packages_potential"])
    conservative = _read_csv(paths["packages_conservative"])
    union = _read_csv(paths["packages_union"])

    for label, frame in [
        ("Balanced packages", balanced),
        ("Potential packages", potential),
        ("Conservative packages", conservative),
        ("q50/q90 union", union),
    ]:
        _require_columns(
            frame,
            ["catalyst_id", "template_id", "combination_id"],
            label,
        )

    balanced_ids = set(balanced["combination_id"].astype(str))
    potential_ids = set(potential["combination_id"].astype(str))
    union_ids = set(union["combination_id"].astype(str))

    if union_ids != balanced_ids | potential_ids:
        raise RuntimeError("L123-002的q50/q90并集与Balanced∪Potential不一致。")
    if union["combination_id"].duplicated().any():
        raise RuntimeError("L123-002并集存在重复combination_id。")

    return {
        "balanced_rows": int(len(balanced)),
        "potential_rows": int(len(potential)),
        "conservative_rows": int(len(conservative)),
        "overlap_rows": int(len(balanced_ids & potential_ids)),
        "union_rows": int(len(union)),
    }


def audit_l123_003(
    recommendations_path: Path,
    union_path: Path,
    cfg: WorldMapConfig,
    audit_path: Path,
) -> dict[str, Any]:
    recommendations = _read_csv(recommendations_path)
    union = _read_csv(union_path)

    _require_columns(
        recommendations,
        [
            "combination_id",
            "recommended_np_size_nm_balanced",
            "recommended_C_over_M_balanced",
            "recommended_np_size_nm_potential",
            "recommended_C_over_M_potential",
        ],
        "L123-003 recommendations",
    )

    recommendation_ids = set(recommendations["combination_id"].astype(str))
    union_ids = set(union["combination_id"].astype(str))
    if recommendation_ids != union_ids:
        raise RuntimeError("L123-003推荐集合与L123-002并集不一致。")
    if recommendations["combination_id"].duplicated().any():
        raise RuntimeError("L123-003推荐存在重复combination_id。")

    id_columns = [
        column for column in [
            "catalyst_id", "template_id", "combination_id",
            "growth_mode", "package_selection_reason",
        ]
        if column in recommendations.columns
    ]
    objective_columns = {
        "balanced": (
            "recommended_np_size_nm_balanced",
            "recommended_C_over_M_balanced",
        ),
        "potential": (
            "recommended_np_size_nm_potential",
            "recommended_C_over_M_potential",
        ),
    }
    np_tol = max(abs(cfg.np_step) * 1e-7, 1e-10)
    cm_tol = max(abs(cfg.cm_step) * 1e-7, 1e-10)
    parts = []

    for objective, (np_column, cm_column) in objective_columns.items():
        part = recommendations[id_columns].copy()
        part["objective"] = objective
        part["recommended_np_size_nm"] = pd.to_numeric(
            recommendations[np_column], errors="coerce"
        )
        part["recommended_C_over_M"] = pd.to_numeric(
            recommendations[cm_column], errors="coerce"
        )
        part["np_at_lower_bound"] = np.isclose(
            part["recommended_np_size_nm"], cfg.np_min, atol=np_tol, rtol=0
        )
        part["np_at_upper_bound"] = np.isclose(
            part["recommended_np_size_nm"], cfg.np_max, atol=np_tol, rtol=0
        )
        part["cm_at_lower_bound"] = np.isclose(
            part["recommended_C_over_M"], cfg.cm_min, atol=cm_tol, rtol=0
        )
        part["cm_at_upper_bound"] = np.isclose(
            part["recommended_C_over_M"], cfg.cm_max, atol=cm_tol, rtol=0
        )
        part["any_boundary"] = part[
            [
                "np_at_lower_bound", "np_at_upper_bound",
                "cm_at_lower_bound", "cm_at_upper_bound",
            ]
        ].any(axis=1)
        parts.append(part)

    audit = pd.concat(parts, ignore_index=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    fractions = audit.groupby("objective")["any_boundary"].mean()
    balanced = float(fractions.get("balanced", np.nan))
    potential = float(fractions.get("potential", np.nan))
    warnings = []
    if np.isfinite(balanced) and balanced >= 0.5:
        warnings.append(
            "Balanced物理状态存在明显边界饱和，不能解释为真实全局最优值。"
        )
    if np.isfinite(potential) and potential >= 0.5:
        warnings.append(
            "Potential物理状态存在明显边界饱和，不能解释为真实全局最优值。"
        )
    return {
        "recommendation_rows": int(len(recommendations)),
        "balanced_boundary_fraction": balanced,
        "potential_boundary_fraction": potential,
        "warnings": warnings,
    }


def merge_final_recommendations(
    comparison_path: Path,
    recommendations_path: Path,
    output_path: Path,
) -> Path:
    comparison = _read_csv(comparison_path)
    recommendations = _read_csv(recommendations_path)
    _require_columns(comparison, ["catalyst_id"], "Catalyst comparison")
    _require_columns(
        recommendations,
        ["catalyst_id", "combination_id"],
        "Physical recommendations",
    )

    extra_columns = [
        column for column in comparison.columns
        if column == "catalyst_id" or column not in recommendations.columns
    ]
    merged = recommendations.merge(
        comparison[extra_columns],
        on="catalyst_id",
        how="left",
        validate="many_to_one",
    )
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


# =============================================================================
# Public pipeline
# =============================================================================

def run_worldmap(
    recipes_df: pd.DataFrame,
    growth_mode: str | int,
    project_dir: str | Path | None = None,
    cfg: WorldMapConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or WorldMapConfig()
    project = (
        Path(project_dir).expanduser().resolve()
        if project_dir is not None else PROJECT_DIR
    )
    mode_letter, mode_number = normalize_growth_mode(growth_mode)

    for value, name in [
        (cfg.topR_balanced, "topR_balanced"),
        (cfg.topR_potential, "topR_potential"),
        (cfg.topK_balanced_001, "topK_balanced_001"),
        (cfg.topK_potential_001, "topK_potential_001"),
        (cfg.aggregate_top_n, "aggregate_top_n"),
        (cfg.topK_balanced, "topK_balanced"),
        (cfg.topK_potential, "topK_potential"),
        (cfg.topK_conservative, "topK_conservative"),
        (cfg.topk_points, "topk_points"),
        (cfg.prediction_chunk, "prediction_chunk"),
    ]:
        _positive_int(value, name)

    _validate_grid(cfg.np_min, cfg.np_max, cfg.np_step, "NP网格")
    _validate_grid(cfg.cm_min, cfg.cm_max, cfg.cm_step, "C/M网格")
    if cfg.prior_q not in {"q10", "q50", "q90"}:
        raise ValueError("prior_q必须是q10、q50或q90。")

    assets = validate_project_assets(project, cfg)
    recipes = prepare_recipe_input(recipes_df)
    run_dir = create_run_directory(project, cfg, mode_letter)

    input_dir = run_dir / "input"
    out_001 = run_dir / "rank_candidate_routes"
    out_002 = run_dir / "rank_catalysts"
    out_003 = run_dir / "optimize_physical_state"
    audit_dir = run_dir / "audit"
    for directory in [input_dir, out_001, out_002, out_003, audit_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    input_csv = input_dir / "catalyst_recipes.csv"
    log_path = run_dir / "run_log.txt"
    manifest_path = run_dir / "run_manifest.json"
    recipes.to_csv(input_csv, index=False, encoding="utf-8-sig")

    manifest: dict[str, Any] = {
        "status": "running",
        "project_dir": str(project),
        "run_dir": str(run_dir),
        "growth_mode_letter": mode_letter,
        "growth_mode_number": mode_number,
        "config": asdict(cfg),
        "input": {"input_csv": str(input_csv), "recipe_rows": int(len(recipes))},
        "definitions": {
            "balanced": "raw q50",
            "potential": "raw q90",
            "conservative": "raw q10",
            "interval_width": "q90-q10; descriptive only",
            "physical_search": "np_size_nm and raw C_over_M",
        },
        "assets": {key: str(value) for key, value in assets.items()},
        "stages": {},
    }
    _write_json(manifest, manifest_path)

    try:
        command_001 = [
            sys.executable, str(assets["l123_001_script"]),
            "--candidates_csv", str(input_csv),
            "--templates_csv", str(assets["templates"]),
            "--prior_q", cfg.prior_q,
            "--candidate_route_mode", "expand",
            "--growth_mode", mode_letter,
            "--L12_q10", str(assets["q10_model"]),
            "--L12_q50", str(assets["q50_model"]),
            "--L12_q90", str(assets["q90_model"]),
            "--L1_model", str(assets["l1_model"]),
            "--outdir", str(out_001),
            "--topR_balanced", str(cfg.topR_balanced),
            "--topR_potential", str(cfg.topR_potential),
            "--topK_balanced", str(cfg.topK_balanced_001),
            "--topK_potential", str(cfg.topK_potential_001),
            "--prediction_chunk", str(cfg.prediction_chunk),
        ]
        manifest["stages"]["L123_001_seconds"] = _run_stage(
            "L123-001", command_001, project, log_path
        )
        shortlist = out_001 / "shortlisted_route_predictions.csv"
        _require_file(shortlist, "L123-001 shortlist")
        audit_001 = validate_l123_001(shortlist)

        command_002 = [
            sys.executable, str(assets["l123_002_script"]),
            "--input_csv", str(shortlist),
            "--outdir", str(out_002),
            "--aggregate_top_n", str(cfg.aggregate_top_n),
            "--topK_balanced", str(cfg.topK_balanced),
            "--topK_potential", str(cfg.topK_potential),
            "--topK_conservative", str(cfg.topK_conservative),
        ]
        manifest["stages"]["L123_002_seconds"] = _run_stage(
            "L123-002", command_002, project, log_path
        )

        stage_002 = {
            "catalyst_balanced": out_002 / "catalyst_rank_balanced.csv",
            "catalyst_potential": out_002 / "catalyst_rank_potential.csv",
            "catalyst_conservative": out_002 / "catalyst_rank_conservative.csv",
            "catalyst_comparison": out_002 / "catalyst_rank_comparison.csv",
            "packages_balanced": out_002 / "process_packages_balanced_topK.csv",
            "packages_potential": out_002 / "process_packages_potential_topK.csv",
            "packages_conservative":
                out_002 / "process_packages_conservative_topK.csv",
            "packages_union": out_002 / "process_packages_union_q50_q90.csv",
            "summary_002": out_002 / "catalyst_summary_L123_002.json",
        }
        for label, path in stage_002.items():
            _require_file(path, label)
        audit_002 = validate_l123_002(stage_002)

        command_003 = [
            sys.executable, str(assets["l123_003_script"]),
            "--input_csv", str(stage_002["packages_union"]),
            "--outdir", str(out_003),
            "--q10_model", str(assets["q10_model"]),
            "--q50_model", str(assets["q50_model"]),
            "--q90_model", str(assets["q90_model"]),
            "--np_min", str(cfg.np_min),
            "--np_max", str(cfg.np_max),
            "--np_step", str(cfg.np_step),
            "--cm_min", str(cfg.cm_min),
            "--cm_max", str(cfg.cm_max),
            "--cm_step", str(cfg.cm_step),
            "--topk_points", str(cfg.topk_points),
        ]
        manifest["stages"]["L123_003_seconds"] = _run_stage(
            "L123-003", command_003, project, log_path
        )

        recommendations = out_003 / "process_packages_with_physical_state.csv"
        top_points = out_003 / "physical_state_top_points.csv"
        summary_003 = out_003 / "run_summary_L123_003.json"
        for label, path in [
            ("Physical recommendations", recommendations),
            ("Physical top points", top_points),
            ("L123-003 summary", summary_003),
        ]:
            _require_file(path, label)

        boundary_audit = audit_dir / "L123_003_boundary_audit.csv"
        audit_003 = audit_l123_003(
            recommendations, stage_002["packages_union"], cfg, boundary_audit
        )

        world_final = run_dir / "WORLD_recommendations_final.csv"
        merge_final_recommendations(
            stage_002["catalyst_comparison"], recommendations, world_final
        )

        paths: dict[str, Path] = {
            "run_dir": run_dir,
            "input_recipes": input_csv,
            "run_log": log_path,
            "run_manifest": manifest_path,
            "shortlist": shortlist,
            **stage_002,
            "physical_recommendations": recommendations,
            "physical_top_points": top_points,
            "summary_003": summary_003,
            "boundary_audit": boundary_audit,
            "world_final": world_final,
        }
        table_keys = [
            "catalyst_balanced", "catalyst_potential",
            "catalyst_conservative", "catalyst_comparison",
            "packages_balanced", "packages_potential",
            "packages_conservative", "packages_union",
            "physical_recommendations", "physical_top_points",
            "boundary_audit", "world_final",
        ]
        tables = {key: _read_csv(paths[key]) for key in table_keys}

        manifest["status"] = "success"
        manifest["audit"] = {
            "L123_001": audit_001,
            "L123_002": audit_002,
            "L123_003": audit_003,
        }
        manifest["outputs"] = {key: str(value) for key, value in paths.items()}
        _write_json(manifest, manifest_path)

        return {
            "status": "success",
            "growth_mode_letter": mode_letter,
            "growth_mode_num": mode_number,
            "run_dir": str(run_dir),
            "paths": {key: str(value) for key, value in paths.items()},
            "tables": tables,
            "audit": {
                "L123_001": audit_001,
                "L123_002": audit_002,
                "L123_003": audit_003,
            },
            "config": asdict(cfg),
        }

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        _write_json(manifest, manifest_path)
        raise
