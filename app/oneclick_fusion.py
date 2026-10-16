# oneclick_fusion.py
# =============================================================================
# CNT 3.0 OneClick fusion — multi-growth-mode catalyst ranking
#
# Each selected mode calls oneclick_core.run_worldmap().
# Balanced=q50, Potential=q90, Conservative=q10.
# Default fusion uses within-mode rank percentiles rather than raw scores.
# Missing modes are tracked explicitly and are never silently filled with zero.
# =============================================================================

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from oneclick_core import WorldMapConfig, run_worldmap


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODE_ORDER = ["A", "B", "C", "D", "E"]
CRITERIA = ["balanced", "potential", "conservative"]
PROFILE_COLUMNS = [
    "recipe_key", "metal_set",
    "metal1", "metal2", "metal3",
    "frac1", "frac2", "frac3",
]


@dataclass
class FusionConfig:
    strategy: str = "rank_percentile"
    min_coverage_fraction: float = 1.0
    out_root: str = "outputs/app/oneclick_fusion"
    run_dir: str | None = None
    overwrite_run_dir: bool = False


# =============================================================================
# Mode parsing
# =============================================================================

def parse_modes_weights(text: str) -> dict[str, float]:
    """
    Supported:
      A+B
      BCE
      A:0.5,B:0.3,C:0.2
      A=0.5 B=0.3 C=0.2
    """
    source = (text or "").strip().upper()
    if not source:
        raise ValueError("模式输入为空。")

    weights: dict[str, float] = {}

    if ":" in source or "=" in source:
        pattern = re.compile(
            r"([A-E])\s*[:=]\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)"
        )
        matches = list(pattern.finditer(source))
        if not matches:
            raise ValueError("未解析到有效权重。示例：A:0.5,B:0.3,C:0.2。")

        residue = re.sub(r"[\s,;+]+", "", pattern.sub("", source))
        if residue:
            raise ValueError(f"存在无法解析的模式权重内容：{residue}")

        for match in matches:
            mode = match.group(1)
            value = float(match.group(2))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"模式{mode}的权重必须是正数。")
            weights[mode] = weights.get(mode, 0.0) + value

    else:
        compact = re.sub(r"[\s,;+]+", "", source)
        if not compact or any(character not in MODE_ORDER for character in compact):
            raise ValueError("模式只能由A–E组成，例如A+B或BCE。")
        for mode in compact:
            weights[mode] = 1.0

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("模式权重总和必须大于0。")

    return {
        mode: weights[mode] / total
        for mode in MODE_ORDER
        if mode in weights
    }


# =============================================================================
# Helpers
# =============================================================================

def _resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (project_dir / path).resolve() if not path.is_absolute() else path.resolve()


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


def _write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)


def _validate_fusion_config(cfg: FusionConfig) -> None:
    if cfg.strategy not in {"rank_percentile", "score_percentile"}:
        raise ValueError(
            "Fusion strategy必须是rank_percentile或score_percentile。"
        )
    if not 0 < cfg.min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction必须位于(0, 1]。")


def _create_fusion_run_dir(project_dir: Path, cfg: FusionConfig) -> Path:
    if cfg.run_dir:
        run_dir = _resolve_path(project_dir, cfg.run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            if not cfg.overwrite_run_dir:
                raise FileExistsError(
                    f"指定的Fusion run_dir已经包含文件：\n  {run_dir}"
                )
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    root = _resolve_path(project_dir, cfg.out_root)
    root.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"run_{tag}"
    run_dir.mkdir()
    return run_dir


# =============================================================================
# Rank-table preparation
# =============================================================================

def _prepare_mode_rank_table(
    frame: pd.DataFrame,
    mode: str,
    criterion: str,
) -> pd.DataFrame:
    score_column = f"primary_catalyst_score_{criterion}"
    rank_column = f"catalyst_rank_{criterion}"
    required = ["catalyst_id", score_column, rank_column]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise RuntimeError(
            f"模式{mode}的{criterion}排名表缺少字段：{missing}"
        )

    table = frame[required].copy()
    table["catalyst_id"] = (
        table["catalyst_id"].fillna("").astype(str).str.strip()
    )
    if table["catalyst_id"].eq("").any():
        raise RuntimeError(f"模式{mode}的{criterion}排名表存在空catalyst_id。")
    if table["catalyst_id"].duplicated().any():
        raise RuntimeError(
            f"模式{mode}的{criterion}排名表存在重复catalyst_id。"
        )

    table[score_column] = pd.to_numeric(table[score_column], errors="coerce")
    table[rank_column] = pd.to_numeric(table[rank_column], errors="coerce")

    if table[[score_column, rank_column]].isna().any().any():
        raise RuntimeError(
            f"模式{mode}的{criterion}排名表存在缺失或非数值分数。"
        )
    if table[rank_column].lt(1).any():
        raise RuntimeError(f"模式{mode}的{criterion}排名必须从1开始。")

    catalyst_count = len(table)
    if catalyst_count == 1:
        rank_percentile = pd.Series([1.0], index=table.index)
    else:
        rank_percentile = (
            1.0 - (table[rank_column] - 1.0) / (catalyst_count - 1.0)
        ).clip(0.0, 1.0)

    score_percentile = table[score_column].rank(
        method="average",
        pct=True,
        ascending=True,
    )

    return pd.DataFrame({
        "catalyst_id": table["catalyst_id"],
        f"raw_score_{mode}": table[score_column],
        f"raw_rank_{mode}": table[rank_column],
        f"rank_percentile_{mode}": rank_percentile,
        f"score_percentile_{mode}": score_percentile,
        f"present_{mode}": True,
    })


def _build_profile_table(
    per_mode_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    parts = []
    for mode, frame in per_mode_frames.items():
        available = [column for column in PROFILE_COLUMNS if column in frame.columns]
        part = frame[["catalyst_id", *available]].copy()
        part["_source_mode"] = mode
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    rows = []

    for catalyst_id, group in combined.groupby("catalyst_id", sort=True):
        record: dict[str, Any] = {"catalyst_id": catalyst_id}

        for column in PROFILE_COLUMNS:
            if column not in group.columns:
                continue

            values = group[column].dropna().astype(str).str.strip()
            values = values[
                ~values.str.upper().isin({"", "NAN", "NONE"})
            ].drop_duplicates()

            if len(values) > 1:
                raise RuntimeError(
                    f"催化剂{catalyst_id}的{column}在不同模式间不一致。"
                )

            record[column] = values.iloc[0] if len(values) else np.nan

        rows.append(record)

    return pd.DataFrame(rows)


# =============================================================================
# Fusion
# =============================================================================

def fuse_rank_tables(
    per_mode: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    criterion: str,
    strategy: str = "rank_percentile",
    min_coverage_fraction: float = 1.0,
) -> pd.DataFrame:
    if criterion not in CRITERIA:
        raise ValueError(f"criterion必须是{CRITERIA}之一。")
    if strategy not in {"rank_percentile", "score_percentile"}:
        raise ValueError(
            "strategy必须是rank_percentile或score_percentile。"
        )
    if set(per_mode) != set(weights):
        raise ValueError("per_mode与weights中的模式集合不一致。")
    if not 0 < min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction必须位于(0, 1]。")

    prepared = {
        mode: _prepare_mode_rank_table(per_mode[mode], mode, criterion)
        for mode in MODE_ORDER
        if mode in weights
    }

    merged: pd.DataFrame | None = None
    for mode in MODE_ORDER:
        if mode not in prepared:
            continue
        merged = (
            prepared[mode]
            if merged is None
            else merged.merge(
                prepared[mode],
                on="catalyst_id",
                how="outer",
                validate="one_to_one",
            )
        )

    if merged is None or merged.empty:
        raise RuntimeError("没有可用于Fusion的催化剂排名。")

    metric_prefix = (
        "rank_percentile"
        if strategy == "rank_percentile"
        else "score_percentile"
    )

    available_weight = pd.Series(0.0, index=merged.index)
    weighted_sum = pd.Series(0.0, index=merged.index)
    available_count = pd.Series(0, index=merged.index, dtype=int)
    missing_modes: list[list[str]] = [[] for _ in range(len(merged))]

    for mode in MODE_ORDER:
        if mode not in weights:
            continue

        present = merged.get(
            f"present_{mode}",
            pd.Series(False, index=merged.index),
        ).fillna(False).astype(bool)

        metric = pd.to_numeric(
            merged[f"{metric_prefix}_{mode}"],
            errors="coerce",
        )
        weight = float(weights[mode])

        available_weight += weight * present.astype(float)
        weighted_sum += weight * metric.fillna(0.0) * present.astype(float)
        available_count += present.astype(int)
        merged[f"weight_{mode}"] = weight

        for index, is_present in enumerate(present.tolist()):
            if not is_present:
                missing_modes[index].append(mode)

    merged["available_mode_count"] = available_count
    merged["available_weight"] = available_weight
    merged["coverage_fraction"] = available_weight
    merged["missing_modes"] = [",".join(modes) for modes in missing_modes]
    merged["fused_score"] = np.where(
        available_weight > 0,
        weighted_sum / available_weight,
        np.nan,
    )
    merged["eligible_for_ranking"] = merged["coverage_fraction"].ge(
        min_coverage_fraction - 1e-12
    )
    merged["coverage_status"] = np.select(
        [
            merged["coverage_fraction"].ge(1.0 - 1e-12),
            merged["eligible_for_ranking"],
        ],
        ["full", "partial_eligible"],
        default="insufficient",
    )

    eligible = merged["eligible_for_ranking"]
    if not eligible.any():
        raise RuntimeError("没有催化剂达到min_coverage_fraction。")

    merged["fusion_rank"] = pd.Series(
        pd.NA,
        index=merged.index,
        dtype="Int64",
    )
    merged.loc[eligible, "fusion_rank"] = (
        merged.loc[eligible, "fused_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    merged.insert(0, "criterion", criterion)
    merged.insert(1, "fusion_strategy", strategy)

    return merged.sort_values(
        [
            "eligible_for_ranking",
            "fusion_rank",
            "coverage_fraction",
            "fused_score",
            "catalyst_id",
        ],
        ascending=[False, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# Full multi-mode run
# =============================================================================

def run_worldmap_fusion(
    recipes_df: pd.DataFrame,
    modes_text: str,
    project_dir: str | Path | None = None,
    core_cfg: WorldMapConfig | None = None,
    fusion_cfg: FusionConfig | None = None,
) -> dict[str, Any]:
    project = (
        Path(project_dir).expanduser().resolve()
        if project_dir is not None
        else PROJECT_DIR
    )
    core_cfg = core_cfg or WorldMapConfig()
    fusion_cfg = fusion_cfg or FusionConfig()
    _validate_fusion_config(fusion_cfg)

    weights = parse_modes_weights(modes_text)
    run_dir = _create_fusion_run_dir(project, fusion_cfg)
    manifest_path = run_dir / "FUSION_manifest.json"

    manifest: dict[str, Any] = {
        "status": "running",
        "project_dir": str(project),
        "run_dir": str(run_dir),
        "weights": weights,
        "fusion_config": asdict(fusion_cfg),
        "core_config": asdict(core_cfg),
        "definitions": {
            "balanced": "q50",
            "potential": "q90",
            "conservative": "q10",
            "default_fusion": "mode-internal rank percentile",
            "missing_mode_rule": (
                "renormalize over available modes; rank only when "
                "coverage threshold is met"
            ),
        },
        "mode_runs": {},
        "outputs": {},
    }
    _write_json(manifest, manifest_path)

    mode_results: dict[str, dict[str, Any]] = {}

    try:
        for mode in MODE_ORDER:
            if mode not in weights:
                continue

            mode_dir = run_dir / f"mode_{mode}"
            mode_cfg = replace(
                core_cfg,
                run_dir=str(mode_dir),
                overwrite_run_dir=False,
            )
            result = run_worldmap(
                recipes_df=recipes_df,
                growth_mode=mode,
                project_dir=project,
                cfg=mode_cfg,
            )
            if result.get("status") != "success":
                raise RuntimeError(f"模式{mode}未返回success状态。")

            mode_results[mode] = result
            manifest["mode_runs"][mode] = {
                "weight": weights[mode],
                "run_dir": result["run_dir"],
                "audit": result.get("audit", {}),
            }
            _write_json(manifest, manifest_path)

        profile = _build_profile_table({
            mode: mode_results[mode]["tables"]["catalyst_comparison"]
            for mode in weights
        })

        fused_tables: dict[str, pd.DataFrame] = {}
        output_paths: dict[str, Path] = {}

        for criterion in CRITERIA:
            per_mode = {
                mode: mode_results[mode]["tables"][f"catalyst_{criterion}"]
                for mode in weights
            }
            fused = fuse_rank_tables(
                per_mode=per_mode,
                weights=weights,
                criterion=criterion,
                strategy=fusion_cfg.strategy,
                min_coverage_fraction=fusion_cfg.min_coverage_fraction,
            )
            fused = fused.merge(
                profile,
                on="catalyst_id",
                how="left",
                validate="one_to_one",
            )

            output_path = run_dir / f"FUSION_rank_{criterion}.csv"
            fused.to_csv(output_path, index=False, encoding="utf-8-sig")
            fused_tables[criterion] = fused
            output_paths[f"fusion_rank_{criterion}"] = output_path

        weights_path = run_dir / "FUSION_mode_weights.csv"
        pd.DataFrame([
            {"mode": mode, "weight": weights[mode]}
            for mode in MODE_ORDER
            if mode in weights
        ]).to_csv(weights_path, index=False, encoding="utf-8-sig")

        mode_rows = []
        for mode in MODE_ORDER:
            if mode not in mode_results:
                continue
            audit = mode_results[mode].get("audit", {})
            mode_rows.append({
                "mode": mode,
                "weight": weights[mode],
                "run_dir": mode_results[mode]["run_dir"],
                "L123_001_routes": audit.get("L123_001", {}).get("rows", np.nan),
                "L123_002_union_rows": (
                    audit.get("L123_002", {}).get("union_rows", np.nan)
                ),
                "balanced_boundary_fraction": (
                    audit.get("L123_003", {}).get(
                        "balanced_boundary_fraction", np.nan
                    )
                ),
                "potential_boundary_fraction": (
                    audit.get("L123_003", {}).get(
                        "potential_boundary_fraction", np.nan
                    )
                ),
            })

        mode_runs_path = run_dir / "FUSION_mode_runs.csv"
        pd.DataFrame(mode_rows).to_csv(
            mode_runs_path,
            index=False,
            encoding="utf-8-sig",
        )

        coverage_rows = []
        for criterion, table in fused_tables.items():
            coverage_rows.append({
                "criterion": criterion,
                "catalyst_rows": int(len(table)),
                "eligible_rows": int(table["eligible_for_ranking"].sum()),
                "full_coverage_rows": int(
                    table["coverage_status"].eq("full").sum()
                ),
                "partial_eligible_rows": int(
                    table["coverage_status"].eq("partial_eligible").sum()
                ),
                "insufficient_rows": int(
                    table["coverage_status"].eq("insufficient").sum()
                ),
                "minimum_coverage_fraction": float(
                    table["coverage_fraction"].min()
                ),
            })

        coverage_path = run_dir / "FUSION_coverage_summary.csv"
        pd.DataFrame(coverage_rows).to_csv(
            coverage_path,
            index=False,
            encoding="utf-8-sig",
        )

        output_paths.update({
            "fusion_weights": weights_path,
            "fusion_mode_runs": mode_runs_path,
            "fusion_coverage_summary": coverage_path,
            "fusion_manifest": manifest_path,
        })

        manifest["status"] = "success"
        manifest["outputs"] = {
            key: str(value) for key, value in output_paths.items()
        }
        manifest["coverage"] = coverage_rows
        _write_json(manifest, manifest_path)

        return {
            "status": "success",
            "run_dir": str(run_dir),
            "weights": weights,
            "fusion_strategy": fusion_cfg.strategy,
            "min_coverage_fraction": fusion_cfg.min_coverage_fraction,
            "paths": {
                key: str(value) for key, value in output_paths.items()
            },
            "tables": {
                "fusion_balanced": fused_tables["balanced"],
                "fusion_potential": fused_tables["potential"],
                "fusion_conservative": fused_tables["conservative"],
            },
            "mode_results": mode_results,
            "coverage_summary": pd.DataFrame(coverage_rows),
            "config": {
                "core": asdict(core_cfg),
                "fusion": asdict(fusion_cfg),
            },
        }

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        _write_json(manifest, manifest_path)
        raise
