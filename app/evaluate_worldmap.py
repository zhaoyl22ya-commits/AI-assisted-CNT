# evaluate_worldmap.py
# =============================================================================
# Audit the frozen CNT WorldMap -> L123-001 -> L123-002 -> L123-003 outputs.
#
# This script does not retrain models, rebuild the composite target, resplit the
# literature dataset, or use q90-q10 as a ranking penalty.
#
# Fixed definitions:
#   Balanced     = raw q50
#   Potential    = raw q90
#   Conservative = raw q10
#   Ordered q90-q10 width is descriptive only
#   Physical-state variables = np_size_nm and raw C_over_M
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTDIR = PROJECT_DIR / "outputs" / "app" / "evaluate_worldmap"
MODE_ORDER = ["A", "B", "C", "D", "E"]
SCRIPT_REVISION = "current_repo_20261001"

VALIDATION_RELATIVE_PATHS = {
    "article_validation":
        Path("outputs/external_validation/validate_generalization/three_split_metrics_with_ci.csv"),
    "uncertainty_calibration":
        Path("outputs/external_validation/uncertainty/uncertainty_calibration.csv"),
    "uncertainty_error_relation":
        Path("outputs/external_validation/uncertainty/uncertainty_error_relation.csv"),
    "retrospective_ranking_summary":
        Path("outputs/external_validation/retrospective_ranking/retrospective_ranking_summary.csv"),
    "composite_threshold_comparison":
        Path("outputs/external_validation/composite_sensitivity/composite_threshold_comparison_table.csv"),
}


LEVEL_ORDER = ["L1", "L2", "L3"]

MODULE_FEATURES = {
    "N": [f"N{i}" for i in range(1, 19)],
    "C": [f"C{i}" for i in range(1, 13)],
    "F": [f"F{i}" for i in range(1, 19)],
    "P": [f"P{i}" for i in range(1, 13)],
}

TARGET_SPECS = [
    ("Growth quality", "y_growth_quality_score"),
    ("Diameter", "y_diameter_nm"),
    ("Length", "y_length_mm"),
    ("Bundle density", "y_bundle_density_per_um"),
    ("CNT density", "y_cnt_density_per_um"),
    ("IG/ID", "y_ig_id_ratio_unified"),
    ("Orientation", "y_orientation_score"),
    ("Purity", "y_purity_score"),
    ("Composite (>=2 components)", "__derived_composite_min2__"),
]

COMPOSITE_COMPONENT_COLUMNS = [
    "y_growth_quality_score",
    "y_ig_id_ratio_unified",
    "y_orientation_score",
    "y_purity_score",
    "y_bundle_density_per_um",
]


# =============================================================================
# Helpers
# =============================================================================

def resolve_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (project_dir / path).resolve() if not path.is_absolute() else path.resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found:\n  {path}")


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{label} is missing required columns:\n  {missing}")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path, "CSV")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def read_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def mode_letter(value: Any) -> str:
    if value is None or pd.isna(value):
        return "?"
    token = str(value).strip().upper()
    if token in MODE_ORDER:
        return token
    try:
        return {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(
            int(float(token)), "?"
        )
    except (TypeError, ValueError):
        return "?"


def add_mode_letter(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    source = (
        "growth_mode_letter"
        if "growth_mode_letter" in output.columns
        else "growth_mode"
        if "growth_mode" in output.columns
        else None
    )
    output["mode_letter"] = (
        output[source].map(mode_letter) if source else "?"
    )
    return output


def safe_corr(x: pd.Series, y: pd.Series, rank: bool = False) -> float:
    pair = pd.DataFrame({"x": to_num(x), "y": to_num(y)}).dropna()
    if len(pair) < 2:
        return np.nan
    if rank:
        pair = pair.rank(method="average")
    return float(pair["x"].corr(pair["y"]))


def latest_worldmap_run(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"WorldMap output root not found:\n  {root}")
    runs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run_* directory found under:\n  {root}")
    return runs[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for label, path in paths.items():
        exists = path.is_file()
        rows.append({
            "input_label": label,
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else np.nan,
            "sha256": file_sha256(path) if exists else "",
        })
    return pd.DataFrame(rows)



# =============================================================================
# Dataset composition and missingness
# =============================================================================

def ensure_igid_unified(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    preferred = (
        to_num(output["y_ig_id_ratio"])
        if "y_ig_id_ratio" in output.columns
        else pd.Series(np.nan, index=output.index, dtype=float)
    )
    legacy = (
        to_num(output["y_IG_ID"])
        if "y_IG_ID" in output.columns
        else pd.Series(np.nan, index=output.index, dtype=float)
    )
    if "y_ig_id_ratio_unified" in output.columns:
        unified = to_num(output["y_ig_id_ratio_unified"])
        output["y_ig_id_ratio_unified"] = unified.where(
            unified.notna(),
            preferred.where(preferred.notna(), legacy),
        )
    else:
        output["y_ig_id_ratio_unified"] = preferred.where(
            preferred.notna(),
            legacy,
        )
    return output


def missing_flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = numeric.gt(0)
    unresolved = numeric.isna()
    if unresolved.any():
        text = (
            series.loc[unresolved]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        output.loc[unresolved] = text.isin(
            {"true", "t", "yes", "y", "missing", "1", "1.0"}
        )
    return output.fillna(False).astype(bool)


def valid_numeric_mask(series: pd.Series) -> pd.Series:
    return to_num(series).notna()


def dataset_year_table(data: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        data,
        ["paper_year_int", "paper_id", "sample_id"],
        "Full CNT dataset",
    )
    work = data.copy()
    work["paper_year_int"] = to_num(work["paper_year_int"])
    work = work[work["paper_year_int"].notna()].copy()
    work["paper_year_int"] = work["paper_year_int"].astype(int)

    return (
        work.groupby("paper_year_int", sort=True)
        .agg(
            paper_count=("paper_id", "nunique"),
            sample_count=("sample_id", "nunique"),
        )
        .reset_index()
    )


def dataset_level_mode_table(data: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        data,
        ["paper_id", "sample_id", "data_level"],
        "Full CNT dataset",
    )
    work = add_mode_letter(data)
    work["data_level"] = (
        work["data_level"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    rows = []
    for level in LEVEL_ORDER:
        subset = work[work["data_level"].eq(level)]
        rows.append({
            "count_type": "data_level",
            "category": level,
            "sample_count": int(subset["sample_id"].nunique()),
            "paper_count": int(subset["paper_id"].nunique()),
        })

    for mode in MODE_ORDER:
        subset = work[work["mode_letter"].eq(mode)]
        rows.append({
            "count_type": "growth_mode",
            "category": mode,
            "sample_count": int(subset["sample_id"].nunique()),
            "paper_count": int(subset["paper_id"].nunique()),
        })

    unknown_mode = work[work["mode_letter"].eq("?")]
    if not unknown_mode.empty:
        rows.append({
            "count_type": "growth_mode",
            "category": "Unknown",
            "sample_count": int(unknown_mode["sample_id"].nunique()),
            "paper_count": int(unknown_mode["paper_id"].nunique()),
        })

    other_levels = sorted(
        set(work.loc[work["data_level"].ne(""), "data_level"]) - set(LEVEL_ORDER)
    )
    for level in other_levels:
        subset = work[work["data_level"].eq(level)]
        rows.append({
            "count_type": "data_level",
            "category": level,
            "sample_count": int(subset["sample_id"].nunique()),
            "paper_count": int(subset["paper_id"].nunique()),
        })

    return pd.DataFrame(rows)


def target_availability_table(data: pd.DataFrame) -> pd.DataFrame:
    work = ensure_igid_unified(data)
    total_samples = int(len(work))
    total_papers = int(work["paper_id"].nunique()) if "paper_id" in work.columns else 0

    component_masks = []
    for column in COMPOSITE_COMPONENT_COLUMNS:
        if column in work.columns:
            component_masks.append(valid_numeric_mask(work[column]))
        else:
            component_masks.append(pd.Series(False, index=work.index))

    component_count = pd.concat(component_masks, axis=1).sum(axis=1)
    composite_min2 = component_count.ge(2)

    rows = []
    for label, column in TARGET_SPECS:
        if column == "__derived_composite_min2__":
            valid = composite_min2
            exists = all(
                name in work.columns
                for name in [
                    "y_growth_quality_score",
                    "y_orientation_score",
                    "y_purity_score",
                    "y_bundle_density_per_um",
                ]
            ) and (
                "y_ig_id_ratio_unified" in work.columns
            )
            source = (
                "derived from at least 2 of 5 formal composite components"
            )
        elif column in work.columns:
            valid = valid_numeric_mask(work[column])
            exists = True
            source = column
        else:
            valid = pd.Series(False, index=work.index)
            exists = False
            source = column

        valid_count = int(valid.sum())
        valid_papers = (
            int(work.loc[valid, "paper_id"].nunique())
            if "paper_id" in work.columns
            else 0
        )
        rows.append({
            "target_label": label,
            "column_name": source,
            "column_exists": bool(exists),
            "valid_sample_count": valid_count,
            "valid_sample_fraction": (
                valid_count / total_samples if total_samples > 0 else np.nan
            ),
            "valid_paper_count": valid_papers,
            "valid_paper_fraction": (
                valid_papers / total_papers if total_papers > 0 else np.nan
            ),
        })

    return pd.DataFrame(rows)


def feature_missingness_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = int(len(data))

    for module, features in MODULE_FEATURES.items():
        feature_rows = []
        for feature in features:
            miss_column = f"miss_{feature}"
            if miss_column in data.columns:
                missing = missing_flag(data[miss_column])
                source = miss_column
            elif feature in data.columns:
                missing = to_num(data[feature]).isna()
                source = feature
            else:
                missing = pd.Series(True, index=data.index)
                source = "feature_and_missing_flag_absent"

            missing_count = int(missing.sum())
            row = {
                "row_type": "feature",
                "module": module,
                "feature": feature,
                "missing_source": source,
                "missing_count": missing_count,
                "observed_count": int(total - missing_count),
                "missing_fraction": (
                    missing_count / total if total > 0 else np.nan
                ),
                "observed_fraction": (
                    1.0 - missing_count / total if total > 0 else np.nan
                ),
            }
            rows.append(row)
            feature_rows.append(row)

        fractions = np.asarray(
            [item["missing_fraction"] for item in feature_rows],
            dtype=float,
        )
        rows.append({
            "row_type": "module_summary",
            "module": module,
            "feature": "",
            "missing_source": "mean across module features",
            "missing_count": np.nan,
            "observed_count": np.nan,
            "missing_fraction": (
                float(np.nanmean(fractions)) if len(fractions) else np.nan
            ),
            "observed_fraction": (
                float(1.0 - np.nanmean(fractions)) if len(fractions) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def add_bar_labels(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    for container in ax.containers:
        try:
            labels = [
                fmt.format(value)
                if np.isfinite(value)
                else ""
                for value in container.datavalues
            ]
            ax.bar_label(container, labels=labels, padding=2, fontsize=8)
        except (AttributeError, TypeError):
            continue


def plot_dataset_composition(
    year_counts: pd.DataFrame,
    level_mode_counts: pd.DataFrame,
    target_counts: pd.DataFrame,
    missingness: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(17.0, 13.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95])

    ax_a = fig.add_subplot(grid[0, 0])
    if not year_counts.empty:
        x = np.arange(len(year_counts))
        width = 0.42
        ax_a.bar(
            x - width / 2,
            year_counts["sample_count"],
            width=width,
            label="Samples",
        )
        ax_a.bar(
            x + width / 2,
            year_counts["paper_count"],
            width=width,
            label="Papers",
        )
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(
            year_counts["paper_year_int"].astype(str),
            rotation=45,
            ha="right",
        )
    ax_a.set_xlabel("Publication year")
    ax_a.set_ylabel("Count")
    ax_a.set_title("a  Temporal coverage")
    ax_a.legend(frameon=False)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    b_grid = grid[0, 1].subgridspec(1, 2)
    ax_b1 = fig.add_subplot(b_grid[0, 0])
    ax_b2 = fig.add_subplot(b_grid[0, 1])

    level = level_mode_counts[
        level_mode_counts["count_type"].eq("data_level")
    ].copy()
    if not level.empty:
        order = LEVEL_ORDER + [
            item for item in level["category"] if item not in LEVEL_ORDER
        ]
        level["_order"] = level["category"].map(
            {name: index for index, name in enumerate(order)}
        )
        level.sort_values("_order", inplace=True, kind="stable")
        ax_b1.bar(level["category"], level["sample_count"])
        add_bar_labels(ax_b1)
    ax_b1.set_xlabel("Data level")
    ax_b1.set_ylabel("Samples")
    ax_b1.set_title("b  Data levels")
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)

    mode = level_mode_counts[
        level_mode_counts["count_type"].eq("growth_mode")
    ].copy()
    if not mode.empty:
        order = MODE_ORDER + ["Unknown"]
        mode["_order"] = mode["category"].map(
            {name: index for index, name in enumerate(order)}
        )
        mode.sort_values("_order", inplace=True, kind="stable")
        ax_b2.bar(mode["category"], mode["sample_count"])
        add_bar_labels(ax_b2)
    ax_b2.set_xlabel("Growth mode")
    ax_b2.set_ylabel("Samples")
    ax_b2.set_title("Growth modes")
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)

    ax_c = fig.add_subplot(grid[1, 0])
    target_plot = target_counts.iloc[::-1].copy()
    y = np.arange(len(target_plot))
    ax_c.barh(y, target_plot["valid_sample_count"])
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(target_plot["target_label"])
    ax_c.set_xlabel("Valid labeled samples")
    ax_c.set_title("c  Target availability")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    for index, (_, row) in enumerate(target_plot.iterrows()):
        fraction = row["valid_sample_fraction"]
        label = (
            f"{int(row['valid_sample_count'])} "
            f"({fraction:.1%})"
            if np.isfinite(fraction)
            else str(int(row["valid_sample_count"]))
        )
        ax_c.text(
            row["valid_sample_count"],
            index,
            "  " + label,
            va="center",
            fontsize=8,
        )

    d_grid = grid[1, 1].subgridspec(2, 1, height_ratios=[0.35, 1.0])
    ax_d1 = fig.add_subplot(d_grid[0, 0])
    ax_d2 = fig.add_subplot(d_grid[1, 0])

    module_summary = missingness[
        missingness["row_type"].eq("module_summary")
    ].copy()
    module_summary["_order"] = module_summary["module"].map(
        {name: index for index, name in enumerate(MODULE_FEATURES)}
    )
    module_summary.sort_values("_order", inplace=True, kind="stable")
    ax_d1.bar(
        module_summary["module"],
        module_summary["missing_fraction"],
    )
    ax_d1.set_ylim(0.0, 1.0)
    ax_d1.set_ylabel("Mean missing fraction")
    ax_d1.set_title("d  N/C/F/P missingness")
    ax_d1.spines["top"].set_visible(False)
    ax_d1.spines["right"].set_visible(False)
    add_bar_labels(ax_d1, fmt="{:.2f}")

    feature = missingness[
        missingness["row_type"].eq("feature")
    ].copy()
    module_rank = {name: index for index, name in enumerate(MODULE_FEATURES)}
    feature["_module_order"] = feature["module"].map(module_rank)
    feature["_feature_number"] = (
        feature["feature"].str.extract(r"(\d+)", expand=False).astype(int)
    )
    feature.sort_values(
        ["_module_order", "_feature_number"],
        inplace=True,
        kind="stable",
    )
    x = np.arange(len(feature))
    ax_d2.bar(x, feature["missing_fraction"])
    ax_d2.set_xticks(x)
    ax_d2.set_xticklabels(feature["feature"], rotation=90, fontsize=7)
    ax_d2.set_ylim(0.0, 1.0)
    ax_d2.set_xlabel("Physical feature")
    ax_d2.set_ylabel("Missing fraction")
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_visible(False)

    boundaries = []
    cumulative = 0
    for module, features in MODULE_FEATURES.items():
        cumulative += len(features)
        boundaries.append(cumulative - 0.5)
    for boundary in boundaries[:-1]:
        ax_d2.axvline(boundary, linewidth=0.8, linestyle="--", alpha=0.6)

    fig.suptitle(
        "CNT literature dataset composition and missingness",
        fontsize=16,
    )
    save_figure(fig, path, dpi)


def audit_dataset_composition(
    dataset_path: Path,
    outdir: Path,
    figdir: Path,
    dpi: int,
) -> dict[str, Any]:
    data = read_csv(dataset_path)
    require_columns(
        data,
        ["sample_id", "paper_id", "paper_year_int", "data_level", "growth_mode"],
        "Full CNT dataset",
    )

    data = data.copy()
    data["sample_id"] = data["sample_id"].fillna("").astype(str)
    data["paper_id"] = data["paper_id"].fillna("").astype(str)
    data["paper_year_int"] = to_num(data["paper_year_int"])
    data["data_level"] = (
        data["data_level"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    year_counts = dataset_year_table(data)
    level_mode_counts = dataset_level_mode_table(data)
    target_counts = target_availability_table(data)
    missingness = feature_missingness_table(data)

    year_counts.to_csv(
        outdir / "dataset_year_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    level_mode_counts.to_csv(
        outdir / "dataset_level_mode_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target_counts.to_csv(
        outdir / "target_availability_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missingness.to_csv(
        outdir / "feature_missingness_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure_path = figdir / "figS_dataset_composition_and_missingness.png"
    plot_dataset_composition(
        year_counts,
        level_mode_counts,
        target_counts,
        missingness,
        figure_path,
        dpi,
    )

    level_counts = {
        row["category"]: int(row["sample_count"])
        for _, row in level_mode_counts[
            level_mode_counts["count_type"].eq("data_level")
        ].iterrows()
    }
    mode_counts = {
        row["category"]: int(row["sample_count"])
        for _, row in level_mode_counts[
            level_mode_counts["count_type"].eq("growth_mode")
        ].iterrows()
    }
    module_missing = {
        row["module"]: float(row["missing_fraction"])
        for _, row in missingness[
            missingness["row_type"].eq("module_summary")
        ].iterrows()
    }

    return {
        "dataset_path": str(dataset_path),
        "sample_rows": int(len(data)),
        "unique_sample_ids": int(data["sample_id"].nunique()),
        "unique_papers": int(data["paper_id"].nunique()),
        "dated_samples": int(data["paper_year_int"].notna().sum()),
        "dated_papers": int(
            data.loc[data["paper_year_int"].notna(), "paper_id"].nunique()
        ),
        "year_min": (
            int(data["paper_year_int"].min())
            if data["paper_year_int"].notna().any()
            else None
        ),
        "year_max": (
            int(data["paper_year_int"].max())
            if data["paper_year_int"].notna().any()
            else None
        ),
        "level_sample_counts": level_counts,
        "mode_sample_counts": mode_counts,
        "target_valid_sample_counts": {
            row["target_label"]: int(row["valid_sample_count"])
            for _, row in target_counts.iterrows()
        },
        "module_mean_missing_fraction": module_missing,
        "figure": str(figure_path),
        "source_tables": {
            "year_counts": str(outdir / "dataset_year_counts.csv"),
            "level_mode_counts": str(outdir / "dataset_level_mode_counts.csv"),
            "target_availability": str(outdir / "target_availability_counts.csv"),
            "feature_missingness": str(outdir / "feature_missingness_counts.csv"),
        },
    }


# =============================================================================
# WorldMap
# =============================================================================

def audit_worldmap_physical_consistency(
    candidates: pd.DataFrame,
    outdir: Path,
) -> dict[str, Any]:
    """Audit the route-dependent N/C identities used by phase_diagram_worldmap.py."""
    required = [
        "N8", "N9", "N16",
        "C1", "C2", "C5", "C6", "C7", "C8", "C9",
        "P1", "P8", "P9", "P10",
        "miss_N8", "miss_N9", "miss_N16",
        "miss_C1", "miss_C2", "miss_C5", "miss_C6",
        "miss_C7", "miss_C8", "miss_C9",
        "miss_P1", "miss_P8", "miss_P9", "miss_P10",
    ]
    require_columns(candidates, required, "WorldMap physical-consistency audit")

    def observed(column: str) -> pd.Series:
        return (
            ~missing_flag(candidates[f"miss_{column}"])
            & to_num(candidates[column]).notna()
        )

    def close(a: pd.Series, b: pd.Series) -> pd.Series:
        a_num = to_num(a)
        b_num = to_num(b)
        return pd.Series(
            np.isclose(
                a_num.to_numpy(float),
                b_num.to_numpy(float),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=False,
            ),
            index=a_num.index,
        )

    rows = []

    def add_direct(check: str, target: str, source: str) -> None:
        expected_observed = observed(source)
        actual_observed = observed(target)
        value_ok = pd.Series(True, index=candidates.index)
        mask = expected_observed & actual_observed
        value_ok.loc[mask] = close(
            candidates.loc[mask, target],
            candidates.loc[mask, source],
        )
        mismatch = actual_observed.ne(expected_observed) | ~value_ok
        rows.append({
            "check": check,
            "evaluable_rows": int(expected_observed.sum()),
            "mismatch_rows": int(mismatch.sum()),
            "passed": bool(mismatch.sum() == 0),
        })

    add_direct("C1 == P1", "C1", "P1")
    add_direct("C5 == P8", "C5", "P8")
    add_direct("C6 == P9", "C6", "P9")
    add_direct("C7 == P10", "C7", "P10")
    add_direct("C8 == N16", "C8", "N16")

    valid_ratio = observed("P1") & observed("N8") & to_num(candidates["N8"]).ne(0)
    expected_ratio = pd.Series(np.nan, index=candidates.index, dtype=float)
    expected_ratio.loc[valid_ratio] = (
        to_num(candidates.loc[valid_ratio, "P1"])
        / to_num(candidates.loc[valid_ratio, "N8"])
    )

    for target, check in [
        ("N9", "N9 == P1 / N8"),
        ("C2", "C2 == P1 / N8"),
    ]:
        actual_observed = observed(target)
        value_ok = pd.Series(True, index=candidates.index)
        mask = valid_ratio & actual_observed
        value_ok.loc[mask] = close(
            candidates.loc[mask, target],
            expected_ratio.loc[mask],
        )
        mismatch = actual_observed.ne(valid_ratio) | ~value_ok
        rows.append({
            "check": check,
            "evaluable_rows": int(valid_ratio.sum()),
            "mismatch_rows": int(mismatch.sum()),
            "passed": bool(mismatch.sum() == 0),
        })

    valid_c9 = observed("N16") & valid_ratio
    expected_c9 = pd.Series(np.nan, index=candidates.index, dtype=float)
    expected_c9.loc[valid_c9] = (
        to_num(candidates.loc[valid_c9, "N16"])
        * expected_ratio.loc[valid_c9]
    )
    actual_c9 = observed("C9")
    value_ok = pd.Series(True, index=candidates.index)
    mask = valid_c9 & actual_c9
    value_ok.loc[mask] = close(
        candidates.loc[mask, "C9"],
        expected_c9.loc[mask],
    )
    mismatch = actual_c9.ne(valid_c9) | ~value_ok
    rows.append({
        "check": "C9 == N16 * (P1 / N8)",
        "evaluable_rows": int(valid_c9.sum()),
        "mismatch_rows": int(mismatch.sum()),
        "passed": bool(mismatch.sum() == 0),
    })

    audit = pd.DataFrame(rows)
    audit.to_csv(
        outdir / "worldmap_physical_consistency_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "physical_consistency_checks": int(len(audit)),
        "physical_consistency_failed_checks": int((~audit["passed"]).sum()),
        "physical_consistency_mismatch_rows": int(audit["mismatch_rows"].sum()),
    }


def audit_worldmap(
    run_dir: Path,
    outdir: Path,
    figdir: Path,
    dpi: int,
) -> dict[str, Any]:
    candidates = add_mode_letter(read_csv(run_dir / "PHASE_CANDIDATES_sampled.csv"))
    catalysts = read_csv(run_dir / "CATALYST_STATES.csv")
    routes = add_mode_letter(read_csv(run_dir / "ROUTE_STATES.csv"))
    pca = read_csv(run_dir / "BLOCK_PCA_AUDIT.csv")

    require_columns(
        candidates,
        [
            "combination_id", "catalyst_state_id", "route_state_id",
            "pred_q10_raw", "pred_q50_raw", "pred_q90_raw",
            "q10_fix", "q50_fix", "q90_fix",
            "interval_width", "interval_width_ordered",
            "pred_L1_composite_final_score", "quantile_fixed", "mode_letter",
        ],
        "WorldMap candidates",
    )
    require_columns(catalysts, ["catalyst_state_id"], "WorldMap catalyst states")
    require_columns(routes, ["route_state_id", "mode_letter"], "WorldMap routes")

    numeric = [
        "pred_q10_raw", "pred_q50_raw", "pred_q90_raw",
        "q10_fix", "q50_fix", "q90_fix",
        "interval_width", "interval_width_ordered",
        "pred_L1_composite_final_score", "quantile_fixed",
    ]
    for column in numeric:
        candidates[column] = to_num(candidates[column])

    raw_crossings = (
        candidates["pred_q10_raw"].gt(candidates["pred_q50_raw"])
        | candidates["pred_q50_raw"].gt(candidates["pred_q90_raw"])
        | candidates["pred_q10_raw"].gt(candidates["pred_q90_raw"])
    )
    ordered_crossings = (
        candidates["q10_fix"].gt(candidates["q50_fix"])
        | candidates["q50_fix"].gt(candidates["q90_fix"])
        | candidates["q10_fix"].gt(candidates["q90_fix"])
    )
    duplicates = int(candidates["combination_id"].duplicated().sum())
    negative_ordered_width = int(candidates["interval_width_ordered"].lt(0).sum())
    physical_consistency = audit_worldmap_physical_consistency(candidates, outdir)

    integrity = {
        "worldmap_run": str(run_dir),
        "candidate_rows": int(len(candidates)),
        "catalyst_state_rows": int(len(catalysts)),
        "route_state_rows": int(len(routes)),
        "unique_combination_ids": int(candidates["combination_id"].nunique()),
        "duplicate_combination_ids": duplicates,
        "raw_quantile_crossings": int(raw_crossings.sum()),
        "ordered_quantile_crossings": int(ordered_crossings.sum()),
        "quantile_fixed_reported_sum": int(candidates["quantile_fixed"].fillna(0).sum()),
        "negative_ordered_interval_width_rows": negative_ordered_width,
        **physical_consistency,
        "q50_q90_spearman": safe_corr(
            candidates["pred_q50_raw"], candidates["pred_q90_raw"], rank=True
        ),
        "q50_L1_spearman": safe_corr(
            candidates["pred_q50_raw"],
            candidates["pred_L1_composite_final_score"],
            rank=True,
        ),
        "q50_L1_pearson": safe_corr(
            candidates["pred_q50_raw"],
            candidates["pred_L1_composite_final_score"],
        ),
    }
    pd.DataFrame([integrity]).to_csv(
        outdir / "worldmap_integrity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mode_table = (
        candidates.groupby("mode_letter", dropna=False)
        .agg(
            candidate_rows=("combination_id", "size"),
            unique_catalyst_states=("catalyst_state_id", "nunique"),
            unique_route_states=("route_state_id", "nunique"),
            q10_mean=("pred_q10_raw", "mean"),
            q50_mean=("pred_q50_raw", "mean"),
            q90_mean=("pred_q90_raw", "mean"),
            interval_width_mean=("interval_width_ordered", "mean"),
        )
        .reset_index()
    )
    mode_table["_order"] = mode_table["mode_letter"].map(
        {mode: index for index, mode in enumerate(MODE_ORDER)}
    )
    mode_table.sort_values("_order", inplace=True, kind="stable")
    mode_table.drop(columns="_order", inplace=True)
    mode_table.to_csv(
        outdir / "worldmap_mode_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    score_stats = (
        candidates[
            [
                "pred_q10_raw", "pred_q50_raw", "pred_q90_raw",
                "interval_width_ordered",
                "pred_L1_composite_final_score",
            ]
        ]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .T.reset_index()
        .rename(columns={"index": "score"})
    )
    score_stats.to_csv(
        outdir / "worldmap_score_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pca.to_csv(
        outdir / "worldmap_block_pca_audit_copy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="constrained")
    for mode in MODE_ORDER:
        subset = candidates[candidates["mode_letter"].eq(mode)]
        if not subset.empty:
            ax.scatter(
                subset["pred_q50_raw"], subset["pred_q90_raw"],
                s=10, alpha=0.45, label=f"Mode {mode}",
            )
    ax.set_xlabel("Balanced score (raw q50)")
    ax.set_ylabel("Potential score (raw q90)")
    ax.set_title("WorldMap candidate score space")
    ax.legend(frameon=False)
    save_figure(fig, figdir / "worldmap_q50_vs_q90.png", dpi)

    fig, ax = plt.subplots(figsize=(8.6, 6.0), layout="constrained")
    values, labels = [], []
    for mode in MODE_ORDER:
        series = candidates.loc[
            candidates["mode_letter"].eq(mode), "pred_q50_raw"
        ].dropna()
        if len(series):
            values.append(series.to_numpy(float))
            labels.append(mode)
    if values:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_xlabel("Growth mode")
    ax.set_ylabel("Balanced score (raw q50)")
    ax.set_title("WorldMap q50 distribution by growth mode")
    save_figure(fig, figdir / "worldmap_q50_by_mode.png", dpi)

    return integrity


# =============================================================================
# L123-001
# =============================================================================

def audit_l123_001(path: Path, outdir: Path) -> dict[str, Any]:
    data = add_mode_letter(read_csv(path))
    require_columns(
        data,
        [
            "catalyst_id", "template_id", "combination_id",
            "pred_q10", "pred_q50", "pred_q90",
        ],
        "L123-001 shortlist",
    )
    for column in ["pred_q10", "pred_q50", "pred_q90"]:
        data[column] = to_num(data[column])

    raw_crossings = (
        data["pred_q10"].gt(data["pred_q50"])
        | data["pred_q50"].gt(data["pred_q90"])
        | data["pred_q10"].gt(data["pred_q90"])
    )

    ordered_columns = ["pred_q10_ordered", "pred_q50_ordered", "pred_q90_ordered"]
    if all(column in data.columns for column in ordered_columns):
        for column in ordered_columns:
            data[column] = to_num(data[column])
        ordered_crossings = (
            data["pred_q10_ordered"].gt(data["pred_q50_ordered"])
            | data["pred_q50_ordered"].gt(data["pred_q90_ordered"])
            | data["pred_q10_ordered"].gt(data["pred_q90_ordered"])
        )
        ordered_crossing_count = int(ordered_crossings.sum())
    else:
        ordered_crossing_count = np.nan

    coverage = (
        data.groupby("catalyst_id")
        .agg(
            route_rows=("combination_id", "size"),
            unique_templates=("template_id", "nunique"),
            unique_combinations=("combination_id", "nunique"),
            q10_missing=("pred_q10", lambda s: int(s.isna().sum())),
            q50_missing=("pred_q50", lambda s: int(s.isna().sum())),
            q90_missing=("pred_q90", lambda s: int(s.isna().sum())),
            q50_mean=("pred_q50", "mean"),
            q90_mean=("pred_q90", "mean"),
        )
        .reset_index()
    )
    coverage.to_csv(
        outdir / "l123_001_coverage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mode_coverage = (
        data.groupby("mode_letter", dropna=False)
        .agg(
            route_rows=("combination_id", "size"),
            catalysts=("catalyst_id", "nunique"),
            templates=("template_id", "nunique"),
        )
        .reset_index()
    )
    mode_coverage.to_csv(
        outdir / "l123_001_mode_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "route_rows": int(len(data)),
        "catalysts": int(data["catalyst_id"].nunique()),
        "templates": int(data["template_id"].nunique()),
        "duplicate_combination_ids": int(data["combination_id"].duplicated().sum()),
        "raw_quantile_crossings": int(raw_crossings.sum()),
        "ordered_quantile_crossings": ordered_crossing_count,
        "missing_pred_q10": int(data["pred_q10"].isna().sum()),
        "missing_pred_q50": int(data["pred_q50"].isna().sum()),
        "missing_pred_q90": int(data["pred_q90"].isna().sum()),
    }
    pd.DataFrame([summary]).to_csv(
        outdir / "l123_001_integrity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


# =============================================================================
# L123-002
# =============================================================================

def audit_l123_002(
    balanced_path: Path,
    potential_path: Path,
    conservative_path: Path,
    union_path: Path,
    outdir: Path,
    figdir: Path,
    dpi: int,
) -> dict[str, Any]:
    balanced = add_mode_letter(read_csv(balanced_path))
    potential = add_mode_letter(read_csv(potential_path))
    conservative = add_mode_letter(read_csv(conservative_path))
    union = add_mode_letter(read_csv(union_path))

    required = ["catalyst_id", "template_id", "combination_id"]
    for label, frame in [
        ("Balanced Top-K", balanced),
        ("Potential Top-K", potential),
        ("Conservative Top-K", conservative),
        ("q50/q90 union", union),
    ]:
        require_columns(frame, required, label)
    require_columns(
        union,
        [
            "selected_by_balanced", "selected_by_potential",
            "package_selection_reason",
        ],
        "L123-002 union",
    )

    union["selected_by_balanced"] = to_bool(union["selected_by_balanced"])
    union["selected_by_potential"] = to_bool(union["selected_by_potential"])

    balanced_ids = set(balanced["combination_id"])
    potential_ids = set(potential["combination_id"])
    conservative_ids = set(conservative["combination_id"])
    union_ids = set(union["combination_id"])
    overlap_ids = balanced_ids & potential_ids
    expected_union = balanced_ids | potential_ids

    selection_counts = (
        union["package_selection_reason"]
        .value_counts(dropna=False)
        .rename_axis("selection_reason")
        .reset_index(name="rows")
    )
    selection_counts.to_csv(
        outdir / "l123_002_selection_reason_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_catalyst = (
        union.groupby("catalyst_id")
        .agg(
            union_rows=("combination_id", "size"),
            unique_templates=("template_id", "nunique"),
            balanced_rows=("selected_by_balanced", "sum"),
            potential_rows=("selected_by_potential", "sum"),
        )
        .reset_index()
    )
    per_catalyst.to_csv(
        outdir / "l123_002_per_catalyst_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mode_diversity = (
        union.groupby("mode_letter", dropna=False)
        .agg(
            union_rows=("combination_id", "size"),
            catalysts=("catalyst_id", "nunique"),
            templates=("template_id", "nunique"),
        )
        .reset_index()
    )
    mode_diversity.to_csv(
        outdir / "l123_002_mode_diversity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "balanced_rows": int(len(balanced)),
        "potential_rows": int(len(potential)),
        "conservative_rows": int(len(conservative)),
        "union_rows": int(len(union)),
        "balanced_potential_overlap_rows": int(len(overlap_ids)),
        "expected_union_rows": int(len(expected_union)),
        "union_matches_set_formula": union_ids == expected_union,
        "jaccard_balanced_potential": (
            len(overlap_ids) / len(expected_union) if expected_union else np.nan
        ),
        "conservative_overlap_with_union": int(len(conservative_ids & union_ids)),
        "union_duplicate_combination_ids": int(
            union["combination_id"].duplicated().sum()
        ),
    }
    pd.DataFrame([summary]).to_csv(
        outdir / "l123_002_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.6), layout="constrained")
    ax.bar(
        ["Balanced", "Potential", "Overlap", "Union"],
        [len(balanced_ids), len(potential_ids), len(overlap_ids), len(union_ids)],
    )
    ax.set_ylabel("Number of route packages")
    ax.set_title("L123-002 q50/q90 selection overlap")
    save_figure(fig, figdir / "l123_002_selection_overlap.png", dpi)

    fig, ax = plt.subplots(figsize=(8.0, 5.6), layout="constrained")
    if not per_catalyst.empty:
        x = np.arange(len(per_catalyst))
        ax.bar(x, per_catalyst["union_rows"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            per_catalyst["catalyst_id"], rotation=30, ha="right"
        )
    ax.set_xlabel("Catalyst")
    ax.set_ylabel("q50/q90 union packages")
    ax.set_title("L123-002 retained packages per catalyst")
    save_figure(fig, figdir / "l123_002_packages_per_catalyst.png", dpi)

    return summary


# =============================================================================
# L123-003
# =============================================================================

def parse_grid(summary: dict[str, Any]) -> dict[str, float]:
    grid = summary.get("grid", {})
    np_grid = grid.get("np_size_nm")
    cm_grid = grid.get("C_over_M_raw")
    if not isinstance(np_grid, list) or len(np_grid) != 3:
        raise RuntimeError("Invalid np_size_nm grid in L123-003 summary.")
    if not isinstance(cm_grid, list) or len(cm_grid) != 3:
        raise RuntimeError("Invalid C_over_M_raw grid in L123-003 summary.")
    return {
        "np_min": float(np_grid[0]),
        "np_max": float(np_grid[1]),
        "np_step": float(np_grid[2]),
        "cm_min": float(cm_grid[0]),
        "cm_max": float(cm_grid[1]),
        "cm_step": float(cm_grid[2]),
    }


def at_boundary(values: pd.Series, boundary: float, step: float) -> pd.Series:
    tolerance = max(abs(step) * 1e-7, 1e-10)
    return pd.Series(
        np.isclose(
            to_num(values).to_numpy(float),
            boundary,
            atol=tolerance,
            rtol=0.0,
        ),
        index=values.index,
    )


def audit_l123_003(
    recommendations_path: Path,
    top_points_path: Path,
    summary_path: Path,
    outdir: Path,
    figdir: Path,
    dpi: int,
) -> dict[str, Any]:
    recommendations = add_mode_letter(read_csv(recommendations_path))
    top_points = add_mode_letter(read_csv(top_points_path))
    run_summary = read_json(summary_path)
    limits = parse_grid(run_summary)

    require_columns(
        recommendations,
        [
            "catalyst_id", "template_id", "combination_id",
            "recommended_np_size_nm_balanced",
            "recommended_C_over_M_balanced",
            "recommended_pred_q50_balanced",
            "recommended_np_size_nm_potential",
            "recommended_C_over_M_potential",
            "recommended_pred_q90_potential",
        ],
        "L123-003 recommendations",
    )
    require_columns(
        top_points,
        [
            "catalyst_id", "template_id", "combination_id",
            "grid_np_size_nm", "grid_C_over_M", "selection_reason",
        ],
        "L123-003 top points",
    )

    id_columns = [
        column for column in [
            "catalyst_id", "template_id", "combination_id",
            "growth_mode", "mode_letter", "package_selection_reason",
        ]
        if column in recommendations.columns
    ]

    objective_map = {
        "balanced": (
            "recommended_np_size_nm_balanced",
            "recommended_C_over_M_balanced",
            "recommended_pred_q50_balanced",
        ),
        "potential": (
            "recommended_np_size_nm_potential",
            "recommended_C_over_M_potential",
            "recommended_pred_q90_potential",
        ),
    }

    parts = []
    for objective, (np_col, cm_col, score_col) in objective_map.items():
        part = recommendations[id_columns].copy()
        part["objective"] = objective
        part["recommended_np_size_nm"] = to_num(recommendations[np_col])
        part["recommended_C_over_M"] = to_num(recommendations[cm_col])
        part["objective_score"] = to_num(recommendations[score_col])
        part["np_at_lower_bound"] = at_boundary(
            part["recommended_np_size_nm"], limits["np_min"], limits["np_step"]
        )
        part["np_at_upper_bound"] = at_boundary(
            part["recommended_np_size_nm"], limits["np_max"], limits["np_step"]
        )
        part["cm_at_lower_bound"] = at_boundary(
            part["recommended_C_over_M"], limits["cm_min"], limits["cm_step"]
        )
        part["cm_at_upper_bound"] = at_boundary(
            part["recommended_C_over_M"], limits["cm_max"], limits["cm_step"]
        )
        part["any_boundary"] = part[
            [
                "np_at_lower_bound", "np_at_upper_bound",
                "cm_at_lower_bound", "cm_at_upper_bound",
            ]
        ].any(axis=1)
        parts.append(part)

    boundary_audit = pd.concat(parts, ignore_index=True)
    boundary_audit.to_csv(
        outdir / "l123_003_boundary_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    boundary_summary = (
        boundary_audit.groupby("objective")
        .agg(
            route_packages=("combination_id", "size"),
            np_lower_count=("np_at_lower_bound", "sum"),
            np_upper_count=("np_at_upper_bound", "sum"),
            cm_lower_count=("cm_at_lower_bound", "sum"),
            cm_upper_count=("cm_at_upper_bound", "sum"),
            any_boundary_count=("any_boundary", "sum"),
        )
        .reset_index()
    )
    for name in ["np_lower", "np_upper", "cm_lower", "cm_upper", "any_boundary"]:
        boundary_summary[f"{name}_fraction"] = (
            boundary_summary[f"{name}_count"]
            / boundary_summary["route_packages"]
        )
    boundary_summary.to_csv(
        outdir / "l123_003_boundary_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    same_state = (
        np.isclose(
            to_num(recommendations["recommended_np_size_nm_balanced"]),
            to_num(recommendations["recommended_np_size_nm_potential"]),
            equal_nan=False,
        )
        & np.isclose(
            to_num(recommendations["recommended_C_over_M_balanced"]),
            to_num(recommendations["recommended_C_over_M_potential"]),
            equal_nan=False,
        )
    )
    agreement = recommendations[id_columns].copy()
    agreement["balanced_and_potential_same_state"] = same_state
    agreement.to_csv(
        outdir / "l123_003_balanced_potential_state_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_audit = (
        top_points.groupby("combination_id")
        .agg(
            selected_grid_points=("combination_id", "size"),
            unique_np_sizes=("grid_np_size_nm", "nunique"),
            unique_C_over_M=("grid_C_over_M", "nunique"),
            balanced_selected_rows=(
                "selection_reason",
                lambda s: int(
                    s.astype(str).isin(["balanced", "balanced+potential"]).sum()
                ),
            ),
            potential_selected_rows=(
                "selection_reason",
                lambda s: int(
                    s.astype(str).isin(["potential", "balanced+potential"]).sum()
                ),
            ),
        )
        .reset_index()
    )
    top_audit.to_csv(
        outdir / "l123_003_top_points_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    indexed = boundary_summary.set_index("objective")
    summary = {
        "route_packages": int(len(recommendations)),
        "top_point_rows": int(len(top_points)),
        "grid": limits,
        "balanced_potential_same_state_rows": int(same_state.sum()),
        "balanced_potential_same_state_fraction": float(same_state.mean()),
        "quantile_crossing_points_detected": int(
            run_summary.get("quantile_crossing_points_detected", 0)
        ),
        "balanced_boundary_fraction": float(
            indexed.loc["balanced", "any_boundary_fraction"]
        ),
        "potential_boundary_fraction": float(
            indexed.loc["potential", "any_boundary_fraction"]
        ),
        "balanced_np_lower_fraction": float(
            indexed.loc["balanced", "np_lower_fraction"]
        ),
        "potential_np_lower_fraction": float(
            indexed.loc["potential", "np_lower_fraction"]
        ),
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.8), layout="constrained")
    categories = ["NP lower", "NP upper", "C/M lower", "C/M upper"]
    balanced_values = [
        indexed.loc["balanced", "np_lower_fraction"],
        indexed.loc["balanced", "np_upper_fraction"],
        indexed.loc["balanced", "cm_lower_fraction"],
        indexed.loc["balanced", "cm_upper_fraction"],
    ]
    potential_values = [
        indexed.loc["potential", "np_lower_fraction"],
        indexed.loc["potential", "np_upper_fraction"],
        indexed.loc["potential", "cm_lower_fraction"],
        indexed.loc["potential", "cm_upper_fraction"],
    ]
    x = np.arange(len(categories))
    width = 0.38
    ax.bar(x - width / 2, balanced_values, width, label="Balanced q50")
    ax.bar(x + width / 2, potential_values, width, label="Potential q90")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Fraction of route packages")
    ax.set_ylim(0, 1.05)
    ax.set_title("L123-003 physical-state boundary saturation")
    ax.legend(frameon=False)
    save_figure(fig, figdir / "l123_003_boundary_saturation.png", dpi)

    return summary


# =============================================================================
# Formal validation outputs
# =============================================================================

def collect_validation_outputs(
    project_dir: Path,
    outdir: Path,
) -> pd.DataFrame:
    target_dir = outdir / "collected_validation"
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for label, relative in VALIDATION_RELATIVE_PATHS.items():
        source = project_dir / relative
        destination = target_dir / f"{label}.csv"
        exists = source.is_file()
        row = {
            "validation_output": label,
            "source_path": str(source),
            "exists": exists,
            "rows": np.nan,
            "columns": np.nan,
            "copied_to": "",
        }
        if exists:
            frame = read_csv(source)
            frame.to_csv(destination, index=False, encoding="utf-8-sig")
            row.update({
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "copied_to": str(destination),
            })
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        outdir / "validation_outputs_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


# =============================================================================
# Pipeline status
# =============================================================================

def pipeline_status(
    worldmap: dict[str, Any],
    stage001: dict[str, Any],
    stage002: dict[str, Any],
    stage003: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        ("WorldMap", "No duplicate combination_id",
         worldmap["duplicate_combination_ids"] == 0,
         worldmap["duplicate_combination_ids"], "integrity"),
        ("WorldMap", "Ordered audit q10<=q50<=q90",
         worldmap["ordered_quantile_crossings"] == 0,
         worldmap["ordered_quantile_crossings"], "integrity"),
        ("WorldMap", "No negative ordered interval width",
         worldmap["negative_ordered_interval_width_rows"] == 0,
         worldmap["negative_ordered_interval_width_rows"], "integrity"),
        ("L123-001", "No duplicate combination_id",
         stage001["duplicate_combination_ids"] == 0,
         stage001["duplicate_combination_ids"], "integrity"),
        ("L123-001", "Ordered audit q10<=q50<=q90",
         pd.isna(stage001["ordered_quantile_crossings"])
         or stage001["ordered_quantile_crossings"] == 0,
         stage001["ordered_quantile_crossings"], "integrity"),
        ("L123-002", "q50/q90 union matches set formula",
         bool(stage002["union_matches_set_formula"]),
         f"{stage002['union_rows']}/{stage002['expected_union_rows']}",
         "integrity"),
        ("L123-002", "No duplicate union combination_id",
         stage002["union_duplicate_combination_ids"] == 0,
         stage002["union_duplicate_combination_ids"], "integrity"),
        ("L123-003", "Physical-state recommendations exist",
         stage003["route_packages"] > 0,
         stage003["route_packages"], "integrity"),
        ("L123-003", "Balanced search is not strongly boundary-saturated",
         stage003["balanced_boundary_fraction"] < 0.5,
         stage003["balanced_boundary_fraction"], "scientific_warning"),
        ("L123-003", "Potential search is not strongly boundary-saturated",
         stage003["potential_boundary_fraction"] < 0.5,
         stage003["potential_boundary_fraction"], "scientific_warning"),
    ]

    output = pd.DataFrame(
        rows,
        columns=["stage", "check", "passed", "value", "check_type"],
    )
    output["status"] = np.where(
        output["passed"],
        "PASS",
        np.where(
            output["check_type"].eq("scientific_warning"),
            "WARN",
            "FAIL",
        ),
    )
    return output


# =============================================================================
# Optional L123 run resolution
# =============================================================================

def l123_paths_from_root(root: Path) -> tuple[str, dict[str, Path]] | None:
    """
    Resolve one complete L123 run.

    Supported layouts
    -----------------
    1. Current formal project:
       <root>/outputs/recommendation/rank_candidate_routes
       <root>/outputs/recommendation/rank_catalysts
       <root>/outputs/recommendation/optimize_physical_state

    2. App run:
       <root>/rank_candidate_routes
       <root>/rank_catalysts
       <root>/optimize_physical_state
    """
    root = root.expanduser().resolve()
    layouts = [
        (
            "current_project",
            root / "outputs" / "recommendation" / "rank_candidate_routes",
            root / "outputs" / "recommendation" / "rank_catalysts",
            root / "outputs" / "recommendation" / "optimize_physical_state",
        ),
        (
            "app_run",
            root / "rank_candidate_routes",
            root / "rank_catalysts",
            root / "optimize_physical_state",
        ),
    ]

    for layout_name, stage001, stage002, stage003 in layouts:
        paths = {
            "l123_001_shortlist":
                stage001 / "shortlisted_route_predictions.csv",
            "l123_002_balanced":
                stage002 / "process_packages_balanced_topK.csv",
            "l123_002_potential":
                stage002 / "process_packages_potential_topK.csv",
            "l123_002_conservative":
                stage002 / "process_packages_conservative_topK.csv",
            "l123_002_union":
                stage002 / "process_packages_union_q50_q90.csv",
            "l123_003_recommendations":
                stage003 / "process_packages_with_physical_state.csv",
            "l123_003_top_points":
                stage003 / "physical_state_top_points.csv",
            "l123_003_summary":
                stage003 / "run_summary_L123_003.json",
        }
        if all(path.is_file() for path in paths.values()):
            return layout_name, paths

    return None


def l123_missing_report(root: Path) -> str:
    root = root.expanduser().resolve()
    lines = [f"No complete L123 layout was found under:\n  {root}"]
    layouts = [
        (
            "current_project",
            root / "outputs" / "recommendation" / "rank_candidate_routes",
            root / "outputs" / "recommendation" / "rank_catalysts",
            root / "outputs" / "recommendation" / "optimize_physical_state",
        ),
        (
            "app_run",
            root / "rank_candidate_routes",
            root / "rank_catalysts",
            root / "optimize_physical_state",
        ),
    ]
    for label, stage001, stage002, stage003 in layouts:
        checks = [
            stage001 / "shortlisted_route_predictions.csv",
            stage002 / "process_packages_balanced_topK.csv",
            stage002 / "process_packages_potential_topK.csv",
            stage002 / "process_packages_conservative_topK.csv",
            stage002 / "process_packages_union_q50_q90.csv",
            stage003 / "process_packages_with_physical_state.csv",
            stage003 / "physical_state_top_points.csv",
            stage003 / "run_summary_L123_003.json",
        ]
        lines.append(f"\nLayout {label}:")
        for path in checks:
            lines.append(f"  [{'OK' if path.is_file() else 'MISSING'}] {path}")
    return "\n".join(lines)


def find_latest_complete_l123_run(project_dir: Path) -> tuple[Path, str, dict[str, Path]]:
    candidate_roots: list[Path] = [project_dir]

    app_base = project_dir / "outputs" / "app"
    if app_base.is_dir():
        candidate_roots.extend(
            path for path in app_base.rglob("*")
            if path.is_dir()
        )

    unique_roots = list(dict.fromkeys(path.resolve() for path in candidate_roots))
    unique_roots.sort(
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )

    # Always prefer the formal project outputs when they are complete.
    formal = l123_paths_from_root(project_dir)
    if formal is not None:
        layout_name, paths = formal
        return project_dir, layout_name, paths

    for root in unique_roots:
        resolved = l123_paths_from_root(root)
        if resolved is not None:
            layout_name, paths = resolved
            return root, layout_name, paths

    raise FileNotFoundError(
        "No complete L123 run was found automatically.\n"
        "Run the recommendation pipeline first and provide a selected app run with:\n"
        "  --scope l123 --l123_run <run_directory>\n"
        "or use the default --scope worldmap, which does not require L123 outputs."
    )


def resolve_l123_run(
    project_dir: Path,
    value: str,
) -> tuple[Path, str, dict[str, Path]]:
    if str(value).strip().lower() == "auto":
        return find_latest_complete_l123_run(project_dir)

    root = resolve_path(project_dir, value)
    if not root.is_dir():
        raise FileNotFoundError(f"L123 run directory not found:\n  {root}")

    resolved = l123_paths_from_root(root)
    if resolved is None:
        raise FileNotFoundError(
            f"The selected directory is not a complete L123 run:\n  {root}\n\n"
            + l123_missing_report(root)
        )

    layout_name, paths = resolved
    return root, layout_name, paths


def pipeline_status_optional(
    worldmap: dict[str, Any] | None,
    stage001: dict[str, Any] | None,
    stage002: dict[str, Any] | None,
    stage003: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []

    if worldmap is not None:
        rows.extend([
            (
                "WorldMap",
                "No duplicate combination_id",
                worldmap["duplicate_combination_ids"] == 0,
                worldmap["duplicate_combination_ids"],
                "integrity",
            ),
            (
                "WorldMap",
                "Ordered audit q10<=q50<=q90",
                worldmap["ordered_quantile_crossings"] == 0,
                worldmap["ordered_quantile_crossings"],
                "integrity",
            ),
            (
                "WorldMap",
                "No negative ordered interval width",
                worldmap["negative_ordered_interval_width_rows"] == 0,
                worldmap["negative_ordered_interval_width_rows"],
                "integrity",
            ),
            (
                "WorldMap",
                "Route-dependent N/C physical consistency",
                worldmap["physical_consistency_failed_checks"] == 0
                and worldmap["physical_consistency_mismatch_rows"] == 0,
                (
                    f"failed_checks={worldmap['physical_consistency_failed_checks']}, "
                    f"mismatch_rows={worldmap['physical_consistency_mismatch_rows']}"
                ),
                "integrity",
            ),
        ])

    if stage001 is not None:
        rows.extend([
            (
                "L123-001",
                "No duplicate combination_id",
                stage001["duplicate_combination_ids"] == 0,
                stage001["duplicate_combination_ids"],
                "integrity",
            ),
            (
                "L123-001",
                "Ordered audit q10<=q50<=q90",
                pd.isna(stage001["ordered_quantile_crossings"])
                or stage001["ordered_quantile_crossings"] == 0,
                stage001["ordered_quantile_crossings"],
                "integrity",
            ),
        ])

    if stage002 is not None:
        rows.extend([
            (
                "L123-002",
                "q50/q90 union matches set formula",
                bool(stage002["union_matches_set_formula"]),
                f"{stage002['union_rows']}/{stage002['expected_union_rows']}",
                "integrity",
            ),
            (
                "L123-002",
                "No duplicate union combination_id",
                stage002["union_duplicate_combination_ids"] == 0,
                stage002["union_duplicate_combination_ids"],
                "integrity",
            ),
        ])

    if stage003 is not None:
        rows.extend([
            (
                "L123-003",
                "Physical-state recommendations exist",
                stage003["route_packages"] > 0,
                stage003["route_packages"],
                "integrity",
            ),
            (
                "L123-003",
                "Balanced search is not strongly boundary-saturated",
                stage003["balanced_boundary_fraction"] < 0.5,
                stage003["balanced_boundary_fraction"],
                "scientific_warning",
            ),
            (
                "L123-003",
                "Potential search is not strongly boundary-saturated",
                stage003["potential_boundary_fraction"] < 0.5,
                stage003["potential_boundary_fraction"],
                "scientific_warning",
            ),
        ])

    output = pd.DataFrame(
        rows,
        columns=["stage", "check", "passed", "value", "check_type"],
    )
    if output.empty:
        output["status"] = pd.Series(dtype=str)
        return output

    output["status"] = np.where(
        output["passed"],
        "PASS",
        np.where(
            output["check_type"].eq("scientific_warning"),
            "WARN",
            "FAIL",
        ),
    )
    return output


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen CNT dataset and WorldMap independently, with an "
            "optional L123 candidate-screening audit."
        )
    )
    parser.add_argument("--project_dir", default=str(PROJECT_DIR))
    parser.add_argument(
        "--scope",
        default="worldmap",
        choices=["worldmap", "l123", "all"],
        help=(
            "worldmap: dataset + frozen WorldMap + formal validation (default); "
            "l123: one selected L123 run only; all: both."
        ),
    )
    parser.add_argument(
        "--worldmap_run",
        default="auto",
        help="'auto' uses the newest outputs/app/phase_diagram_worldmap/run_*.",
    )
    parser.add_argument(
        "--l123_run",
        default="auto",
        help=(
            "L123 run directory. 'auto' finds the newest complete web/Fusion/fixed "
            "run. Ignored when --scope worldmap."
        ),
    )
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    outdir = resolve_path(project_dir, args.outdir)

    run_worldmap_stage = args.scope in {"worldmap", "all"}
    run_l123_stage = args.scope in {"l123", "all"}

    worldmap_run: Path | None = None
    l123_run: Path | None = None
    l123_layout = "not_requested"
    paths: dict[str, Path] = {}

    if run_worldmap_stage:
        worldmap_root = project_dir / "outputs" / "app" / "phase_diagram_worldmap"
        worldmap_run = (
            latest_worldmap_run(worldmap_root)
            if args.worldmap_run == "auto"
            else resolve_path(project_dir, args.worldmap_run)
        )
        if not worldmap_run.is_dir():
            raise FileNotFoundError(f"WorldMap run not found:\n  {worldmap_run}")

        paths.update({
            "dataset_full60D":
                project_dir / "data" / "11_flattened" / "ml_dataset_3p0_full60D.csv",
            "worldmap_candidates":
                worldmap_run / "PHASE_CANDIDATES_sampled.csv",
            "worldmap_catalyst_states":
                worldmap_run / "CATALYST_STATES.csv",
            "worldmap_route_states":
                worldmap_run / "ROUTE_STATES.csv",
            "worldmap_pca_audit":
                worldmap_run / "BLOCK_PCA_AUDIT.csv",
        })

    if run_l123_stage:
        l123_run, l123_layout, l123_paths = resolve_l123_run(
            project_dir,
            args.l123_run,
        )
        paths.update(l123_paths)

    # Only files required by the requested scope are checked.
    for label, path in paths.items():
        require_file(path, label)

    if outdir.exists():
        shutil.rmtree(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("[CNT WORLDMAP OUTPUT AUDIT START]")
    print(f"Script revision   : {SCRIPT_REVISION}")
    print(f"Audit scope       : {args.scope}")
    print(f"Project directory : {project_dir}")
    print(f"WorldMap run      : {worldmap_run if worldmap_run else 'NOT REQUESTED'}")
    print(f"L123 run          : {l123_run if l123_run else 'NOT REQUESTED'}")
    print(f"L123 layout       : {l123_layout}")
    print(f"Output directory  : {outdir}")
    print("=" * 112)

    build_input_manifest(paths).to_csv(
        outdir / "input_file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dataset_info: dict[str, Any] | None = None
    worldmap: dict[str, Any] | None = None
    stage001: dict[str, Any] | None = None
    stage002: dict[str, Any] | None = None
    stage003: dict[str, Any] | None = None

    if run_worldmap_stage:
        dataset_info = audit_dataset_composition(
            paths["dataset_full60D"],
            outdir,
            figdir,
            args.dpi,
        )
        worldmap = audit_worldmap(
            worldmap_run,
            outdir,
            figdir,
            args.dpi,
        )
        validation_manifest = collect_validation_outputs(project_dir, outdir)
    else:
        validation_manifest = pd.DataFrame(
            columns=[
                "validation_output",
                "source_path",
                "searched_paths",
                "exists",
                "rows",
                "columns",
                "copied_to",
            ]
        )
        validation_manifest.to_csv(
            outdir / "validation_outputs_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if run_l123_stage:
        stage001 = audit_l123_001(
            paths["l123_001_shortlist"],
            outdir,
        )
        stage002 = audit_l123_002(
            paths["l123_002_balanced"],
            paths["l123_002_potential"],
            paths["l123_002_conservative"],
            paths["l123_002_union"],
            outdir,
            figdir,
            args.dpi,
        )
        stage003 = audit_l123_003(
            paths["l123_003_recommendations"],
            paths["l123_003_top_points"],
            paths["l123_003_summary"],
            outdir,
            figdir,
            args.dpi,
        )

    status = pipeline_status_optional(
        worldmap,
        stage001,
        stage002,
        stage003,
    )
    status.to_csv(
        outdir / "pipeline_integrity_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failures = status[status["status"].eq("FAIL")] if not status.empty else status
    warnings = status[status["status"].eq("WARN")] if not status.empty else status

    summary = {
        "script_revision": SCRIPT_REVISION,
        "audit_scope": args.scope,
        "project_directory": str(project_dir),
        "worldmap_run": str(worldmap_run) if worldmap_run else None,
        "l123_run": str(l123_run) if l123_run else None,
        "l123_layout": l123_layout,
        "definitions": {
            "balanced": "raw q50",
            "potential": "raw q90",
            "conservative": "raw q10",
            "interval_width": "ordered q90-q10; descriptive only",
            "physical_search": "np_size_nm and raw C_over_M",
        },
        "dataset_composition": dataset_info,
        "worldmap": worldmap,
        "l123_001": stage001,
        "l123_002": stage002,
        "l123_003": stage003,
        "pipeline_checks": {
            "pass": int(status["status"].eq("PASS").sum()) if not status.empty else 0,
            "warn": int(status["status"].eq("WARN").sum()) if not status.empty else 0,
            "fail": int(status["status"].eq("FAIL").sum()) if not status.empty else 0,
            "failed_checks": (
                failures[["stage", "check", "value"]].to_dict("records")
                if not failures.empty else []
            ),
            "warnings": (
                warnings[["stage", "check", "value"]].to_dict("records")
                if not warnings.empty else []
            ),
        },
        "formal_validation_outputs": {
            "available": (
                int(validation_manifest["exists"].sum())
                if "exists" in validation_manifest.columns else 0
            ),
            "expected": int(len(validation_manifest)),
        },
        "output_directory": str(outdir),
    }
    write_json(summary, outdir / "evaluation_summary.json")

    print("\n[RESULT SUMMARY]")

    if dataset_info is not None and worldmap is not None:
        print(
            "Dataset papers/samples       : "
            f"{dataset_info['unique_papers']}/{dataset_info['unique_sample_ids']}"
        )
        print(
            "Dataset levels L1/L2/L3      : "
            + "/".join(
                str(dataset_info["level_sample_counts"].get(level, 0))
                for level in LEVEL_ORDER
            )
        )
        print(
            "Dataset modes A/B/C/D/E      : "
            + "/".join(
                str(dataset_info["mode_sample_counts"].get(mode, 0))
                for mode in MODE_ORDER
            )
        )
        print(f"WorldMap candidates          : {worldmap['candidate_rows']}")
        print(f"WorldMap catalyst states     : {worldmap['catalyst_state_rows']}")
        print(f"WorldMap route states        : {worldmap['route_state_rows']}")
    else:
        print("Dataset/WorldMap audit       : NOT REQUESTED")

    if stage001 is not None and stage002 is not None and stage003 is not None:
        print(
            "L123-001 routes/catalysts    : "
            f"{stage001['route_rows']}/{stage001['catalysts']}"
        )
        print(
            "L123-002 q50/q90 overlap     : "
            f"{stage002['balanced_potential_overlap_rows']}"
        )
        print(f"L123-002 q50/q90 union       : {stage002['union_rows']}")
        print(
            "L123-003 balanced boundary   : "
            f"{stage003['balanced_boundary_fraction']:.3f}"
        )
        print(
            "L123-003 potential boundary  : "
            f"{stage003['potential_boundary_fraction']:.3f}"
        )
    else:
        print("L123 audit                   : NOT REQUESTED")

    print(
        "Pipeline PASS/WARN/FAIL      : "
        f"{int(status['status'].eq('PASS').sum()) if not status.empty else 0}/"
        f"{int(status['status'].eq('WARN').sum()) if not status.empty else 0}/"
        f"{int(status['status'].eq('FAIL').sum()) if not status.empty else 0}"
    )
    print(
        "Formal validation files found: "
        f"{int(validation_manifest['exists'].sum()) if 'exists' in validation_manifest.columns else 0}/"
        f"{len(validation_manifest)}"
    )

    if not warnings.empty:
        print("\n[SCIENTIFIC WARNINGS]")
        for _, row in warnings.iterrows():
            print(f"- {row['stage']} | {row['check']} | value={row['value']}")

    if not failures.empty:
        print("\n[INTEGRITY FAILURES]")
        for _, row in failures.iterrows():
            print(f"- {row['stage']} | {row['check']} | value={row['value']}")

    print(f"\nOutput directory             : {outdir}")
    print(f"Summary JSON                 : {outdir / 'evaluation_summary.json'}")
    print(f"Pipeline status              : {outdir / 'pipeline_integrity_status.csv'}")
    print("\n[OK] CNT WorldMap output audit completed.")
    print("=" * 112)


if __name__ == "__main__":
    main()
