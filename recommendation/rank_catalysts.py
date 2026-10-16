# rank_catalysts.py
# =============================================================================
# CNT 3.0 — catalyst-level ranking and q50/q90 route-package union
#
# Input:
#   outputs/recommendation/rank_candidate_routes/shortlisted_route_predictions.csv
#
# Outputs:
#   outputs/recommendation/rank_catalysts/catalyst_rank_balanced.csv
#   outputs/recommendation/rank_catalysts/catalyst_rank_potential.csv
#   outputs/recommendation/rank_catalysts/catalyst_rank_conservative.csv
#   outputs/recommendation/rank_catalysts/catalyst_rank_comparison.csv
#   outputs/recommendation/rank_catalysts/process_packages_balanced_topK.csv
#   outputs/recommendation/rank_catalysts/process_packages_potential_topK.csv
#   outputs/recommendation/rank_catalysts/process_packages_conservative_topK.csv
#   outputs/recommendation/rank_catalysts/process_packages_union_q50_q90.csv
#   outputs/recommendation/rank_catalysts/catalyst_summary_L123_002.json
#
# Definitions:
#   Balanced     = raw pred_q50
#   Potential    = raw pred_q90
#   Conservative = raw pred_q10
#   Catalyst score = mean of its top-N route scores
#   ordered q90-q10 = descriptive interval width only
#
# Intentionally removed:
#   legacy inputs/outputs, risk penalties, beta_risk, use_topk_input,
#   score fallbacks, primary-metric switching, route-audit duplication,
#   and repeated quantile correction.
# =============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "recommendation"
    / "rank_candidate_routes"
    / "shortlisted_route_predictions.csv"
)
OUTDIR = PROJECT_DIR / "outputs" / "recommendation" / "rank_catalysts"

AGGREGATE_TOP_N = 5
TOPK_BALANCED = 5
TOPK_POTENTIAL = 5
TOPK_CONSERVATIVE = 5

CORE_COLUMNS = [
    "catalyst_id", "template_id", "combination_id",
    "pred_q10", "pred_q50", "pred_q90",
]

PROFILE_COLUMNS = [
    "recipe_key", "metal_set",
    "metal1", "metal2", "metal3",
    "frac1", "frac2", "frac3",
    "candidate_growth_mode", "growth_mode",
]


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(
            "rank_candidate_routes output is missing required columns:\n  "
            + "\n  ".join(missing)
        )


def positive_integer(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"rank_candidate_routes input not found:\n  {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, CORE_COLUMNS)
    data = df.copy()

    for column in ["catalyst_id", "template_id", "combination_id"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
        invalid = data[column].eq("") | data[column].str.upper().eq("NAN")
        if invalid.any():
            raise RuntimeError(
                f"{column} contains {int(invalid.sum())} invalid rows."
            )

    duplicates = int(data["combination_id"].duplicated().sum())
    if duplicates:
        raise RuntimeError(
            f"combination_id contains {duplicates} duplicates. "
            "Fix rank_candidate_routes rather than removing them in rank_catalysts."
        )

    for column in ["pred_q10", "pred_q50", "pred_q90"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        missing = int(data[column].isna().sum())
        if missing:
            raise RuntimeError(
                f"{column} contains {missing} missing/non-numeric values."
            )

    raw_matrix = data[["pred_q10", "pred_q50", "pred_q90"]].to_numpy(dtype=float)
    ordered = np.sort(raw_matrix, axis=1)

    data["pred_q10_ordered"] = ordered[:, 0]
    data["pred_q50_ordered"] = ordered[:, 1]
    data["pred_q90_ordered"] = ordered[:, 2]
    data["quantile_crossing_detected"] = (
        data["pred_q10"].gt(data["pred_q50"])
        | data["pred_q50"].gt(data["pred_q90"])
        | data["pred_q10"].gt(data["pred_q90"])
    )

    data["score_balanced"] = data["pred_q50"]
    data["score_potential"] = data["pred_q90"]
    data["score_conservative"] = data["pred_q10"]
    data["interval_width"] = data["pred_q90_ordered"] - data["pred_q10_ordered"]
    return data.reset_index(drop=True)


# =============================================================================
# CATALYST-LEVEL RANKING
# =============================================================================

def catalyst_profile(df: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in PROFILE_COLUMNS if column in df.columns]
    if not available:
        return df[["catalyst_id"]].drop_duplicates().reset_index(drop=True)
    return (
        df.groupby("catalyst_id", as_index=False, sort=True)[available]
        .first()
    )


def add_rank(
    df: pd.DataFrame,
    score_column: str,
    rank_column: str,
) -> pd.DataFrame:
    output = df.copy()
    output[rank_column] = (
        output[score_column]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    return output


def aggregate_catalysts(
    df: pd.DataFrame,
    score_column: str,
    label: str,
    top_n: int,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    ordered = df.sort_values(
        ["catalyst_id", score_column, "template_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    ordered["_route_rank"] = ordered.groupby("catalyst_id").cumcount() + 1

    best = (
        ordered.loc[
            ordered["_route_rank"].eq(1),
            ["catalyst_id", "template_id", "combination_id", score_column],
        ]
        .rename(
            columns={
                "template_id": f"best_template_id_{label}",
                "combination_id": f"best_combination_id_{label}",
                score_column: f"best_score_{label}",
            }
        )
    )

    top_stats = (
        ordered.loc[ordered["_route_rank"].le(top_n)]
        .groupby("catalyst_id")[score_column]
        .agg(
            mean_topN="mean",
            std_topN="std",
            min_topN="min",
            max_topN="max",
            n_topN="count",
        )
        .reset_index()
        .rename(
            columns={
                "mean_topN": f"mean_top{top_n}_score_{label}",
                "std_topN": f"std_top{top_n}_score_{label}",
                "min_topN": f"min_top{top_n}_score_{label}",
                "max_topN": f"max_top{top_n}_score_{label}",
                "n_topN": f"n_top{top_n}_routes_{label}",
            }
        )
    )

    all_stats = (
        ordered.groupby("catalyst_id")[score_column]
        .agg(
            median_score="median",
            mean_all="mean",
            std_all="std",
            min_all="min",
            max_all="max",
            route_count="count",
        )
        .reset_index()
        .rename(
            columns={
                "median_score": f"median_score_{label}",
                "mean_all": f"mean_all_score_{label}",
                "std_all": f"std_all_score_{label}",
                "min_all": f"min_all_score_{label}",
                "max_all": f"max_all_score_{label}",
                "route_count": f"route_count_{label}",
            }
        )
    )

    leaderboard = (
        best.merge(top_stats, on="catalyst_id", how="outer")
        .merge(all_stats, on="catalyst_id", how="outer")
        .merge(profile, on="catalyst_id", how="left")
    )

    primary = f"mean_top{top_n}_score_{label}"
    leaderboard[f"primary_catalyst_score_{label}"] = leaderboard[primary]
    leaderboard = add_rank(
        leaderboard,
        f"primary_catalyst_score_{label}",
        f"catalyst_rank_{label}",
    )

    return leaderboard.sort_values(
        [f"catalyst_rank_{label}", f"best_score_{label}"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def build_comparison(
    balanced: pd.DataFrame,
    potential: pd.DataFrame,
    conservative: pd.DataFrame,
    profile: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    def cols(label: str) -> list[str]:
        return [
            "catalyst_id",
            f"catalyst_rank_{label}",
            f"primary_catalyst_score_{label}",
            f"best_score_{label}",
            f"mean_top{top_n}_score_{label}",
            f"median_score_{label}",
            f"best_template_id_{label}",
            f"best_combination_id_{label}",
        ]

    comparison = (
        balanced[cols("balanced")]
        .merge(potential[cols("potential")], on="catalyst_id", how="outer")
        .merge(
            conservative[cols("conservative")],
            on="catalyst_id",
            how="outer",
        )
        .merge(profile, on="catalyst_id", how="left")
    )

    comparison["mean_rank_balanced_potential"] = comparison[
        ["catalyst_rank_balanced", "catalyst_rank_potential"]
    ].astype(float).mean(axis=1)

    comparison["rank_gap_balanced_potential"] = (
        comparison["catalyst_rank_balanced"].astype(float)
        - comparison["catalyst_rank_potential"].astype(float)
    ).abs()

    return comparison.sort_values(
        ["mean_rank_balanced_potential", "catalyst_rank_balanced"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# ROUTE-PACKAGE SELECTION
# =============================================================================

def select_topk_routes(
    df: pd.DataFrame,
    score_column: str,
    top_k: int,
    rank_column: str,
) -> pd.DataFrame:
    selected = df.sort_values(
        ["catalyst_id", score_column, "template_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    selected[rank_column] = selected.groupby("catalyst_id").cumcount() + 1
    return selected[selected[rank_column].le(top_k)].reset_index(drop=True)


def build_q50_q90_union(
    df: pd.DataFrame,
    balanced_topk: pd.DataFrame,
    potential_topk: pd.DataFrame,
) -> pd.DataFrame:
    balanced_rank = balanced_topk.set_index("combination_id")[
        "package_rank_balanced"
    ].to_dict()
    potential_rank = potential_topk.set_index("combination_id")[
        "package_rank_potential"
    ].to_dict()

    union = df[
        df["combination_id"].isin(set(balanced_rank) | set(potential_rank))
    ].copy()

    union["package_rank_balanced"] = union["combination_id"].map(balanced_rank)
    union["package_rank_potential"] = union["combination_id"].map(potential_rank)
    union["selected_by_balanced"] = union["combination_id"].isin(balanced_rank)
    union["selected_by_potential"] = union["combination_id"].isin(potential_rank)

    union["package_selection_reason"] = np.select(
        [
            union["selected_by_balanced"] & union["selected_by_potential"],
            union["selected_by_balanced"],
            union["selected_by_potential"],
        ],
        ["balanced+potential", "balanced", "potential"],
        default="",
    )

    union["best_available_package_rank"] = union[
        ["package_rank_balanced", "package_rank_potential"]
    ].min(axis=1, skipna=True)

    return union.sort_values(
        [
            "catalyst_id",
            "best_available_package_rank",
            "score_balanced",
            "score_potential",
        ],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# MAIN
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CNT 3.0 catalyst ranking and q50/q90 route-package union"
        )
    )
    parser.add_argument("--input_csv", default=str(INPUT_CSV))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument(
        "--aggregate_top_n",
        type=int,
        default=AGGREGATE_TOP_N,
    )
    parser.add_argument(
        "--topK_balanced",
        type=int,
        default=TOPK_BALANCED,
    )
    parser.add_argument(
        "--topK_potential",
        type=int,
        default=TOPK_POTENTIAL,
    )
    parser.add_argument(
        "--topK_conservative",
        type=int,
        default=TOPK_CONSERVATIVE,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    input_path = Path(args.input_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    aggregate_top_n = positive_integer(
        args.aggregate_top_n,
        "aggregate_top_n",
    )
    topk_balanced = positive_integer(args.topK_balanced, "topK_balanced")
    topk_potential = positive_integer(args.topK_potential, "topK_potential")
    topk_conservative = positive_integer(
        args.topK_conservative,
        "topK_conservative",
    )

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("[CATALYST RANKING START]")
    print(f"Project directory      : {PROJECT_DIR}")
    print(f"Input route-level CSV  : {input_path}")
    print(f"Output directory       : {outdir}")
    print(f"Aggregate top N        : {aggregate_top_n}")
    print(f"Balanced package Top-K : {topk_balanced}")
    print(f"Potential package Top-K: {topk_potential}")
    print(f"Conservative Top-K     : {topk_conservative}")
    print("=" * 108)

    routes = validate_and_prepare(read_input(input_path))
    profile = catalyst_profile(routes)

    balanced = aggregate_catalysts(
        routes,
        "score_balanced",
        "balanced",
        aggregate_top_n,
        profile,
    )
    potential = aggregate_catalysts(
        routes,
        "score_potential",
        "potential",
        aggregate_top_n,
        profile,
    )
    conservative = aggregate_catalysts(
        routes,
        "score_conservative",
        "conservative",
        aggregate_top_n,
        profile,
    )

    comparison = build_comparison(
        balanced,
        potential,
        conservative,
        profile,
        aggregate_top_n,
    )

    packages_balanced = select_topk_routes(
        routes,
        "score_balanced",
        topk_balanced,
        "package_rank_balanced",
    )
    packages_potential = select_topk_routes(
        routes,
        "score_potential",
        topk_potential,
        "package_rank_potential",
    )
    packages_conservative = select_topk_routes(
        routes,
        "score_conservative",
        topk_conservative,
        "package_rank_conservative",
    )
    packages_union = build_q50_q90_union(
        routes,
        packages_balanced,
        packages_potential,
    )

    output_paths = {
        "catalyst_balanced": outdir / "catalyst_rank_balanced.csv",
        "catalyst_potential": outdir / "catalyst_rank_potential.csv",
        "catalyst_conservative": outdir / "catalyst_rank_conservative.csv",
        "catalyst_comparison": outdir / "catalyst_rank_comparison.csv",
        "packages_balanced": outdir / "process_packages_balanced_topK.csv",
        "packages_potential": outdir / "process_packages_potential_topK.csv",
        "packages_conservative": (
            outdir / "process_packages_conservative_topK.csv"
        ),
        "packages_union": outdir / "process_packages_union_q50_q90.csv",
        "summary": outdir / "catalyst_summary_L123_002.json",
    }

    tables = {
        "catalyst_balanced": balanced,
        "catalyst_potential": potential,
        "catalyst_conservative": conservative,
        "catalyst_comparison": comparison,
        "packages_balanced": packages_balanced,
        "packages_potential": packages_potential,
        "packages_conservative": packages_conservative,
        "packages_union": packages_union,
    }

    for key, table in tables.items():
        table.to_csv(
            output_paths[key],
            index=False,
            encoding="utf-8-sig",
        )

    balanced_ids = set(packages_balanced["combination_id"])
    potential_ids = set(packages_potential["combination_id"])

    summary = {
        "project_directory": str(PROJECT_DIR),
        "input_csv": str(input_path),
        "n_route_rows": int(len(routes)),
        "n_catalysts": int(routes["catalyst_id"].nunique()),
        "n_templates": int(routes["template_id"].nunique()),
        "aggregate_top_n": int(aggregate_top_n),
        "primary_catalyst_metric": f"mean_top{aggregate_top_n}",
        "ranking_definition": {
            "balanced": "raw pred_q50",
            "potential": "raw pred_q90",
            "conservative": "raw pred_q10",
            "interval_width": (
                "ordered q90 - ordered q10; descriptive only; "
                "not used as a ranking penalty"
            ),
        },
        "quantile_crossing_rows_detected": int(
            routes["quantile_crossing_detected"].sum()
        ),
        "package_selection": {
            "balanced_topK_per_catalyst": int(topk_balanced),
            "potential_topK_per_catalyst": int(topk_potential),
            "conservative_topK_per_catalyst": int(topk_conservative),
            "union_rule": "Balanced Top-K union Potential Top-K",
            "n_balanced_package_rows": int(len(packages_balanced)),
            "n_potential_package_rows": int(len(packages_potential)),
            "n_conservative_package_rows": int(len(packages_conservative)),
            "n_union_package_rows": int(len(packages_union)),
            "n_q50_q90_overlap_rows": int(
                len(balanced_ids & potential_ids)
            ),
        },
        "outputs": {
            key: str(path)
            for key, path in output_paths.items()
        },
    }

    with output_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n[RESULT SUMMARY]")
    print(f"Route rows                  : {len(routes)}")
    print(f"Catalysts                   : {routes['catalyst_id'].nunique()}")
    print(f"Templates                   : {routes['template_id'].nunique()}")
    print(f"Balanced Top-K package rows : {len(packages_balanced)}")
    print(f"Potential Top-K package rows: {len(packages_potential)}")
    print(f"Conservative reference rows : {len(packages_conservative)}")
    print(f"q50/q90 union package rows  : {len(packages_union)}")
    print(f"q50/q90 overlap rows        : {len(balanced_ids & potential_ids)}")
    print("\n[OUTPUTS]")

    for label, path in output_paths.items():
        print(f"{label:24s}: {path}")

    print("\n[OK] Catalyst ranking completed.")
    print("=" * 108)


if __name__ == "__main__":
    main()
