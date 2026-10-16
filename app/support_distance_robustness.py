# -*- coding: utf-8 -*-
"""CNT WorldMap support-distance robustness audit (no CNT model retraining or prediction).

Place this script beside app/phase_diagram_worldmap.py and run it directly.
It reuses the original WorldMap baseline, then recomputes nearest-reference
physical distances with three alternative, explicitly defined metrics.
The original candidates, scores, and support labels are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import phase_diagram_worldmap as wm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SCHEMES = ["baseline", "equal_modules", "nc_emphasis", "std_scaling"]
SCHEME_LABELS = {
    "baseline": "Original",
    "equal_modules": "Equal N/C/F/P",
    "nc_emphasis": "N/C emphasis",
    "std_scaling": "Std scaling",
}
STATUS = ["supported", "near-boundary", "out-of-domain"]
STATUS_LABELS = ["Supported", "Near boundary", "Out of domain"]
BLOCKS = {"N": wm.N_COLS, "C": wm.C_COLS, "F": wm.F_COLS, "P": wm.P_COLS}
WEIGHTS = {
    "equal_modules": {"N": .25, "C": .25, "F": .25, "P": .25},
    "nc_emphasis": {"N": .30, "C": .30, "F": .20, "P": .20},
}


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", default=str(PROJECT_DIR))
    parser.add_argument("--dataset", default="data/11_flattened/ml_dataset_3p0_full60D.csv")
    parser.add_argument("--worldmap_run", default="", help="WorldMap run_* folder; default: latest")
    parser.add_argument("--outdir", default="", help="Default: <WorldMap run>/support_distance_robustness")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def absolute_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def find_run(root: Path, specified: str) -> Path:
    if specified:
        run = absolute_path(root, specified)
    else:
        parent = root / "outputs" / "app" / "phase_diagram_worldmap"
        runs = sorted(
            (p for p in parent.glob("run_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not runs:
            raise FileNotFoundError(f"WorldMap run_* folder not found: {parent}")
        run = runs[0]
    if not run.is_dir():
        raise FileNotFoundError(f"WorldMap folder not found: {run}")
    return run


def load_inputs(root: Path, run: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = absolute_path(root, dataset)
    candidates_path = run / "PHASE_CANDIDATES_sampled.csv"
    data = wm.prepare_dataset(wm.read_csv_smart(dataset_path))
    reference = wm.build_support_reference(data)
    candidates = wm.read_csv_smart(candidates_path).reset_index(drop=True)
    required = [
        "combination_id", "domain_status", "support_distance_full",
        "pred_q50_raw", "pred_q90_raw", *wm.MODEL_FEATURES,
    ]
    wm.require_columns(candidates, required, "WorldMap candidate CSV")
    if candidates.empty or candidates["combination_id"].isna().any():
        raise ValueError("Candidate CSV is empty or contains missing combination_id")
    if candidates["combination_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate combination_id; Top-K audit requires unique candidates")
    return reference, candidates


def recalculate_baseline(
    reference: pd.DataFrame, candidates: pd.DataFrame, run: Path,
) -> tuple[np.ndarray, np.ndarray, float, float, int, dict]:
    """Call the SAME fit and apply functions as the original WorldMap."""
    space = wm.fit_physical_support_space(reference, wm.PHYSICAL_COLS, "full")
    result = wm.apply_physical_support(candidates, space)
    distance = result["distance"].to_numpy(float)
    status = wm.classify_domain(pd.Series(distance), space.p95, space.p99).to_numpy()
    original_distance = pd.to_numeric(candidates["support_distance_full"], errors="coerce").to_numpy(float)
    original_status = candidates["domain_status"].astype(str).str.strip().str.lower().to_numpy()
    if not np.isfinite(original_distance).all():
        raise ValueError("Original support_distance_full contains missing/nonfinite values")
    abs_error = np.abs(distance - original_distance)
    different = int(np.count_nonzero(status != original_status))
    if not np.allclose(distance, original_distance, rtol=1e-7, atol=1e-8) or different:
        raise RuntimeError(
            "Original WorldMap support was NOT reproduced. Do not compare schemes.\n"
            f"  changed support labels: {different}\n"
            f"  maximum distance difference: {abs_error.max():.8g}\n"
            "Check that the dataset and the WorldMap run were generated together."
        )
    threshold_path = run / "SUPPORT_THRESHOLDS.csv"
    threshold_check: dict = {"threshold_csv_found": threshold_path.is_file()}
    if threshold_path.is_file():
        table = wm.read_csv_smart(threshold_path)
        wm.require_columns(table, ["space", "nn_distance_p95", "nn_distance_p99"], "SUPPORT_THRESHOLDS")
        row = table.loc[table["space"].astype(str).str.lower().eq("full")]
        if len(row) != 1:
            raise RuntimeError("SUPPORT_THRESHOLDS.csv must have exactly one 'full' row")
        p95_old = float(row.iloc[0]["nn_distance_p95"])
        p99_old = float(row.iloc[0]["nn_distance_p99"])
        if not np.allclose([space.p95, space.p99], [p95_old, p99_old], rtol=1e-7, atol=1e-8):
            raise RuntimeError("Original p95/p99 thresholds differ from saved WorldMap thresholds")
        threshold_check.update({"original_p95": p95_old, "original_p99": p99_old})
    threshold_check.update({
        "baseline_verified": True,
        "n_mismatched_labels": different,
        "max_absolute_distance_difference": float(abs_error.max()),
    })
    return distance, status, space.p95, space.p99, len(space.reference_vectors), threshold_check


def recalculate_variant(
    reference: pd.DataFrame, candidates: pd.DataFrame, scheme: str,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    features = wm.PHYSICAL_COLS
    raw_ref = wm.restore_physical_missing(reference, features)
    raw_cand = wm.restore_physical_missing(candidates, features)
    median = raw_ref.median().fillna(0.0)
    filled_ref = raw_ref.fillna(median)

    std = filled_ref.std(ddof=0).abs()
    if scheme == "std_scaling":
        scale = std.where(std > 1e-12, 1.0)
    else:
        iqr = (filled_ref.quantile(.75) - filled_ref.quantile(.25)).abs()
        scale = iqr.where(iqr > 1e-12, std).where(lambda s: s > 1e-12, 1.0)

    multipliers = pd.Series(1.0, index=features)
    if scheme in WEIGHTS:
        # Block-wise mean squared distance, followed by weighted sum.
        # The square-root converts the weighted sum to Euclidean distance.
        for block, cols in BLOCKS.items():
            multipliers.loc[cols] = np.sqrt(WEIGHTS[scheme][block] / len(cols))

    scaled_ref = (((filled_ref - median) / scale) * multipliers).to_numpy(float)
    scaled_cand = (((raw_cand.fillna(median) - median) / scale) * multipliers).to_numpy(float)
    if not np.isfinite(scaled_ref).all() or not np.isfinite(scaled_cand).all():
        raise ValueError(f"Nonfinite transformed physical values for scheme={scheme}")

    unique_rows = wm.deduplicate_vectors(scaled_ref)
    unique_ref = scaled_ref[unique_rows]
    if len(unique_ref) < 2:
        raise RuntimeError(f"Too few unique reference states for scheme={scheme}")

    # Leave-one-out nearest reference distances, matching original definitions.
    loo = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(unique_ref)
    train_nn = loo.kneighbors(unique_ref, return_distance=True)[0][:, 1]
    p95, p99 = np.quantile(train_nn, [wm.SUPPORT_P95, wm.SUPPORT_P99])
    model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(unique_ref)
    distances = model.kneighbors(scaled_cand, return_distance=True)[0][:, 0]
    labels = wm.classify_domain(pd.Series(distances), float(p95), float(p99)).to_numpy()
    return distances, labels, float(p95), float(p99), len(unique_ref)


def topk_index(candidates: pd.DataFrame, score: str, k: int) -> np.ndarray:
    # Stable scores; support classification is never used to select Top-K.
    work = candidates[["combination_id", score]].copy()
    work[score] = pd.to_numeric(work[score], errors="coerce")
    work = work.dropna(subset=[score]).sort_values(
        [score, "combination_id"], ascending=[False, True], kind="stable",
    )
    return work.head(k).index.to_numpy(dtype=int)


def assemble_tables(
    candidates: pd.DataFrame,
    results: dict[str, dict],
) -> dict[str, pd.DataFrame]:
    n = len(candidates)
    old_status = results["baseline"]["status"]
    baseline_supported = old_status == "supported"
    old_n_supported = int(baseline_supported.sum())
    scheme_rows, transition_rows, correlations, topk_rows, selected_rows = [], [], [], [], []

    candidate_table = candidates[["combination_id", "pred_q50_raw", "pred_q90_raw", "domain_status"]].copy()
    for scheme in SCHEMES:
        info = results[scheme]
        new_status = info["status"]
        dist = info["distance"]
        candidate_table[f"distance_{scheme}"] = dist
        candidate_table[f"domain_{scheme}"] = new_status
        counts = {s: int(np.count_nonzero(new_status == s)) for s in STATUS}
        retained = int(np.count_nonzero(baseline_supported & (new_status == "supported")))
        scheme_rows.append({
            "scheme": scheme, "description": SCHEME_LABELS[scheme],
            "reference_unique_states": info["n_unique"], "candidate_count": n,
            "p95": info["p95"], "p99": info["p99"],
            "n_supported": counts["supported"], "pct_supported": 100 * counts["supported"] / n,
            "n_near_boundary": counts["near-boundary"], "pct_near_boundary": 100 * counts["near-boundary"] / n,
            "n_out_of_domain": counts["out-of-domain"], "pct_out_of_domain": 100 * counts["out-of-domain"] / n,
            "original_n_supported": old_n_supported,
            "original_supported_retained_n": retained,
            "original_supported_retained_pct": 100 * retained / old_n_supported if old_n_supported else np.nan,
            "all_status_unchanged_n": int(np.count_nonzero(old_status == new_status)),
            "all_status_unchanged_pct": 100 * np.mean(old_status == new_status),
        })
        if scheme != "baseline":
            correlations.append({
                "scheme": scheme, "description": SCHEME_LABELS[scheme],
                "spearman_vs_original": float(pd.Series(results["baseline"]["distance"]).corr(
                    pd.Series(dist), method="spearman"
                )),
            })
            for before in STATUS:
                for after in STATUS:
                    count = int(np.count_nonzero((old_status == before) & (new_status == after)))
                    total = int(np.count_nonzero(old_status == before))
                    transition_rows.append({
                        "scheme": scheme, "original_class": before, "new_class": after,
                        "candidate_count": count,
                        "pct_within_original_class": 100 * count / total if total else np.nan,
                    })

    for criterion, col in [("q50", "pred_q50_raw"), ("q90", "pred_q90_raw")]:
        indices20 = topk_index(candidates, col, 20)
        for rank, idx in enumerate(indices20, start=1):
            row = {
                "criterion": criterion, "rank": rank,
                "in_top10": rank <= 10,
                "combination_id": candidates.iloc[idx]["combination_id"],
                "model_score": float(candidates.iloc[idx][col]),
            }
            for scheme in SCHEMES:
                row[f"domain_{scheme}"] = results[scheme]["status"][idx]
            selected_rows.append(row)
        for k in (10, 20):
            indices = topk_index(candidates, col, k)
            before = old_status[indices]
            for scheme in SCHEMES:
                after = results[scheme]["status"][indices]
                originally_supported = before == "supported"
                ns = int(originally_supported.sum())
                topk_rows.append({
                    "criterion": criterion, "k": k, "scheme": scheme,
                    "description": SCHEME_LABELS[scheme],
                    "n_candidates": len(indices),
                    "n_unchanged_class": int(np.count_nonzero(before == after)),
                    "pct_unchanged_class": 100 * np.mean(before == after),
                    "original_supported_n": ns,
                    "original_supported_retained_n": int(np.count_nonzero(originally_supported & (after == "supported"))),
                    "original_supported_retained_pct": (
                        100 * np.mean(after[originally_supported] == "supported") if ns else np.nan
                    ),
                    "new_supported_n": int(np.count_nonzero(after == "supported")),
                    "new_near_boundary_n": int(np.count_nonzero(after == "near-boundary")),
                    "new_out_of_domain_n": int(np.count_nonzero(after == "out-of-domain")),
                })

    return {
        "support_scheme_summary.csv": pd.DataFrame(scheme_rows),
        "support_class_transition.csv": pd.DataFrame(transition_rows),
        "support_distance_correlation.csv": pd.DataFrame(correlations),
        "support_topk_stability.csv": pd.DataFrame(topk_rows),
        "support_topk_candidate_changes.csv": pd.DataFrame(selected_rows),
        "support_candidate_comparison.csv": candidate_table,
    }


def plot_results(tables: dict[str, pd.DataFrame], outdir: Path, dpi: int) -> None:
    summary = tables["support_scheme_summary.csv"].set_index("scheme").loc[SCHEMES]
    topk = tables["support_topk_stability.csv"]
    corr = tables["support_distance_correlation.csv"].set_index("scheme")
    fig, axs = plt.subplots(2, 2, figsize=(15.2, 10.4), constrained_layout=True)
    x = np.arange(len(SCHEMES))
    names = [SCHEME_LABELS[s] for s in SCHEMES]

    # a: full candidate class distribution
    ax = axs[0, 0]
    bottom = np.zeros(len(SCHEMES))
    for key, label, color in [
        ("pct_supported", "Supported", "#5d9c82"),
        ("pct_near_boundary", "Near boundary", "#e3ae58"),
        ("pct_out_of_domain", "Out of domain", "#8d80a9"),
    ]:
        heights = summary[key].to_numpy(float)
        bars = ax.bar(x, heights, bottom=bottom, width=.67, label=label, color=color)
        for bar, base, v in zip(bars, bottom, heights):
            if v >= 8:
                ax.text(bar.get_x() + bar.get_width()/2, base+v/2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += heights
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Candidates (%)")
    ax.set_title("a  Support-class composition", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=9)

    # b: retention of original supported set vs all-status agreement
    ax = axs[0, 1]
    w = .35
    ax.bar(x-w/2, summary["original_supported_retained_pct"], width=w,
           color="#2873a9", label="Original Supported retained")
    ax.bar(x+w/2, summary["all_status_unchanged_pct"], width=w,
           color="#66b2a1", label="All classes unchanged")
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0, 125)
    ax.set_ylabel("Agreement with original (%)")
    ax.set_title("b  Classification stability", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    # c: unchanged support class among original model-ranked Top-10/Top-20
    ax = axs[1, 0]
    row_names = ["q50 Top-10", "q50 Top-20", "q90 Top-10", "q90 Top-20"]
    row_ids = [("q50", 10), ("q50", 20), ("q90", 10), ("q90", 20)]
    alternative = SCHEMES[1:]
    heat = np.empty((4, 3))
    for i, (q, k) in enumerate(row_ids):
        for j, scheme in enumerate(alternative):
            row = topk.loc[(topk["criterion"] == q) & (topk["k"] == k) & (topk["scheme"] == scheme)]
            heat[i, j] = float(row.iloc[0]["pct_unchanged_class"])
    im = ax.imshow(heat, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(3), [SCHEME_LABELS[s] for s in alternative], rotation=14)
    ax.set_yticks(np.arange(4), row_names)
    for i in range(4):
        for j in range(3):
            val = heat[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color="white" if val > 68 else "black", fontsize=11, fontweight="bold")
    ax.set_title("c  Original Top-K support-class retention", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Unchanged class (%)", fraction=.046, pad=.035)

    # d: Spearman correlation of physical nearest-neighbour distances
    ax = axs[1, 1]
    ys = np.arange(len(alternative))
    values = corr.loc[alternative, "spearman_vs_original"].to_numpy(float)
    ax.barh(ys, values, color="#527fa8", height=.51)
    ax.set_yticks(ys, [SCHEME_LABELS[s] for s in alternative])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman correlation vs original distance")
    for y, value in zip(ys, values):
        ax.text(max(-.99, min(.99, value + (.055 if value >= 0 else -.055))), y,
                f"{value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=10)
    ax.set_title("d  Distance-order stability", loc="left", fontweight="bold")

    for ax in axs.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Robustness of WorldMap training-domain support", fontsize=16, fontweight="bold")
    fig.savefig(outdir / "figS_support_distance_robustness.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / "figS_support_distance_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = args_parser()
    root = Path(args.project_dir).expanduser().resolve()
    run = find_run(root, args.worldmap_run)
    outdir = absolute_path(root, args.outdir) if args.outdir else run / "support_distance_robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    print("[START] CNT support-distance robustness (NO retraining, NO reprediction)")
    print(f"WorldMap run: {run}")
    reference, candidates = load_inputs(root, run, args.dataset)
    print(f"Reference rows: {len(reference)} | Candidates: {len(candidates)}")

    print("[1/4] Reproducing original support scheme ...", flush=True)
    distance, status, p95, p99, unique_n, baseline_check = recalculate_baseline(reference, candidates, run)
    results = {"baseline": {"distance": distance, "status": status,
                            "p95": p95, "p99": p99, "n_unique": unique_n}}
    for index, scheme in enumerate(SCHEMES[1:], start=2):
        print(f"[{index}/4] Recalculating {scheme} ...", flush=True)
        distance, status, p95, p99, unique_n = recalculate_variant(reference, candidates, scheme)
        results[scheme] = {"distance": distance, "status": status,
                           "p95": p95, "p99": p99, "n_unique": unique_n}

    tables = assemble_tables(candidates, results)
    for filename, frame in tables.items():
        frame.to_csv(outdir / filename, encoding="utf-8-sig", index=False)
    plot_results(tables, outdir, args.dpi)
    summary = {
        "reference_definition": "L1+L2, Composite >= 2 (identical to phase_diagram_worldmap.py)",
        "reference_rows": len(reference), "candidate_rows": len(candidates),
        "original_worldmap_run": str(run),
        "distance_definitions": {
            "baseline": "median imputation; IQR with SD fallback; 60D Euclidean",
            "equal_modules": "same scaling; sqrt(weight/number_of_dimensions), weight=.25 for each N/C/F/P",
            "nc_emphasis": "same scaling; N=.30 C=.30 F=.20 P=.20 with within-block dimension normalization",
            "std_scaling": "median imputation; population SD scaling; 60D Euclidean",
            "thresholds": "per-scheme leave-one-out training NN p95/p99; unique transformed reference states",
        },
        "baseline_verification": baseline_check,
        "interpretation": "Support class indicates proximity to observed training states, not predictive accuracy or confidence.",
        "figure": "figS_support_distance_robustness.png",
    }
    (outdir / "support_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = tables["support_scheme_summary.csv"]
    print("\n[RESULT]", flush=True)
    print(report[["description", "pct_supported", "original_supported_retained_pct", "all_status_unchanged_pct"]].to_string(index=False))
    print(f"\n[OK] Figures (PNG+PDF) and CSV tables saved to: {outdir}")


if __name__ == "__main__":
    main()
