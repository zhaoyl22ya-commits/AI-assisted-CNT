# phase_diagram_interactive_3d.py
# =============================================================================
# CNT 3.0 WorldMap — interactive 3D explorer
#
# Reads the newest:
#   outputs/app/phase_diagram_worldmap/run_*/PHASE_CANDIDATES_sampled.csv
#
# raw q50 = Balanced, raw q90 = Potential, raw q10 = Conservative.
# q90-q10 is descriptive only and is not used for marker size or ranking.
# Raw C/M remains C3; log10(C3) is created only for display.
# =============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PHASE_OUT = "outputs/app/phase_diagram_worldmap"
INPUT_NAME = "PHASE_CANDIDATES_sampled.csv"
OUTPUT_NAME = "dashboard_highD_3D.html"

MODE_ORDER = ["A", "B", "C", "D", "E"]

N_COLS = [f"N{i}" for i in range(1, 19)]
C_COLS = [f"C{i}" for i in range(1, 13)]
F_COLS = [f"F{i}" for i in range(1, 19)]
P_COLS = [f"P{i}" for i in range(1, 13)]

BLOCKS = {
    "nuc": N_COLS,
    "cat": C_COLS,
    "flow": F_COLS,
    "proc": P_COLS,
}

AXIS_LABELS = {
    "pred_q10_raw": "Conservative score (raw q10)",
    "pred_q50_raw": "Balanced score (raw q50)",
    "pred_q90_raw": "Potential score (raw q90)",
    "pred_L1_composite_final_score": "L1 predicted composite score",
    "interval_width": "Interval width (q90-q10)",
    "N11": "Catalyst particle size N11 (nm)",
    "C3": "Raw C/M (C3)",
    "log10_C3_display": "log10(raw C/M, C3)",
    "P1": "Temperature-related feature P1",
    "nuc_PC1": "Nucleation block PC1",
    "nuc_PC2": "Nucleation block PC2",
    "cat_PC1": "Carbon-supply block PC1",
    "cat_PC2": "Carbon-supply block PC2",
    "flow_PC1": "Flow block PC1",
    "flow_PC2": "Flow block PC2",
    "proc_PC1": "Process block PC1",
    "proc_PC2": "Process block PC2",
}


# =============================================================================
# IO
# =============================================================================

def resolve_path(project_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def latest_run_dir(out_base: Path) -> Path:
    if not out_base.is_dir():
        raise FileNotFoundError(
            f"WorldMap output folder does not exist:\n  {out_base}"
        )

    runs = sorted(
        [
            path
            for path in out_base.iterdir()
            if path.is_dir() and path.name.startswith("run_")
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError(
            f"No run_* folder was found under:\n  {out_base}"
        )

    return runs[0]


def read_csv_smart(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate CSV not found:\n  {path}")

    last_error = None

    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def require_columns(
    frame: pd.DataFrame,
    columns: List[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns:\n  {missing}"
        )


# =============================================================================
# LABELS
# =============================================================================

def mode_to_letter(value) -> str:
    if value is None or pd.isna(value):
        return "?"

    token = str(value).strip().upper()

    direct = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "5": "E",
    }

    if token in direct:
        return direct[token]

    try:
        number = int(float(token))
        return {
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "E",
        }.get(number, "?")
    except (TypeError, ValueError):
        return "?"


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"", "nan", "none"}:
        return ""

    return text


def candidate_label(row: pd.Series) -> str:
    combination_id = clean_text(
        row.get("combination_id")
    )

    if combination_id:
        return combination_id

    catalyst_id = clean_text(
        row.get("catalyst_state_id")
    )

    route_id = clean_text(
        row.get("route_state_id")
    )

    if catalyst_id or route_id:
        return " | ".join(
            value
            for value in [catalyst_id, route_id]
            if value
        )

    catalyst_title = clean_text(
        row.get("catalyst_paper_title")
    )

    route_title = clean_text(
        row.get("route_paper_title")
    )

    if catalyst_title or route_title:
        return " | ".join(
            value
            for value in [catalyst_title, route_title]
            if value
        )

    return "CNT candidate"


# =============================================================================
# PCA
# =============================================================================

def usable_block_features(
    frame: pd.DataFrame,
    columns: List[str],
) -> List[str]:
    usable = []

    for feature in columns:
        if feature not in frame.columns:
            continue

        miss_column = f"miss_{feature}"

        if miss_column in frame.columns:
            observed = (
                to_numeric(frame[miss_column])
                .fillna(1)
                .eq(0)
            )
        else:
            observed = to_numeric(
                frame[feature]
            ).notna()

        values = to_numeric(
            frame.loc[observed, feature]
        ).dropna()

        if len(values) >= 2 and values.nunique() > 1:
            usable.append(feature)

    return usable


def recompute_block_pca(
    frame: pd.DataFrame,
    columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    usable = usable_block_features(
        frame,
        columns,
    )

    if not usable:
        return (
            np.zeros(len(frame)),
            np.zeros(len(frame)),
        )

    matrix = pd.DataFrame(index=frame.index)

    for feature in usable:
        values = to_numeric(
            frame[feature]
        ).copy()

        miss_column = f"miss_{feature}"

        if miss_column in frame.columns:
            missing = (
                to_numeric(frame[miss_column])
                .fillna(1)
                .gt(0)
            )

            values.loc[missing] = np.nan

        median = values.median(skipna=True)

        if not np.isfinite(median):
            median = 0.0

        values = values.fillna(median)

        mean = values.mean()
        std = values.std(ddof=1)

        if not np.isfinite(mean):
            mean = 0.0

        if not np.isfinite(std) or std == 0:
            std = 1.0

        matrix[feature] = (
            values - mean
        ) / std

    array = matrix.to_numpy(dtype=float)
    array -= np.mean(
        array,
        axis=0,
        keepdims=True,
    )

    u, singular_values, _ = np.linalg.svd(
        array,
        full_matrices=False,
    )

    component_count = min(
        2,
        len(singular_values),
    )

    coordinates = (
        u[:, :component_count]
        * singular_values[:component_count]
    )

    pc1 = coordinates[:, 0]

    pc2 = (
        coordinates[:, 1]
        if component_count > 1
        else np.zeros(len(frame))
    )

    return pc1, pc2


def ensure_block_pca_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for prefix, columns in BLOCKS.items():
        pc1 = f"{prefix}_PC1"
        pc2 = f"{prefix}_PC2"

        if pc1 in output.columns and pc2 in output.columns:
            output[pc1] = to_numeric(
                output[pc1]
            ).fillna(0.0)

            output[pc2] = to_numeric(
                output[pc2]
            ).fillna(0.0)

            continue

        output[pc1], output[pc2] = recompute_block_pca(
            output,
            columns,
        )

    return output


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_candidate_data(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        [
            "pred_q10_raw",
            "pred_q50_raw",
            "pred_q90_raw",
            "pred_L1_composite_final_score",
        ],
        "PHASE_CANDIDATES_sampled.csv",
    )

    data = frame.copy().reset_index(drop=True)

    numeric_columns = [
        "pred_q10_raw",
        "pred_q50_raw",
        "pred_q90_raw",
        "pred_L1_composite_final_score",
        "interval_width",
        "N11",
        "C3",
        "P1",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = to_numeric(
                data[column]
            )

    if "growth_mode_letter" in data.columns:
        data["mode_letter"] = (
            data["growth_mode_letter"]
            .map(mode_to_letter)
        )
    elif "growth_mode" in data.columns:
        data["mode_letter"] = (
            data["growth_mode"]
            .map(mode_to_letter)
        )
    else:
        data["mode_letter"] = "?"

    if "interval_width" not in data.columns:
        data["interval_width"] = (
            data["pred_q90_raw"]
            - data["pred_q10_raw"]
        )

    data["candidate_label"] = data.apply(
        candidate_label,
        axis=1,
    )

    if "C3" in data.columns:
        valid_c3 = to_numeric(
            data["C3"]
        ).gt(0)

        if "miss_C3" in data.columns:
            valid_c3 &= (
                to_numeric(data["miss_C3"])
                .fillna(1)
                .eq(0)
            )

        data["log10_C3_display"] = np.nan

        data.loc[
            valid_c3,
            "log10_C3_display",
        ] = np.log10(
            to_numeric(
                data.loc[valid_c3, "C3"]
            )
        )

    return ensure_block_pca_columns(
        data
    )


# =============================================================================
# AXIS SETS
# =============================================================================

def build_axis_sets(
    frame: pd.DataFrame,
) -> Dict[str, Tuple[str, str, str]]:
    axis_sets: Dict[
        str,
        Tuple[str, str, str],
    ] = {}

    if all(
        column in frame.columns
        for column in [
            "flow_PC1",
            "cat_PC1",
            "nuc_PC1",
        ]
    ):
        axis_sets[
            "Latent atlas: Flow-Carbon supply-Nucleation"
        ] = (
            "flow_PC1",
            "cat_PC1",
            "nuc_PC1",
        )

    if all(
        column in frame.columns
        for column in [
            "nuc_PC1",
            "proc_PC1",
            "flow_PC1",
        ]
    ):
        axis_sets[
            "Latent atlas: Nucleation-Process-Flow"
        ] = (
            "nuc_PC1",
            "proc_PC1",
            "flow_PC1",
        )

    axis_sets[
        "Decision space: raw q10-q50-q90"
    ] = (
        "pred_q10_raw",
        "pred_q50_raw",
        "pred_q90_raw",
    )

    if all(
        column in frame.columns
        for column in [
            "N11",
            "log10_C3_display",
            "P1",
        ]
    ):
        axis_sets[
            "Physical space: N11-log10(C/M)-P1"
        ] = (
            "N11",
            "log10_C3_display",
            "P1",
        )

    if all(
        column in frame.columns
        for column in [
            "proc_PC1",
            "flow_PC1",
            "pred_q50_raw",
        ]
    ):
        axis_sets[
            "Process-flow slice with q50"
        ] = (
            "proc_PC1",
            "flow_PC1",
            "pred_q50_raw",
        )

    if all(
        column in frame.columns
        for column in [
            "cat_PC1",
            "nuc_PC1",
            "pred_q50_raw",
        ]
    ):
        axis_sets[
            "Catalyst-nucleation slice with q50"
        ] = (
            "cat_PC1",
            "nuc_PC1",
            "pred_q50_raw",
        )

    return axis_sets


# =============================================================================
# DETAIL PANEL
# =============================================================================

def build_detail_sections(
    frame: pd.DataFrame,
) -> List[Dict[str, object]]:
    requested = [
        {
            "title": "Candidate identity",
            "keys": [
                "candidate_label",
                "combination_id",
                "catalyst_state_id",
                "route_state_id",
                "growth_mode",
                "growth_mode_letter",
                "mode_letter",
            ],
        },
        {
            "title": "Literature support",
            "keys": [
                "catalyst_frequency",
                "route_frequency",
                "catalyst_paper_doi",
                "catalyst_paper_title",
                "catalyst_paper_year",
                "catalyst_paper_journal",
                "catalyst_data_level",
                "catalyst_synthesis_id",
                "catalyst_source_file",
                "route_paper_doi",
                "route_paper_title",
                "route_paper_year",
                "route_paper_journal",
                "route_data_level",
                "route_synthesis_id",
                "route_source_file",
            ],
        },
        {
            "title": "Model outputs",
            "keys": [
                "pred_q10_raw",
                "pred_q50_raw",
                "pred_q90_raw",
                "q10_fix",
                "q50_fix",
                "q90_fix",
                "balanced_score",
                "potential_score",
                "conservative_score",
                "interval_width",
                "quantile_fixed",
                "pred_L1_composite_final_score",
            ],
        },
        {
            "title": "Physical state",
            "keys": [
                "N11",
                "C3",
                "log10_C3_display",
                "P1",
                "P3",
                "P4",
                "P5",
                "P6",
                "P8",
                "P9",
                "P10",
            ],
        },
        {
            "title": "Latent PCA coordinates",
            "keys": [
                "nuc_PC1",
                "nuc_PC2",
                "cat_PC1",
                "cat_PC2",
                "flow_PC1",
                "flow_PC2",
                "proc_PC1",
                "proc_PC2",
            ],
        },
        {
            "title": "Selected N/C/F/P features",
            "keys": [
                "N4",
                "N7",
                "N9",
                "N12",
                "N13",
                "N15",
                "N18",
                "C2",
                "C5",
                "C6",
                "C9",
                "C10",
                "C11",
                "C12",
                "F1",
                "F6",
                "F7",
                "F10",
                "F11",
                "F14",
                "F15",
                "F17",
            ],
        },
    ]

    sections = []

    for section in requested:
        keys = [
            key
            for key in section["keys"]
            if key in frame.columns
        ]

        if keys:
            sections.append(
                {
                    "title": section["title"],
                    "keys": keys,
                }
            )

    return sections


def detail_panel_template() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CNT WorldMap — Interactive 3D Explorer</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #ffffff;
      color: #111827;
    }
    .page-wrap {
      padding: 10px 12px 18px 12px;
      box-sizing: border-box;
    }
    .plot-wrap {
      width: 100%;
      min-width: 0;
      background: #ffffff;
    }
    .detail-panel {
      width: 100%;
      min-height: 280px;
      max-height: 520px;
      overflow-y: auto;
      overflow-x: hidden;
      margin-top: 14px;
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 10px;
      padding: 14px 16px 18px 16px;
      box-sizing: border-box;
      background: #ffffff;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .detail-panel h2 {
      margin: 0 0 10px 0;
      font-size: 18px;
      line-height: 1.35;
      word-break: break-word;
    }
    .hint {
      color: #4b5563;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 10px;
    }
    .section {
      margin-top: 14px;
      padding-top: 10px;
      border-top: 1px solid #e5e7eb;
    }
    .section-title {
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 8px;
      color: #111827;
    }
    .kv-row {
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr);
      gap: 10px;
      padding: 4px 0;
      align-items: start;
    }
    .kv-key {
      color: #374151;
      font-weight: 600;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .kv-val {
      color: #111827;
      font-size: 13px;
      white-space: normal;
      word-break: break-word;
      overflow-wrap: anywhere;
      line-height: 1.45;
    }
    .note {
      margin-top: 10px;
      font-size: 12px;
      color: #6b7280;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="page-wrap">
    <div class="plot-wrap">
      __PLOT_HTML__
    </div>

    <div id="point_detail" class="detail-panel">
      <h2>CNT candidate detail</h2>
      <div class="hint">
        Click a point in the 3D plot to display its catalyst state,
        route state, literature support and model outputs.
      </div>
      <div class="note">
        q90-q10 is shown only as a descriptive interval width.
        Marker size does not represent uncertainty or confidence.
      </div>
    </div>
  </div>

  <script>
    const DETAIL_FIELDS = __DETAIL_FIELDS__;
    const DETAIL_SECTIONS = __DETAIL_SECTIONS__;
    const plot = document.getElementById("cnt_worldmap_plot");
    const panel = document.getElementById("point_detail");

    const LABEL_MAP = {
      "pred_q10_raw": "Conservative raw q10",
      "pred_q50_raw": "Balanced raw q50",
      "pred_q90_raw": "Potential raw q90",
      "q10_fix": "Ordered q10 (audit)",
      "q50_fix": "Ordered q50 (audit)",
      "q90_fix": "Ordered q90 (audit)",
      "interval_width": "q90-q10 width",
      "pred_L1_composite_final_score": "L1 predicted composite",
      "log10_C3_display": "log10(raw C/M)",
      "N11": "Catalyst particle size N11",
      "C3": "Raw C/M (C3)",
      "growth_mode": "Growth mode",
      "growth_mode_letter": "Growth mode letter"
    };

    function escapeHtml(value) {
      if (value === null || value === undefined) return "";

      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") {
        return "—";
      }

      if (typeof value === "number") {
        if (!Number.isFinite(value)) return "—";

        const absolute = Math.abs(value);

        if (
          absolute >= 10000
          || (absolute > 0 && absolute < 0.001)
        ) {
          return value.toExponential(4);
        }

        return String(Number(value.toFixed(6)));
      }

      return escapeHtml(value);
    }

    function renderSection(title, keys, dataObject) {
      let html = `
        <div class="section">
          <div class="section-title">${escapeHtml(title)}</div>
      `;

      keys.forEach((key) => {
        const label = LABEL_MAP[key] || key;

        html += `
          <div class="kv-row">
            <div class="kv-key">${escapeHtml(label)}</div>
            <div class="kv-val">${formatValue(dataObject[key])}</div>
          </div>
        `;
      });

      html += "</div>";

      return html;
    }

    function renderPoint(dataObject) {
      const title = (
        dataObject.candidate_label
        || dataObject.combination_id
        || "CNT candidate detail"
      );

      let html = `<h2>${formatValue(title)}</h2>`;

      html += `
        <div class="hint">
          The selected point remains fixed while the 3D view is rotated
          or switched to another axis set.
        </div>
      `;

      DETAIL_SECTIONS.forEach((section) => {
        html += renderSection(
          section.title,
          section.keys,
          dataObject
        );
      });

      html += `
        <div class="note">
          Click another point to update this panel.
        </div>
      `;

      panel.innerHTML = html;
    }

    if (plot && plot.on) {
      plot.on("plotly_click", function(eventData) {
        if (
          !eventData
          || !eventData.points
          || !eventData.points.length
        ) {
          return;
        }

        const point = eventData.points[0];
        const customData = Array.isArray(point.customdata)
          ? point.customdata
          : [];

        const dataObject = {};

        DETAIL_FIELDS.forEach((key, index) => {
          dataObject[key] = (
            index < customData.length
            ? customData[index]
            : null
          );
        });

        renderPoint(dataObject);
      });
    }
  </script>
</body>
</html>
"""


# =============================================================================
# FIGURE
# =============================================================================

def axis_title(column: str) -> str:
    return AXIS_LABELS.get(
        column,
        column,
    )


def scene_style(
    x_column: str,
    y_column: str,
    z_column: str,
) -> Dict[str, object]:
    return {
        "xaxis": {
            "title": axis_title(x_column),
            "showbackground": False,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zeroline": False,
        },
        "yaxis": {
            "title": axis_title(y_column),
            "showbackground": False,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zeroline": False,
        },
        "zaxis": {
            "title": axis_title(z_column),
            "showbackground": False,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zeroline": False,
        },
        "camera": {
            "eye": {
                "x": 1.55,
                "y": 1.35,
                "z": 1.10,
            }
        },
        "aspectmode": "cube",
    }


def marker_size_for_rows(
    row_count: int,
) -> float:
    if row_count >= 8000:
        return 3.2

    if row_count >= 4000:
        return 3.8

    if row_count >= 1500:
        return 4.5

    return 5.5


def make_interactive_3d(
    frame: pd.DataFrame,
    output_html: Path,
) -> None:
    import plotly.graph_objects as go

    data = prepare_candidate_data(
        frame
    )

    axis_sets = build_axis_sets(
        data
    )

    if not axis_sets:
        raise RuntimeError(
            "No valid 3D axis set could be built."
        )

    detail_sections = build_detail_sections(
        data
    )

    detail_fields = []

    for section in detail_sections:
        for key in section["keys"]:
            if key not in detail_fields:
                detail_fields.append(key)

    customdata_frame = (
        data[detail_fields]
        .copy()
        .where(
            pd.notna(data[detail_fields]),
            None,
        )
    )

    customdata = customdata_frame.to_numpy(
        dtype=object
    )

    default_name = next(
        iter(axis_sets)
    )

    default_x, default_y, default_z = axis_sets[
        default_name
    ]

    color_column = "pred_q50_raw"
    color_values = to_numeric(
        data[color_column]
    )

    finite_color = color_values[
        np.isfinite(color_values)
    ]

    if finite_color.empty:
        color_min = 0.0
        color_max = 1.0
    else:
        color_min = float(
            finite_color.quantile(0.01)
        )

        color_max = float(
            finite_color.quantile(0.99)
        )

        if color_min == color_max:
            color_max = color_min + 1.0

    modes = [
        mode
        for mode in [*MODE_ORDER, "?"]
        if data["mode_letter"].eq(mode).any()
    ]

    marker_size = marker_size_for_rows(
        len(data)
    )

    def custom_index(column: str) -> Optional[int]:
        if column not in detail_fields:
            return None

        return detail_fields.index(
            column
        )

    label_index = custom_index(
        "candidate_label"
    )

    q50_index = custom_index(
        "pred_q50_raw"
    )

    q90_index = custom_index(
        "pred_q90_raw"
    )

    mode_index = custom_index(
        "mode_letter"
    )

    hover_lines = []

    if label_index is not None:
        hover_lines.append(
            f"<b>%{{customdata[{label_index}]}}</b>"
        )

    if mode_index is not None:
        hover_lines.append(
            f"Mode: %{{customdata[{mode_index}]}}"
        )

    if q50_index is not None:
        hover_lines.append(
            f"Balanced q50: %{{customdata[{q50_index}]:.5g}}"
        )

    if q90_index is not None:
        hover_lines.append(
            f"Potential q90: %{{customdata[{q90_index}]:.5g}}"
        )

    hover_lines.append(
        "<extra>Click for full details</extra>"
    )

    hover_template = "<br>".join(
        hover_lines
    )

    traces = []

    for mode in modes:
        subset = data[
            data["mode_letter"].eq(mode)
        ]

        traces.append(
            go.Scatter3d(
                x=to_numeric(
                    subset[default_x]
                ),
                y=to_numeric(
                    subset[default_y]
                ),
                z=to_numeric(
                    subset[default_z]
                ),
                mode="markers",
                name=f"Mode {mode}",
                customdata=customdata[
                    subset.index.to_numpy()
                ],
                hovertemplate=hover_template,
                marker={
                    "size": marker_size,
                    "color": to_numeric(
                        subset[color_column]
                    ),
                    "coloraxis": "coloraxis",
                    "opacity": 0.56,
                    "line": {
                        "width": 0.15,
                        "color": "rgba(255,255,255,0.45)",
                    },
                },
            )
        )

    figure = go.Figure(
        data=traces
    )

    figure.update_layout(
        title=(
            "CNT WorldMap — Interactive High-Dimensional "
            "Candidate Explorer"
        ),
        height=860,
        margin={
            "l": 10,
            "r": 120,
            "t": 70,
            "b": 10,
        },
        font={
            "size": 12,
        },
        scene=scene_style(
            default_x,
            default_y,
            default_z,
        ),
        legend={
            "x": 1.01,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": "rgba(0,0,0,0.12)",
            "borderwidth": 1,
            "font": {
                "size": 12,
                "color": "black",
            },
            "itemsizing": "constant",
        },
        coloraxis={
            "colorscale": "Turbo",
            "cmin": color_min,
            "cmax": color_max,
            "colorbar": {
                "title": "Balanced score (raw q50)",
                "x": 1.0,
                "y": 0.48,
                "len": 0.78,
                "thickness": 18,
                "tickfont": {
                    "size": 11,
                },
            },
        },
    )

    buttons = []

    for name, (
        x_column,
        y_column,
        z_column,
    ) in axis_sets.items():
        x_values = []
        y_values = []
        z_values = []

        for mode in modes:
            subset = data[
                data["mode_letter"].eq(mode)
            ]

            x_values.append(
                to_numeric(
                    subset[x_column]
                )
            )

            y_values.append(
                to_numeric(
                    subset[y_column]
                )
            )

            z_values.append(
                to_numeric(
                    subset[z_column]
                )
            )

        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {
                        "x": x_values,
                        "y": y_values,
                        "z": z_values,
                    },
                    {
                        "scene": scene_style(
                            x_column,
                            y_column,
                            z_column,
                        )
                    },
                ],
            }
        )

    figure.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "y": 1.04,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.88)",
                "bordercolor": "rgba(0,0,0,0.12)",
                "borderwidth": 1,
                "font": {
                    "size": 12,
                    "color": "black",
                },
            }
        ]
    )

    plot_html = figure.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="cnt_worldmap_plot",
    )

    page_html = detail_panel_template()
    page_html = page_html.replace(
        "__PLOT_HTML__",
        plot_html,
    )

    page_html = page_html.replace(
        "__DETAIL_FIELDS__",
        json.dumps(
            detail_fields,
            ensure_ascii=False,
        ),
    )

    page_html = page_html.replace(
        "__DETAIL_SECTIONS__",
        json.dumps(
            detail_sections,
            ensure_ascii=False,
        ),
    )

    output_html.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_html.write_text(
        page_html,
        encoding="utf-8",
    )

    print(
        f"[OK] Saved interactive 3D dashboard: {output_html}"
    )

    print(
        f"[OK] Candidate rows: {len(data)}"
    )

    print(
        "[OK] Axis sets: "
        + " | ".join(axis_sets.keys())
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNT 3.0 interactive 3D WorldMap"
    )

    parser.add_argument(
        "--project_dir",
        default=str(PROJECT_DIR),
        help="CNT project root",
    )

    parser.add_argument(
        "--phase_out",
        default=PHASE_OUT,
        help="WorldMap output base folder",
    )

    parser.add_argument(
        "--input_csv",
        default="auto",
        help=(
            "Candidate CSV path or 'auto'. "
            "Auto reads the newest run_*/PHASE_CANDIDATES_sampled.csv."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = Path(
        args.project_dir
    ).expanduser().resolve()

    if args.input_csv == "auto":
        out_base = resolve_path(
            project_dir,
            args.phase_out,
        )

        run_dir = latest_run_dir(
            out_base
        )

        csv_path = (
            run_dir
            / INPUT_NAME
        )

        if not csv_path.is_file():
            raise FileNotFoundError(
                "The newest WorldMap run does not contain:\n"
                f"  {csv_path}\n"
                "Run phase_diagram_worldmap.py first."
            )
    else:
        csv_path = resolve_path(
            project_dir,
            args.input_csv,
        )

        run_dir = csv_path.parent

    output_html = (
        run_dir
        / OUTPUT_NAME
    )

    print("=" * 108)
    print("[CNT INTERACTIVE 3D START]")
    print(f"Project directory : {project_dir}")
    print(f"Candidate CSV     : {csv_path}")
    print(f"Output HTML       : {output_html}")
    print("=" * 108)

    candidates = read_csv_smart(
        csv_path
    )

    make_interactive_3d(
        candidates,
        output_html,
    )

    print("\n[OK] CNT interactive 3D explorer completed.")
    print("=" * 108)


if __name__ == "__main__":
    main()
