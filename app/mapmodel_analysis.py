# mapmodel_analysis.py
# =============================================================================
# DeepSeek analysis for the frozen CNT OneClick pipeline.
#
# Supported objects:
#   1) One selected catalyst from a single growth-mode result.
#   2) One selected catalyst from a multi-mode Fusion result.
#
# Fixed definitions:
#   Balanced=q50; Potential=q90; Conservative=q10.
#   q90-q10 is descriptive only, never a calibrated error/confidence or penalty.
#
# Removed:
#   TOP-3 compatibility, robust strategy, final_score/screen_score fallbacks,
#   old output discovery and log_C_NP_ratio.
#
# API key:
#   Read DEEPSEEK_API_KEY from the project-root .env file.
#   Never hardcode a real API key in this source file.
# =============================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

CRITERIA = ("balanced", "potential", "conservative")
CRITERION_LABELS = {
    "balanced": "Balanced (q50)",
    "potential": "Potential (q90)",
    "conservative": "Conservative (q10)",
}
# =============================================================================
# DEEPSEEK API CONFIG
# =============================================================================

DEFAULT_API_BASE = (
    os.getenv("DEEPSEEK_BASE_URL")
    or os.getenv("DEEPSEEK_API_BASE")
    or "https://api.deepseek.com"
).rstrip("/")
DEFAULT_MODEL = os.getenv(
    "DEEPSEEK_MODEL", "deepseek-flash"
).strip()


# =============================================================================
# API key and optional project context
# =============================================================================

def resolve_deepseek_api_key() -> str:
    """Read DEEPSEEK_API_KEY from the project-root .env/environment."""
    return os.getenv("DEEPSEEK_API_KEY", "").strip()


@dataclass
class DeepSeekConfig:
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = DEFAULT_MODEL
    chat_path: str = "/chat/completions"
    timeout_sec: int = 600
    temperature: float | None = None
    max_output_tokens: int | None = 32000

    def __post_init__(self) -> None:
        if not str(self.api_key or "").strip():
            self.api_key = resolve_deepseek_api_key()


def _read_optional_text(path: Path) -> str:
    return (
        path.read_text(encoding="utf-8", errors="replace").strip()
        if path.is_file()
        else ""
    )


def load_project_context(project_dir: str | Path) -> dict[str, str]:
    """
    Only current promptsV2 files are read. Missing optional files do not stop
    the analysis; the prompt then forbids inventing feature definitions.
    """
    root = Path(project_dir).expanduser().resolve()
    paths = {
        "selected_analysis_prompt":
            root / "promptsV2/prompt_deepseek_selected_analysis.txt",
        "growth_mode_definitions":
            root / "promptsV2/prompt_classify_mode.txt",
        "latent_space_definitions":
            root / "promptsV2/CNT_latent_space_explanation.txt",
        "feature_dictionary":
            root / "promptsV2/CNT_feature_dictionary.txt",
    }
    return {key: _read_optional_text(path) for key, path in paths.items()}


# =============================================================================
# General helpers
# =============================================================================

def normalize_criterion(value: str) -> str:
    aliases = {
        "balanced": "balanced", "balance": "balanced", "q50": "balanced",
        "potential": "potential", "q90": "potential",
        "conservative": "conservative", "q10": "conservative",
    }
    token = str(value or "").strip().lower()
    if token not in aliases:
        raise ValueError(
            "criterion必须是balanced、potential或conservative。"
        )
    return aliases[token]


def _require_success(result: Mapping[str, Any], label: str) -> None:
    if result.get("status") != "success":
        raise RuntimeError(f"{label}不是成功运行结果。")


def _require_dataframe(
    tables: Mapping[str, Any],
    key: str,
    label: str,
) -> pd.DataFrame:
    frame = tables.get(key)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"{label}缺少有效表格：{key}")
    return frame.copy()


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label}缺少字段：{missing}")


def _clean_id(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _jsonable(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, default=_json_default)
    )


def _sort_ranking(
    frame: pd.DataFrame,
    criterion: str,
    fusion: bool,
) -> pd.DataFrame:
    rank_col = "fusion_rank" if fusion else f"catalyst_rank_{criterion}"
    score_col = (
        "fused_score"
        if fusion
        else f"primary_catalyst_score_{criterion}"
    )
    _require_columns(
        frame, ["catalyst_id", rank_col, score_col], "Catalyst ranking"
    )

    work = frame.copy()
    work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")

    if fusion and "eligible_for_ranking" in work.columns:
        eligible = work["eligible_for_ranking"]
        if not pd.api.types.is_bool_dtype(eligible):
            eligible = (
                eligible.fillna("").astype(str).str.strip().str.lower()
                .isin({"true", "1", "yes"})
            )
        work = work[eligible].copy()

    work = work.dropna(subset=[rank_col, score_col])
    if work.empty:
        raise RuntimeError("没有可供选择的有效排名行。")

    return work.sort_values(
        [rank_col, score_col, "catalyst_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _select_by_position(
    ranking: pd.DataFrame,
    rank_position: int,
) -> pd.DataFrame:
    position = int(rank_position)
    if position < 1 or position > len(ranking):
        raise IndexError(
            f"rank_position={position}超出范围；可选行数为{len(ranking)}。"
        )
    selected = ranking.iloc[[position - 1]].copy()
    selected.insert(0, "user_selected_position", position)
    return selected


def _match_one(
    frame: pd.DataFrame,
    column: str,
    value: str,
    label: str,
    required: bool,
) -> pd.DataFrame:
    if frame.empty:
        if required:
            raise RuntimeError(f"{label}为空。")
        return pd.DataFrame()

    _require_columns(frame, [column], label)
    matched = frame[
        frame[column].astype(str).str.strip().eq(value)
    ].copy()

    if matched.empty:
        if required:
            raise RuntimeError(f"{label}中未找到{column}={value}。")
        return matched
    if len(matched) > 1:
        raise RuntimeError(
            f"{label}中{column}={value}对应{len(matched)}行，无法唯一匹配。"
        )
    return matched.reset_index(drop=True)


def _selected_boundary_warning(
    boundary_rows: pd.DataFrame,
    criterion: str,
) -> list[str]:
    if boundary_rows.empty:
        return (
            [
                "当前L123-003只对q50/q90并集搜索物理状态；"
                "Conservative路线不一定具有独立物理状态结果。"
            ]
            if criterion == "conservative"
            else []
        )

    if "any_boundary" not in boundary_rows.columns:
        return []

    values = boundary_rows["any_boundary"]
    if not pd.api.types.is_bool_dtype(values):
        values = (
            values.fillna("").astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes"})
        )

    return (
        [
            f"{CRITERION_LABELS[criterion]}物理状态位于预设搜索边界；"
            "该数值只能解释为boundary-limited方向，"
            "不能解释为真实全局最优条件。"
        ]
        if values.any()
        else []
    )


def _relevant_stage_warnings(
    run_result: Mapping[str, Any],
    criterion: str,
) -> list[str]:
    warnings = (
        run_result.get("audit", {})
        .get("L123_003", {})
        .get("warnings", [])
    )
    if not isinstance(warnings, list):
        return []

    token = criterion.capitalize()
    return [
        str(item)
        for item in warnings
        if token in str(item)
    ]


# =============================================================================
# Single-mode selected-catalyst bundle
# =============================================================================

def _single_bundle_for_catalyst(
    run_result: Mapping[str, Any],
    criterion: str,
    catalyst_id: str,
    selected_position: int | None,
) -> dict[str, Any]:
    _require_success(run_result, "Single-mode result")
    criterion = normalize_criterion(criterion)
    tables = run_result.get("tables")
    if not isinstance(tables, Mapping):
        raise RuntimeError("Single-mode result缺少tables。")

    ranking = _sort_ranking(
        _require_dataframe(
            tables, f"catalyst_{criterion}", "Single-mode result"
        ),
        criterion,
        fusion=False,
    )
    selected_rank = _match_one(
        ranking, "catalyst_id", catalyst_id, "Catalyst ranking", True
    )
    if selected_position is not None:
        selected_rank.insert(
            0, "user_selected_position", int(selected_position)
        )

    best_comb_col = f"best_combination_id_{criterion}"
    best_template_col = f"best_template_id_{criterion}"
    _require_columns(
        selected_rank,
        [best_comb_col, best_template_col],
        "Selected catalyst ranking",
    )
    combination_id = _clean_id(selected_rank.iloc[0][best_comb_col])
    template_id = _clean_id(selected_rank.iloc[0][best_template_col])
    if not combination_id:
        raise RuntimeError("Selected ranking row缺少best_combination_id。")

    selected_package = _match_one(
        _require_dataframe(
            tables, f"packages_{criterion}", "Single-mode result"
        ),
        "combination_id",
        combination_id,
        f"{criterion} package table",
        True,
    )

    physical = _require_dataframe(
        tables, "physical_recommendations", "Single-mode result"
    )
    selected_physical = _match_one(
        physical,
        "combination_id",
        combination_id,
        "Physical recommendations",
        criterion in {"balanced", "potential"},
    )

    boundary = _require_dataframe(
        tables, "boundary_audit", "Single-mode result"
    )
    _require_columns(
        boundary, ["combination_id", "objective"], "Boundary audit"
    )
    selected_boundary = boundary[
        boundary["combination_id"].astype(str).str.strip().eq(combination_id)
        & boundary["objective"].astype(str).str.strip().str.lower()
        .eq(criterion)
    ].copy()

    warnings = _selected_boundary_warning(
        selected_boundary, criterion
    ) + _relevant_stage_warnings(run_result, criterion)

    return {
        "kind": "single_selected",
        "criterion": criterion,
        "criterion_label": CRITERION_LABELS[criterion],
        "growth_mode_letter": run_result.get("growth_mode_letter"),
        "growth_mode_num": run_result.get("growth_mode_num"),
        "run_dir": run_result.get("run_dir"),
        "catalyst_id": catalyst_id,
        "template_id": template_id,
        "combination_id": combination_id,
        "selected_rank_row": selected_rank,
        "selected_package_row": selected_package,
        "selected_physical_row": selected_physical,
        "selected_boundary_rows": selected_boundary,
        "scientific_warnings": list(dict.fromkeys(warnings)),
        "audit": _jsonable(run_result.get("audit", {})),
        "config": _jsonable(run_result.get("config", {})),
    }


def build_selected_catalyst_bundle_single(
    run_result: Mapping[str, Any],
    criterion: str = "balanced",
    rank_position: int = 1,
) -> dict[str, Any]:
    """Select the displayed 1-based position from one criterion ranking."""
    _require_success(run_result, "Single-mode result")
    criterion = normalize_criterion(criterion)
    tables = run_result.get("tables")
    if not isinstance(tables, Mapping):
        raise RuntimeError("Single-mode result缺少tables。")

    ranking = _sort_ranking(
        _require_dataframe(
            tables, f"catalyst_{criterion}", "Single-mode result"
        ),
        criterion,
        fusion=False,
    )
    selected = _select_by_position(ranking, rank_position)
    catalyst_id = _clean_id(selected.iloc[0]["catalyst_id"])

    return _single_bundle_for_catalyst(
        run_result, criterion, catalyst_id, int(rank_position)
    )


# =============================================================================
# Fusion selected-catalyst bundle
# =============================================================================

def build_selected_catalyst_bundle_fusion(
    fusion_result: Mapping[str, Any],
    criterion: str = "balanced",
    rank_position: int = 1,
) -> dict[str, Any]:
    _require_success(fusion_result, "Fusion result")
    criterion = normalize_criterion(criterion)
    tables = fusion_result.get("tables")
    if not isinstance(tables, Mapping):
        raise RuntimeError("Fusion result缺少tables。")

    ranking = _sort_ranking(
        _require_dataframe(
            tables, f"fusion_{criterion}", "Fusion result"
        ),
        criterion,
        fusion=True,
    )
    selected_fusion = _select_by_position(ranking, rank_position)
    catalyst_id = _clean_id(selected_fusion.iloc[0]["catalyst_id"])

    mode_results = fusion_result.get("mode_results")
    weights = fusion_result.get("weights")
    if not isinstance(mode_results, Mapping) or not isinstance(weights, Mapping):
        raise RuntimeError("Fusion result缺少mode_results或weights。")

    mode_bundles: dict[str, dict[str, Any]] = {}
    missing_modes: list[str] = []

    for mode, weight in weights.items():
        mode_result = mode_results.get(mode)
        mode_tables = (
            mode_result.get("tables")
            if isinstance(mode_result, Mapping)
            else None
        )
        mode_rank = (
            mode_tables.get(f"catalyst_{criterion}")
            if isinstance(mode_tables, Mapping)
            else None
        )

        if not isinstance(mode_rank, pd.DataFrame):
            missing_modes.append(str(mode))
            continue

        present_ids = (
            set(mode_rank["catalyst_id"].astype(str).str.strip())
            if "catalyst_id" in mode_rank.columns
            else set()
        )
        if catalyst_id not in present_ids:
            missing_modes.append(str(mode))
            continue

        bundle = _single_bundle_for_catalyst(
            mode_result, criterion, catalyst_id, None
        )
        bundle["fusion_mode_weight"] = float(weight)
        mode_bundles[str(mode)] = bundle

    if not mode_bundles:
        raise RuntimeError(
            "所选Fusion催化剂在任何模式中都没有可分析的单模式结果。"
        )

    warnings = [
        f"Mode {mode}: {warning}"
        for mode, bundle in mode_bundles.items()
        for warning in bundle["scientific_warnings"]
    ]
    if missing_modes:
        warnings.append(
            "所选Fusion催化剂缺少以下模式结果："
            + ", ".join(missing_modes)
            + "。Fusion分数只代表可用模式及其覆盖规则。"
        )

    return {
        "kind": "fusion_selected",
        "criterion": criterion,
        "criterion_label": CRITERION_LABELS[criterion],
        "run_dir": fusion_result.get("run_dir"),
        "catalyst_id": catalyst_id,
        "selected_fusion_row": selected_fusion,
        "mode_bundles": mode_bundles,
        "missing_modes": missing_modes,
        "weights": _jsonable(weights),
        "fusion_strategy": fusion_result.get("fusion_strategy"),
        "min_coverage_fraction":
            fusion_result.get("min_coverage_fraction"),
        "scientific_warnings": list(dict.fromkeys(warnings)),
        "coverage_summary": _jsonable(
            fusion_result.get("coverage_summary", {})
        ),
        "config": _jsonable(fusion_result.get("config", {})),
    }


# =============================================================================
# Prompt construction
# =============================================================================

def column_glossary_markdown() -> str:
    return """
## Fixed interpretation glossary
- `pred_q10`: Conservative reference prediction.
- `pred_q50`: Balanced prediction and primary median estimate.
- `pred_q90`: Potential/high-upside prediction.
- `interval_width = pred_q90 - pred_q10`: descriptive predictive spread only;
  it is not a calibrated sample-level error, confidence or risk penalty.
- `primary_catalyst_score_balanced/potential/conservative`: catalyst-level
  top-N route aggregation under q50/q90/q10.
- `N1..N18`, `C1..C12`, `F1..F18`, `P1..P12`: structured blocks for
  nucleation, catalyst/carbon, flow/transport and process.
- `miss_*`: missingness indicators, not physical values.
- `N11`: catalyst nanoparticle-size variable used by physical-state search.
- `C3`: raw carbon-to-metal ratio (raw C/M); log transforms are display-only.
- `recommended_np_size_nm_*` and `recommended_C_over_M_*`: grid-limited
  recommendations; boundary points are directions, not validated global optima.
"""


def _frame_to_csv(frame: Any, missing_message: str) -> str:
    return (
        frame.to_csv(index=False)
        if isinstance(frame, pd.DataFrame) and not frame.empty
        else f"[{missing_message}]"
    )


def _bundle_tables_markdown(bundle: Mapping[str, Any]) -> str:
    parts: list[str] = []

    if bundle.get("kind") == "single_selected":
        data = [
            ("Selected catalyst ranking row", "selected_rank_row",
             "No selected catalyst ranking row"),
            ("Matched process-package row", "selected_package_row",
             "No matched process-package row"),
            ("Matched physical-state recommendation", "selected_physical_row",
             "No physical-state recommendation for this route"),
            ("Selected-route boundary audit", "selected_boundary_rows",
             "No boundary-audit row for this criterion"),
        ]
        for title, key, message in data:
            parts += [f"## {title}", _frame_to_csv(bundle.get(key), message)]
        return "\n\n".join(parts)

    parts += [
        "## Selected Fusion ranking row",
        _frame_to_csv(
            bundle.get("selected_fusion_row"), "No selected Fusion row"
        ),
    ]
    for mode, mode_bundle in bundle.get("mode_bundles", {}).items():
        data = [
            (f"Mode {mode} selected catalyst ranking row",
             "selected_rank_row", "No mode ranking row"),
            (f"Mode {mode} matched process-package row",
             "selected_package_row", "No matched mode package row"),
            (f"Mode {mode} physical-state recommendation",
             "selected_physical_row", "No mode physical-state recommendation"),
            (f"Mode {mode} boundary audit",
             "selected_boundary_rows", "No mode boundary-audit row"),
        ]
        for title, key, message in data:
            parts += [f"## {title}", _frame_to_csv(
                mode_bundle.get(key), message
            )]
    return "\n\n".join(parts)


def build_reasoner_prompt(
    bundle: Mapping[str, Any],
    project_dir: str | Path,
    user_question: str = "",
    response_language: str = "Chinese",
) -> tuple[str, str]:
    context = load_project_context(project_dir)
    criterion = normalize_criterion(str(bundle.get("criterion", "")))
    warnings = bundle.get("scientific_warnings", [])

    system_prompt = (
        "You are a scientific assistant for carbon nanotube growth and "
        "materials informatics. Analyze only the supplied rows. Distinguish "
        "observational literature associations, model predictions and "
        "experimentally validated facts. Never invent missing feature "
        "definitions or unsupported causal mechanisms."
    )

    metadata_keys = [
        "kind", "criterion", "criterion_label", "growth_mode_letter",
        "growth_mode_num", "run_dir", "catalyst_id", "template_id",
        "combination_id", "weights", "fusion_strategy",
        "min_coverage_fraction", "missing_modes",
    ]
    metadata = {
        key: bundle.get(key)
        for key in metadata_keys
        if bundle.get(key) is not None
    }

    sections = [
        "# CNT OneClick selected-catalyst analysis",
        f"Respond in {response_language}.",
        column_glossary_markdown(),
        "## Run metadata",
        json.dumps(
            metadata, ensure_ascii=False, indent=2, default=_json_default
        ),
        "## Scientific warnings",
        (
            "\n".join(f"- {warning}" for warning in warnings)
            if warnings
            else "- No selected-row boundary warning was recorded."
        ),
    ]

    optional_sections = [
        ("Project growth-mode definitions",
         context["growth_mode_definitions"]),
        ("Project latent-space definitions",
         context["latent_space_definitions"]),
        ("Formal feature dictionary",
         context["feature_dictionary"]),
        ("Project-specific selected-analysis instructions",
         context["selected_analysis_prompt"]),
    ]
    for title, text in optional_sections:
        if text:
            sections += [f"## {title}", text]

    if not context["feature_dictionary"]:
        sections += [
            "## Feature-definition limitation",
            (
                "No formal CNT_feature_dictionary.txt was found. Do not assign "
                "specific meanings to individual N/C/F/P indices beyond the "
                "block-level definitions and explicitly defined N11/C3."
            ),
        ]

    sections += [
        _bundle_tables_markdown(bundle),
        "## Required analysis",
        (
            "1. Explain why this catalyst and route were selected under "
            f"{CRITERION_LABELS[criterion]}.\n"
            "2. Interpret q10/q50/q90 without treating interval width as a "
            "calibrated confidence or error measure.\n"
            "3. Explain only trends supported by supplied feature definitions "
            "and rows. Treat literature-derived patterns as conditional "
            "associations, not causal laws.\n"
            "4. Translate the selected route into an ordered experimental-"
            "planning checklist while labeling assumptions and missing data.\n"
            "5. Explain nanoparticle-size and raw C/M recommendations only "
            "within the search grid and discuss every boundary flag.\n"
            "6. For Fusion, compare available modes and explain coverage and "
            "weights; do not average physical conditions across modes.\n"
            "7. State that this is a literature-knowledge-driven ranking "
            "assistant, not an exact predictor or validated optimizer.\n"
            "8. End with: Recommended direction, Evidence limitations, and "
            "Parameters to verify next."
        ),
        "## User additional request",
        (user_question or "").strip() or "No additional request.",
    ]
    return system_prompt, "\n\n".join(sections)


# =============================================================================
# DeepSeek API
# =============================================================================

def call_deepseek_reasoner(
    config: DeepSeekConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    api_key = (config.api_key or "").strip()
    if not api_key:
        raise ValueError(
            "DeepSeek API key为空。请在项目根目录.env中设置"
            "DEEPSEEK_API_KEY。"
        )
    if config.timeout_sec <= 0:
        raise ValueError("timeout_sec必须大于0。")
    if config.max_output_tokens is not None and config.max_output_tokens <= 0:
        raise ValueError("max_output_tokens必须大于0。")

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "thinking": {
            "type": "enabled"
        },
        "reasoning_effort": "high",
    }
    if config.temperature is not None:
        payload["temperature"] = float(config.temperature)
    if config.max_output_tokens is not None:
        payload["max_tokens"] = int(config.max_output_tokens)

    response = requests.post(
        config.api_base.rstrip("/") + "/" + config.chat_path.lstrip("/"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.timeout_sec,
    )
    if not response.ok:
        raise RuntimeError(
            f"DeepSeek API请求失败：HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )

    data = response.json()
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek API响应缺少choices。")

    content = str(
        (choices[0].get("message") or {}).get("content") or ""
    ).strip()
    if not content:
        raise RuntimeError("DeepSeek API返回了空content。")

    # reasoning_content is intentionally not exposed or stored.
    return {
        "content": content,
        "model": data.get("model", config.model),
        "usage": data.get("usage", {}),
        "request_id": data.get("id"),
    }


# =============================================================================
# Saving and Streamlit panel
# =============================================================================

def save_analysis_markdown(
    bundle: Mapping[str, Any],
    response: Mapping[str, Any],
    user_question: str = "",
) -> Path | None:
    run_dir_value = bundle.get("run_dir")
    if not run_dir_value:
        return None

    output_dir = (
        Path(str(run_dir_value)).expanduser().resolve()
        / "deepseek_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = normalize_criterion(str(bundle.get("criterion", "")))
    catalyst_id = str(bundle.get("catalyst_id", "candidate"))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / (
        f"selected_{criterion}_{catalyst_id}_{tag}.md"
    )

    metadata = {
        "kind": bundle.get("kind"),
        "criterion": criterion,
        "catalyst_id": catalyst_id,
        "model": response.get("model"),
        "request_id": response.get("request_id"),
        "usage": response.get("usage"),
        "user_question": user_question,
    }
    text = (
        "# DeepSeek selected-catalyst analysis\n\n"
        "## Metadata\n\n```json\n"
        + json.dumps(
            metadata, ensure_ascii=False, indent=2, default=_json_default
        )
        + "\n```\n\n## Analysis\n\n"
        + str(response.get("content", ""))
        + "\n"
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path


def render_deepseek_panel_streamlit(
    *,
    bundle: Mapping[str, Any],
    project_dir: str | Path,
    deepseek_config: DeepSeekConfig | None = None,
    default_question: str = (
        "请基于所选结果给出严谨、可执行且明确标注局限性的CNT生长建议。"
    ),
    title: str = "DeepSeek selected-catalyst analysis",
    key_prefix: str = "cnt_deepseek",
) -> None:
    import streamlit as st

    config = deepseek_config or DeepSeekConfig()
    if not config.api_key:
        config.api_key = resolve_deepseek_api_key()

    st.markdown(f"### {title}")
    st.caption(
        "只发送当前选中的催化剂、路线包、物理状态与边界审计；"
        "不发送完整数据库。"
    )
    for warning in bundle.get("scientific_warnings", []):
        st.warning(str(warning))

    with st.expander("查看将用于分析的选中数据", expanded=False):
        if bundle.get("kind") == "single_selected":
            for title_text, key in [
                ("Catalyst ranking", "selected_rank_row"),
                ("Matched route package", "selected_package_row"),
                ("Physical-state recommendation", "selected_physical_row"),
                ("Boundary audit", "selected_boundary_rows"),
            ]:
                st.write(title_text)
                st.dataframe(
                    bundle.get(key, pd.DataFrame()),
                    use_container_width=True,
                )
        else:
            st.write("Fusion ranking")
            st.dataframe(
                bundle.get("selected_fusion_row", pd.DataFrame()),
                use_container_width=True,
            )
            for mode, mode_bundle in bundle.get("mode_bundles", {}).items():
                st.write(f"Mode {mode}")
                st.dataframe(
                    mode_bundle.get(
                        "selected_package_row", pd.DataFrame()
                    ),
                    use_container_width=True,
                )

    question = st.text_area(
        "给DeepSeek的补充要求",
        value=default_question,
        height=130,
        key=f"{key_prefix}_question",
    )

    if not config.api_key:
        st.error(
            "未找到DeepSeek API密钥。请在项目根目录.env中设置"
            "DEEPSEEK_API_KEY。"
        )
        return

    consent = st.checkbox(
        "我同意将上方选中结果发送至DeepSeek API。",
        value=False,
        key=f"{key_prefix}_consent",
    )
    if not st.button(
        "开始DeepSeek分析",
        type="primary",
        use_container_width=True,
        disabled=not consent,
        key=f"{key_prefix}_button",
    ):
        return

    system_prompt, user_prompt = build_reasoner_prompt(
        bundle=bundle,
        project_dir=project_dir,
        user_question=question,
        response_language="Chinese",
    )
    with st.spinner("正在调用DeepSeek..."):
        response = call_deepseek_reasoner(
            config, system_prompt, user_prompt
        )

    saved_path = save_analysis_markdown(
        bundle, response, question
    )
    st.success("DeepSeek分析完成。")
    st.markdown(response["content"])

    if saved_path is not None:
        st.caption(f"本地保存：{saved_path}")

    st.download_button(
        "下载分析结果（Markdown）",
        data=str(response["content"]).encode("utf-8"),
        file_name=(
            f"deepseek_{bundle.get('criterion', 'selected')}_"
            f"{bundle.get('catalyst_id', 'candidate')}.md"
        ),
        mime="text/markdown",
        use_container_width=True,
        key=f"{key_prefix}_download",
    )

# =============================================================================
# Direct-run key check
# =============================================================================

def main() -> None:
    key = resolve_deepseek_api_key()

    print("=" * 88)
    print("[CNT DEEPSEEK ANALYSIS MODULE]")
    print(f"API base : {DEFAULT_API_BASE}")
    print(f"Model    : {DEFAULT_MODEL}")

    if key:
        masked = (
            key[:5] + "*" * max(0, len(key) - 9) + key[-4:]
            if len(key) > 9
            else "********"
        )
        print(f"API key  : loaded ({masked})")
        print(
            "Status   : key configuration is ready. "
            "The Streamlit page will call this module after a catalyst is selected."
        )
    else:
        print("API key  : not configured")
        print(
            "Action   : set DEEPSEEK_API_KEY in the project-root .env file."
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
