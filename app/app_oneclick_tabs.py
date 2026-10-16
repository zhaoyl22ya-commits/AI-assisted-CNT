# app_oneclick_tabs.py
# =============================================================================
# DeepCNT 3.0 — top-level Streamlit application
#
# Sections:
#   Overview
#   Single Mode
#   Fusion
#   Evaluation
#   3D WorldMap
#
# Current scientific definitions:
#   Balanced     = q50
#   Potential    = q90
#   Conservative = q10
#   q90-q10      = descriptive interval width only
#
# Required local modules:
#   oneclick_core.py
#   oneclick_fusion.py
#   mapmodel_analysis.py
#   app_oneclick_worldmap.py
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mapmodel_analysis import (
    DeepSeekConfig,
    build_selected_catalyst_bundle_fusion,
    render_deepseek_panel_streamlit,
)
from oneclick_core import (
    WorldMapConfig,
    make_template_csv_bytes,
    parse_manual_recipe,
    prepare_recipe_input,
)
from oneclick_fusion import (
    FusionConfig,
    parse_modes_weights,
    run_worldmap_fusion,
)
from app_oneclick_worldmap import render_single_mode_page


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

NAV_OPTIONS = [
    "📖 Overview",
    "✅ Single Mode",
    "🧩 Fusion",
    "📊 Evaluation",
    "🧊 3D WorldMap",
]

FUSION_STATE_RESULT = "cnt_fusion_result"
FUSION_STATE_ERROR = "cnt_fusion_error"
FUSION_STATE_EDITOR = "cnt_fusion_editor"

RECIPE_COLUMNS = [
    "metal_1",
    "pct_1",
    "metal_2",
    "pct_2",
    "metal_3",
    "pct_3",
]

CRITERION_LABELS = {
    "balanced": "Balanced（q50）",
    "potential": "Potential（q90）",
    "conservative": "Conservative（q10）",
}

FUSION_TABLE_KEYS = {
    "balanced": "fusion_balanced",
    "potential": "fusion_potential",
    "conservative": "fusion_conservative",
}


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="DeepCNT 3.0",
    page_icon="🗺️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        font-family: "Microsoft YaHei", system-ui, sans-serif;
    }
    .stButton > button {
        border-radius: 10px;
        min-height: 2.7rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 12px;
        padding: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🗺️ DeepCNT 3.0 — Literature-Knowledge-Driven CNT WorldMap")
st.caption(
    "千余组CNT文献数据 → 物理状态编码 → 分层预测 → "
    "催化剂—工艺—状态候选排序。该系统是候选筛选助手，不是精确预测器。"
)

navigation = st.radio(
    "页面导航",
    options=NAV_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="cnt_top_navigation",
)

st.divider()


# =============================================================================
# General helpers
# =============================================================================

def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def _blank_recipe_table(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                column: ""
                for column in RECIPE_COLUMNS
            }
            for _ in range(max(1, int(rows)))
        ]
    )


def _initialize_fusion_state() -> None:
    st.session_state.setdefault(
        FUSION_STATE_RESULT,
        None,
    )

    st.session_state.setdefault(
        FUSION_STATE_ERROR,
        None,
    )

    if FUSION_STATE_EDITOR not in st.session_state:
        st.session_state[
            FUSION_STATE_EDITOR
        ] = _blank_recipe_table()


def _read_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    filename = str(
        uploaded_file.name
    ).lower()

    if filename.endswith(".csv"):
        last_error: Exception | None = None

        for encoding in (
            "utf-8-sig",
            "utf-8",
            "gb18030",
        ):
            try:
                uploaded_file.seek(0)

                return pd.read_csv(
                    uploaded_file,
                    encoding=encoding,
                    low_memory=False,
                )

            except UnicodeDecodeError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "CSV文件无法读取。"
        )

    uploaded_file.seek(0)

    return pd.read_excel(
        uploaded_file
    )


def _drop_blank_recipe_rows(
    frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=RECIPE_COLUMNS
        )

    work = frame.copy()

    for column in RECIPE_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA

    blank = pd.Series(
        True,
        index=work.index,
    )

    for column in RECIPE_COLUMNS:
        values = (
            work[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        blank &= values.isin(
            {
                "",
                "nan",
                "None",
                "<NA>",
            }
        )

    return work.loc[
        ~blank
    ].reset_index(
        drop=True
    )


def _validate_recipes(
    frame: pd.DataFrame | None,
) -> tuple[
    pd.DataFrame | None,
    str | None,
]:
    cleaned = _drop_blank_recipe_rows(
        frame
    )

    if cleaned.empty:
        return None, (
            "当前没有有效催化剂配方。"
        )

    try:
        normalized = prepare_recipe_input(
            cleaned
        )

    except Exception as exc:
        return None, str(
            exc
        )

    return normalized, None


def _download_file(
    label: str,
    path_value: Any,
    key: str,
) -> None:
    if not path_value:
        return

    path = Path(
        str(path_value)
    )

    if not path.is_file():
        return

    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=(
            "text/csv"
            if path.suffix.lower() == ".csv"
            else "application/octet-stream"
        ),
        use_container_width=True,
        key=key,
    )


def _show_table(
    title: str,
    frame: Any,
    height: int = 280,
) -> None:
    st.markdown(
        f"#### {title}"
    )

    if not isinstance(
        frame,
        pd.DataFrame,
    ) or frame.empty:
        st.warning(
            f"{title}为空或未生成。"
        )

        return

    st.dataframe(
        frame,
        use_container_width=True,
        height=height,
    )


def _sort_fusion_ranking(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if "fusion_rank" not in frame.columns:
        return frame

    work = frame.copy()

    work[
        "fusion_rank"
    ] = pd.to_numeric(
        work[
            "fusion_rank"
        ],
        errors="coerce",
    )

    return work.sort_values(
        [
            "eligible_for_ranking",
            "fusion_rank",
            "coverage_fraction",
            "fused_score",
        ],
        ascending=[
            False,
            True,
            False,
            False,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def _latest_run_directory(
    root: Path,
) -> Path | None:
    if not root.is_dir():
        return None

    runs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(
            "run_"
        )
    ]

    if not runs:
        return None

    return max(
        runs,
        key=lambda path: path.stat().st_mtime,
    )


def _latest_phase_html() -> Path | None:
    latest_run = _latest_run_directory(
        PROJECT_DIR
        / "outputs"
        / "app"
        / "phase_diagram_worldmap"
    )

    if latest_run is None:
        return None

    expected = (
        latest_run
        / "dashboard_highD_3D.html"
    )

    return (
        expected
        if expected.is_file()
        else None
    )


def _ordered_pngs(
    directory: Path,
    preferred_names: list[str],
) -> list[Path]:
    if not directory.is_dir():
        return []

    all_pngs = {
        path.name: path
        for path in directory.glob(
            "*.png"
        )
        if path.is_file()
    }

    ordered = [
        all_pngs.pop(
            filename
        )
        for filename in preferred_names
        if filename in all_pngs
    ]

    ordered.extend(
        sorted(
            all_pngs.values(),
            key=lambda path: path.name.lower(),
        )
    )

    return ordered


def _render_image_gallery(
    title: str,
    directory: Path,
    preferred_names: list[str],
    caption: str,
) -> None:
    st.markdown(
        f"### {title}"
    )

    st.caption(
        caption
    )

    images = _ordered_pngs(
        directory,
        preferred_names,
    )

    if not images:
        st.warning(
            f"未找到图像目录或PNG文件：{directory}"
        )

        return

    for index in range(
        0,
        len(
            images
        ),
        2,
    ):
        columns = st.columns(
            2,
            gap="large",
        )

        for offset in range(
            2
        ):
            position = (
                index
                + offset
            )

            if position >= len(
                images
            ):
                continue

            image_path = images[
                position
            ]

            with columns[
                offset
            ]:
                st.markdown(
                    f"#### {image_path.stem}"
                )

                st.image(
                    image_path,
                    use_container_width=True,
                )

                st.caption(
                    str(
                        image_path.relative_to(
                            PROJECT_DIR
                        )
                    )
                )


# =============================================================================
# Overview
# =============================================================================

def render_overview() -> None:
    st.header(
        "模型概述"
    )

    st.info(
        "当前系统基于千余组异构CNT文献数据进行知识提取和候选排序。"
        "WorldMap与L123输出代表literature-derived候选空间，"
        "不是8800组真实实验，也不是实验闭环验证结果。"
    )

    figure_paths = [
        PROJECT_DIR
        / "worldmap001.png",
        PROJECT_DIR
        / "worldmap002.png",
        PROJECT_DIR
        / "worldmap003.png",
    ]

    available_figures = [
        path
        for path in figure_paths
        if path.is_file()
    ]

    if available_figures:
        columns = st.columns(
            min(
                len(
                    available_figures
                ),
                3,
            ),
            gap="large",
        )

        for column, path in zip(
            columns,
            available_figures,
        ):
            with column:
                st.image(
                    path,
                    use_container_width=True,
                )

                st.caption(
                    path.name
                )

    st.markdown(
        "### 当前冻结的数据与模型结构"
    )

    first, second, third = st.columns(
        3,
        gap="large",
    )

    with first:
        st.markdown(
            """
            #### 120维模型输入
            - 60个物理特征
            - 60个缺失掩码
            - N18：成核/催化剂状态
            - C12：催化剂与供碳状态
            - F18：流动与传质状态
            - P12：工艺边界状态
            """
        )

    with second:
        st.markdown(
            """
            #### 分层模型
            - **L1**：完整信息条件下的点预测基线
            - **L12**：缺失感知的q10/q50/q90预测
            - **L123**：调用L1/L12评分器进行搜索与推荐
            - L123不是独立监督学习模型
            """
        )

    with third:
        st.markdown(
            """
            #### 决策定义
            - **Balanced = q50**
            - **Potential = q90**
            - **Conservative = q10**
            - q90−q10仅表示预测跨度
            - 不使用风险惩罚排名
            """
        )

    st.divider()

    st.markdown(
        "### 单模式推荐链"
    )

    st.code(
        """
催化剂配方
  → L123-001：催化剂 × 文献路线模板
  → q50 Balanced预筛 + q90 Potential预筛
  → L123-002：催化剂三类排名与工艺包
  → Balanced/Potential路线并集
  → L123-003：颗粒尺寸 × raw C/M网格搜索
  → 边界审计与最终合表
        """.strip(),
        language="text",
    )

    st.markdown(
        "### 多模式Fusion"
    )

    st.write(
        "A–E模式分别运行完整单模式流程，随后在每个模式内部"
        "将催化剂排名转换为0–1百分位，再按照用户权重融合。"
        "默认不直接相加不同模式的原始模型分数。"
    )

    st.latex(
        r"""
        S_i^{\mathrm{fusion}}
        =
        \frac{
        \sum_m w_m s_{i,m} I_{i,m}
        }{
        \sum_m w_m I_{i,m}
        }
        """
    )

    st.markdown(
        "### 结果解释边界"
    )

    st.warning(
        "L123-003当前可能出现颗粒尺寸或raw C/M边界饱和。"
        "位于网格边界的推荐值只能解释为模型指示的搜索方向，"
        "不能写成真实全局最优实验条件。"
    )

    st.markdown(
        """
        - 文献数据中的条件关系应称为**关联、条件趋势或模型指示规律**；
        - 不应从观察性文献数据直接宣称因果机制；
        - 当前系统用于缩小候选空间和组织CNT生长知识；
        - 实验验证不是当前文章代码链的一部分。
        """
    )


# =============================================================================
# Fusion page
# =============================================================================

def _render_fusion_core_config() -> WorldMapConfig:
    cfg = WorldMapConfig()

    with st.expander(
        "单模式子流程参数",
        expanded=False,
    ):
        first, second, third = st.columns(
            3
        )

        with first:
            st.markdown(
                "#### L123-001"
            )

            cfg.topR_balanced = int(
                st.number_input(
                    "Balanced Top-R",
                    min_value=1,
                    max_value=100,
                    value=int(
                        cfg.topR_balanced
                    ),
                    step=1,
                    key="fusion_topR_balanced",
                )
            )

            cfg.topR_potential = int(
                st.number_input(
                    "Potential Top-R",
                    min_value=1,
                    max_value=100,
                    value=int(
                        cfg.topR_potential
                    ),
                    step=1,
                    key="fusion_topR_potential",
                )
            )

            cfg.topK_balanced_001 = int(
                st.number_input(
                    "Balanced 001 Top-K",
                    min_value=1,
                    max_value=cfg.topR_balanced,
                    value=min(
                        int(
                            cfg.topK_balanced_001
                        ),
                        cfg.topR_balanced,
                    ),
                    step=1,
                    key="fusion_topK_balanced_001",
                )
            )

            cfg.topK_potential_001 = int(
                st.number_input(
                    "Potential 001 Top-K",
                    min_value=1,
                    max_value=cfg.topR_potential,
                    value=min(
                        int(
                            cfg.topK_potential_001
                        ),
                        cfg.topR_potential,
                    ),
                    step=1,
                    key="fusion_topK_potential_001",
                )
            )

        with second:
            st.markdown(
                "#### L123-002"
            )

            cfg.aggregate_top_n = int(
                st.number_input(
                    "催化剂聚合Top-N",
                    min_value=1,
                    max_value=50,
                    value=int(
                        cfg.aggregate_top_n
                    ),
                    step=1,
                    key="fusion_aggregate_top_n",
                )
            )

            cfg.topK_balanced = int(
                st.number_input(
                    "Balanced工艺包Top-K",
                    min_value=1,
                    max_value=20,
                    value=int(
                        cfg.topK_balanced
                    ),
                    step=1,
                    key="fusion_topK_balanced",
                )
            )

            cfg.topK_potential = int(
                st.number_input(
                    "Potential工艺包Top-K",
                    min_value=1,
                    max_value=20,
                    value=int(
                        cfg.topK_potential
                    ),
                    step=1,
                    key="fusion_topK_potential",
                )
            )

            cfg.topK_conservative = int(
                st.number_input(
                    "Conservative工艺包Top-K",
                    min_value=1,
                    max_value=20,
                    value=int(
                        cfg.topK_conservative
                    ),
                    step=1,
                    key="fusion_topK_conservative",
                )
            )

        with third:
            st.markdown(
                "#### L123-003"
            )

            cfg.np_min = float(
                st.number_input(
                    "NP最小值（nm）",
                    value=float(
                        cfg.np_min
                    ),
                    step=0.5,
                    key="fusion_np_min",
                )
            )

            cfg.np_max = float(
                st.number_input(
                    "NP最大值（nm）",
                    value=float(
                        cfg.np_max
                    ),
                    step=1.0,
                    key="fusion_np_max",
                )
            )

            cfg.np_step = float(
                st.number_input(
                    "NP步长",
                    min_value=0.1,
                    value=float(
                        cfg.np_step
                    ),
                    step=0.1,
                    key="fusion_np_step",
                )
            )

            cfg.cm_min = float(
                st.number_input(
                    "raw C/M最小值",
                    value=float(
                        cfg.cm_min
                    ),
                    step=0.5,
                    key="fusion_cm_min",
                )
            )

            cfg.cm_max = float(
                st.number_input(
                    "raw C/M最大值",
                    value=float(
                        cfg.cm_max
                    ),
                    step=0.5,
                    key="fusion_cm_max",
                )
            )

            cfg.cm_step = float(
                st.number_input(
                    "raw C/M步长",
                    min_value=0.1,
                    value=float(
                        cfg.cm_step
                    ),
                    step=0.1,
                    key="fusion_cm_step",
                )
            )

            cfg.topk_points = int(
                st.number_input(
                    "每条路线物理状态Top点",
                    min_value=1,
                    max_value=100,
                    value=int(
                        cfg.topk_points
                    ),
                    step=1,
                    key="fusion_topk_points",
                )
            )

    return cfg


def _render_fusion_input() -> tuple[
    pd.DataFrame | None,
    str,
    FusionConfig,
    WorldMapConfig,
]:
    st.subheader(
        "Fusion输入"
    )

    modes_text = st.text_input(
        "① 选择模式与权重",
        value="A+B",
        help=(
            "支持A+B、BCE、"
            "A:0.5,C:0.3,E:0.2。"
        ),
        key="fusion_modes_text",
    )

    try:
        normalized_weights = parse_modes_weights(
            modes_text
        )

        st.caption(
            "归一化权重："
            + ", ".join(
                f"{mode}={weight:.3f}"
                for mode, weight in normalized_weights.items()
            )
        )

    except Exception as exc:
        st.warning(
            f"模式权重尚未通过检查：{exc}"
        )

    strategy_label = st.selectbox(
        "② Fusion策略",
        options=[
            "模式内排名百分位",
            "模式内分数百分位",
        ],
        index=0,
        key="fusion_strategy_label",
    )

    fusion_strategy = (
        "rank_percentile"
        if strategy_label
        == "模式内排名百分位"
        else "score_percentile"
    )

    coverage = float(
        st.slider(
            "③ 最低加权模式覆盖率",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help=(
                "默认1.0：只有覆盖全部所选模式的催化剂"
                "进入正式Fusion排名。"
            ),
            key="fusion_min_coverage",
        )
    )

    input_mode = st.radio(
        "④ 催化剂输入方式",
        options=[
            "单个催化剂",
            "批量输入",
        ],
        horizontal=True,
        key="fusion_input_mode",
    )

    recipes: pd.DataFrame | None = None

    if input_mode == "单个催化剂":
        recipe_text = st.text_input(
            "⑤ 输入催化剂",
            value="Fe 60, Co 40",
            help=(
                "示例：Fe 100；"
                "Fe 60, Co 40；"
                "Fe 60, Co 30, Ni 10。"
            ),
            key="fusion_manual_recipe",
        )

        if recipe_text.strip():
            try:
                recipes = parse_manual_recipe(
                    recipe_text
                )

            except Exception as exc:
                st.error(
                    f"输入解析失败：{exc}"
                )

    else:
        st.download_button(
            "下载催化剂输入模板",
            data=make_template_csv_bytes(
                n_blank_rows=12
            ),
            file_name=(
                "catalyst_input_template.csv"
            ),
            mime="text/csv",
            use_container_width=True,
            key="fusion_download_template",
        )

        uploaded_file = st.file_uploader(
            "上传CSV或Excel",
            type=[
                "csv",
                "xlsx",
                "xls",
            ],
            key="fusion_uploaded_file",
        )

        uploaded_frame: pd.DataFrame | None = None

        if uploaded_file is not None:
            try:
                uploaded_frame = _read_uploaded_table(
                    uploaded_file
                )

                st.success(
                    f"已读取{uploaded_file.name}："
                    f"{len(uploaded_frame)}行。"
                )

                with st.expander(
                    "预览上传文件",
                    expanded=False,
                ):
                    st.dataframe(
                        uploaded_frame.head(
                            30
                        ),
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(
                    f"文件读取失败：{exc}"
                )

        st.markdown(
            "#### 网页表格"
        )

        edited = st.data_editor(
            st.session_state[
                FUSION_STATE_EDITOR
            ],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="fusion_recipe_editor",
        )

        st.session_state[
            FUSION_STATE_EDITOR
        ] = edited

        source_options = [
            "网页表格",
        ]

        if uploaded_frame is not None:
            source_options.append(
                "上传文件"
            )

        selected_source = st.radio(
            "⑤ 本次使用的数据源",
            options=source_options,
            horizontal=True,
            key="fusion_batch_source",
        )

        recipes = (
            uploaded_frame
            if selected_source == "上传文件"
            else edited
        )

    normalized, validation_error = _validate_recipes(
        recipes
    )

    if validation_error:
        st.warning(
            validation_error
        )

    elif normalized is not None:
        st.info(
            f"有效且去重后的催化剂数："
            f"{len(normalized)}"
        )

        with st.expander(
            "查看规范化后的催化剂配方",
            expanded=False,
        ):
            st.dataframe(
                normalized,
                use_container_width=True,
            )

    core_cfg = _render_fusion_core_config()

    fusion_cfg = FusionConfig(
        strategy=fusion_strategy,
        min_coverage_fraction=coverage,
    )

    return (
        recipes,
        modes_text,
        fusion_cfg,
        core_cfg,
    )


def _render_fusion_audit(
    result: dict[str, Any],
) -> None:
    weights = result.get(
        "weights",
        {},
    )

    coverage = result.get(
        "coverage_summary"
    )

    first, second, third = st.columns(
        3
    )

    first.metric(
        "所选模式数",
        len(
            weights
        ),
    )

    first.caption(
        ", ".join(
            f"{mode}={weight:.2f}"
            for mode, weight in weights.items()
        )
    )

    if isinstance(
        coverage,
        pd.DataFrame,
    ) and not coverage.empty:
        balanced_row = coverage[
            coverage[
                "criterion"
            ].eq(
                "balanced"
            )
        ]

        if not balanced_row.empty:
            row = balanced_row.iloc[
                0
            ]

            second.metric(
                "可排名催化剂",
                int(
                    row[
                        "eligible_rows"
                    ]
                ),
            )

            third.metric(
                "完整模式覆盖",
                int(
                    row[
                        "full_coverage_rows"
                    ]
                ),
            )

    mode_results = result.get(
        "mode_results",
        {},
    )

    warnings = []

    for mode, mode_result in mode_results.items():
        audit = mode_result.get(
            "audit",
            {}
        ).get(
            "L123_003",
            {},
        )

        balanced_boundary = audit.get(
            "balanced_boundary_fraction"
        )

        potential_boundary = audit.get(
            "potential_boundary_fraction"
        )

        if isinstance(
            balanced_boundary,
            (
                int,
                float,
            ),
        ) and balanced_boundary >= 0.5:
            warnings.append(
                f"Mode {mode}: Balanced边界饱和率"
                f"{balanced_boundary:.1%}。"
            )

        if isinstance(
            potential_boundary,
            (
                int,
                float,
            ),
        ) and potential_boundary >= 0.5:
            warnings.append(
                f"Mode {mode}: Potential边界饱和率"
                f"{potential_boundary:.1%}。"
            )

    for warning in warnings:
        st.warning(
            warning
            + " 该状态只能解释为搜索方向。"
        )


def _render_fusion_rankings(
    result: dict[str, Any],
) -> None:
    tables = result.get(
        "tables",
        {},
    )

    st.markdown(
        "### Fusion催化剂排名"
    )

    tabs = st.tabs(
        [
            "Balanced（q50）",
            "Potential（q90）",
            "Conservative（q10）",
        ]
    )

    for tab, criterion in zip(
        tabs,
        (
            "balanced",
            "potential",
            "conservative",
        ),
    ):
        with tab:
            frame = tables.get(
                FUSION_TABLE_KEYS[
                    criterion
                ]
            )

            if isinstance(
                frame,
                pd.DataFrame,
            ):
                frame = _sort_fusion_ranking(
                    frame
                )

            _show_table(
                CRITERION_LABELS[
                    criterion
                ],
                frame,
                height=320,
            )


def _render_fusion_mode_results(
    result: dict[str, Any],
) -> None:
    st.markdown(
        "### 分模式运行结果"
    )

    mode_results = result.get(
        "mode_results",
        {},
    )

    if not isinstance(
        mode_results,
        dict,
    ) or not mode_results:
        st.warning(
            "Fusion结果中没有mode_results。"
        )

        return

    mode_tabs = st.tabs(
        [
            f"Mode {mode}"
            for mode in mode_results
        ]
    )

    for tab, (
        mode,
        mode_result,
    ) in zip(
        mode_tabs,
        mode_results.items(),
    ):
        with tab:
            st.caption(
                f"运行目录：{mode_result.get('run_dir', '')}"
            )

            audit = mode_result.get(
                "audit",
                {},
            )

            st.json(
                audit,
                expanded=False,
            )

            tables = mode_result.get(
                "tables",
                {},
            )

            if isinstance(
                tables,
                dict,
            ):
                _show_table(
                    "催化剂三类对比",
                    tables.get(
                        "catalyst_comparison"
                    ),
                    height=260,
                )

                _show_table(
                    "q50/q90路线并集",
                    tables.get(
                        "packages_union"
                    ),
                    height=260,
                )


def _render_fusion_downloads(
    result: dict[str, Any],
) -> None:
    paths = result.get(
        "paths",
        {},
    )

    if not isinstance(
        paths,
        dict,
    ):
        return

    st.markdown(
        "### 下载Fusion结果"
    )

    first, second = st.columns(
        2
    )

    with first:
        _download_file(
            "下载Balanced Fusion排名",
            paths.get(
                "fusion_rank_balanced"
            ),
            "download_fusion_balanced",
        )

        _download_file(
            "下载Potential Fusion排名",
            paths.get(
                "fusion_rank_potential"
            ),
            "download_fusion_potential",
        )

        _download_file(
            "下载Conservative Fusion排名",
            paths.get(
                "fusion_rank_conservative"
            ),
            "download_fusion_conservative",
        )

    with second:
        _download_file(
            "下载模式权重",
            paths.get(
                "fusion_weights"
            ),
            "download_fusion_weights",
        )

        _download_file(
            "下载模式运行汇总",
            paths.get(
                "fusion_mode_runs"
            ),
            "download_fusion_mode_runs",
        )

        _download_file(
            "下载覆盖率汇总",
            paths.get(
                "fusion_coverage_summary"
            ),
            "download_fusion_coverage",
        )

        _download_file(
            "下载Fusion manifest",
            paths.get(
                "fusion_manifest"
            ),
            "download_fusion_manifest",
        )


def _render_fusion_deepseek(
    result: dict[str, Any],
) -> None:
    st.divider()

    st.markdown(
        "### DeepSeek Fusion单催化剂分析"
    )

    if not st.checkbox(
        "启用Fusion DeepSeek分析",
        value=False,
        help=(
            "发送所选Fusion催化剂及其各模式"
            "最佳路线、物理状态和边界审计。"
        ),
        key="fusion_enable_deepseek",
    ):
        return

    criterion = st.selectbox(
        "① 选择分析策略",
        options=[
            "balanced",
            "potential",
            "conservative",
        ],
        format_func=lambda value: (
            CRITERION_LABELS[
                value
            ]
        ),
        key="fusion_deepseek_criterion",
    )

    tables = result.get(
        "tables",
        {},
    )

    ranking = (
        tables.get(
            FUSION_TABLE_KEYS[
                criterion
            ]
        )
        if isinstance(
            tables,
            dict,
        )
        else None
    )

    if not isinstance(
        ranking,
        pd.DataFrame,
    ) or ranking.empty:
        st.warning(
            "当前策略没有可分析的Fusion排名。"
        )

        return

    ranking = _sort_fusion_ranking(
        ranking
    )

    eligible = ranking

    if "eligible_for_ranking" in eligible.columns:
        column = eligible[
            "eligible_for_ranking"
        ]

        if not pd.api.types.is_bool_dtype(
            column
        ):
            column = (
                column
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(
                    {
                        "true",
                        "1",
                        "yes",
                    }
                )
            )

        eligible = eligible[
            column
        ].reset_index(
            drop=True
        )

    if eligible.empty:
        st.warning(
            "没有满足覆盖率阈值的Fusion催化剂。"
        )

        return

    rank_position = int(
        st.number_input(
            "② 选择Fusion排名位置",
            min_value=1,
            max_value=len(
                eligible
            ),
            value=1,
            step=1,
            key="fusion_deepseek_rank_position",
        )
    )

    st.dataframe(
        eligible.iloc[
            [
                rank_position
                - 1
            ]
        ],
        use_container_width=True,
    )

    try:
        bundle = (
            build_selected_catalyst_bundle_fusion(
                fusion_result=result,
                criterion=criterion,
                rank_position=rank_position,
            )
        )

    except Exception as exc:
        st.error(
            f"构造Fusion分析对象失败：{exc}"
        )

        return

    render_deepseek_panel_streamlit(
        bundle=bundle,
        project_dir=PROJECT_DIR,
        deepseek_config=DeepSeekConfig(),
        default_question=(
            "请比较该催化剂在各可用生长模式下的"
            "路线差异、Fusion排序依据、边界限制与"
            "下一步需要核实的实验参数。"
        ),
        title=(
            "DeepSeek Fusion selected-catalyst analysis"
        ),
        key_prefix=(
            f"fusion_deepseek_"
            f"{criterion}_"
            f"{rank_position}"
        ),
    )


def render_fusion_page() -> None:
    _initialize_fusion_state()

    st.header(
        "多生长模式Fusion"
    )

    st.caption(
        "各模式分别运行完整单模式推荐链，"
        "再在模式内部标准化排名并按权重融合。"
    )

    input_column, output_column = st.columns(
        [
            0.40,
            0.60,
        ],
        gap="large",
    )

    with input_column:
        (
            recipes,
            modes_text,
            fusion_cfg,
            core_cfg,
        ) = _render_fusion_input()

        st.divider()

        run_column, clear_column = st.columns(
            2
        )

        with run_column:
            run_clicked = st.button(
                "开始Fusion推荐",
                type="primary",
                use_container_width=True,
                key="fusion_run_button",
            )

        with clear_column:
            clear_clicked = st.button(
                "清空Fusion结果",
                use_container_width=True,
                key="fusion_clear_button",
            )

        if clear_clicked:
            st.session_state[
                FUSION_STATE_RESULT
            ] = None

            st.session_state[
                FUSION_STATE_ERROR
            ] = None

            st.rerun()

        if run_clicked:
            st.session_state[
                FUSION_STATE_ERROR
            ] = None

            normalized, validation_error = (
                _validate_recipes(
                    recipes
                )
            )

            try:
                parse_modes_weights(
                    modes_text
                )

            except Exception as exc:
                validation_error = str(
                    exc
                )

            if validation_error:
                st.session_state[
                    FUSION_STATE_ERROR
                ] = validation_error

            elif normalized is None:
                st.session_state[
                    FUSION_STATE_ERROR
                ] = (
                    "没有有效催化剂输入。"
                )

            else:
                try:
                    with st.spinner(
                        "正在逐模式运行L123-001/002/003"
                        "并融合排名..."
                    ):
                        result = run_worldmap_fusion(
                            recipes_df=recipes,
                            modes_text=modes_text,
                            project_dir=PROJECT_DIR,
                            core_cfg=core_cfg,
                            fusion_cfg=fusion_cfg,
                        )

                    st.session_state[
                        FUSION_STATE_RESULT
                    ] = result

                except Exception as exc:
                    st.session_state[
                        FUSION_STATE_ERROR
                    ] = str(
                        exc
                    )

            st.rerun()

    with output_column:
        st.subheader(
            "Fusion输出"
        )

        error = st.session_state.get(
            FUSION_STATE_ERROR
        )

        if error:
            st.error(
                error
            )

        result = st.session_state.get(
            FUSION_STATE_RESULT
        )

        if result is None:
            st.info(
                "尚未运行Fusion。"
            )

            return

        if result.get(
            "status"
        ) != "success":
            st.error(
                "当前Fusion结果不是成功状态。"
            )

            return

        st.success(
            "Fusion运行完成。"
        )

        st.caption(
            f"运行目录：{result.get('run_dir', '')}"
        )

        _render_fusion_audit(
            result
        )

        _render_fusion_rankings(
            result
        )

        _render_fusion_mode_results(
            result
        )

        _render_fusion_downloads(
            result
        )

        _render_fusion_deepseek(
            result
        )


# =============================================================================
# Evaluation page
# =============================================================================

def _render_integrity_status(
    evaluation_dir: Path,
) -> None:
    status_path = (
        evaluation_dir
        / "pipeline_integrity_status.csv"
    )

    summary_path = (
        evaluation_dir
        / "evaluation_summary.json"
    )

    if not status_path.is_file():
        st.warning(
            f"未找到完整性审计：{status_path}"
        )

        return

    status = _read_csv(
        status_path
    )

    pass_count = int(
        status[
            "status"
        ].eq(
            "PASS"
        ).sum()
    )

    warn_count = int(
        status[
            "status"
        ].eq(
            "WARN"
        ).sum()
    )

    fail_count = int(
        status[
            "status"
        ].eq(
            "FAIL"
        ).sum()
    )

    columns = st.columns(
        4
    )

    columns[0].metric(
        "PASS",
        pass_count,
    )

    columns[1].metric(
        "WARN",
        warn_count,
    )

    columns[2].metric(
        "FAIL",
        fail_count,
    )

    columns[3].metric(
        "总检查",
        len(
            status
        ),
    )

    st.dataframe(
        status,
        use_container_width=True,
        height=320,
    )

    if summary_path.is_file():
        with st.expander(
            "查看evaluation_summary.json",
            expanded=False,
        ):
            st.json(
                _read_json(
                    summary_path
                ),
                expanded=False,
            )


def _render_validation_tables() -> None:
    manifest_path = (
        PROJECT_DIR
        / "outputs"
        / "app"
        / "evaluate_worldmap"
        / "validation_outputs_manifest.csv"
    )

    if not manifest_path.is_file():
        return

    st.markdown(
        "### 正式验证输出清单"
    )

    manifest = _read_csv(
        manifest_path
    )

    st.dataframe(
        manifest,
        use_container_width=True,
        height=260,
    )

    collected_dir = (
        PROJECT_DIR
        / "outputs"
        / "app"
        / "evaluate_worldmap"
        / "collected_validation"
    )

    if not collected_dir.is_dir():
        return

    files = sorted(
        collected_dir.glob(
            "*.csv"
        )
    )

    if not files:
        return

    selected = st.selectbox(
        "查看正式验证CSV",
        options=files,
        format_func=lambda path: path.name,
        key="evaluation_selected_validation_csv",
    )

    try:
        st.dataframe(
            _read_csv(
                selected
            ),
            use_container_width=True,
            height=340,
        )

    except Exception as exc:
        st.error(
            f"读取{selected.name}失败：{exc}"
        )


def render_evaluation_page() -> None:
    st.header(
        "Evaluation与审计"
    )

    st.caption(
        "该页面只读取当前冻结目录，"
        "不再回退到2.0或旧随机切分评估。"
    )

    evaluation_dir = (
        PROJECT_DIR
        / "outputs"
        / "app"
        / "evaluate_worldmap"
    )

    _render_integrity_status(
        evaluation_dir
    )

    st.divider()

    _render_image_gallery(
        title="WorldMap与L123完整性审计",
        directory=(
            evaluation_dir
            / "figures"
        ),
        preferred_names=[
            "worldmap_q50_vs_q90.png",
            "worldmap_q50_by_mode.png",
            "l123_002_selection_overlap.png",
            "l123_002_packages_per_catalyst.png",
            "l123_003_boundary_saturation.png",
        ],
        caption=(
            "展示WorldMap分数空间、模式分布、"
            "q50/q90选择重叠及物理状态边界饱和。"
        ),
    )

    st.divider()

    _render_image_gallery(
        title="跨论文与时间泛化",
        directory=(
            PROJECT_DIR
            / "outputs"
            / "external_validation"
            / "validate_generalization"
            / "figures"
        ),
        preferred_names=[
            "fig3a_three_split_r2.png",
            "fig3b_three_split_spearman.png",
            "fig3c_grouped_parity.png",
            "fig3d_temporal_parity.png",
            "figS1_metrics_by_growth_mode.png",
        ],
        caption=(
            "正式证据应来自random、paper-grouped与temporal"
            "验证，而不是在本页面重新随机切分。"
        ),
    )

    st.divider()

    _render_image_gallery(
        title="消融与特征块贡献",
        directory=(
            PROJECT_DIR
            / "outputs"
            / "external_validation"
            / "ablation"
            / "figures"
        ),
        preferred_names=[
            "fig4a_retraining_ablation_heatmap.png",
            "fig4b_drop_one_delta_r2.png",
        ],
        caption=(
            "用于检验N/C/F/P及缺失掩码模块的模型贡献。"
        ),
    )

    st.divider()

    _render_image_gallery(
        title="分位数与预测区间",
        directory=(
            PROJECT_DIR
            / "outputs"
            / "external_validation"
            / "uncertainty"
            / "figures"
        ),
        preferred_names=[
            "fig5a_quantile_calibration.png",
            "fig5b_interval_coverage_width.png",
            "fig5c_uncertainty_error_relation.png",
        ],
        caption=(
            "区间宽度用于描述q10–q90跨度；"
            "当前结果不支持把宽度作为单样本误差代理。"
        ),
    )

    st.divider()

    _render_image_gallery(
        title="历史回放排序",
        directory=(
            PROJECT_DIR
            / "outputs"
            / "external_validation"
            / "retrospective_ranking"
            / "figures"
        ),
        preferred_names=[
            "fig5d_historical_ndcg_timeline.png",
            "fig5e_recall_by_cutoff.png",
            "fig5f_enrichment_over_random.png",
        ],
        caption=(
            "检验模型能否仅依据较早文献优先排序后续高价值候选。"
        ),
    )

    st.divider()

    _render_image_gallery(
        title="Composite阈值敏感性",
        directory=(
            PROJECT_DIR
            / "outputs"
            / "external_validation"
            / "composite_sensitivity"
            / "figures"
        ),
        preferred_names=[
            "fig_composite_sensitivity_validation_spearman.png",
            "fig_composite_sensitivity_historical_enrichment.png",
        ],
        caption=(
            "比较不同Composite阈值下的数据保留、"
            "泛化与历史富集表现。"
        ),
    )

    st.divider()

    _render_validation_tables()


# =============================================================================
# 3D WorldMap page
# =============================================================================

def render_phase_page() -> None:
    st.header(
        "高维3D WorldMap"
    )

    st.caption(
        "严格读取outputs/app/phase_diagram_worldmap中最新run_*的"
        "dashboard_highD_3D.html。"
    )

    html_path = _latest_phase_html()

    if html_path is None:
        st.warning(
            "未找到：outputs/app/phase_diagram_worldmap/run_*/"
            "dashboard_highD_3D.html。\n\n"
            "请先依次运行phase_diagram_worldmap.py和"
            "phase_diagram_interactive_3d.py。"
        )

        return

    st.success(
        "已加载："
        + str(
            html_path.relative_to(
                PROJECT_DIR
            )
        )
    )

    first, second = st.columns(
        [
            0.75,
            0.25,
        ]
    )

    with first:
        st.write(
            "该3D页面包含潜在空间、决策空间、"
            "物理状态空间与工艺/催化剂切片。"
        )

    with second:
        _download_file(
            "下载3D HTML",
            html_path,
            "download_3d_html",
        )

    html = html_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    components.html(
        html,
        height=920,
        scrolling=True,
    )

    run_dir = html_path.parent

    with st.expander(
        "查看本次WorldMap运行文件",
        expanded=False,
    ):
        files = sorted(
            [
                path
                for path in run_dir.iterdir()
                if path.is_file()
            ],
            key=lambda path: path.name.lower(),
        )

        file_table = pd.DataFrame(
            [
                {
                    "file": path.name,
                    "size_MB": (
                        path.stat().st_size
                        / 1024
                        / 1024
                    ),
                }
                for path in files
            ]
        )

        st.dataframe(
            file_table,
            use_container_width=True,
        )


# =============================================================================
# Router
# =============================================================================

if navigation == "📖 Overview":
    render_overview()

elif navigation == "✅ Single Mode":
    render_single_mode_page()

elif navigation == "🧩 Fusion":
    render_fusion_page()

elif navigation == "📊 Evaluation":
    render_evaluation_page()

elif navigation == "🧊 3D WorldMap":
    render_phase_page()
