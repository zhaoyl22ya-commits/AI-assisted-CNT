# app_oneclick_worldmap.py
# =============================================================================
# Streamlit page for the frozen CNT 3.0 single-mode OneClick pipeline.
#
# Balanced=q50; Potential=q90; Conservative=q10.
# q90-q10 is descriptive only and is never used as a ranking penalty.
#
# Dependencies:
#   oneclick_core.py
#   mapmodel_analysis.py
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mapmodel_analysis import (
    DeepSeekConfig,
    build_selected_catalyst_bundle_single,
    render_deepseek_panel_streamlit,
)
from oneclick_core import (
    WorldMapConfig,
    make_template_csv_bytes,
    parse_manual_recipe,
    prepare_recipe_input,
    run_worldmap,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

STATE_RESULT = "cnt_single_result"
STATE_ERROR = "cnt_single_error"
STATE_EDITOR = "cnt_single_editor"

RECIPE_COLUMNS = [
    "metal_1", "pct_1",
    "metal_2", "pct_2",
    "metal_3", "pct_3",
]

CRITERION_LABELS = {
    "balanced": "Balanced（q50）",
    "potential": "Potential（q90）",
    "conservative": "Conservative（q10）",
}

RANK_KEYS = {
    "balanced": "catalyst_balanced",
    "potential": "catalyst_potential",
    "conservative": "catalyst_conservative",
}

PACKAGE_KEYS = {
    "balanced": "packages_balanced",
    "potential": "packages_potential",
    "conservative": "packages_conservative",
}


# =============================================================================
# Helpers
# =============================================================================

def _blank_recipe_table(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: "" for column in RECIPE_COLUMNS} for _ in range(rows)]
    )


def _initialize_state() -> None:
    st.session_state.setdefault(STATE_RESULT, None)
    st.session_state.setdefault(STATE_ERROR, None)
    if STATE_EDITOR not in st.session_state:
        st.session_state[STATE_EDITOR] = _blank_recipe_table()


def _read_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    name = str(uploaded_file.name).lower()

    if name.endswith(".csv"):
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
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

        raise RuntimeError("CSV文件无法读取。")

    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)


def _drop_blank_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=RECIPE_COLUMNS)

    work = frame.copy()
    for column in RECIPE_COLUMNS:
        if column not in work.columns:
            work[column] = pd.NA

    blank = pd.Series(True, index=work.index)
    for column in RECIPE_COLUMNS:
        values = work[column].fillna("").astype(str).str.strip()
        blank &= values.isin({"", "nan", "None", "<NA>"})

    return work.loc[~blank].reset_index(drop=True)


def _validate_recipes(
    frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, str | None]:
    cleaned = _drop_blank_rows(frame)

    if cleaned.empty:
        return None, "当前没有有效催化剂配方。"

    try:
        return prepare_recipe_input(cleaned), None
    except Exception as exc:
        return None, str(exc)


def _download_file(label: str, path_value: Any, key: str) -> None:
    if not path_value:
        return

    path = Path(str(path_value))
    if not path.is_file():
        return

    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv" if path.suffix.lower() == ".csv"
        else "application/octet-stream",
        use_container_width=True,
        key=key,
    )


def _sort_ranking(frame: pd.DataFrame, criterion: str) -> pd.DataFrame:
    rank_column = f"catalyst_rank_{criterion}"

    if rank_column not in frame.columns:
        return frame

    work = frame.copy()
    work[rank_column] = pd.to_numeric(work[rank_column], errors="coerce")

    return work.sort_values(
        rank_column,
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)


def _show_table(title: str, frame: Any, height: int = 280) -> None:
    st.markdown(f"#### {title}")

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.warning(f"{title}为空或未生成。")
        return

    st.dataframe(frame, use_container_width=True, height=height)


# =============================================================================
# Sidebar
# =============================================================================

def _render_config_sidebar() -> WorldMapConfig:
    cfg = WorldMapConfig()

    with st.sidebar:
        st.header("单模式推荐参数")
        st.caption(
            "Balanced=q50，Potential=q90，Conservative=q10。"
        )

        with st.expander("L123-001 路线筛选", expanded=True):
            cfg.topR_balanced = int(st.number_input(
                "Balanced预筛Top-R",
                min_value=1, max_value=100,
                value=int(cfg.topR_balanced), step=1,
                key="single_topR_balanced",
            ))
            cfg.topR_potential = int(st.number_input(
                "Potential预筛Top-R",
                min_value=1, max_value=100,
                value=int(cfg.topR_potential), step=1,
                key="single_topR_potential",
            ))
            cfg.topK_balanced_001 = int(st.number_input(
                "Balanced进入001并集Top-K",
                min_value=1, max_value=cfg.topR_balanced,
                value=min(int(cfg.topK_balanced_001), cfg.topR_balanced),
                step=1,
                key="single_topK_balanced_001",
            ))
            cfg.topK_potential_001 = int(st.number_input(
                "Potential进入001并集Top-K",
                min_value=1, max_value=cfg.topR_potential,
                value=min(int(cfg.topK_potential_001), cfg.topR_potential),
                step=1,
                key="single_topK_potential_001",
            ))

        with st.expander("L123-002 催化剂与工艺包", expanded=True):
            cfg.aggregate_top_n = int(st.number_input(
                "催化剂聚合使用的Top-N路线",
                min_value=1, max_value=50,
                value=int(cfg.aggregate_top_n), step=1,
                key="single_aggregate_top_n",
            ))
            cfg.topK_balanced = int(st.number_input(
                "每个催化剂Balanced工艺包Top-K",
                min_value=1, max_value=20,
                value=int(cfg.topK_balanced), step=1,
                key="single_topK_balanced",
            ))
            cfg.topK_potential = int(st.number_input(
                "每个催化剂Potential工艺包Top-K",
                min_value=1, max_value=20,
                value=int(cfg.topK_potential), step=1,
                key="single_topK_potential",
            ))
            cfg.topK_conservative = int(st.number_input(
                "每个催化剂Conservative工艺包Top-K",
                min_value=1, max_value=20,
                value=int(cfg.topK_conservative), step=1,
                key="single_topK_conservative",
            ))

        with st.expander("L123-003 物理状态网格", expanded=False):
            cfg.np_min = float(st.number_input(
                "NP最小尺寸（nm）",
                value=float(cfg.np_min), step=0.5,
                key="single_np_min",
            ))
            cfg.np_max = float(st.number_input(
                "NP最大尺寸（nm）",
                value=float(cfg.np_max), step=1.0,
                key="single_np_max",
            ))
            cfg.np_step = float(st.number_input(
                "NP步长（nm）",
                min_value=0.1,
                value=float(cfg.np_step), step=0.1,
                key="single_np_step",
            ))
            cfg.cm_min = float(st.number_input(
                "raw C/M最小值",
                value=float(cfg.cm_min), step=0.5,
                key="single_cm_min",
            ))
            cfg.cm_max = float(st.number_input(
                "raw C/M最大值",
                value=float(cfg.cm_max), step=0.5,
                key="single_cm_max",
            ))
            cfg.cm_step = float(st.number_input(
                "raw C/M步长",
                min_value=0.1,
                value=float(cfg.cm_step), step=0.1,
                key="single_cm_step",
            ))
            cfg.topk_points = int(st.number_input(
                "每条路线保留物理状态Top点",
                min_value=1, max_value=100,
                value=int(cfg.topk_points), step=1,
                key="single_topk_points",
            ))

        st.divider()
        st.caption(
            "每次运行写入独立目录：outputs/app/oneclick_core/run_*_mode_X"
        )

    return cfg


# =============================================================================
# Input panel
# =============================================================================

def _render_input_panel() -> tuple[pd.DataFrame | None, str]:
    st.subheader("催化剂输入")

    growth_mode = st.selectbox(
        "① 选择生长模式",
        ["A", "B", "C", "D", "E"],
        index=3,
        key="single_growth_mode",
    )

    input_mode = st.radio(
        "② 选择输入方式",
        ["单个催化剂", "批量输入"],
        horizontal=True,
        key="single_input_mode",
    )

    recipes: pd.DataFrame | None = None

    if input_mode == "单个催化剂":
        text = st.text_input(
            "③ 输入催化剂",
            value="Fe 60, Co 40",
            help="示例：Fe 100；Fe 60, Co 40；Fe 60, Co 30, Ni 10。",
            key="single_manual_recipe",
        )

        if text.strip():
            try:
                recipes = parse_manual_recipe(text)
            except Exception as exc:
                st.error(f"输入解析失败：{exc}")

    else:
        st.caption(
            "可上传CSV/Excel，也可直接在网页表格中粘贴。"
        )

        st.download_button(
            "下载催化剂输入模板",
            data=make_template_csv_bytes(n_blank_rows=12),
            file_name="catalyst_input_template.csv",
            mime="text/csv",
            use_container_width=True,
            key="single_download_template",
        )

        uploaded_file = st.file_uploader(
            "上传CSV或Excel",
            type=["csv", "xlsx", "xls"],
            key="single_uploaded_file",
        )

        uploaded_frame: pd.DataFrame | None = None

        if uploaded_file is not None:
            try:
                uploaded_frame = _read_uploaded_table(uploaded_file)
                st.success(
                    f"已读取{uploaded_file.name}：{len(uploaded_frame)}行。"
                )
                with st.expander("预览上传文件", expanded=False):
                    st.dataframe(
                        uploaded_frame.head(30),
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"文件读取失败：{exc}")

        st.markdown("#### 网页表格")
        edited = st.data_editor(
            st.session_state[STATE_EDITOR],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="single_recipe_editor",
        )
        st.session_state[STATE_EDITOR] = edited

        if st.button(
            "清空网页表格",
            use_container_width=True,
            key="single_clear_editor",
        ):
            st.session_state[STATE_EDITOR] = _blank_recipe_table()
            st.rerun()

        source_options = ["网页表格"]
        if uploaded_frame is not None:
            source_options.append("上传文件")

        source = st.radio(
            "③ 本次使用的数据源",
            source_options,
            horizontal=True,
            key="single_batch_source",
        )
        recipes = uploaded_frame if source == "上传文件" else edited

    normalized, validation_error = _validate_recipes(recipes)

    if validation_error:
        st.warning(validation_error)
    elif normalized is not None:
        st.info(f"有效且去重后的催化剂数：{len(normalized)}")
        with st.expander("查看规范化后的催化剂配方", expanded=False):
            st.dataframe(normalized, use_container_width=True)

    return recipes, growth_mode


# =============================================================================
# Result display
# =============================================================================

def _render_audit(result: dict[str, Any]) -> None:
    audit = result.get("audit", {})
    stage_001 = audit.get("L123_001", {})
    stage_002 = audit.get("L123_002", {})
    stage_003 = audit.get("L123_003", {})

    columns = st.columns(5)
    columns[0].metric("催化剂", stage_001.get("catalysts", "—"))
    columns[1].metric("001路线", stage_001.get("rows", "—"))
    columns[2].metric("q50/q90交集", stage_002.get("overlap_rows", "—"))
    columns[3].metric("q50/q90并集", stage_002.get("union_rows", "—"))

    balanced_boundary = stage_003.get("balanced_boundary_fraction")
    potential_boundary = stage_003.get("potential_boundary_fraction")

    columns[4].metric(
        "Balanced边界率",
        f"{balanced_boundary:.1%}"
        if isinstance(balanced_boundary, (int, float))
        else "—",
    )

    if isinstance(balanced_boundary, (int, float)) and balanced_boundary >= 0.5:
        st.warning(
            "Balanced物理状态存在明显边界饱和；"
            "边界值只能解释为方向性推荐。"
        )

    if isinstance(potential_boundary, (int, float)) and potential_boundary >= 0.5:
        st.warning(
            "Potential物理状态存在明显边界饱和；"
            "边界值只能解释为方向性推荐。"
        )


def _render_rankings(tables: dict[str, Any]) -> None:
    st.markdown("### 催化剂排名")
    tabs = st.tabs([
        "Balanced（q50）",
        "Potential（q90）",
        "Conservative（q10）",
        "三类对比",
    ])

    for tab, criterion in zip(
        tabs[:3],
        ("balanced", "potential", "conservative"),
    ):
        with tab:
            frame = tables.get(RANK_KEYS[criterion])
            if isinstance(frame, pd.DataFrame):
                frame = _sort_ranking(frame, criterion)
            _show_table(CRITERION_LABELS[criterion], frame)

    with tabs[3]:
        _show_table(
            "Balanced / Potential / Conservative对比",
            tables.get("catalyst_comparison"),
            height=300,
        )


def _render_packages(tables: dict[str, Any]) -> None:
    st.markdown("### 工艺包")
    tabs = st.tabs([
        "Balanced Top-K",
        "Potential Top-K",
        "Conservative Top-K",
        "q50/q90并集",
    ])

    for tab, criterion in zip(
        tabs[:3],
        ("balanced", "potential", "conservative"),
    ):
        with tab:
            _show_table(
                f"{CRITERION_LABELS[criterion]}工艺包",
                tables.get(PACKAGE_KEYS[criterion]),
                height=300,
            )

    with tabs[3]:
        _show_table(
            "Balanced∪Potential路线包",
            tables.get("packages_union"),
            height=320,
        )


def _render_physical(tables: dict[str, Any]) -> None:
    st.markdown("### 物理状态推荐")
    st.caption(
        "变量为催化剂颗粒尺寸与raw C/M；推荐值只在当前网格内成立。"
    )

    tabs = st.tabs([
        "路线级推荐",
        "Top网格点",
        "边界审计",
        "最终合表",
    ])

    items = [
        ("process_packages_with_physical_state.csv",
         "physical_recommendations", 320),
        ("physical_state_top_points.csv",
         "physical_top_points", 320),
        ("L123-003 boundary audit",
         "boundary_audit", 320),
        ("WORLD_recommendations_final.csv",
         "world_final", 340),
    ]

    for tab, (title, key, height) in zip(tabs, items):
        with tab:
            _show_table(title, tables.get(key), height=height)


def _render_downloads(paths: dict[str, Any]) -> None:
    st.markdown("### 下载输出")
    first, second, third = st.columns(3)

    with first:
        _download_file(
            "下载催化剂三类对比",
            paths.get("catalyst_comparison"),
            "download_catalyst_comparison",
        )
        _download_file(
            "下载Balanced排名",
            paths.get("catalyst_balanced"),
            "download_catalyst_balanced",
        )
        _download_file(
            "下载Potential排名",
            paths.get("catalyst_potential"),
            "download_catalyst_potential",
        )
        _download_file(
            "下载Conservative排名",
            paths.get("catalyst_conservative"),
            "download_catalyst_conservative",
        )

    with second:
        _download_file(
            "下载Balanced工艺包",
            paths.get("packages_balanced"),
            "download_packages_balanced",
        )
        _download_file(
            "下载Potential工艺包",
            paths.get("packages_potential"),
            "download_packages_potential",
        )
        _download_file(
            "下载Conservative工艺包",
            paths.get("packages_conservative"),
            "download_packages_conservative",
        )
        _download_file(
            "下载q50/q90路线并集",
            paths.get("packages_union"),
            "download_packages_union",
        )

    with third:
        _download_file(
            "下载物理状态推荐",
            paths.get("physical_recommendations"),
            "download_physical_recommendations",
        )
        _download_file(
            "下载物理状态Top点",
            paths.get("physical_top_points"),
            "download_physical_top_points",
        )
        _download_file(
            "下载边界审计",
            paths.get("boundary_audit"),
            "download_boundary_audit",
        )
        _download_file(
            "下载最终合表",
            paths.get("world_final"),
            "download_world_final",
        )

    with st.expander("运行日志与清单", expanded=False):
        _download_file(
            "下载run_log.txt",
            paths.get("run_log"),
            "download_run_log",
        )
        _download_file(
            "下载run_manifest.json",
            paths.get("run_manifest"),
            "download_run_manifest",
        )


def _render_deepseek(
    result: dict[str, Any],
    tables: dict[str, Any],
) -> None:
    st.divider()
    st.markdown("### DeepSeek单催化剂分析")

    if not st.checkbox(
        "启用DeepSeek分析",
        value=False,
        help=(
            "只发送选中的催化剂、最佳路线、物理状态和边界审计。"
        ),
        key="single_enable_deepseek",
    ):
        return

    criterion = st.selectbox(
        "① 选择分析策略",
        ["balanced", "potential", "conservative"],
        format_func=lambda value: CRITERION_LABELS[value],
        key="single_deepseek_criterion",
    )

    ranking = tables.get(RANK_KEYS[criterion])
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        st.warning("当前策略没有可用催化剂排名。")
        return

    ranking = _sort_ranking(ranking, criterion)
    rank_position = int(st.number_input(
        "② 选择排名位置",
        min_value=1,
        max_value=len(ranking),
        value=1,
        step=1,
        key="single_deepseek_rank_position",
    ))

    st.markdown("#### 当前选择")
    st.dataframe(
        ranking.iloc[[rank_position - 1]],
        use_container_width=True,
    )

    try:
        bundle = build_selected_catalyst_bundle_single(
            run_result=result,
            criterion=criterion,
            rank_position=rank_position,
        )
    except Exception as exc:
        st.error(f"构造DeepSeek分析对象失败：{exc}")
        return

    render_deepseek_panel_streamlit(
        bundle=bundle,
        project_dir=PROJECT_DIR,
        deepseek_config=DeepSeekConfig(),
        default_question=(
            "请重点解释所选催化剂及路线的可执行设置、"
            "物理趋势、边界限制和下一步需要核实的参数。"
            "并按照该CVD工艺的类别给我具体的工艺操作的指导"
        ),
        title="DeepSeek selected-catalyst analysis",
        key_prefix=f"single_deepseek_{criterion}_{rank_position}",
    )


def _render_result_panel() -> None:
    st.subheader("输出结果")

    error = st.session_state.get(STATE_ERROR)
    if error:
        st.error(error)

    result = st.session_state.get(STATE_RESULT)
    if result is None:
        st.info("尚未运行。请完成输入后点击“开始推荐”。")
        return

    if result.get("status") != "success":
        st.error("当前结果不是成功状态。")
        return

    st.success(
        f"单模式推荐完成：Mode {result.get('growth_mode_letter', '—')}"
    )
    st.caption(f"本次运行目录：{result.get('run_dir', '')}")

    _render_audit(result)

    tables = result.get("tables", {})
    paths = result.get("paths", {})

    if not isinstance(tables, dict):
        st.error("运行结果缺少tables。")
        return
    if not isinstance(paths, dict):
        paths = {}

    _render_rankings(tables)
    _render_packages(tables)
    _render_physical(tables)
    _render_downloads(paths)
    _render_deepseek(result, tables)


# =============================================================================
# Public page
# =============================================================================

def render_single_mode_page() -> None:
    _initialize_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        h1, h2, h3, h4 {
            font-family: "Microsoft YaHei", system-ui, sans-serif;
        }
        .stButton > button {
            border-radius: 10px;
            min-height: 2.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("CNT单生长模式推荐")
    st.caption(
        "催化剂输入 → L123-001路线筛选 → "
        "L123-002三类排名与工艺包 → "
        "L123-003颗粒尺寸/raw C/M状态搜索。"
    )

    cfg = _render_config_sidebar()
    input_column, result_column = st.columns([0.40, 0.60], gap="large")

    with input_column:
        recipes, growth_mode = _render_input_panel()
        st.divider()

        run_column, clear_column = st.columns(2)

        with run_column:
            run_clicked = st.button(
                "开始推荐",
                type="primary",
                use_container_width=True,
                key="single_run_button",
            )

        with clear_column:
            clear_clicked = st.button(
                "清空结果",
                use_container_width=True,
                key="single_clear_result",
            )

        if clear_clicked:
            st.session_state[STATE_RESULT] = None
            st.session_state[STATE_ERROR] = None
            st.rerun()

        if run_clicked:
            st.session_state[STATE_ERROR] = None
            normalized, validation_error = _validate_recipes(recipes)

            if validation_error:
                st.session_state[STATE_ERROR] = validation_error
            elif normalized is None:
                st.session_state[STATE_ERROR] = "没有有效催化剂输入。"
            else:
                try:
                    with st.spinner(
                        "正在运行L123-001 → L123-002 → L123-003..."
                    ):
                        result = run_worldmap(
                            recipes_df=recipes,
                            growth_mode=growth_mode,
                            project_dir=PROJECT_DIR,
                            cfg=cfg,
                        )
                    st.session_state[STATE_RESULT] = result
                except Exception as exc:
                    st.session_state[STATE_ERROR] = str(exc)

            st.rerun()

    with result_column:
        _render_result_panel()


if __name__ == "__main__":
    render_single_mode_page()