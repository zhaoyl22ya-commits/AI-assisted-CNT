# -*- coding: utf-8 -*-

# ============================================================
# 06d_clean_D.py
#
# 功能：
#   对已经完成 growth mode 与 L1/L2/L3 划分的 D 模式 JSON
#   执行历史 D-mode canonical cleaning。
#
# 输入：
#   data/04_leveled/D/L1/*.json
#   data/04_leveled/D/L2/*.json
#   data/04_leveled/D/L3/*.json
#
# 输出：
#   data/05_cleaned/D/L1/CLEANED/*.json
#   data/05_cleaned/D/L2/CLEANED/*.json
#   data/05_cleaned/D/L3/CLEANED/*.json
#
# 原则：
#   1. 保留历史 D-mode cleaning 科学逻辑。
#   2. 保留 D 模式 nanoparticle-size 双来源兼容。
#   3. 保留 D 模式 NP 级物理量字段统一。
#   4. 不修改、不移动 04_leveled 源 JSON。
#   5. 保持原始文件名不变。
#   6. nonuse 不进入后续处理。
# ============================================================

from pathlib import Path
import json


# ============================================================
# 1. 项目路径
# ============================================================

# 当前脚本：
# project_root/pipeline/06d_clean_D.py
#
# 所以 parents[1] = project_root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "04_leveled"
    / "D"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "05_cleaned"
    / "D"
)

LEVELS = (
    "L1",
    "L2",
    "L3",
)

CLEAN_SUBDIR = "CLEANED"


# ============================================================
# 2. 基础工具
# ============================================================

def to_float(x):
    """
    尝试转为 float。
    无法转换时保持历史规则：返回空字符串。
    """
    try:
        return float(x)
    except Exception:
        return ""


def normalize_temperature(val, unit):
    """
    温度统一为 °C。
    """

    if val in ["", None]:
        return ""

    try:
        v = float(val)
    except Exception:
        return ""

    if unit and "k" in str(unit).lower():
        return v - 273.15

    return v


def normalize_calc_block(calc_block):
    """
    D 模式 calc_block 清洗。

    保留历史逻辑：

    1. 字段名包含：
       ratio
       concentration
       density
       flux

       的值尽可能统一为 float。

    2. D 模式特殊字段：

       Average_size_M_nanoparticle
           → np_size_nm

       Metal_atoms_per_nanoparticle
           → metal_atoms_per_np

       C_NP_ratio
           → C_NP_ratio（float）
    """

    if not isinstance(calc_block, dict):
        return {}

    new_calc = {}

    # --------------------------------------------------------
    # 基础数值字段规范化
    # --------------------------------------------------------

    for k, v in calc_block.items():

        if any(
            tag in k.lower()
            for tag in [
                "ratio",
                "concentration",
                "density",
                "flux",
            ]
        ):

            new_calc[k] = to_float(v)

        else:

            new_calc[k] = v

    # --------------------------------------------------------
    # D 模式关键 NP 级物理量
    # --------------------------------------------------------

    if "Average_size_M_nanoparticle" in calc_block:

        new_calc["np_size_nm"] = to_float(
            calc_block.get(
                "Average_size_M_nanoparticle"
            )
        )

    if "Metal_atoms_per_nanoparticle" in calc_block:

        new_calc["metal_atoms_per_np"] = to_float(
            calc_block.get(
                "Metal_atoms_per_nanoparticle"
            )
        )

    if "C_NP_ratio" in calc_block:

        new_calc["C_NP_ratio"] = to_float(
            calc_block.get(
                "C_NP_ratio"
            )
        )

    return new_calc


def normalize_gas_composition(gas_flows):
    """
    保留历史 D-mode gas composition 逻辑。

    D 使用普通 gas 字段读取方式，
    不使用 C/E 的 gas1 / gas2 / gasX 扩展规则。

    单气体但没有有效流量：
        {gas: 1.0}

    多气体全部没有有效流量：
        {}

    存在有效流量：
        按流量归一化。
    """

    if not gas_flows:
        return {}

    if not isinstance(gas_flows, list):
        return {}

    flows = {}
    gases = []

    for g in gas_flows:

        if not isinstance(g, dict):
            continue

        gas = g.get("gas")

        if gas:

            gases.append(gas)

            try:
                flow = float(
                    g.get(
                        "flow",
                        "",
                    )
                )

            except Exception:
                flow = None

            if flow and flow > 0:
                flows[gas] = flow

    # --------------------------------------------------------
    # 单气体，无有效流量
    # --------------------------------------------------------

    if len(gases) == 1 and not flows:

        return {
            gases[0]: 1.0
        }

    # --------------------------------------------------------
    # 多气体，无有效流量
    # --------------------------------------------------------

    if len(gases) > 1 and not flows:

        return {}

    # --------------------------------------------------------
    # 有有效流量，归一化
    # --------------------------------------------------------

    total = sum(
        flows.values()
    )

    if total <= 0:
        return {}

    return {
        gas: flow / total
        for gas, flow in flows.items()
    }


# ============================================================
# 3. D 模式单文件清洗
# ============================================================

def clean_one_file(data, data_level):

    clean = {
        "growth_mode": "D",
        "data_level": data_level,
        "paper": data.get(
            "paper",
            {},
        ),
    }

    syntheses = []

    # --------------------------------------------------------
    # 原始：
    #
    # paper
    # synthesis_1
    # synthesis_2
    # ...
    #
    # ↓
    #
    # canonical：
    #
    # growth_mode
    # data_level
    # paper
    # syntheses[
    #     synthesis_id
    #     catalysts
    #     process
    #     calc_block
    #     product
    # ]
    # --------------------------------------------------------

    for key, syn in data.items():

        if (
            not key.startswith("synthesis_")
            or
            not isinstance(syn, dict)
        ):
            continue

        syn_clean = {
            "synthesis_id": key
        }

        # ====================================================
        # catalysts
        # ====================================================

        catalysts_clean = []

        catalysts = syn.get(
            "catalysts",
            [],
        )

        if not isinstance(catalysts, list):
            catalysts = []

        for c in catalysts:

            if not isinstance(c, dict):
                continue

            c2 = {

                "precursor": c.get(
                    "precursor",
                    "",
                ),

                "catalyst_type": c.get(
                    "catalyst_type",
                    "",
                ),
            }

            # ------------------------------------------------
            # metals
            # ------------------------------------------------

            metals = []

            metal_items = c.get(
                "metals",
                [],
            )

            if not isinstance(
                metal_items,
                list,
            ):
                metal_items = []

            for m in metal_items:

                if not isinstance(
                    m,
                    dict,
                ):
                    continue

                metals.append(
                    {
                        "element": m.get(
                            "element",
                            "",
                        ),

                        "atomic_percentage": to_float(
                            m.get(
                                "atomic_percentage"
                            )
                        ),
                    }
                )

            c2["metals"] = metals

            # ------------------------------------------------
            # D 模式 nanoparticle size 双来源兼容
            #
            # 优先：
            # 1. catalyst_nanoparticle_size_nm
            #
            # 若缺失：
            # 2. calc_block.Average_size_M_nanoparticle
            # ------------------------------------------------

            np_size = ""

            if (
                c.get(
                    "catalyst_nanoparticle_size_nm"
                )
                not in ["", None]
            ):

                np_size = to_float(
                    c.get(
                        "catalyst_nanoparticle_size_nm"
                    )
                )

            calc = syn.get(
                "calc_block",
                {},
            )

            if (
                np_size in ["", None]
                and isinstance(calc, dict)
            ):

                if (
                    calc.get(
                        "Average_size_M_nanoparticle"
                    )
                    not in ["", None]
                ):

                    np_size = to_float(
                        calc.get(
                            "Average_size_M_nanoparticle"
                        )
                    )

            if np_size not in ["", None]:

                c2[
                    "nanoparticle_size_nm"
                ] = np_size

            catalysts_clean.append(
                c2
            )

        syn_clean[
            "catalysts"
        ] = catalysts_clean

        # ====================================================
        # process
        # ====================================================

        steps = syn.get(
            "synthesis_steps",
            [],
        )

        if isinstance(
            steps,
            dict,
        ):

            steps = [
                steps
            ]

        if not isinstance(
            steps,
            list,
        ):

            steps = []

        process = {}

        # ----------------------------------------------------
        # 保留历史规则：
        # 只读取第一个 synthesis step
        # ----------------------------------------------------

        if steps:

            s0 = steps[0]

            if isinstance(
                s0,
                dict,
            ):

                process[
                    "temperature_C"
                ] = normalize_temperature(
                    s0.get(
                        "temperature"
                    ),
                    s0.get(
                        "temperature_unit"
                    ),
                )

                process[
                    "pressure_Pa"
                ] = to_float(
                    s0.get(
                        "pressure"
                    )
                )

                process[
                    "growth_time_min"
                ] = to_float(
                    s0.get(
                        "Growth time"
                    )
                )

                process[
                    "gas_composition"
                ] = normalize_gas_composition(
                    s0.get(
                        "gas_flows",
                        [],
                    )
                )

                carbon_source = (
                    s0.get(
                        "carbon_source"
                    )
                )

                process[
                    "carbon_source"
                ] = (
                    carbon_source.get(
                        "name"
                    )
                    if isinstance(
                        carbon_source,
                        dict,
                    )
                    else carbon_source
                )

        syn_clean[
            "process"
        ] = process

        # ====================================================
        # calc_block
        # ====================================================

        syn_clean[
            "calc_block"
        ] = normalize_calc_block(
            syn.get(
                "calc_block",
                {},
            )
        )

        # ====================================================
        # product
        # ====================================================

        if (
            "product" in syn
            and
            isinstance(
                syn.get(
                    "product"
                ),
                dict,
            )
        ):

            p = syn[
                "product"
            ]

            syn_clean[
                "product"
            ] = {

                "form": p.get(
                    "form",
                    "",
                ),

                "cnt_type": p.get(
                    "cnt_type",
                    "",
                ),

                "diameter_nm": to_float(
                    p.get(
                        "diameter_nm"
                    )
                ),

                "length_mm": to_float(
                    p.get(
                        "length_mm"
                    )
                ),

                "ig_id_ratio": to_float(
                    p.get(
                        "ig_id_ratio"
                    )
                ),

                "notes": p.get(
                    "notes",
                    "",
                ),
            }

        syntheses.append(
            syn_clean
        )

    clean[
        "syntheses"
    ] = syntheses

    return clean


# ============================================================
# 4. 单个 Level
# ============================================================

def process_level(level):

    # --------------------------------------------------------
    # 输入：
    # data/04_leveled/D/L1
    # data/04_leveled/D/L2
    # data/04_leveled/D/L3
    # --------------------------------------------------------

    src_dir = (
        INPUT_ROOT
        / level
    )

    # --------------------------------------------------------
    # 输出：
    # data/05_cleaned/D/L1/CLEANED
    # data/05_cleaned/D/L2/CLEANED
    # data/05_cleaned/D/L3/CLEANED
    # --------------------------------------------------------

    dst_dir = (
        OUTPUT_ROOT
        / level
        / CLEAN_SUBDIR
    )

    if not src_dir.exists():

        print(
            f"[WARN] 输入目录不存在：{src_dir}"
        )

        return {
            "level": level,
            "input": 0,
            "done": 0,
            "failed": 0,
        }

    dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 只读取当前 level 根目录里的 JSON，
    # 不递归扫描，避免误读其他目录。
    files = sorted(
        src_dir.glob("*.json"),
        key=lambda p: p.name.lower(),
    )

    print()
    print(
        f"===== D / {level} ====="
    )

    print(
        f"输入目录：{src_dir}"
    )

    print(
        f"输出目录：{dst_dir}"
    )

    print(
        f"输入 JSON：{len(files)}"
    )

    done = 0
    failed = 0

    for index, src_path in enumerate(
        files,
        start=1,
    ):

        try:

            with src_path.open(
                "r",
                encoding="utf-8-sig",
            ) as f:

                data = json.load(f)

            if not isinstance(
                data,
                dict,
            ):

                raise ValueError(
                    "JSON 顶层不是 object"
                )

            cleaned = clean_one_file(
                data,
                level,
            )

            # 保持原始文件名不变
            dst_path = (
                dst_dir
                / src_path.name
            )

            with dst_path.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    cleaned,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

                f.write("\n")

            done += 1

            print(
                f"[OK] "
                f"[{index}/{len(files)}] "
                f"{src_path.name}"
            )

        except Exception as e:

            failed += 1

            print(
                f"[FAIL] "
                f"{src_path.name} "
                f"| {e}"
            )

    return {
        "level": level,
        "input": len(files),
        "done": done,
        "failed": failed,
    }


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 80)

    print(
        "CNT Mode-D Canonical Cleaning"
    )

    print("=" * 80)

    print(
        f"项目根目录：{PROJECT_ROOT}"
    )

    print(
        f"输入根目录：{INPUT_ROOT}"
    )

    print(
        f"输出根目录：{OUTPUT_ROOT}"
    )

    print(
        "处理等级：L1 / L2 / L3"
    )

    print(
        "nonuse：不处理"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 检查总输入目录
    # --------------------------------------------------------

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            "D 模式输入目录不存在："
            f"\n{INPUT_ROOT}"
        )

    results = []

    # --------------------------------------------------------
    # L1 / L2 / L3
    # --------------------------------------------------------

    for level in LEVELS:

        result = process_level(
            level
        )

        results.append(
            result
        )

    # ========================================================
    # 6. 汇总
    # ========================================================

    total_input = sum(
        r["input"]
        for r in results
    )

    total_done = sum(
        r["done"]
        for r in results
    )

    total_failed = sum(
        r["failed"]
        for r in results
    )

    print()
    print("=" * 80)

    print(
        "Mode-D cleaning finished"
    )

    print("=" * 80)

    for r in results:

        print(
            f"{r['level']}: "
            f"input={r['input']} | "
            f"done={r['done']} | "
            f"failed={r['failed']}"
        )

    print("-" * 80)

    print(
        f"输入总数：{total_input}"
    )

    print(
        f"成功清洗：{total_done}"
    )

    print(
        f"失败：{total_failed}"
    )

    print(
        f"最终输出：{OUTPUT_ROOT}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 计数闭合检查
    # --------------------------------------------------------

    if (
        total_done
        + total_failed
        != total_input
    ):

        print(
            "[WARN] 计数不闭合，请检查。"
        )

    elif total_failed == 0:

        print(
            "[OK] D 模式所有 L1/L2/L3 "
            "JSON 均成功完成 canonical cleaning。"
        )


if __name__ == "__main__":
    main()