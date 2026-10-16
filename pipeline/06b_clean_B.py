# -*- coding: utf-8 -*-

# ============================================================
# 06b_clean_B.py
#
# 功能：
#   对已经完成 growth mode 与 L1/L2/L3 划分的 B 模式 JSON
#   执行历史 B-mode canonical cleaning。
#
# 输入：
#   data/04_leveled/B/L1/*.json
#   data/04_leveled/B/L2/*.json
#   data/04_leveled/B/L3/*.json
#
# 输出：
#   data/05_cleaned/B/L1/CLEANED/*.json
#   data/05_cleaned/B/L2/CLEANED/*.json
#   data/05_cleaned/B/L3/CLEANED/*.json
#
# 原则：
#   1. 保留历史 B-mode cleaning 科学逻辑。
#   2. 保留 substrate-supported catalyst 定义。
#   3. 保留历史 T_CAT = 0.1 s effective C/M 规则。
#   4. 不修改、不移动 04_leveled 源 JSON。
#   5. 保持原始文件名不变。
#   6. nonuse 不进入后续处理。
# ============================================================

from pathlib import Path
import json


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "04_leveled"
    / "B"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "05_cleaned"
    / "B"
)

LEVELS = (
    "L1",
    "L2",
    "L3",
)

CLEAN_SUBDIR = "CLEANED"


# ============================================================
# 2. B 模式历史物理参数
# ============================================================

# 基底负载催化剂特征时间，严格保留旧代码规则
T_CAT = 0.1


# ============================================================
# 3. 基础工具
# ============================================================

def to_float(x):
    try:
        return float(x)
    except Exception:
        return ""


def normalize_temperature(val, unit):
    """
    温度统一到 °C。
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


def normalize_gas_composition(gas_flows):
    """
    历史 B-mode 规则：

    gas_flows
        ↓
    摩尔 / 体积分数

    单气体但没有有效流量：
        {gas: 1.0}

    多气体但没有有效流量：
        {}

    存在有效流量：
        按流量归一化。
    """

    if not gas_flows:
        return {}

    flows = {}
    gases = []

    if not isinstance(gas_flows, list):
        return {}

    for g in gas_flows:

        if not isinstance(g, dict):
            continue

        gas = g.get("gas")

        if gas:
            gases.append(gas)

            try:
                flow = float(
                    g.get("flow", "")
                )

            except Exception:
                flow = None

            if flow and flow > 0:
                flows[gas] = flow

    # 单气体、没有有效 flow
    if len(gases) == 1 and not flows:
        return {
            gases[0]: 1.0
        }

    # 多气体、完全没有有效 flow
    if len(gases) > 1 and not flows:
        return {}

    total = sum(
        flows.values()
    )

    if total <= 0:
        return {}

    return {
        gas: flow / total
        for gas, flow in flows.items()
    }


def normalize_calc_block(calc):
    """
    B 模式 calc_block 清洗。

    保留历史逻辑：
    1. ratio / concentration / density / flux
       尽可能转成 float；
    2. C_M_ratio_per_second × T_CAT
       → C_M_ratio_effective；
    3. C_M_nanoparticle_ratio_per_second × T_CAT
       → C_M_nanoparticle_ratio_effective；
    4. 写入 effective_time_s 和 C_M_definition。
    """

    if not isinstance(calc, dict):
        return {}

    new_calc = {}

    for k, v in calc.items():

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
    # 历史 B-mode effective C/M
    # --------------------------------------------------------

    if (
        new_calc.get(
            "C_M_ratio_per_second",
            "",
        )
        != ""
    ):

        new_calc[
            "effective_time_s"
        ] = T_CAT

        new_calc[
            "C_M_ratio_effective"
        ] = (
            new_calc[
                "C_M_ratio_per_second"
            ]
            * T_CAT
        )

    # --------------------------------------------------------
    # 历史 B-mode effective C/NP
    # --------------------------------------------------------

    if (
        new_calc.get(
            "C_M_nanoparticle_ratio_per_second",
            "",
        )
        != ""
    ):

        new_calc[
            "C_M_nanoparticle_ratio_effective"
        ] = (
            new_calc[
                "C_M_nanoparticle_ratio_per_second"
            ]
            * T_CAT
        )

    # --------------------------------------------------------
    # 历史定义说明
    # --------------------------------------------------------

    if (
        "C_M_ratio_per_second"
        in new_calc
        or
        "C_M_nanoparticle_ratio_per_second"
        in new_calc
    ):

        new_calc[
            "C_M_definition"
        ] = (
            "Effective ratios computed as flux × t_cat "
            "(substrate-supported catalyst, transient carbon supply)"
        )

    return new_calc


# ============================================================
# 4. B 模式单文件清洗
# ============================================================

def clean_one_file(data, level):

    clean = {
        "growth_mode": "B",
        "data_level": level,
        "paper": data.get(
            "paper",
            {},
        ),
        "syntheses": [],
    }

    # --------------------------------------------------------
    # 原始：
    #
    # synthesis_1
    # synthesis_2
    # ...
    #
    # ↓
    #
    # syntheses: [
    #     {
    #         synthesis_id,
    #         catalysts,
    #         process,
    #         calc_block,
    #         product
    #     }
    # ]
    # --------------------------------------------------------

    for key, syn in data.items():

        if not key.startswith(
            "synthesis_"
        ):
            continue

        if not isinstance(
            syn,
            dict,
        ):
            continue

        syn_clean = {
            "synthesis_id": key
        }

        # ====================================================
        # catalysts
        # ====================================================

        cats = []

        catalysts = syn.get(
            "catalysts",
            [],
        )

        if not isinstance(
            catalysts,
            list,
        ):
            catalysts = []

        for c in catalysts:

            if not isinstance(
                c,
                dict,
            ):
                continue

            c2 = {
                "catalyst_geometry":
                    "substrate_supported",

                "precursor": c.get(
                    "precursor",
                    "",
                ),

                "catalyst_type": c.get(
                    "catalyst_type",
                    "",
                ),

                "metals": [],
            }

            metals = c.get(
                "metals",
                [],
            )

            if not isinstance(
                metals,
                list,
            ):
                metals = []

            for m in metals:

                if not isinstance(
                    m,
                    dict,
                ):
                    continue

                c2[
                    "metals"
                ].append(
                    {
                        "element": m.get(
                            "element",
                            "",
                        ),

                        "atomic_percentage":
                            to_float(
                                m.get(
                                    "atomic_percentage"
                                )
                            ),
                    }
                )

            # -----------------------------------------------
            # B 模式历史 NP size 字段
            # -----------------------------------------------

            if (
                "catalyst_nanoparticle_size_nm"
                in c
            ):

                c2[
                    "nanoparticle_size_nm"
                ] = to_float(
                    c.get(
                        "catalyst_nanoparticle_size_nm"
                    )
                )

            cats.append(
                c2
            )

        syn_clean[
            "catalysts"
        ] = cats

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
            steps = [steps]

        if not isinstance(
            steps,
            list,
        ):
            steps = []

        process = {}

        # 历史规则：
        # 使用第一个 synthesis step
        if steps:

            s0 = steps[0]

            if isinstance(
                s0,
                dict,
            ):

                process = {

                    "temperature_C":
                        normalize_temperature(
                            s0.get(
                                "temperature"
                            ),
                            s0.get(
                                "temperature_unit"
                            ),
                        ),

                    "pressure_Pa":
                        to_float(
                            s0.get(
                                "pressure"
                            )
                        ),

                    "growth_time_min":
                        to_float(
                            s0.get(
                                "Growth time"
                            )
                        ),

                    "gas_composition":
                        normalize_gas_composition(
                            s0.get(
                                "gas_flows",
                                [],
                            )
                        ),

                    "carbon_source":
                        (
                            s0.get(
                                "carbon_source",
                                {},
                            ).get(
                                "name"
                            )

                            if isinstance(
                                s0.get(
                                    "carbon_source"
                                ),
                                dict,
                            )

                            else s0.get(
                                "carbon_source"
                            )
                        ),
                }

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

                "diameter_nm":
                    to_float(
                        p.get(
                            "diameter_nm"
                        )
                    ),

                "length_mm":
                    to_float(
                        p.get(
                            "length_mm"
                        )
                    ),

                "ig_id_ratio":
                    to_float(
                        p.get(
                            "ig_id_ratio"
                        )
                    ),

                "notes": p.get(
                    "notes",
                    "",
                ),
            }

        clean[
            "syntheses"
        ].append(
            syn_clean
        )

    return clean


# ============================================================
# 5. 单个 Level
# ============================================================

def process_level(level):

    src_dir = (
        INPUT_ROOT
        / level
    )

    dst_dir = (
        OUTPUT_ROOT
        / level
        / CLEAN_SUBDIR
    )

    if not src_dir.exists():

        print(
            f"[WARN] 输入目录不存在："
            f"{src_dir}"
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

    files = sorted(
        src_dir.glob(
            "*.json"
        ),
        key=lambda p:
            p.name.lower(),
    )

    print()

    print(
        f"===== B / {level} ====="
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

                data = json.load(
                    f
                )

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
# 6. 主程序
# ============================================================

def main():

    print("=" * 80)

    print(
        "CNT Mode-B Canonical Cleaning"
    )

    print("=" * 80)

    print(
        f"输入目录：{INPUT_ROOT}"
    )

    print(
        f"输出目录：{OUTPUT_ROOT}"
    )

    print(
        "处理等级：L1 / L2 / L3"
    )

    print(
        f"T_CAT：{T_CAT} s"
    )

    print(
        "nonuse：不处理"
    )

    print("=" * 80)

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"B 模式输入目录不存在："
            f"\n{INPUT_ROOT}"
        )

    results = []

    for level in LEVELS:

        result = process_level(
            level
        )

        results.append(
            result
        )

    # ========================================================
    # 7. 汇总
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
        "Mode-B cleaning finished"
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
        f"输出目录：{OUTPUT_ROOT}"
    )

    print("=" * 80)

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
            "B 模式所有 L1/L2/L3 "
            "JSON 均成功完成 canonical cleaning。"
        )


if __name__ == "__main__":
    main()