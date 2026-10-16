# -*- coding: utf-8 -*-

# ============================================================
# 06a_clean_A.py
#
# 功能：
#   对已经完成 growth mode 与 L1/L2/L3 划分的 A 模式 JSON
#   执行历史 A-mode canonical cleaning。
#
# 输入：
#   data/04_leveled/A/L1/*.json
#   data/04_leveled/A/L2/*.json
#   data/04_leveled/A/L3/*.json
#
# 输出：
#   data/05_cleaned/A/L1/CLEANED/*.json
#   data/05_cleaned/A/L2/CLEANED/*.json
#   data/05_cleaned/A/L3/CLEANED/*.json
#
# 原则：
#   1. 保留历史 A-mode cleaning 科学逻辑。
#   2. 不修改、不移动 04_leveled 源 JSON。
#   3. 保持原始文件名不变。
#   4. nonuse 不进入后续处理。
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
    / "A"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "05_cleaned"
    / "A"
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
    try:
        return float(x)
    except Exception:
        return ""


def normalize_temperature(val, unit):
    """
    历史规则：
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


def normalize_calc_block(calc_block):
    """
    历史规则：
    calc_block 中名称包含
    ratio / concentration / density / flux
    的字段尝试统一为 float。
    """
    if not isinstance(calc_block, dict):
        return {}

    new_calc = {}

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

    return new_calc


def normalize_gas_composition(gas_flows):
    """
    历史 A-mode 规则：

    gas_flows
        ↓
    摩尔/体积分数表示

    单气体但没有有效流量：
        {gas: 1.0}

    多气体但没有有效流量：
        {}

    存在有效流量：
        按各气体流量归一化。
    """

    if not gas_flows:
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
                    g.get("flow", "")
                )

            except Exception:
                flow = None

            if flow and flow > 0:
                flows[gas] = flow

    # 单气体无流量
    if len(gases) == 1 and not flows:
        return {
            gases[0]: 1.0
        }

    # 多气体无流量
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


# ============================================================
# 3. A 模式单文件清洗
# ============================================================

def clean_one_file(data, data_level):

    clean = {
        "growth_mode": data.get(
            "growth_mode",
            "A",
        ),
        "data_level": data_level,
        "paper": data.get(
            "paper",
            {},
        ),
    }

    syntheses = []

    # --------------------------------------------------------
    # 原始 JSON：
    #
    # synthesis_1
    # synthesis_2
    # ...
    #
    # ↓
    #
    # syntheses: [
    #   {
    #       synthesis_id,
    #       catalysts,
    #       process,
    #       calc_block,
    #       product
    #   }
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
                "precursor": c.get(
                    "precursor",
                    "",
                ),
                "catalyst_type": c.get(
                    "catalyst_type",
                    "",
                ),
            }

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

            # 历史 A-mode nanoparticle-size 字段
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

            # 历史 A-mode promoter 字段
            if c.get(
                "catalyst promoter precursor"
            ):

                c2["promoter"] = {
                    "name": c.get(
                        "catalyst promoter precursor"
                    ),
                    "feed_rate_mol_s": to_float(
                        c.get(
                            "catalyst_promoter_feed_rate"
                        )
                    ),
                }

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
            steps = [steps]

        if not isinstance(
            steps,
            list,
        ):
            steps = []

        process = {}

        # 保留历史规则：
        # 只使用第一个 synthesis step
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

                carbon_source = s0.get(
                    "carbon_source"
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
                syn.get("product"),
                dict,
            )
        ):

            p = syn["product"]

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
# 4. 单个 level
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
        src_dir.glob("*.json"),
        key=lambda p: p.name.lower(),
    )

    print()
    print(
        f"===== A / {level} ====="
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
        "CNT Mode-A Canonical Cleaning"
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
        "nonuse：不处理"
    )

    print("=" * 80)

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"A 模式输入目录不存在："
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
    # 汇总
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
        "Mode-A cleaning finished"
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
            "A 模式所有 L1/L2/L3 "
            "JSON 均成功完成 canonical cleaning。"
        )


if __name__ == "__main__":
    main()