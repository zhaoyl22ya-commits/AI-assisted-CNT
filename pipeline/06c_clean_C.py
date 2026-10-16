# -*- coding: utf-8 -*-

# ============================================================
# 06c_clean_C.py
#
# 功能：
#   对已经完成 growth mode 与 L1/L2/L3 划分的 C 模式 JSON
#   执行历史 C-mode canonical cleaning。
#
# 输入：
#   data/04_leveled/C/L1/*.json
#   data/04_leveled/C/L2/*.json
#   data/04_leveled/C/L3/*.json
#
# 输出：
#   data/05_cleaned/C/L1/CLEANED/*.json
#   data/05_cleaned/C/L2/CLEANED/*.json
#   data/05_cleaned/C/L3/CLEANED/*.json
#
# 原则：
#   1. 保留历史 C-mode cleaning 科学逻辑。
#   2. 保留 substrate-supported catalyst 定义。
#   3. 保留 T_CAT = 0.1 s effective C/M 规则。
#   4. 保留 gas / gas1 / gas2 / gasX 兼容逻辑。
#   5. 不修改、不移动 04_leveled 源 JSON。
#   6. 保持原始文件名不变。
#   7. nonuse 不进入后续处理。
# ============================================================

from pathlib import Path
import json


# ============================================================
# 1. 项目路径
# ============================================================

# 当前脚本：
# project_root/pipeline/06c_clean_C.py
#
# 所以 parents[1] = project_root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "04_leveled"
    / "C"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "05_cleaned"
    / "C"
)

LEVELS = (
    "L1",
    "L2",
    "L3",
)

CLEAN_SUBDIR = "CLEANED"


# ============================================================
# 2. C 模式历史物理参数
# ============================================================

# 基底负载催化剂特征时间（秒）
# 严格保留历史 C-mode 定义。
T_CAT = 0.1


# ============================================================
# 3. 基础工具
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


def normalize_gas_composition(gas_flows):
    """
    历史 C-mode gas composition 规则。

    兼容：
        gas
        gas1
        gas2
        gasX
        ...

    情况：
    1. 单气体，无有效 flow：
       {gas: 1.0}

    2. 多气体，全部无有效 flow：
       {}

    3. 存在有效 flow：
       按 flow 归一化为摩尔/体积分数。

    注意：
    保留 C cleaner 的 gas* 字段兼容逻辑。
    """

    if not gas_flows:
        return {}

    if not isinstance(
        gas_flows,
        list,
    ):
        return {}

    flows = {}
    gases_seen = []

    for g in gas_flows:

        if not isinstance(
            g,
            dict,
        ):
            continue

        # ----------------------------------------------------
        # 读取当前 gas entry 的 flow
        # ----------------------------------------------------

        try:

            flow = float(
                g.get(
                    "flow",
                    "",
                )
            )

        except Exception:

            flow = None

        # ----------------------------------------------------
        # 查找所有 gas* 字段
        # gas / gas1 / gas2 / gasX ...
        # ----------------------------------------------------

        for k, v in g.items():

            if not isinstance(
                k,
                str,
            ):
                continue

            if not k.lower().startswith(
                "gas"
            ):
                continue

            if not v:
                continue

            gas_name = v

            gases_seen.append(
                gas_name
            )

            if (
                flow is not None
                and flow > 0
            ):

                flows[
                    gas_name
                ] = flow

    # --------------------------------------------------------
    # 情况 1：单气体，没有有效流量
    # --------------------------------------------------------

    if (
        len(gases_seen) == 1
        and not flows
    ):

        return {
            gases_seen[0]: 1.0
        }

    # --------------------------------------------------------
    # 情况 2：多气体，全部无有效流量
    # --------------------------------------------------------

    if (
        len(gases_seen) > 1
        and not flows
    ):

        return {}

    # --------------------------------------------------------
    # 情况 3：存在有效流量
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


def normalize_calc_block(calc):
    """
    C 模式 calc_block 清洗。

    保留历史逻辑：

    1. ratio / concentration / density / flux
       尽可能转成 float；

    2. C_M_ratio_per_second × T_CAT
       → C_M_ratio_effective；

    3. C_M_nanoparticle_ratio_per_second × T_CAT
       → C_M_nanoparticle_ratio_effective；

    4. 保留 effective_time_s 和 C_M_definition。
    """

    if not isinstance(
        calc,
        dict,
    ):
        return {}

    new_calc = {}

    # --------------------------------------------------------
    # 基础数值字段统一
    # --------------------------------------------------------

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

            new_calc[
                k
            ] = to_float(
                v
            )

        else:

            new_calc[
                k
            ] = v

    # --------------------------------------------------------
    # effective C/M
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
    # effective C/NP
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
# 4. C 模式单文件清洗
# ============================================================

def clean_one_file(
    data,
    level,
):

    clean = {
        "growth_mode": "C",
        "data_level": level,
        "paper": data.get(
            "paper",
            {},
        ),
        "syntheses": [],
    }

    # --------------------------------------------------------
    # 输入：
    #
    # paper
    # synthesis_1
    # synthesis_2
    # ...
    #
    # ↓
    #
    # 输出：
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

                "precursor":
                    c.get(
                        "precursor",
                        "",
                    ),

                "catalyst_type":
                    c.get(
                        "catalyst_type",
                        "",
                    ),

                "metals": [],
            }

            # ------------------------------------------------
            # metals
            # ------------------------------------------------

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
                        "element":
                            m.get(
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

            # ------------------------------------------------
            # C 模式 nanoparticle size
            # ------------------------------------------------

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
        # 仅采用第一个 synthesis step
        # ----------------------------------------------------

        if steps:

            s0 = steps[0]

            if isinstance(
                s0,
                dict,
            ):

                carbon_source = (
                    s0.get(
                        "carbon_source"
                    )
                )

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
                            carbon_source.get(
                                "name"
                            )
                            if isinstance(
                                carbon_source,
                                dict,
                            )
                            else carbon_source
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

                "form":
                    p.get(
                        "form",
                        "",
                    ),

                "cnt_type":
                    p.get(
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

                "notes":
                    p.get(
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

    # --------------------------------------------------------
    # 输入：
    # data/04_leveled/C/L1
    # data/04_leveled/C/L2
    # data/04_leveled/C/L3
    # --------------------------------------------------------

    src_dir = (
        INPUT_ROOT
        / level
    )

    # --------------------------------------------------------
    # 输出：
    # data/05_cleaned/C/L1/CLEANED
    # data/05_cleaned/C/L2/CLEANED
    # data/05_cleaned/C/L3/CLEANED
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

    # 只读取当前 level 根目录 *.json
    # 不递归，避免误读其他子目录。
    files = sorted(
        src_dir.glob(
            "*.json"
        ),
        key=lambda p:
            p.name.lower(),
    )

    print()

    print(
        f"===== C / {level} ====="
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

            # 保持文件名不变
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

                f.write(
                    "\n"
                )

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

    print(
        "=" * 80
    )

    print(
        "CNT Mode-C Canonical Cleaning"
    )

    print(
        "=" * 80
    )

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
        f"T_CAT：{T_CAT} s"
    )

    print(
        "nonuse：不处理"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 总输入目录检查
    # --------------------------------------------------------

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            "C 模式输入目录不存在："
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

    print(
        "=" * 80
    )

    print(
        "Mode-C cleaning finished"
    )

    print(
        "=" * 80
    )

    for r in results:

        print(
            f"{r['level']}: "
            f"input={r['input']} | "
            f"done={r['done']} | "
            f"failed={r['failed']}"
        )

    print(
        "-" * 80
    )

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

    print(
        "=" * 80
    )

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
            "[OK] C 模式所有 L1/L2/L3 "
            "JSON 均成功完成 canonical cleaning。"
        )


if __name__ == "__main__":
    main()