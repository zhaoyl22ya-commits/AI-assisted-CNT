# ============================================================
# 05_assign_L123.py
#
# 功能：
#   读取 data/03_routed 中 A-E 的 JSON，
#   按历史 L1/L2/L3/nonuse 判据进行数据等级划分。
#
# 输入：
#   data/03_routed/A-E/*.json
#
# 输出：
#   data/04_leveled/A-E/L1
#   data/04_leveled/A-E/L2
#   data/04_leveled/A-E/L3
#   data/04_leveled/A-E/nonuse
#
# 注意：
#   1. 保留历史 L1/L2/L3 判据，不改变科学逻辑。
#   2. 不移动、不修改 03_routed 中的源 JSON。
#   3. 使用 copy2 建立下一阶段工作副本。
# ============================================================

from pathlib import Path
import json
import shutil


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "03_routed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "04_leveled"
)

MODES = ("A", "B", "C", "D", "E")

LEVELS = (
    "L1",
    "L2",
    "L3",
    "nonuse",
)

RATIO_KEYS = (
    "C_M_ratio",
    "C_M_ratio_per_second",
    "C_M_nanoparticle_ratio_per_second",
)


# ============================================================
# 2. 基础工具
# ============================================================

def is_nonempty(value):
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip() != ""

    return True


def iter_syntheses(data):
    for key, synthesis in data.items():

        if (
            key.startswith("synthesis_")
            and isinstance(synthesis, dict)
        ):
            yield synthesis


# ============================================================
# 3. 判据 1：C/M ratio
# ============================================================

def has_cm_ratio(data):

    for synthesis in iter_syntheses(data):

        calc = synthesis.get(
            "calc_block",
            {},
        )

        if not isinstance(calc, dict):
            continue

        for key in RATIO_KEYS:

            if is_nonempty(
                calc.get(key)
            ):
                return True

    return False


# ============================================================
# 4. 判据 2：nanoparticle size
# ============================================================

def has_np_size(data):

    for synthesis in iter_syntheses(data):

        # A/B/C/E 常见位置
        catalysts = synthesis.get(
            "catalysts",
            [],
        )

        if isinstance(catalysts, list):

            for catalyst in catalysts:

                if not isinstance(
                    catalyst,
                    dict,
                ):
                    continue

                if is_nonempty(
                    catalyst.get(
                        "catalyst_nanoparticle_size_nm"
                    )
                ):
                    return True

        # D 模式历史字段
        calc = synthesis.get(
            "calc_block",
            {},
        )

        if isinstance(calc, dict):

            if is_nonempty(
                calc.get(
                    "Average_size_M_nanoparticle"
                )
            ):
                return True

    return False


# ============================================================
# 5. 判据 3：完整金属组成
# ============================================================

def has_metal_composition(data):

    for synthesis in iter_syntheses(data):

        catalysts = synthesis.get(
            "catalysts",
            [],
        )

        if not isinstance(
            catalysts,
            list,
        ):
            continue

        for catalyst in catalysts:

            if not isinstance(
                catalyst,
                dict,
            ):
                continue

            metals = catalyst.get(
                "metals",
                [],
            )

            if not isinstance(
                metals,
                list,
            ):
                continue

            for metal in metals:

                if not isinstance(
                    metal,
                    dict,
                ):
                    continue

                if (
                    is_nonempty(
                        metal.get("element")
                    )
                    and
                    is_nonempty(
                        metal.get(
                            "atomic_percentage"
                        )
                    )
                ):
                    return True

    return False


# ============================================================
# 6. 判定数据等级
# ============================================================

def assign_level(data):

    if has_cm_ratio(data):

        if has_np_size(data):
            return "L1"

        return "L2"

    if has_metal_composition(data):
        return "L3"

    return "nonuse"


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print("=" * 80)
    print("CNT Data-Level Assignment")
    print("=" * 80)

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"输入目录不存在：\n{INPUT_ROOT}"
        )

    counts = {
        mode: {
            level: 0
            for level in LEVELS
        }
        for mode in MODES
    }

    failed = {
        mode: 0
        for mode in MODES
    }

    total_input = 0

    # 创建目录
    for mode in MODES:

        for level in LEVELS:

            (
                OUTPUT_ROOT
                / mode
                / level
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    # --------------------------------------------------------
    # A-E
    # --------------------------------------------------------

    for mode in MODES:

        src_dir = (
            INPUT_ROOT
            / mode
        )

        if not src_dir.exists():

            print(
                f"[WARN] 不存在模式目录："
                f"{src_dir}"
            )
            continue

        files = sorted(
            src_dir.glob("*.json"),
            key=lambda p: p.name.lower(),
        )

        print()
        print(
            f"===== Mode {mode} | "
            f"{len(files)} JSON ====="
        )

        total_input += len(files)

        for index, src_path in enumerate(
            files,
            start=1,
        ):

            try:

                with open(
                    src_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

                if not isinstance(
                    data,
                    dict,
                ):
                    raise ValueError(
                        "JSON 顶层不是 object"
                    )

                level = assign_level(
                    data
                )

                dst_path = (
                    OUTPUT_ROOT
                    / mode
                    / level
                    / src_path.name
                )

                shutil.copy2(
                    src_path,
                    dst_path,
                )

                counts[
                    mode
                ][
                    level
                ] += 1

                print(
                    f"[OK] "
                    f"[{index}/{len(files)}] "
                    f"{src_path.name} "
                    f"→ {mode}/{level}/"
                )

            except Exception as e:

                failed[mode] += 1

                print(
                    f"[FAIL] "
                    f"{mode}/{src_path.name} "
                    f"| {e}"
                )

    # ========================================================
    # 8. 汇总
    # ========================================================

    print()
    print("=" * 80)
    print("L1/L2/L3 assignment finished")
    print("=" * 80)

    total_assigned = 0
    total_failed = 0

    for mode in MODES:

        mode_total = sum(
            counts[mode].values()
        )

        total_assigned += mode_total
        total_failed += failed[mode]

        print(
            f"{mode}: "
            f"L1={counts[mode]['L1']} | "
            f"L2={counts[mode]['L2']} | "
            f"L3={counts[mode]['L3']} | "
            f"nonuse={counts[mode]['nonuse']} | "
            f"failed={failed[mode]} | "
            f"total={mode_total}"
        )

    print("-" * 80)

    print(
        f"A-E 输入总数：{total_input}"
    )

    print(
        f"成功分级：{total_assigned}"
    )

    print(
        f"读取/处理失败：{total_failed}"
    )

    print(
        f"输出目录：{OUTPUT_ROOT}"
    )

    print("=" * 80)

    if (
        total_assigned
        + total_failed
        != total_input
    ):

        print(
            "[WARN] 计数不一致，请检查。"
        )

    elif total_failed == 0:

        print(
            "全部 A-E JSON "
            "均成功归入 L1/L2/L3/nonuse。"
        )


if __name__ == "__main__":
    main()