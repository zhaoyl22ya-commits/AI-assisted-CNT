# ============================================================
# 04_split_by_growth_mode.py
#
# 功能：
#   读取 data/02_extracted_json 中的最终初始 JSON，
#   根据顶层 growth_mode 字段，
#   将文件复制到：
#
#   data/03_routed/A
#   data/03_routed/B
#   ...
#   data/03_routed/I
#
# 注意：
#   1. 不重新调用 LLM。
#   2. 不重新判断 growth mode。
#   3. 不修改或移动 02_extracted_json 中的原始 JSON。
#   4. 可重复运行；目标文件已存在时直接覆盖更新。
# ============================================================

from pathlib import Path
import json
import shutil


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "02_extracted_json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "03_routed"
)

VALID_MODES = tuple("ABCDEFGHI")


# ============================================================
# 2. 读取 growth_mode
# ============================================================

def extract_growth_mode(json_path: Path):
    """
    从最终 JSON 顶层读取 growth_mode。

    返回：
        A-I
        或 None
    """

    try:
        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except Exception as e:
        print(
            f"[FAIL] JSON 读取失败："
            f"{json_path.name} | {e}"
        )
        return None

    if not isinstance(data, dict):
        print(
            f"[FAIL] 顶层不是 JSON object："
            f"{json_path.name}"
        )
        return None

    mode = data.get(
        "growth_mode",
        ""
    )

    if mode is None:
        return None

    mode = str(mode).strip().upper()

    if mode in VALID_MODES:
        return mode

    return None


# ============================================================
# 3. 主程序
# ============================================================

def main():

    print(
        "=" * 80
    )
    print(
        "CNT JSON Split by Growth Mode"
    )
    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 检查输入目录
    # --------------------------------------------------------

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"输入目录不存在：\n{INPUT_DIR}"
        )

    # --------------------------------------------------------
    # 创建 A-I 输出目录
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for mode in VALID_MODES:
        (
            OUTPUT_ROOT
            / mode
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # 获取全部 JSON
    # --------------------------------------------------------

    files = sorted(
        INPUT_DIR.glob("*.json"),
        key=lambda p: p.name.lower(),
    )

    print(
        f"输入 JSON 总数：{len(files)}"
    )
    print(
        f"输入目录：{INPUT_DIR}"
    )
    print(
        f"输出目录：{OUTPUT_ROOT}"
    )
    print(
        "=" * 80
    )

    if not files:
        print(
            "没有发现 JSON 文件。"
        )
        return

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    counts = {
        mode: 0
        for mode in VALID_MODES
    }

    failed = 0

    # --------------------------------------------------------
    # 分类
    # --------------------------------------------------------

    for index, src_path in enumerate(
        files,
        start=1,
    ):

        mode = extract_growth_mode(
            src_path
        )

        if mode not in VALID_MODES:
            print(
                f"[SKIP] [{index}/{len(files)}] "
                f"未识别 growth_mode："
                f"{src_path.name}"
            )

            failed += 1
            continue

        dst_path = (
            OUTPUT_ROOT
            / mode
            / src_path.name
        )

        try:
            # 使用 copy2：
            # 保留原始 JSON，
            # 仅建立 routed 工作副本。
            #
            # 如果目标文件已经存在，
            # copy2 会覆盖更新，
            # 因此可以安全重复运行。
            shutil.copy2(
                src_path,
                dst_path,
            )

            counts[mode] += 1

            print(
                f"[OK]   [{index}/{len(files)}] "
                f"{src_path.name} → {mode}/"
            )

        except Exception as e:
            print(
                f"[FAIL] [{index}/{len(files)}] "
                f"{src_path.name} | {e}"
            )

            failed += 1

    # ========================================================
    # 4. 汇总
    # ========================================================

    print()
    print(
        "=" * 80
    )
    print(
        "Growth-mode split finished"
    )
    print(
        "=" * 80
    )

    total_routed = 0

    for mode in VALID_MODES:
        count = counts[mode]

        print(
            f"{mode}: {count}"
        )

        total_routed += count

    print(
        "-" * 80
    )
    print(
        f"成功分类：{total_routed}"
    )
    print(
        f"未识别/失败：{failed}"
    )
    print(
        f"输入总数：{len(files)}"
    )
    print(
        f"输出目录：{OUTPUT_ROOT}"
    )
    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 最基本的一致性检查
    # --------------------------------------------------------

    if (
        total_routed
        + failed
        != len(files)
    ):
        print(
            "[WARN] 计数不一致，请检查。"
        )

    elif failed == 0:
        print(
            "全部 JSON 均成功归入 A-I。"
        )

    else:
        print(
            "存在未识别文件，"
            "请先检查这些 JSON 的 growth_mode。"
        )


if __name__ == "__main__":
    main()