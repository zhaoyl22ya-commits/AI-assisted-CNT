# -*- coding: utf-8 -*-
"""
07b_merge_image_patches.py

将 data/05_cleaned 中 A-E / L1-L3 的全部 JSON
复制到 data/07_image_merged。

若对应的 data/07_image_patches/{MODE}/{LEVEL}/*.rejson 存在，
则按 synthesis_id 将图像补充字段加入原 JSON。

原则：
1. 所有 L1/L2/L3 cleaned JSON 都必须进入输出。
2. 没有 rejson 的文件原样复制。
3. rejson 只补空缺字段，不覆盖已有非空值。
4. synthesis 只按 synthesis_id 匹配，不做 index 猜测。
5. 保持当前正式字段名，不做旧版 IG_ID 等字段转换。
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# 1. 配置
# ============================================================

MODES = (
    "A",
    "B",
    "C",
    "D",
    "E",
)

LEVELS = (
    "L1",
    "L2",
    "L3",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "05_cleaned"
)

PATCH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "07_image_patches"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "07_image_merged"
)


PRODUCT_FIELDS = {
    "length_mm",
    "diameter_nm",
    "orientation_score",
    "ig_id_ratio",
    "purity_score",
    "bundle_density_per_um",
    "cnt_density_per_um",
}

CALC_FIELDS = {
    "np_size_nm",
}


# ============================================================
# 2. 基础工具
# ============================================================

def has_value(value: Any) -> bool:

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True


def load_json(path: Path) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON root must be object: {path}"
        )

    return data


def dump_json(
    path: Path,
    data: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def synthesis_map(
    document: dict[str, Any],
) -> dict[str, dict[str, Any]]:

    out = {}

    syntheses = (
        document.get("syntheses")
        or []
    )

    if not isinstance(
        syntheses,
        list,
    ):
        raise ValueError(
            "syntheses must be list."
        )

    for synthesis in syntheses:

        if not isinstance(
            synthesis,
            dict,
        ):
            raise ValueError(
                "Invalid synthesis object."
            )

        synthesis_id = (
            synthesis.get(
                "synthesis_id"
            )
        )

        if not isinstance(
            synthesis_id,
            str,
        ) or not synthesis_id.strip():

            raise ValueError(
                "Missing synthesis_id."
            )

        synthesis_id = (
            synthesis_id.strip()
        )

        if synthesis_id in out:

            raise ValueError(
                f"Duplicate synthesis_id: "
                f"{synthesis_id}"
            )

        out[
            synthesis_id
        ] = synthesis

    return out


def original_np_exists(
    synthesis: dict[str, Any],
) -> bool:

    calc_block = (
        synthesis.get(
            "calc_block"
        )
        or {}
    )

    if isinstance(
        calc_block,
        dict,
    ):

        if (
            has_value(
                calc_block.get(
                    "np_size_nm"
                )
            )
            or has_value(
                calc_block.get(
                    "Average_size_M_nanoparticle"
                )
            )
        ):

            return True

    catalysts = (
        synthesis.get(
            "catalysts"
        )
        or []
    )

    if isinstance(
        catalysts,
        list,
    ):

        for catalyst in catalysts:

            if not isinstance(
                catalyst,
                dict,
            ):
                continue

            if (
                has_value(
                    catalyst.get(
                        "nanoparticle_size_nm"
                    )
                )
                or has_value(
                    catalyst.get(
                        "catalyst_nanoparticle_size_nm"
                    )
                )
            ):

                return True

    return False


# ============================================================
# 3. Patch 合并
# ============================================================

def merge_patch(
    base_doc: dict[str, Any],
    patch_doc: dict[str, Any],
) -> tuple[int, int]:

    """
    Returns:
        added_fields
        blocked_existing_fields
    """

    base_map = synthesis_map(
        base_doc
    )

    patch_syntheses = (
        patch_doc.get(
            "syntheses"
        )
    )

    if not isinstance(
        patch_syntheses,
        list,
    ):

        raise ValueError(
            "Patch syntheses must be list."
        )

    added = 0
    blocked = 0

    for patch_synthesis in (
        patch_syntheses
    ):

        if not isinstance(
            patch_synthesis,
            dict,
        ):

            raise ValueError(
                "Invalid patch synthesis."
            )

        invalid_keys = (
            set(
                patch_synthesis
            )
            - {
                "synthesis_id",
                "product",
                "calc_block",
            }
        )

        if invalid_keys:

            raise ValueError(
                f"Unexpected patch fields: "
                f"{invalid_keys}"
            )

        synthesis_id = (
            patch_synthesis.get(
                "synthesis_id"
            )
        )

        if (
            not isinstance(
                synthesis_id,
                str,
            )
            or synthesis_id
            not in base_map
        ):

            raise ValueError(
                f"Unknown synthesis_id: "
                f"{synthesis_id!r}"
            )

        base_synthesis = (
            base_map[
                synthesis_id
            ]
        )

        # ----------------------------------------------------
        # product
        # ----------------------------------------------------

        patch_product = (
            patch_synthesis.get(
                "product"
            )
        )

        if patch_product is not None:

            if not isinstance(
                patch_product,
                dict,
            ):

                raise ValueError(
                    f"{synthesis_id}.product "
                    f"must be object."
                )

            invalid_product = (
                set(
                    patch_product
                )
                - PRODUCT_FIELDS
            )

            if invalid_product:

                raise ValueError(
                    f"{synthesis_id}: "
                    f"invalid product fields: "
                    f"{invalid_product}"
                )

            base_product = (
                base_synthesis.setdefault(
                    "product",
                    {},
                )
            )

            if not isinstance(
                base_product,
                dict,
            ):

                raise ValueError(
                    f"{synthesis_id}.product "
                    f"in base JSON is invalid."
                )

            for field, value in (
                patch_product.items()
            ):

                if not has_value(value):
                    continue

                if has_value(
                    base_product.get(
                        field
                    )
                ):

                    blocked += 1
                    continue

                base_product[
                    field
                ] = value

                added += 1

        # ----------------------------------------------------
        # calc_block.np_size_nm
        # ----------------------------------------------------

        patch_calc = (
            patch_synthesis.get(
                "calc_block"
            )
        )

        if patch_calc is not None:

            if not isinstance(
                patch_calc,
                dict,
            ):

                raise ValueError(
                    f"{synthesis_id}."
                    f"calc_block must be object."
                )

            invalid_calc = (
                set(
                    patch_calc
                )
                - CALC_FIELDS
            )

            if invalid_calc:

                raise ValueError(
                    f"{synthesis_id}: "
                    f"invalid calc fields: "
                    f"{invalid_calc}"
                )

            if (
                "np_size_nm"
                in patch_calc
                and has_value(
                    patch_calc[
                        "np_size_nm"
                    ]
                )
            ):

                if original_np_exists(
                    base_synthesis
                ):

                    blocked += 1

                else:

                    base_calc = (
                        base_synthesis.setdefault(
                            "calc_block",
                            {},
                        )
                    )

                    if not isinstance(
                        base_calc,
                        dict,
                    ):

                        raise ValueError(
                            f"{synthesis_id}."
                            f"calc_block in base "
                            f"JSON is invalid."
                        )

                    base_calc[
                        "np_size_nm"
                    ] = (
                        patch_calc[
                            "np_size_nm"
                        ]
                    )

                    added += 1

    return added, blocked


# ============================================================
# 4. 主程序
# ============================================================

def main() -> None:

    # 07_image_merged 是可重复生成的数据层。
    # 每次从 05_cleaned + 当前 rejson 全新重建，
    # 避免上一次运行留下旧文件。
    if OUT_ROOT.exists():

        shutil.rmtree(
            OUT_ROOT
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = Counter()

    for mode in MODES:

        for level in LEVELS:

            base_dir = (
                BASE_ROOT
                / mode
                / level
                / "CLEANED"
            )

            if not base_dir.exists():
                continue

            patch_dir = (
                PATCH_ROOT
                / mode
                / level
            )

            out_dir = (
                OUT_ROOT
                / mode
                / level
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            patch_map = {}

            if patch_dir.exists():

                for patch_path in (
                    patch_dir.glob(
                        "*.rejson"
                    )
                ):

                    patch_map[
                        patch_path.stem
                    ] = patch_path

            base_files = sorted(
                base_dir.glob(
                    "*.json"
                ),
                key=lambda p:
                    p.name.lower(),
            )

            print(
                f"\n[{mode}/{level}] "
                f"BASE={len(base_files)} "
                f"PATCH={len(patch_map)}"
            )

            used_patch_stems = set()

            for base_path in base_files:

                stats[
                    "base_files"
                ] += 1

                out_path = (
                    out_dir
                    / base_path.name
                )

                patch_path = (
                    patch_map.get(
                        base_path.stem
                    )
                )

                # ------------------------------------------------
                # 没有 patch：原样复制
                # ------------------------------------------------

                if patch_path is None:

                    shutil.copy2(
                        base_path,
                        out_path,
                    )

                    stats[
                        "copied_without_patch"
                    ] += 1

                    continue

                stats[
                    "patch_files_found"
                ] += 1

                used_patch_stems.add(
                    base_path.stem
                )

                base_doc = load_json(
                    base_path
                )

                patch_doc = load_json(
                    patch_path
                )

                added, blocked = (
                    merge_patch(
                        base_doc,
                        patch_doc,
                    )
                )

                stats[
                    "fields_added"
                ] += added

                stats[
                    "blocked_existing"
                ] += blocked

                if added == 0:

                    # patch 合法但没有任何新增内容，
                    # 保持原文件完全不变。
                    shutil.copy2(
                        base_path,
                        out_path,
                    )

                    stats[
                        "patch_no_change"
                    ] += 1

                else:

                    dump_json(
                        out_path,
                        base_doc,
                    )

                    stats[
                        "files_changed"
                    ] += 1

            # ----------------------------------------------------
            # 检查有没有找不到 base JSON 的孤立 patch
            # ----------------------------------------------------

            orphan = (
                set(
                    patch_map
                )
                - used_patch_stems
            )

            if orphan:

                stats[
                    "orphan_patches"
                ] += len(
                    orphan
                )

                print(
                    f"[WARNING] "
                    f"{mode}/{level} "
                    f"orphan patches: "
                    f"{len(orphan)}"
                )

    output_files = list(
        OUT_ROOT.rglob(
            "*.json"
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "IMAGE PATCH MERGE FINISHED"
    )

    print(
        f"BASE FILES            : "
        f"{stats['base_files']}"
    )

    print(
        f"OUTPUT FILES          : "
        f"{len(output_files)}"
    )

    print(
        f"PATCH FILES FOUND     : "
        f"{stats['patch_files_found']}"
    )

    print(
        f"COPIED WITHOUT PATCH  : "
        f"{stats['copied_without_patch']}"
    )

    print(
        f"FILES ACTUALLY CHANGED: "
        f"{stats['files_changed']}"
    )

    print(
        f"PATCH NO CHANGE       : "
        f"{stats['patch_no_change']}"
    )

    print(
        f"FIELDS ADDED          : "
        f"{stats['fields_added']}"
    )

    print(
        f"BLOCKED EXISTING      : "
        f"{stats['blocked_existing']}"
    )

    print(
        f"ORPHAN PATCHES        : "
        f"{stats['orphan_patches']}"
    )

    print(
        f"OUTPUT ROOT           : "
        f"{OUT_ROOT}"
    )

    print(
        "=" * 80
    )

    if (
        len(output_files)
        != stats[
            "base_files"
        ]
    ):

        raise RuntimeError(
            "Output file count does not "
            "match base file count."
        )


if __name__ == "__main__":
    main()