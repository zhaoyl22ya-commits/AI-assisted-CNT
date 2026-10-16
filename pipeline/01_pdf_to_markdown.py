import os
import re
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path


# ============================================================
# 项目路径
# ============================================================

# 兼容两种情况：
# 1. 当前脚本仍然放在项目根目录
# 2. 后续脚本移动到 pipeline/ 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "data").exists():
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "00_raw_papers"
OUTPUT_DIR = PROJECT_ROOT / "data" / "01_markdown"


# ============================================================
# 安全提取 PDF 文本
# ============================================================

def extract_text_safe(page):
    try:
        text = page.get_text("text")
        if len(text.strip()) > 10:
            return text
    except:
        pass

    try:
        blocks = page.get_text("blocks")
        if blocks:
            blocks_text = "\n".join(
                [b[4] for b in blocks if len(b[4].strip()) > 0]
            )

            if len(blocks_text.strip()) > 10:
                return blocks_text
    except:
        pass

    try:
        raw = page.get_text("rawdict")
        items = []

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(
                    [
                        span["text"]
                        for span in line.get("spans", [])
                    ]
                )

                if line_text.strip():
                    items.append(line_text)

        return "\n".join(items)

    except:
        pass

    return ""


# ============================================================
# PDF → Markdown
# ============================================================

def pdf_to_markdown(pdf_path, md_path):
    try:
        doc = fitz.open(pdf_path)

    except Exception as e:
        print(
            f"[ERROR] 无法打开 PDF：{pdf_path}  原因：{e}"
        )
        return False

    print(
        f"[INFO] PDF：{os.path.basename(pdf_path)} "
        f"（{doc.page_count} 页）"
    )

    try:
        with open(md_path, "w", encoding="utf-8") as out:

            for i in range(doc.page_count):

                page = doc.load_page(i)

                text = extract_text_safe(page)

                out.write(
                    f"\n\n---\n\n# Page {i + 1}\n\n"
                )

                out.write(text)

        print(
            f"[OK] 已输出：{md_path}"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Markdown 写入失败："
            f"{md_path}  原因：{e}"
        )

        return False

    finally:
        doc.close()


# ============================================================
# Word → Markdown
# ============================================================

def word_to_markdown(doc_path, md_path):
    try:
        doc = Document(doc_path)

    except Exception as e:

        print(
            f"[ERROR] 无法打开 Word："
            f"{doc_path}  原因：{e}"
        )

        return False

    print(
        f"[INFO] Word：{os.path.basename(doc_path)}"
    )

    try:
        with open(md_path, "w", encoding="utf-8") as out:

            for para in doc.paragraphs:

                text = para.text.strip()

                if text:
                    out.write(text + "\n\n")

        print(
            f"[OK] 已输出：{md_path}"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Markdown 写入失败："
            f"{md_path}  原因：{e}"
        )

        return False


# ============================================================
# 批量 PDF / Word → Markdown
# ============================================================

def batch_convert(input_folder, output_folder):

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(
            (".pdf", ".doc", ".docx")
        )
    ]

    if not files:

        print(
            "[WARN] 未找到 PDF 或 Word 文件"
        )

        return

    files.sort()

    print("=" * 80)

    print(
        f"[INFO] 输入目录：{input_folder}"
    )

    print(
        f"[INFO] 输出目录：{output_folder}"
    )

    print(
        f"[INFO] 找到 {len(files)} 个文件"
    )

    print("=" * 80)

    success_count = 0
    failed_count = 0

    for index, fname in enumerate(
        files,
        start=1
    ):

        print(
            f"\n[{index}/{len(files)}] "
            f"{fname}"
        )

        fpath = input_folder / fname

        md_path = output_folder / (
            fname.rsplit(".", 1)[0] + ".md"
        )

        if fname.lower().endswith(".pdf"):

            success = pdf_to_markdown(
                str(fpath),
                str(md_path)
            )

        else:

            success = word_to_markdown(
                str(fpath),
                str(md_path)
            )

        if success:
            success_count += 1

        else:
            failed_count += 1


    print("\n" + "=" * 80)

    print("PDF / Word → Markdown 完成")

    print("=" * 80)

    print(
        f"输入文件：{len(files)}"
    )

    print(
        f"成功转换：{success_count}"
    )

    print(
        f"失败：{failed_count}"
    )

    print(
        f"Markdown 输出目录：{output_folder}"
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    batch_convert(
        INPUT_DIR,
        OUTPUT_DIR
    )