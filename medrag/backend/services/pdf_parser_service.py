from pathlib import Path

import fitz

from config import (
    OCR_DPI,
    OCR_ENABLED,
    OCR_LANGUAGES,
    OCR_MIN_TEXT_CHARS,
    OCR_TESSDATA_DIR,
)


def page_needs_ocr(page: fitz.Page, text: str) -> bool:
    return len(text.strip()) < OCR_MIN_TEXT_CHARS and bool(page.get_images(full=True))


def extract_page_with_ocr(page: fitz.Page) -> str:
    try:
        text_page = page.get_textpage_ocr(
            language=OCR_LANGUAGES,
            dpi=OCR_DPI,
            full=True,
            tessdata=OCR_TESSDATA_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            "OCR 识别失败。请确认已安装 Tesseract，并配置正确的 TESSDATA_PREFIX 和语言包。"
        ) from exc

    return page.get_text("text", textpage=text_page).strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")

    pages = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            image_count = len(page.get_images(full=True))
            extraction_method = "text"

            if page_needs_ocr(page, text):
                if OCR_ENABLED:
                    text = extract_page_with_ocr(page)
                    extraction_method = "ocr" if text else "ocr_empty"
                else:
                    extraction_method = "ocr_required"
            elif not text:
                extraction_method = "empty"

            pages.append(
                {
                    "页码": index,
                    "文本": text,
                    "提取方式": extraction_method,
                    "图片数量": image_count,
                }
            )

    return pages


def extract_pdf_text(pdf_path: str | Path) -> str:
    pages = extract_pdf_pages(pdf_path)
    return "\n\n".join(
        f"[第 {item['页码']} 页]\n{item['文本']}" for item in pages if item["文本"]
    )
