import hashlib
import json
from collections import Counter
from pathlib import Path

import fitz


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
ALLOWED_TYPES = {"fact", "summary", "unanswerable", "term", "comparison", "follow_up"}


def load_json(filename: str) -> list[dict]:
    return json.loads((BASE_DIR / filename).read_text(encoding="utf-8"))


def main() -> None:
    documents = load_json("documents.json")
    questions = load_json("questions.json")
    errors = []

    document_map = {item["document_key"]: item for item in documents}
    if len(document_map) != 3:
        errors.append("documents.json 必须包含 3 篇不同文档")
    if len(questions) != 15:
        errors.append("questions.json 必须包含 15 道题")

    for document in documents:
        pdf_path = PDF_DIR / document["filename"]
        if not pdf_path.exists():
            errors.append(f"缺少 PDF：{pdf_path.name}")
            continue

        actual_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if actual_hash != document["sha256"]:
            errors.append(f"PDF 哈希不匹配：{pdf_path.name}")

        with fitz.open(pdf_path) as pdf:
            if len(pdf) != document["page_count"]:
                errors.append(f"PDF 页数不匹配：{pdf_path.name}")
            if not any(page.get_text("text").strip() for page in pdf):
                errors.append(f"PDF 没有可解析文本：{pdf_path.name}")

    question_ids = [item["question_id"] for item in questions]
    if len(question_ids) != len(set(question_ids)):
        errors.append("question_id 必须唯一")

    counts = Counter(item["document_key"] for item in questions)
    for document_key in document_map:
        if counts[document_key] != 5:
            errors.append(f"{document_key} 必须恰好包含 5 道题")

    for question in questions:
        question_id = question["question_id"]
        document = document_map.get(question["document_key"])
        if document is None:
            errors.append(f"{question_id}: document_key 不存在")
            continue
        if question["question_type"] not in ALLOWED_TYPES:
            errors.append(f"{question_id}: 非法 question_type")
        if not question["question"].strip():
            errors.append(f"{question_id}: question 不能为空")
        if not question["gold_answer_points"]:
            errors.append(f"{question_id}: gold_answer_points 不能为空")
        if question["question_type"] == "follow_up" and not question["history"]:
            errors.append(f"{question_id}: 追问题必须提供 history")
        if question["expected_answerable"] and not question["source_pages"]:
            errors.append(f"{question_id}: 可回答问题必须提供 source_pages")
        if not question["expected_answerable"] and question["source_pages"]:
            errors.append(f"{question_id}: 无答案问题的 source_pages 应为空")
        for page in question["source_pages"]:
            if page < 1 or page > document["page_count"]:
                errors.append(f"{question_id}: source page {page} 超出文档范围")

    if errors:
        print("评估集校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("评估集校验通过：3 篇 PDF，15 道问题，每篇 5 道。")
    print("题型统计：", dict(Counter(item["question_type"] for item in questions)))


if __name__ == "__main__":
    main()
