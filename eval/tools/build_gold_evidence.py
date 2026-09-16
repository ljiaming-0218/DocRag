"""Build a human-annotation draft from configured gold source pages."""

from pathlib import Path

import fitz

from eval.core import DATASET_DIR, PDF_DIR, RUNS_DIR, load_json, save_json

OUTPUT_PATH = RUNS_DIR / "drafts" / "gold_evidence.json"


def extract_pages(pdf_path: Path, page_numbers: list[int]) -> list[dict]:
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_number in page_numbers:
            pages.append({
                "page": page_number,
                "text": pdf[page_number - 1].get_text("text").strip(),
            })
    return pages


def main() -> None:
    documents = {
        item["document_key"]: item
        for item in load_json(DATASET_DIR / "documents.json")
    }
    cases = []
    for filename in ("single_turn.json", "follow_up.json"):
        cases.extend(load_json(DATASET_DIR / filename)["cases"])
    draft = []

    for case in cases:
        gold_pages = case.get("gold_pages") or []
        if not case.get("answerable") or not gold_pages:
            continue
        document = documents[case["document_key"]]
        draft.append({
            "case_id": case["case_id"],
            "document_key": case["document_key"],
            "query": case["query"],
            "answer_points": case["answer_points"],
            "gold_pages": gold_pages,
            "existing_gold_evidence": case.get("gold_evidence") or [],
            "page_candidates": extract_pages(
                PDF_DIR / document["filename"],
                gold_pages,
            ),
        })

    save_json(draft, OUTPUT_PATH)
    print(f"已生成 {len(draft)} 个案例的标注草稿：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
