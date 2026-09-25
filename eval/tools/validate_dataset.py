"""Validate the canonical DocRAG evaluation dataset."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter

import fitz

from eval.core import DATASET_DIR, PDF_DIR, case_document_keys, load_json
from eval.metrics.retrieval import (
    evidence_alternatives,
    normalize_evidence_text,
)


ALLOWED_CATEGORIES = {
    "fact",
    "summary",
    "unanswerable",
    "term",
    "comparison",
    "source_check",
}
ALLOWED_SCENARIOS = {"single_turn", "follow_up"}
MIN_DOCUMENTS = 3
MIN_CASES = 19
MIN_CASES_PER_DOCUMENT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 DocRAG 评估数据")
    parser.add_argument(
        "--require-gold-evidence",
        action="store_true",
        help="要求所有可回答案例都包含原文证据",
    )
    return parser.parse_args()


def load_all_cases() -> list[dict]:
    cases = []
    for filename in ("single_turn.json", "follow_up.json"):
        payload = load_json(DATASET_DIR / filename)
        cases.extend(payload.get("cases") or [])
    return cases


def validate_document(document: dict, errors: list[str]) -> dict[int, str]:
    document_key = document.get("document_key", "unknown")
    pdf_path = PDF_DIR / document.get("filename", "")
    if not pdf_path.exists():
        errors.append(f"{document_key}: 缺少 PDF {pdf_path.name}")
        return {}

    actual_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if actual_hash != document.get("sha256"):
        errors.append(f"{document_key}: PDF SHA-256 不匹配")

    with fitz.open(pdf_path) as pdf:
        if len(pdf) != document.get("page_count"):
            errors.append(f"{document_key}: PDF 页数不匹配")
        pages = {
            number: normalize_evidence_text(page.get_text("text"))
            for number, page in enumerate(pdf, start=1)
        }
    if not any(pages.values()):
        errors.append(f"{document_key}: PDF 没有可解析文本")
    return pages


def validate_evidence(
    case: dict,
    document_pages: dict[str, dict[int, str]],
    *,
    require_gold_evidence: bool,
    errors: list[str],
) -> bool:
    case_id = case["case_id"]
    answerable = case["answerable"]
    gold_pages = case.get("gold_pages") or []
    evidence = case.get("gold_evidence") or []
    if require_gold_evidence and answerable and not evidence:
        errors.append(f"{case_id}: 缺少 gold_evidence")
    if not answerable and evidence:
        errors.append(f"{case_id}: 无答案案例的 gold_evidence 应为空")

    evidence_ids = []
    for item in evidence:
        evidence_id = item.get("evidence_id")
        evidence_ids.append(evidence_id)
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            errors.append(f"{case_id}: evidence_id 不能为空")
        alternatives = item.get("alternatives")
        if alternatives is not None and (
            not isinstance(alternatives, list) or not alternatives
        ):
            errors.append(f"{case_id}: {evidence_id} alternatives 不能为空")
            continue
        if alternatives is not None and (
            item.get("page") is not None or item.get("text") is not None
        ):
            errors.append(
                f"{case_id}: {evidence_id} 不能同时使用直接证据和 alternatives"
            )

        for index, alternative in enumerate(
            evidence_alternatives(item),
            start=1,
        ):
            location = f"{evidence_id} alternative-{index}"
            case_scope = case_document_keys(case)
            document_key = alternative.get("document_key")
            if len(case_scope) > 1:
                if document_key not in case_scope:
                    errors.append(
                        f"{case_id}: {location} document_key 必须属于 document_keys"
                    )
                    continue
            else:
                document_key = document_key or case_scope[0]
            page = alternative.get("page")
            if page not in gold_pages:
                errors.append(
                    f"{case_id}: {location} page 必须属于 gold_pages"
                )
            text = alternative.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{case_id}: {location} text 不能为空")
            elif (
                normalize_evidence_text(text)
                not in document_pages.get(document_key, {}).get(page, "")
            ):
                errors.append(
                    f"{case_id}: {location} 不在 PDF 第 {page} 页"
                )
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append(f"{case_id}: evidence_id 必须唯一")
    return bool(evidence)


def validate_case(
    case: dict,
    documents: dict[str, dict],
    document_pages: dict[str, dict[int, str]],
    *,
    require_gold_evidence: bool,
    errors: list[str],
) -> bool:
    case_id = case.get("case_id", "unknown")
    required = {
        "case_id",
        "scenario",
        "category",
        "query",
        "answerable",
        "expected_task",
    }
    missing = sorted(required - set(case))
    if missing:
        errors.append(f"{case_id}: 缺少字段 {missing}")
        return False
    try:
        document_keys = case_document_keys(case)
    except ValueError as error:
        errors.append(f"{case_id}: {error}")
        return False
    missing_documents = [key for key in document_keys if key not in documents]
    if missing_documents:
        errors.append(f"{case_id}: 未定义文档 {missing_documents}")
        return False
    if case["scenario"] not in ALLOWED_SCENARIOS:
        errors.append(f"{case_id}: 非法 scenario")
    if case["category"] not in ALLOWED_CATEGORIES:
        errors.append(f"{case_id}: 非法 category")
    if not str(case["query"]).strip():
        errors.append(f"{case_id}: query 不能为空")
    if case["scenario"] == "follow_up" and not case.get("history"):
        errors.append(f"{case_id}: 追问案例必须包含 history")
    if case["scenario"] == "single_turn" and case.get("history"):
        errors.append(f"{case_id}: 单轮案例不应包含 history")
    if not case.get("answer_points"):
        errors.append(f"{case_id}: answer_points 不能为空")
    if not 1 <= case.get("n_results", 3) <= 10:
        errors.append(f"{case_id}: n_results 必须在 1 到 10 之间")

    gold_pages = case.get("gold_pages") or []
    if case["answerable"] and not gold_pages:
        errors.append(f"{case_id}: 可回答案例必须包含 gold_pages")
    if not case["answerable"] and gold_pages:
        errors.append(f"{case_id}: 无答案案例的 gold_pages 应为空")
    for page in gold_pages:
        if not isinstance(page, int) or page < 1:
            errors.append(f"{case_id}: gold page {page} 非法")
        elif not any(
            page <= documents[key]["page_count"] for key in document_keys
        ):
            errors.append(f"{case_id}: gold page {page} 超出范围")

    return validate_evidence(
        case,
        document_pages,
        require_gold_evidence=require_gold_evidence,
        errors=errors,
    )


def main() -> None:
    args = parse_args()
    documents_list = load_json(DATASET_DIR / "documents.json")
    cases = load_all_cases()
    errors = []
    documents = {
        item["document_key"]: item
        for item in documents_list
    }
    if len(documents) != len(documents_list):
        errors.append("document_key 必须唯一")
    if len(documents) < MIN_DOCUMENTS:
        errors.append(f"至少需要 {MIN_DOCUMENTS} 篇文档")
    if len(cases) < MIN_CASES:
        errors.append(f"至少需要 {MIN_CASES} 个案例")

    document_pages = {
        key: validate_document(document, errors)
        for key, document in documents.items()
    }
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("case_id 必须唯一")
    counts = Counter(
        key
        for case in cases
        for key in case_document_keys(case)
    )
    for key in documents:
        if counts[key] < MIN_CASES_PER_DOCUMENT:
            errors.append(f"{key}: 至少需要 {MIN_CASES_PER_DOCUMENT} 个案例")

    annotated = sum(
        validate_case(
            case,
            documents,
            document_pages,
            require_gold_evidence=args.require_gold_evidence,
            errors=errors,
        )
        and case["answerable"]
        for case in cases
    )
    if errors:
        print("评估数据校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    answerable = sum(case["answerable"] for case in cases)
    print(f"评估数据校验通过：{len(documents)} 篇 PDF，{len(cases)} 个案例。")
    print("场景统计：", dict(Counter(case["scenario"] for case in cases)))
    print("类别统计：", dict(Counter(case["category"] for case in cases)))
    print(f"gold_evidence 覆盖：{annotated}/{answerable}")


if __name__ == "__main__":
    main()
