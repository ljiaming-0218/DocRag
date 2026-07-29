import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "rag_dataset" / "results"
OUTPUT_PATH = RESULTS_DIR / "metrics_summary.json"
QUESTIONS_PATH = BASE_DIR / "rag_dataset" / "questions.json"

RESULT_FILES = [
    RESULTS_DIR / "normal_results.json",
    RESULTS_DIR / "follow_up_results.json",
]

def apply_current_gold_pages(
    results: list[dict],
) -> list[dict]:
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    questions_by_id = {
        item["question_id"]: item
        for item in questions
    }

    normalized_results = []

    for result in results:
        question_id = result.get("question_id")
        question = questions_by_id.get(question_id)

        normalized_result = dict(result)

        if question is not None:
            normalized_result["gold_source_pages"] = (
                question.get("source_pages") or []
            )

        normalized_results.append(normalized_result)

    return normalized_results

def load_results() -> list[dict]:
    results = []

    for path in RESULT_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        results.extend(data.get("results", []))

    return results


def calculate_hit_rate(
    gold_pages: list[int],
    retrieved_pages: list[int],
    k: int = 3,
) -> float:
    top_k_pages = retrieved_pages[:k]
    return 1.0 if set(gold_pages) & set(top_k_pages) else 0.0


def calculate_recall(
    gold_pages: list[int],
    retrieved_pages: list[int],
    k: int = 3,
) -> float:
    if not gold_pages:
        return 0.0

    matched_pages = set(gold_pages) & set(retrieved_pages[:k])
    return len(matched_pages) / len(set(gold_pages))


def calculate_mrr(
    gold_pages: list[int],
    retrieved_pages: list[int],
    k: int = 3,
) -> float:
    for rank, page_number in enumerate(retrieved_pages[:k], start=1):
        if page_number in gold_pages:
            return 1.0 / rank

    return 0.0


def average(values: list[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)

def calculate_case_metrics(
    item: dict,
    k: int = 3,
) -> dict:
    question_id = item.get("question_id")
    status = item.get("status")
    gold_pages = item.get("gold_source_pages") or []

    result = {
        "question_id": question_id,
        "document_key": item.get("document_key"),
        "question_type": item.get("question_type"),
        "status": status,
        "retrieval_evaluable": False,
        "gold_source_pages": gold_pages,
        "returned_source_pages": [],
        f"hit_rate_at_{k}": None,
        f"recall_at_{k}": None,
        f"mrr_at_{k}": None,
    }

    if status != "success":
        result["error"] = item.get("error")
        return result

    actual = item.get("actual") or {}
    retrieved_pages = actual.get("returned_source_pages", [])
    result["returned_source_pages"] = retrieved_pages

    # 无答案题没有 gold，不计算检索指标。
    if not gold_pages:
        return result

    result["retrieval_evaluable"] = True
    result[f"hit_rate_at_{k}"] = calculate_hit_rate(
        gold_pages,
        retrieved_pages,
        k,
    )
    result[f"recall_at_{k}"] = calculate_recall(
        gold_pages,
        retrieved_pages,
        k,
    )
    result[f"mrr_at_{k}"] = calculate_mrr(
        gold_pages,
        retrieved_pages,
        k,
    )

    return result

def summarize_by_field(
    case_metrics: list[dict],
    field_name: str,
    k: int = 3,
) -> dict:
    groups = defaultdict(list)

    for item in case_metrics:
        group_name = item.get(field_name) or "unknown"
        groups[group_name].append(item)

    grouped_summary = {}

    for group_name, group_items in groups.items():
        successful = [
            item for item in group_items
            if item.get("status") == "success"
        ]

        evaluable = [
            item for item in group_items
            if item.get("retrieval_evaluable") is True
        ]

        hit_rates = [
            item[f"hit_rate_at_{k}"]
            for item in evaluable
        ]

        recalls = [
            item[f"recall_at_{k}"]
            for item in evaluable
        ]

        mrr_scores = [
            item[f"mrr_at_{k}"]
            for item in evaluable
        ]

        grouped_summary[group_name] = {
            "total_cases": len(group_items),
            "successful_cases": len(successful),
            "retrieval_evaluable_cases": len(evaluable),
            "execution_success_rate": (
                len(successful) / len(group_items)
                if group_items else 0.0
            ),
            f"hit_rate_at_{k}": average(hit_rates),
            f"recall_at_{k}": average(recalls),
            f"mrr_at_{k}": average(mrr_scores),
        }

    return grouped_summary


def summarize_results(results: list[dict], k: int = 3) -> dict:
    successful = [
        item for item in results
        if item.get("status") == "success"
    ]

    # 无答案题没有 gold 页码，不参与检索指标。
    evaluable = [
        item for item in successful
        if item.get("gold_source_pages")
    ]

    hit_rates = []
    recalls = []
    mrr_scores = []

    for item in evaluable:
        gold_pages = item["gold_source_pages"]
        actual = item.get("actual") or {}
        retrieved_pages = actual.get("returned_source_pages", [])

        hit_rates.append(
            calculate_hit_rate(gold_pages, retrieved_pages, k)
        )
        recalls.append(
            calculate_recall(gold_pages, retrieved_pages, k)
        )
        mrr_scores.append(
            calculate_mrr(gold_pages, retrieved_pages, k)
        )

    case_metrics = [
        calculate_case_metrics(item, k)
        for item in results
    ]
    by_document = summarize_by_field(
    case_metrics,
        "document_key",
        k,
    )

    by_question_type = summarize_by_field(
        case_metrics,
        "question_type",
        k,
    )
    return {
        "total_cases": len(results),
        "successful_cases": len(successful),
        "failed_cases": len(results) - len(successful),
        "retrieval_evaluable_cases": len(evaluable),
        "execution_success_rate": (
            len(successful) / len(results) if results else 0.0
        ),
        "case_metrics": case_metrics,
        "by_document": by_document,
        "by_question_type": by_question_type,
        f"hit_rate_at_{k}": average(hit_rates),
        f"recall_at_{k}": average(recalls),
        f"mrr_at_{k}": average(mrr_scores),
    }


def validate_result_coverage(results: list[dict]) -> None:
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    expected_ids = {
        question["question_id"]
        for question in questions
    }

    actual_ids = [
        item.get("question_id")
        for item in results
    ]

    id_counts = Counter(actual_ids)

    duplicate_ids = {
        question_id
        for question_id, count in id_counts.items()
        if count > 1
    }

    actual_id_set = set(actual_ids)
    missing_ids = expected_ids - actual_id_set
    unexpected_ids = actual_id_set - expected_ids

    if missing_ids or duplicate_ids or unexpected_ids:
        raise ValueError(
            "评估结果不完整："
            f"missing={sorted(missing_ids)}, "
            f"duplicate={sorted(duplicate_ids)}, "
            f"unexpected={sorted(unexpected_ids)}"
        )

def main() -> None:
    results = load_results()
    results = apply_current_gold_pages(results)
    validate_result_coverage(results)
    summary = summarize_results(results, k=3)

    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()