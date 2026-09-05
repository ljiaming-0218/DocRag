import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "rag_dataset"
DEFAULT_RESULTS_PATH = DATASET_DIR / "results" / "normal_results.json"
DEFAULT_OUTPUT_PATH = (
    DATASET_DIR / "results" / "evidence_threshold_calibration.json"
)
QUESTIONS_PATH = DATASET_DIR / "questions.json"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_question_map(questions: list[dict]) -> dict[str, dict]:
    return {
        question["question_id"]: question
        for question in questions
    }


def get_successful_cases(
    result_data: dict,
    questions_by_id: dict[str, dict],
) -> list[dict]:
    cases = []
    for result in result_data.get("results", []):
        if result.get("status") != "success":
            continue

        question = questions_by_id.get(result.get("question_id"))
        if question is None:
            continue

        sources = (result.get("actual") or {}).get("sources", [])
        missing_scores = [
            source
            for source in sources
            if not isinstance(source.get("rerank_score"), (int, float))
        ]
        if missing_scores:
            raise ValueError(
                f"{result['question_id']} 存在缺少 rerank_score 的 source"
            )

        cases.append({
            "question_id": result["question_id"],
            "question_type": question.get("question_type"),
            "expected_answerable": question.get(
                "expected_answerable",
                bool(question.get("source_pages")),
            ),
            "gold_source_pages": question.get("source_pages") or [],
            "sources": sources,
        })
    return cases


def build_thresholds(cases: list[dict]) -> list[float]:
    scores = {
        float(source["rerank_score"])
        for case in cases
        for source in case["sources"]
    }
    return sorted(scores)


def retained_sources(
    case: dict,
    threshold: float | None,
) -> list[dict]:
    if threshold is None:
        return case["sources"]
    return [
        source
        for source in case["sources"]
        if source["rerank_score"] >= threshold
    ]


def source_page(source: dict) -> int | None:
    metadata = source.get("元数据") or {}
    return metadata.get("page_number")


def safe_divide(numerator: int | float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_threshold(
    cases: list[dict],
    threshold: float | None,
    top_k: int = 3,
) -> dict:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    answerable_recalls = []
    retained_source_counts = []

    for case in cases:
        retained = retained_sources(case, threshold)[:top_k]
        retained_source_counts.append(len(retained))
        retained_pages = {
            page
            for page in (source_page(source) for source in retained)
            if page is not None
        }

        if case["expected_answerable"]:
            gold_pages = set(case["gold_source_pages"])
            matched_pages = gold_pages & retained_pages
            if matched_pages:
                true_positive += 1
            else:
                false_negative += 1
            if gold_pages:
                answerable_recalls.append(
                    len(matched_pages) / len(gold_pages)
                )
        elif retained:
            false_positive += 1
        else:
            true_negative += 1

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "mean_gold_page_recall": (
            sum(answerable_recalls) / len(answerable_recalls)
            if answerable_recalls
            else 0.0
        ),
        "mean_retained_sources": (
            sum(retained_source_counts) / len(retained_source_counts)
            if retained_source_counts
            else 0.0
        ),
    }


def build_case_decisions(
    cases: list[dict],
    threshold: float,
    top_k: int = 3,
) -> list[dict]:
    decisions = []
    for case in cases:
        retained = retained_sources(case, threshold)[:top_k]
        retained_pages = [
            page
            for page in (source_page(source) for source in retained)
            if page is not None
        ]
        gold_pages = set(case["gold_source_pages"])
        matched_gold_pages = sorted(gold_pages & set(retained_pages))
        expected_answerable = case["expected_answerable"]

        decisions.append({
            "question_id": case["question_id"],
            "question_type": case["question_type"],
            "expected_answerable": expected_answerable,
            "system_would_answer": bool(retained),
            "decision_correct": (
                bool(matched_gold_pages)
                if expected_answerable
                else not retained
            ),
            "gold_source_pages": case["gold_source_pages"],
            "retained_source_pages": retained_pages,
            "matched_gold_pages": matched_gold_pages,
            "retained_scores": [
                source["rerank_score"] for source in retained
            ],
        })
    return decisions


def select_recommended_threshold(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            row["balanced_accuracy"],
            row["recall"],
            -row["threshold"],
        ),
    )


def calibrate(
    result_data: dict,
    questions: list[dict],
    top_k: int = 3,
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    cases = get_successful_cases(
        result_data,
        build_question_map(questions),
    )
    thresholds = build_thresholds(cases)
    rows = [
        evaluate_threshold(cases, threshold, top_k)
        for threshold in thresholds
    ]
    answerable_count = sum(
        case["expected_answerable"] for case in cases
    )
    unanswerable_count = sum(
        not case["expected_answerable"] for case in cases
    )
    calibration_ready = answerable_count > 0 and unanswerable_count > 0
    recommended = (
        select_recommended_threshold(rows)
        if calibration_ready
        else None
    )
    recommended_threshold = (
        recommended["threshold"] if recommended else None
    )

    return {
        "source_results": result_data.get("run_id"),
        "top_k": top_k,
        "successful_cases": len(cases),
        "answerable_cases": answerable_count,
        "unanswerable_cases": unanswerable_count,
        "calibration_ready": calibration_ready,
        "baseline_metrics": evaluate_threshold(cases, None, top_k),
        "recommended_threshold": recommended_threshold,
        "recommended_metrics": recommended,
        "recommended_case_decisions": (
            build_case_decisions(
                cases,
                recommended_threshold,
                top_k,
            )
            if recommended_threshold is not None
            else []
        ),
        "threshold_results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线校准 CrossEncoder 证据阈值。"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = calibrate(
        load_json(args.results),
        load_json(args.questions),
        args.top_k,
    )
    report["source_results_file"] = str(args.results.resolve())
    save_json(report, args.output)

    print(f"成功样本数: {report['successful_cases']}")
    print(f"可回答题: {report['answerable_cases']}")
    print(f"无答案题: {report['unanswerable_cases']}")
    print(f"推荐阈值: {report['recommended_threshold']}")
    print(f"结果文件: {args.output}")


if __name__ == "__main__":
    main()
