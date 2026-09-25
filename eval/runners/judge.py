"""Run LLM-as-a-Judge against an existing generation evaluation run."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Callable

from eval.core import (
    BACKEND_ROOT,
    RUNS_DIR,
    build_run_id,
    load_json,
    save_json,
    utc_now_iso,
)
from eval.judges.llm_judge import (
    JUDGE_CRITERIA,
    JUDGE_VERSION,
    build_unscored_scores,
    judge_case,
)
from eval.metrics.generation import (
    apply_no_answer_judge_policy,
    summarize_generation_results,
)


logger = logging.getLogger(__name__)


def load_judge_dependencies() -> tuple[Callable[..., str], dict]:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from config import ANSWER_LLM_MODEL, ANSWER_LLM_PROVIDER
    from services.llm_service import generate_answer

    return generate_answer, {
        "provider": ANSWER_LLM_PROVIDER,
        "model": ANSWER_LLM_MODEL,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LLM 评价已有生成结果，不重新执行 RAG。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="generation 运行生成的 results.json",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="只评价指定案例，可重复传入",
    )
    parser.add_argument("--request-interval", type=float, default=10.0)
    parser.add_argument("--max-source-chars", type=int, default=1500)
    return parser.parse_args()


def validate_judge_config(args: argparse.Namespace) -> None:
    if args.request_interval < 0:
        raise ValueError("request_interval 不能小于 0")
    if args.max_source_chars <= 0:
        raise ValueError("max_source_chars 必须大于 0")


def select_cases(results: list[dict], case_ids: list[str] | None) -> list[dict]:
    if not case_ids:
        return results
    requested = set(case_ids)
    selected = [case for case in results if case.get("case_id") in requested]
    found = {case.get("case_id") for case in selected}
    missing = requested - found
    if missing:
        raise ValueError("找不到评估案例: " + ", ".join(sorted(missing)))
    return selected


def main() -> None:
    args = parse_args()
    validate_judge_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    source_path = args.input.resolve()
    source_data = load_json(source_path)
    source_results = source_data.get("results")
    if not isinstance(source_results, list):
        raise ValueError("输入文件缺少 results 列表")
    selected = select_cases(source_results, args.case_ids)
    generate, judge_model = load_judge_dependencies()
    run_id = build_run_id("judge")
    run_dir = RUNS_DIR / "judge" / run_id
    judged_data = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "source_generation_run_id": source_data.get("run_id"),
        "source_results_path": str(source_path),
        "dataset_fingerprint": source_data.get("dataset_fingerprint"),
        "source_generation_snapshot": {
            "config": source_data.get("config"),
            "runtime_snapshot": source_data.get("runtime_snapshot"),
            "evaluation_contract": source_data.get("evaluation_contract"),
        },
        "config": {
            "case_ids": args.case_ids,
            "request_interval": args.request_interval,
            "max_source_chars": args.max_source_chars,
            "judge_model": judge_model,
            "judge_version": JUDGE_VERSION,
            "score_scale": [0.0, 1.0],
            "unscored_value": None,
            "criteria": JUDGE_CRITERIA,
            "applicability_policy": "correct_refusal_without_sources_v1",
        },
        "results": [],
    }

    print(f"评价 {len(selected)} 道已有生成结果")
    for index, source_case in enumerate(selected, start=1):
        case = copy.deepcopy(source_case)
        print(f"[{index}/{len(selected)}] {case.get('case_id')}")
        try:
            case["llm_judge"] = apply_no_answer_judge_policy(
                case,
                judge_case(
                    case,
                    generate,
                    max_source_chars=args.max_source_chars,
                ),
            )
        except Exception as error:
            logger.exception(
                "judge_case_failed case_id=%s",
                case.get("case_id"),
            )
            case["llm_judge"] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "scores": build_unscored_scores("judge_failed"),
            }
        judged_data["results"].append(case)
        save_json(judged_data, run_dir / "results.json")
        if index < len(selected) and args.request_interval > 0:
            time.sleep(args.request_interval)

    save_json(
        summarize_generation_results(judged_data),
        run_dir / "summary.json",
    )
    save_json(judged_data["config"], run_dir / "config.json")
    print(f"LLM Judge 完成: {run_dir}")


if __name__ == "__main__":
    main()
