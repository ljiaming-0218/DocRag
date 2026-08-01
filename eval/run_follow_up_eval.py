import logging
import sys
import time

import requests

from run_eval import (
    BASE_URL,
    DATASET_DIR,
    REQUEST_INTERVAL_SECONDS,
    RESULTS_DIR,
    check_api,
    create_eval_user,
    index_documents,
    load_json,
    run_follow_up_case,
    save_json,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    documents = load_json(DATASET_DIR / "documents.json")
    questions = load_json(DATASET_DIR / "questions.json")
    follow_up_questions = [
        question
        for question in questions
        if question["question_type"] == "follow_up"
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with requests.Session() as session:
        print("1. 检查后端服务")
        check_api(session, BASE_URL)
        print("后端服务正常")

        print("2. 创建或获取评估用户")
        user_id = create_eval_user(session, BASE_URL)
        print(f"user_id: {user_id}")

        print("3. 上传并索引评估文档")
        document_ids = index_documents(
            session,
            BASE_URL,
            user_id,
            documents,
        )

        batch_result = {
            "user_id": user_id,
            "document_ids": document_ids,
            "results": [],
        }
        output_path = RESULTS_DIR / "follow_up_results.json"

        print(f"4. 开始执行 {len(follow_up_questions)} 道追问题")

        for index, question in enumerate(follow_up_questions, start=1):
            question_id = question["question_id"]
            print(
                f"[{index}/{len(follow_up_questions)}] "
                f"执行追问题 {question_id}"
            )

            try:
                case_result = run_follow_up_case(
                    session,
                    BASE_URL,
                    user_id,
                    document_ids,
                    question,
                )
                print("实际改写：")
                print(case_result["actual"]["rewritten_query"])
                print("系统回答：")
                print(case_result["actual"]["answer"])
            except Exception as exc:
                case_result = {
                    "question_id": question_id,
                    "document_key": question["document_key"],
                    "question_type": question["question_type"],
                    "question": question["question"],
                    "status": "error",
                    "error": str(exc),
                    "actual": None,
                }
                print(f"{question_id} 失败：{exc}")
            finally:
                time.sleep(REQUEST_INTERVAL_SECONDS)

            batch_result["results"].append(case_result)
            save_json(batch_result, output_path)

        print(f"追问评估完成：{output_path}")


if __name__ == "__main__":
    main()
