import json
import logging
import requests
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "rag_dataset"
PDF_DIR = DATASET_DIR / "pdfs"
BASE_URL = "http://127.0.0.1:8000"
RESULTS_DIR = DATASET_DIR / "results"
REQUEST_INTERVAL_SECONDS = 30
logger = logging.getLogger(__name__)

def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

def request_json(session, method, url, **kwargs) -> dict:
    started_at = time.perf_counter()
    kwargs.setdefault("timeout", 180)

    try:
        response = session.request(method, url, **kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started_at
        logger.error(
            "http_request_failed method=%s url=%s "
            "elapsed_seconds=%.2f error_type=%s",
            method,
            url,
            elapsed,
            type(exc).__name__,
        )
        raise

    elapsed = time.perf_counter() - started_at
    logger.info(
        "http_request_succeeded method=%s url=%s "
        "status_code=%s elapsed_seconds=%.2f",
        method,
        url,
        response.status_code,
        elapsed,
    )

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        logger.error(
            "http_response_invalid_json method=%s url=%s",
            method,
            url,
        )
        raise ValueError(
            f"无法解析 JSON 响应: {response.text}"
        ) from exc

    
def check_api(session, base_url) -> None:
    try:   
        response = session.get(f"{base_url}/health")
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"无法连接到 API: {base_url}") from exc

def create_eval_user(session, base_url) -> str:
    user_data ={"username": "docrag-eval-v1", "default_user_type": "general"}
    response = session.post(f"{base_url}/users", json=user_data)
    response.raise_for_status()

    return response.json()["user_id"]

def index_documents(session, base_url, user_id, documents) -> dict:
    index_results = {}

    for doc in documents:
        pdf_path = PDF_DIR / doc["filename"]

        params = {
            "user_id": user_id,
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        with pdf_path.open("rb") as pdf_file:
            files = {
                "file": (
                    doc["filename"],
                    pdf_file,
                    "application/pdf",
                )
            }

            result = request_json(
                session,
                "POST",
                f"{base_url}/pdf/index",
                params=params,
                files=files,
            )

        index_results[doc["document_key"]] = result["document_id"]

    return index_results

def create_eval_conversation(
    session,
    base_url,
    user_id,
    document_id,
    question_id,
) -> str:
    request_body = {
        "user_id": user_id,
        "document_id": document_id,
        "title": f"Eval {question_id}",
    }

    result = request_json(
        session,
        "POST",
        f"{base_url}/conversations",
        json=request_body,
    )

    return result["conversation_id"]

def ask_question(
    session,
    base_url,
    user_id,
    conversation_id,
    question,
) -> dict:
    request_body = {
        "user_id": user_id,
        "query": question,
        "history_limit": 6,
        "n_results": 3,
        "user_type": "general",
    }

    return request_json(
        session,
        "POST",
        f"{base_url}/conversations/{conversation_id}/ask",
        json=request_body,
    )

def extract_source_pages(sources: list[dict]) -> list[int]:
    page_numbers = []

    for source in sources:
        metadata = source.get("元数据", {})
        page_number = metadata.get("page_number")

        if (
            page_number is not None
            and page_number not in page_numbers
        ):
            page_numbers.append(page_number)

    return page_numbers


def run_normal_case(
    session,
    base_url,
    user_id,
    document_ids,
    question,
) -> dict:
    document_key = question["document_key"]
    document_id = document_ids[document_key]

    conversation_id = create_eval_conversation(
        session,
        base_url,
        user_id,
        document_id,
        question["question_id"],
    )

    response = ask_question(
        session,
        base_url,
        user_id,
        conversation_id,
        question["question"],
    )

    returned_pages = extract_source_pages(
        response.get("sources", [])
    )

    return {
        "question_id": question["question_id"],
        "document_key": document_key,
        "document_id": document_id,
        "conversation_id": conversation_id,
        "question_type": question["question_type"],
        "question": question["question"],
        "expected_rewritten_query": question["expected_rewritten_query"],
        "gold_answer": question["gold_answer"],
        "gold_answer_points": question["gold_answer_points"],
        "gold_source_pages": question["source_pages"],
        "expected_answerable": question["expected_answerable"],
        "status": "success",
        "error": None,
        "actual": {
            "rewritten_query": response.get("rewritten_query"),
            "retrieval_queries": response.get(
                "retrieval_queries",
                [],
            ),
            "answer": response.get("answer"),
            "sources": response.get("sources", []),
            "returned_source_pages": returned_pages,
            "task_type": response.get("task_type"),
        },
    }

def run_follow_up_case(
    session,
    base_url,
    user_id,
    document_ids,
    question,
) -> dict:
    document_key = question["document_key"]
    document_id = document_ids[document_key]

    conversation_id = create_eval_conversation(
        session,
        base_url,
        user_id,
        document_id,
        question["question_id"],
    )

    generated_history = []

    for history_message in question["history"]:
        if history_message["role"] != "user":
            continue

        history_response = ask_question(
            session,
            base_url,
            user_id,
            conversation_id,
            history_message["content"],
        )

        generated_history.append({
            "question": history_message["content"],
            "rewritten_query": history_response.get("rewritten_query"),
            "retrieval_queries": history_response.get(
                "retrieval_queries",
                [],
            ),
            "answer": history_response.get("answer"),
            "sources": history_response.get("sources", []),
            "returned_source_pages": extract_source_pages(
                history_response.get("sources", [])
            ),
        })
        time.sleep(REQUEST_INTERVAL_SECONDS)

    response = ask_question(
        session,
        base_url,
        user_id,
        conversation_id,
        question["question"],
    )

    returned_pages = extract_source_pages(
        response.get("sources", [])
    )

    return {
        "question_id": question["question_id"],
        "document_key": document_key,
        "document_id": document_id,
        "conversation_id": conversation_id,
        "question_type": question["question_type"],
        "question": question["question"],
        "expected_history": question["history"],
        "generated_history": generated_history,
        "expected_rewritten_query": question["expected_rewritten_query"],
        "gold_answer": question["gold_answer"],
        "gold_answer_points": question["gold_answer_points"],
        "gold_source_pages": question["source_pages"],
        "status": "success",
        "error": None,
        "actual": {
            "rewritten_query": response.get("rewritten_query"),
            "retrieval_queries": response.get(
                "retrieval_queries",
                [],
            ),
            "answer": response.get("answer"),
            "sources": response.get("sources", []),
            "returned_source_pages": returned_pages,
            "task_type": response.get("task_type"),
        },
    }

def main() -> None:
    documents_path = DATASET_DIR / "documents.json"
    questions_path = DATASET_DIR / "questions.json"

    documents = load_json(documents_path)
    questions = load_json(questions_path)



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

        

        print("文档索引完成：")
        for document_key, document_id in document_ids.items():
            print(f"{document_key}: {document_id}")

        normal_questions = [
            question
            for question in questions
            if question["question_type"] != "follow_up"
        ]

        print(f"4. 开始执行 {len(normal_questions)} 道非追问题")

        batch_result = {
            "user_id": user_id,
            "document_ids": document_ids,
            "config": {
                "base_url": BASE_URL,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "history_limit": 6,
                "n_results": 3,
                "user_type": "general",
            },
            "results": [],
        }

        output_path = RESULTS_DIR / "normal_results.json"

        for index, question in enumerate(normal_questions, start=1):
            question_id = question["question_id"]
            print(
                f"[{index}/{len(normal_questions)}] "
                f"正在执行 {question_id}"
            )

            try:
                case_result = run_normal_case(
                    session,
                    BASE_URL,
                    user_id,
                    document_ids,
                    question,
                )

                returned_pages = case_result["actual"][
                    "returned_source_pages"
                ]

                print(
                    f"{question_id} 完成，"
                    f"返回页码：{returned_pages}"
                )

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

 

if __name__ == "__main__":
    main()
