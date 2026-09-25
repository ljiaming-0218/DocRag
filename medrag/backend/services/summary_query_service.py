import json
import logging
import re
from pathlib import Path
from time import perf_counter

from config import SUMMARY_MAX_SUBQUERIES
from services.llm_service import generate_answer


logger = logging.getLogger("uvicorn.error")
TECHNICAL_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}(?:[-_.][A-Z0-9]+)*|"
    r"[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|"
    r"\d+(?:\.\d+)?(?:[BMK]|%|@[0-9]+)?)\b"
)


def load_summary_query_prompt_template() -> str:
    prompt_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "summary_query_prompt.txt"
    )
    return prompt_path.read_text(encoding="utf-8")


def build_seed_context(
    sources: list[dict],
    max_sources: int = 5,
) -> str:
    context_parts = []

    for source in sources[:max_sources]:
        metadata = source.get("元数据", {})
        page_number = metadata.get("page_number", "未知")
        text = source.get("文本块", "").strip()

        if text:
            context_parts.append(
                f"[第{page_number}页]\n{text}"
            )

    return "\n\n".join(context_parts)


def build_default_summary_queries(query: str) -> list[str]:
    """Provide stable coverage dimensions before optional LLM refinement."""
    if re.search(r"[\u4e00-\u9fff]", query):
        asks_for_task_types = "任务类型" in query
        asks_for_findings = any(
            term in query
            for term in (
                "实验结论", "主要结论", "研究结论", "结论是",
                "事实性", "知识更新", "幻觉",
            )
        )
        if asks_for_task_types and asks_for_findings:
            if re.search(r"\bRAG\b", query, flags=re.IGNORECASE):
                return [
                    "Which open-domain and abstractive question answering tasks are evaluated?",
                    "Which open-domain question generation tasks are evaluated?",
                    "Which fact verification tasks are evaluated?",
                    "What state-of-the-art results are reported for open-domain QA?",
                    "How do factuality and hallucination compare with the baseline?",
                    "Can external or non-parametric memory update world knowledge?",
                ]
            return [
                "Which question answering tasks and datasets are evaluated?",
                "Which generation tasks and datasets are evaluated?",
                "Which classification or fact verification tasks are evaluated?",
                "Which results establish state-of-the-art performance?",
                "What findings concern factuality and hallucination?",
                "What findings concern updating external knowledge?",
            ]

        queries = [
            "What research problem does the paper address?",
            "What is the paper's core method, model architecture, and key mechanism?",
            "What tasks, datasets, and benchmarks are evaluated?",
            "What are the main experimental results and baseline comparisons?",
        ]
        asks_for_reasoning_tasks = any(
            term in query
            for term in (
                "算术", "数学", "常识", "符号", "推理任务", "推理类型",
            )
        )
        if asks_for_reasoning_tasks:
            queries = [
                "What arithmetic reasoning tasks and datasets are evaluated?",
                "What commonsense reasoning tasks and datasets are evaluated?",
                "Which symbolic reasoning benchmarks and task examples are evaluated?",
                "What tasks, datasets, and benchmarks are evaluated?",
                "What are the main experimental results and baseline comparisons?",
            ]
        if any(
            term in query
            for term in (
                "效率", "参数", "显存", "推理效率", "推理延迟", "推理速度",
            )
        ):
            queries.append(
                "What are the trainable components, parameter efficiency, "
                "GPU memory requirements, weight merging, and inference latency?"
            )
        if asks_for_task_types:
            queries.extend([
                "Which question answering tasks and datasets are evaluated?",
                "Which generation tasks and datasets are evaluated?",
                "Which classification or fact verification tasks are evaluated?",
            ])
        if asks_for_findings:
            queries.extend([
                "Which results establish state-of-the-art performance?",
                "What findings concern factuality and hallucination?",
                "What findings concern updating external knowledge?",
            ])
        return queries
    return [
        "What problem or task does the paper study?",
        "What core method does the paper propose or use?",
        "Which tasks, datasets, or experimental settings are evaluated?",
        "What are the main experimental results and comparisons with baselines?",
        "What findings concern factuality, limitations, conclusions, or knowledge updates?",
    ]


def _normalize_query(text: str) -> str:
    return " ".join(text.lower().split()).rstrip("?？.!。")


def _extract_technical_entities(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in TECHNICAL_ENTITY_PATTERN.finditer(text)
    }


def is_valid_summary_subquery(
    subquery: str,
    query: str,
    seed_context: str,
    existing_queries: list[str],
) -> bool:
    normalized = _normalize_query(subquery)
    if len(normalized) < 8:
        return False
    if normalized in {
        _normalize_query(item)
        for item in existing_queries
    }:
        return False

    allowed_entities = _extract_technical_entities(
        f"{query}\n{seed_context}"
    )
    introduced_entities = (
        _extract_technical_entities(subquery) - allowed_entities
    )
    return not introduced_entities


def build_summary_subqueries(
    query: str,
    seed_sources: list[dict],
) -> list[str]:
    started_at = perf_counter()
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")

    default_queries = build_default_summary_queries(query)
    selected_queries = default_queries[:SUMMARY_MAX_SUBQUERIES]
    seed_context = build_seed_context(seed_sources)
    if not seed_context or len(selected_queries) >= SUMMARY_MAX_SUBQUERIES:
        logger.info(
            "summary_queries_completed template_count=%s llm_count=0 "
            "summary_query_generation_ms=%.2f",
            len(selected_queries),
            (perf_counter() - started_at) * 1000,
        )
        return selected_queries

    prompt = load_summary_query_prompt_template().format(
        query=query,
        seed_context=seed_context,
    )

    try:
        raw_result = generate_answer(
            prompt,
            operation="summary_query",
        )
        parsed_result = json.loads(raw_result)
    except Exception as exc:
        logger.warning(
            "summary_query_failed error_type=%s "
            "summary_query_generation_ms=%.2f",
            type(exc).__name__,
            (perf_counter() - started_at) * 1000,
        )
        return selected_queries

    if not isinstance(parsed_result, list):
        logger.warning(
            "summary_query_invalid_response response_type=%s "
            "summary_query_generation_ms=%.2f",
            type(parsed_result).__name__,
            (perf_counter() - started_at) * 1000,
        )
        return selected_queries

    for item in parsed_result:
        if not isinstance(item, str):
            continue

        subquery = item.strip()
        if is_valid_summary_subquery(
            subquery,
            query,
            seed_context,
            selected_queries,
        ):
            selected_queries.append(subquery)

        if len(selected_queries) >= SUMMARY_MAX_SUBQUERIES:
            break

    logger.info(
        "summary_queries_completed template_count=%s total_count=%s "
        "summary_query_generation_ms=%.2f",
        len(default_queries),
        len(selected_queries),
        (perf_counter() - started_at) * 1000,
    )
    return selected_queries
