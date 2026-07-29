import json
import logging
from pathlib import Path

from services.llm_service import generate_answer


logger = logging.getLogger("uvicorn.error")


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


def build_summary_subqueries(
    query: str,
    seed_sources: list[dict],
) -> list[str]:
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")

    seed_context = build_seed_context(seed_sources)
    if not seed_context:
        return [query]

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
            "summary_query_failed error_type=%s",
            type(exc).__name__,
        )
        return [query]

    if not isinstance(parsed_result, list):
        return [query]

    subqueries = []
    for item in parsed_result:
        if not isinstance(item, str):
            continue

        subquery = item.strip()
        if (
            subquery
            and subquery not in subqueries
        ):
            subqueries.append(subquery)

        if len(subqueries) >= 4:
            break

    return subqueries or [query]
