import json
import logging
from pathlib import Path

from services.llm_service import generate_answer


logger = logging.getLogger(__name__)
SUPPORTED_TASK_TYPES = {
    "qa",
    "summary",
    "report",
    "term",
    "source_check",
    "comparison",
}
SUPPORTED_TASK_SCOPES = {"focused", "knowledge_base_overview"}


def load_router_prompt_template() -> str:
    prompt_path = (
        Path(__file__).parent.parent / "prompts" / "task_router_prompt.txt"
    )
    return prompt_path.read_text(encoding="utf-8")


def _route_scope_by_rules(query: str, task_type: str) -> str:
    """Infer only the summary retrieval scope as a fallback."""
    if task_type != "summary":
        return "focused"

    normalized_query = " ".join(query.strip().lower().split())
    overview_phrases = (
        "这个库里面的文档",
        "这个知识库里的文档",
        "知识库中的文档",
        "库里的论文",
        "what do the documents",
        "documents in the current literature",
        "overview of this knowledge base",
    )
    if any(phrase in normalized_query for phrase in overview_phrases):
        return "knowledge_base_overview"
    knowledge_base_terms = (
        "知识库", "库里", "库中", "库里面", "knowledge base",
    )
    document_terms = (
        "文档", "论文", "资料", "documents", "papers",
    )
    overview_terms = (
        "分别", "逐篇", "每篇", "都讲", "各自", "所有", "全部",
        "each", "all", "overview",
    )
    if (
        any(term in normalized_query for term in knowledge_base_terms)
        and any(term in normalized_query for term in document_terms)
        and any(term in normalized_query for term in overview_terms)
    ):
        return "knowledge_base_overview"
    return "focused"


def is_collective_document_question(query: str) -> bool:
    normalized_query = query.strip().lower()
    return (
        any(term in normalized_query for term in ("分别", "各自", "都"))
        and any(term in normalized_query for term in ("模型", "方法", "任务"))
        and any(term in normalized_query for term in ("哪些", "什么"))
        and not any(
            term in normalized_query
            for term in ("比较", "对比", "区别", "差异")
        )
    )


def route_task_by_rules(query: str) -> dict[str, str]:
    """Provide a deterministic fallback when the LLM router is unavailable."""
    normalized_query = query.strip().lower()

    if is_collective_document_question(query):
        return {"task_type": "comparison", "scope": "focused"}

    if any(keyword in normalized_query for keyword in ["阅读报告", "分析这篇文献"]):
        task_type = "report"
    elif any(
        keyword in normalized_query
        for keyword in [
            "依据", "出处", "引用", "来源", "证据", "citation", "evidence",
        ]
    ):
        task_type = "source_check"
    elif any(
        keyword in normalized_query
        for keyword in [
            "术语", "关键词", "关键概念", "专业概念", "概念解释",
            "名词解释", "keywords", "key terms", "terminology",
        ]
    ):
        task_type = "term"
    elif any(
        keyword in normalized_query
        for keyword in [
            "比较", "对比", "区别", "差异", "compare", "comparison",
            "difference", "differences", " versus ", " vs ",
        ]
    ):
        task_type = "comparison"
    elif any(
        keyword in normalized_query
        for keyword in [
            "总结", "摘要", "概括", "归纳", "讲了什么", "讲了些什么",
            "都讲了些什么", "主要讲什么", "主要内容", "summarize",
            "summary", "overview", "what do the documents",
        ]
    ):
        task_type = "summary"
    else:
        task_type = "qa"

    return {
        "task_type": task_type,
        "scope": _route_scope_by_rules(query, task_type),
    }


def parse_task_route(raw_result: str) -> dict[str, str] | None:
    text = raw_result.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    task_type = parsed.get("task_type")
    scope = parsed.get("scope")
    if not isinstance(task_type, str) or not isinstance(scope, str):
        return None
    task_type = task_type.strip().lower()
    scope = scope.strip().lower()
    if task_type not in SUPPORTED_TASK_TYPES:
        return None
    if scope not in SUPPORTED_TASK_SCOPES:
        return None
    if task_type != "summary" and scope != "focused":
        return None
    return {"task_type": task_type, "scope": scope}


def route_task(query: str) -> dict[str, str]:
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")

    fallback_route = route_task_by_rules(query)
    if is_collective_document_question(query):
        return fallback_route
    prompt = load_router_prompt_template().format(query=query)
    try:
        raw_result = generate_answer(prompt, operation="task_route")
        route = parse_task_route(raw_result)
        if route is not None:
            logger.info(
                "task_route_completed strategy=llm task_type=%s scope=%s",
                route["task_type"],
                route["scope"],
            )
            return route
        logger.warning(
            "task_route_invalid_response fallback_task=%s fallback_scope=%s",
            fallback_route["task_type"],
            fallback_route["scope"],
        )
    except Exception as exc:
        logger.warning(
            "task_route_failed error_type=%s fallback_task=%s fallback_scope=%s",
            type(exc).__name__,
            fallback_route["task_type"],
            fallback_route["scope"],
        )
    return fallback_route
