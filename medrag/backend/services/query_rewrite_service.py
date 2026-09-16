
import re
import logging
from pathlib import Path
from time import perf_counter

from services.language_service import detect_text_language
from services.llm_service import generate_answer


logger = logging.getLogger(__name__)

UNRESOLVED_REFERENCE_PATTERNS = (
    r"它",
    r"该方法",
    r"这个方法",
    r"该模型",
    r"这个模型",
    r"这些任务",
    r"上述任务",
    r"另一个变体",
    r"\bit\b",
    r"\bthe model\b",
    r"\bthis model\b",
    r"\bthis method\b",
    r"\bthat method\b",
    r"\bthese tasks\b",
    r"\bthose tasks\b",
    r"\bthe other variant\b",
)
CRITICAL_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}(?:[-_.][A-Z0-9]+)*|"
    r"[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|"
    r"[A-Za-z][A-Za-z0-9_-]*@[0-9]+|"
    r"\d+(?:\.\d+)?(?:[BMK]|%)?)\b|"
    r"(?:(?i:Table|Figure|Fig\.?)|表|图)\s*\d+"
)
CRITICAL_PHRASE_PATTERN = re.compile(
    r"\b(?:full[ -]fine[ -]tuning|fine[ -]tuning)\b|全量微调",
    flags=re.IGNORECASE,
)


def load_query_rewrite_template() -> str:
    file = Path(__file__).parent.parent / "prompts" / "query_rewrite_prompt.txt"
    with open(file, "r", encoding="utf-8") as f:
        return f.read()

def build_history_text(history: list[dict]) -> str:
    if not history:
        return "无历史对话"

    lines = []
    for message in history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            role_name = "用户"
        elif role == "assistant":
            role_name = "助手"
        else:
            role_name = role or "未知角色"

        if content:
            lines.append(f"{role_name}: {content}")

    return "\n".join(lines) if lines else "无历史对话"


def build_query_rewrite_prompt(
    history: list[dict],
    query: str,
    target_language: str,
) -> str:
    template = load_query_rewrite_template()
    history_text = build_history_text(history)
    language_name = {
        "zh": "中文",
        "en": "英文",
    }.get(target_language, "保持原语言")
    return template.format(
        history=history_text,
        query=query,
        target_language=language_name,
    )


def has_unresolved_reference(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in UNRESOLVED_REFERENCE_PATTERNS
    )


def extract_critical_entities(
    history: list[dict],
    query: str,
) -> list[str]:
    """Extract explicit technical identifiers without asking an LLM."""
    relevant_texts = [query]
    recent_user_messages = [
        message.get("content", "")
        for message in history
        if message.get("role") == "user"
        and message.get("content")
    ][-3:]
    relevant_texts.extend(recent_user_messages)

    entities = []
    seen = set()
    for match in CRITICAL_ENTITY_PATTERN.finditer("\n".join(relevant_texts)):
        entity = match.group(0).strip()
        normalized = entity.casefold()
        if entity and normalized not in seen:
            entities.append(entity)
            seen.add(normalized)
    for match in CRITICAL_PHRASE_PATTERN.finditer("\n".join(relevant_texts)):
        entity = match.group(0).strip()
        normalized = entity.casefold()
        if normalized not in seen:
            entities.append(entity)
            seen.add(normalized)
    return entities


def preserve_critical_entities(
    rewritten_query: str,
    critical_entities: list[str],
) -> str:
    normalized_query = rewritten_query.casefold()
    missing_entities = [
        entity
        for entity in critical_entities
        if entity.casefold() not in normalized_query
    ]
    if not missing_entities:
        return rewritten_query

    logger.warning(
        "query_rewrite_entities_restored missing_count=%s",
        len(missing_entities),
    )
    return (
        f"{rewritten_query.rstrip()} "
        f"[Required entities: {', '.join(missing_entities)}]"
    )


def build_query_repair_prompt(
    history: list[dict],
    query: str,
    rewritten_query: str,
    target_language: str,
) -> str:
    history_text = build_history_text(history)
    language_name = {
        "zh": "中文",
        "en": "英文",
    }.get(target_language, "保持原语言")
    return (
        "你是一个 RAG 检索查询修复助手。\n"
        "上一次改写结果仍然包含依赖历史的未解析指代。\n\n"
        "最近对话：\n"
        f"{history_text}\n\n"
        "用户原始问题：\n"
        f"{query}\n\n"
        "上一次改写结果：\n"
        f"{rewritten_query}\n\n"
        f"目标输出语言：{language_name}\n\n"
        "请使用历史中明确出现的实体、方法和任务名称替换所有指代。\n"
        "修复结果必须脱离历史也能独立理解，不要回答问题，"
        "不要添加历史中不存在的事实。\n"
        "只输出修复后的检索问题。"
    )



def rewrite_query(
    history: list[dict],
    query: str,
    target_language: str = "unknown",
) -> str:
    started_at = perf_counter()
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")

    query_language = detect_text_language(query)

    language_mismatch = (
        target_language in {"zh", "en"}
        and query_language != "unknown"
        and query_language != target_language
    )

    if not history and not language_mismatch:
        return query

    critical_entities = extract_critical_entities(history, query)

    try:
        prompt = build_query_rewrite_prompt(
            history,
            query,
            target_language,
        )
        rewritten_query = generate_answer(prompt, operation="query_rewrite").strip()
    except Exception as exc:
        logger.warning(
            "query_rewrite_failed error_type=%s query_rewrite_ms=%.2f",
            type(exc).__name__,
            (perf_counter() - started_at) * 1000,
        )
        return query

    if not rewritten_query:
        return query

    if history and has_unresolved_reference(rewritten_query):
        try:
            repair_prompt = build_query_repair_prompt(
                history,
                query,
                rewritten_query,
                target_language,
            )
            repaired_query = generate_answer(
                repair_prompt,
                operation="query_rewrite",
            ).strip()
            if repaired_query:
                rewritten_query = repaired_query
        except Exception as exc:
            logger.warning(
                "query_rewrite_repair_failed error_type=%s",
                type(exc).__name__,
            )

    rewritten_query = preserve_critical_entities(
        rewritten_query,
        critical_entities,
    )
    logger.info(
        "query_rewrite_completed history_count=%s source_length=%s "
        "result_length=%s query_rewrite_ms=%.2f",
        len(history),
        len(query),
        len(rewritten_query),
        (perf_counter() - started_at) * 1000,
    )
    return rewritten_query
