from pathlib import Path


PROMPT_DIR = Path(__file__).parent.parent / "prompts"

USER_TYPE_INSTRUCTIONS = {
    "undergraduate": "少术语，多解释基础概念",
    "graduate_student": "强调研究问题、方法、创新点、局限",
    "researcher": "强调相关工作、贡献、实验设置、可复现性",
    "developer": "强调实现流程、技术路线、工程落地",
    "teacher": "强调知识结构、教学讲解",
    "general": "简洁概括",
}


def load_prompt_template(filename: str) -> str:
    prompt_path = PROMPT_DIR / filename
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict],
    user_type: str = "general",
    history: list[dict] | None = None,
) -> str:
    prompt_template = load_prompt_template("qa_prompt.txt")
    context = build_context(retrieved_chunks)
    history_context = build_history_context(history or [])
    user_type_instruction = build_user_type_instruction(user_type)
    return prompt_template.format(
        history=history_context,
        question=question,
        context=context,
        user_type=user_type,
        user_type_instruction=user_type_instruction,
    )


def build_report_prompt(
    query: str,
    sources: list[dict],
    user_type: str,
    history: list[dict],
) -> str:
    prompt_template = load_prompt_template("report_prompt.txt")
    return prompt_template.format(
        history=build_history_context(history or []),
        query=query,
        context=build_context(sources),
        user_type=user_type,
        user_type_instruction=build_user_type_instruction(user_type),
    )


def build_summary_prompt(
    query: str,
    sources: list[dict],
    user_type: str,
    history: list[dict],
) -> str:
    prompt_template = load_prompt_template("summary_prompt.txt")
    return prompt_template.format(
        history=build_history_context(history or []),
        query=query,
        context=build_context(sources),
        user_type=user_type,
        user_type_instruction=build_user_type_instruction(user_type),
    )


def build_term_prompt(
    query: str,
    sources: list[dict],
    user_type: str,
    history: list[dict],
) -> str:
    prompt_template = load_prompt_template("term_prompt.txt")
    return prompt_template.format(
        history=build_history_context(history or []),
        query=query,
        context=build_context(sources),
        user_type=user_type,
        user_type_instruction=build_user_type_instruction(user_type),
    )


def build_source_check_prompt(
    query: str,
    sources: list[dict],
    user_type: str,
    history: list[dict],
) -> str:
    prompt_template = load_prompt_template("source_check_prompt.txt")
    return prompt_template.format(
        history=build_history_context(history or []),
        query=query,
        context=build_context(sources),
        user_type=user_type,
        user_type_instruction=build_user_type_instruction(user_type),
    )


def build_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for chunk in retrieved_chunks:
        metadata = chunk["元数据"]
        context_parts.append(
            f"[第{metadata['page_number']}页, "
            f"distance {chunk['距离']}, "
            f"chunk {metadata['chunk_index']}]\n"
            f"{chunk['文本块']}"
        )

    return "\n".join(context_parts)


def build_history_context(history: list[dict]) -> str:
    if not history:
        return "无历史对话"

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )


def build_user_type_instruction(user_type: str) -> str:
    return USER_TYPE_INSTRUCTIONS.get(
        user_type,
        USER_TYPE_INSTRUCTIONS["general"],
    )
