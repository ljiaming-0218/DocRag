def route_task(query: str) -> str:
    normalized_query = query.strip().lower()

    if any(
        keyword in normalized_query
        for keyword in ["阅读报告", "分析这篇文献"]
    ):
        return "report"

    if any(
        keyword in normalized_query
        for keyword in [
            "依据",
            "出处",
            "引用",
            "来源",
            "证据",
            "citation",
            "evidence",
        ]
    ):
        return "source_check"

    if any(
        keyword in normalized_query
        for keyword in [
            "术语",
            "关键词",
            "关键概念",
            "专业概念",
            "概念解释",
            "名词解释",
            "keywords",
            "key terms",
            "terminology",
        ]
    ):
        return "term"

    if any(
        keyword in normalized_query
        for keyword in ["总结", "摘要", "概括", "归纳"]
    ):
        return "summary"

    return "qa"
