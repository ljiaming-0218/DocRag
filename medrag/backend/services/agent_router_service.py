def route_task(query: str) -> str:
    if any(
        keyword in query
        for keyword in ["阅读报告", "分析这篇文献"]
    ):
        return "report"

    if any(
        keyword in query
        for keyword in ["总结", "摘要", "概括", "归纳"]
    ):
        return "summary"

    return "qa"
