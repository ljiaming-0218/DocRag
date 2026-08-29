import logging

from services.chunk_validation_service import validate_chunk_params


logger = logging.getLogger(__name__)

DEFAULT_CHUNK_STRATEGY = "fixed"
SUPPORTED_CHUNK_STRATEGIES = {"fixed", "recursive"}
CHUNK_STRATEGY_VERSIONS = {
    "fixed": "fixed:v1",
    "recursive": "recursive:v1",
}

RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    "! ",
    "? ",
    "; ",
    " ",
    "",
]


def resolve_chunk_strategy(strategy: str | None) -> str:

    if not strategy:
        return DEFAULT_CHUNK_STRATEGY

    normalized_strategy = strategy.strip().lower()
    if normalized_strategy in SUPPORTED_CHUNK_STRATEGIES:
        return normalized_strategy

    logger.warning(
        "unsupported_chunk_strategy requested=%s fallback=%s",
        strategy,
        DEFAULT_CHUNK_STRATEGY,
    )
    return DEFAULT_CHUNK_STRATEGY


def get_chunk_version(strategy: str | None) -> str:
    actual_strategy = resolve_chunk_strategy(strategy)
    return CHUNK_STRATEGY_VERSIONS[actual_strategy]


def split_text_fixed(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text by a fixed character window."""
    validate_chunk_params(chunk_size, chunk_overlap)

    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks


def split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
) -> list[str]:

    validate_chunk_params(chunk_size, chunk_overlap)

    normalized_text = text.strip()
    if not normalized_text:
        return []

    content_size = chunk_size - chunk_overlap
    base_chunks = _split_recursively(
        normalized_text,
        content_size,
        RECURSIVE_SEPARATORS,
    )

    return apply_chunk_overlap(
        base_chunks,
        chunk_size,
        chunk_overlap,
    )


def _split_recursively(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        return [
            text[start:start + chunk_size]
            for start in range(0, len(text), chunk_size)
        ]

    parts = text.split(separator)
    if len(parts) == 1:
        return _split_recursively(
            text,
            chunk_size,
            remaining_separators,
        )

    units = [
        part + separator if index < len(parts) - 1 else part
        for index, part in enumerate(parts)
        if part
    ]

    chunks = []
    current = ""

    for unit in units:
        candidate = current + unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())

        if len(unit) > chunk_size:
            chunks.extend(
                _split_recursively(
                    unit.strip(),
                    chunk_size,
                    remaining_separators,
                )
            )
            current = ""
        else:
            current = unit

    if current.strip():
        chunks.append(current.strip())

    return chunks


def apply_chunk_overlap(
    chunks: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not chunks or chunk_overlap == 0:
        return chunks

    overlapped_chunks = [chunks[0]]

    for index in range(1, len(chunks)):
        previous_chunk = chunks[index - 1]
        current_chunk = chunks[index]
        available_space = chunk_size - len(current_chunk)

        if available_space <= 0:
            overlapped_chunks.append(current_chunk)
            continue

        actual_overlap = min(
            chunk_overlap,
            available_space,
            len(previous_chunk),
        )
        overlap_text = previous_chunk[-actual_overlap:]
        overlapped_chunks.append(overlap_text + current_chunk)

    return overlapped_chunks


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str | None = DEFAULT_CHUNK_STRATEGY,
) -> list[str]:

    actual_strategy = resolve_chunk_strategy(strategy)

    if actual_strategy == "recursive":
        return split_text_recursive(
            text,
            chunk_size,
            chunk_overlap,
        )

    return split_text_fixed(
        text,
        chunk_size,
        chunk_overlap,
    )


def split_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str | None = DEFAULT_CHUNK_STRATEGY,
) -> list[dict]:

    actual_strategy = resolve_chunk_strategy(strategy)
    all_chunks = []
    chunk_index = 0

    for page in pages:
        chunks = split_text(
            page["文本"],
            chunk_size,
            chunk_overlap,
            actual_strategy,
        )

        for chunk in chunks:
            chunk_index += 1
            all_chunks.append({
                "页码": page["页码"],
                "文本块": chunk,
                "块索引": chunk_index,
                "document_id": page["document_id"],
                "user_id": page["user_id"],
                "提取方式": page.get("提取方式", "text"),
                "图片数量": page.get("图片数量", 0),
                "strategy": actual_strategy,
            })

    return all_chunks
