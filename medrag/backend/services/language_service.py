MAX_LANGUAGE_TEXT_LENGTH = 20_000


def detect_document_language(pages: list[dict]) -> str:
    text_parts = []
    collected_length = 0

    for page in pages:
        page_text = page.get("文本", "")

        if not isinstance(page_text, str):
            continue

        page_text = page_text.strip()
        if not page_text:
            continue

        remaining_length = (
            MAX_LANGUAGE_TEXT_LENGTH - collected_length
        )

        if remaining_length <= 0:
            break

        selected_text = page_text[:remaining_length]
        text_parts.append(selected_text)
        collected_length += len(selected_text)

    text = "".join(text_parts)

    return detect_text_language(
        text,
        minimum_character_count=100,
    )


def detect_text_language(
    text: str,
    minimum_character_count: int = 1,
) -> str:
    chinese_count = sum(
        "\u4e00" <= char <= "\u9fff"
        for char in text
    )

    english_count = sum(
        char.isascii() and char.isalpha()
        for char in text
    )

    valid_count = chinese_count + english_count

    if valid_count < minimum_character_count:
        return "unknown"

    chinese_ratio = chinese_count / valid_count

    return "zh" if chinese_ratio >= 0.3 else "en"