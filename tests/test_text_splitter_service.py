import pytest

from services.text_splitter_service import (
    apply_chunk_overlap,
    get_chunk_version,
    resolve_chunk_strategy,
    split_pages,
    split_text,
    split_text_fixed,
    split_text_recursive,
)


def compact_text(text: str) -> str:
    return "".join(text.split())


def assert_chunks_preserve_text(text: str, chunks: list[str]) -> None:
    assert chunks
    assert all(chunks)
    assert compact_text("".join(chunks)) == compact_text(text)


def test_resolve_chunk_strategy_supports_fixed_and_recursive(caplog):
    assert resolve_chunk_strategy(None) == "fixed"
    assert resolve_chunk_strategy(" Recursive ") == "recursive"
    assert resolve_chunk_strategy("semantic") == "fixed"
    assert "unsupported_chunk_strategy" in caplog.text


def test_chunk_strategy_versions_are_explicit():
    assert get_chunk_version("fixed") == "fixed:v1"
    assert get_chunk_version("recursive") == "recursive:v1"


def test_fixed_splitter_keeps_existing_overlap_behavior():
    chunks = split_text_fixed(
        "abcdefghij",
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks == ["abcd", "defg", "ghij", "j"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_fixed_splitter_rejects_invalid_parameters(
    chunk_size: int,
    chunk_overlap: int,
):
    with pytest.raises(ValueError):
        split_text_fixed("test", chunk_size, chunk_overlap)


def test_recursive_splitter_returns_empty_list_for_blank_text():
    assert split_text_recursive("   \n\t", chunk_size=20) == []


def test_recursive_splitter_returns_short_text_as_one_chunk():
    text = "短文本不需要继续切分。"
    assert split_text_recursive(text, chunk_size=50) == [text]


def test_recursive_splitter_preserves_chinese_content_and_size():
    text = (
        "第一段介绍RAG。这里说明检索过程。\n\n"
        "第二段介绍生成过程。模型根据证据回答问题。"
    )

    chunks = split_text_recursive(text, chunk_size=20)

    assert_chunks_preserve_text(text, chunks)
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert any(chunk.endswith("。") for chunk in chunks)


def test_recursive_splitter_preserves_english_content_and_size():
    text = (
        "Retrieval finds relevant evidence. "
        "Generation uses that evidence. "
        "The answer cites its sources."
    )

    chunks = split_text_recursive(text, chunk_size=30)

    assert_chunks_preserve_text(text, chunks)
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_recursive_splitter_handles_mixed_language_text():
    text = (
        "RAG 的核心是 Retrieval-Augmented Generation。"
        "The retriever retrieves relevant chunks."
    )

    chunks = split_text_recursive(text, chunk_size=25)

    assert_chunks_preserve_text(text, chunks)
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_recursive_splitter_falls_back_to_character_splitting():
    text = "A" * 53
    chunks = split_text_recursive(text, chunk_size=20)

    assert chunks == ["A" * 20, "A" * 20, "A" * 13]


def test_recursive_splitter_applies_overlap_without_exceeding_size():
    chunks = split_text_recursive(
        "abcdefghijklmnop",
        chunk_size=6,
        chunk_overlap=2,
    )

    assert chunks == ["abcd", "cdefgh", "ghijkl", "klmnop"]
    assert all(len(chunk) <= 6 for chunk in chunks)


def test_overlap_shrinks_when_current_chunk_has_limited_space():
    chunks = apply_chunk_overlap(
        ["abcdef", "123456789"],
        chunk_size=10,
        chunk_overlap=4,
    )

    assert chunks == ["abcdef", "f123456789"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_recursive_splitter_rejects_invalid_parameters(
    chunk_size: int,
    chunk_overlap: int,
):
    with pytest.raises(ValueError):
        split_text_recursive("test", chunk_size, chunk_overlap)


def test_recursive_strategy_is_available_through_dispatcher():
    text = "第一句介绍检索。第二句介绍生成。"

    expected = split_text_recursive(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )
    actual = split_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
        strategy="recursive",
    )

    assert actual == expected


def test_split_pages_preserves_metadata_and_strategy():
    pages = [{
        "页码": 2,
        "文本": "abcdefgh",
        "document_id": "document-1",
        "user_id": "user-1",
        "提取方式": "ocr",
        "图片数量": 1,
    }]

    chunks = split_pages(
        pages,
        chunk_size=4,
        chunk_overlap=0,
        strategy="fixed",
    )

    assert [chunk["文本块"] for chunk in chunks] == ["abcd", "efgh"]
    assert [chunk["块索引"] for chunk in chunks] == [1, 2]
    assert all(chunk["页码"] == 2 for chunk in chunks)
    assert all(chunk["document_id"] == "document-1" for chunk in chunks)
    assert all(chunk["user_id"] == "user-1" for chunk in chunks)
    assert all(chunk["提取方式"] == "ocr" for chunk in chunks)
    assert all(chunk["图片数量"] == 1 for chunk in chunks)
    assert all(chunk["strategy"] == "fixed" for chunk in chunks)
