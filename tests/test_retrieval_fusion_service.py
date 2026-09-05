import pytest

from services.retrieval_fusion_service import (
    build_candidate_key,
    reciprocal_rank_fusion,
)


def make_candidate(
    document_id: str,
    chunk_index: int,
    *,
    distance: float | None = None,
    sparse_score: float | None = None,
) -> dict:
    candidate = {
        "chunk_id": f"{document_id}-{chunk_index}",
        "文本块": f"chunk {document_id}-{chunk_index}",
        "距离": distance,
        "元数据": {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "page_number": 1,
        },
    }
    if sparse_score is not None:
        candidate["sparse_score"] = sparse_score
    return candidate


def test_build_candidate_key_prefers_chunk_id():
    candidate = make_candidate("document-1", 3)

    assert build_candidate_key(candidate) == (
        "chunk_id",
        "document-1-3",
    )


def test_rrf_merges_duplicate_candidate_and_preserves_scores():
    dense = [make_candidate("document-1", 0, distance=0.2)]
    sparse = [make_candidate("document-1", 0, sparse_score=4.5)]

    result = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse},
        limit=3,
        rrf_k=60,
    )

    assert len(result) == 1
    assert result[0]["距离"] == 0.2
    assert result[0]["sparse_score"] == 4.5
    assert result[0]["dense_rank"] == 1
    assert result[0]["sparse_rank"] == 1
    assert result[0]["retrieval_sources"] == ["dense", "sparse"]
    assert result[0]["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_uses_rank_instead_of_incompatible_raw_scores():
    dense_first = make_candidate("document-1", 0, distance=100.0)
    sparse_first = make_candidate("document-2", 0, sparse_score=0.01)

    result = reciprocal_rank_fusion(
        {
            "dense": [dense_first, sparse_first],
            "sparse": [sparse_first, dense_first],
        },
        limit=2,
    )

    assert {item["chunk_id"] for item in result} == {
        "document-1-0",
        "document-2-0",
    }
    assert result[0]["rrf_score"] == result[1]["rrf_score"]


def test_rrf_deduplicates_repeated_candidate_in_same_source():
    candidate = make_candidate("document-1", 0, distance=0.2)

    result = reciprocal_rank_fusion(
        {"dense": [candidate, candidate]},
        limit=3,
    )

    assert len(result) == 1
    assert result[0]["rrf_score"] == pytest.approx(1 / 61)


def test_rrf_applies_global_limit():
    result = reciprocal_rank_fusion(
        {
            "dense": [
                make_candidate("document-1", 0),
                make_candidate("document-1", 1),
            ],
            "sparse": [
                make_candidate("document-2", 0),
                make_candidate("document-2", 1),
            ],
        },
        limit=2,
    )

    assert len(result) == 2


@pytest.mark.parametrize(
    ("ranked_candidates", "limit", "rrf_k", "message"),
    [
        ({"dense": []}, 0, 60, "limit must be greater than 0"),
        ({"dense": []}, 3, 0, "rrf_k must be greater than 0"),
        ({"": []}, 3, 60, "source name cannot be empty"),
    ],
)
def test_rrf_rejects_invalid_parameters(
    ranked_candidates,
    limit,
    rrf_k,
    message,
):
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(
            ranked_candidates,
            limit,
            rrf_k,
        )
