from services import evidence_filter_service as service


def make_chunk(score: float | None) -> dict:
    chunk = {"文本块": "evidence"}
    if score is not None:
        chunk["rerank_score"] = score
    return chunk


def test_disabled_threshold_preserves_candidates(monkeypatch):
    monkeypatch.setattr(service, "EVIDENCE_RERANK_MIN_SCORE", None)
    chunks = [make_chunk(-2.0), make_chunk(None)]

    assert service.filter_relevant_evidence(chunks) is chunks


def test_threshold_keeps_only_relevant_candidates(monkeypatch):
    monkeypatch.setattr(service, "EVIDENCE_RERANK_MIN_SCORE", 0.0)
    chunks = [make_chunk(1.5), make_chunk(0.0), make_chunk(-0.1)]

    result = service.filter_relevant_evidence(chunks)

    assert [chunk["rerank_score"] for chunk in result] == [1.5, 0.0]


def test_threshold_rejects_candidate_without_rerank_score(monkeypatch):
    monkeypatch.setattr(service, "EVIDENCE_RERANK_MIN_SCORE", 0.0)

    assert service.filter_relevant_evidence([make_chunk(None)]) == []


def test_explicit_threshold_overrides_configuration(monkeypatch):
    monkeypatch.setattr(service, "EVIDENCE_RERANK_MIN_SCORE", 10.0)

    result = service.filter_relevant_evidence(
        [make_chunk(0.8), make_chunk(0.4)],
        min_score=0.5,
    )

    assert [chunk["rerank_score"] for chunk in result] == [0.8]
