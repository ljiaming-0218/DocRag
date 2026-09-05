def build_candidate_key(candidate: dict) -> tuple:
    """Build a stable identity used to deduplicate retrieval candidates."""
    chunk_id = candidate.get("chunk_id")
    if chunk_id:
        return "chunk_id", chunk_id

    metadata = candidate.get("元数据") or {}
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index")
    if document_id is not None and chunk_index is not None:
        return "document_chunk", document_id, chunk_index

    return (
        "fallback",
        document_id,
        metadata.get("page_number"),
        candidate.get("文本块", ""),
    )


def reciprocal_rank_fusion(
    ranked_candidates: dict[str, list[dict]],
    limit: int,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse ranked lists without comparing their incompatible raw scores."""
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if not ranked_candidates:
        return []

    fused_candidates = {}
    source_order = {
        source_name: index
        for index, source_name in enumerate(ranked_candidates)
    }

    for source_name, candidates in ranked_candidates.items():
        if not source_name.strip():
            raise ValueError("retrieval source name cannot be empty")

        seen_keys = set()
        for rank, candidate in enumerate(candidates, start=1):
            candidate_key = build_candidate_key(candidate)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)

            fused = fused_candidates.get(candidate_key)
            if fused is None:
                fused = {
                    "candidate": {
                        **candidate,
                        "元数据": dict(candidate.get("元数据") or {}),
                    },
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "first_source": source_order[source_name],
                    "retrieval_sources": [],
                }
                fused_candidates[candidate_key] = fused
            else:
                base_candidate = fused["candidate"]
                for field, value in candidate.items():
                    if field in {"元数据", "retrieval_sources"}:
                        continue
                    if value is not None and base_candidate.get(field) is None:
                        base_candidate[field] = value
                base_candidate["元数据"].update(
                    candidate.get("元数据") or {}
                )

            fused["rrf_score"] += 1.0 / (rrf_k + rank)
            fused["best_rank"] = min(fused["best_rank"], rank)
            if source_name not in fused["retrieval_sources"]:
                fused["retrieval_sources"].append(source_name)

            if source_name == "dense":
                fused["candidate"]["dense_rank"] = rank
            elif source_name == "sparse":
                fused["candidate"]["sparse_rank"] = rank

    ordered = sorted(
        fused_candidates.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["best_rank"],
            item["first_source"],
        ),
    )

    results = []
    for item in ordered[:limit]:
        candidate = item["candidate"]
        candidate["rrf_score"] = item["rrf_score"]
        candidate["retrieval_sources"] = item["retrieval_sources"]
        results.append(candidate)
    return results
