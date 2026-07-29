import json
from pathlib import Path

import sys
from copy import deepcopy



BASE_DIR = Path(__file__).resolve().parent
RESULT_PATH = (
    BASE_DIR
    / "rag_dataset"
    / "results"
    / "normal_results.json"
)
QUESTIONS_PATH = (
    BASE_DIR
    / "rag_dataset"
    / "questions.json"
)
PROJECT_ROOT = BASE_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "medrag" / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from services.rerank_service import rerank_chunks
from services.embedding_service import get_embedding
from services.vector_store_service import query_chunks

def load_eval_case(question_id: str) -> dict:
    result_data = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    result_item = next(
        item for item in result_data["results"]
        if item["question_id"] == question_id
    )
    question_item = next(
        item for item in questions
        if item["question_id"] == question_id
    )

    return {
        "user_id": result_data["user_id"],
        "document_id": result_item["document_id"],
        "question_id": question_id,
        "question": question_item["question"],
        "gold_source_pages": question_item["source_pages"],
    }
def build_query_variants(original_query: str) -> dict:
    english_query = (
        "What is the core mechanism of LoRA? "
        "Which pretrained parameters are frozen, "
        "and which low-rank matrices are updated during training?"
    )

    bilingual_query = (
        f"{original_query}\n"
        "English keywords: Low-Rank Adaptation, "
        "frozen pretrained weights, "
        "trainable low-rank matrices A and B."
    )

    return {
        "chinese": original_query,
        "english": english_query,
        "bilingual": bilingual_query,
    }

def retrieve_candidates(
    user_id: str,
    document_id: str,
    query: str,
    candidate_k: int = 15,
) -> list[dict]:
    query_embedding = get_embedding(query)

    raw_result = query_chunks(
        user_id=user_id,
        document_id=document_id,
        query_embedding=query_embedding,
        n_results=candidate_k,
    )

    documents = raw_result["documents"][0]
    distances = raw_result["distances"][0]
    metadatas = raw_result["metadatas"][0]

    candidates = []

    for rank, (text, distance, metadata) in enumerate(
        zip(documents, distances, metadatas),
        start=1,
    ):
        candidates.append({
            "vector_rank": rank,
            "文本块": text,
            "距离": distance,
            "元数据": metadata,
        })

    return candidates

def rerank_candidates(
    query: str,
    candidates: list[dict],
) -> list[dict]:
    rerank_input = deepcopy(candidates)

    ranked = rerank_chunks(
        query=query,
        chunks=rerank_input,
        top_k=len(rerank_input),
    )

    for rank, candidate in enumerate(ranked, start=1):
        candidate["rerank_rank"] = rank

    return ranked

def print_reranked_candidates(
    candidates: list[dict],
    gold_pages: list[int],
) -> None:
    for candidate in candidates:
        metadata = candidate["元数据"]
        page = metadata["page_number"]
        marker = "GOLD" if page in gold_pages else ""

        print(
            f"rerank={candidate['rerank_rank']:2d} "
            f"vector={candidate['vector_rank']:2d} "
            f"page={page:2d} "
            f"chunk={metadata['chunk_index']:3d} "
            f"score={candidate['rerank_score']:.4f} "
            f"{marker}"
        )

def print_candidates(
    candidates: list[dict],
    gold_pages: list[int],
) -> None:
    for candidate in candidates:
        metadata = candidate["元数据"]
        page_number = metadata["page_number"]
        chunk_index = metadata["chunk_index"]

        is_gold_page = page_number in gold_pages
        gold_marker = "GOLD" if is_gold_page else ""

        text_preview = (
            candidate["文本块"]
            .replace("\n", " ")
            [:100]
        )

        print(
            f"rank={candidate['vector_rank']:2d} "
            f"page={page_number:2d} "
            f"chunk={chunk_index:3d} "
            f"distance={candidate['距离']:.4f} "
            f"{gold_marker}"
        )
        print(f"  {text_preview}")

def main() -> None:
    case = load_eval_case("LORA-01")
    query_variants = build_query_variants(case["question"])

    print(f"question_id: {case['question_id']}")
    print(f"user_id: {case['user_id']}")
    print(f"document_id: {case['document_id']}")
    print(f"gold pages: {case['gold_source_pages']}")

    for variant_name, query in query_variants.items():
        print(f"\n[{variant_name}]")
        print(query)

        candidates = retrieve_candidates(
            user_id=case["user_id"],
            document_id=case["document_id"],
            query=query,
            candidate_k=15,
        )

        print_candidates(
            candidates,
            case["gold_source_pages"],
        )
        reranked = rerank_candidates(query, candidates)

        print("\nRerank 后：")
        print_reranked_candidates(
            reranked,
            case["gold_source_pages"],
        )


if __name__ == "__main__":
    main()