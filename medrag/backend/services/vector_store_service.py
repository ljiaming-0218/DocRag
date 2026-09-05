import logging
from time import perf_counter

import chromadb
from chromadb.errors import InternalError, NotFoundError

from config import CHROMA_DIR, VECTOR_WRITE_BATCH_SIZE


logger = logging.getLogger(__name__)
COLLECTION_NAME = "medical_chunks"


def _empty_query_result() -> dict:
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _normalize_document_ids(document_ids: list[str]) -> list[str]:
    if not document_ids:
        raise ValueError("document_ids cannot be empty")

    normalized_document_ids = []
    for document_id in document_ids:
        document_id = document_id.strip()
        if not document_id:
            raise ValueError("document_id cannot be empty")
        if document_id not in normalized_document_ids:
            normalized_document_ids.append(document_id)
    return normalized_document_ids


def _build_document_scope_filter(
    user_id: str,
    document_ids: list[str],
    index_generations: dict[str, str | None] | None = None,
) -> dict:
    """Build a user-scoped filter for active or legacy document indexes."""
    normalized_document_ids = _normalize_document_ids(document_ids)
    has_versioned_scope = bool(
        index_generations
        and any(
            index_generations.get(document_id)
            for document_id in normalized_document_ids
        )
    )
    if not has_versioned_scope:
        document_scope = (
            {"document_id": normalized_document_ids[0]}
            if len(normalized_document_ids) == 1
            else {
                "document_id": {
                    "$in": normalized_document_ids,
                }
            }
        )
        return {
            "$and": [
                {"user_id": user_id},
                document_scope,
            ]
        }

    document_filters = []
    for document_id in normalized_document_ids:
        generation_id = (
            index_generations.get(document_id)
            if index_generations is not None
            else None
        )
        conditions = [{"document_id": document_id}]
        if generation_id:
            conditions.append({"index_generation_id": generation_id})
        document_filters.append(
            conditions[0]
            if len(conditions) == 1
            else {"$and": conditions}
        )

    document_scope = (
        document_filters[0]
        if len(document_filters) == 1
        else {"$or": document_filters}
    )
    return {
        "$and": [
            {"user_id": user_id},
            document_scope,
        ]
    }


def _raise_chroma_error(operation: str, error: Exception) -> None:
    raise RuntimeError(
        f"向量数据库{operation}失败，可能是本地 Chroma 索引损坏。"
        f"请备份并重建 {CHROMA_DIR} 后重新索引 PDF。原始错误: {error}"
    ) from error



def build_chunk_metadata(chunk: dict) -> dict:
    metadata = {
        "page_number": chunk["页码"],
        "chunk_index": chunk["块索引"],
        "document_id": chunk["document_id"],
        "user_id": chunk["user_id"],
        "extraction_method": chunk.get("提取方式", "text"),
        "image_count": chunk.get("图片数量", 0),
        "chunk_strategy": chunk.get("strategy", "fixed"),
    }
    optional_fields = (
        "chunk_size",
        "chunk_overlap",
        "chunk_version",
        "embedding_model",
        "embedding_version",
        "index_version",
        "index_fingerprint",
        "index_generation_id",
    )
    for field in optional_fields:
        value = chunk.get(field)
        if value is not None:
            metadata[field] = value
    return metadata



def has_chunks(
    user_id: str,
    document_id: str,
    index_generation_id: str | None = None,
) -> bool:
    if not user_id:
        raise ValueError("user_id 不能为空")
    if not document_id:
        raise ValueError("document_id 不能为空")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return False
    except InternalError as exc:
        _raise_chroma_error("读取集合", exc)

    try:
        result = collection.get(
            where=_build_document_scope_filter(
                user_id,
                [document_id],
                {document_id: index_generation_id},
            ),
            limit=1,
        )
    except InternalError as exc:
        _raise_chroma_error("检查文档索引", exc)

    return bool(result.get("ids"))


def save_chunks(
    chunks: list[dict],
    batch_size: int = VECTOR_WRITE_BATCH_SIZE,
) -> int:
    if not chunks:
        return 0
    if batch_size <= 0:
        raise ValueError("vector write batch_size 必须大于 0")
    
    document_id = chunks[0]["document_id"]
    user_id = chunks[0]["user_id"]
    index_generation_id = chunks[0].get("index_generation_id")
    for chunk in chunks:
        if chunk["document_id"] != document_id:
            raise ValueError(f"所有块必须具有相同的 document_id，发现不匹配的 document_id: {chunk['document_id']}")
        if chunk["user_id"] != user_id:
            raise ValueError(f"所有块必须具有相同的 user_id，发现不匹配的 user_id: {chunk['user_id']}")
        if chunk.get("index_generation_id") != index_generation_id:
            raise ValueError("所有块必须具有相同的 index_generation_id")

            
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except InternalError as exc:
        _raise_chroma_error("初始化", exc)

    old_ids = set()
    if index_generation_id is None:
        try:
            old_ids = get_document_chunk_ids(
                collection,
                user_id,
                document_id,
            )
        except InternalError as exc:
            _raise_chroma_error("读取旧索引", exc)

    new_ids = [
        build_chunk_id(chunk)
        for chunk in chunks
    ]
    new_id_set = set(new_ids)
    if len(new_id_set) != len(new_ids):
        raise ValueError("文本块生成了重复的 chunk_id")

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(
        range(0, len(chunks), batch_size),
        start=1,
    ):
        batch_started_at = perf_counter()
        batch = chunks[start:start + batch_size]
        batch_ids = [build_chunk_id(chunk) for chunk in batch]
        try:
            collection.upsert(
                ids=batch_ids,
                documents=[chunk["文本块"] for chunk in batch],
                embeddings=[chunk["embedding"] for chunk in batch],
                metadatas=[build_chunk_metadata(chunk) for chunk in batch],
            )
        except InternalError as exc:
            _raise_chroma_error("写入索引", exc)

        verify_chunk_ids(collection, set(batch_ids))
        logger.info(
            "vector_write_batch_completed batch_index=%s total_batches=%s "
            "batch_size=%s elapsed_seconds=%.3f",
            batch_index,
            total_batches,
            len(batch),
            perf_counter() - batch_started_at,
        )

    stale_ids = old_ids - new_id_set
    if index_generation_id is None and stale_ids:
        try:
            collection.delete(
                ids=sorted(stale_ids),
            )
        except InternalError as exc:
            _raise_chroma_error("清理过期索引", exc)
    return len(new_ids)

def query_chunks(
    user_id: str,
    document_id: str,
    query_embedding: list[float],
    n_results: int = 3,
    index_generation_id: str | None = None,
) -> dict:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    if not document_id:
        raise ValueError("document_id 不能为空")
    if not query_embedding:
        raise ValueError("query_embedding 不能为空")
    if n_results <= 0:
        raise ValueError("n_results 必须大于 0")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))


    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return _empty_query_result()
    except InternalError as exc:
        _raise_chroma_error("读取集合", exc)

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=_build_document_scope_filter(
                user_id,
                [document_id],
                {document_id: index_generation_id},
            ),
        )
    except InternalError as exc:
        _raise_chroma_error("查询", exc)
    return result


def query_chunks_by_documents(
    user_id: str,
    document_ids: list[str],
    query_embedding: list[float],
    n_results: int = 3,
    index_generations: dict[str, str | None] | None = None,
) -> dict:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    if not query_embedding:
        raise ValueError("query_embedding cannot be empty")
    if n_results <= 0:
        raise ValueError("n_results must be greater than 0")

    normalized_document_ids = _normalize_document_ids(document_ids)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return _empty_query_result()
    except InternalError as exc:
        _raise_chroma_error("read collection", exc)

    try:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=_build_document_scope_filter(
                user_id,
                normalized_document_ids,
                index_generations,
            ),
        )
    except InternalError as exc:
        _raise_chroma_error("query knowledge base", exc)


def get_chunks_by_documents(
    user_id: str,
    document_ids: list[str],
    index_generations: dict[str, str | None] | None = None,
) -> list[dict]:
    """Read chunk text and metadata for a user-scoped document set."""
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id cannot be empty")
    normalized_document_ids = _normalize_document_ids(document_ids)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return []
    except InternalError as exc:
        _raise_chroma_error("read collection", exc)

    try:
        result = collection.get(
            where=_build_document_scope_filter(
                user_id,
                normalized_document_ids,
                index_generations,
            ),
            include=["documents", "metadatas"],
        )
    except InternalError as exc:
        _raise_chroma_error("read sparse retrieval corpus", exc)

    return [
        {
            "chunk_id": chunk_id,
            "文本块": document,
            "元数据": metadata,
        }
        for chunk_id, document, metadata in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        )
        if document and document.strip()
    ]


def build_chunk_id(chunk: dict) -> str:
    parts = [
        chunk["user_id"],
        chunk["document_id"],
    ]
    index_generation_id = chunk.get("index_generation_id")
    if index_generation_id:
        parts.append(index_generation_id)
    parts.extend([
        f"page_{chunk['页码']}",
        f"chunk_{chunk['块索引']}",
    ])
    return "_".join(parts)


def get_document_chunk_ids(
    collection,
    user_id: str,
    document_id: str,
) -> set[str]:

    result = collection.get(
    where={
            "$and": [
                {"user_id": user_id},
                {"document_id": document_id},
            ]
        },
        include=[],
    )

    return set(result.get("ids", []))


def delete_index_generation(
    user_id: str,
    document_id: str,
    index_generation_id: str,
) -> int:
    """Delete an inactive or failed index generation by exact scope."""
    user_id = user_id.strip()
    document_id = document_id.strip()
    index_generation_id = index_generation_id.strip()
    if not user_id:
        raise ValueError("user_id 不能为空")
    if not document_id:
        raise ValueError("document_id 不能为空")
    if not index_generation_id:
        raise ValueError("index_generation_id 不能为空")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        return 0
    except InternalError as exc:
        _raise_chroma_error("读取集合", exc)

    try:
        result = collection.get(
            where=_build_document_scope_filter(
                user_id,
                [document_id],
                {document_id: index_generation_id},
            ),
            include=[],
        )
        generation_ids = result.get("ids", [])
        if generation_ids:
            collection.delete(ids=generation_ids)
    except InternalError as exc:
        _raise_chroma_error("清理失败索引代次", exc)
    return len(generation_ids)

def verify_chunk_ids(
    collection,
    expected_ids: set[str],
) -> None:

    result = collection.get(
    ids=sorted(expected_ids),
    include=[],
)

    actual_ids = set(result.get("ids", []))
    missing_ids = expected_ids - actual_ids

    if missing_ids:
        raise RuntimeError(
            "向量数据库写入校验失败，"
            f"缺少 {len(missing_ids)} 个文本块"
        )
