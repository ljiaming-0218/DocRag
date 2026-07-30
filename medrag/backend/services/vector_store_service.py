

import chromadb
from chromadb.errors import InternalError, NotFoundError

from config import CHROMA_DIR

COLLECTION_NAME = "medical_chunks"


def _empty_query_result() -> dict:
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _raise_chroma_error(operation: str, error: Exception) -> None:
    raise RuntimeError(
        f"向量数据库{operation}失败，可能是本地 Chroma 索引损坏。"
        f"请备份并重建 {CHROMA_DIR} 后重新索引 PDF。原始错误: {error}"
    ) from error

def has_chunks(user_id: str, document_id: str) -> bool:
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
            where={
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id},
                ]
            },
            limit=1,
        )
    except InternalError as exc:
        _raise_chroma_error("检查文档索引", exc)

    return bool(result.get("ids"))


def save_chunks(chunks:list[dict]) -> int:
    if not chunks:
        return 0
    
    document_id = chunks[0]["document_id"]
    user_id = chunks[0]["user_id"]
    for chunk in chunks:
        if chunk["document_id"] != document_id:
            raise ValueError(f"所有块必须具有相同的 document_id，发现不匹配的 document_id: {chunk['document_id']}")
        if chunk["user_id"] != user_id:
            raise ValueError(f"所有块必须具有相同的 user_id，发现不匹配的 user_id: {chunk['user_id']}")

            
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except InternalError as exc:
        _raise_chroma_error("初始化", exc)

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

    documents = [chunk["文本块"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    metadatas = [
        {
            "page_number": chunk["页码"],
            "chunk_index": chunk["块索引"],
            "document_id": chunk["document_id"],
            "user_id": chunk["user_id"],
            "extraction_method": chunk.get("提取方式", "text"),
            "image_count": chunk.get("图片数量", 0),
        }
        for chunk in chunks
    ]
    try:
        collection.upsert(
            ids=new_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except InternalError as exc:
        _raise_chroma_error("写入索引", exc)

    verify_chunk_ids(collection, new_id_set)

    stale_ids = old_ids - new_id_set
    if stale_ids:
        try:
            collection.delete(
                ids=sorted(stale_ids),
            )
        except InternalError as exc:
            _raise_chroma_error("清理过期索引", exc)
    return len(new_ids)


def query_chunks(user_id: str, document_id: str, query_embedding: list[float], n_results: int = 3) -> dict:
    if not query_embedding:
        raise ValueError("query_embedding 不能为空")
    if n_results <= 0:
        raise ValueError("n_results 必须大于 0")
    if not document_id:
        raise ValueError("document_id 不能为空")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if not user_id:
        raise ValueError("user_id 不能为空")


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
            where={
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id}
                ]
            }
        )
    except InternalError as exc:
        _raise_chroma_error("查询", exc)
    return result


def build_chunk_id(chunk: dict) -> str:
    return (
        f"{chunk['user_id']}_"
        f"{chunk['document_id']}_"
        f"page_{chunk['页码']}_"
        f"chunk_{chunk['块索引']}"
    )


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