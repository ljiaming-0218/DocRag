

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
        collection.delete(where={"$and": [
                    {"user_id": user_id},
                    {"document_id": document_id}
                ]})
    except InternalError as exc:
        _raise_chroma_error("删除旧索引", exc)
                
    ids =  [f"{chunk['user_id']}_{chunk['document_id']}_page_{chunk['页码']}_chunk_{chunk['块索引']}" for chunk in chunks]  
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
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except InternalError as exc:
        _raise_chroma_error("写入索引", exc)
    
    return len(chunks)


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
