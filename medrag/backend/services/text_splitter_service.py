
def split_text(text: str, chunk_size, chunk_overlap) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 必须大于或等于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def split_pages(pages: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    all_chunks = []
    chunk_index = 0
    for page in pages:
        text = page["文本"]
        index = page["页码"]
        chunks = split_text(text, chunk_size, chunk_overlap)
        document_id = page["document_id"]
        user_id = page["user_id"]
        extraction_method = page.get("提取方式", "text")
        image_count = page.get("图片数量", 0)
        for chunk in chunks:
            chunk_index += 1
            all_chunks.append({
                "页码": index,
                "文本块": chunk,
                "块索引": chunk_index,
                "document_id": document_id,
                "user_id": user_id,
                "提取方式": extraction_method,
                "图片数量": image_count,
            })
    return all_chunks
