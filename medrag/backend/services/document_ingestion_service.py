from fastapi import HTTPException, UploadFile
from services.user_service import get_existing_user
from services.pdf_parser_service import extract_pdf_pages
from services.pdf_storage_service import upload_pdf
from services.document_service import get_or_create_document
from pathlib import Path



async def build_pdf_pages(user_id: str, file: UploadFile) -> dict:
    file = await upload_pdf(file)  # 调用上传函数，确保文件已保存
    safe_name = file["文件名"]
    save_path = Path(file["保存路径"])
    document_hash = file["document_hash"]

    try:
        user = await get_existing_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get or create the document
    document_info = await get_or_create_document(user_id, safe_name, document_hash)

    try:
        pages = extract_pdf_pages(save_path) 
        for page in pages:
            page["document_id"] = document_info["document_id"]
            page["user_id"] = user_id
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {exc}") from exc
    
    return {
        "文件名": safe_name,
        "总页数": len(pages),
        "每页内容": pages,
        "document_id": document_info["document_id"],
        "user_id": user["user_id"],
        "document_hash": document_info["document_hash"],
        "existing_document": document_info["existing_document"],
    }


async def prepare_document(user_id: str, file: UploadFile) -> dict:
    file = await upload_pdf(file)  # 调用上传函数，确保文件已保存
    safe_name = file["文件名"]
    save_path = Path(file["保存路径"])
    document_hash = file["document_hash"]

    try:
        user = await get_existing_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get or create the document
    document_info = await get_or_create_document(user_id, safe_name, document_hash)

    return {
        "文件名": safe_name,
        "save_path": str(save_path),
        "document_hash": document_hash,
        "document_id": document_info["document_id"],
        "user_id": user["user_id"],
        "existing_document": document_info["existing_document"],
        "document_language": document_info["language"],
    }


def parse_document_pages(save_path, document_info) -> list[dict]:
    try:
        pages = extract_pdf_pages(save_path) 
        for page in pages:
            page["document_id"] = document_info["document_id"]
            page["user_id"] = document_info["user_id"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {exc}") from exc
    
    return pages
