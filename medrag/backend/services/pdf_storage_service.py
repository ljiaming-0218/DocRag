from hashlib import sha256
from pathlib import Path


from fastapi import HTTPException, UploadFile
from config import UPLOAD_DIR

async def upload_pdf(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")
    pdf_path = Path(file.filename)


    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = pdf_path.name
    

    content = await file.read()

    document_hash = sha256(content).hexdigest()
    save_path = UPLOAD_DIR / f"{document_hash}_{safe_name}"
    save_path.write_bytes(content)
    
    return {
        "文件名": safe_name,
        "保存路径": str(save_path),
        "document_hash": document_hash
    }
