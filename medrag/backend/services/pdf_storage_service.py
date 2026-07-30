from asyncio import to_thread
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import UploadFile
from config import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
)

async def upload_pdf(file: UploadFile) -> dict:
    return await to_thread(
        save_pdf_stream,
        file.filename,
        file.file,
    )

def save_pdf_stream(
    filename: str | None,
    source: BinaryIO,
) -> dict:
    if not filename:
        raise ValueError("文件名不能为空")

    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError("请上传 PDF 文件")
    if MAX_UPLOAD_SIZE_BYTES <= 0:
        raise RuntimeError("MAX_UPLOAD_SIZE_MB 必须大于 0")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOAD_DIR / f".upload_{uuid4().hex}.tmp"
    hasher = sha256()
    total_size = 0
    read_size = 1024 * 1024

    try:
        source.seek(0)
        with temp_path.open("xb") as destination:
            while True:
                chunk = source.read(read_size)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    raise ValueError("上传文件包含无效的二进制数据")

                if total_size == 0:
                    first_chunk = bytes(chunk)
                    if b"%PDF-" not in first_chunk[:1024]:
                        raise ValueError("文件内容不是有效的 PDF")

                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise ValueError(
                        f"PDF 大小不能超过 {MAX_UPLOAD_SIZE_MB}MB"
                    )

                hasher.update(chunk)
                destination.write(chunk)

        if total_size == 0:
            raise ValueError("上传的 PDF 不能为空")

        document_hash = hasher.hexdigest()
        save_path = UPLOAD_DIR / f"{document_hash}_{safe_name}"

        if save_path.exists():
            temp_path.unlink()
        else:
            temp_path.replace(save_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "文件名": safe_name,
        "保存路径": str(save_path),
        "document_hash": document_hash,
    }
