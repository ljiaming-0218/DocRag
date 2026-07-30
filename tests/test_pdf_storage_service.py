from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from services import pdf_storage_service as service


VALID_PDF = b"%PDF-1.4\nminimal pdf content"


def configure_storage(
    monkeypatch,
    tmp_path: Path,
    *,
    max_size: int = 1024,
) -> None:
    monkeypatch.setattr(service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(service, "MAX_UPLOAD_SIZE_BYTES", max_size)
    monkeypatch.setattr(service, "MAX_UPLOAD_SIZE_MB", 1)


def test_valid_pdf_is_saved(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)

    result = service.save_pdf_stream(
        "paper.pdf",
        BytesIO(VALID_PDF),
    )

    expected_hash = sha256(VALID_PDF).hexdigest()
    saved_path = Path(result["保存路径"])
    assert result["文件名"] == "paper.pdf"
    assert result["document_hash"] == expected_hash
    assert saved_path.name == f"{expected_hash}_paper.pdf"
    assert saved_path.read_bytes() == VALID_PDF


def test_non_pdf_extension_is_rejected(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="请上传 PDF 文件"):
        service.save_pdf_stream(
            "paper.txt",
            BytesIO(VALID_PDF),
        )

    assert list(tmp_path.iterdir()) == []


def test_fake_pdf_content_is_rejected(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="不是有效的 PDF"):
        service.save_pdf_stream(
            "paper.pdf",
            BytesIO(b"plain text"),
        )

    assert list(tmp_path.iterdir()) == []


def test_empty_pdf_is_rejected(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="PDF 不能为空"):
        service.save_pdf_stream(
            "paper.pdf",
            BytesIO(b""),
        )

    assert list(tmp_path.iterdir()) == []


def test_pdf_at_size_limit_is_allowed(monkeypatch, tmp_path):
    configure_storage(
        monkeypatch,
        tmp_path,
        max_size=len(VALID_PDF),
    )

    result = service.save_pdf_stream(
        "paper.pdf",
        BytesIO(VALID_PDF),
    )

    assert Path(result["保存路径"]).exists()


def test_oversized_pdf_is_rejected_and_temp_file_removed(
    monkeypatch,
    tmp_path,
):
    configure_storage(
        monkeypatch,
        tmp_path,
        max_size=len(VALID_PDF) - 1,
    )

    with pytest.raises(ValueError, match="PDF 大小不能超过"):
        service.save_pdf_stream(
            "paper.pdf",
            BytesIO(VALID_PDF),
        )

    assert list(tmp_path.iterdir()) == []


def test_filename_is_sanitized(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)

    result = service.save_pdf_stream(
        "../../paper.pdf",
        BytesIO(VALID_PDF),
    )

    saved_path = Path(result["保存路径"])
    assert result["文件名"] == "paper.pdf"
    assert saved_path.parent == tmp_path


def test_duplicate_file_reuses_saved_path(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)

    first = service.save_pdf_stream(
        "paper.pdf",
        BytesIO(VALID_PDF),
    )
    second = service.save_pdf_stream(
        "paper.pdf",
        BytesIO(VALID_PDF),
    )

    assert first == second
    assert len(list(tmp_path.glob("*.pdf"))) == 1
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_upload_pdf_uses_stream_storage(monkeypatch, tmp_path):
    configure_storage(monkeypatch, tmp_path)
    upload = UploadFile(
        filename="paper.pdf",
        file=BytesIO(VALID_PDF),
    )

    result = await service.upload_pdf(upload)

    assert result["document_hash"] == sha256(VALID_PDF).hexdigest()
    assert Path(result["保存路径"]).read_bytes() == VALID_PDF
