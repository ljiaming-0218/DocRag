from unittest.mock import Mock

import pytest
import requests

from eval.runners import hiv_acceptance as acceptance
from eval.runners.hiv_acceptance import AcceptanceClient


def test_acceptance_client_records_network_error(monkeypatch):
    client = AcceptanceClient("https://example.com")
    monkeypatch.setattr(
        client.session,
        "request",
        Mock(
            side_effect=requests.exceptions.ProxyError(
                "proxy disconnected"
            )
        ),
    )

    result = client.request("POST", "/ask")

    assert result["status_code"] == 0
    assert result["payload"]["error_type"] == "ProxyError"
    assert "proxy disconnected" in result["payload"]["message"]


def test_acceptance_client_returns_json_response(monkeypatch):
    client = AcceptanceClient("https://example.com")
    response = Mock(status_code=200)
    response.json.return_value = {"status": "ok"}
    monkeypatch.setattr(
        client.session,
        "request",
        Mock(return_value=response),
    )

    result = client.request("GET", "/health")

    assert result["status_code"] == 200
    assert result["payload"] == {"status": "ok"}


def test_find_documents_accepts_identical_upload_copies(
    monkeypatch,
    tmp_path,
):
    original = tmp_path / "paper.pdf"
    copied = tmp_path / ("hash_" * 3 + "paper.pdf")
    original.write_bytes(b"same pdf content")
    copied.write_bytes(b"same pdf content")
    monkeypatch.setattr(acceptance, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        acceptance,
        "HIV_DOCUMENT_PATTERNS",
        {"paper": "*paper.pdf"},
    )

    result = acceptance.find_documents()

    assert result == {"paper": original}


def test_find_documents_rejects_different_matching_pdfs(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "first-paper.pdf").write_bytes(b"first")
    (tmp_path / "second-paper.pdf").write_bytes(b"second")
    monkeypatch.setattr(acceptance, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        acceptance,
        "HIV_DOCUMENT_PATTERNS",
        {"paper": "*paper.pdf"},
    )

    with pytest.raises(RuntimeError, match="Expected one unique PDF"):
        acceptance.find_documents()
