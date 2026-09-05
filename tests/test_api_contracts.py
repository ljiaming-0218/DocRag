from unittest.mock import ANY, AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from api_errors import (
    APIError,
    api_error_handler,
    request_validation_error_handler,
)
from routers import (
    conversation_router,
    pdf_router,
    user_router,
)
from services.llm_service import LLMServiceError

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(user_router.router)
    app.include_router(conversation_router.router)
    app.include_router(pdf_router.router)
    app.add_exception_handler(
        APIError,
        api_error_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_create_user_returns_user_contract(
    client: TestClient,
    monkeypatch,
):
    create_user = AsyncMock(return_value={
        "user_id": "user-1",
        "username": "test-user",
        "default_user_type": "general",
        "created": True,
    })
    monkeypatch.setattr(
        user_router,
        "create_or_get_user",
        create_user,
    )

    response = client.post(
        "/users",
        json={
            "username": "test-user",
            "default_user_type": "general",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "user-1",
        "username": "test-user",
        "default_user_type": "general",
        "created": True,
    }
    create_user.assert_awaited_once_with(
        "test-user",
        "general",
    )


def test_create_user_rejects_empty_username(
    client: TestClient,
    monkeypatch,
):
    create_user = AsyncMock()
    monkeypatch.setattr(
        user_router,
        "create_or_get_user",
        create_user,
    )

    response = client.post(
        "/users",
        json={
            "username": "",
            "default_user_type": "general",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": "VALIDATION_ERROR",
        "message": "请求参数校验失败",
    }
    create_user.assert_not_awaited()


def test_create_user_maps_invalid_request_to_error_contract(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        user_router,
        "create_or_get_user",
        AsyncMock(side_effect=ValueError("非法的用户类型")),
    )

    response = client.post(
        "/users",
        json={
            "username": "test-user",
            "default_user_type": "invalid",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "INVALID_USER_REQUEST",
        "message": "非法的用户类型",
    }


def test_list_user_documents_returns_document_contract(
    client: TestClient,
    monkeypatch,
):
    list_documents = AsyncMock(return_value=[
        {
            "document_id": "document-1",
            "user_id": "user-1",
            "filename": "rag-paper.pdf",
            "document_hash": "hash-1",
            "language": "en",
            "created_at": "2026-07-30T10:00:00Z",
            "updated_at": "2026-07-30T11:00:00Z",
        }
    ])
    monkeypatch.setattr(
        user_router,
        "list_user_documents",
        list_documents,
    )

    response = client.get(
        "/users/user-1/documents",
        params={"limit": 20},
    )

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "document-1"
    assert "_id" not in response.json()[0]
    list_documents.assert_awaited_once_with(
        "user-1",
        20,
    )


def test_list_user_documents_maps_missing_user_to_404(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        user_router,
        "list_user_documents",
        AsyncMock(side_effect=ValueError("用户不存在")),
    )

    response = client.get(
        "/users/user-404/documents",
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": "USER_NOT_FOUND",
        "message": "用户不存在",
    }


def test_list_messages_maps_permission_error_to_403(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "list_messages",
        AsyncMock(
            side_effect=PermissionError(
                "当前用户无权访问该会话"
            )
        ),
    )

    response = client.get(
        "/conversations/conversation-1/messages",
        params={"user_id": "user-2"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": "CONVERSATION_FORBIDDEN",
        "message": "当前用户无权访问该会话",
    }


def test_list_messages_maps_missing_conversation_to_404(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "list_messages",
        AsyncMock(side_effect=LookupError("会话不存在")),
    )

    response = client.get(
        "/conversations/conversation-404/messages",
        params={"user_id": "user-1"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": "CONVERSATION_NOT_FOUND",
        "message": "会话不存在",
    }


def test_ask_passes_user_type_to_chat_service(
    client: TestClient,
    monkeypatch,
):
    ask_conversation = AsyncMock(return_value={
        "conversation_id": "conversation-1",
        "user_type": "developer",
        "answer": "实现时需要拆分检索和生成流程。",
        "sources": [],
        "sources_count": 0,
        "rewritten_query": "如何实现该方法？",
        "retrieval_queries": ["如何实现该方法？"],
        "assistant_message": {},
        "task_type": "qa",
    })
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        ask_conversation,
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-1",
            "query": "如何实现该方法？",
            "history_limit": 6,
            "n_results": 3,
            "user_type": "developer",
        },
    )

    assert response.status_code == 200
    assert response.json()["user_type"] == "developer"
    ask_conversation.assert_awaited_once_with(
        "user-1",
        "conversation-1",
        "如何实现该方法？",
        6,
        3,
        "developer",
    )


def test_ask_maps_permission_error_to_403(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        AsyncMock(
            side_effect=PermissionError(
                "当前用户无权访问该会话"
            )
        ),
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-2",
            "query": "总结这篇文档",
        },
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": "CONVERSATION_FORBIDDEN",
        "message": "当前用户无权访问该会话",
    }


def test_ask_maps_invalid_request_to_400(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        AsyncMock(side_effect=ValueError("非法的 user_type")),
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-1",
            "query": "总结这篇文档",
            "user_type": "invalid",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "INVALID_CONVERSATION_REQUEST",
        "message": "非法的 user_type",
    }


def test_ask_maps_document_not_ready_to_409(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        AsyncMock(
            side_effect=conversation_router.DocumentNotReadyError(
                "document-1",
                "processing",
                "embedding",
                "文档索引尚未完成，当前阶段: embedding，请稍后再试",
            )
        ),
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-1",
            "query": "总结这篇文档",
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "error": "DOCUMENT_NOT_READY",
        "message": "文档索引尚未完成，当前阶段: embedding，请稍后再试",
    }


def test_ask_maps_validation_error_to_422(
    client: TestClient,
    monkeypatch,
):
    ask_conversation = AsyncMock()
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        ask_conversation,
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-1",
            "query": "总结这篇文档",
            "n_results": 0,
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": "VALIDATION_ERROR",
        "message": "请求参数校验失败",
    }
    ask_conversation.assert_not_awaited()


def test_ask_maps_llm_rate_limit_to_503(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        conversation_router,
        "ask_conversation",
        AsyncMock(
            side_effect=LLMServiceError(
                "LLM_RATE_LIMITED",
                "大模型服务繁忙，请稍后重试",
                503,
            )
        ),
    )

    response = client.post(
        "/conversations/conversation-1/ask",
        json={
            "user_id": "user-1",
            "query": "总结这篇文档",
            "history_limit": 6,
            "n_results": 3,
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": "LLM_RATE_LIMITED",
        "message": "大模型服务繁忙，请稍后重试",
    }


def test_index_pdf_maps_validation_error_to_400(
    client: TestClient,
    monkeypatch,
):
    index_document = AsyncMock(
        side_effect=ValueError(
            "文件内容不是有效的 PDF"
        )
    )
    monkeypatch.setattr(
        pdf_router,
        "index_document",
        index_document,
    )

    response = client.post(
        "/pdf/index",
        params={
            "user_id": "user-1",
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        files={
            "file": (
                "invalid.pdf",
                b"not a real pdf",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "INVALID_PDF_REQUEST",
        "message": "文件内容不是有效的 PDF",
    }
    index_document.assert_awaited_once()


def test_index_pdf_forwards_chunk_strategy(
    client: TestClient,
    monkeypatch,
):
    index_document = AsyncMock(return_value={
        "document_id": "document-1",
        "chunk_strategy": "recursive",
    })
    monkeypatch.setattr(
        pdf_router,
        "index_document",
        index_document,
    )

    response = client.post(
        "/pdf/index",
        params={
            "user_id": "user-1",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "strategy": "recursive",
            "force_reindex": "true",
        },
        files={
            "file": (
                "paper.pdf",
                b"%PDF-1.7",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    index_document.assert_awaited_once_with(
        user_id="user-1",
        file=ANY,
        chunk_size=500,
        chunk_overlap=50,
        strategy="recursive",
        force_reindex=True,
    )


def test_index_pdf_maps_runtime_error_to_500(
    client: TestClient,
    monkeypatch,
):
    index_document = AsyncMock(
        side_effect=RuntimeError("向量数据库写入失败")
    )
    monkeypatch.setattr(
        pdf_router,
        "index_document",
        index_document,
    )

    response = client.post(
        "/pdf/index",
        params={
            "user_id": "user-1",
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        files={
            "file": (
                "paper.pdf",
                b"%PDF-1.7",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "PDF_OPERATION_FAILED",
        "message": "向量数据库写入失败",
    }
    index_document.assert_awaited_once()


def test_parse_pdf_returns_non_persistent_preview(
    client: TestClient,
    monkeypatch,
):
    preview = AsyncMock(return_value={
        "文件名": "paper.pdf",
        "总页数": 1,
        "每页内容": [{"页码": 1, "文本": "content"}],
        "document_id": None,
        "user_id": "user-1",
        "document_hash": "hash-1",
        "persisted": False,
    })
    monkeypatch.setattr(pdf_router, "preview_pdf_pages", preview)

    response = client.post(
        "/pdf/parse",
        params={"user_id": "user-1"},
        files={
            "file": (
                "paper.pdf",
                b"%PDF-1.7",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["document_id"] is None
    assert response.json()["persisted"] is False
    preview.assert_awaited_once()


def test_chunk_preview_does_not_create_persistent_document(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(pdf_router, "validate_chunk_params", Mock())
    monkeypatch.setattr(
        pdf_router,
        "preview_pdf_pages",
        AsyncMock(return_value={
            "文件名": "paper.pdf",
            "总页数": 1,
            "每页内容": [{
                "页码": 1,
                "文本": "content",
                "document_id": None,
                "user_id": "user-1",
            }],
            "document_id": None,
            "user_id": "user-1",
            "persisted": False,
        }),
    )
    monkeypatch.setattr(
        pdf_router,
        "split_pages",
        Mock(return_value=[{
            "页码": 1,
            "文本块": "content",
            "document_id": None,
        }]),
    )

    response = client.post(
        "/pdf/chunks",
        params={
            "user_id": "user-1",
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        files={
            "file": (
                "paper.pdf",
                b"%PDF-1.7",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["document_id"] is None
    assert response.json()["persisted"] is False


def test_pdf_answer_preserves_llm_error_contract(
    client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(
        pdf_router,
        "get_active_index_generation",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        pdf_router,
        "search_relevant_chunks",
        lambda *args: [],
    )
    monkeypatch.setattr(
        pdf_router,
        "build_rag_prompt",
        lambda *args: "prompt",
    )
    monkeypatch.setattr(
        pdf_router,
        "generate_answer",
        lambda *args: (_ for _ in ()).throw(
            LLMServiceError(
                "LLM_TIMEOUT",
                "大模型请求超时",
                504,
            )
        ),
    )

    response = client.post(
        "/pdf/answer",
        params={
            "user_id": "user-1",
            "document_id": "document-1",
            "query": "总结文档",
            "n_results": 3,
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "error": "LLM_TIMEOUT",
        "message": "大模型请求超时",
    }
