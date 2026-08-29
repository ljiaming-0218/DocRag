import json
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import Route, sync_playwright


FRONTEND_DIR = Path(__file__).parents[1] / "medrag" / "frontend"
EDGE_PATH = Path(
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


@contextmanager
def frontend_server():
    handler = lambda *args, **kwargs: QuietHandler(
        *args,
        directory=str(FRONTEND_DIR),
        **kwargs,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def json_response(route: Route, data, status: int = 200):
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(data, ensure_ascii=False),
    )


def test_frontend_complete_smoke(tmp_path: Path):
    assert EDGE_PATH.exists(), "本机未找到 Microsoft Edge"

    errors = []
    index_calls = 0
    document = {
        "document_id": "document-smoke",
        "filename": "smoke.pdf",
        "language": "zh",
        "updated_at": "2026-08-01T10:00:00Z",
    }
    conversation = {
        "conversation_id": "conversation-smoke",
        "document_id": "document-smoke",
        "title": "smoke",
        "user_type": "general",
        "updated_at": "2026-08-01T10:00:00Z",
    }
    source = {
        "文本块": "RAG 通过检索外部证据辅助生成。",
        "距离": 0.2,
        "元数据": {"page_number": 1, "chunk_index": 0},
    }

    def handle(route: Route):
        nonlocal index_calls
        request = route.request
        url = request.url

        if url.endswith("/users") and request.method == "POST":
            return json_response(route, {
                "user_id": "user-smoke",
                "username": "smoke-user",
                "default_user_type": "general",
                "created": True,
            })
        if "/users/user-smoke/documents" in url:
            return json_response(route, [document] if index_calls else [])
        if "/knowledge-bases?" in url and request.method == "GET":
            return json_response(route, [])
        if "/pdf/index" in url:
            index_calls += 1
            return json_response(route, {
                "document_id": "document-smoke",
                "document_hash": "hash-smoke",
                "existing_document": index_calls > 1,
                "conversations": [conversation] if index_calls > 1 else [],
                "文件名": "smoke.pdf",
                "总页数": 1,
                "总块数": 1,
                "成功保存块数": 1,
                "message": "PDF 索引完成。",
            })
        if url.endswith("/conversations") and request.method == "POST":
            return json_response(route, conversation, 201)
        if "/conversations?" in url:
            return json_response(route, [conversation] if index_calls else [])
        if url.endswith("/messages?user_id=user-smoke"):
            return json_response(route, [])
        if url.endswith("/ask"):
            return json_response(route, {
                "conversation_id": "conversation-smoke",
                "answer": "RAG 使用检索证据回答问题。",
                "sources": [source],
                "sources_count": 1,
                "task_type": "qa",
                "rewritten_query": "RAG 如何使用检索证据？",
            })
        return route.continue_()

    with frontend_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE_PATH),
            headless=True,
        )
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.route("**/*", handle)
        page.goto(base_url, wait_until="networkidle")

        page.locator("#usernameInput").fill("smoke-user")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()

        pdf_path = tmp_path / "smoke.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        page.locator("#pdfFile").set_input_files(str(pdf_path))
        page.locator("#indexButton").click()
        page.locator("[data-conversation-id='conversation-smoke']").wait_for()

        page.locator("#indexButton").click()
        page.wait_for_function(
            "document.getElementById('indexButton').disabled === false"
        )
        page.locator("[data-conversation-id='conversation-smoke']").click()

        page.locator("#answerQuery").fill("它如何使用证据？")
        page.locator("#answerButton").click()
        page.get_by_text("RAG 使用检索证据回答问题。", exact=True).wait_for()
        page.locator(".sources-panel summary").click()
        page.get_by_text("RAG 通过检索外部证据辅助生成。", exact=True).wait_for()

        assert index_calls == 2
        assert errors == []
        browser.close()


def test_frontend_knowledge_base_smoke(tmp_path: Path):
    assert EDGE_PATH.exists(), "Microsoft Edge is required"

    errors = []
    knowledge_base_created = False
    document_linked = False
    conversation_created = False
    selected_scope_created = False
    knowledge_base = {
        "kb_id": "kb-smoke",
        "user_id": "user-smoke",
        "name": "RAG Papers",
        "description": "multi-document test",
        "created_at": "2026-08-28T10:00:00Z",
        "updated_at": "2026-08-28T10:00:00Z",
    }
    document = {
        "document_id": "document-smoke",
        "user_id": "user-smoke",
        "kb_id": "kb-smoke",
        "filename": "rag.pdf",
        "document_hash": "hash-smoke",
        "language": "en",
        "added_at": "2026-08-28T10:00:00Z",
    }
    conversation = {
        "conversation_id": "conversation-kb-smoke",
        "document_id": None,
        "kb_id": "kb-smoke",
        "selected_document_ids": None,
        "title": "RAG Papers",
        "user_type": "general",
        "updated_at": "2026-08-28T10:00:00Z",
    }

    def handle(route: Route):
        nonlocal knowledge_base_created
        nonlocal document_linked
        nonlocal conversation_created
        nonlocal selected_scope_created
        request = route.request
        url = request.url

        if url.endswith("/users") and request.method == "POST":
            return json_response(route, {
                "user_id": "user-smoke",
                "username": "smoke-user",
                "default_user_type": "general",
                "created": True,
            })
        if "/users/user-smoke/documents" in url:
            return json_response(route, [document] if document_linked else [])
        if url.endswith("/knowledge-bases") and request.method == "POST":
            knowledge_base_created = True
            return json_response(route, knowledge_base, 201)
        if "/knowledge-bases?" in url and request.method == "GET":
            return json_response(
                route,
                [knowledge_base] if knowledge_base_created else [],
            )
        if "/knowledge-bases/kb-smoke/documents?" in url:
            return json_response(route, [document] if document_linked else [])
        if (
            url.endswith(
                "/knowledge-bases/kb-smoke/documents/document-smoke"
            )
            and request.method == "POST"
        ):
            document_linked = True
            return json_response(route, {
                "kb_id": "kb-smoke",
                "document_id": "document-smoke",
                "filename": "rag.pdf",
                "linked": True,
                "knowledge_base_name": "RAG Papers",
            })
        if "/pdf/index" in url:
            return json_response(route, {
                "document_id": "document-smoke",
                "document_hash": "hash-smoke",
                "existing_document": False,
                "conversations": [],
                "文件名": "rag.pdf",
                "总页数": 2,
                "总块数": 4,
                "成功保存块数": 4,
                "message": "PDF 索引完成。",
            })
        if url.endswith("/conversations") and request.method == "POST":
            payload = request.post_data_json
            assert payload["kb_id"] == "kb-smoke"
            assert payload.get("document_id") is None
            if payload.get("selected_document_ids") == ["document-smoke"]:
                selected_scope_created = True
            conversation_created = True
            return json_response(route, conversation, 201)
        if "/conversations?" in url:
            return json_response(
                route,
                [conversation] if conversation_created else [],
            )
        if url.endswith("/ask"):
            return json_response(route, {
                "conversation_id": "conversation-kb-smoke",
                "kb_id": "kb-smoke",
                "knowledge_base_name": "RAG Papers",
                "document_ids": ["document-smoke"],
                "answer": "The knowledge base contains RAG evidence.",
                "sources": [{
                    "文本块": "RAG retrieves evidence before generation.",
                    "距离": 0.1,
                    "元数据": {
                        "filename": "rag.pdf",
                        "document_id": "document-smoke",
                        "page_number": 1,
                        "chunk_index": 0,
                    },
                }],
                "sources_count": 1,
                "task_type": "qa",
            })
        return route.continue_()

    with frontend_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE_PATH),
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.route("**/*", handle)
        page.goto(base_url, wait_until="networkidle")

        page.locator("#usernameInput").fill("smoke-user")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()

        page.locator("#knowledgeBaseName").fill("RAG Papers")
        page.locator("#knowledgeBaseDescription").fill(
            "multi-document test"
        )
        page.locator("#createKnowledgeBaseButton").click()
        page.locator("[data-kb-id='kb-smoke']").wait_for()

        pdf_path = tmp_path / "rag.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        page.locator("#pdfFile").set_input_files(str(pdf_path))
        page.locator("#uploadKnowledgeBaseSelect").select_option("kb-smoke")
        page.locator("#indexButton").click()
        page.locator(
            "[data-conversation-id='conversation-kb-smoke']"
        ).wait_for()

        assert page.locator("#answerConversationId").input_value() == (
            "conversation-kb-smoke"
        )
        page.locator("#answerQuery").fill("What does it contain?")
        page.locator("#answerButton").click()
        page.get_by_text(
            "The knowledge base contains RAG evidence.",
            exact=True,
        ).wait_for()
        page.locator(".sources-panel summary").click()
        page.locator(".sources-panel .chunk-title").get_by_text(
            "rag.pdf",
            exact=False,
        ).wait_for()

        page.locator("[data-kb-document-id='document-smoke']").check()
        page.locator("#newKnowledgeBaseConversationButton").click()
        page.wait_for_function(
            "document.getElementById('answerQuery').disabled === false"
        )
        assert page.locator("#answerStatus").text_content() == ""

        assert knowledge_base_created is True
        assert document_linked is True
        assert conversation_created is True
        assert selected_scope_created is True
        assert errors == []
        browser.close()


def test_frontend_user_switch_clears_draft_and_error_state():
    assert EDGE_PATH.exists(), "Microsoft Edge is required"

    errors = []

    def handle(route: Route):
        request = route.request
        url = request.url

        if url.endswith("/users") and request.method == "POST":
            payload = request.post_data_json
            username = payload["username"]
            return json_response(route, {
                "user_id": f"user-{username}",
                "username": username,
                "default_user_type": "general",
                "created": True,
            })
        if "/documents" in url and request.method == "GET":
            return json_response(route, [])
        if "/knowledge-bases?" in url and request.method == "GET":
            return json_response(route, [])
        if "/conversations?" in url and request.method == "GET":
            return json_response(route, [])
        return route.continue_()

    with frontend_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE_PATH),
            headless=True,
        )
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url, wait_until="networkidle")
        page.route("**/*", handle)

        page.locator("#usernameInput").fill("first")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()
        page.evaluate("""
            () => {
              const query = document.getElementById("answerQuery");
              query.value = "上一个用户未发送的问题";
              const status = document.getElementById("answerStatus");
              status.textContent = "大模型服务返回异常";
              status.className = "status show error";
            }
        """)

        page.locator("#logoutButton").click()
        page.locator("#usernameInput").fill("second")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()

        assert page.locator("#activeUserName").text_content() == "second"
        assert page.locator("#answerQuery").input_value() == ""
        assert page.locator("#answerStatus").text_content() == ""
        assert "show" not in (page.locator("#answerStatus").get_attribute("class") or "")
        assert page.locator("#answerConversationId").input_value() == ""
        assert errors == []
        browser.close()


def test_frontend_multi_pdf_upload_is_sequential(tmp_path: Path):
    assert EDGE_PATH.exists(), "Microsoft Edge is required"

    errors = []
    index_calls = 0
    conversation_creations = 0

    def make_document(index: int) -> dict:
        return {
            "document_id": f"document-{index}",
            "filename": f"paper-{index}.pdf",
            "language": "en",
            "updated_at": "2026-08-29T10:00:00Z",
        }

    def handle(route: Route):
        nonlocal index_calls
        nonlocal conversation_creations
        request = route.request
        url = request.url

        if url.endswith("/users") and request.method == "POST":
            return json_response(route, {
                "user_id": "user-batch",
                "username": "batch-user",
                "default_user_type": "general",
                "created": True,
            })
        if "/users/user-batch/documents" in url:
            return json_response(
                route,
                [make_document(index) for index in range(1, index_calls + 1)],
            )
        if "/knowledge-bases?" in url and request.method == "GET":
            return json_response(route, [])
        if "/pdf/index" in url:
            index_calls += 1
            return json_response(route, {
                "document_id": f"document-{index_calls}",
                "document_hash": f"hash-{index_calls}",
                "existing_document": False,
                "conversations": [],
                "文件名": f"paper-{index_calls}.pdf",
                "总页数": 1,
                "总块数": 1,
                "成功保存块数": 1,
                "message": "indexed",
            })
        if url.endswith("/conversations") and request.method == "POST":
            conversation_creations += 1
            return json_response(route, {}, 201)
        if "/conversations?" in url:
            return json_response(route, [])
        return route.continue_()

    with frontend_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE_PATH),
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.route("**/*", handle)
        page.goto(base_url, wait_until="networkidle")

        page.locator("#usernameInput").fill("batch-user")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()

        first_pdf = tmp_path / "paper-1.pdf"
        second_pdf = tmp_path / "paper-2.pdf"
        first_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        second_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        page.locator("#pdfFile").set_input_files(
            [str(first_pdf), str(second_pdf)]
        )

        assert page.locator(".selected-file-item").count() == 2
        page.locator("#indexButton").click()
        page.locator("#uploadProgressCount").get_by_text(
            "2 / 2",
            exact=True,
        ).wait_for()
        page.locator(".batch-result-item").nth(1).wait_for()

        assert index_calls == 2
        assert conversation_creations == 0
        assert errors == []
        browser.close()
