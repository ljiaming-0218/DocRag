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
        page.get_by_text("检测到该文件已索引", exact=False).wait_for()
        page.locator("[data-conversation-id='conversation-smoke']").click()

        page.locator("#answerQuery").fill("它如何使用证据？")
        page.locator("#answerButton").click()
        page.get_by_text("RAG 使用检索证据回答问题。", exact=True).wait_for()
        page.locator(".sources-panel summary").click()
        page.get_by_text("RAG 通过检索外部证据辅助生成。", exact=True).wait_for()

        assert index_calls == 2
        assert errors == []
        browser.close()
