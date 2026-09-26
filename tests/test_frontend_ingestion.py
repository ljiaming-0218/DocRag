from pathlib import Path

from playwright.sync_api import Route, sync_playwright

from test_frontend_smoke import frontend_server, json_response, launch_browser


def login(page):
    page.locator("#usernameInput").fill("ingestion-user")
    page.locator("#passwordInput").fill("password-123")
    page.locator("#authButton").click()
    page.locator("#appView:not(.hidden)").wait_for()


def upload(page, tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")
    page.locator("#pdfFile").set_input_files(str(pdf))
    page.locator("#indexButton").click()
    page.locator(".batch-result-item").first.wait_for()


def make_routes(state):
    def handle(route: Route):
        request = route.request
        url = request.url
        if url.endswith("/auth/login") and request.method == "POST":
            return json_response(route, {
                "access_token": "token-ingestion",
                "user": {
                    "user_id": "user-ingestion",
                    "username": "ingestion-user",
                    "default_user_type": "general",
                },
            })
        if url.endswith("/auth/me"):
            return json_response(route, {
                "user_id": "user-ingestion",
                "username": "ingestion-user",
                "default_user_type": "general",
            })
        if "/users/user-ingestion/documents" in url:
            docs = [{
                "document_id": "doc-1", "filename": "paper.pdf",
                "language": "en", "updated_at": "2026-09-26T10:00:00Z",
            }] if state["status"] == "completed" else []
            return json_response(route, docs)
        if "/knowledge-bases?" in url:
            return json_response(route, [])
        if "/conversations?" in url:
            return json_response(route, [])
        if url.endswith("/conversations") and request.method == "POST":
            return json_response(route, {
                "conversation_id": "conversation-1", "document_id": "doc-1",
                "title": "paper", "user_type": "general",
            }, 201)
        if "/pdf/index" in url:
            return json_response(route, {
                "job_id": "job-1", "document_id": "doc-1",
                "status": "queued", "stage": "queued",
            }, 202)
        if url.endswith("/ingestion-jobs/job-1/retry"):
            state.update(status="queued", stage="queued", progress=0)
            state["retry_calls"] += 1
            return json_response(route, {
                "job_id": "job-1", "document_id": "doc-1",
                "status": "queued", "stage": "queued", "progress": 0,
            })
        if url.endswith("/ingestion-jobs/job-1"):
            state["query_calls"] += 1
            if state.get("query_error"):
                return json_response(route, {"message": "temporary failure"}, 503)
            return json_response(route, {
                "job_id": "job-1", "document_id": "doc-1",
                "status": state["status"], "stage": state["stage"],
                "progress": state["progress"],
                "error_message": "向量服务不可用" if state["status"] == "failed" else None,
            })
        return route.continue_()
    return handle


def test_ingestion_progress_failure_retry_and_timer_cleanup(tmp_path: Path):
    state = {
        "status": "running", "stage": "embedding", "progress": 50,
        "retry_calls": 0, "query_calls": 0,
    }
    with frontend_server() as url, sync_playwright() as playwright:
        browser = launch_browser(playwright)
        page = browser.new_page()
        page.route("**/*", make_routes(state))
        page.goto(url, wait_until="networkidle")
        login(page)
        upload(page, tmp_path)
        page.get_by_text("正在生成向量", exact=True).wait_for()
        assert page.locator(".ingestion-track").get_attribute("aria-valuenow") == "50"

        state.update(status="failed", stage="embedding", progress=50)
        page.evaluate("() => pollUploadJob(uploadJobs.get('job-1'))")
        page.get_by_text("向量服务不可用", exact=True).wait_for()
        assert page.evaluate("() => uploadJobs.get('job-1').timer === null")
        assert page.evaluate("() => localStorage.getItem('docrag.ingestionJobs.user-ingestion')") is None

        page.get_by_role("button", name="重新处理").click()
        page.get_by_text("等待处理", exact=True).wait_for()
        assert state["retry_calls"] == 1
        state.update(status="completed", stage="completed", progress=100)
        page.evaluate("() => pollUploadJob(uploadJobs.get('job-1'))")
        page.get_by_text("文档处理完成", exact=True).wait_for()
        assert page.evaluate("() => uploadJobs.get('job-1').timer === null")
        browser.close()


def test_ingestion_refresh_restores_pending_job(tmp_path: Path):
    state = {
        "status": "queued", "stage": "queued", "progress": 0,
        "retry_calls": 0, "query_calls": 0,
    }
    with frontend_server() as url, sync_playwright() as playwright:
        browser = launch_browser(playwright)
        page = browser.new_page()
        page.route("**/*", make_routes(state))
        page.goto(url, wait_until="networkidle")
        login(page)
        upload(page, tmp_path)
        page.get_by_text("等待处理", exact=True).wait_for()
        assert page.evaluate("() => JSON.parse(localStorage.getItem('docrag.ingestionJobs.user-ingestion'))[0].jobId") == "job-1"

        state.update(status="running", stage="chunking", progress=25)
        page.reload(wait_until="networkidle")
        page.get_by_text("正在切分文档", exact=True).wait_for()
        assert page.locator(".ingestion-track").get_attribute("aria-valuenow") == "25"
        state.update(status="completed", stage="completed", progress=100)
        page.evaluate("() => pollUploadJob(uploadJobs.get('job-1'))")
        page.get_by_text("文档处理完成", exact=True).wait_for()
        assert page.evaluate("() => localStorage.getItem('docrag.ingestionJobs.user-ingestion')") is None
        browser.close()


def test_transient_query_errors_do_not_mark_job_failed(tmp_path: Path):
    state = {
        "status": "running", "stage": "extracting", "progress": 5,
        "retry_calls": 0, "query_calls": 0, "query_error": True,
    }
    with frontend_server() as url, sync_playwright() as playwright:
        browser = launch_browser(playwright)
        page = browser.new_page()
        page.route("**/*", make_routes(state))
        page.goto(url, wait_until="networkidle")
        login(page)
        upload(page, tmp_path)
        page.wait_for_function("() => uploadJobs.get('job-1')?.queryFailures === 1")
        page.evaluate("() => pollUploadJob(uploadJobs.get('job-1'))")
        page.evaluate("() => pollUploadJob(uploadJobs.get('job-1'))")
        page.get_by_role("button", name="重新查询").wait_for()
        assert page.get_by_role("button", name="重新处理").count() == 0
        assert page.evaluate("() => uploadJobs.get('job-1').timer === null")

        state["query_error"] = False
        page.get_by_role("button", name="重新查询").click()
        page.get_by_text("正在解析文档", exact=True).wait_for()
        page.locator("#logoutButton").click()
        assert page.evaluate("() => uploadJobs.size") == 0
        browser.close()
