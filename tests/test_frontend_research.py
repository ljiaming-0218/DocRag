"""Research UI contract smoke; API responses are fixtures, not a live model run.

Run without pytest: python tests/test_frontend_research.py
Screenshots are saved outside the repository by default.
"""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from test_frontend_smoke import frontend_server, json_response, launch_browser


def run_smoke(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {"creates": 0, "runs": 0, "gets": 0, "asks": 0,
             "create_error": False, "query_error": False, "missing": False, "run_response_error": False,
             "owner": "ui-user", "tasks": {}, "payloads": [], "hold_queue": False, "cancels": 0,
             "deleted_chat": False, "deleted_research": [], "delete_error": False}
    errors = []
    console_errors = []
    documents = [
        {"document_id": "doc-rag", "filename": "RAG.pdf", "language": "en"},
        {"document_id": "doc-lora", "filename": "LoRA.pdf", "language": "en"},
        {"document_id": "doc-cot", "filename": "Chain-of-Thought.pdf", "language": "en"},
    ]
    filenames = {doc["document_id"]: doc["filename"] for doc in documents}
    conversation = {"conversation_id": "chat-ui", "kb_id": "kb-ui", "title": "问答验收", "user_type": "general"}

    def user():
        return {"user_id": state["owner"], "username": "界面验收用户", "default_user_type": "general"}

    def result_task(task_id):
        task = copy.deepcopy(state["tasks"][task_id])
        task.update(status="partial" if task_id == "task-2" else "completed", model_turns=2, model_actions=2, tool_calls=3)
        if not any(item["evidence_id"] == "e-cot" for item in task["evidence"]):
            task["evidence"].append({"evidence_id": "e-cot", "document_id": "doc-cot", "page_number": 3,
                                     "text": "CoT 使用中间推理步骤。"})
        missing = [doc for doc in task["document_ids"] if not any(e["document_id"] == doc for e in task["evidence"])]
        task["report"] = {
            "title": "检索增强与推理提示的比较", "verification_status": "needs_semantic_review",
            "run_limitations": [{"code": "retrieval_gap", "document_ids": missing}] if missing else [],
            "claims": [
                {"role": "comparison", "status": "supported", "text": "RAG 借助外部文档，CoT 使用中间推理步骤。", "evidence_ids": ["e-rag", "e-cot"]},
                {"role": "limitation", "status": "unknown" if task_id == "task-2" else "supported",
                 "text": "现有证据不足以比较两者的效果。", "evidence_ids": [] if task_id == "task-2" else ["e-rag", "e-cot"]},
            ],
            "question_coverage": {"status": "partial" if task_id == "task-2" else "structurally_complete", "gaps": [
                {"code": "missing_comparison_limitations"},
            ] if task_id == "task-2" else []},
        }
        task["stop_reason"] = "report_incomplete" if task_id == "task-2" else None
        task["events"].append({"actor": "model", "name": "finish_report", "args_summary": {"claim_count": 2},
                               "result": {"claim_count": 2}, "elapsed_ms": 400})
        return task

    def handle(route):
        request = route.request
        path = urlparse(request.url).path
        if path == "/auth/login":
            return json_response(route, {"access_token": "fixture-token", "user": user()})
        if path == "/auth/me":
            return json_response(route, user())
        if path.startswith(("/research-tasks", "/knowledge-bases", "/users/", "/conversations")):
            assert request.headers.get("authorization") == "Bearer fixture-token"
        if path == "/knowledge-bases":
            return json_response(route, [{"kb_id": "kb-ui", "name": "论文集 · 界面验收"}])
        if path == "/knowledge-bases/kb-ui/documents" or path.endswith("/documents"):
            return json_response(route, documents)
        if path == "/conversations":
            return json_response(route, [] if state["deleted_chat"] else [conversation])
        if path == "/conversations/chat-ui" and request.method == "DELETE":
            state["deleted_chat"] = True
            return json_response(route, {"deleted": True})
        if path.endswith("/messages"):
            return json_response(route, [])
        if path.endswith("/ask"):
            state["asks"] += 1
            return json_response(route, {"answer": "原有文档问答仍可使用。", "sources": [], "task_type": "qa"})
        if path == "/research-tasks" and request.method == "POST":
            if state["create_error"]:
                return json_response(route, {"message": "验收用的创建失败"}, 503)
            state["creates"] += 1
            payload = request.post_data_json
            state["payloads"].append(payload)
            task_id = f"task-{state['creates']}"
            task = {
                "task_id": task_id, "kb_id": payload["kb_id"], "query": payload["query"], "status": "ready",
                "created_at": f"2026-10-16T02:{state['creates']:02d}:00Z", "document_ids": payload["selected_document_ids"] or list(filenames),
                "document_filenames": filenames, "model_turns": 0, "model_actions": 0, "tool_calls": 0,
                "events": [], "evidence": [], "report": None,
                "parent_task_id": payload.get("parent_task_id"),
                "parent_query": state["tasks"].get(payload.get("parent_task_id"), {}).get("query"),
            }
            state["tasks"][task_id] = task
            return json_response(route, task, 201)
        if path.startswith("/research-tasks/"):
            task_id = path.split("/")[2]
            task = state["tasks"][task_id]
            if request.method == "DELETE":
                if state["delete_error"]:
                    return json_response(route, {"message": "删除失败验收"}, 503)
                state["deleted_research"].append(task_id)
                del state["tasks"][task_id]
                return json_response(route, {"deleted": True})
            if path.endswith("/cancel"):
                state["cancels"] += 1
                task.update(status="cancelled", stop_reason="user_cancelled")
                return json_response(route, task)
            if path.endswith("/run"):
                state["runs"] += 1
                task.update(status="running", model_turns=1, model_actions=1, tool_calls=1)
                task["events"] = [{"name": "search_documents", "actor": "model", "args_summary": {"query": "RAG 方法"}, "result": {"evidence_ids": ["e-rag"]}, "elapsed_ms": 200}]
                task["evidence"] = [{"evidence_id": "e-rag", "document_id": "doc-rag", "page_number": 2, "text": "RAG 使用检索文档辅助生成。"}]
                if task_id == "task-2":
                    state["tasks"][task_id] = result_task(task_id)
                if state["hold_queue"]:
                    task.update(status="queued", model_turns=0, model_actions=0, tool_calls=0,
                                events=[], evidence=[])
                if state["run_response_error"]:
                    return json_response(route, {"message": "验收用的启动响应失败"}, 503)
                return json_response(route, state["tasks"][task_id], 202)
            state["gets"] += 1
            if state["missing"]:
                return json_response(route, {"message": "Research task not found"}, 404)
            if state["query_error"]:
                return json_response(route, {"message": "验收用的查询失败"}, 503)
            if task["status"] == "running":
                if not any(item["actor"] == "harness" for item in task["events"]):
                    task["events"].append({"name": "search_documents", "actor": "harness", "args_summary": {"query": "比较 RAG 与 CoT", "document_ids": ["doc-cot"]}, "result": {"evidence_ids": ["e-cot"]}, "elapsed_ms": 180})
                    task["evidence"].append({"evidence_id": "e-cot", "document_id": "doc-cot", "page_number": 3, "text": 'CoT 使用中间推理步骤。<img src=x onerror="window.researchInjected=true">'})
                task["tool_calls"] = 2
                if state["gets"] >= 2:
                    state["tasks"][task_id] = result_task(task_id)
            return json_response(route, state["tasks"][task_id])
        return route.continue_()

    with frontend_server() as base_url, sync_playwright() as playwright:
        browser = launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.route("**/*", handle)
        page.goto(base_url, wait_until="networkidle")
        assert "DocRAG" in page.title()
        assert page.locator("#authView").is_visible()
        page.locator("#usernameInput").fill("ui-user")
        page.locator("#passwordInput").fill("fixture-password")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()
        page.locator("#conversationHistoryPanel > summary").click()
        assert page.locator("#conversationList").is_hidden()
        page.locator("#conversationHistoryPanel > summary").click()
        assert page.locator('[data-conversation-id="chat-ui"]').is_visible()
        page.locator("#researchWorkspaceButton").click()
        assert page.locator("#chatWindow").is_hidden()
        assert page.locator("#chatComposer").is_visible()
        assert page.locator("#answerButton").is_disabled()
        assert page.locator("textarea:visible").count() == 1
        assert page.locator("#researchForm, #researchKnowledgeBase").count() == 0
        page.locator('[data-kb-id="kb-ui"]').click()
        page.locator('[data-kb-document-id="doc-cot"]').wait_for()
        page.locator('[data-kb-document-id="doc-rag"]').check()
        page.locator('[data-kb-document-id="doc-cot"]').check()
        page.locator("[data-research-suggestion]").first.click()
        assert "比较所选文档" in page.locator("#answerQuery").input_value()
        page.locator("#answerQuery").fill("比较 RAG 与 CoT 的方法和限制")
        page.locator("#answerButton").click()
        page.locator(".harness-step").first.wait_for(state="attached")
        assert page.locator("#researchTimeline").get_attribute("open") is None
        page.locator("#researchTimeline > summary").click()
        assert state["payloads"][0]["selected_document_ids"] == ["doc-rag", "doc-cot"]
        assert page.locator("#researchTask").get_by_text("Harness 主动补查", exact=True).is_visible()
        page.screenshot(path=str(output_dir / "research-progress.png"), full_page=False)
        page.locator("#chatWorkspaceButton").click()
        get_count = state["gets"]
        page.wait_for_timeout(2300)
        assert state["gets"] == get_count
        page.locator("#researchWorkspaceButton").click()
        page.locator(".research-report").wait_for()
        assert "待核验" in page.locator(".research-report").inner_text()
        assert page.locator(".research-run-limitations").count() == 0
        assert "Chain-of-Thought.pdf" in page.locator("#researchTask").inner_text()
        page.locator(".research-report [data-research-citation='1']").first.click()
        assert page.locator("#researchSourceDrawer").is_visible()
        assert "Chain-of-Thought.pdf" in page.locator("#researchSourceContent").inner_text()
        assert page.locator("#researchSourceContent img").count() == 0
        assert not page.evaluate("Boolean(window.researchInjected)")
        page.keyboard.press("Escape")
        assert page.locator("#researchSourceDrawer").is_hidden()
        page.locator(".research-report [data-research-citation='0']").first.click()
        page.screenshot(path=str(output_dir / "research-source.png"), full_page=False)
        page.locator("#researchSourceClose").click()
        if page.locator("#researchTimeline").get_attribute("open") is not None:
            page.locator("#researchTimeline > summary").click()
        page.locator("#researchView").evaluate("element => element.scrollTop = 0")
        page.screenshot(path=str(output_dir / "research-desktop.png"), full_page=False)

        # Existing chat remains reachable and sends through its original endpoint.
        page.locator('[data-conversation-id="chat-ui"]').click()
        assert page.locator("#researchView").is_hidden()
        assert page.locator("#chatComposer").is_visible()
        page.locator("#answerQuery").fill("继续问答")
        page.locator("#answerButton").click()
        page.get_by_text("原有文档问答仍可使用。", exact=True).wait_for()
        assert state["asks"] == 1

        # Reload retrieves the saved task ID and never restarts its run.
        page.reload(wait_until="networkidle")
        page.locator("#appView:not(.hidden)").wait_for()
        # History is loaded during authentication restoration, while chat stays active.
        page.locator('[data-research-task="task-1"]').wait_for(state="visible")
        assert page.locator("#researchView").is_hidden()
        page.locator("#researchWorkspaceButton").click()
        page.locator(".research-report").wait_for()
        assert state["creates"] == state["runs"] == 1

        # Transient query failures preserve the last known report.
        state["query_error"] = True
        page.locator("[data-research-refresh]").click()
        page.locator("#researchStatus.error").wait_for()
        assert page.locator(".research-report").is_visible()
        state["query_error"] = False
        page.locator("[data-research-refresh]").click()
        page.locator("#researchStatus:not(.show)").wait_for(state="attached")

        # Partial reports show gaps, while failed creation does not start a run.
        page.locator("#newResearchButton").click()
        page.locator('[data-kb-id="kb-ui"]').click()
        page.locator('[data-kb-document-id="doc-cot"]').wait_for()
        page.locator("#answerQuery").fill("比较这些论文的实验限制")
        page.locator("#answerButton").click()
        page.locator('[data-state="partial"]').wait_for()
        assert "本次尚未从以下文档取得证据片段：LoRA.pdf" in page.locator(".research-run-limitations").inner_text()
        assert "这不代表原文没有相关内容" in page.locator(".research-run-limitations").inner_text()
        page.screenshot(path=str(output_dir / "research-run-limitations.png"), full_page=False)
        assert "缺少有跨文档引用的比较限制说明" in page.locator(".research-gaps").inner_text()
        assert page.get_by_text("证据不足", exact=True).is_visible()
        state["create_error"] = True
        page.locator("#answerQuery").fill("比较这些论文的实验限制")
        page.locator("#answerButton").click()
        page.locator("#researchStatus.error").wait_for()
        assert state["creates"] == state["runs"] == 2
        state["create_error"] = False

        # An ambiguous start response is reconciled with GET, never an automatic POST retry.
        state["run_response_error"] = True
        page.locator('[data-kb-document-id="doc-rag"]').check()
        page.locator('[data-kb-document-id="doc-cot"]').check()
        page.locator("#answerButton").click()
        page.locator("#researchStatus.error").wait_for()
        assert "启动结果未确认" in page.locator("#researchStatus").inner_text()
        assert state["runs"] == 3
        state["run_response_error"] = False
        page.locator("[data-research-refresh]").click()
        page.locator(".research-report").wait_for()
        assert state["runs"] == 3

        # H2 queued tasks keep polling and expose an owner-scoped cancellation.
        state["hold_queue"] = True
        page.locator("#answerQuery").fill("查阅所选文档的方法")
        page.locator("#answerButton").click()
        page.locator('[data-state="queued"]').wait_for()
        assert page.locator("#answerButton").is_disabled()
        assert "任务已排队，等待开始" in page.locator("#researchTask").inner_text()
        page.locator('[data-delete-research="task-4"]').click()
        page.locator("#researchStatus.error").wait_for()
        assert "请先取消任务" in page.locator("#researchStatus").inner_text()
        assert not state["deleted_research"]
        page.screenshot(path=str(output_dir / "research-queued.png"), full_page=False)
        page.locator("[data-research-cancel]").click()
        page.locator('[data-state="cancelled"]').wait_for()
        assert page.locator("#answerButton").is_enabled()
        get_count = state["gets"]
        page.wait_for_timeout(2300)
        assert state["gets"] == get_count and state["cancels"] == 1
        state["hold_queue"] = False

        # A missing task permits a new run and keeps references honest.
        state["missing"] = True
        page.locator("[data-research-refresh]").click()
        page.get_by_text("任务暂不可用", exact=True).wait_for()
        assert page.locator("#answerButton").is_enabled()
        state["missing"] = False
        page.locator('[data-research-task="task-1"]').click()
        page.locator(".research-report").wait_for()

        # Reading older records must not reorder history; identical questions
        # remain distinguishable by their creation time and short task ID.
        def history_order():
            return page.locator("#researchHistory [data-research-task]").evaluate_all(
                "items => items.map(item => item.dataset.researchTask)")

        expected_order = ["task-4", "task-3", "task-2", "task-1"]
        assert history_order() == expected_order
        row2 = page.locator('[data-research-task="task-2"]')
        row3 = page.locator('[data-research-task="task-3"]')
        assert row2.locator(".conversation-title").inner_text() == row3.locator(".conversation-title").inner_text()
        assert row2.locator(".research-history-time").inner_text() != row3.locator(".research-history-time").inner_text()
        assert "#task-2" in row2.locator(".research-history-time").inner_text()
        assert row2.get_attribute("title").endswith("任务 task-2")
        if page.locator("#researchTimeline").get_attribute("open") is None:
            page.locator("#researchTimeline > summary").click()
        page.locator("#researchView").evaluate("element => element.scrollTop = 300")
        row2.click()
        page.locator('[data-research-task="task-2"][aria-current="true"]').wait_for()
        assert page.locator("#researchTimeline").get_attribute("open") is None
        assert page.locator("#researchView").evaluate("element => element.scrollTop") == 0
        assert history_order() == expected_order

        # An inactive task cached as running is refreshed from the server when
        # opening research, without replacing the currently selected report.
        page.evaluate("""() => {
          const key = 'docrag.researchTasks.ui-user';
          const refs = JSON.parse(localStorage.getItem(key));
          refs.find(item => item.task_id === 'task-3').status = 'running';
          localStorage.setItem(key, JSON.stringify(refs));
        }""")
        page.locator("#researchWorkspaceButton").click()
        row3.locator(".research-history-status").get_by_text("报告已生成 · 待核验", exact=True).wait_for()
        assert page.locator('[data-research-task="task-2"]').get_attribute("aria-current") == "true"
        assert history_order() == expected_order
        page.locator("#researchHistoryPanel").scroll_into_view_if_needed()
        page.screenshot(path=str(output_dir / "research-history-desktop.png"), full_page=False)
        page.locator('[data-research-task="task-1"]').click()
        page.locator('[data-research-task="task-1"][aria-current="true"]').wait_for()

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_function("document.getElementById('sidebar').getBoundingClientRect().right <= 1")
        page.locator("#researchView").evaluate("element => element.scrollTop = 0")
        assert page.locator("#researchWorkspaceButton").is_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.locator("#mobileMenuButton").click()
        assert "is-open" in page.locator("#sidebar").get_attribute("class")
        page.locator("#researchHistoryPanel").scroll_into_view_if_needed()
        page.screenshot(path=str(output_dir / "research-history-mobile.png"), full_page=False)
        page.locator("#sidebarBackdrop").click(position={"x": 370, "y": 300})
        page.wait_for_function("document.getElementById('sidebar').getBoundingClientRect().right <= 1")
        page.screenshot(path=str(output_dir / "research-mobile.png"), full_page=False)
        page.locator(".research-report").scroll_into_view_if_needed()
        page.locator(".research-report [data-research-citation='1']").first.click()
        assert page.locator("#researchSourceDrawer").is_visible()
        page.locator("#researchSourceClose").click()

        # Logging out clears rendered artifacts; the next user has their own history.
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator("#logoutButton").click()
        page.locator("#authView:not(.hidden)").wait_for()
        assert "检索增强与推理提示的比较" not in page.locator("#researchTask").inner_text()
        page.locator("#usernameInput").fill("ui-user")
        page.locator("#passwordInput").fill("fixture-password")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()
        page.locator('[data-research-task="task-1"]').wait_for(state="visible")
        assert page.locator("#researchView").is_hidden()
        page.locator('[data-research-task="task-1"]').click()
        page.locator(".research-report").wait_for()
        assert page.locator("#answerQuery").is_enabled()
        assert "沿用原范围" in page.locator("#researchScopeHint").inner_text()
        page.locator("#answerQuery").fill("进一步解释前述方法的区别")
        page.locator("#answerButton").click()
        page.locator('[data-research-parent="task-1"]').wait_for()
        page.locator(".research-report").wait_for()
        assert state["payloads"][-1]["parent_task_id"] == "task-1"
        assert state["payloads"][-1]["selected_document_ids"] == state["tasks"]["task-1"]["document_ids"]
        page.screenshot(path=str(output_dir / "research-continuation.png"), full_page=False)
        # Deletion cancellation/failure preserve the record; success removes it.
        page.once("dialog", lambda dialog: dialog.dismiss())
        page.locator('[data-delete-research="task-5"]').click()
        assert page.locator('[data-research-task="task-5"]').count() == 1
        state["delete_error"] = True
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator('[data-delete-research="task-5"]').click()
        page.locator("#researchStatus.error").wait_for()
        assert page.locator('[data-research-task="task-5"]').count() == 1
        state["delete_error"] = False
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator('[data-delete-research="task-5"]').click()
        page.locator('[data-research-task="task-5"]').wait_for(state="detached")
        assert page.locator(".research-welcome").is_visible()
        page.locator('[data-research-task="task-1"]').click()
        page.locator(".research-report").wait_for()
        page.wait_for_function("document.querySelector('[data-research-task=\"task-1\"]').getAttribute('aria-current') === 'true'")
        page.locator("#newResearchButton").click()
        assert page.locator(".research-welcome").is_visible()
        page.locator('[data-conversation-id="chat-ui"]').click()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator('[data-delete-conversation="chat-ui"]').click()
        page.locator('[data-conversation-id="chat-ui"]').wait_for(state="detached")
        assert page.locator("#answerConversationId").input_value() == ""
        assert page.locator("#emptyState").is_visible()
        page.reload(wait_until="networkidle")
        assert page.locator('[data-conversation-id="chat-ui"]').count() == 0
        assert page.locator('[data-research-task="task-5"]').count() == 0
        assert page.locator('[data-research-task="task-1"]').is_visible()
        assert "本机保存任务入口" not in page.locator("#sidebar").inner_text()
        page.screenshot(path=str(output_dir / "history-deletion.png"), full_page=False)
        page.locator("#logoutButton").click()
        page.locator("#authView:not(.hidden)").wait_for()
        state["owner"] = "other-ui-user"
        page.locator("#usernameInput").fill("other-ui-user")
        page.locator("#passwordInput").fill("fixture-password")
        page.locator("#authButton").click()
        page.locator("#appView:not(.hidden)").wait_for()
        page.locator("#researchWorkspaceButton").click()
        assert page.locator("[data-research-task]").count() == 0
        assert not errors, errors
        unexpected = [item for item in console_errors if "503" not in item and "404" not in item]
        assert not unexpected, unexpected
        browser.close()
    return {"result": "passed", "api": "fixture responses", "viewports": ["1440x1000", "390x844"],
            "creates": state["creates"], "runs": state["runs"], "cancels": state["cancels"], "chat_requests": state["asks"],
            "deleted_chat": state["deleted_chat"], "deleted_research": state["deleted_research"],
            "page_errors": errors, "unexpected_console_errors": unexpected, "screenshots": str(output_dir)}


def test_research_workspace_contract(tmp_path):
    run_smoke(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "docrag-research-ui")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir), ensure_ascii=False, indent=2))
