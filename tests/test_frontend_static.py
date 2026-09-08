from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_JS = ROOT_DIR / "medrag" / "frontend" / "app.js"
INDEX_HTML = ROOT_DIR / "medrag" / "frontend" / "index.html"
STYLES_CSS = ROOT_DIR / "medrag" / "frontend" / "styles.css"


def test_upload_finalizers_do_not_read_conversation_before_creation():
    source = APP_JS.read_text(encoding="utf-8")

    for function_name in (
        "finalizeKnowledgeBaseUpload",
        "finalizeStandaloneUpload",
    ):
        function_start = source.index(f"async function {function_name}")
        function_end = source.index(
            "\nasync function ",
            function_start + 1,
        )
        function_source = source[function_start:function_end]
        conversation_creation = function_source.index(
            "const conversation = await createConversation"
        )

        assert "conversation.user_type" not in (
            function_source[:conversation_creation]
        )


def test_knowledge_base_controls_are_present():
    html = INDEX_HTML.read_text(encoding="utf-8")

    for element_id in (
        "knowledgeBaseName",
        "createKnowledgeBaseButton",
        "knowledgeBaseList",
        "uploadKnowledgeBaseSelect",
        "knowledgeBaseDocumentList",
        "newKnowledgeBaseConversationButton",
    ):
        assert f'id="{element_id}"' in html


def test_frontend_calls_knowledge_base_contracts():
    source = APP_JS.read_text(encoding="utf-8")

    assert 'buildUrl("/knowledge-bases"' in source
    assert "/knowledge-bases/${encodeURIComponent(kbId)}/documents" in source
    assert "selected_document_ids" in source
    assert "currentKnowledgeBaseId" in source


def test_frontend_uses_bearer_authentication_for_business_requests():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")

    assert 'id="passwordInput"' in html
    assert 'id="registerButton"' in html
    assert "async function authorizedFetch" in source
    assert 'headers.set("Authorization"' in source
    assert 'authenticateUser("/auth/login"' in source
    assert 'authenticateUser("/auth/register"' in source


def test_register_button_is_visible_on_light_auth_card():
    styles = STYLES_CSS.read_text(encoding="utf-8")

    rule_start = styles.index(".auth-card .ghost-button {")
    rule_end = styles.index("}", rule_start)
    rule = styles[rule_start:rule_end]

    assert "background: #fff" in rule
    assert "color: var(--primary-dark)" in rule


def test_document_library_explains_single_document_conversation_flow():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")

    assert "文档库" in html
    assert "基于所选文档新建对话" in html
    assert "该文档暂无历史会话" in source


def test_user_chat_hides_internal_rag_parameters():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="answerConversationId" type="hidden"' in html
    assert 'id="answerTopK" type="hidden" value="3"' in html
    assert '<details class="side-panel" hidden>' in html
    assert "系统会得到 document_id" not in html
    assert "保存到 conversation" not in html


def test_conversation_history_is_not_filtered_by_active_scope():
    source = APP_JS.read_text(encoding="utf-8")
    function_start = source.index("async function loadConversations()")
    function_end = source.index("\nfunction renderConversationList", function_start)
    function_source = source[function_start:function_end]

    assert "params.document_id" not in function_source
    assert "params.kb_id" not in function_source


def test_composer_hides_answer_style_and_requires_conversation():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")

    assert 'id="answerUserType" type="hidden"' in html
    assert 'class="composer-actions"' not in html
    assert 'class="answer-mode"' not in html
    assert 'id="answerQuery"' in html and "disabled" in html
    assert "function updateComposerAvailability()" in source


def test_frontend_supports_sequential_multi_pdf_upload():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")

    assert 'id="pdfFile"' in html
    assert "multiple" in html
    assert "async function requestPdfIndex" in source
    assert "for (const [index, file] of files.entries())" in source
    assert "await requestPdfIndex(" in source
    assert "renderBatchUploadResults(results)" in source
