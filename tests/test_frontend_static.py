from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_JS = ROOT_DIR / "medrag" / "frontend" / "app.js"


def test_index_pdf_does_not_read_conversation_before_creation():
    source = APP_JS.read_text(encoding="utf-8")

    function_start = source.index("async function indexPdf()")
    conversation_creation = source.index(
        "const conversation = await createConversation",
        function_start,
    )

    unsafe_read = source.find(
        "conversation.user_type",
        function_start,
        conversation_creation,
    )

    assert unsafe_read == -1
