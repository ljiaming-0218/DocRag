const USER_STORAGE_KEY = "docrag.currentUser";

let currentUser = null;
let currentDocumentId = "";
let currentConversationId = "";
let documents = [];
let conversations = [];

function $(id) {
  return document.getElementById(id);
}

function getApiBase() {
  return window.location.origin;
}

function setStatus(id, message, type = "info") {
  const box = $(id);
  if (!box) return;
  box.textContent = message;
  box.className = `status show ${type}`;
}

function clearStatus(id) {
  const box = $(id);
  if (!box) return;
  box.textContent = "";
  box.className = "status";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function parseResponse(response) {
  const text = await response.text();
  let data;

  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text || "响应不是 JSON 格式" };
  }

  if (!response.ok) {
    const detailMessage =
      typeof data.detail === "object"
        ? data.detail?.message
        : data.detail;

    const message =
      data.message ||
      detailMessage ||
      `请求失败，状态码：${response.status}`;
    throw new Error(message);
  }

  return data;
}

function buildUrl(path, params = {}) {
  const url = new URL(`${getApiBase()}${path}`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  return url;
}

function saveCurrentUser(user) {
  currentUser = user;
  localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
}

function loadSavedUser() {
  const raw = localStorage.getItem(USER_STORAGE_KEY);
  if (!raw) return null;

  try {
    return JSON.parse(raw);
  } catch {
    localStorage.removeItem(USER_STORAGE_KEY);
    return null;
  }
}

function requireUser() {
  if (!currentUser || !currentUser.user_id) {
    throw new Error("请先登录或注册用户。");
  }
  return currentUser;
}

function showAuthView() {
  $("authView").classList.remove("hidden");
  $("appView").classList.add("hidden");
}

function showAppView() {
  $("authView").classList.add("hidden");
  $("appView").classList.remove("hidden");
  renderActiveUser();
}

function renderActiveUser() {
  if (!currentUser) return;
  $("activeUserName").textContent = currentUser.username;
  $("activeUserType").textContent = currentUser.default_user_type || "general";
  $("activeUserAvatar").textContent = currentUser.username.slice(0, 1).toUpperCase();
  $("answerUserType").value = currentUser.default_user_type || "general";
}

async function createOrLoginUser(username, defaultUserType) {
  const response = await fetch(`${getApiBase()}/users`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username,
      default_user_type: defaultUserType,
    }),
  });

  return parseResponse(response);
}

async function handleAuthSubmit(event) {
  event.preventDefault();
  clearStatus("authStatus");

  const username = $("usernameInput").value.trim();
  const defaultUserType = $("userTypeInput").value;

  if (!username) {
    setStatus("authStatus", "请输入用户名。", "error");
    return;
  }

  $("authButton").disabled = true;

  try {
    const user = await createOrLoginUser(username, defaultUserType);
    saveCurrentUser(user);
    showAppView();
    await loadDocuments();
    await loadConversations();
  } catch (error) {
    setStatus("authStatus", error.message, "error");
  } finally {
    $("authButton").disabled = false;
  }
}

function logout() {
  localStorage.removeItem(USER_STORAGE_KEY);
  currentUser = null;
  currentDocumentId = "";
  currentConversationId = "";
  documents = [];
  conversations = [];
  $("answerResult").innerHTML = "";
  $("documentList").innerHTML = "";
  $("conversationList").innerHTML = "";
  $("answerConversationId").value = "";
  $("searchDocumentId").value = "";
  $("newDocumentConversationButton").classList.add("hidden");
  showAuthView();
}

async function loadDocuments() {
  const user = requireUser();
  clearStatus("documentStatus");

  try {
    const url = buildUrl(
      `/users/${encodeURIComponent(user.user_id)}/documents`,
      { limit: 50 }
    );
    const response = await fetch(url);
    const data = await parseResponse(response);
    documents = Array.isArray(data) ? data : [];
    renderDocumentList();
  } catch (error) {
    documents = [];
    renderDocumentList();
    setStatus("documentStatus", error.message, "error");
  }
}

function renderDocumentList() {
  const target = $("documentList");

  if (!documents.length) {
    target.innerHTML = '<p class="empty-list">还没有历史文档。</p>';
    return;
  }

  target.innerHTML = documents
    .map((document) => {
      const activeClass =
        document.document_id === currentDocumentId
          ? " active"
          : "";
      return `
        <button
          class="document-item${activeClass}"
          type="button"
          data-document-id="${escapeHtml(document.document_id)}"
        >
          <span class="document-title">${escapeHtml(document.filename || "未命名文档")}</span>
          <span class="document-meta">
            ${escapeHtml(document.language || "unknown")}
            · ${escapeHtml(formatDate(document.updated_at))}
          </span>
        </button>
      `;
    })
    .join("");
}

async function selectDocument(documentId) {
  const document = documents.find(
    (item) => item.document_id === documentId
  );

  currentDocumentId = documentId;
  currentConversationId = "";
  $("searchDocumentId").value = documentId;
  $("answerConversationId").value = "";
  $("answerResult").innerHTML = "";
  $("answerQuery").value = "";
  $("newDocumentConversationButton").classList.remove("hidden");
  $("emptyState").classList.remove("hidden");
  renderDocumentList();
  updateContextText({
    title: document?.filename || "当前文档",
  });
  await loadConversations(documentId);
}

async function createConversation(userId, documentId, title) {
  const response = await fetch(`${getApiBase()}/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: userId,
      document_id: documentId,
      title,
    }),
  });
  return parseResponse(response);
}

async function loadConversations(documentId = null) {
  const user = requireUser();
  clearStatus("conversationStatus");

  try {
    const params = {
      user_id: user.user_id,
      limit: 50,
    };

    if (documentId) {
      params.document_id = documentId;
    }

    const url = buildUrl("/conversations", params);
    const response = await fetch(url);

    conversations = await parseResponse(response);
    renderConversationList();
  } catch (error) {
    conversations = [];
    renderConversationList();
    setStatus("conversationStatus", error.message, "error");
  }
}

function renderConversationList() {
  const target = $("conversationList");

  if (!conversations.length) {
    target.innerHTML = '<p class="empty-list">还没有历史会话。上传 PDF 后会自动创建。</p>';
    return;
  }

  target.innerHTML = conversations
    .map((conversation) => {
      const activeClass = conversation.conversation_id === currentConversationId ? " active" : "";
      return `
        <button class="conversation-item${activeClass}" type="button" data-conversation-id="${escapeHtml(conversation.conversation_id)}">
          <span class="conversation-title">${escapeHtml(conversation.title || "新会话")}</span>
          <span class="conversation-meta">${escapeHtml(formatDate(conversation.updated_at))}</span>
          <span class="conversation-doc">${escapeHtml(conversation.document_id || "")}</span>
        </button>
      `;
    })
    .join("");
}

async function selectConversation(conversationId) {
  const user = requireUser();
  const conversation = conversations.find((item) => item.conversation_id === conversationId);
  $("answerUserType").value =
  conversation?.user_type ||
  currentUser.default_user_type ||
  "general";
  currentConversationId = conversationId;
  currentDocumentId = conversation?.document_id || "";
  $("newDocumentConversationButton").classList.remove("hidden");
  $("answerConversationId").value = currentConversationId;
  $("searchDocumentId").value = currentDocumentId;
  updateContextText(conversation);
  renderDocumentList();
  renderConversationList();

  try {
    const url = buildUrl(`/conversations/${encodeURIComponent(conversationId)}/messages`, {
      user_id: user.user_id,
    });
    const response = await fetch(url);
    const messages = await parseResponse(response);
    renderMessages(messages);
  } catch (error) {
    setStatus("answerStatus", error.message, "error");
  }
}

function updateContextText(conversation) {
  if (!conversation && !currentDocumentId) {
    $("currentContextText").textContent = "先上传 PDF，或从左侧选择历史会话。";
    return;
  }

  const title = conversation?.title || "当前文档";
  $("currentContextText").textContent = `${title} · document_id: ${currentDocumentId}`;
}

function renderMessages(messages) {
  const target = $("answerResult");
  target.innerHTML = "";

  if (!messages.length) {
    $("emptyState").classList.remove("hidden");
    return;
  }

  $("emptyState").classList.add("hidden");
  messages.forEach((message) => {
    appendMessage(message.role, message.content, message.sources || [], {
      taskType: message.task_type,
    });
  });
  scrollChatToBottom();
}

async function startBlankConversation() {
  currentDocumentId = "";
  currentConversationId = "";

  $("pdfFile").value = "";
  $("searchDocumentId").value = "";
  $("answerConversationId").value = "";
  $("answerResult").innerHTML = "";
  $("indexResult").innerHTML = "";
  $("searchResult").innerHTML = "";
  $("answerQuery").value = "";
  $("searchQuery").value = "";
  $("answerUserType").value =
    currentUser?.default_user_type || "general";

  clearStatus("indexStatus");
  clearStatus("searchStatus");
  clearStatus("answerStatus");
  $("newDocumentConversationButton").classList.add("hidden");
  $("emptyState").classList.remove("hidden");
  updateContextText(null);
  renderDocumentList();
  await loadConversations();
}


async function startNewConversation() {
  if (!currentDocumentId) {
    setStatus("answerStatus", "请先上传并索引一个 PDF 文件。", "error");
    return;
  }

  try {
    const user = requireUser();
    const file = $("pdfFile").files[0];
    const title = file ? file.name.replace(/\.pdf$/i, "") : "新会话";
    const conversation = await createConversation(user.user_id, currentDocumentId, title);
    $("answerUserType").value =
      conversation.user_type ||
      currentUser.default_user_type ||
      "general";
    currentConversationId = conversation.conversation_id;
    $("answerConversationId").value = currentConversationId;
    $("answerResult").innerHTML = "";
    $("answerQuery").value = "";
    $("emptyState").classList.remove("hidden");
    updateContextText(conversation);
    setStatus("answerStatus", "已创建新会话，可以开始提问。", "success");
    await loadConversations(currentDocumentId);
    $("answerQuery").focus();
  } catch (error) {
    setStatus("answerStatus", error.message, "error");
  }
}

function scrollChatToBottom() {
  const chatWindow = $("chatWindow");
  requestAnimationFrame(() => {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  });
}

function renderMeta(targetId, data) {
  const target = $(targetId);
  const savedChunks = data["成功保存块数"] ?? "-";
  const totalChunks = data["总块数"] ?? "-";
  const totalPages = data["总页数"] ?? "-";

  target.innerHTML = `
    <div class="meta">
      <div class="meta-item">
        <span>文件名</span>
        <strong>${escapeHtml(data["文件名"] || "-")}</strong>
      </div>
      <div class="meta-item">
        <span>document_id</span>
        <strong>${escapeHtml(data.document_id)}</strong>
      </div>
      <div class="meta-item">
        <span>总页数</span>
        <strong>${escapeHtml(totalPages)}</strong>
      </div>
      <div class="meta-item">
        <span>文本块</span>
        <strong>${escapeHtml(savedChunks)} / ${escapeHtml(totalChunks)}</strong>
      </div>
    </div>
  `;
}

function getChunkText(chunk) {
  return chunk["文本块"] || chunk.text || chunk.document || "";
}

function getChunkMeta(chunk) {
  return chunk["元数据"] || chunk.metadata || {};
}

function getChunkDistance(chunk) {
  return chunk["距离"] ?? chunk.distance ?? "";
}

function renderSearchResults(targetId, chunks) {
  const target = $(targetId);

  if (!chunks || chunks.length === 0) {
    target.innerHTML = '<p class="hint">没有检索到相关内容。</p>';
    return;
  }

  target.innerHTML = chunks
    .map((chunk, index) => {
      const meta = getChunkMeta(chunk);
      const distance = getChunkDistance(chunk);
      const formattedDistance = typeof distance === "number" ? distance.toFixed(4) : distance;
      return `
        <div class="chunk">
          <div class="chunk-title">
            <span>结果 ${index + 1} · 第 ${escapeHtml(meta.page_number ?? "-")} 页 · 块 ${escapeHtml(meta.chunk_index ?? "-")}</span>
            <span>距离：${escapeHtml(formattedDistance || "-")}</span>
          </div>
          <p>${escapeHtml(getChunkText(chunk))}</p>
        </div>
      `;
    })
    .join("");
}

async function indexPdf() {
  const user = requireUser();
  const fileInput = $("pdfFile");
  const chunkSize = $("chunkSize").value;
  const chunkOverlap = $("chunkOverlap").value;
  const button = $("indexButton");


  if (!fileInput.files.length) {
    setStatus("indexStatus", "请先选择一个 PDF 文件。", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  clearStatus("indexStatus");
  $("indexResult").innerHTML = "";
  button.disabled = true;
  setStatus("indexStatus", "正在索引 PDF，embedding 可能需要一点时间。", "info");

  try {
    const url = buildUrl("/pdf/index", {
      user_id: user.user_id,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
    });
    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });
    const data = await parseResponse(response);

    currentDocumentId = data.document_id || "";
    $("searchDocumentId").value = currentDocumentId;
    $("newDocumentConversationButton").classList.remove("hidden");

    renderMeta("indexResult", data);
    await loadDocuments();

    if (data.existing_document) {
      currentConversationId = "";
      $("answerConversationId").value = "";
      $("answerResult").innerHTML = "";
      $("emptyState").classList.remove("hidden");
      setStatus("indexStatus", "检测到该文件已索引。你可以从左侧历史会话继续，或点击新建对话。", "success");
      conversations = Array.isArray(data.conversations)
                    ? data.conversations
                    : [];

      renderConversationList();
      renderDocumentList();
      updateContextText(null);
      return;
    }

    const title = fileInput.files[0].name.replace(/\.pdf$/i, "");
    const conversation = await createConversation(user.user_id, currentDocumentId, title);
    currentConversationId = conversation.conversation_id;
    $("answerUserType").value = conversation.user_type || currentUser.default_user_type || "general";
    $("answerConversationId").value = currentConversationId;
    $("answerResult").innerHTML = "";
    $("emptyState").classList.remove("hidden");
    updateContextText(conversation);
    setStatus("indexStatus", `${data.message || "PDF 索引完成。"} 已创建问答会话。`, "success");
    await loadConversations(currentDocumentId);
  } catch (error) {
    setStatus("indexStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function searchChunks() {
  const user = requireUser();
  const documentId = $("searchDocumentId").value.trim();
  const query = $("searchQuery").value.trim();
  const topK = $("searchTopK").value;
  const button = $("searchButton");

  if (!documentId || !query) {
    setStatus("searchStatus", "请填写 document_id 和问题。", "error");
    return;
  }

  clearStatus("searchStatus");
  $("searchResult").innerHTML = "";
  button.disabled = true;
  setStatus("searchStatus", "正在检索相关文本块。", "info");

  try {
    const url = buildUrl("/pdf/search", {
      user_id: user.user_id,
      document_id: documentId,
      query,
      n_results: topK,
    });
    const response = await fetch(url, { method: "POST" });
    const data = await parseResponse(response);

    setStatus("searchStatus", `检索完成，共找到 ${data.sources_count ?? 0} 条相关内容。`, "success");
    renderSearchResults("searchResult", data.sources || []);
  } catch (error) {
    setStatus("searchStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function appendMessage(role, content, sources = [], options = {}) {
  $("emptyState").classList.add("hidden");
  const isUser = role === "user";
  const sourceTargetId = `sources-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const taskType = options.taskType;

  $("answerResult").insertAdjacentHTML(
    "beforeend",
    `
      <div class="message ${isUser ? "message-user" : "message-assistant"}">
        <div class="avatar ${isUser ? "user" : "assistant"}">${isUser ? "你" : "AI"}</div>
        <div class="message-content">
          <div class="message-role">${isUser ? "You" : "DocRAG Assistant"}</div>
          ${
            !isUser && taskType
              ? `<div class="message-meta"><span class="task-badge">任务类型：${escapeHtml(taskType)}</span></div>`
              : ""
          }
          <p>${escapeHtml(content)}</p>
          ${
            !isUser
              ? `
                <details class="sources-panel">
                  <summary>
                    <span>引用来源 · ${escapeHtml(sources.length)} 条</span>
                    <span class="sources-toggle">展开/收起</span>
                  </summary>
                  <div id="${sourceTargetId}" class="sources-body"></div>
                </details>
              `
              : ""
          }
        </div>
      </div>
    `
  );

  if (!isUser) {
    renderSearchResults(sourceTargetId, sources);
  }

  scrollChatToBottom();
}

async function answerQuestion() {
  const user = requireUser();
  const conversationId = $("answerConversationId").value.trim();
  const query = $("answerQuery").value.trim();
  const topK = Number($("answerTopK").value);
  const button = $("answerButton");
  const userType = $("answerUserType").value;

  if (!conversationId || !query) {
    setStatus("answerStatus", "请先建立或选择会话，并输入问题。", "error");
    return;
  }

  clearStatus("answerStatus");
  button.disabled = true;
  setStatus("answerStatus", "正在生成回答。", "info");

  try {
    const response = await fetch(`${getApiBase()}/conversations/${encodeURIComponent(conversationId)}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: user.user_id,
        query,
        history_limit: 6,
        n_results: topK,
        user_type: userType,
      }),
    });
    const data = await parseResponse(response);

    appendMessage("user", query);
    appendMessage("assistant", data.answer, data.sources || [], {
      taskType: data.task_type,
    });
    $("answerQuery").value = "";
    setStatus("answerStatus", "回答生成完成。", "success");
    await loadConversations(currentDocumentId);
  } catch (error) {
    setStatus("answerStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  $("authForm").addEventListener("submit", handleAuthSubmit);
  $("logoutButton").addEventListener("click", logout);
  $("newConversationButton").addEventListener("click", startBlankConversation);
  $("indexButton").addEventListener("click", indexPdf);
  $("searchButton").addEventListener("click", searchChunks);
  $("answerButton").addEventListener("click", answerQuestion);
  $("newDocumentConversationButton").addEventListener("click",startNewConversation);
  $("conversationList").addEventListener("click", (event) => {
    const item = event.target.closest("[data-conversation-id]");
    if (!item) return;
    selectConversation(item.dataset.conversationId);
  });
  $("documentList").addEventListener("click", (event) => {
    const item = event.target.closest("[data-document-id]");
    if (!item) return;
    selectDocument(item.dataset.documentId);
  });
  $("answerQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      answerQuestion();
    }
  });
}

async function initApp() {
  bindEvents();
  const savedUser = loadSavedUser();

  if (!savedUser?.user_id) {
    showAuthView();
    return;
  }

  currentUser = savedUser;
  showAppView();
  await loadDocuments();
  await loadConversations();
}

initApp();
