const USER_STORAGE_KEY = "docrag.currentUser";

let currentUser = null;
let currentDocumentId = "";
let currentKnowledgeBaseId = "";
let currentConversationId = "";
let documents = [];
let knowledgeBases = [];
let knowledgeBaseDocuments = [];
let selectedKnowledgeBaseDocumentIds = new Set();
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
  if (!currentUser || !currentUser.user_id || !currentUser.access_token) {
    throw new Error("请先登录或注册用户。");
  }
  return currentUser;
}

async function authorizedFetch(input, options = {}) {
  const user = requireUser();
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${user.access_token}`);
  return fetch(input, { ...options, headers });
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

async function authenticateUser(path, payload) {
  const response = await fetch(`${getApiBase()}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse(response);
}

async function completeAuthentication(authResult) {
  resetUserWorkspace();
  saveCurrentUser({
    ...authResult.user,
    access_token: authResult.access_token,
  });
  showAppView();
  await loadKnowledgeBases();
  await loadDocuments();
  await loadConversations();
}

function readCredentials() {
  return {
    username: $("usernameInput").value.trim(),
    password: $("passwordInput").value,
    defaultUserType: $("userTypeInput").value,
  };
}

function setAuthButtonsDisabled(disabled) {
  $("authButton").disabled = disabled;
  $("registerButton").disabled = disabled;
}

async function handleAuthSubmit(event) {
  event.preventDefault();
  clearStatus("authStatus");

  const { username, password } = readCredentials();

  if (!username || password.length < 8) {
    setStatus("authStatus", "请输入用户名和至少 8 位密码。", "error");
    return;
  }

  setAuthButtonsDisabled(true);

  try {
    const result = await authenticateUser("/auth/login", {
      username,
      password,
    });
    await completeAuthentication(result);
  } catch (error) {
    setStatus("authStatus", error.message, "error");
  } finally {
    setAuthButtonsDisabled(false);
  }
}

async function handleRegister() {
  clearStatus("authStatus");
  const { username, password, defaultUserType } = readCredentials();
  if (!username || password.length < 8) {
    setStatus("authStatus", "请输入用户名和至少 8 位密码。", "error");
    return;
  }

  setAuthButtonsDisabled(true);
  try {
    const result = await authenticateUser("/auth/register", {
      username,
      password,
      default_user_type: defaultUserType,
    });
    await completeAuthentication(result);
  } catch (error) {
    setStatus("authStatus", error.message, "error");
  } finally {
    setAuthButtonsDisabled(false);
  }
}

function resetUserWorkspace() {
  currentDocumentId = "";
  currentKnowledgeBaseId = "";
  currentConversationId = "";
  documents = [];
  knowledgeBases = [];
  knowledgeBaseDocuments = [];
  selectedKnowledgeBaseDocumentIds = new Set();
  conversations = [];

  $("pdfFile").value = "";
  $("uploadKnowledgeBaseSelect").value = "";
  $("answerConversationId").value = "";
  $("searchDocumentId").value = "";
  $("answerQuery").value = "";
  $("searchQuery").value = "";
  $("answerResult").innerHTML = "";
  $("indexResult").innerHTML = "";
  $("searchResult").innerHTML = "";
  $("documentList").innerHTML = "";
  $("knowledgeBaseList").innerHTML = "";
  $("knowledgeBaseDocumentList").innerHTML = "";
  $("conversationList").innerHTML = "";
  $("uploadProgress").classList.add("hidden");
  $("uploadProgressText").textContent = "准备上传";
  $("uploadProgressCount").textContent = "0 / 0";
  $("uploadProgressBar").style.width = "0%";

  [
    "indexStatus",
    "searchStatus",
    "answerStatus",
    "documentStatus",
    "conversationStatus",
    "knowledgeBaseStatus",
    "knowledgeBaseDocumentStatus",
  ].forEach(clearStatus);

  renderSelectedPdfFiles();
  $("newDocumentConversationButton").classList.add("hidden");
  $("newKnowledgeBaseConversationButton").classList.add("hidden");
  $("emptyState").classList.remove("hidden");
  updateContextText(null);
}

function logout() {
  localStorage.removeItem(USER_STORAGE_KEY);
  currentUser = null;
  resetUserWorkspace();
  showAuthView();
}

async function loadKnowledgeBases() {
  const user = requireUser();
  clearStatus("knowledgeBaseStatus");

  try {
    const url = buildUrl("/knowledge-bases", {
      user_id: user.user_id,
      limit: 50,
    });
    const response = await authorizedFetch(url);
    const data = await parseResponse(response);
    knowledgeBases = Array.isArray(data) ? data : [];
    renderKnowledgeBaseList();
    renderKnowledgeBaseOptions();
  } catch (error) {
    knowledgeBases = [];
    renderKnowledgeBaseList();
    renderKnowledgeBaseOptions();
    setStatus("knowledgeBaseStatus", error.message, "error");
  }
}

function renderKnowledgeBaseList() {
  const target = $("knowledgeBaseList");
  if (!knowledgeBases.length) {
    target.innerHTML = '<p class="empty-list">还没有知识库。</p>';
    return;
  }

  target.innerHTML = knowledgeBases
    .map((knowledgeBase) => {
      const activeClass =
        knowledgeBase.kb_id === currentKnowledgeBaseId ? " active" : "";
      return `
        <button
          class="knowledge-base-item${activeClass}"
          type="button"
          data-kb-id="${escapeHtml(knowledgeBase.kb_id)}"
        >
          <span class="knowledge-base-title">${escapeHtml(knowledgeBase.name)}</span>
          <span class="knowledge-base-meta">${escapeHtml(knowledgeBase.description || "暂无描述")}</span>
        </button>
      `;
    })
    .join("");
}

function renderKnowledgeBaseOptions() {
  const select = $("uploadKnowledgeBaseSelect");
  const previousValue = select.value || currentKnowledgeBaseId;
  select.innerHTML = [
    '<option value="">仅保存为个人文档</option>',
    ...knowledgeBases.map(
      (knowledgeBase) =>
        `<option value="${escapeHtml(knowledgeBase.kb_id)}">${escapeHtml(knowledgeBase.name)}</option>`
    ),
  ].join("");

  if (knowledgeBases.some((item) => item.kb_id === previousValue)) {
    select.value = previousValue;
  }
}

async function handleCreateKnowledgeBase() {
  const user = requireUser();
  const name = $("knowledgeBaseName").value.trim();
  const description = $("knowledgeBaseDescription").value.trim();
  const button = $("createKnowledgeBaseButton");

  if (!name) {
    setStatus("knowledgeBaseStatus", "请输入知识库名称。", "error");
    return;
  }

  button.disabled = true;
  clearStatus("knowledgeBaseStatus");
  try {
    const response = await authorizedFetch(`${getApiBase()}/knowledge-bases`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: user.user_id,
        name,
        description,
      }),
    });
    const knowledgeBase = await parseResponse(response);
    $("knowledgeBaseName").value = "";
    $("knowledgeBaseDescription").value = "";
    await loadKnowledgeBases();
    await selectKnowledgeBase(knowledgeBase.kb_id);
    setStatus("knowledgeBaseStatus", "知识库创建完成。", "success");
  } catch (error) {
    setStatus("knowledgeBaseStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function selectKnowledgeBase(kbId) {
  const knowledgeBase = knowledgeBases.find((item) => item.kb_id === kbId);
  currentKnowledgeBaseId = kbId;
  currentDocumentId = "";
  currentConversationId = "";
  selectedKnowledgeBaseDocumentIds = new Set();
  $("uploadKnowledgeBaseSelect").value = kbId;
  $("searchDocumentId").value = "";
  $("answerConversationId").value = "";
  $("answerResult").innerHTML = "";
  $("newDocumentConversationButton").classList.add("hidden");
  $("newKnowledgeBaseConversationButton").classList.remove("hidden");
  $("emptyState").classList.remove("hidden");
  renderKnowledgeBaseList();
  renderDocumentList();
  updateContextText({
    title: knowledgeBase?.name || "当前知识库",
    kb_id: kbId,
  });
  await loadKnowledgeBaseDocuments(kbId);
  await loadConversations();
}

async function loadKnowledgeBaseDocuments(kbId = currentKnowledgeBaseId) {
  const user = requireUser();
  clearStatus("knowledgeBaseDocumentStatus");
  if (!kbId) {
    knowledgeBaseDocuments = [];
    renderKnowledgeBaseDocuments();
    return;
  }

  try {
    const url = buildUrl(
      `/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
      { user_id: user.user_id, limit: 100 }
    );
    const response = await authorizedFetch(url);
    const data = await parseResponse(response);
    knowledgeBaseDocuments = Array.isArray(data) ? data : [];
    const availableIds = new Set(
      knowledgeBaseDocuments.map((document) => document.document_id)
    );
    selectedKnowledgeBaseDocumentIds = new Set(
      [...selectedKnowledgeBaseDocumentIds].filter((id) => availableIds.has(id))
    );
    renderKnowledgeBaseDocuments();
  } catch (error) {
    knowledgeBaseDocuments = [];
    renderKnowledgeBaseDocuments();
    setStatus("knowledgeBaseDocumentStatus", error.message, "error");
  }
}

function renderKnowledgeBaseDocuments() {
  const target = $("knowledgeBaseDocumentList");
  if (!currentKnowledgeBaseId) {
    target.innerHTML = '<p class="empty-list">请先选择知识库。</p>';
    return;
  }
  if (!knowledgeBaseDocuments.length) {
    target.innerHTML = '<p class="empty-list">当前知识库还没有文档。</p>';
    return;
  }

  target.innerHTML = knowledgeBaseDocuments
    .map((document) => {
      const checked = selectedKnowledgeBaseDocumentIds.has(document.document_id)
        ? " checked"
        : "";
      return `
        <label class="knowledge-base-document-item">
          <input
            type="checkbox"
            data-kb-document-id="${escapeHtml(document.document_id)}"
            ${checked}
          />
          <span>
            <strong>${escapeHtml(document.filename || "未命名文档")}</strong>
            <small>${escapeHtml(document.language || "unknown")}</small>
          </span>
        </label>
      `;
    })
    .join("");
}

async function linkDocumentToKnowledgeBase(kbId, documentId) {
  const user = requireUser();
  const response = await authorizedFetch(
    `${getApiBase()}/knowledge-bases/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(documentId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: user.user_id }),
    }
  );
  return parseResponse(response);
}

async function loadDocuments() {
  const user = requireUser();
  clearStatus("documentStatus");

  try {
    const url = buildUrl(
      `/users/${encodeURIComponent(user.user_id)}/documents`,
      { limit: 50 }
    );
    const response = await authorizedFetch(url);
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
  currentKnowledgeBaseId = "";
  currentConversationId = "";
  knowledgeBaseDocuments = [];
  selectedKnowledgeBaseDocumentIds = new Set();
  $("uploadKnowledgeBaseSelect").value = "";
  $("searchDocumentId").value = documentId;
  $("answerConversationId").value = "";
  $("answerResult").innerHTML = "";
  $("answerQuery").value = "";
  $("newDocumentConversationButton").classList.remove("hidden");
  $("newKnowledgeBaseConversationButton").classList.add("hidden");
  $("emptyState").classList.remove("hidden");
  renderDocumentList();
  renderKnowledgeBaseList();
  renderKnowledgeBaseDocuments();
  updateContextText({
    title: document?.filename || "当前文档",
  });
  await loadConversations();
  const hasRelatedConversation = conversations.some(
    (conversation) => conversation.document_id === documentId
  );
  if (
    !hasRelatedConversation
    && !$("conversationStatus").classList.contains("show")
  ) {
    setStatus(
      "conversationStatus",
      "该文档暂无历史会话，可点击“基于所选文档新建对话”。",
      "info"
    );
  }
}

async function createConversation(userId, scope, title) {
  const payload = {
    user_id: userId,
    title,
    ...scope,
  };
  const response = await authorizedFetch(`${getApiBase()}/conversations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

async function loadConversations() {
  const user = requireUser();
  clearStatus("conversationStatus");

  try {
    const params = {
      user_id: user.user_id,
      limit: 50,
    };

    const url = buildUrl("/conversations", params);
    const response = await authorizedFetch(url);

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
          <span class="conversation-doc">${escapeHtml(
            conversation.kb_id
              ? `知识库：${getKnowledgeBaseName(conversation.kb_id)}`
              : `文档：${getDocumentName(conversation.document_id)}`
          )}</span>
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
  currentKnowledgeBaseId = conversation?.kb_id || "";
  selectedKnowledgeBaseDocumentIds = new Set(
    conversation?.selected_document_ids || []
  );
  $("newDocumentConversationButton").classList.toggle(
    "hidden",
    !currentDocumentId
  );
  $("newKnowledgeBaseConversationButton").classList.toggle(
    "hidden",
    !currentKnowledgeBaseId
  );
  $("answerConversationId").value = currentConversationId;
  $("searchDocumentId").value = currentDocumentId;
  if (currentKnowledgeBaseId) {
    $("uploadKnowledgeBaseSelect").value = currentKnowledgeBaseId;
    await loadKnowledgeBaseDocuments(currentKnowledgeBaseId);
  } else {
    knowledgeBaseDocuments = [];
    renderKnowledgeBaseDocuments();
  }
  updateContextText(conversation);
  renderDocumentList();
  renderKnowledgeBaseList();
  renderConversationList();

  try {
    const url = buildUrl(`/conversations/${encodeURIComponent(conversationId)}/messages`, {
      user_id: user.user_id,
    });
    const response = await authorizedFetch(url);
    const messages = await parseResponse(response);
    renderMessages(messages);
  } catch (error) {
    setStatus("answerStatus", error.message, "error");
  }
}

function updateContextText(conversation) {
  updateComposerAvailability();
  if (!conversation && !currentDocumentId && !currentKnowledgeBaseId) {
    $("currentContextText").textContent = "先上传 PDF，或从左侧选择历史会话。";
    return;
  }

  const title = conversation?.title || "当前文档";
  if (conversation?.kb_id || currentKnowledgeBaseId) {
    const kbId = conversation?.kb_id || currentKnowledgeBaseId;
    const knowledgeBaseName = getKnowledgeBaseName(kbId);
    $("currentContextText").textContent =
      title === knowledgeBaseName
        ? `知识库：${knowledgeBaseName}`
        : `${title} · 知识库：${knowledgeBaseName}`;
    return;
  }
  $("currentContextText").textContent = title;
}

function updateComposerAvailability() {
  const hasConversation = Boolean(currentConversationId);
  $("answerQuery").disabled = !hasConversation;
  $("answerButton").disabled = !hasConversation;
  $("answerQuery").placeholder = hasConversation
    ? "输入问题，按 Enter 发送，Shift + Enter 换行"
    : "请先选择或新建会话";
}

function getKnowledgeBaseName(kbId) {
  return knowledgeBases.find((item) => item.kb_id === kbId)?.name || kbId;
}

function getDocumentName(documentId) {
  return documents.find((item) => item.document_id === documentId)?.filename
    || "未命名文档";
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
  currentKnowledgeBaseId = "";
  currentConversationId = "";
  knowledgeBaseDocuments = [];
  selectedKnowledgeBaseDocumentIds = new Set();

  $("pdfFile").value = "";
  renderSelectedPdfFiles();
  $("uploadKnowledgeBaseSelect").value = "";
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
  $("newKnowledgeBaseConversationButton").classList.add("hidden");
  $("emptyState").classList.remove("hidden");
  updateContextText(null);
  renderDocumentList();
  renderKnowledgeBaseList();
  renderKnowledgeBaseDocuments();
  await loadConversations();
}


async function startNewConversation() {
  if (!currentDocumentId) {
    setStatus("answerStatus", "请先上传并索引一个 PDF 文件。", "error");
    return;
  }

  try {
    const user = requireUser();
    const activeDocument = documents.find(
      (item) => item.document_id === currentDocumentId
    );
    const title = activeDocument?.filename
      ? activeDocument.filename.replace(/\.pdf$/i, "")
      : "新会话";
    const conversation = await createConversation(
      user.user_id,
      { document_id: currentDocumentId },
      title
    );
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
    clearStatus("answerStatus");
    await loadConversations();
    $("answerQuery").focus();
  } catch (error) {
    setStatus("answerStatus", error.message, "error");
  }
}

async function startKnowledgeBaseConversation() {
  if (!currentKnowledgeBaseId) {
    setStatus(
      "knowledgeBaseDocumentStatus",
      "请先选择知识库。",
      "error"
    );
    return;
  }
  if (!knowledgeBaseDocuments.length) {
    setStatus(
      "knowledgeBaseDocumentStatus",
      "当前知识库没有可检索文档。",
      "error"
    );
    return;
  }

  const user = requireUser();
  const selectedIds = [...selectedKnowledgeBaseDocumentIds];
  const scope = { kb_id: currentKnowledgeBaseId };
  if (selectedIds.length) {
    scope.selected_document_ids = selectedIds;
  }

  try {
    const conversation = await createConversation(
      user.user_id,
      scope,
      getKnowledgeBaseName(currentKnowledgeBaseId)
    );
    currentConversationId = conversation.conversation_id;
    $("answerConversationId").value = currentConversationId;
    $("answerResult").innerHTML = "";
    $("answerQuery").value = "";
    $("emptyState").classList.remove("hidden");
    updateContextText(conversation);
    clearStatus("answerStatus");
    await loadConversations();
    $("answerQuery").focus();
  } catch (error) {
    setStatus("knowledgeBaseDocumentStatus", error.message, "error");
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
      const sourceName = meta.filename || "未知文档";
      return `
        <div class="chunk">
          <div class="chunk-title">
            <span>${escapeHtml(sourceName)} · 第 ${escapeHtml(meta.page_number ?? "-")} 页 · 块 ${escapeHtml(meta.chunk_index ?? "-")}</span>
            <span>距离：${escapeHtml(formattedDistance || "-")}</span>
          </div>
          <p>${escapeHtml(getChunkText(chunk))}</p>
        </div>
      `;
    })
    .join("");
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderSelectedPdfFiles() {
  const files = Array.from($("pdfFile").files || []);
  const target = $("selectedFileList");
  const button = $("indexButton");

  button.textContent = files.length > 1
    ? `建立 ${files.length} 个 PDF 索引`
    : "建立 PDF 索引";

  if (!files.length) {
    target.innerHTML = '<span class="empty-list">尚未选择文件</span>';
    return;
  }

  target.innerHTML = files
    .map(
      (file) => `
        <div class="selected-file-item">
          <span title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
          <small>${escapeHtml(formatFileSize(file.size))}</small>
        </div>
      `
    )
    .join("");
}

function updateUploadProgress(completed, total, filename = "") {
  const percent = total ? Math.round((completed / total) * 100) : 0;
  $("uploadProgress").classList.remove("hidden");
  $("uploadProgressText").textContent = filename
    ? `正在处理：${filename}`
    : "批量索引完成";
  $("uploadProgressCount").textContent = `${completed} / ${total}`;
  $("uploadProgressBar").style.width = `${percent}%`;
}

function renderBatchUploadResults(results) {
  $("indexResult").innerHTML = `
    <div class="batch-result-list">
      ${results
        .map((result) => {
          const statusLabel = result.status === "success"
            ? "成功"
            : result.status === "partial"
              ? "部分成功"
              : "失败";
          const detail = result.status === "success"
            ? result.data?.existing_document
              ? "已复用现有索引"
              : `已保存 ${result.data?.["成功保存块数"] ?? "-"} 个文本块`
            : result.error;
          return `
            <div class="batch-result-item ${result.status}">
              <strong title="${escapeHtml(result.file.name)}">${escapeHtml(result.file.name)}</strong>
              <span>${statusLabel}</span>
              <small>${escapeHtml(detail || "处理完成")}</small>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

async function requestPdfIndex(userId, file, chunkSize, chunkOverlap) {
  const formData = new FormData();
  formData.append("file", file);
  const url = buildUrl("/pdf/index", {
    user_id: userId,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
  });
  const response = await authorizedFetch(url, {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}

async function finalizeKnowledgeBaseUpload(
  user,
  targetKnowledgeBaseId,
  successfulResults
) {
  currentKnowledgeBaseId = targetKnowledgeBaseId;
  currentDocumentId = "";
  selectedKnowledgeBaseDocumentIds = new Set();
  $("searchDocumentId").value = "";
  $("newDocumentConversationButton").classList.add("hidden");
  $("newKnowledgeBaseConversationButton").classList.remove("hidden");

  await loadDocuments();
  await loadKnowledgeBases();
  await loadKnowledgeBaseDocuments(targetKnowledgeBaseId);

  const hasNewContent = successfulResults.some(
    (result) => result.linkResult?.linked || !result.data.existing_document
  );
  if (hasNewContent) {
    const conversation = await createConversation(
      user.user_id,
      { kb_id: targetKnowledgeBaseId },
      getKnowledgeBaseName(targetKnowledgeBaseId)
    );
    currentConversationId = conversation.conversation_id;
    $("answerConversationId").value = currentConversationId;
    $("answerResult").innerHTML = "";
    $("emptyState").classList.remove("hidden");
    updateContextText(conversation);
  } else {
    currentConversationId = "";
    $("answerConversationId").value = "";
    updateContextText({
      title: getKnowledgeBaseName(targetKnowledgeBaseId),
      kb_id: targetKnowledgeBaseId,
    });
  }

  renderKnowledgeBaseList();
  renderDocumentList();
  await loadConversations();
}

async function finalizeStandaloneUpload(user, successfulResults) {
  const lastResult = successfulResults.at(-1);
  if (!lastResult) return;

  currentKnowledgeBaseId = "";
  knowledgeBaseDocuments = [];
  selectedKnowledgeBaseDocumentIds = new Set();
  $("newKnowledgeBaseConversationButton").classList.add("hidden");
  renderKnowledgeBaseList();
  renderKnowledgeBaseDocuments();
  await loadDocuments();

  currentDocumentId = lastResult.data.document_id || "";
  $("searchDocumentId").value = currentDocumentId;
  $("newDocumentConversationButton").classList.remove("hidden");

  if (successfulResults.length > 1) {
    await selectDocument(currentDocumentId);
    return;
  }

  if (lastResult.data.existing_document) {
    currentConversationId = "";
    $("answerConversationId").value = "";
    $("answerResult").innerHTML = "";
    $("emptyState").classList.remove("hidden");
    conversations = Array.isArray(lastResult.data.conversations)
      ? lastResult.data.conversations
      : [];
    renderConversationList();
    renderDocumentList();
    updateContextText(null);
    return;
  }

  const conversation = await createConversation(
    user.user_id,
    { document_id: currentDocumentId },
    lastResult.file.name.replace(/\.pdf$/i, "")
  );
  currentConversationId = conversation.conversation_id;
  $("answerUserType").value =
    conversation.user_type || user.default_user_type || "general";
  $("answerConversationId").value = currentConversationId;
  $("answerResult").innerHTML = "";
  $("emptyState").classList.remove("hidden");
  updateContextText(conversation);
  await loadConversations();
}

async function indexPdf() {
  const user = requireUser();
  const fileInput = $("pdfFile");
  const files = Array.from(fileInput.files || []);
  const chunkSize = $("chunkSize").value;
  const chunkOverlap = $("chunkOverlap").value;
  const targetKnowledgeBaseId = $("uploadKnowledgeBaseSelect").value;
  const button = $("indexButton");

  if (!files.length) {
    setStatus("indexStatus", "请至少选择一个 PDF 文件。", "error");
    return;
  }

  const invalidFile = files.find(
    (file) => !file.name.toLowerCase().endsWith(".pdf")
  );
  if (invalidFile) {
    setStatus("indexStatus", `${invalidFile.name} 不是 PDF 文件。`, "error");
    return;
  }

  clearStatus("indexStatus");
  $("indexResult").innerHTML = "";
  button.disabled = true;
  fileInput.disabled = true;
  $("uploadKnowledgeBaseSelect").disabled = true;
  $("chunkSize").disabled = true;
  $("chunkOverlap").disabled = true;
  updateUploadProgress(0, files.length, files[0].name);

  const results = [];
  for (const [index, file] of files.entries()) {
    button.textContent = `正在索引 ${index + 1} / ${files.length}`;
    updateUploadProgress(index, files.length, file.name);
    try {
      const data = await requestPdfIndex(
        user.user_id,
        file,
        chunkSize,
        chunkOverlap
      );
      const result = { file, data, status: "success", linkResult: null };

      if (targetKnowledgeBaseId) {
        try {
          result.linkResult = await linkDocumentToKnowledgeBase(
            targetKnowledgeBaseId,
            data.document_id
          );
        } catch (error) {
          result.status = "partial";
          result.error = `索引成功，但加入知识库失败：${error.message}`;
        }
      }
      results.push(result);
    } catch (error) {
      results.push({ file, status: "error", error: error.message });
    }
    updateUploadProgress(index + 1, files.length, file.name);
  }

  const indexedResults = results.filter(
    (result) => result.status !== "error"
  );
  const scopedResults = results.filter(
    (result) => result.status === "success"
  );

  try {
    if (targetKnowledgeBaseId && scopedResults.length) {
      await finalizeKnowledgeBaseUpload(
        user,
        targetKnowledgeBaseId,
        scopedResults
      );
    } else if (!targetKnowledgeBaseId && indexedResults.length) {
      await finalizeStandaloneUpload(user, indexedResults);
    } else {
      await loadDocuments();
    }
  } catch (error) {
    results.push({
      file: { name: "上传后状态同步" },
      status: "partial",
      error: error.message,
    });
  } finally {
    renderBatchUploadResults(results);
    updateUploadProgress(files.length, files.length);
    const failedCount = results.filter(
      (result) => result.status === "error"
    ).length;
    const partialCount = results.filter(
      (result) => result.status === "partial"
    ).length;
    const completedCount = files.length - failedCount;
    const reusedSingleFile = files.length === 1
      && indexedResults[0]?.data?.existing_document;
    const statusType = failedCount === files.length
      ? "error"
      : failedCount || partialCount
        ? "info"
        : "success";
    setStatus(
      "indexStatus",
      `${reusedSingleFile ? "检测到该文件已经索引，已复用现有资源。" : ""}批量处理完成：${completedCount} 个已索引，${failedCount} 个失败${partialCount ? `，${partialCount} 个未完成知识库关联` : ""}。`,
      statusType
    );
    button.disabled = false;
    fileInput.disabled = false;
    $("uploadKnowledgeBaseSelect").disabled = false;
    $("chunkSize").disabled = false;
    $("chunkOverlap").disabled = false;
    renderSelectedPdfFiles();
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
    const response = await authorizedFetch(url, { method: "POST" });
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
    const response = await authorizedFetch(`${getApiBase()}/conversations/${encodeURIComponent(conversationId)}/ask`, {
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
    clearStatus("answerStatus");
    await loadConversations();
  } catch (error) {
    setStatus("answerStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  $("authForm").addEventListener("submit", handleAuthSubmit);
  $("registerButton").addEventListener("click", handleRegister);
  $("logoutButton").addEventListener("click", logout);
  $("newConversationButton").addEventListener("click", startBlankConversation);
  $("createKnowledgeBaseButton").addEventListener(
    "click",
    handleCreateKnowledgeBase
  );
  $("indexButton").addEventListener("click", indexPdf);
  $("pdfFile").addEventListener("change", renderSelectedPdfFiles);
  $("searchButton").addEventListener("click", searchChunks);
  $("answerButton").addEventListener("click", answerQuestion);
  $("newDocumentConversationButton").addEventListener("click",startNewConversation);
  $("newKnowledgeBaseConversationButton").addEventListener(
    "click",
    startKnowledgeBaseConversation
  );
  $("knowledgeBaseList").addEventListener("click", (event) => {
    const item = event.target.closest("[data-kb-id]");
    if (!item) return;
    selectKnowledgeBase(item.dataset.kbId);
  });
  $("knowledgeBaseDocumentList").addEventListener("change", (event) => {
    const input = event.target.closest("[data-kb-document-id]");
    if (!input) return;
    const documentId = input.dataset.kbDocumentId;
    if (input.checked) {
      selectedKnowledgeBaseDocumentIds.add(documentId);
    } else {
      selectedKnowledgeBaseDocumentIds.delete(documentId);
    }
  });
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
  try {
    const response = await authorizedFetch(`${getApiBase()}/auth/me`);
    const user = await parseResponse(response);
    saveCurrentUser({
      ...user,
      access_token: savedUser.access_token,
    });
    showAppView();
    await loadKnowledgeBases();
    await loadDocuments();
    await loadConversations();
  } catch {
    logout();
  }
}

initApp();
