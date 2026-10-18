/* Research tasks share authentication and document scope with the existing workspace. */
const ResearchUI = (() => {
  const STORAGE_PREFIX = "docrag.researchTasks.";
  const POLL_MS = 2000;
  const labels = {
    ready: "待启动", queued: "排队中", cancelled: "已取消", running: "调查中", completed: "报告已生成 · 待核验",
    partial: "部分完成", stopped: "已停止", failed: "执行失败",
  };
  const actionLabels = {
    search_documents: "检索文档", read_evidence: "读取原文",
    save_research_note: "保存研究笔记", finish_report: "提交报告",
  };
  const stopLabels = {
    report_incomplete: "报告仍有未覆盖的要求，请查看下方缺口。",
    budget_exhausted: "执行次数已达上限，已有证据仍可查看。",
    wall_time_exhausted: "任务达到时间上限，已有证据仍可查看。",
    no_new_evidence: "连续检索没有新增证据，已结束调查。",
    index_generation_changed: "文档索引版本已变化，请重新发起任务。",
    document_scope_unavailable: "文档范围已不可用，请检查文档后重新发起。",
    index_generation_unavailable: "任务固定的索引已不可用，请重新发起。",
    user_cancelled: "任务已取消，已保存的证据仍可查看。",
    interrupted_action_retry_exhausted: "中断动作重试已达上限。",
    process_interrupted: "任务执行被中断。",
  };
  let active = false;
  let task = null;
  let refs = [];
  let busy = false;
  let timer = null;
  let controller = null;
  let readVersion = 0;
  let sessionVersion = 0;
  let lastRender = "";
  let citationOpener = null;
  let historySync = null;

  function stopPolling() {
    clearTimeout(timer);
    timer = null;
    controller?.abort();
    controller = null;
    readVersion += 1;
  }

  function showChat() {
    active = false;
    $("newResearchButton").classList.add("hidden");
    stopPolling();
    closeSource();
    $("researchView").classList.add("hidden");
    $("chatWindow").classList.remove("hidden");
    $("researchScopeHint").classList.add("hidden");
    $("researchStatus").classList.add("hidden");
    $("chatComposer").dataset.mode = "chat";
    $("answerQuery").removeAttribute("maxlength");
    $("answerButton").setAttribute("aria-label", "发送");
    $("chatWorkspaceButton").classList.add("active");
    $("chatWorkspaceButton").setAttribute("aria-pressed", "true");
    $("researchWorkspaceButton").classList.remove("active");
    $("researchWorkspaceButton").setAttribute("aria-pressed", "false");
    updateComposerAvailability();
  }

  function loadRefs() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_PREFIX + currentUser.user_id) || "[]");
      refs = Array.isArray(value) ? value.filter((item) => item && typeof item.task_id === "string").slice(0, 20) : [];
      sortRefs();
    } catch {
      refs = [];
    }
  }

  function sortRefs() {
    const seen = new Set();
    refs = refs.filter((ref) => {
      if (seen.has(ref.task_id)) return false;
      seen.add(ref.task_id);
      return true;
    })
      .sort((a, b) => (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0)
        || b.task_id.localeCompare(a.task_id)).slice(0, 20);
  }

  function remember(data) {
    const ref = {
      task_id: data.task_id, kb_id: data.kb_id, query: data.query,
      status: data.status, created_at: data.created_at, stop_reason: data.stop_reason,
    };
    refs = [ref, ...refs.filter((item) => item.task_id !== ref.task_id)];
    sortRefs();
    try {
      localStorage.setItem(STORAGE_PREFIX + currentUser.user_id, JSON.stringify(refs));
    } catch {
      // Task execution can continue if this browser disallows local storage writes.
    }
    renderHistory();
  }

  function loadHistory() {
    if (!currentUser) return;
    loadRefs();
    renderHistory();
    $("researchHistoryPanel").open = refs.length > 0;
    syncPendingHistory();
  }

  function open(requestedId = null) {
    if (!currentUser) return;
    active = true;
    $("researchView").classList.remove("hidden");
    $("chatWindow").classList.add("hidden");
    $("researchScopeHint").classList.remove("hidden");
    $("researchStatus").classList.remove("hidden");
    $("chatComposer").dataset.mode = "research";
    $("answerQuery").maxLength = 2000;
    $("answerButton").setAttribute("aria-label", "开始深入研究");
    $("researchWorkspaceButton").classList.add("active");
    $("researchWorkspaceButton").setAttribute("aria-pressed", "true");
    $("chatWorkspaceButton").classList.remove("active");
    $("chatWorkspaceButton").setAttribute("aria-pressed", "false");
    loadRefs();
    syncContext();
    renderHistory();
    const latest = typeof requestedId === "string" ? requestedId
      : task?.task_id || refs.find((item) => !currentKnowledgeBaseId || item.kb_id === currentKnowledgeBaseId)?.task_id;
    const loading = latest && !busy ? fetchTask(latest) : Promise.resolve();
    syncPendingHistory(latest);
    return loading;
  }

  function syncPendingHistory(excludeId = task?.task_id) {
    if (!currentUser || historySync) return;
    const session = sessionVersion;
    const owner = currentUser.user_id;
    const pending = refs.filter((ref) => ref.task_id !== excludeId && ["ready", "queued", "running"].includes(ref.status));
    async function update() {
      while (pending.length && session === sessionVersion && currentUser?.user_id === owner) {
        const ref = pending.shift();
        try {
          const response = await authorizedFetch(`${getApiBase()}/research-tasks/${encodeURIComponent(ref.task_id)}`);
          const data = await parseResponse(response);
          if (session !== sessionVersion || currentUser?.user_id !== owner) return;
          const cached = refs.find((item) => item.task_id === ref.task_id);
          if (cached?.status === ref.status && task?.task_id !== ref.task_id) remember(data);
        } catch {
          // A failed refresh preserves the last known status and the saved entry.
        }
      }
    }
    historySync = Promise.all([update(), update()]).finally(() => {
      if (session === sessionVersion) historySync = null;
    });
  }

  function canFollowup() {
    return task && !["ready", "queued", "running"].includes(task.status);
  }

  function newResearch() {
    if (busy) return;
    stopPolling();
    closeSource();
    task = null;
    lastRender = "";
    $("researchTask").innerHTML = '<div class="research-welcome"><h1>开始新的研究</h1><p>沿用左侧知识库和文档选择，在下方输入新的研究问题。</p></div>';
    clearStatus("researchStatus");
    renderHistory();
    syncContext();
  }

  function syncContext() {
    const count = selectedKnowledgeBaseDocumentIds.size || knowledgeBaseDocuments.length;
    $("researchScopeHint").textContent = canFollowup()
      ? `继续研究 · ${getKnowledgeBaseName(task.kb_id)} · ${task.document_ids.length} 篇文档（沿用原范围）`
      : currentKnowledgeBaseId
      ? `${getKnowledgeBaseName(currentKnowledgeBaseId)} · ${count} 篇文档`
      : "请先从左侧选择知识库";
    updateCreateButton();
  }

  function updateCreateButton() {
    if (!active) return;
    $("newResearchButton").classList.toggle("hidden", !task);
    $("newResearchButton").disabled = busy;
    const hasScope = canFollowup() || Boolean(currentKnowledgeBaseId && knowledgeBaseDocuments.length);
    const running = busy || ["ready", "queued", "running"].includes(task?.status);
    $("answerButton").disabled = !hasScope || running;
    $("answerQuery").disabled = !hasScope || running;
    $("answerQuery").placeholder = running ? "研究正在进行，完成后可发起新的研究"
      : canFollowup() ? "基于这份研究继续提问…" : hasScope ? "提出需要多步检索的研究问题…" : "请先从左侧选择已有索引的知识库";
  }

  function renderHistory() {
    $("researchHistory").innerHTML = refs.length ? refs.map((ref) => `
      <div class="history-row"><button type="button" class="conversation-item research-history-item${ref.task_id === task?.task_id ? " active" : ""}"
        data-research-task="${escapeHtml(ref.task_id)}" ${ref.task_id === task?.task_id ? 'aria-current="true"' : ""}
        title="${escapeHtml(ref.query || "研究任务")}&#10;${escapeHtml(formatHistoryDate(ref.created_at))}&#10;任务 ${escapeHtml(ref.task_id)}">
        <span class="conversation-title">${escapeHtml(ref.query || "研究任务")}</span>
        <span class="conversation-meta research-history-status">${escapeHtml(labels[ref.status] || "待查询")}</span>
        <span class="conversation-meta research-history-time">${escapeHtml(formatHistoryDate(ref.created_at))} · #${escapeHtml(ref.task_id.slice(0, 6))}</span>
      </button><button type="button" class="history-delete" data-delete-research="${escapeHtml(ref.task_id)}" aria-label="删除研究：${escapeHtml(ref.query || "研究任务")}" title="删除研究记录">×</button></div>`).join("") : '<p class="empty-list">暂无研究记录。</p>';
  }

  async function deleteResearch(id) {
    if (busy) return;
    const ref = refs.find((item) => item.task_id === id);
    if (["queued", "running"].includes(ref?.status)) {
      await open(id);
      setStatus("researchStatus", "请先取消任务，再删除研究记录。", "error");
      return;
    }
    if (!window.confirm("删除这次研究及其报告、步骤和证据？已有续问会保留。此操作不可撤销。")) return;
    const session = sessionVersion;
    busy = true;
    if (task?.task_id === id) stopPolling();
    updateCreateButton();
    try {
      const response = await authorizedFetch(`${getApiBase()}/research-tasks/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (response.status !== 404) await parseResponse(response);
      if (session !== sessionVersion) return;
      refs = refs.filter((item) => item.task_id !== id);
      localStorage.setItem(STORAGE_PREFIX + currentUser.user_id, JSON.stringify(refs));
      busy = false;
      if (task?.task_id === id) newResearch();
      renderHistory();
    } catch (error) {
      if (session === sessionVersion) setStatus("researchStatus", `删除失败：${error.message}`, "error");
    } finally {
      if (session === sessionVersion) { busy = false; updateCreateButton(); schedulePoll(); }
    }
  }

  function formatHistoryDate(value) {
    if (!value) return "时间未知";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "时间未知" : date.toLocaleString("zh-CN", {
      month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }

  function documentName(id) {
    return task?.document_filenames?.[id] || id;
  }

  function gapText(gap) {
    const gaps = {
      uncited_documents: "缺少文档引用",
      missing_cross_document_comparison: "缺少有跨文档引用的比较结论",
      missing_comparison_limitations: "缺少有跨文档引用的比较限制说明",
    };
    return (gaps[gap.code] || gap.code) + (gap.document_ids?.length
      ? `：${gap.document_ids.map(documentName).join("、")}` : "");
  }

  function renderTask() {
    if (!task) return;
    const serialized = JSON.stringify(task);
    if (serialized === lastRender) return;
    const opened = new Set([...$("researchTask").querySelectorAll("details[open][id]")].map((item) => item.id));
    lastRender = serialized;
    const evidence = task.evidence || [];
    const covered = new Set(evidence.map((item) => item.document_id));
    const report = task.report;
    const events = task.events || [];
    const roles = { finding: "各文档的主要观点", comparison: "方法对比", limitation: "适用边界与局限" };
    const claimGroups = [];
    for (const claim of report?.claims || []) {
      let group = claimGroups.find((item) => item.role === claim.role);
      if (!group) { group = { role: claim.role, claims: [] }; claimGroups.push(group); }
      group.claims.push(claim);
    }
    const citations = (ids) => (ids || []).map((id) => {
      const index = evidence.findIndex((item) => item.evidence_id === id);
      if (index < 0) return '<span class="research-hint">引用不可用</span>';
      const item = evidence[index];
      const label = `${documentName(item.document_id)} · 第 ${item.page_number ?? "未知"} 页`;
      return `<button type="button" class="research-citation" data-research-citation="${index}" title="${escapeHtml(label)}" aria-label="查看来源：${escapeHtml(label)}">${index + 1}</button>`;
    }).join("");
    $("researchTask").innerHTML = `
      ${task.parent_task_id ? `<button type="button" class="research-text-button" data-research-parent="${escapeHtml(task.parent_task_id)}">继续自：${escapeHtml(task.parent_query || "上一轮研究")}</button>` : ""}
      <div class="research-question">${escapeHtml(task.query)}</div>
      <article class="research-answer">
      <div class="research-answer-brand"><span class="research-brand-mark" aria-hidden="true">D</span><strong>DocRAG</strong><span>深入研究</span></div>
      <details id="researchTimeline" class="research-process">
        <summary><span class="research-state" data-state="${escapeHtml(task.status)}">${escapeHtml(labels[task.status] || task.status)}</span><span>研究过程 · ${events.length} 步</span><span class="research-process-caret" aria-hidden="true">⌄</span></summary>
        <p class="research-hint">${escapeHtml(getKnowledgeBaseName(task.kb_id))} · 已检索 ${covered.size} / ${(task.document_ids || []).length} 篇文档 · 模型请求 ${task.model_turns || 0} 次 · 工具执行 ${task.tool_calls || 0} 次</p>
        <ol class="research-timeline">${events.map((event) => {
          const args = event.args_summary || {};
          const scope = args.document_ids?.map(documentName).join("、") || "任务固定的文档范围";
          return `<li class="${event.actor === "harness" ? "harness-step" : ""}">
            <div><strong>${escapeHtml(actionLabels[event.name] || "模型请求")}</strong>
              <span class="research-actor">${event.actor === "harness" ? "Harness 主动补查" : "模型动作"}</span></div>
            ${args.query ? `<p>${escapeHtml(args.query)}</p><small>范围：${escapeHtml(scope)}</small>` : ""}
            ${event.name === "read_evidence" ? `<div class="research-citations">${citations([args.evidence_id])}</div>` : ""}
            <small>${event.result?.evidence_ids ? `返回 ${event.result.evidence_ids.length} 条证据 · ` : ""}${event.elapsed_ms != null ? `${(event.elapsed_ms / 1000).toFixed(1)} 秒` : ""}</small>
            ${event.error_code ? `<p class="research-warning">${escapeHtml(event.error_code)}</p>` : ""}
          </li>`;
        }).join("") || '<li class="research-hint">等待第一个步骤完成。</li>'}</ol>
        <p class="research-code">任务 ${escapeHtml(task.task_id)} · ${escapeHtml(task.stop_reason || "无停止错误")}</p>
      </details>
      ${task.status === "queued" ? '<p class="research-hint" role="status">任务已排队，等待开始。</p>' : ""}
      ${task.status === "running" ? '<p class="research-running" role="status"><span class="research-pulse" aria-hidden="true"></span>正在查阅文档并补充证据…</p>' : ""}
      ${task.stop_reason ? `<p class="research-warning">${escapeHtml(stopLabels[task.stop_reason] || "任务未正常完成，请查看研究过程。")}</p>` : ""}
      ${report ? `<section class="research-report"><h1>${escapeHtml(report.title || "研究报告")}</h1>
        <p class="research-review-note">待核验 · 点击编号可核对引用原文</p>
        ${(report.question_coverage?.gaps || []).length ? `<ul class="research-gaps">${report.question_coverage.gaps.map((gap) => `<li>${escapeHtml(gapText(gap))}</li>`).join("")}</ul>` : ""}
        ${(report.run_limitations || []).filter((item) => item.code === "retrieval_gap").map((item) => `<section class="research-claim research-run-limitations"><h2>本次运行限制</h2><p class="research-hint">本次尚未从以下文档取得证据片段：${escapeHtml((item.document_ids || []).map(documentName).join("、"))}。这不代表原文没有相关内容。</p></section>`).join("")}
        ${claimGroups.map((group) => `<section class="research-claim"><h2>${escapeHtml(roles[group.role] || "研究结论")}</h2>
          ${group.claims.map((claim) => `<p>${claim.status === "unknown" ? '<span class="research-unknown">证据不足</span> ' : ""}${escapeHtml(claim.text)} <span class="research-citations">${citations(claim.evidence_ids)}</span></p>`).join("")}</section>`).join("")}
      </section>` : task.status === "running" ? "" : '<p class="research-hint">尚无报告，可查看已有步骤与来源。</p>'}
      <details id="researchEvidenceSection" class="research-sources">
        <summary>查看 ${evidence.length} 条原文证据</summary>
        <div class="research-source-list">${evidence.map((item, index) => `<button type="button" class="research-source-item" data-research-citation="${index}">
          <span class="research-source-number">${index + 1}</span><span><strong>${escapeHtml(documentName(item.document_id))}</strong><small>第 ${escapeHtml(item.page_number ?? "未知")} 页</small></span></button>`).join("") || '<p class="research-hint">尚无检索证据。</p>'}</div>
      </details>
      <div class="research-footer"><span>${escapeHtml(formatDate(task.created_at))} · ${task.parent_task_id ? "继续研究" : "独立研究"}</span>
        <button type="button" class="research-text-button" data-research-refresh>刷新状态</button>
        ${["queued", "running"].includes(task.status) ? '<button type="button" class="research-text-button" data-research-cancel>取消任务</button>' : ""}
        ${task.status === "ready" ? '<button type="button" class="research-text-button" data-research-run>启动此任务</button>' : ""}</div>
      </article>`;
    $("researchTask").querySelectorAll("details[id]").forEach((item) => { item.open = opened.has(item.id); });
    renderHistory();
    syncContext();
  }

  function schedulePoll() {
    if (active && ["queued", "running"].includes(task?.status) && !document.hidden) {
      timer = setTimeout(() => fetchTask(task.task_id), POLL_MS);
    }
  }

  async function fetchTask(id) {
    stopPolling();
    const version = readVersion;
    const owner = currentUser?.user_id;
    let notFound = false;
    controller = new AbortController();
    try {
      const response = await authorizedFetch(`${getApiBase()}/research-tasks/${encodeURIComponent(id)}`, { signal: controller.signal });
      notFound = response.status === 404;
      const data = await parseResponse(response);
      if (version !== readVersion || currentUser?.user_id !== owner) return;
      if (task?.task_id !== data.task_id) {
        closeSource();
        lastRender = "";
        $("researchTask").replaceChildren();
        $("researchView").scrollTop = 0;
      }
      task = data;
      remember(data);
      renderTask();
      clearStatus("researchStatus");
      schedulePoll();
    } catch (error) {
      if (error.name === "AbortError" || version !== readVersion || currentUser?.user_id !== owner) return;
      setStatus("researchStatus", `无法查询任务：${error.message}。可手动刷新；H2 新任务状态保存在服务端。`, "error");
      if (notFound || task?.task_id !== id) {
        task = null;
        lastRender = "";
        closeSource();
        $("researchTask").innerHTML = `<div class="research-welcome"><h1>任务暂不可用</h1>
          <p class="research-hint">未将查询失败视为任务完成。可点击近期任务重新查询。</p></div>`;
      }
      renderHistory();
      updateCreateButton();
    }
  }

  async function cancelTask() {
    if (busy || !["queued", "running"].includes(task?.status)) return;
    const session = sessionVersion;
    const id = task.task_id;
    stopPolling();
    busy = true;
    updateCreateButton();
    try {
      const response = await authorizedFetch(`${getApiBase()}/research-tasks/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      const data = await parseResponse(response);
      if (session !== sessionVersion || task?.task_id !== id) return;
      task = data;
      remember(data);
      renderTask();
      clearStatus("researchStatus");
    } catch (error) {
      if (session === sessionVersion) setStatus("researchStatus", `取消结果未确认：${error.message}。请刷新任务状态。`, "error");
    } finally {
      if (session === sessionVersion) { busy = false; updateCreateButton(); schedulePoll(); }
    }
  }

  async function runReadyTask() {
    if (busy || task?.status !== "ready") return;
    const session = sessionVersion;
    stopPolling();
    busy = true;
    updateCreateButton();
    try {
      const response = await authorizedFetch(`${getApiBase()}/research-tasks/${encodeURIComponent(task.task_id)}/run`, { method: "POST" });
      const data = await parseResponse(response);
      if (session !== sessionVersion) return;
      task = data;
      remember(data);
      renderTask();
      clearStatus("researchStatus");
      schedulePoll();
    } catch (error) {
      if (session === sessionVersion) setStatus("researchStatus", `启动结果未确认：${error.message}。请刷新任务状态后再操作，系统不会自动重复启动。`, "error");
    } finally {
      if (session === sessionVersion) {
        busy = false;
        updateCreateButton();
      }
    }
  }

  async function submit() {
    if (!active || busy || $("answerButton").disabled) return;
    const query = $("answerQuery").value.trim();
    if (!query || query.length > 2000) {
      setStatus("researchStatus", "请输入 1～2000 字的研究问题。", "error");
      return;
    }
    const session = sessionVersion;
    const payload = canFollowup()
      ? { kb_id: task.kb_id, query, selected_document_ids: task.document_ids, parent_task_id: task.task_id }
      : { kb_id: currentKnowledgeBaseId, query,
          selected_document_ids: selectedKnowledgeBaseDocumentIds.size ? [...selectedKnowledgeBaseDocumentIds] : null };
    busy = true;
    stopPolling();
    clearStatus("researchStatus");
    updateCreateButton();
    try {
      const response = await authorizedFetch(`${getApiBase()}/research-tasks`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await parseResponse(response);
      if (session !== sessionVersion) return;
      closeSource();
      task = data;
      lastRender = "";
      remember(data);
      renderTask();
      if (active && $("answerQuery").value.trim() === query) $("answerQuery").value = "";
      $("researchHistoryPanel").open = true;
      busy = false;
      await runReadyTask();
    } catch (error) {
      if (session === sessionVersion) setStatus("researchStatus", `未能创建任务：${error.message}`, "error");
    } finally {
      if (session === sessionVersion) {
        busy = false;
        updateCreateButton();
      }
    }
  }

  function reset() {
    sessionVersion += 1;
    historySync = null;
    showChat();
    task = null;
    refs = [];
    busy = false;
    lastRender = "";
    closeSource();
    $("researchTask").innerHTML = '<div class="research-welcome"><h1>让文档回答更深入的问题</h1><p>选好左侧知识库，在下方提出问题。系统会分步检索，补查缺失证据，再整理回答。</p><button type="button" class="research-suggestion" data-research-suggestion="比较所选文档各自解决的问题、方法与局限。">比较论文的方法与局限 <span aria-hidden="true">↗</span></button><button type="button" class="research-suggestion" data-research-suggestion="归纳所选文档的共同结论，指出证据不足的地方。">归纳共识与证据缺口 <span aria-hidden="true">↗</span></button></div>';
    clearStatus("researchStatus");
    renderHistory();
  }

  function closeSource() {
    $("researchSourceDrawer").classList.add("hidden");
    $("researchSourceBackdrop").classList.add("hidden");
    $("researchSourceContent").innerHTML = "";
    citationOpener?.isConnected && citationOpener.focus();
    citationOpener = null;
  }

  function openSource(index, opener) {
    const evidence = task?.evidence?.[index];
    if (!evidence) return;
    citationOpener = opener;
    $("researchSourceContent").innerHTML = `<p class="research-source-filename">${escapeHtml(documentName(evidence.document_id))}</p>
      <p class="research-hint">第 ${escapeHtml(evidence.page_number ?? "未知")} 页 · 来源 ${index + 1}</p>
      <blockquote>${escapeHtml(evidence.text)}</blockquote>
      <details class="research-source-identifiers"><summary>证据标识</summary><p class="research-code">${escapeHtml(evidence.evidence_id)}</p></details>`;
    $("researchSourceDrawer").classList.remove("hidden");
    $("researchSourceBackdrop").classList.remove("hidden");
    $("researchSourceClose").focus();
  }

  function init() {
    $("researchHistoryPanel").addEventListener("toggle", () => {
      if ($("researchHistoryPanel").open) syncPendingHistory();
    });
    $("newResearchButton").addEventListener("click", newResearch);
    $("chatWorkspaceButton").addEventListener("click", showChat);
    $("researchWorkspaceButton").addEventListener("click", open);
    $("researchSourceClose").addEventListener("click", closeSource);
    $("researchSourceBackdrop").addEventListener("click", closeSource);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeSource();
      if (event.key === "Tab" && !$("researchSourceDrawer").classList.contains("hidden")) {
        const focusable = [...$("researchSourceDrawer").querySelectorAll("button, summary")];
        const first = focusable[0], last = focusable.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    });
    $("researchHistory").addEventListener("click", (event) => {
      const remove = event.target.closest("[data-delete-research]");
      if (remove) { deleteResearch(remove.dataset.deleteResearch); return; }
      const button = event.target.closest("[data-research-task]");
      if (button && !busy) open(button.dataset.researchTask);
    });
    $("researchTask").addEventListener("click", (event) => {
      if (event.target.closest("[data-research-refresh]") && task && !busy) fetchTask(task.task_id);
      if (event.target.closest("[data-research-run]")) runReadyTask();
      if (event.target.closest("[data-research-cancel]")) cancelTask();
      const parent = event.target.closest("[data-research-parent]");
      if (parent && !busy) fetchTask(parent.dataset.researchParent);
      const suggestion = event.target.closest("[data-research-suggestion]");
      if (suggestion) { $("answerQuery").value = suggestion.dataset.researchSuggestion; $("answerQuery").focus(); }
      const citation = event.target.closest("[data-research-citation]");
      if (citation) openSource(Number(citation.dataset.researchCitation), citation);
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopPolling();
      else if (active && ["queued", "running"].includes(task?.status)) fetchTask(task.task_id);
    });
  }

  return { init, open, showChat, loadHistory, syncContext, reset, stopPolling, submit, isActive: () => active };
})();
