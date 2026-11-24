'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const api = window.inferaStudio;
  const textReveal = window.InferaStudioTextReveal;
  const startup = $('startup');
  const startupStatus = $('startup-status');
  const draft = $('draft');
  const filePicker = $('file-picker');
  const imagePicker = $('image-picker');
  const maxFiles = 8;
  const maxFileBytes = 256 * 1024;
  const maxTotalFileBytes = 512 * 1024;
  const maxImages = 4;
  const maxImageBytes = 4 * 1024 * 1024;
  const maxTotalImageBytes = 8 * 1024 * 1024;
  const utf8 = new TextDecoder('utf-8', { fatal: true });
  const utf8Encoder = new TextEncoder();
  const transcript = $('transcript');
  const sidebar = $('sidebar');
  const historyMenu = $('history-menu');
  const pinButton = $('pin-chat');
  const pinError = $('pin-error');
  const statsPanel = $('system-stats');
  const statsToggle = $('system-stats-toggle');
  const statRows = [...statsPanel.querySelectorAll('[data-stat]')];
  const gpuPicker = $('stats-gpu');
  const sidebarOpen = $('sidebar-open');
  const sidebarClose = $('sidebar-close');
  const drafts = new Map();
  const listeners = new Map();
  const cards = new Map();
  const state = {
    catalog: null, chat: null, messages: [], history: [], pinnedLimit: 0, budget: null,
    server: 'Opening', running: false, available: true, navigation: false, features: null,
  };
  let revision = 0;
  let sequence = 0;
  let operation;
  let budgetRequest;
  let budgetTimer;
  let searchTimer;
  let searchTicket = 0;
  let historyMenuTarget;
  let noticeKind = '';
  let healthPending = false;
  let statsPending = false;
  let lastStats = null;
  let selectedGpu = '';
  let performanceSnapshot = null;
  let performanceUpdatedAt = 0;
  let generation;
  let draftFiles = [];
  let webSearch = false;
  let readingFiles = false;
  let choosingWorkspace = false;

  const voice = window.InferaStudioVoice.create({
    draft, call, startRequest: start, changed: controls,
    setup: () => dialogs.preferences(state.catalog, { section: 'voice' }),
  });
  const dialogs = window.InferaStudioDialogs.create({
    call, start, savePreferences, rescan, selectModel, createChat, renameChat, deleteChat,
    refreshModels: async () => adoptCatalog(await call('rescan')),
    applySettings, changeContext, unload, reload: () => openChat(state.chat.id), fork,
    configureVoice: voice.configure, voiceSource: voice.source,
  });

  function start(action, params = {}, receive) {
    const id = ++sequence;
    if (receive) listeners.set(id, receive);
    const response = api.request(id, action, params).finally(() => listeners.delete(id));
    return { id, response };
  }

  async function call(action, params) {
    const answer = await start(action, params).response;
    if (!answer.ok) throw new Error(answer.error || 'Operation stopped.');
    return answer.value;
  }

  api.onEvent(({ id, event, value }) => listeners.get(id)?.(event, value));
  api.onExit((error) => {
    state.available = false;
    renderStats();
    setStatus('Error', 'Backend closed');
    notice(error);
  });

  function model() {
    return state.catalog?.models.find((item) => item.path === state.chat?.model && !item.issue);
  }

  function renderStats() {
    const preferences = state.catalog?.preferences;
    statsToggle.hidden = statsPanel.hidden = !state.available || !preferences?.visible_stats.length;
    if (statsPanel.hidden) {
      statsPanel.hidePopover();
      return;
    }
    const devices = lastStats?.gpus ?? [];
    const gpu = devices.find((device) => gpuKey(device) === selectedGpu) ?? lastStats?.gpu;
    gpuPicker.hidden = devices.length < 2 || !preferences.visible_stats.some((key) => key.startsWith('gpu_'));
    if (!gpuPicker.hidden) {
      const keys = devices.map(gpuKey);
      if (keys.join('|') !== gpuPicker.dataset.devices) {
        gpuPicker.replaceChildren(...devices.map((device) => new Option(
          device.name || device.vendor + ' ' + device.device, gpuKey(device),
        )));
        gpuPicker.dataset.devices = keys.join('|');
      }
      gpuPicker.value = gpuKey(gpu);
    }
    const percent = (value) => value == null ? '—' : Math.round(value) + '%';
    const degrees = (value) => value == null ? '—' : Math.round(value) + ' °C';
    const gib = (bytes) => bytes == null ? '—' : (bytes / 2 ** 30).toFixed(1);
    const memory = (used, total) => total != null ? gib(used) + ' / ' + gib(total) + ' GiB'
      : used != null ? gib(used) + ' GiB' : '—';
    const values = {
      gpu_load: percent(gpu?.load),
      gpu_vram: memory(gpu?.vram_used, gpu?.vram_total),
      gpu_temp: degrees(gpu?.temperature_c),
      cpu_load: percent(lastStats?.cpu_load),
      cpu_temp: degrees(lastStats?.cpu_temperature_c),
      ram: memory(lastStats?.ram_used, lastStats?.ram_total),
    };
    for (const row of statRows) {
      const key = row.dataset.stat;
      row.hidden = !preferences.visible_stats.includes(key);
      row.querySelector('strong').textContent = values[key];
      const device = key.startsWith('gpu_') && gpu ? ' · ' + (gpu.name || gpu.vendor + ' GPU') + ' · ' + gpu.device : '';
      row.title = state.catalog.statLabels[key] + device + (values[key] === '—' ? ' · Unavailable' : '');
    }
    const occupancy = lastStats?.kv_occupancy;
    const kv = $('kv-occupancy');
    kv.hidden = occupancy == null || !['Ready', 'Generating'].includes(state.server);
    if (!kv.hidden) kv.querySelector('strong').textContent = percentOf(occupancy, 1);
  }

  function gpuKey(device) {
    return device.vendor + ':' + device.device;
  }

  gpuPicker.addEventListener('change', () => {
    selectedGpu = gpuPicker.value;
    renderStats();
  });

  async function refreshStats() {
    if (!state.available || statsPending || document.hidden || !state.catalog?.preferences.visible_stats.length) return;
    statsPending = true;
    try {
      lastStats = await call('system_stats');
    } catch {
      lastStats = null;
    } finally {
      statsPending = false;
      renderStats();
    }
  }

  function notice(text = '', kind = '') {
    noticeKind = kind;
    $('notice-text').textContent = text;
    $('notice').hidden = !text;
    const action = { library: 'Model library', server: 'Diagnostics', settings: 'Settings',
      context: 'Context', model: 'Model settings' }[kind] || '';
    $('notice-action').hidden = !action;
    $('notice-action').textContent = action;
  }

  function renderServerStatus() {
    const backend = state.chat?.settings.backend || state.catalog?.preferences.backend;
    const label = ['Inference server', state.server, backend].filter(Boolean).join(' · ');
    const server = $('server');
    server.title = label;
    server.setAttribute('aria-label', label);
    $('status-dot').dataset.state = state.server;
  }

  function setStatus(status, detail = '') {
    state.server = status;
    renderServerStatus();
    $('detail').textContent = detail;
    $('detail').title = detail;
    $('loading').toggleAttribute('hidden', status !== 'Loading');
    if (!['Ready', 'Generating'].includes(status)) $('kv-occupancy').hidden = true;
    controls();
  }

  function controls() {
    const ready = Boolean(state.catalog && state.available);
    const busy = state.running || state.navigation || Boolean(generation);
    for (const id of ['new-chat', 'search', 'settings', 'models', 'server']) $(id).disabled = !ready;
    $('chat-actions').disabled = !ready || !state.chat;
    $('model-settings').disabled = !ready || !state.chat || busy;
    $('context').disabled = !ready || !state.chat || busy;
    $('advanced-stats').disabled = !state.chat || state.navigation;
    $('stop').hidden = !ready || !(generation?.request || operation);
    draft.disabled = !ready || !state.chat || busy || choosingWorkspace;
    const dictating = voice.sync(!draft.disabled);
    const workspace = state.chat?.workspace || '';
    $('mode-ask').disabled = $('mode-agent').disabled = $('workspace').disabled = draft.disabled;
    $('mode-ask').setAttribute('aria-pressed', String(!workspace));
    $('mode-agent').setAttribute('aria-pressed', String(Boolean(workspace)));
    $('workspace').hidden = !workspace;
    $('workspace').title = workspace ? `Change project folder · ${workspace}` : '';
    $('workspace-name').textContent = workspace.split('/').at(-1) || workspace;
    draft.placeholder = workspace ? 'Give Agent a task for this project…' : 'Message your local model…';
    const selected = model();
    const tavilyConfigured = state.catalog?.preferences.has_tavily_api_key;
    const searchButton = $('web-search');
    searchButton.disabled = draft.disabled || !selected;
    if (tavilyConfigured) {
      searchButton.removeAttribute('aria-haspopup');
      searchButton.setAttribute('aria-label', 'Web search');
      searchButton.setAttribute('aria-pressed', String(webSearch));
      searchButton.title = webSearch
        ? 'Web search on for this message · Queries may be sent to Tavily'
        : 'Web search off · Enable for this message';
    } else {
      searchButton.removeAttribute('aria-pressed');
      searchButton.setAttribute('aria-haspopup', 'dialog');
      searchButton.setAttribute('aria-label', 'Set up web search');
      searchButton.title = 'Set up web search with a Tavily API key';
    }
    const settings = state.chat?.settings;
    const images = draftFiles.filter((file) => file.media_type).length;
    const vision = Boolean(selected && state.features?.vision);
    const thinking = Boolean(selected && (state.features?.thinking || selected.thinking));
    const mtp = Boolean(selected && (state.features?.mtpAvailable ?? (selected.mtp || settings.mtp_draft)));
    $('attach').disabled = draft.disabled || readingFiles || draftFiles.length - images >= maxFiles;
    $('vision').disabled = draft.disabled || readingFiles || !selected || (vision && images >= maxImages);
    $('vision').dataset.active = String(vision);
    $('vision').title = vision ? 'Attach PNG or JPEG images' : 'Set up a vision projector';
    $('mtp').disabled = draft.disabled || !mtp;
    $('mtp').setAttribute('aria-pressed', String(Boolean(selected && state.features?.mtpEnabled)));
    $('mtp').title = state.features?.mtp || (mtp ? 'Toggle automatic multi-token prediction'
      : 'Multi-token prediction requires a compatible model and draft head');
    $('thinking').disabled = draft.disabled || !thinking;
    $('thinking').setAttribute('aria-pressed', String(thinking && settings.thinking === 'On'));
    $('thinking').title = thinking ? 'Toggle Thinking for this conversation' : 'Thinking requires a compatible chat template';
    $('send').disabled = draft.disabled || dictating || readingFiles || !selected || (images > 0 && !vision)
      || (!draft.value.trim() && !draftFiles.length);
    for (const button of $('draft-files').querySelectorAll('button')) button.disabled = draft.disabled || readingFiles;
    for (const card of cards.values()) card.action.disabled = !ready || busy;
  }

  function renderDraftFiles() {
    const list = $('draft-files');
    const rows = draftFiles.map((file, index) => {
      const row = document.createElement('li');
      const name = document.createElement('span');
      name.textContent = file.name;
      name.title = file.name;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'quiet';
      remove.textContent = '×';
      remove.setAttribute('aria-label', 'Remove ' + file.name);
      remove.addEventListener('click', () => {
        draftFiles = draftFiles.filter((_, position) => position !== index);
        renderDraftFiles();
        scheduleBudget();
        $(file.media_type ? 'vision' : 'attach').focus();
      });
      row.append(name, remove);
      return row;
    });
    list.replaceChildren(...rows);
    list.hidden = !rows.length;
    controls();
  }

  function readImage(file) {
    const mediaType = /\.png$/i.test(file.name) ? 'image/png'
      : /\.jpe?g$/i.test(file.name) ? 'image/jpeg' : file.type;
    if (!['image/png', 'image/jpeg'].includes(mediaType)) throw new Error();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = reject;
      reader.onload = () => resolve({ name: file.name, content: reader.result.split(',')[1], media_type: mediaType });
      reader.readAsDataURL(file);
    });
  }

  async function attachFiles(selected, images = false) {
    if ($(images ? 'vision' : 'attach').disabled || !selected.length) return;
    const token = revision;
    const existing = draftFiles.filter((file) => Boolean(file.media_type) === images);
    const available = (images ? maxImages : maxFiles) - existing.length;
    const sizeLimit = images ? maxImageBytes : maxFileBytes;
    const totalLimit = images ? maxTotalImageBytes : maxTotalFileBytes;
    const added = [];
    const errors = [];
    let bytes = existing.reduce((total, file) => total + (images
      ? (file.content.length * 3 / 4) - (file.content.endsWith('==') ? 2 : file.content.endsWith('=') ? 1 : 0)
      : utf8Encoder.encode(file.content).length), 0);
    readingFiles = true;
    controls();
    try {
      for (const file of selected.slice(0, available)) {
        if (token !== revision) return;
        if (file.size > sizeLimit) {
          errors.push(file.name + (images ? ' exceeds 4 MiB.' : ' exceeds 256 KiB.'));
          continue;
        }
        if (bytes + file.size > totalLimit) {
          errors.push(file.name + (images ? ' would exceed the 8 MiB total.' : ' would exceed the 512 KiB total.'));
          continue;
        }
        try {
          if (images) {
            added.push(await readImage(file));
          } else {
            const content = utf8.decode(await file.arrayBuffer());
            const nonText = /[\x00-\x08\x0b\x0e-\x1f\x7f]/;
            if (content.startsWith('%PDF-') || nonText.test(content) || nonText.test(file.name)) throw new Error();
            added.push({ name: file.name, content });
          }
          bytes += file.size;
        } catch {
          errors.push(file.name + (images ? ' could not be read as PNG or JPEG.' : ' could not be read as UTF-8 text.'));
        }
      }
      if (token !== revision) return;
      if (selected.length > available) errors.push(images ? 'Attach up to four images per message.' : 'Attach up to eight files per message.');
      draftFiles = [...draftFiles, ...added];
      if (errors.length) notice(errors.join(' '), 'attachment');
      else if (noticeKind === 'attachment') notice();
      scheduleBudget();
    } finally {
      readingFiles = false;
      renderDraftFiles();
    }
  }

  function updateModel() {
    const selected = model();
    const unavailable = state.catalog?.models.find((item) => item.path === state.chat?.model);
    const name = selected?.name || (state.chat?.model
      ? 'Unavailable · ' + (unavailable?.name || state.chat.model.split('/').at(-1))
      : 'Select a model');
    $('active-model').textContent = name;
    $('models').title = name;
    $('models').dataset.selected = Boolean(selected);
    renderServerStatus();
    $('welcome-note').textContent = selected
      ? 'Send a message to ' + selected.name + '.'
      : 'Choose a local model below to get started.';
    if (!$('model-menu').hidden) renderModelOptions();
  }

  function sizeModelMenu() {
    const menu = $('model-menu');
    if (!menu.hidden) {
      const above = $('models').getBoundingClientRect().top;
      menu.style.maxHeight = `${Math.max(100, Math.floor(above - 20))}px`;
    }
  }

  function modelMenu(open) {
    $('model-menu').hidden = !open;
    $('models').setAttribute('aria-expanded', String(open));
    if (open) {
      $('model-search').value = '';
      renderModelOptions();
      sizeModelMenu();
      $('model-search').focus();
    }
  }

  function renderModelOptions() {
    const query = $('model-search').value.trim().toLowerCase();
    const options = $('model-options');
    options.replaceChildren();
    for (const item of state.catalog?.models || []) {
      if (item.issue || ![item.name, item.architecture, item.quant, item.path]
        .some((value) => value.toLowerCase().includes(query))) continue;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'quiet model-option';
      button.setAttribute('aria-current', String(state.chat?.model === item.path));
      button.title = item.path;
      button.append(item.name);
      const extra = document.createElement('small');
      extra.textContent = [item.architecture, item.quant].filter(Boolean).join(' · ');
      button.append(extra);
      button.addEventListener('click', () => {
        modelMenu(false);
        $('models').focus();
        void selectModel(item.path);
      });
      options.append(button);
    }
    if (!options.childElementCount) {
      const empty = document.createElement('p');
      empty.className = 'history-meta';
      empty.textContent = query ? 'No matching models' : 'No models available';
      options.append(empty);
    }
  }

  function adoptCatalog(catalog) {
    state.catalog = catalog;
    if (!catalog.preferences.has_tavily_api_key) webSearch = false;
    document.documentElement.dataset.theme = catalog.preferences.theme;
    updateModel();
    controls();
    renderStats();
    void refreshStats();
  }

  function showHistoryMenu(chat, trigger) {
    if (historyMenu.matches(':popover-open')) {
      const sameTrigger = historyMenuTarget.trigger === trigger;
      historyMenu.hidePopover();
      if (sameTrigger) return;
    }
    historyMenuTarget = { chat, trigger };
    $('pin-chat-label').textContent = chat.pinned_at ? 'Unpin' : 'Pin to top';
    pinError.hidden = true;
    pinButton.disabled = false;
    trigger.focus({ preventScroll: true });
    historyMenu.showPopover({ source: trigger });
    trigger.setAttribute('aria-expanded', 'true');
  }

  function renderHistoryList(list, chats) {
    const rows = document.createDocumentFragment();
    for (const chat of chats) {
      const row = document.createElement('li');
      row.className = 'history-entry';
      row.dataset.chatId = chat.id;
      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'quiet history-name';
      open.textContent = chat.title;
      open.title = chat.title;
      open.addEventListener('click', () => {
        if (chat.id !== state.chat?.id || state.navigation) void openChat(chat.id);
      });
      const menu = document.createElement('button');
      menu.type = 'button';
      menu.className = 'quiet icon-button history-actions';
      menu.append($('chat-actions').firstElementChild.cloneNode(true));
      menu.setAttribute('aria-label', 'Actions for ' + chat.title);
      menu.setAttribute('aria-controls', 'history-menu');
      menu.setAttribute('aria-expanded', 'false');
      menu.addEventListener('click', () => showHistoryMenu(chat, menu));
      row.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        showHistoryMenu(chat, menu);
      });
      row.append(open, menu);
      rows.append(row);
    }
    list.replaceChildren(rows);
  }

  function selectHistoryRow() {
    const previous = $('history').querySelector('.history-entry.active');
    const current = $('history').querySelector(`.history-entry[data-chat-id="${state.chat?.id}"]`);
    if (previous === current) return;
    if (previous) {
      previous.classList.remove('active');
      previous.firstElementChild.removeAttribute('aria-current');
    }
    if (current) {
      current.classList.add('active');
      current.firstElementChild.setAttribute('aria-current', 'true');
    }
  }

  function renderHistory() {
    const filtering = Boolean($('search').value.trim());
    const pinned = state.history.filter((chat) => chat.pinned_at);
    pinned.sort((a, b) => b.pinned_at.localeCompare(a.pinned_at));
    const recent = state.history.filter((chat) => !chat.pinned_at);
    const menuRow = historyMenuTarget?.trigger.closest('.history-entry');
    const focused = menuRow ? historyMenuTarget.trigger : document.activeElement;
    const focusedId = focused?.closest('.history-entry')?.dataset.chatId;
    if (menuRow) historyMenu.hidePopover();

    $('pinned-history').hidden = !pinned.length;
    $('recent-history').hidden = !recent.length;
    $('pinned-count').hidden = filtering;
    $('pinned-count').textContent = pinned.length + '/' + state.pinnedLimit;
    $('pinned-count').title = pinned.length + ' of ' + state.pinnedLimit + ' pin slots used';
    renderHistoryList($('pinned-chats'), pinned);
    renderHistoryList($('recent-chats'), recent);
    selectHistoryRow();

    const empty = $('history-empty');
    empty.hidden = Boolean(state.history.length);
    empty.textContent = filtering ? 'No matching chats' : 'No chats yet';
    if (focusedId) {
      const control = focused.classList.contains('history-actions') ? '.history-actions' : '.history-name';
      const replacement = $('history').querySelector(`[data-chat-id="${focusedId}"] ${control}`);
      (replacement ?? $('search')).focus();
    }
  }

  async function refreshHistory() {
    const ticket = ++searchTicket;
    try {
      const history = await call('search', { query: $('search').value.trim() });
      if (ticket !== searchTicket) return;
      state.history = history;
      renderHistory();
    } catch (error) {
      if (ticket === searchTicket) notice(error.message);
    }
  }

  function atBottom() {
    return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 60;
  }

  function scrollBottom() {
    transcript.scrollTop = transcript.scrollHeight;
  }

  function setSidebarExpanded(expanded) {
    if (sidebar.hidden === !expanded) return;
    const follow = atBottom();
    if (!expanded && sidebar.contains(historyMenuTarget?.trigger ?? null)) historyMenu.hidePopover();
    sidebar.hidden = !expanded;
    sidebarOpen.hidden = expanded;
    sidebarOpen.setAttribute('aria-expanded', String(expanded));
    sidebarClose.setAttribute('aria-expanded', String(expanded));
    (expanded ? sidebarClose : sidebarOpen).focus();
    if (follow) scrollBottom();
  }

  function responseStep(agent) {
    const item = document.createElement('div');
    item.className = 'response-step';
    const reasoning = document.createElement('details');
    reasoning.className = 'message-reasoning';
    reasoning.open = Boolean(state.catalog?.preferences.show_reasoning);
    const title = document.createElement('summary');
    title.textContent = 'Reasoning';
    const thoughts = document.createElement('div');
    thoughts.className = 'message-body';
    reasoning.append(title, thoughts);
    const body = document.createElement('div');
    body.className = 'message-body';
    const activity = document.createElement('ul');
    activity.className = 'tool-activity';
    activity.setAttribute('aria-label', 'Project file activity');
    const work = document.createElement('div');
    work.className = 'step-work';
    work.append(activity, reasoning);
    if (agent) {
      item.classList.add('agent-step');
      item.append(body, work);
    } else {
      item.append(work, body);
    }
    return { item, reasoning, thoughts, body, work, activity, tools: [] };
  }

  function toolActivityRow({ name, path, ok }) {
    const labels = {
      read_file: ['Read', 'Could not read'], create_file: ['Created', 'Could not create'],
      search_files: ['Searched', 'Could not search'],
      web_search: ['Searched the web with Tavily', 'Tavily web search failed'],
    };
    const verbs = Object.hasOwn(labels, name) ? labels[name] : ['Ran tool', 'Tool call failed'];
    const entry = document.createElement('li');
    entry.dataset.ok = String(ok);
    const mark = document.createElement('span');
    mark.className = 'tool-result-mark';
    mark.setAttribute('aria-hidden', 'true');
    mark.textContent = ok ? '✓' : '!';
    const label = document.createElement('span');
    label.textContent = verbs[ok ? 0 : 1];
    if (name === 'search_files' && path === '.') {
      label.append(' project');
    } else if (path) {
      const target = document.createElement('code');
      target.textContent = path;
      label.append(' ', target);
    }
    entry.append(mark, label);
    return entry;
  }

  function updateResponse(card, message, reveal) {
    const agent = Boolean(message.agentProgress || message.tool_activity?.length);
    const groups = [];
    for (const tool of message.tool_activity || []) {
      // Activity saved before offsets were available stays above the response.
      const contentEnd = tool.content_end ?? 0;
      const reasoningEnd = tool.reasoning_end ?? 0;
      const previous = groups.at(-1);
      if (previous?.contentEnd === contentEnd && previous.reasoningEnd === reasoningEnd) {
        previous.tools.push(tool);
      } else {
        groups.push({ contentEnd, reasoningEnd, tools: [tool] });
      }
    }
    groups.push({ contentEnd: message.content.length, reasoningEnd: message.reasoning.length, tools: [] });

    let contentStart = 0;
    let reasoningStart = 0;
    for (const [index, { contentEnd, reasoningEnd, tools }] of groups.entries()) {
      if (!card.steps[index]) {
        const step = responseStep(agent);
        card.steps.push(step);
        card.body.append(step.item);
      }
      const step = card.steps[index];
      const content = message.content.slice(contentStart, contentEnd);
      const reasoning = message.reasoning.slice(reasoningStart, reasoningEnd);
      if (step.content !== content) window.InferaStudioMarkdown.render(step.body, content, reveal);
      if (step.thought !== reasoning) window.InferaStudioMarkdown.render(step.thoughts, reasoning, reveal);
      step.content = content;
      step.thought = reasoning;
      step.reasoning.hidden = !reasoning.trim();
      step.body.hidden = !content.trim();
      if (step.tools.length !== tools.length || step.tools.some((tool, i) => tool !== tools[i])) {
        const rows = tools.map(toolActivityRow);
        step.activity.replaceChildren(...rows);
        if (reveal) for (const row of rows.slice(step.tools.length)) textReveal.play(row);
        step.tools = tools;
      }
      step.activity.hidden = !tools.length;
      step.work.hidden = step.reasoning.hidden && step.activity.hidden;
      step.item.hidden = step.work.hidden && step.body.hidden;
      contentStart = contentEnd;
      reasoningStart = reasoningEnd;
    }
    while (card.steps.length > groups.length) card.steps.pop().item.remove();
  }

  function messageCard(message) {
    const item = document.createElement('article');
    const user = message.role === 'user';
    item.className = 'message ' + (user ? 'user' : 'assistant');
    if (!user) {
      const speaker = document.createElement('div');
      speaker.className = 'speaker';
      const icon = document.createElement('img');
      icon.src = './icon.png';
      icon.alt = '';
      icon.width = 19;
      icon.height = 19;
      speaker.append(icon, 'Infera Studio');
      item.append(speaker);
    } else {
      item.setAttribute('aria-label', 'You');
    }
    const body = document.createElement('div');
    body.className = user ? 'message-body' : 'assistant-flow';
    const files = document.createElement('div');
    files.className = 'message-files';
    for (const file of message.attachments || []) {
      const details = document.createElement('details');
      const name = document.createElement('summary');
      name.textContent = file.name;
      const image = ['image/png', 'image/jpeg'].includes(file.media_type);
      const content = document.createElement(image ? 'img' : 'pre');
      if (image) {
        content.className = 'message-image';
        content.loading = 'lazy';
        content.alt = file.name;
        content.src = `data:${file.media_type};base64,${file.content}`;
      } else {
        content.textContent = file.content;
      }
      details.append(name, content);
      files.append(details);
    }
    files.hidden = !files.childElementCount;
    const status = document.createElement('div');
    status.className = 'message-state muted';
    status.setAttribute('role', 'status');
    const speed = document.createElement('div');
    speed.className = 'message-speed muted';
    const performance = document.createElement('details');
    performance.className = 'message-performance';
    const performanceTitle = document.createElement('summary');
    performanceTitle.textContent = 'Request performance';
    const performanceBody = document.createElement('div');
    performanceBody.className = 'performance-details';
    const graphs = document.createElement('button');
    graphs.className = 'quiet analytics-link';
    graphs.type = 'button';
    graphs.textContent = 'View request charts';
    graphs.setAttribute('aria-haspopup', 'dialog');
    graphs.setAttribute('aria-controls', 'inference-analytics');
    graphs.addEventListener('click', () => openAnalytics(graphs, card.message.id));
    performance.append(performanceTitle, performanceBody, graphs);
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    const copy = document.createElement('button');
    copy.className = 'quiet';
    copy.textContent = 'Copy';
    copy.addEventListener('click', async () => {
      await api.copy(card.message.content);
      copy.textContent = 'Copied';
      setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
    });
    const action = document.createElement('button');
    action.className = 'quiet';
    action.textContent = user ? 'Edit and resend' : 'Regenerate';
    action.addEventListener('click', () => {
      if (!card.message.id || !state.chat) return;
      if (user) dialogs.edit(state.chat.id, card.message);
      else void fork(state.chat.id, card.message.id, null);
    });
    actions.append(copy, action);
    item.append(body, files, status, speed, performance, actions);
    const card = { item, body, steps: [], status, speed, performance, performanceBody,
      copy, action, actions, message };
    updateCard(card, message);
    return card;
  }

  function updateCard(card, message, reveal = false) {
    card.message = message;
    if (message.role === 'user') {
      card.body.textContent = message.content;
      card.body.hidden = !message.content;
      card.copy.hidden = !message.content;
    } else {
      updateResponse(card, message, reveal);
    }
    const states = { streaming: message.content ? '' : 'Generating…', stopped: 'Stopped',
      error: 'Response interrupted', interrupted: 'Response interrupted', no_answer: 'No final answer' };
    const live = message.state === 'streaming' && Boolean(message.agentProgress);
    let detail = states[message.state] || '';
    if (live) {
      const step = card.steps.at(-1);
      const writing = message.agentProgress === 'Streaming response';
      if (writing) detail = step.content.trim() ? 'Writing response…' : step.thought.trim() ? 'Thinking…' : 'Generating…';
      else detail = `${message.agentProgress}…`;
    }
    const statusChanged = card.status.textContent !== detail;
    if (statusChanged) card.status.textContent = detail;
    card.status.classList.toggle('message-progress', live);
    card.status.hidden = !detail;
    if (reveal && statusChanged && detail) textReveal.play(card.status);
    const speed = message.generation_per_second;
    card.speed.hidden = message.role !== 'assistant' || message.state !== 'complete'
      || !Number.isFinite(speed) || Boolean(message.performance);
    card.speed.textContent = card.speed.hidden ? '' : `${speed.toFixed(1)} generation tok/s`;
    card.performance.hidden = message.role !== 'assistant' || !message.performance || message.state === 'streaming';
    if (!card.performance.hidden) {
      const generationRate = message.performance.generation_per_second;
      card.performance.firstElementChild.textContent = generationRate == null ? 'Request performance' :
        'Request performance · ' + generationRate.toFixed(1) + ' generation tok/s';
      renderPerformanceDetails(card.performanceBody, message.performance);
    }
    card.actions.hidden = !message.id || message.state === 'streaming';
    card.action.disabled = state.running || state.navigation || Boolean(generation);
  }

  function renderMessages(forceScroll = false) {
    const follow = forceScroll || atBottom();
    const container = $('messages');
    const elements = [];
    const fresh = new Map();
    for (const message of state.messages) {
      const card = cards.get(message.id) || messageCard(message);
      if (card.message !== message) updateCard(card, message, card.message.state === 'streaming');
      fresh.set(message.id, card);
      elements.push(card.item);
    }
    cards.clear();
    for (const [id, card] of fresh) cards.set(id, card);
    container.replaceChildren(...elements);
    container.hidden = !elements.length;
    $('welcome').hidden = Boolean(elements.length);
    if (follow) scrollBottom();
  }

  function showConversation(value, switching, focus = false) {
    const [chat, messages] = value;
    if (switching) cards.clear();
    state.chat = chat;
    state.messages = messages;
    if (switching) {
      const saved = drafts.get(chat.id);
      draft.value = saved?.text || '';
      draftFiles = saved?.files || [];
    }
    $('chat-title').textContent = chat.title;
    renderMessages(switching);
    updateModel();
    selectHistoryRow();
    renderDraftFiles();
    if (focus) draft.focus();
  }

  function showBudget(budget) {
    state.budget = budget;
    renderContextButton();
  }

  function renderContextButton() {
    const button = $('context');
    const budget = state.budget;
    const live = activeStages.has(performanceSnapshot?.stage) && requestContext(performanceSnapshot);
    const used = live ? live.used : budget?.prompt;
    const total = live ? performanceSnapshot.context : budget?.context;
    const prefix = live ? live.prefix : budget?.estimated ? '~' : '';
    const fill = total > 0 ? Math.min(100, 100 * used / total) : 0;
    $('context-fill').setAttribute('stroke-dasharray', fill + ' 100');
    button.dataset.overLimit = String(Boolean(budget && budget.remaining < 0));
    const label = total ? 'Context usage · ' + prefix + tokens(used) + ' of ' + tokens(total) +
      ' tokens · ' + prefix + percentOf(used, total) : 'Context and memory';
    button.setAttribute('aria-label', label);
    button.title = budget && !live ? label + ' · ' + tokens(budget.reserved) + ' reserved · ' +
      tokens(budget.remaining) + ' remaining' : label;
  }

  const performanceStages = {
    model: 'Preparing model', token_check: 'Checking tokens', prompt: 'Processing prompt',
    generation: 'Generating',
  };
  const activeStages = new Set(Object.keys(performanceStages));
  const tokens = (value) => value == null ? '—' : value.toLocaleString();
  const seconds = (value) => value == null ? '—' : value.toFixed(2) + ' s';
  const rate = (value) => value == null ? '—' : value.toFixed(1) + ' tok/s';

  function percentOf(value, total) {
    return (100 * value / total).toFixed(1) + '%';
  }

  function requestContext(metrics) {
    const prompt = metrics.prompt_tokens ?? metrics.budget_prompt;
    const estimated = metrics.prompt_tokens == null && metrics.budget_estimated;
    if (!metrics.context || prompt == null) return null;
    if (metrics.generated_tokens != null) {
      return { used: prompt + metrics.generated_tokens, prefix: estimated ? '~' : '' };
    }
    if (metrics.first_token_seconds != null) {
      return { used: prompt, prefix: estimated ? '~' : '≥' };
    }
    if (metrics.prompt_progress_tokens != null) {
      return { used: metrics.prompt_progress_tokens, prefix: '' };
    }
    return null;
  }

  function processedPromptTokens(metrics) {
    const cached = metrics.cached_prompt_tokens;
    return metrics.processed_prompt_tokens ??
      (metrics.prompt_progress_tokens != null && cached != null
        ? Math.max(0, metrics.prompt_progress_tokens - cached) : null);
  }

  const analytics = window.InferaStudioAnalytics.create({
    requestContext, processedPromptTokens, rate, percentOf, renderPerformanceDetails,
  });

  function analyticsData() {
    const stage = performanceSnapshot?.stage;
    return {
      title: state.chat?.title,
      messages: state.messages,
      metrics: performanceSnapshot,
      live: Boolean(generation && generation.chatId === state.chat?.id && (!stage || activeStages.has(stage))),
      stage: performanceStages[stage],
    };
  }

  function openAnalytics(trigger, messageId) {
    if (state.chat && !state.navigation) analytics.open(analyticsData(), trigger, messageId);
  }

  function renderPerformanceDetails(element, metrics, { showContextMeter = true, formatTiming } = {}) {
    const prompt = metrics.prompt_tokens ?? metrics.budget_prompt;
    const estimated = metrics.prompt_tokens == null && metrics.budget_estimated;
    const used = requestContext(metrics);
    const cached = metrics.cached_prompt_tokens;
    const processed = processedPromptTokens(metrics);
    const timings = [
      ['Model readiness', metrics.model_seconds],
      ['Token check', metrics.token_check_seconds],
      ['Turn setup', metrics.turn_setup_seconds],
      ['Prompt processing', metrics.prompt_seconds],
      ['Time to first token', metrics.first_token_seconds],
      ['Generation', metrics.generation_seconds],
      ['Response stream', metrics.seconds || null],
      ['Request total', metrics.request_seconds],
    ];
    const usage = [
      ['Prompt speed', rate(metrics.prompt_per_second)],
      ['Generation speed', rate(metrics.generation_per_second)],
      ['Prompt tokens', (estimated ? '~' : '') + tokens(prompt)],
      ['New prompt tokens processed', tokens(processed)],
      ['Generated tokens', tokens(metrics.generated_tokens)],
      ['KV prompt reuse', cached == null ? '—' : tokens(cached) + ' tokens' +
        (prompt > 0 && !estimated ? ' · ' + percentOf(cached, prompt) : '')],
      ['KV buffer allocation', metrics.kv_allocated_mib == null ? '—' : metrics.kv_allocated_mib.toFixed(1) + ' MiB'],
      ['Request context', used ? used.prefix + tokens(used.used) + ' / ' + tokens(metrics.context) +
        ' · ' + used.prefix + percentOf(used.used, metrics.context) : '—'],
    ];
    const peak = Math.max(0, ...timings.map(([, value]) => value ?? 0));
    const groups = [timings, usage].map((entries, index) => {
      const list = document.createElement('dl');
      const bars = index === 0 && formatTiming;
      if (bars) list.className = 'analytics-timing-bars';
      for (const [label, value] of entries) {
        const term = document.createElement('dt');
        const description = document.createElement('dd');
        term.textContent = label;
        description.textContent = index === 0 ? (formatTiming ?? seconds)(value) : value;
        if (bars) {
          const track = document.createElement('span');
          track.className = 'analytics-timing-track';
          track.setAttribute('aria-hidden', 'true');
          track.style.setProperty('--duration-share', (peak > 0 ? 100 * (value ?? 0) / peak : 0) + '%');
          if (value == null) track.style.visibility = 'hidden';
          description.prepend(track);
        }
        list.append(term, description);
      }
      return list;
    });
    if (used && showContextMeter) {
      const contextValue = groups[1].lastElementChild;
      const meter = document.createElement('meter');
      meter.className = 'context-meter';
      meter.max = metrics.context;
      meter.value = Math.min(used.used, metrics.context);
      meter.setAttribute('aria-label', 'Request context utilization');
      contextValue.append(meter);
    }
    element.replaceChildren(...groups);
  }

  function performanceSummary(metrics) {
    if (!metrics) return '';
    const entries = [];
    if (metrics.prompt_per_second != null) entries.push(metrics.prompt_per_second.toFixed(1) + ' prompt tok/s');
    if (metrics.generation_per_second != null) {
      entries.push(metrics.generation_per_second.toFixed(1) + ' generation tok/s');
    }
    const used = requestContext(metrics);
    if (used) entries.push(used.prefix + percentOf(used.used, metrics.context) + ' context');
    if (metrics.request_seconds != null && !activeStages.has(metrics.stage)) {
      entries.push(seconds(metrics.request_seconds) + ' total');
    }
    return entries.join('  ·  ');
  }

  function showMetrics(metrics, observedAt = performance.now()) {
    performanceSnapshot = metrics;
    performanceUpdatedAt = observedAt;
    renderContextButton();
    if (analytics.dialog.open) analytics.update(analyticsData());
  }

  setInterval(() => {
    if (document.hidden || !analytics.dialog.open || !activeStages.has(performanceSnapshot?.stage)) return;
    const elapsed = performanceSnapshot.request_seconds + (performance.now() - performanceUpdatedAt) / 1000;
    analytics.elapsed(elapsed);
  }, 250);

  function cancelBudget() {
    clearTimeout(budgetTimer);
    if (budgetRequest) api.cancel(budgetRequest.id);
    budgetRequest = null;
  }

  function scheduleBudget(delay = 330) {
    cancelBudget();
    if (!state.chat || state.server !== 'Ready' || state.running || state.navigation || generation || choosingWorkspace) return;
    budgetTimer = setTimeout(async () => {
      const request = start('budget', {
        chat_id: state.chat.id, draft: voice.draftText(), attachments: draftFiles, web_search: webSearch,
      });
      budgetRequest = request;
      try {
        const answer = await request.response;
        if (budgetRequest !== request) return;
        if (answer.ok && answer.value) {
          showBudget(answer.value);
          if (answer.value.remaining < 0) {
            notice('Context needs ' + (-answer.value.remaining).toLocaleString() + ' more tokens.', 'context');
          } else if (noticeKind === 'context' || noticeKind === 'image-budget') {
            notice();
          }
        } else if (!answer.cancelled) {
          showBudget(null);
          if (answer.error && draftFiles.some((file) => file.media_type)) notice(answer.error, 'image-budget');
        }
      } catch (error) {
        if (budgetRequest === request) notice(error.message);
      } finally {
        if (budgetRequest === request) budgetRequest = null;
      }
    }, delay);
  }

  function saveDraft() {
    if (state.chat) drafts.set(state.chat.id, { text: voice.draftText(), files: draftFiles });
  }

  function stop() {
    const request = generation?.request || operation;
    if (request) api.cancel(request.id);
  }

  function eventFor(token, event, value) {
    if (token !== revision) return;
    if (event === 'status') setStatus(...value);
    if (event === 'budget') showBudget(value);
    if (event === 'notice') notice(value, 'library');
    if (event === 'features') {
      state.features = value;
      controls();
    }
  }

  async function foreground(action, params, receive) {
    const request = start(action, params, receive);
    operation = request;
    controls();
    try {
      return await request.response;
    } finally {
      if (operation === request) {
        operation = null;
        controls();
      }
    }
  }

  async function openChat(target, options = {}) {
    saveDraft();
    webSearch = false;
    if (operation) api.cancel(operation.id);
    cancelBudget();
    const token = ++revision;
    state.navigation = true;
    state.running = false;
    state.features = null;
    notice();
    if (generation) controls();
    else setStatus('Stopped', 'Opening conversation');
    try {
      const id = typeof target === 'function' ? await target() : target;
      if (token !== revision) return;
      if (options.draft != null) drafts.set(id, { text: options.draft, files: options.files });
      const ongoing = generation?.chatId === id ? generation : null;
      const value = await call('conversation', { chat_id: id });
      if (token !== revision) return;
      if (ongoing) showMetrics(ongoing.performance ?? null, ongoing.performanceUpdatedAt);
      else showMetrics(value[1].findLast((message) => message.performance)?.performance ?? null);
      showBudget(ongoing?.budget ?? null);
      state.features = ongoing?.features ?? null;
      showConversation(ongoing?.conversation || value, true);
      if (generation?.chatId === id) return;
      options.onRestored?.();
      const answer = await foreground('open', { chat_id: id },
        (event, value) => eventFor(token, event, value));
      if (token !== revision) return;
      if (answer.ok && model()) setStatus('Ready');
      else if (answer.ok || answer.cancelled) setStatus('Stopped', answer.cancelled ? 'Stopped' : '');
      else {
        setStatus('Error', 'Could not load model');
        notice(answer.error, 'server');
      }
      if ((answer.ok || answer.cancelled) && !model() && !noticeKind) {
        notice('Choose a GGUF model to start chatting.', 'library');
      }
      if (answer.ok && options.generate) {
        state.navigation = false;
        await generate(options.draft ?? null, options.files);
      }
    } catch (error) {
      if (token === revision) {
        setStatus('Error');
        notice(error.message);
      }
    } finally {
      if (token === revision) {
        state.navigation = false;
        controls();
        if (!options.onRestored || document.activeElement === document.body) draft.focus();
        scheduleBudget();
      }
      void refreshHistory();
    }
  }

  function createChat(modelPath) {
    const source = state.chat?.id ?? null;
    const model = typeof modelPath === 'string' ? modelPath : null;
    return openChat(() => call('new', { source, model }));
  }

  function selectModel(path) {
    if (!state.chat) return createChat(path);
    if (state.chat.model === path && state.server === 'Ready') return Promise.resolve();
    const id = state.chat.id;
    return openChat(async () => {
      await call('configure', { chat_id: id, model: path });
      return id;
    });
  }

  function applySettings(id, settings) {
    return openChat(async () => {
      await call('configure', { chat_id: id, settings });
      return id;
    });
  }

  function fork(id, message, text, files = []) {
    return openChat(() => call('fork', { chat_id: id, message_id: message }), {
      draft: text,
      files,
      generate: true,
    });
  }

  async function generate(text, files = []) {
    if (!state.chat || !model() || state.running || state.navigation || generation || choosingWorkspace) return;
    const job = { chatId: state.chat.id, budget: null, conversation: null, message: null, request: null };
    const useWebSearch = webSearch;
    const visible = () => state.chat?.id === job.chatId;
    cancelBudget();
    generation = job;
    notice();
    showMetrics(null);
    setStatus('Loading', 'Preparing the request');
    let messageTimer;
    const refreshMessage = () => {
      messageTimer = undefined;
      const card = visible() && cards.get(job.message.id);
      if (!card) return;
      const follow = atBottom();
      updateCard(card, job.message, true);
      if (follow) scrollBottom();
    };
    const queueMessageRefresh = () => {
      if (visible() && messageTimer === undefined) messageTimer = setTimeout(refreshMessage, 40);
    };
    try {
      job.request = start('generate', {
        chat_id: job.chatId, draft: text, attachments: files, web_search: useWebSearch,
      }, (event, value) => {
        if (event === 'status') {
          setStatus(...value);
          if (value[0] === 'Generating' && job.message?.agentProgress) {
            job.message.agentProgress = value[1];
            queueMessageRefresh();
          }
        }
        if (event === 'performance') {
          job.performance = value;
          job.performanceUpdatedAt = performance.now();
          if (visible()) showMetrics(value, job.performanceUpdatedAt);
        }
        if (event === 'features') {
          job.features = value;
          if (visible()) {
            state.features = value;
            controls();
          }
        }
        if (event === 'budget') {
          job.budget = value;
          if (visible()) showBudget(value);
        }
        if (event === 'started') {
          const [chat, messages, messageId] = value;
          job.conversation = [chat, messages];
          job.message = messages.find((message) => message.id === messageId);
          if ((chat.workspace || useWebSearch) && job.message) job.message.agentProgress = 'Streaming response';
          const saved = drafts.get(job.chatId);
          if (text !== null && saved?.text === text && saved.files === files) drafts.delete(job.chatId);
          if (visible()) {
            if (text !== null && draft.value === text && draftFiles === files) {
              draft.value = '';
              draftFiles = [];
              webSearch = false;
            }
            showConversation(job.conversation, false);
            if (text !== null) {
              const user = messages.findLast((message) => message.role === 'user');
              if (user) textReveal.play(cards.get(user.id).item);
            }
            if (job.message) textReveal.play(cards.get(job.message.id).status);
            scrollBottom();
          }
        }
        if ((event === 'delta' || event === 'tool_activity') && job.message) {
          if (event === 'delta') [job.message.content, job.message.reasoning] = value;
          else job.message.tool_activity = value;
          queueMessageRefresh();
        }
      });
      controls();
      const answer = await job.request.response;
      if (answer.ok) {
        const [chat, messages, result] = answer.value;
        job.conversation = [chat, messages];
        job.message = null;
        if (!visible()) return;
        showConversation(job.conversation, false);
        showMetrics(result.metrics);
        if (result.error) notice(result.error, 'server');
        else if (result.notice) notice(result.notice, 'model');
        setStatus(result.error ? 'Error' : 'Ready', result.state === 'stopped' ? 'Stopped' : '');
      } else if (!visible()) {
        return;
      } else if (answer.cancelled) {
        setStatus('Stopped', 'Stopped');
      } else if (state.budget?.remaining < 0) {
        setStatus('Ready', 'Context limit reached');
        notice(answer.error, 'context');
      } else {
        setStatus('Error');
        notice(answer.error, 'server');
      }
    } catch (error) {
      if (visible()) {
        setStatus('Error');
        notice(error.message, 'server');
      }
    } finally {
      clearTimeout(messageTimer);
      if (visible() && activeStages.has(performanceSnapshot?.stage)) {
        showMetrics({ ...performanceSnapshot, stage: 'error' });
      }
      generation = null;
      if (job.message) {
        job.message.state = 'error';
        refreshMessage();
      }
      controls();
      if (visible() && !state.navigation) {
        draft.focus();
        if (noticeKind !== 'context') scheduleBudget(0);
      }
      void refreshHistory();
    }
  }

  async function changeContext(id, start, summarize) {
    if (state.chat?.id !== id || state.running || state.navigation || generation) return;
    cancelBudget();
    const token = revision;
    state.running = true;
    notice();
    try {
      const answer = await foreground('change_context', { chat_id: id, start, summarize },
        (event, value) => eventFor(token, event, value));
      if (token !== revision) return;
      if (answer.ok) await openChat(id);
      else {
        setStatus(model() ? 'Ready' : 'Stopped', answer.cancelled ? 'Stopped' : '');
        if (!answer.cancelled) notice(answer.error, 'context');
      }
    } finally {
      if (token === revision) {
        state.running = false;
        controls();
        scheduleBudget();
      }
    }
  }

  async function unload() {
    const token = revision;
    stop();
    if (generation && operation) api.cancel(operation.id);
    cancelBudget();
    await call('unload');
    if (token === revision) {
      showBudget(null);
      setStatus('Stopped', 'Model unloaded');
    }
  }

  async function renameChat(id, title) {
    await call('rename', { chat_id: id, title });
    if (state.chat?.id === id) {
      state.chat.title = title;
      $('chat-title').textContent = title;
    }
    await refreshHistory();
  }

  async function toggleChatPin() {
    const target = historyMenuTarget;
    if (!target) return;
    const { chat, trigger } = target;
    pinButton.disabled = true;
    pinError.hidden = true;
    try {
      const pinnedAt = await call('set_pinned', { chat_id: chat.id, pinned: !chat.pinned_at });
      if (state.chat?.id === chat.id) state.chat.pinned_at = pinnedAt;
      if (historyMenuTarget === target) {
        historyMenu.hidePopover();
        trigger.focus({ preventScroll: true });
      }
      await refreshHistory();
    } catch (error) {
      if (historyMenuTarget !== target) {
        notice(error.message);
        return;
      }
      pinError.textContent = error.message;
      pinError.hidden = false;
      pinButton.disabled = false;
      pinButton.focus();
    }
  }

  async function deleteChat(id) {
    const active = id === null || id === state.chat?.id;
    const token = active ? ++revision : revision;
    if (generation && (id === null || id === generation.chatId)) api.cancel(generation.request.id);
    if (active) {
      if (operation) api.cancel(operation.id);
      cancelBudget();
      state.navigation = true;
      controls();
    }
    try {
      await call('delete', { chat_id: id, unload: active });
    } catch (failure) {
      if (active && token === revision) {
        state.running = state.navigation = false;
        controls();
      }
      throw failure;
    }
    if (id === null) drafts.clear();
    else drafts.delete(id);
    if (active && token === revision) {
      state.chat = null;
      state.messages = [];
      state.features = null;
      state.running = state.navigation = false;
      draft.value = '';
      draftFiles = [];
      $('chat-title').textContent = 'Local chat';
      updateModel();
      renderMessages(true);
      renderDraftFiles();
      showBudget(null);
      showMetrics(null);
      setStatus('Stopped');
    }
    await refreshHistory();
  }

  function catalogNotice() {
    if (!state.catalog.models.some((item) => !item.issue)) {
      notice('No chat-ready GGUF models found. Add a model folder in Settings.', 'settings');
    } else if (state.catalog.errors.length) {
      notice('Some model folders could not be scanned.', 'library');
    }
  }

  async function rescan() {
    notice('Scanning local GGUF folders…');
    adoptCatalog(await call('rescan'));
    notice();
    if (state.chat?.model && !model()) await openChat(state.chat.id);
    else catalogNotice();
  }

  async function savePreferences(preferences) {
    const previous = state.catalog.preferences;
    adoptCatalog(await call('preferences', { preferences }));
    if (previous.show_reasoning !== preferences.show_reasoning) {
      for (const card of cards.values()) {
        for (const step of card.steps) step.reasoning.open = preferences.show_reasoning;
      }
    }
    if (state.chat && (!model() || JSON.stringify(previous.binaries || {}) !== JSON.stringify(preferences.binaries))) {
      await openChat(state.chat.id);
    } else {
      notice();
      catalogNotice();
      if (previous.has_tavily_api_key !== state.catalog.preferences.has_tavily_api_key) scheduleBudget(0);
    }
  }

  async function safely(action) {
    try { await action(); } catch (error) { notice(error.message); }
  }

  async function changeMode(mode) {
    const id = state.chat?.id;
    if (!id || draft.disabled) return;
    choosingWorkspace = true;
    cancelBudget();
    controls();
    try {
      const workspace = await call('workspace', { chat_id: id, mode });
      if (workspace !== null && state.chat?.id === id) {
        state.chat.workspace = workspace;
        notice();
        showBudget(null);
      }
    } finally {
      choosingWorkspace = false;
      controls();
      if (state.chat?.id === id) {
        scheduleBudget(0);
        draft.focus();
      }
    }
  }

  $('model-search').addEventListener('input', renderModelOptions);
  $('models').addEventListener('click', () => modelMenu($('model-menu').hidden));
  window.addEventListener('resize', sizeModelMenu);
  document.addEventListener('pointerdown', (event) => {
    if (!event.target.closest('.model-control')) modelMenu(false);
  });
  sidebarOpen.addEventListener('click', () => setSidebarExpanded(true));
  sidebarClose.addEventListener('click', () => setSidebarExpanded(false));
  $('new-chat').addEventListener('click', () => void createChat());
  $('search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTicket++;
    searchTimer = setTimeout(refreshHistory, 210);
  });
  $('settings').addEventListener('click', () => dialogs.preferences(state.catalog));
  $('library').addEventListener('click', () => {
    modelMenu(false);
    dialogs.library(state.catalog);
  });
  $('rescan').addEventListener('click', () => {
    modelMenu(false);
    void safely(rescan);
  });
  $('stop').addEventListener('click', stop);
  $('chat-actions').addEventListener('click', (event) => {
    const chat = state.history.find((item) => item.id === state.chat.id) ?? state.chat;
    showHistoryMenu(chat, event.currentTarget);
  });
  historyMenu.addEventListener('beforetoggle', (event) => {
    if (event.newState === 'closed') {
      historyMenuTarget?.trigger.setAttribute('aria-expanded', 'false');
      historyMenuTarget = null;
    }
  });
  pinButton.addEventListener('click', () => void toggleChatPin());
  for (const [id, open] of [
    ['rename-chat', dialogs.renameConversation], ['delete-chat', dialogs.confirmDelete],
    ['delete-all-chats', () => dialogs.confirmDelete(null)],
  ]) {
    $(id).addEventListener('click', () => {
      const chat = historyMenuTarget.chat;
      historyMenu.hidePopover();
      open(chat);
    });
  }
  $('model-settings').addEventListener('click', () => void safely(() => dialogs.model(state.chat, model(), state.catalog)));
  $('advanced-stats').addEventListener('click', (event) => openAnalytics(event.currentTarget));
  $('context').addEventListener('click', () => void safely(() => dialogs.context(
    state.chat, state.messages, state.budget, performanceSummary(performanceSnapshot), model(), state.catalog)));
  $('server').addEventListener('click', () => dialogs.server({
    reload: Boolean(model() && !operation && !generation),
    unload: Boolean(operation || generation || state.server === 'Ready'),
  }));
  $('notice-action').addEventListener('click', () => {
    if (noticeKind === 'server') void safely(dialogs.diagnostics);
    else if (noticeKind === 'library') dialogs.library(state.catalog);
    else if (noticeKind === 'model') $('model-settings').click();
    else if (noticeKind === 'context') $('context').click();
    else $('settings').click();
  });
  draft.addEventListener('input', () => {
    controls();
    scheduleBudget();
  });
  $('attach').addEventListener('click', () => filePicker.click());
  $('mode-ask').addEventListener('click', () => {
    if (state.chat?.workspace) void safely(() => changeMode('ask'));
  });
  $('mode-agent').addEventListener('click', () => {
    if (!state.chat?.workspace) void safely(() => changeMode('agent'));
  });
  $('workspace').addEventListener('click', () => void safely(() => changeMode('agent')));
  $('web-search').addEventListener('click', () => {
    if (!state.catalog.preferences.has_tavily_api_key) {
      dialogs.preferences(state.catalog, { section: 'web-search' });
      return;
    }
    webSearch = !webSearch;
    controls();
    scheduleBudget(0);
  });
  $('vision').addEventListener('click', () => {
    if (state.features?.vision) imagePicker.click();
    else dialogs.vision(state.chat, state.catalog);
  });
  $('mtp').addEventListener('click', () => {
    const { id, settings } = state.chat;
    void applySettings(id, { ...settings, mtp: settings.mtp === 'Off' ? 'Automatic' : 'Off' });
  });
  $('thinking').addEventListener('click', () => {
    const { id, settings } = state.chat;
    void applySettings(id, { ...settings, thinking: settings.thinking === 'On' ? 'Off' : 'On' });
  });
  filePicker.addEventListener('change', () => {
    const selected = Array.from(filePicker.files);
    filePicker.value = '';
    void attachFiles(selected);
  });
  imagePicker.addEventListener('change', () => {
    const selected = Array.from(imagePicker.files);
    imagePicker.value = '';
    void attachFiles(selected, true);
  });
  draft.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('composer').requestSubmit();
    }
  });
  $('composer').addEventListener('submit', (event) => {
    event.preventDefault();
    if (!$('send').disabled) void generate(draft.value, draftFiles);
  });
  document.addEventListener('keydown', (event) => {
    if (!startup.hidden || $('modal').open || analytics.dialog.open || event.isComposing) return;
    const shortcut = (event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey
      ? event.key.toLowerCase() : '';
    if (event.key === 'Escape') {
      if (statsPanel.matches(':popover-open') || historyMenu.matches(':popover-open')) return;
      if (!$('model-menu').hidden) {
        modelMenu(false);
        $('models').focus();
      } else if (!voice.cancel()) stop();
    } else if (shortcut === 'b') {
      event.preventDefault();
      if (!event.repeat) setSidebarExpanded(sidebar.hidden);
    } else if (shortcut === 'k') {
      event.preventDefault();
      setSidebarExpanded(true);
      $('search').focus();
    } else if (shortcut === 'n') {
      event.preventDefault();
      if (state.catalog) void createChat();
    }
  });

  setInterval(async () => {
    if (healthPending || operation || generation || state.navigation || state.server !== 'Ready') return;
    healthPending = true;
    const token = revision;
    try {
      const error = await call('health');
      if (error && token === revision && !operation && !generation) {
        showBudget(null);
        setStatus('Error', 'Inference server stopped');
        notice(error, 'server');
      }
    } catch (error) {
      if (token === revision && !operation && !generation) notice(error.message, 'server');
    } finally {
      healthPending = false;
    }
  }, 4500);

  function showWorkspace() {
    startup.hidden = true;
    sidebar.inert = $('chat-workspace').inert = false;
  }

  async function initialize() {
    try {
      const result = await call('initialize');
      state.history = result.history;
      state.pinnedLimit = result.pinned_limit;
      adoptCatalog(result);
      renderHistory();
      setStatus('Stopped');
      const finish = () => {
        if (startup.hidden) return;
        showWorkspace();
        if (result.note) notice(result.note, 'settings');
        else if (!noticeKind) catalogNotice();
        if (!result.preferences.setup_complete) dialogs.preferences(result, { setup: true });
      };
      if (result.history.length) {
        startupStatus.textContent = 'Restoring your latest conversation…';
        await openChat(result.history[0].id, { onRestored: finish });
      }
      finish();
    } catch (error) {
      setStatus('Error', 'Could not open Infera Studio data');
      notice(error.message);
    } finally {
      if (!startup.hidden) showWorkspace();
    }
  }

  document.addEventListener('visibilitychange', refreshStats);
  setInterval(refreshStats, 2000);
  void voice.refresh();
  void initialize();
})();
