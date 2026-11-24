'use strict';

window.InferaStudioDialogs = (() => {
  const $ = (id) => document.getElementById(id);
  const make = (tag, text = '', className = '') => {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    if (className) element.className = className;
    return element;
  };

  const count = (value) => value.toLocaleString();
  const gib = (value) => value == null ? 'Unavailable' : (value / 2 ** 30).toFixed(2) + ' GiB';
  const downloadSize = (value) => value >= 2 ** 30 ? gib(value) : (value / 2 ** 20).toFixed(1) + ' MiB';

  function choice(entries, current) {
    const input = make('select');
    setChoices(input, entries, current);
    return input;
  }

  function setChoices(input, entries, current = '') {
    input.replaceChildren(...entries.map((item) => {
      const [value, label] = Array.isArray(item) ? item : [item, item];
      const option = make('option', label);
      option.value = value;
      return option;
    }));
    input.value = current;
  }

  function input(type, value = '') {
    const field = make('input');
    field.type = type;
    field.value = value ?? '';
    return field;
  }

  function number(value, min, max, fractional = false) {
    const field = input('number', value);
    field.min = min;
    field.max = max;
    field.step = fractional ? 'any' : '1';
    field.required = value != null;
    return field;
  }

  function field(title, control, help = '') {
    const label = make('label', '', 'field');
    label.append(make('span', title), control);
    if (help) label.append(make('small', help));
    return label;
  }

  function button(title, action, style = '') {
    const result = make('button', title, style);
    result.type = 'button';
    result.addEventListener('click', action);
    return result;
  }

  function icon(name) {
    const namespace = 'http://www.w3.org/2000/svg';
    const graphic = document.createElementNS(namespace, 'svg');
    graphic.classList.add('icon');
    graphic.setAttribute('aria-hidden', 'true');
    const symbol = document.createElementNS(namespace, 'use');
    symbol.setAttribute('href', '#icon-' + name);
    graphic.append(symbol);
    return graphic;
  }

  function memory(info) {
    const { plan, gpu, slot, allocations, observed, kvAllocatedMib } = info;
    const lines = [];
    if (gpu) {
      lines.push('GPU VRAM total  ' + gib(gpu.total), 'Currently used  ' + gib(gpu.used),
        'Currently free  ' + gib(gpu.total - gpu.used));
    } else lines.push('AMDGPU VRAM counters are unavailable.');
    if (plan) {
      lines.push('', 'Estimate at ' + count(plan.context) + ' tokens (' + (plan.offload_fraction ? 'GPU' : 'host') + ')',
        'Model weights  ' + gib(plan.weights), 'KV cache  ' + gib(plan.kv));
      if (plan.recurrent) lines.push('Recurrent state  ' + gib(plan.recurrent));
      lines.push('Runtime margin  ' + gib(plan.runtime_margin), 'Estimated total  ' + gib(plan.required));
      if (plan.headroom != null) lines.push('Estimated headroom  ' + gib(plan.headroom));
    }
    if (slot) {
      lines.push('', 'Successfully loaded slot  ' + count(slot) + ' tokens');
      if (kvAllocatedMib != null) lines.push('Total KV buffer allocation  ' + kvAllocatedMib.toFixed(1) + ' MiB');
      for (const [device, kind, size] of allocations) lines.push(device + ' ' + kind + ' allocation  ' + size.toFixed(1) + ' MiB');
      if (observed != null) lines.push('Observed GPU usage change during load  ' + gib(observed));
    }
    return lines.join('\n');
  }

  function modelDownloads(actions) {
    const query = input('search');
    query.id = 'hub-query';
    query.maxLength = 200;
    query.placeholder = 'Search models';
    query.setAttribute('aria-label', 'Model name or repository ID');
    const searchInput = make('div', '', 'hub-search-input');
    searchInput.append(icon('search'), query);
    const searchButton = button('Search', () => search(query.value.trim()), 'hub-search-button');
    const searchRow = make('div', '', 'hub-search-row');
    searchRow.append(searchInput, searchButton);
    const searchStatus = make('p', '', 'hub-status');
    searchStatus.setAttribute('role', 'status');
    const resultList = make('ul', '', 'hub-results');
    resultList.setAttribute('aria-label', 'Matching repositories');
    resultList.hidden = true;
    const browse = button('Browse popular models', () => {
      query.value = '';
      query.focus();
      void search('');
    }, 'hub-link');
    const empty = make('div', '', 'hub-empty');
    empty.append(icon('search'), make('p', 'Search by model name or repository ID.'), browse);
    const discovery = make('section', '', 'settings-card hub-card');
    discovery.append(make('h4', 'Find a model'), searchRow, searchStatus, empty, resultList);

    const repository = make('p', 'Select a repository to see its GGUF files.', 'hub-repository');
    const options = choice([['', 'Select a repository first']], '');
    const variantField = field('GGUF variant', options);
    const fileDetails = make('div', '', 'hub-file');
    const filename = make('span', '', 'hub-filename');
    const fileSize = make('strong', '', 'hub-size');
    fileDetails.append(filename, fileSize);
    fileDetails.hidden = true;
    const folder = choice([], '');
    const folderField = field('Download folder', folder);
    let openModelFolders;
    const folderHelp = make('div', '', 'hub-folder-help');
    folderHelp.append(make('span', 'Save a model folder to enable downloads.'),
      button('Open Model folders', () => openModelFolders(), 'hub-link'));
    const downloadButton = button('Download GGUF', download, 'primary hub-download-button');
    downloadButton.prepend(icon('download'));
    const status = make('p', '', 'hub-status');
    status.setAttribute('role', 'status');
    const stopButton = button('Cancel download', () => {
      stopButton.disabled = true;
      status.textContent = 'Canceling download…';
      window.inferaStudio.cancel(transfer.id);
    }, 'hub-download-button');
    stopButton.hidden = true;
    const meter = make('progress');
    meter.setAttribute('aria-label', 'Download progress');
    const bytes = make('span');
    const percent = make('strong');
    const progressDetails = make('div', '', 'hub-progress-details');
    progressDetails.append(bytes, percent);
    const progress = make('div', '', 'hub-progress');
    progress.hidden = true;
    progress.append(progressDetails, meter);
    const delivery = make('section', '', 'settings-card hub-card');
    delivery.append(make('h4', 'Download'), repository, variantField, fileDetails, folderField, folderHelp,
      downloadButton, stopButton, progress, status);
    const view = make('div', '', 'hub-downloads');
    const columns = make('div', '', 'hub-columns');
    columns.append(discovery, delivery);
    view.append(columns);

    let looking = false;
    let transfer;
    let selectedRepo = '';
    let revision;
    let variants = [];

    function message(element, label, tone = '') {
      element.textContent = label;
      element.dataset.tone = tone;
    }

    function updateControls() {
      const variant = variants.find(({ path }) => path === options.value);
      searchButton.disabled = looking || Boolean(transfer);
      query.disabled = Boolean(transfer);
      for (const control of resultList.querySelectorAll('button')) {
        control.setAttribute('aria-pressed', String(control.value === selectedRepo));
        control.disabled = Boolean(transfer) || (looking && control.value !== selectedRepo);
      }
      options.disabled = Boolean(transfer) || !variants.length;
      folder.disabled = Boolean(transfer) || !folder.value;
      folder.title = folder.value;
      downloadButton.disabled = looking || Boolean(transfer) || !variant || !folder.value;
      downloadButton.hidden = Boolean(transfer);
      stopButton.hidden = !transfer;
      fileDetails.hidden = !variant;
      filename.textContent = variant?.path || '';
      filename.title = filename.textContent;
      fileSize.textContent = variant ? downloadSize(variant.size) : '';
    }

    function setFolders(folders, showModelFolders) {
      openModelFolders = showModelFolders;
      const selected = folders.includes(folder.value) ? folder.value : folders[0] || '';
      setChoices(folder, folders.length ? folders : [['', 'No saved folders']], selected);
      folderHelp.hidden = Boolean(folders.length);
      updateControls();
    }

    function resetVariants(label) {
      variants = [];
      revision = '';
      setChoices(options, [['', label]]);
    }

    async function search(term) {
      if (looking || transfer) return;
      looking = true;
      selectedRepo = '';
      resultList.replaceChildren();
      resultList.hidden = true;
      empty.hidden = true;
      repository.textContent = 'Select a repository to see its GGUF files.';
      resetVariants('Select a repository first');
      message(searchStatus, 'Searching Hugging Face…');
      message(status, '');
      progress.hidden = true;
      updateControls();
      try {
        const results = await actions.call('hub_search', { query: term });
        for (const repo of results) {
          const [owner, name] = repo.split('/');
          const label = make('span', '', 'hub-result-label');
          label.append(make('strong', name || owner));
          if (name) label.append(make('span', owner, 'hub-result-owner'));
          const result = button('', () => selectRepository(repo), 'hub-result');
          result.value = repo;
          result.title = repo;
          result.setAttribute('aria-label', repo);
          result.append(icon('chip'), label, icon('chevron-right'));
          const item = make('li');
          item.append(result);
          resultList.append(item);
        }
        resultList.hidden = !results.length;
        message(searchStatus, results.length
          ? results.length + (results.length === 1 ? ' repository found' : ' repositories found')
          : 'No public GGUF repositories found. Try a different search.');
      } catch (failure) {
        message(searchStatus, failure.message, 'error');
      } finally {
        looking = false;
        updateControls();
      }
    }

    async function selectRepository(repo) {
      if (looking || transfer || repo === selectedRepo) return;
      selectedRepo = repo;
      repository.textContent = repo;
      resetVariants('Loading GGUF files…');
      progress.hidden = true;
      looking = true;
      message(status, 'Checking available GGUF files…');
      updateControls();
      try {
        ({ revision, variants } = await actions.call('hub_variants', { repo }));
        setChoices(options, [['', variants.length ? 'Choose a quantization or file' : 'No complete GGUF variants'], ...variants.map(
          ({ path, quant, size, parts }) =>
            [path, [quant, downloadSize(size), parts > 1 ? parts + ' parts' : '', path].filter(Boolean).join(' · ')],
        )]);
        message(status, variants.length ? '' : 'This repository has no complete GGUF variants.');
      } catch (failure) {
        selectedRepo = '';
        resetVariants('Could not load GGUF variants');
        message(status, failure.message, 'error');
      } finally {
        looking = false;
        updateControls();
      }
    }

    async function download() {
      const variant = variants.find(({ path }) => path === options.value);
      if (!variant || !folder.value || transfer || looking) return;
      meter.max = variant.size;
      meter.value = 0;
      bytes.textContent = '0 MiB / ' + downloadSize(variant.size);
      percent.textContent = '0%';
      progress.hidden = false;
      message(status, 'Starting download…');
      stopButton.disabled = false;
      let saved;
      try {
        const focusCancel = document.activeElement === downloadButton;
        transfer = actions.start('hub_download', {
          repo: selectedRepo, revision, path: variant.path, folder: folder.value,
        }, (event, value) => {
          if (event !== 'progress') return;
          if (status.textContent === 'Starting download…') status.textContent = 'Downloading…';
          meter.value = value.received;
          percent.textContent = Math.floor(meter.position * 100) + '%';
          bytes.textContent = downloadSize(value.received) + ' / ' + downloadSize(value.total)
            + (value.parts > 1 ? ' · Part ' + value.part + ' of ' + value.parts : '');
        });
        updateControls();
        if (focusCancel) stopButton.focus({ preventScroll: true });
        const answer = await transfer.response;
        if (!answer.ok) {
          message(status, answer.cancelled ? 'Download canceled.' : answer.error, answer.cancelled ? '' : 'error');
          progress.hidden = true;
          return;
        }
        saved = answer.value;
        meter.value = meter.max;
        percent.textContent = '100%';
        bytes.textContent = downloadSize(variant.size) + ' downloaded';
        message(status, 'Downloaded to ' + saved, 'success');
        stopButton.disabled = true;
        await actions.refreshModels();
      } catch (failure) {
        progress.hidden = !saved;
        message(status, saved
          ? 'Downloaded to ' + saved + '. Could not refresh the local model library. ' + failure.message
          : failure.message, 'error');
      } finally {
        transfer = null;
        updateControls();
      }
    }

    query.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' || event.isComposing) return;
      event.preventDefault();
      void search(query.value.trim());
    });
    options.addEventListener('change', () => {
      progress.hidden = true;
      message(status, '');
      updateControls();
    });
    folder.addEventListener('change', () => { folder.title = folder.value; });
    return { view, setFolders };
  }

  function create(actions) {
    const modal = $('modal');
    const form = $('modal-form');
    const footer = $('modal-actions');
    const errorText = $('modal-error');
    const downloads = modelDownloads(actions);
    let session = 0;
    let submit;

    function error(message = '') {
      errorText.hidden = !message;
      errorText.textContent = message;
    }

    function show(title, ...content) {
      session++;
      submit = null;
      error();
      $('modal-title').textContent = title;
      $('modal-subtitle').hidden = true;
      $('modal-content').replaceChildren(...content);
      footer.replaceChildren(button('Close', () => modal.close(), 'quiet'));
      if (!modal.open) modal.showModal();
      $('modal-content').scrollTop = 0;
    }

    function showSettings(title, subtitle, sections, { initial = sections[0].id, model = false } = {}) {
      const layout = make('div', '', 'settings-layout');
      const navigation = make(model ? 'div' : 'nav', '', 'settings-nav');
      navigation.setAttribute('aria-label', title + ' sections');
      if (model) {
        layout.classList.add('model-settings');
        navigation.setAttribute('role', 'tablist');
      }
      layout.append(navigation);

      function select(id, focus = false) {
        for (const { id: sectionId, panel, control } of panels) {
          const selected = sectionId === id;
          panel.hidden = !selected;
          control.setAttribute(model ? 'aria-selected' : 'aria-current', String(selected));
          if (model) control.tabIndex = selected ? 0 : -1;
          if (selected && focus) control.focus();
        }
      }

      const panels = sections.map(({ id, label, glyph, heading = label, description, featured = false, content }) => {
        const panel = make('section', '', 'settings-section');
        panel.id = 'settings-' + id;
        const control = button('', () => select(id, true), 'quiet settings-nav-button');
        control.setAttribute('aria-controls', panel.id);
        control.append(icon(glyph), label);
        if (model) {
          control.id = panel.id + '-tab';
          control.setAttribute('role', 'tab');
          panel.setAttribute('role', 'tabpanel');
          panel.setAttribute('aria-labelledby', control.id);
          panel.tabIndex = 0;
          panel.append(...content);
        } else {
          const sectionTitle = make('h3', heading);
          sectionTitle.id = panel.id + '-heading';
          panel.setAttribute('aria-labelledby', sectionTitle.id);
          const header = make('header', '', featured ? 'hub-hero' : 'settings-section-heading');
          const copy = make('div');
          copy.append(sectionTitle);
          if (description) copy.append(make('p', description));
          header.append(copy);
          if (featured) {
            const mark = make('div', '', 'hub-hero-mark');
            mark.append(icon(glyph));
            mark.setAttribute('aria-hidden', 'true');
            header.append(mark);
            panel.append(header, ...content);
          } else {
            const card = make('div', '', 'settings-card');
            card.append(...content);
            panel.append(header, card);
          }
        }
        navigation.append(control);
        layout.append(panel);
        return { id, panel, control };
      });

      if (model) {
        navigation.addEventListener('keydown', (event) => {
          if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
          const index = panels.findIndex(({ control }) => control === event.target);
          const last = panels.length - 1;
          const next = {
            ArrowLeft: index === 0 ? last : index - 1,
            ArrowRight: index === last ? 0 : index + 1,
            Home: 0,
            End: last,
          }[event.key];
          if (index < 0 || !Number.isInteger(next)) return;
          event.preventDefault();
          select(panels[next].id, true);
        });
      }

      layout.addEventListener('invalid', (event) => {
        // Reveal the first invalid control before the browser moves focus to it.
        if (event.target !== layout.querySelector('input:invalid, select:invalid, textarea:invalid')) return;
        const section = panels.find(({ panel }) => panel.contains(event.target));
        if (section) select(section.id);
      }, true);

      select(initial);
      show(title, layout);
      $('modal-title').prepend(icon('settings'));
      $('modal-subtitle').textContent = subtitle;
      $('modal-subtitle').title = subtitle;
      $('modal-subtitle').hidden = !subtitle;
      panels.find(({ id }) => id === initial).control.focus();
      return select;
    }

    function setSubmit(title, action, style = 'primary') {
      submit = action;
      const control = make('button', title, style);
      control.type = 'submit';
      footer.append(control);
    }

    async function run(action, close = true) {
      const active = session;
      const controls = [...footer.querySelectorAll('button')].filter((item) => !item.disabled);
      controls.forEach((item) => { item.disabled = true; });
      error();
      try {
        await action();
        if (session === active && close && modal.open) modal.close();
      } catch (failure) {
        if (session === active) error(failure.message);
      } finally {
        controls.forEach((item) => { item.disabled = false; });
      }
    }

    $('modal-close').addEventListener('click', () => modal.close());
    modal.addEventListener('close', () => {
      if (modal.open) return;
      session++;
      for (const password of form.querySelectorAll('input[type="password"]')) password.value = '';
    });
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (submit && event.submitter && !event.submitter.disabled) void run(submit);
    });

    function preferences(catalog, { setup = false, section = setup ? 'folders' : 'appearance' } = {}) {
      const prefs = catalog.preferences;
      const original = prefs.binaries || {};
      const backend = choice(catalog.backends, prefs.backend);
      const theme = choice(['Dark', 'Light'], prefs.theme);
      const reasoning = input('checkbox');
      reasoning.checked = prefs.show_reasoning;
      const reasoningLabel = make('label', '', 'settings-check');
      reasoningLabel.append(reasoning, 'Expand reasoning by default');
      const voiceStatus = make('p', '', 'muted');
      voiceStatus.setAttribute('role', 'status');
      const voiceFiles = button('', () => run(async () => {
        voiceFiles.disabled = true;
        try {
          await actions.configureVoice();
          renderVoice();
        } finally {
          voiceFiles.disabled = false;
        }
      }, false));
      function renderVoice() {
        const source = actions.voiceSource();
        voiceFiles.textContent = source === 'custom' ? 'Change custom Whisper files…'
          : source === 'bundled' ? 'Choose custom Whisper files…' : 'Choose Whisper files…';
        voiceStatus.textContent = source === 'bundled' ? 'Ready with the bundled multilingual Whisper model.'
          : source === 'custom' ? 'Ready with your custom Whisper files.' : 'Whisper files are unavailable.';
      }
      renderVoice();
      const tavilyKey = input('password');
      tavilyKey.autocomplete = 'off';
      tavilyKey.spellcheck = false;
      tavilyKey.placeholder = prefs.has_tavily_api_key ? 'Saved key' : 'Enter your Tavily API key';
      const removeTavilyKey = input('checkbox');
      const removeKeyLabel = make('label', '', 'settings-check');
      removeKeyLabel.append(removeTavilyKey, 'Remove saved key');
      removeKeyLabel.hidden = !prefs.has_tavily_api_key;
      tavilyKey.addEventListener('input', () => { removeTavilyKey.checked = false; });
      removeTavilyKey.addEventListener('change', () => {
        if (removeTavilyKey.checked) tavilyKey.value = '';
      });
      const statControls = new Map();
      const statOptions = Object.entries(catalog.statLabels).map(([key, name]) => {
        const checkbox = input('checkbox');
        checkbox.checked = prefs.visible_stats.includes(key);
        const label = make('label', '', 'settings-check');
        label.append(checkbox, name);
        statControls.set(key, checkbox);
        return label;
      });
      const paths = new Map();
      const executables = make('fieldset', '', 'settings-group');
      executables.append(make('legend', 'Inference executables'));
      for (const name of catalog.backends) {
        const executable = input('text', original[name] || '');
        executable.setAttribute('aria-label', name + ' executable');
        executable.placeholder = catalog.discovered[name] || 'Auto-detect';
        const row = make('div', '', 'field-row');
        row.append(executable, button('Browse…', async () => {
          try {
            const paths = await window.inferaStudio.browse('executable');
            if (paths.length) executable.value = paths[0];
          } catch (failure) { error(failure.message); }
        }));
        const group = make('div', '', 'field');
        group.append(make('span', name), row);
        executables.append(group);
        paths.set(name, executable);
      }

      const dirs = new Set(prefs.model_dirs);
      const directoryList = make('div', '', 'settings-folders');
      function renderDirs() {
        directoryList.replaceChildren();
        if (!dirs.size) directoryList.append(make('p', 'No model folders added.', 'muted'));
        for (const directory of dirs) {
          const row = make('div', '', 'list-row');
          const remove = button('Remove', () => {
            dirs.delete(directory);
            renderDirs();
          }, 'quiet');
          remove.setAttribute('aria-label', 'Remove ' + directory);
          row.append(icon('folder'), make('span', directory), remove);
          directoryList.append(row);
        }
      }
      const addFolder = button('Add folder…', async () => {
        try {
          for (const directory of await window.inferaStudio.browse('directory')) dirs.add(directory);
          renderDirs();
        } catch (failure) { error(failure.message); }
      });
      renderDirs();
      const select = showSettings(setup ? 'Set up Infera Studio' : 'Settings', '', [
        { id: 'appearance', label: 'Appearance', glyph: 'appearance', content: [field('Theme', theme), reasoningLabel] },
        { id: 'folders', label: 'Model folders', glyph: 'folder', heading: 'GGUF model folders', content: [directoryList, addFolder] },
        { id: 'hugging-face', label: 'Hugging Face', glyph: 'download', featured: true,
          description: 'Find public GGUF models and download them to your local library.', content: [downloads.view] },
        { id: 'inference', label: 'Inference', glyph: 'chip', content: [field('Default backend', backend), executables] },
        { id: 'web-search', label: 'Web search', glyph: 'search', content: [
          field('Tavily API key', tavilyKey, 'Web search sends the model’s search queries to Tavily.'),
          removeKeyLabel,
        ] },
        { id: 'voice', label: 'Voice input', glyph: 'mic', description:
          'Dictate into the message box. Transcription runs offline on your computer.',
        content: [voiceStatus, voiceFiles] },
        { id: 'stats', label: 'Stats', glyph: 'stats', heading: 'Toolbar stats', content: statOptions },
      ], { initial: section });
      downloads.setFolders(prefs.model_dirs, () => select('folders', true));
      setSubmit(setup ? 'Finish setup' : 'Save settings', () => {
        const binaries = Object.fromEntries([...paths].flatMap(([name, control]) =>
          control.value.trim() ? [[name, control.value.trim()]] : []));
        return actions.savePreferences({
          model_dirs: [...dirs], binaries, backend: backend.value, theme: theme.value,
          show_reasoning: reasoning.checked, setup_complete: true,
          visible_stats: [...statControls].filter(([, checkbox]) => checkbox.checked).map(([key]) => key),
          ...(tavilyKey.value.trim() || removeTavilyKey.checked ? { tavily_api_key: tavilyKey.value.trim() } : {}),
        });
      });
    }

    function library(catalog) {
      const items = choice([['', 'Select a model'], ...catalog.models.map((item) =>
        [item.path, item.name + (item.issue ? ' · Unavailable' : '')])], '');
      const report = make('pre', '', 'report');
      const issues = make('pre', catalog.errors.join('\n'), 'report muted');
      issues.hidden = !catalog.errors.length;
      const select = button('Use model', () => run(() => actions.selectModel(items.value)), 'primary');
      function update() {
        const item = catalog.models.find((item) => item.path === items.value);
        select.disabled = !item || Boolean(item.issue);
        report.textContent = item
          ? [item.path, [item.architecture, item.quant].filter(Boolean).join(' · '),
            count(item.parameters) + ' parameters · ' + gib(item.size) + ' · ' + count(item.context)
              + ' native context tokens', item.mtp && !item.issue ? 'Embedded MTP head' : '',
            ...item.shards.slice(1), item.issue].filter(Boolean).join('\n')
          : 'Select a model to see its metadata and location.';
      }
      items.addEventListener('change', update);
      update();
      show('Local model library', field('Discovered models', items), report, issues);
      footer.append(button('Model folders…', () => preferences(catalog, { section: 'folders' })),
        button('Rescan', () => run(actions.rescan)), select);
    }

    function projectorChoice(catalog, current) {
      const projectors = catalog.models.filter((item) => item.visionProjector);
      const entries = [['', 'Off'], ...projectors.map((item) => [item.path, item.name + ' · ' + item.path.split('/').at(-1)])];
      if (current && !projectors.some((item) => item.path === current)) {
        entries.push([current, 'Unavailable · ' + current.split('/').at(-1)]);
      }
      return choice(entries, current);
    }

    function vision(chat, catalog) {
      const current = chat.settings.vision_projector;
      const projector = projectorChoice(catalog, current);
      const instructions = make('p', 'Add the matching mmproj GGUF to a model folder and rescan, then select it here.');
      show('Vision', instructions, field('Vision projector', projector));
      footer.append(button('Model folders…', () => preferences(catalog, { section: 'folders' })));
      if (projector.options.length > 1) {
        setSubmit('Apply', () => actions.applySettings(chat.id, { ...chat.settings, vision_projector: projector.value }));
      }
    }

    async function model(chat, item, catalog) {
      const info = await actions.call('context', { chat_id: chat.id });
      const config = chat.settings;
      const fields = new Map();
      const budgetGrid = make('div', '', 'field-grid model-token-budget');
      const runtimeGrid = make('div', '', 'model-settings-row');
      const addOnGrid = make('div', '', 'field-grid');
      const estimate = make('p', '', 'model-memory-estimate');
      estimate.setAttribute('role', 'status');
      function group(title, ...content) {
        const group = make('fieldset', '', 'model-settings-group');
        group.append(make('legend', title), ...content);
        return group;
      }
      function add(container, key, label, control, help) {
        const row = field(label, control, help);
        if (help) {
          const [name, , description] = row.children;
          name.id = 'model-' + key + '-label';
          description.id = 'model-' + key + '-help';
          control.setAttribute('aria-labelledby', name.id);
          control.setAttribute('aria-describedby', description.id);
        }
        fields.set(key, control);
        container.append(row);
        return control;
      }
      const nativeContext = Math.min(item?.context || 0, 2_147_483_647);
      const presets = nativeContext
        ? [...catalog.contextSizes.filter((size) => size < nativeContext), nativeContext] : catalog.contextSizes;
      const contextSizes = [...new Set([0, ...presets, config.context])].sort((a, b) => a - b);
      const capacity = input('range', contextSizes.indexOf(config.context));
      capacity.min = 0;
      capacity.max = contextSizes.length - 1;
      capacity.step = 1;
      const stops = make('datalist');
      stops.id = 'model-context-stops';
      stops.append(...contextSizes.map((_, index) => make('option', String(index))));
      capacity.setAttribute('list', stops.id);
      add(budgetGrid, 'context', 'Context tokens', capacity, item?.context
        ? 'GGUF advertises ' + count(item.context) + ' tokens' : 'Automatic is the first stop.');
      const contextField = capacity.closest('.field');
      contextField.classList.add('model-context-field');
      const contextReadout = make('span', '', 'model-context-value');
      contextReadout.setAttribute('aria-hidden', 'true');
      contextField.append(contextReadout, stops);
      function showContextSize() {
        const tokens = contextSizes[capacity.valueAsNumber];
        const label = tokens ? count(tokens) + ' tokens' : 'Automatic';
        contextReadout.textContent = label;
        capacity.setAttribute('aria-valuetext', label);
      }
      showContextSize();
      capacity.addEventListener('input', showContextSize);
      const output = add(budgetGrid, 'output', 'Maximum output tokens', number(config.output, 1, 2_147_483_647));
      output.classList.add('model-compact-control');
      const backend = add(runtimeGrid, 'backend', 'Backend', choice(catalog.backends, config.backend));
      const layers = add(runtimeGrid, 'gpu_layers', 'GPU layers', number(config.gpu_layers, -1, 100_000), '−1 for automatic');
      add(runtimeGrid, 'flash', 'Flash Attention', choice([['auto', 'Automatic'], ['on', 'On'], ['off', 'Off']], config.flash));
      const key = add(runtimeGrid, 'cache_k', 'Key cache', choice(catalog.cacheTypes, config.cache_k));
      const value = add(runtimeGrid, 'cache_v', 'Value cache', choice(catalog.cacheTypes, config.cache_v));
      const drafts = info.mtpDrafts;
      if (item?.mtp || drafts.length || config.mtp_draft) {
        const mtp = add(addOnGrid, 'mtp', 'Multi-token prediction (MTP)', choice(['Automatic', 'Off'], config.mtp),
          info.mtp || 'Automatic uses an available MTP head with compatible llama.cpp builds.');
        mtp.classList.add('model-compact-control');
      }
      if (drafts.length || config.mtp_draft) {
        const choices = [['', item?.mtp ? 'Embedded head' : 'None'], ...drafts];
        if (config.mtp_draft && !drafts.some(([path]) => path === config.mtp_draft)) {
          choices.push([config.mtp_draft, 'Unavailable · ' + config.mtp_draft.split('/').at(-1)]);
        }
        add(addOnGrid, 'mtp_draft', 'MTP draft head', choice(choices, config.mtp_draft),
          'Choose the draft GGUF made for this exact model.');
      }
      if (config.vision_projector || catalog.models.some((candidate) => candidate.visionProjector)) {
        add(addOnGrid, 'vision_projector', 'Vision projector', projectorChoice(catalog, config.vision_projector),
          'Choose the mmproj GGUF made for this model.');
      }

      const samplingGrid = make('div', '', 'model-settings-row model-samplers');
      const samplers = [
        ['temperature', 'Temperature', 10, true], ['top_p', 'Top-p', 1, true],
        ['top_k', 'Top-k', 1_000_000, false], ['min_p', 'Min-p', 1, true],
        ['repeat_penalty', 'Repetition penalty', 10, true], ['seed', 'Seed', 4_294_967_295, false],
      ];
      for (const [key, label, max, fractional] of samplers) {
        if (!info.samplers.includes(key)) continue;
        const control = number(config[key], 0, max, fractional);
        control.required = false;
        control.placeholder = 'Model default';
        add(samplingGrid, key, label, control);
      }
      const sampling = [];
      if (samplingGrid.childElementCount) sampling.push(samplingGrid);
      if (info.thinking) {
        const thinking = make('div', '', 'model-settings-row model-thinking');
        add(thinking, 'thinking', 'Thinking', choice(['Automatic', 'On', 'Off'], config.thinking));
        sampling.push(thinking);
      }
      if (!sampling.length) sampling.push(make('p', 'No sampling overrides are available for this model.', 'muted'));

      const instructionField = make('div', '', 'model-instructions');
      const instructions = make('textarea');
      instructions.rows = 9;
      instructions.value = config.system;
      instructions.placeholder = 'Optional';
      add(instructionField, 'system', 'System instructions', instructions, 'Applies to requests in this conversation.');
      function values() {
        return { ...config, ...Object.fromEntries([...fields].map(([key, control]) => {
          if (control === capacity) return [key, contextSizes[capacity.valueAsNumber]];
          if (control.type === 'number') return [key, control.value === '' ? null : control.valueAsNumber];
          return [key, control.value];
        })) };
      }
      function renderEstimate(plan) {
        if (!plan) {
          estimate.textContent = 'Memory estimate unavailable.';
          return;
        }
        const location = plan.offload_fraction ? 'GPU' : 'host';
        const headroom = plan.headroom == null ? '' : ' · ' + gib(plan.headroom) + ' headroom';
        estimate.replaceChildren(
          make('strong', gib(plan.required) + ' estimated ' + location + ' memory'),
          make('span', ' at ' + count(plan.context) + ' tokens' + headroom),
        );
      }
      const runtime = [group('Token budget', budgetGrid, estimate), group('Inference', runtimeGrid)];
      if (addOnGrid.childElementCount) runtime.push(group('Model add-ons', addOnGrid));
      renderEstimate(info.plan);
      showSettings('Model settings', item?.name || 'No model selected', [
        { id: 'runtime', label: 'Context and GPU', glyph: 'chip', content: runtime },
        { id: 'sampling', label: 'Sampling', glyph: 'sliders', content: sampling },
        { id: 'instructions', label: 'Instructions', glyph: 'instructions', content: [instructionField] },
      ], { model: true });
      let timer;
      let request = 0;
      for (const control of [backend, capacity, layers, key, value]) {
        control.addEventListener('input', () => {
          clearTimeout(timer);
          const token = ++request;
          const active = session;
          timer = setTimeout(async () => {
            if (active !== session || !modal.open || !form.matches(':valid')) return;
            try {
              const info = await actions.call('context', { chat_id: chat.id, settings: values() });
              if (token === request && active === session) renderEstimate(info.plan);
            } catch {
              if (token === request && active === session) renderEstimate(null);
            }
          }, 200);
        });
      }
      setSubmit('Apply settings', () => actions.applySettings(chat.id, values()));
    }

    async function context(chat, messages, budget, metrics, item, catalog) {
      const info = await actions.call('context', { chat_id: chat.id });
      const summary = budget ? [
        budget.estimated ? 'Estimated token count' : 'Tokenizer count',
        'Configured  ' + count(budget.context) + ' tokens',
        'Conversation and template  ' + count(budget.conversation),
        'System instructions  ' + count(budget.system),
        'Reserved output  ' + count(budget.reserved),
        'Remaining  ' + count(budget.remaining),
      ].join('\n') : 'Context usage appears after a model loads.';
      const userTurns = messages.filter((message) => message.role === 'user');
      const cut = choice([['0', 'Beginning of conversation'], ...userTurns.map((message) =>
        [message.id, (message.content.trim() || message.attachments?.[0]?.name || 'User message')
          .replace(/\s+/g, ' ').slice(0, 85)])], String(chat.active_from));
      if (!cut.value) cut.value = '0';
      const apply = button('Apply context start', () =>
        run(() => actions.changeContext(chat.id, Number(cut.value), false)));
      const summarize = button('Summarize earlier turns', () =>
        run(() => actions.changeContext(chat.id, Number(cut.value), true)));
      function change() {
        apply.disabled = Number(cut.value) === chat.active_from;
        summarize.disabled = !info.slot || Number(cut.value) <= Math.max(chat.active_from, userTurns[0]?.id || 0);
      }
      cut.addEventListener('change', change);
      change();
      const row = make('div', '', 'inline-actions');
      row.append(apply, summarize);
      const note = chat.summary ? 'An earlier-turn summary is active. Stored messages remain in the conversation.'
        : 'Stored messages remain in the conversation when you change the active context.';
      show('Context and memory', make('pre', summary, 'report'), make('p', note, 'muted'),
        field('First retained user turn', cut), row, make('pre', memory(info), 'report'),
        ...(info.mtp ? [make('p', info.mtp, 'muted')] : []), make('p', metrics, 'muted'));
      footer.append(button('Model settings', () => run(() => model(chat, item, catalog), false)),
        button('New chat', () => run(actions.createChat)));
    }

    async function diagnostics() {
      const logs = await actions.call('diagnostics');
      show('Inference diagnostics', make('pre', logs || 'No inference startup log is available.', 'report'));
      footer.append(button('Unload model', () => run(actions.unload), 'danger'));
    }

    function server(available) {
      show('Inference server');
      const reload = button('Reload model', () => run(actions.reload));
      reload.disabled = !available.reload;
      const unload = button('Unload model', () => run(actions.unload));
      unload.disabled = !available.unload;
      footer.append(button('Diagnostics', () => run(diagnostics, false)), reload, unload);
    }

    function confirmDelete(chat) {
      show(chat ? 'Delete conversation?' : 'Delete all conversations?',
        make('p', chat ? 'This permanently removes “' + chat.title + '” from Infera Studio.'
          : 'This permanently removes every conversation from Infera Studio.'));
      setSubmit(chat ? 'Delete conversation' : 'Delete all', () => actions.deleteChat(chat?.id ?? null), 'danger');
    }

    function renameConversation(chat) {
      const title = input('text', chat.title);
      title.required = true;
      show('Rename conversation', field('Conversation title', title));
      setSubmit('Rename', () => {
        const name = title.value.trim();
        if (!name) throw new Error('Enter a title.');
        return actions.renameChat(chat.id, name);
      });
    }

    function edit(id, message) {
      const text = make('textarea');
      text.rows = 6;
      const files = message.attachments || [];
      text.required = !files.length;
      text.value = message.content;
      const names = make('ul');
      for (const file of files) names.append(make('li', file.name));
      show('Edit and resend', field('Message', text), ...(files.length ? [names] : []));
      setSubmit('Resend in a branch', () => {
        if (!text.value.trim() && !files.length) throw new Error('Enter a message.');
        return actions.fork(id, message.id, text.value, files);
      });
      text.focus();
    }

    return { preferences, library, vision, model, context, diagnostics, server, renameConversation, confirmDelete, edit };
  }

  return { create };
})();
