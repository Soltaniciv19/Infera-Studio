'use strict';

window.InferaStudioMarkdown = (() => {
  const markdown = window.markdownit({ html: false, linkify: false });
  const escape = markdown.utils.escapeHtml;
  markdown.validateLink = (href) => {
    try {
      const link = new URL(href);
      return link.protocol === 'https:' || link.protocol === 'http:';
    } catch {
      return false;
    }
  };
  markdown.renderer.rules.image = (tokens, index) => escape(tokens[index].content);
  markdown.renderer.rules.fence = (tokens, index, _options, env) => {
    const { content, info } = tokens[index];
    const language = info.trim().split(/\s+/, 1)[0];
    const highlighted = language && window.hljs.getLanguage(language)
      ? window.hljs.highlight(content, { language, ignoreIllegals: true }).value
      : escape(content);
    const id = env.snippets.push(content) - 1;
    return '<div class="code-block"><div class="code-heading"><span>'
      + escape(language || 'Code')
      + '</span><button class="quiet" type="button" data-copy="' + id + '">Copy code</button>'
      + '</div><pre><code>' + highlighted + '</code></pre></div>\n';
  };

  function render(target, source, reveal = false) {
    const env = { snippets: [] };
    const template = document.createElement('template');
    // Markdown-It escapes raw HTML and allows only absolute HTTP(S) links above.
    template.innerHTML = markdown.render(source, env);
    for (const button of template.content.querySelectorAll('[data-copy]')) {
      const snippet = env.snippets[Number(button.dataset.copy)];
      button.addEventListener('click', async () => {
        await window.inferaStudio.copy(snippet);
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = 'Copy code'; }, 1200);
      });
    }
    for (const link of template.content.querySelectorAll('a')) {
      const url = link.href;
      link.removeAttribute('href');
      link.setAttribute('role', 'link');
      link.tabIndex = 0;
      const open = () => window.inferaStudio.openLink(url).catch(() => {});
      link.addEventListener('click', open);
      link.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') open();
      });
    }
    window.InferaStudioTextReveal.replace(target, template.content, reveal);
  }

  return { render };
})();
