'use strict';

window.InferaStudioTextReveal = (() => {
  const duration = 180;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const animations = new WeakMap();
  const playing = new Set();
  const tails = new WeakMap();

  function splitsSurrogate(text, offset) {
    const code = text.charCodeAt(offset);
    return code >= 0xdc00 && code <= 0xdfff;
  }

  function cancel(element) {
    animations.get(element)?.cancel();
  }

  function play(element, { elapsed = 0, from = 0.92 } = {}) {
    if (motion.matches || elapsed >= duration) return;
    cancel(element);
    const animation = element.animate([{ opacity: from }, { opacity: 1 }], {
      duration, easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
    });
    animation.currentTime = elapsed;
    animations.set(element, animation);
    playing.add(animation);
    const forget = () => {
      playing.delete(animation);
      if (animations.get(element) === animation) animations.delete(element);
    };
    animation.onfinish = animation.oncancel = forget;
  }

  motion.addEventListener('change', () => {
    if (motion.matches) for (const animation of playing) animation.cancel();
  });

  function replace(target, content, reveal) {
    if (!reveal || motion.matches) {
      tails.delete(target);
      target.replaceChildren(content);
      return;
    }

    const before = target.textContent;
    const after = content.textContent;
    const now = performance.now();
    let common = 0;
    while (common < before.length && common < after.length && before[common] === after[common]) common++;
    if (common && splitsSurrogate(after, common)) common--;
    // Carry unfinished fades through the Markdown reparse without restarting them.
    const tail = (tails.get(target) || []).filter(({ start, time }) => start < common && now - time < duration)
      .map(({ start, end, time }) => ({ start, end: Math.min(end, common), time }));
    if (common < after.length) tail.push({ start: common, end: after.length, time: now });
    tails.set(target, tail);

    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    const spans = [];
    const first = tail[0]?.start ?? after.length;
    let offset = 0;
    for (const node of nodes) {
      const text = node.data;
      const start = offset;
      offset += text.length;
      if (offset <= first || !text.trim() || node.parentElement?.closest('button')) continue;
      const parts = document.createDocumentFragment();
      let used = 0;
      for (const range of tail) {
        const from = Math.max(0, range.start - start);
        const to = Math.min(text.length, range.end - start);
        if (to <= from || !text.slice(from, to).trim()) continue;
        const span = document.createElement('span');
        span.textContent = text.slice(from, to);
        parts.append(text.slice(used, from), span);
        spans.push([span, range.time]);
        used = to;
      }
      if (used) {
        parts.append(text.slice(used));
        node.replaceWith(parts);
      }
    }
    target.replaceChildren(content);
    for (const [span, time] of spans) {
      // Keep lower-contrast text legible at the start of the fade.
      const from = span.closest('a, blockquote, code, details') ? 0.92 : 0.68;
      play(span, { elapsed: now - time, from });
    }
  }

  return { play, cancel, replace };
})();
