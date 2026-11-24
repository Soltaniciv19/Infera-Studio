'use strict';

window.InferaStudioLineChart = (() => {
  const measured = (value) => Number.isFinite(value) && value >= 0;

  function scale(peak) {
    if (!peak) return { end: 1, ticks: [0, .25, .5, .75, 1] };
    const rough = Math.max(peak / 4, 1e-9);
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 2.5, 5, 10].find((candidate) => candidate * magnitude >= rough) * magnitude;
    const intervals = Math.ceil(peak / step);
    return { end: step * intervals, ticks: Array.from({ length: intervals + 1 }, (_, i) => step * i) };
  }

  function trace(ctx, points) {
    const slopes = points.slice(1).map(([x, y], index) => (y - points[index][1]) / (x - points[index][0]));
    // Limit the slope at each point so the curve stays between the measured values.
    const tangents = points.map((_, index) => {
      const before = slopes[index - 1];
      const after = slopes[index];
      return before * after > 0 ? Math.sign(before) * Math.min(Math.abs(before), Math.abs(after)) : 0;
    });
    ctx.moveTo(...points[0]);
    for (let i = 1; i < points.length; i++) {
      const [left, first] = points[i - 1];
      const [right, last] = points[i];
      const third = (right - left) / 3;
      ctx.bezierCurveTo(left + third, first + third * tangents[i - 1],
        right - third, last - third * tangents[i], right, last);
    }
  }

  function create(canvas, reading, { series, formatX, formatY, formatValue = formatY, discrete = false }) {
    const ctx = canvas.getContext('2d');
    let data = [];
    let activeX = null;
    let invertX;

    function draw() {
      const { clientWidth: width, clientHeight: height } = canvas;
      if (!width || !height || !data.length) return;
      const ratio = window.devicePixelRatio;
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const style = getComputedStyle(canvas);
      const color = (name) => style.getPropertyValue(name).trim();
      const peak = Math.max(0, ...data.flatMap((point) => series.map(({ key }) => point[key]).filter(measured)));
      const yScale = scale(peak);
      const xScale = scale(data.at(-1).x);
      const start = discrete ? data[0].x : 0;
      const end = discrete ? data.at(-1).x : xScale.end;
      ctx.font = '11px ' + style.fontFamily;
      const left = Math.ceil(Math.max(...yScale.ticks.map((tick) => ctx.measureText(formatY(tick)).width))) + 12;
      const right = width - Math.max(15, ctx.measureText(formatX(end)).width / 2 + 1);
      const top = 13;
      const bottom = height - 25;
      if (right <= left || bottom <= top) return;
      const x = (value) => end === start ? (left + right) / 2 : left + (right - left) * (value - start) / (end - start);
      const y = (value) => bottom - (bottom - top) * value / yScale.end;
      invertX = (position) => start + (position - left) * (end - start) / (right - left);

      const stride = Math.ceil(data.length / 6);
      const xTicks = discrete ? data.filter((_, index) => index % stride === 0).map((point) => point.x)
        : xScale.ticks;
      if (discrete && xTicks.at(-1) !== end) xTicks.push(end);
      ctx.fillStyle = color('--muted');
      ctx.strokeStyle = color('--line');
      ctx.lineWidth = 1;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const tick of yScale.ticks) {
        const position = Math.round(y(tick)) + .5;
        ctx.beginPath();
        ctx.moveTo(left, position);
        ctx.lineTo(right, position);
        ctx.stroke();
        ctx.fillText(formatY(tick), left - 9, position);
      }
      ctx.textAlign = 'center';
      for (const tick of xTicks) {
        ctx.fillText(formatX(tick), x(tick), height - 9);
      }

      function dot(px, py, radius, fill) {
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, 2 * Math.PI);
        ctx.fillStyle = fill;
        ctx.fill();
      }

      for (const { key, tone } of series) {
        const stroke = color(tone);
        const runs = [];
        let run = [];
        for (const point of data) {
          if (measured(point[key])) {
            run.push([x(point.x), y(point[key])]);
          } else if (run.length) {
            runs.push(run);
            run = [];
          }
        }
        if (run.length) runs.push(run);
        for (const points of runs) {
          if (points.length > 1) {
            const gradient = ctx.createLinearGradient(0, top, 0, bottom);
            gradient.addColorStop(0, stroke);
            gradient.addColorStop(1, 'transparent');
            ctx.beginPath();
            trace(ctx, points);
            ctx.lineTo(points.at(-1)[0], bottom);
            ctx.lineTo(points[0][0], bottom);
            ctx.closePath();
            ctx.fillStyle = gradient;
            ctx.globalAlpha = .24;
            ctx.fill();
            ctx.globalAlpha = 1;
            ctx.beginPath();
            trace(ctx, points);
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 2.25;
            ctx.lineCap = 'round';
            ctx.stroke();
          }
          if (discrete || points.length === 1) {
            for (const [px, py] of points) dot(px, py, 3, stroke);
          } else {
            dot(...points.at(-1), 3, stroke);
          }
        }
      }

      const active = data.find((point) => point.x === activeX);
      if (!active) return;
      const position = x(active.x);
      ctx.strokeStyle = color('--muted');
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(position, top);
      ctx.lineTo(position, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const { key, tone } of series) {
        if (measured(active[key])) {
          dot(position, y(active[key]), 5, color('--panel'));
          dot(position, y(active[key]), 3.5, color(tone));
        }
      }
    }

    function select(point) {
      activeX = point?.x ?? null;
      const values = point && series.filter(({ key }) => measured(point[key]))
        .map(({ key, label }) => label + ' ' + formatValue(point[key]));
      reading.textContent = point ? [formatX(point.x), ...values].join(' · ') : '';
      draw();
    }

    const readable = (point) => series.some(({ key }) => measured(point[key]));
    canvas.addEventListener('pointermove', (event) => {
      if (!invertX) return;
      const target = invertX(event.offsetX);
      let nearest;
      for (const point of data) {
        if (readable(point) && (!nearest || Math.abs(point.x - target) < Math.abs(nearest.x - target))) nearest = point;
      }
      if (nearest?.x !== activeX) select(nearest);
    });
    canvas.addEventListener('pointerleave', () => select(null));
    canvas.addEventListener('blur', () => select(null));
    canvas.addEventListener('keydown', (event) => {
      const points = data.filter(readable);
      const index = points.findIndex((point) => point.x === activeX);
      const next = {
        ArrowLeft: index < 0 ? points.length - 1 : Math.max(0, index - 1),
        ArrowRight: Math.min(points.length - 1, index + 1),
        Home: 0,
        End: points.length - 1,
      }[event.key];
      if (next == null) return;
      event.preventDefault();
      select(points[next]);
    });

    return {
      draw,
      update(points) {
        data = points;
        if (activeX !== null) select(data.find((point) => point.x === activeX && readable(point)));
        else draw();
      },
    };
  }

  return { create };
})();
