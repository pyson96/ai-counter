/* Result dashboard: /result?job=job_000123
 *
 * Everything is drawn by hand in SVG and canvas - no chart library, no CDN.
 * The festival site may have no internet at all, and the page still has to work.
 *
 * Palette (validated with the data-viz validator against this page's card
 * surface #1a212b, dark mode):
 *   series-1 blue   #3987e5   entry / male / age bars
 *   series-2 orange #d95926   exit / female
 *   no-data  gray   #898781   unknown
 *     worst adjacent pair CVD dE 11.3, normal-vision dE 16.7, all >= 4:1 contrast.
 *   heatmap: the documented blue sequential ramp, one hue, monotone lightness.
 * Every chart is paired with a table view, so nothing is encoded by colour alone.
 */
'use strict';

const SERIES_1 = '#3987e5';
const SERIES_2 = '#d95926';
const NO_DATA = '#898781';
// the boundary itself, and the two sides of it
const LINE_COLOR = '#f2c14e';      // against both dark floors and bright ones
const IN_COLOR = '#6ee7a8';
const OUT_COLOR = '#f7a072';
const INK = '#e8edf4';
const INK_2 = '#c3c2b7';
const MUTED = '#898781';
const GRID = '#2e3a4a';

// Blue sequential ramp (palette steps 600 -> 100). Over the dimmed still the
// high end has to be the BRIGHT one to be visible at all, so the ramp runs
// dark -> light with magnitude; one hue, monotone lightness either way.
const HEAT_RAMP = ['#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4'];

const $ = (id) => document.getElementById(id);
const JOB_ID = new URLSearchParams(location.search).get('job');

const esc = (s) => String(s === null || s === undefined ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const fmtTime = (s) => {
  if (s === null || s === undefined || Number.isNaN(Number(s))) return '-';
  const t = Math.max(0, Math.round(Number(s)));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
};
const num = (n) => (n === null || n === undefined ? '-' : Number(n).toLocaleString('ko-KR'));

const AGE_LABEL = {
  under_20: '20세 미만', '20s': '20대', '30s': '30대',
  '40s': '40대', '50s': '50대', '60_plus': '60대 이상', unknown: '미상',
};
// used when the column band is too narrow for the full label to fit
const AGE_SHORT = {
  under_20: '~19', '20s': '20', '30s': '30',
  '40s': '40', '50s': '50', '60_plus': '60+', unknown: '?',
};
const GENDER_LABEL = { male: '남성', female: '여성', unknown: '미상' };

/* ------------------------------------------------------------------ shell */

function card(title, note, bodyHtml, id) {
  return `<section class="card viz" ${id ? `id="${id}"` : ''}>
    <div class="viz-head">
      <h2>${esc(title)}</h2>
      ${note ? `<p class="note">${note}</p>` : ''}
    </div>
    ${bodyHtml}
  </section>`;
}

function tableToggle(key) {
  return `<button class="ghost small" data-table="${key}" aria-expanded="false">표로 보기</button>`;
}

function dataTable(headers, rows, key) {
  return `<div class="tableview" id="table-${key}" hidden>
    <div class="scroll"><table>
      <thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c, i) =>
        `<t${i === 0 ? 'h scope="row"' : 'd'}>${esc(c)}</t${i === 0 ? 'h' : 'd'}>`).join('')}</tr>`).join('')}
      </tbody>
    </table></div>
  </div>`;
}

function legend(items) {
  return `<ul class="legend">${items.map((it) =>
    `<li><i style="background:${it.color}"></i>${esc(it.label)}</li>`).join('')}</ul>`;
}

/* -------------------------------------------------------------- stat tiles */

function statRow(tiles) {
  return `<div class="stats">${tiles.map((t) => `
    <div class="stat">
      <b>${esc(t.value)}</b>
      <span>${esc(t.label)}</span>
      ${t.sub ? `<em>${esc(t.sub)}</em>` : ''}
    </div>`).join('')}</div>`;
}

/* ---------------------------------------------------------------- heatmap */

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * p)));
  return sorted[i];
}

function mixHex(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.substr(i, 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.substr(i, 2), 16));
  return pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
}

function rampColor(t) {
  // t in 0..1 -> [r,g,b] along HEAT_RAMP
  const x = Math.max(0, Math.min(1, t)) * (HEAT_RAMP.length - 1);
  const i = Math.min(HEAT_RAMP.length - 2, Math.floor(x));
  return mixHex(HEAT_RAMP[i], HEAT_RAMP[i + 1], x - i);
}

function drawHeatmap(canvas, img, heat, opts) {
  const { cols, rows, cells } = heat;
  const W = img.naturalWidth || heat.video_width || 1280;
  const H = img.naturalHeight || heat.video_height || 720;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(img, 0, 0, W, H);
  if (opts.showHeat) {
    // dim the still so the ramp reads against it
    ctx.fillStyle = 'rgba(9,13,19,0.62)';
    ctx.fillRect(0, 0, W, H);

    const off = document.createElement('canvas');
    off.width = cols; off.height = rows;
    const octx = off.getContext('2d');
    const px = octx.createImageData(cols, rows);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const v = cells[y][x];
        const t = opts.vmax > 0 ? Math.min(1, v / opts.vmax) : 0;
        const [r, g, b] = rampColor(t);
        const o = (y * cols + x) * 4;
        px.data[o] = r; px.data[o + 1] = g; px.data[o + 2] = b;
        // empty cells stay fully transparent: the still shows through where
        // nobody ever stood, which is information too
        px.data[o + 3] = v <= 0 ? 0 : Math.round(60 + 195 * t);
      }
    }
    octx.putImageData(px, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, W, H);
  }

  if (opts.showRoi) {
    if (heat.crossing_line && heat.crossing_line.length >= 2) {
      drawCrossingLine(ctx, heat.crossing_line, heat.inside_side || 1, W);
    } else {
      drawPolygon(ctx, heat.entry_roi, SERIES_1, '입장 ROI', W);
      drawPolygon(ctx, heat.exit_roi, SERIES_2, '퇴장 ROI', W);
    }
  }
}

/* Which side of the line a point is on: +1, -1, or 0 when it cannot be
 * judged (past either end, or exactly on it). This mirrors
 * pipeline.signed_side_of_line so the picture agrees with the count. */
function sideOfLine(x, y, pts) {
  if (!pts || pts.length < 2) return 0;
  let best = null;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const [ax, ay] = pts[i];
    const [bx, by] = pts[i + 1];
    const vx = bx - ax;
    const vy = by - ay;
    const len2 = vx * vx + vy * vy;
    if (len2 <= 0) continue;
    const t = ((x - ax) * vx + (y - ay) * vy) / len2;
    const c = Math.max(0, Math.min(1, t));
    const dx = x - (ax + vx * c);
    const dy = y - (ay + vy * c);
    const d = Math.hypot(dx, dy);
    if (!best || d < best.d) best = { d, cross: vx * (y - ay) - vy * (x - ax), t, i };
  }
  if (!best) return 0;
  if ((best.i === 0 && best.t < 0) || (best.i === pts.length - 2 && best.t > 1)) return 0;
  if (best.cross === 0) return 0;
  return best.cross > 0 ? 1 : -1;
}

/* The crossing line, with arrows showing which way is IN.
 *
 * Which side counts as inside is the whole meaning of the line, so it is drawn
 * rather than described: an arrow at every segment points into the interior,
 * and the two sides are labelled. The +1 side is always (-vy, vx) - the same
 * convention the server uses. */
function drawCrossingLine(ctx, pts, inside, W) {
  if (!pts || pts.length < 2) return;
  const lw = Math.max(2, W / 400);
  ctx.setLineDash([]);
  ctx.lineWidth = lw * 2.2;
  ctx.strokeStyle = 'rgba(0,0,0,.55)';
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.stroke();
  ctx.lineWidth = lw;
  ctx.strokeStyle = LINE_COLOR;
  ctx.stroke();

  const arrow = Math.max(18, W / 26);
  const sign = inside >= 0 ? 1 : -1;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const [ax, ay] = pts[i];
    const [bx, by] = pts[i + 1];
    const vx = bx - ax;
    const vy = by - ay;
    const len = Math.hypot(vx, vy) || 1;
    // (-vy, vx) is the +1 side; flip it when the inside is the other one
    const nx = (-vy / len) * sign;
    const ny = (vx / len) * sign;
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    const tipX = mx + nx * arrow;
    const tipY = my + ny * arrow;
    ctx.lineWidth = Math.max(2, lw * 0.9);
    ctx.strokeStyle = IN_COLOR;
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    const head = arrow * 0.38;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - nx * head + ny * head * 0.6, tipY - ny * head - nx * head * 0.6);
    ctx.lineTo(tipX - nx * head - ny * head * 0.6, tipY - ny * head + nx * head * 0.6);
    ctx.closePath();
    ctx.fillStyle = IN_COLOR;
    ctx.fill();
  }

  // label both sides off the middle segment
  const mid = Math.floor((pts.length - 1) / 2);
  const [ax, ay] = pts[mid];
  const [bx, by] = pts[mid + 1];
  const vx = bx - ax;
  const vy = by - ay;
  const len = Math.hypot(vx, vy) || 1;
  const nx = (-vy / len) * sign;
  const ny = (vx / len) * sign;
  const mx = (ax + bx) / 2;
  const my = (ay + by) / 2;
  const off = arrow * 1.9;
  const f = Math.max(14, W / 48);
  ctx.font = `700 ${f}px system-ui, -apple-system, "Malgun Gothic", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const tag = (tx, ty, text, color) => {
    const w = ctx.measureText(text).width + f * 0.8;
    ctx.fillStyle = 'rgba(0,0,0,.6)';
    ctx.fillRect(tx - w / 2, ty - f * 0.75, w, f * 1.5);
    ctx.fillStyle = color;
    ctx.fillText(text, tx, ty);
  };
  tag(mx + nx * off, my + ny * off, '내부', IN_COLOR);
  tag(mx - nx * off, my - ny * off, '외부', OUT_COLOR);
  ctx.textAlign = 'start';
  ctx.textBaseline = 'alphabetic';
}

function drawPolygon(ctx, points, color, label, W) {
  if (!points || points.length < 3) return;
  ctx.lineWidth = Math.max(2, W / 500);
  ctx.strokeStyle = color;
  ctx.setLineDash([]);
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.closePath();
  ctx.stroke();
  const f = Math.max(15, W / 44);
  ctx.font = `700 ${f}px system-ui, -apple-system, "Malgun Gothic", sans-serif`;
  ctx.lineWidth = Math.max(3, f / 5);
  ctx.strokeStyle = 'rgba(0,0,0,.8)';
  ctx.strokeText(label, points[0][0] + 6, points[0][1] - 8);
  ctx.fillStyle = color;
  ctx.fillText(label, points[0][0] + 6, points[0][1] - 8);
}

function heatScaleLegend(vmax, fps) {
  const stops = 24;
  const bar = Array.from({ length: stops }, (_, i) => {
    const [r, g, b] = rampColor(i / (stops - 1));
    return `<i style="background:rgb(${r},${g},${b})"></i>`;
  }).join('');
  const secs = (v) => (fps ? `${(v / fps).toFixed(0)}초` : `${Math.round(v)}`);
  return `<div class="scale">
    <span class="scale-end">0</span>
    <div class="scale-bar">${bar}</div>
    <span class="scale-end">${esc(secs(vmax))} 이상</span>
  </div>
  <p class="note">칸마다 사람이 머문 누적 시간. 상위 2 % 구간은 최댓값으로 잘라 표시합니다.</p>`;
}

function heatmapCard(result) {
  const heat = result.heatmap;
  if (!heat || !result.frame_url) return '';
  const flat = [];
  heat.cells.forEach((row) => row.forEach((v) => { if (v > 0) flat.push(v); }));
  flat.sort((a, b) => a - b);
  const vmax = Math.max(1, percentile(flat, 0.98));
  const fps = (result.summary && result.summary.fps) || 0;

  // Table twin: dwell split by ROI - numbers a reader can act on, unlike a
  // 64x36 grid of cells.
  const split = roiDwell(heat);
  const isLine = !!(heat.crossing_line && heat.crossing_line.length >= 2);
  const total = split.entry + split.exit + split.other || 1;
  const pct = (v) => `${(v / total * 100).toFixed(1)}%`;
  const rowsTbl = isLine
    ? [['안쪽', secs(split.entry, fps), pct(split.entry)],
       ['바깥쪽', secs(split.exit, fps), pct(split.exit)],
       ['판정 불가(선 끝 너머)', secs(split.other, fps), pct(split.other)]]
    : [['입장 ROI 안', secs(split.entry, fps), pct(split.entry)],
       ['퇴장 ROI 안', secs(split.exit, fps), pct(split.exit)],
       ['그 외 영역', secs(split.other, fps), pct(split.other)]];

  return card('체류 히트맵', '영상 첫 프레임 위에 사람이 서 있던 위치(바운딩 박스 하단 15 % 지점)의 누적 시간을 표시합니다.', `
    <div class="viz-actions">
      <label class="chk"><input type="checkbox" id="opt-heat" checked> 히트맵</label>
      <label class="chk"><input type="checkbox" id="opt-roi" checked> 기준선</label>
      ${tableToggle('heat')}
    </div>
    <div class="canvas-wrap"><canvas id="heat-canvas"></canvas></div>
    ${heatScaleLegend(vmax, fps)}
    <div id="roi-legend">${legend(isLine
      ? [{ color: LINE_COLOR, label: '기준선' }, { color: IN_COLOR, label: '안쪽' },
         { color: OUT_COLOR, label: '바깥쪽' }]
      : [{ color: SERIES_1, label: '입장 ROI' }, { color: SERIES_2, label: '퇴장 ROI' }])}</div>
    ${dataTable(['영역', '누적 체류', '비중'], rowsTbl, 'heat')}
  `, 'card-heat');
}

function secs(v, fps) { return fps ? `${(v / fps).toFixed(0)}초` : `${Math.round(v)}`; }

function roiDwell(heat) {
  const out = { entry: 0, exit: 0, other: 0 };
  const W = heat.video_width || 0;
  const H = heat.video_height || 0;
  for (let y = 0; y < heat.rows; y += 1) {
    for (let x = 0; x < heat.cols; x += 1) {
      const v = heat.cells[y][x];
      if (!v) continue;
      const px = (x + 0.5) / heat.cols * W;
      const py = (y + 0.5) / heat.rows * H;
      if (heat.crossing_line && heat.crossing_line.length >= 2) {
        // with a line there are only two places to be
        const side = sideOfLine(px, py, heat.crossing_line);
        if (side === 0) out.other += v;
        else if (side === (heat.inside_side || 1)) out.entry += v;
        else out.exit += v;
      } else if (inPoly(px, py, heat.entry_roi)) out.entry += v;
      else if (inPoly(px, py, heat.exit_roi)) out.exit += v;
      else out.other += v;
    }
  }
  return out;
}

function inPoly(x, y, poly) {
  if (!poly || poly.length < 3) return false;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < xi + (y - yi) * (xj - xi) / ((yj - yi) || 1e-9)) inside = !inside;
  }
  return inside;
}

/* --------------------------------------------------------- chart plumbing */

const SVG_NS = 'http://www.w3.org/2000/svg';
function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** A bar rounded at its DATA end only - the baseline end stays square, so the
 *  mark reads as anchored to the axis instead of floating above it. */
function barPath(x, y, w, h, r, side) {
  const rad = Math.max(0, Math.min(r, w / 2, h));
  if (side === 'top') {
    return `M${x},${y + h} L${x},${y + rad} Q${x},${y} ${x + rad},${y} `
         + `L${x + w - rad},${y} Q${x + w},${y} ${x + w},${y + rad} L${x + w},${y + h} Z`;
  }
  if (side === 'left') {   // horizontal bar, rounded on its left end
    return `M${x + w},${y} L${x + rad},${y} Q${x},${y} ${x},${y + rad} `
         + `L${x},${y + h - rad} Q${x},${y + h} ${x + rad},${y + h} L${x + w},${y + h} Z`;
  }
  if (side === 'right') {  // horizontal bar, rounded on its right end
    return `M${x},${y} L${x + w - rad},${y} Q${x + w},${y} ${x + w},${y + rad} `
         + `L${x + w},${y + h - rad} Q${x + w},${y + h} ${x + w - rad},${y + h} L${x},${y + h} Z`;
  }
  if (side === 'both') {
    return `M${x + rad},${y} L${x + w - rad},${y} Q${x + w},${y} ${x + w},${y + rad} `
         + `L${x + w},${y + h - rad} Q${x + w},${y + h} ${x + w - rad},${y + h} `
         + `L${x + rad},${y + h} Q${x},${y + h} ${x},${y + h - rad} L${x},${y + rad} Q${x},${y} ${x + rad},${y} Z`;
  }
  return `M${x},${y} h${w} v${h} h${-w} Z`;
}

function niceTicks(max, count) {
  if (max <= 0) return [0, 1];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  const out = [];
  for (let v = 0; v <= max + step * 0.001; v += step) out.push(Math.round(v * 1000) / 1000);
  return out.length > 1 ? out : [0, max];
}

/** Grouped columns or two lines, chosen by how many buckets there are: a
 *  one-point "line" reads as broken, and 60 grouped columns read as a smear. */
function timeChart(host, buckets, fps) {
  const data = buckets.map((b) => ({
    label: fmtTime(b.start_seconds),
    end: fmtTime(b.end_seconds),
    entry: b.entry,
    exit: b.exit,
  }));
  const asColumns = data.length <= 8;
  const W = Math.max(host.clientWidth || 640, 320);
  const padL = 44; const padR = 16; const padT = 14; const padB = 42;
  const H = 260;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const max = Math.max(1, ...data.map((d) => Math.max(d.entry, d.exit)));
  const ticks = niceTicks(max, 4);
  const top = ticks[ticks.length - 1];
  const y = (v) => padT + plotH - (v / top) * plotH;

  const svg = el('svg', {
    viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
    role: 'img', 'aria-label': '시간대별 입장·퇴장 추이',
  });

  ticks.forEach((t) => {
    svg.appendChild(el('line', {
      x1: padL, x2: W - padR, y1: y(t), y2: y(t),
      stroke: t === 0 ? '#3c4c63' : GRID, 'stroke-width': 1,
    }));
    svg.appendChild(el('text', {
      x: padL - 8, y: y(t) + 4, 'text-anchor': 'end',
      fill: MUTED, 'font-size': 11, 'font-variant-numeric': 'tabular-nums',
    }, String(t)));
  });

  const band = plotW / data.length;
  if (asColumns) {
    const gap = 2;                       // 2px surface gap between adjacent bars
    const bw = Math.max(5, Math.min(26, band / 2 - gap));
    data.forEach((d, i) => {
      const cx = padL + band * (i + 0.5);
      [['entry', SERIES_1, -1], ['exit', SERIES_2, 1]].forEach(([key, color, side]) => {
        const v = d[key];
        const h = Math.max(v > 0 ? 2 : 0, (v / top) * plotH);
        if (!h) return;
        svg.appendChild(el('path', {
          d: barPath(cx + (side < 0 ? -bw - gap / 2 : gap / 2), padT + plotH - h,
                     bw, h, 4, 'top'),
          fill: color,
        }));
      });
    });
  } else {
    [['entry', SERIES_1], ['exit', SERIES_2]].forEach(([key, color]) => {
      const pts = data.map((d, i) => `${padL + band * (i + 0.5)},${y(d[key])}`).join(' ');
      svg.appendChild(el('polyline', {
        points: pts, fill: 'none', stroke: color, 'stroke-width': 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      }));
    });
  }

  // x labels - thinned so they never collide
  const every = Math.max(1, Math.ceil(data.length / Math.floor(plotW / 56)));
  data.forEach((d, i) => {
    if (i % every) return;
    svg.appendChild(el('text', {
      x: padL + band * (i + 0.5), y: H - padB + 18, 'text-anchor': 'middle',
      fill: MUTED, 'font-size': 11, 'font-variant-numeric': 'tabular-nums',
    }, d.label));
  });
  // hover layer: nearest bucket, crosshair + tooltip
  const hair = el('line', { y1: padT, y2: padT + plotH, stroke: '#5a6b84', 'stroke-width': 1, opacity: 0 });
  svg.appendChild(hair);
  const dots = [SERIES_1, SERIES_2].map((c) => {
    const d = el('circle', { r: 4.5, fill: c, stroke: '#1a212b', 'stroke-width': 2, opacity: 0 });
    svg.appendChild(d);
    return d;
  });
  const hit = el('rect', { x: padL, y: padT, width: plotW, height: plotH, fill: 'transparent' });
  svg.appendChild(hit);
  host.appendChild(svg);

  const tip = document.createElement('div');
  tip.className = 'tip';
  tip.hidden = true;
  host.appendChild(tip);

  const move = (ev) => {
    const r = svg.getBoundingClientRect();
    const cx = ev.touches ? ev.touches[0].clientX : ev.clientX;
    const sx = (cx - r.left) / r.width * W;
    const i = Math.max(0, Math.min(data.length - 1, Math.floor((sx - padL) / band)));
    const d = data[i];
    const px = padL + band * (i + 0.5);
    hair.setAttribute('x1', px); hair.setAttribute('x2', px); hair.setAttribute('opacity', 1);
    dots[0].setAttribute('cx', px); dots[0].setAttribute('cy', y(d.entry)); dots[0].setAttribute('opacity', 1);
    dots[1].setAttribute('cx', px); dots[1].setAttribute('cy', y(d.exit)); dots[1].setAttribute('opacity', 1);
    tip.hidden = false;
    tip.innerHTML = `<b>${esc(d.label)} – ${esc(d.end)}</b>
      <span><i style="background:${SERIES_1}"></i>입장 ${num(d.entry)}</span>
      <span><i style="background:${SERIES_2}"></i>퇴장 ${num(d.exit)}</span>`;
    const leftPct = px / W * 100;
    tip.style.left = `${Math.min(88, Math.max(4, leftPct))}%`;
  };
  const leave = () => {
    hair.setAttribute('opacity', 0);
    dots.forEach((d) => d.setAttribute('opacity', 0));
    tip.hidden = true;
  };
  hit.addEventListener('mousemove', move);
  hit.addEventListener('mouseleave', leave);
  hit.addEventListener('touchstart', move, { passive: true });
  hit.addEventListener('touchmove', move, { passive: true });
  hit.addEventListener('touchend', leave);
}

/** Part-to-whole: one horizontal stacked bar, 2px surface gaps, direct labels. */
function stackedBar(host, segments, total) {
  const W = Math.max(host.clientWidth || 640, 280);
  const H = 54;
  const gap = 2;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
    role: 'img', 'aria-label': '성별 분포' });
  let x = 0;
  const shown = segments.filter((s) => s.value > 0);
  const usable = W - gap * Math.max(0, shown.length - 1);
  shown.forEach((s, idx) => {
    const w = Math.max(3, s.value / (total || 1) * usable);
    // only the two ends of the whole bar are data ends; inner joins stay square
    const side = shown.length === 1 ? 'both'
      : idx === 0 ? 'left' : idx === shown.length - 1 ? 'right' : 'square';
    svg.appendChild(el('path', { d: barPath(x, 0, w, 28, 4, side), fill: s.color }));
    // label inside only when it actually fits, otherwise it goes under the bar
    const pct = `${(s.value / (total || 1) * 100).toFixed(0)}%`;
    if (w >= 52) {
      svg.appendChild(el('text', {
        x: x + w / 2, y: 19, 'text-anchor': 'middle', fill: '#07121f',
        'font-size': 12, 'font-weight': 700,
      }, pct));
    }
    svg.appendChild(el('text', {
      x: x + w / 2, y: 47, 'text-anchor': 'middle', fill: INK_2, 'font-size': 12,
    }, `${s.label} ${num(s.value)}`));
    x += w + gap;
  });
  host.appendChild(svg);
}

/** Ordered age bands: columns, one hue (bar length already carries magnitude),
 *  with the "미상" column in the no-data gray because it is not an age band. */
function columnChart(host, items) {
  const W = Math.max(host.clientWidth || 640, 300);
  const padL = 40; const padR = 12; const padT = 20; const padB = 38;
  const H = 240;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const max = Math.max(1, ...items.map((d) => d.value));
  const ticks = niceTicks(max, 4);
  const top = ticks[ticks.length - 1];
  const y = (v) => padT + plotH - (v / top) * plotH;

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
    role: 'img', 'aria-label': '연령대 분포' });
  ticks.forEach((t) => {
    svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: y(t), y2: y(t),
      stroke: t === 0 ? '#3c4c63' : GRID, 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: padL - 8, y: y(t) + 4, 'text-anchor': 'end',
      fill: MUTED, 'font-size': 11, 'font-variant-numeric': 'tabular-nums' }, String(t)));
  });

  const band = plotW / items.length;
  const bw = Math.max(10, Math.min(34, band - 14));
  items.forEach((d, i) => {
    const cx = padL + band * (i + 0.5);
    const h = d.value > 0 ? Math.max(2, (d.value / top) * plotH) : 0;
    if (h) {
      svg.appendChild(el('path', {
        d: barPath(cx - bw / 2, padT + plotH - h, bw, h, 4, 'top'), fill: d.color }));
      svg.appendChild(el('text', { x: cx, y: padT + plotH - h - 7, 'text-anchor': 'middle',
        fill: INK, 'font-size': 12, 'font-weight': 700 }, num(d.value)));
    }
    // a label only goes in if it fits; otherwise the short form does
    svg.appendChild(el('text', { x: cx, y: H - padB + 18, 'text-anchor': 'middle',
      fill: MUTED, 'font-size': 11 }, band >= 52 ? d.label : (d.short || d.label)));
  });
  host.appendChild(svg);
}

/* --------------------------------------------------------------- renderers */

/** The time card - or, when the video is shorter than one bucket, an honest
 *  note instead of a two-bar "trend" that the stat tiles already state. */
function timeCard(buckets, s, what, inLabel, outLabel) {
  const title = `시간대별 ${what}`;
  if (s.roi_configured === false) {
    return card(title, null,
      '<p class="note">ROI를 설정하면 시간대별 입퇴장 분포가 계산됩니다.</p>');
  }
  if (buckets.length < 2) {
    return card(title, null,
      `<p class="note">영상 길이(${fmtTime(s.video_duration_seconds)})가 집계 구간
       ${s.time_bucket_seconds}초보다 짧아 시간대별 분포가 없습니다.
       총계는 위 요약 카드를 참고하세요. 구간 폭은 config.yaml의
       <code>time_bucket_seconds</code>로 조정합니다.</p>`);
  }
  return card(title,
    `${s.time_bucket_seconds}초 단위 구간별 ${inLabel}·${outLabel} 수 · 가로축은 영상 경과 시간`, `
    <div class="viz-actions">${legend([
      { color: SERIES_1, label: inLabel }, { color: SERIES_2, label: outLabel },
    ])}${tableToggle('time')}</div>
    <div class="plot" id="plot-time"></div>
    ${dataTable(['구간 시작', '구간 끝', inLabel, outLabel],
      buckets.map((b) => [fmtTime(b.start_seconds), fmtTime(b.end_seconds), num(b.entry), num(b.exit)]),
      'time')}
  `);
}

/** Exits with no matching entry - usually people who were already inside when
 *  the recording started. Shown rather than hidden, so the number is auditable. */
function unmatchedNote(s) {
  if (s.roi_mode === 'line') return '';      // a crossing is the count; nothing is dropped
  if (!s.require_entry_before_exit || !s.unmatched_exit_count) return '';
  return `<section class="card viz note-card">
    <p class="note"><b>${num(s.unmatched_exit_count)}건</b>의 퇴장이
    입장 기록 없이 발생해 총 퇴장에서 제외됐습니다
    (<code>require_entry_before_exit: true</code>).
    보통 촬영 시작 시점에 이미 안에 있던 사람입니다.
    입구가 화각 밖이라면 config.yaml에서 이 옵션을 끄세요.</p>
  </section>`;
}

function renderPerson(result) {
  const s = result.summary;
  const buckets = (s.entries_by_time || []).map((b, i) => ({
    start_seconds: b.start_seconds,
    end_seconds: b.end_seconds,
    entry: b.count,
    exit: ((s.exits_by_time || [])[i] || {}).count || 0,
  }));

  const genderTotal = Object.values(s.gender).reduce((a, b) => a + b, 0);
  const genderSegs = [
    { key: 'male', label: GENDER_LABEL.male, value: s.gender.male, color: SERIES_1 },
    { key: 'female', label: GENDER_LABEL.female, value: s.gender.female, color: SERIES_2 },
    { key: 'unknown', label: GENDER_LABEL.unknown, value: s.gender.unknown, color: NO_DATA },
  ];
  const ageItems = Object.entries(s.age_groups).map(([k, v]) => ({
    key: k, label: AGE_LABEL[k] || k, short: AGE_SHORT[k] || k, value: v,
    color: k === 'unknown' ? NO_DATA : SERIES_1,
  }));

  const html = `
    ${statRow([
      { value: s.roi_configured ? num(s.entry_count) : '—', label: '총 입장',
        sub: !s.roi_configured ? '기준선 미설정'
          : (s.roi_mode === 'line' ? '밖 → 안 통과' : '입장 ROI 진입') },
      { value: s.roi_configured ? num(s.exit_count) : '—', label: '총 퇴장',
        sub: !s.roi_configured ? '기준선 미설정'
          : (s.roi_mode === 'line' ? '안 → 밖 통과'
             : (s.require_entry_before_exit ? '입장 후 퇴장 ROI 진입' : '퇴장 ROI 진입')) },
      { value: s.roi_configured ? num(s.unique_persons_entered) : num(s.unique_persons_detected),
        label: s.roi_configured ? '고유 방문객' : '검출 인원',
        sub: s.roi_configured ? `검출 ${num(s.unique_persons_detected)}명 중` : 'ROI 무관' },
      { value: fmtTime(s.video_duration_seconds), label: '영상 길이', sub: `${num(s.frames_processed)} 프레임` },
    ])}

    ${roiCard(result)}

    ${unmatchedNote(s)}

    ${videoCard(result)}

    ${heatmapCard(result)}

    ${timeCard(buckets, s, '방문객', '입장', '퇴장')}

    ${card('성별 분포', `${s.roi_configured ? '입장으로 집계된 방문객' : '검출된 인원'} ${num(genderTotal)}명 기준. "미상"은 추정 신뢰도가 기준 미만인 경우입니다.`, `
      <div class="viz-actions">${legend(genderSegs.map((g) => ({ color: g.color, label: g.label })))}${tableToggle('gender')}</div>
      <div class="plot" id="plot-gender"></div>
      ${dataTable(['성별', '인원', '비중'],
        genderSegs.map((g) => [g.label, num(g.value),
          `${(g.value / (genderTotal || 1) * 100).toFixed(1)}%`]), 'gender')}
    `)}

    ${card('연령대 분포', 'MiVOLO v2가 추정한 외형 기반 연령대이며, 신원 정보가 아닙니다.', `
      <div class="viz-actions">${legend([
        { color: SERIES_1, label: '연령대' }, { color: NO_DATA, label: '미상' },
      ])}${tableToggle('age')}</div>
      <div class="plot" id="plot-age"></div>
      ${dataTable(['연령대', '인원'], ageItems.map((a) => [a.label, num(a.value)]), 'age')}
    `)}

    ${card('방문객 목록', '익명 별칭입니다. 실제 신원과 무관합니다.', `
      <div class="scroll"><table>
        <thead><tr><th>별칭</th><th>입장</th><th>퇴장</th><th>추정 나이</th><th>연령대</th><th>성별</th><th>체류</th></tr></thead>
        <tbody>${(result.persons || []).map((p) => `<tr>
          <th scope="row">${esc(p.person_name)}</th>
          <td>${p.entry_time === null || p.entry_time === undefined ? '—' : fmtTime(p.entry_time)}</td>
          <td>${p.exit_time === null || p.exit_time === undefined ? '—' : fmtTime(p.exit_time)}</td>
          <td>${p.estimated_age === null || p.estimated_age === undefined ? '-' : Number(p.estimated_age).toFixed(1)}</td>
          <td>${esc(AGE_LABEL[p.age_group] || p.age_group)}</td>
          <td>${esc(GENDER_LABEL[p.estimated_gender] || p.estimated_gender)}</td>
          <td>${p.total_visible_seconds === null || p.total_visible_seconds === undefined ? '-' : `${Number(p.total_visible_seconds).toFixed(1)}초`}</td>
        </tr>`).join('')}</tbody>
      </table></div>
    `)}

    ${filesCard(result)}
  `;
  $('main').innerHTML = html;

  mountRoi(result);
  if (buckets.length >= 2) timeChart($('plot-time'), buckets, s.fps);
  stackedBar($('plot-gender'), genderSegs, genderTotal);
  columnChart($('plot-age'), ageItems);
  mountHeatmap(result);
}

function renderVehicle(result) {
  const s = result.summary;
  const buckets = (s.entries_by_time || []).map((b, i) => ({
    start_seconds: b.start_seconds,
    end_seconds: b.end_seconds,
    entry: b.count,
    exit: ((s.exits_by_time || [])[i] || {}).count || 0,
  }));
  const html = `
    ${statRow([
      { value: num(s.total_vehicle_entries), label: '차량 입차' },
      { value: num(s.total_vehicle_exits), label: '차량 출차' },
      { value: num(s.current_entered_vehicle_count), label: '현재 내부', sub: '출차 미확인' },
      { value: num(s.unique_plates), label: '고유 번호판', sub: `추적 ${num(s.vehicles_tracked)}대` },
    ])}

    ${videoCard(result)}

    ${result.frame_url ? card('촬영 구간', '전체 프레임이 ROI입니다. 영상 첫 프레임을 보관한 것입니다.',
      `<div class="canvas-wrap"><img src="${esc(result.frame_url)}" alt="영상 첫 프레임"></div>`) : ''}

    ${timeCard(buckets, s, '입·출차', '입차', '출차')}

    ${card('차량 목록', '번호판은 여러 프레임의 OCR 결과를 가중 투표해 확정한 값입니다.', `
      <div class="scroll"><table>
        <thead><tr><th>번호판</th><th>입차</th><th>출차</th><th>최초</th><th>최종</th><th>상태</th></tr></thead>
        <tbody>${(result.vehicles || []).map((v) => `<tr>
          <th scope="row">${esc(v.plate_number)}</th>
          <td>${fmtTime(v.entry_time)}</td><td>${fmtTime(v.exit_time)}</td>
          <td>${fmtTime(v.first_seen)}</td><td>${fmtTime(v.last_seen)}</td>
          <td>${esc(v.status)}</td>
        </tr>`).join('')}</tbody>
      </table></div>
    `)}

    ${filesCard(result)}
  `;
  $('main').innerHTML = html;
  if (buckets.length >= 2) timeChart($('plot-time'), buckets, s.fps);
}

/* ------------------------------------------------------------- ROI editor */

/** Draw / redraw the Entry and Exit polygons on the stored first frame.
 *
 *  Applying them replays tracks.jsonl.gz through the ROI state machine on the
 *  server - no GPU, no decoder. The source video is usually already deleted by
 *  the time this runs, which is exactly the point.
 */
const roi = { line: [], inside: 1, frame: null, busy: false };

function roiCard(result) {
  if (result.analysis_type !== 'person') return '';
  if (!result.frame_url) {
    return card('입퇴장 기준선', null,
      '<p class="err">첫 프레임 이미지가 없어 기준선을 설정할 수 없습니다.</p>');
  }
  if (!result.tracks_available) {
    return card('입퇴장 기준선', null,
      '<p class="err">이 작업에는 트랙 기록(tracks.jsonl.gz)이 없어 입퇴장을 '
      + '계산할 수 없습니다.</p>');
  }
  const configured = !!result.roi_configured;
  return card('입퇴장 기준선',
    configured
      ? '설정된 기준선입니다. 다시 그린 뒤 적용하면 입퇴장만 재계산됩니다 — GPU 분석은 다시 돌지 않습니다.'
      : '첫 프레임 위에 선을 긋고, 어느 쪽이 안인지 정하세요. 저장된 트랙 기록으로 입퇴장을 계산합니다.',
    `
    <div class="viz-actions">
      <button id="roi-flip" class="ghost small">안 / 밖 바꾸기</button>
      <button id="roi-undo" class="ghost small">Undo</button>
      <button id="roi-clear" class="ghost small">Clear</button>
    </div>
    <p class="note" id="roi-hint"></p>
    <div class="canvas-wrap"><canvas id="roi-canvas"></canvas></div>
    <div class="roi-status" id="roi-status"></div>
    <button id="roi-apply" class="big go" disabled>
      ${configured ? '기준선 적용 · 입퇴장 재계산' : '기준선 적용 · 입퇴장 계산'}
    </button>
    <p class="note" id="roi-msg"></p>
  `, 'card-roi');
}

function mountRoi(result) {
  const canvas = $('roi-canvas');
  if (!canvas) return;
  const cctx = canvas.getContext('2d');
  const saved = result.crossing_line || null;
  roi.line = (saved && saved.line ? saved.line : []).map((p) => [p[0], p[1]]);
  roi.inside = saved && saved.inside ? saved.inside : 1;
  roi.busy = false;

  const draw = () => {
    if (!roi.frame) return;
    cctx.drawImage(roi.frame, 0, 0);
    drawCrossingLine(cctx, roi.line, roi.inside, canvas.width);
    const r = Math.max(4, canvas.width / 180);
    roi.line.forEach((p) => {
      cctx.beginPath();
      cctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      cctx.fillStyle = LINE_COLOR;
      cctx.fill();
      cctx.lineWidth = 1;
      cctx.strokeStyle = '#000';
      cctx.stroke();
    });
    const ok = roi.line.length >= 2;
    $('roi-apply').disabled = !ok || roi.busy;
    $('roi-flip').disabled = !ok;
    $('roi-status').textContent =
      `${roi.line.length}점 · 프레임 ${canvas.width}x${canvas.height}`
      + (ok ? ' · 적용 가능' : ' · 2점 이상 필요');
  };

  const img = new Image();
  img.onload = () => {
    roi.frame = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    $('roi-hint').innerHTML =
      '화면을 탭하거나 클릭해 <b>선</b>을 그으세요 (2점 이상, 꺾어도 됩니다). '
      + '화살표가 가리키는 쪽이 <b>안</b>입니다 — 반대면 「안 / 밖 바꾸기」를 누르세요. '
      + '바운딩 박스 하단 15 % 지점이 밖에서 안으로 넘어오면 <b>입장</b>, '
      + '안에서 밖으로 나가면 <b>퇴장</b>입니다.';
    draw();
  };
  img.onerror = () => { $('roi-msg').textContent = '첫 프레임을 불러오지 못했습니다.'; };
  img.src = result.frame_url;

  $('roi-flip').addEventListener('click', () => { roi.inside = -roi.inside; draw(); });
  $('roi-undo').addEventListener('click', () => { roi.line.pop(); draw(); });
  $('roi-clear').addEventListener('click', () => { roi.line = []; draw(); });

  const addPoint = (ev) => {
    if (!roi.frame || roi.busy) return;
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const src = ev.touches && ev.touches.length ? ev.touches[0] : ev;
    // the canvas is sized to the frame's natural resolution, so the points we
    // push are already in ORIGINAL video pixels - the server never rescales
    const x = (src.clientX - rect.left) / rect.width * canvas.width;
    const y = (src.clientY - rect.top) / rect.height * canvas.height;
    roi.line.push([Math.round(Math.max(0, Math.min(canvas.width, x))),
                   Math.round(Math.max(0, Math.min(canvas.height, y)))]);
    draw();
  };
  canvas.addEventListener('click', addPoint);
  canvas.addEventListener('touchstart', addPoint, { passive: false });

  $('roi-apply').addEventListener('click', async () => {
    roi.busy = true;
    $('roi-apply').disabled = true;
    $('roi-msg').textContent = '저장된 트랙 기록으로 입퇴장을 계산하는 중...';
    try {
      const res = await fetch(`/api/jobs/${encodeURIComponent(JOB_ID)}/person-roi`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ line: roi.line, inside: roi.inside }),
      });
      const text = await res.text();
      if (!res.ok) throw new Error(text);
      $('roi-msg').textContent = '완료. 결과를 새로 불러옵니다...';
      boot();
    } catch (err) {
      roi.busy = false;
      $('roi-apply').disabled = false;
      $('roi-msg').textContent = `실패: ${err.message}`;
    }
  });
}

/** The annotated video, when the job opted into one. */
function videoCard(result) {
  if (!result.video_url) return '';
  return card('결과 영상',
    '박스 · 지속 ID · 접지점 · ROI · 누적 입퇴장 카운트가 그려진 영상입니다. '
    + '표시된 카운트는 아래 표와 같은 최종 수치입니다.', `
    <video class="resultvideo" src="${esc(result.video_url)}" controls preload="metadata"
           playsinline ${result.frame_url ? `poster="${esc(result.frame_url)}"` : ''}></video>
    <p class="note"><a class="dl" href="${esc(result.video_url)}" download>내려받기</a></p>
  `);
}

function filesCard(result) {
  const files = result.analysis_type === 'person'
    ? ['summary.json', 'persons.json', 'person_events.json', 'heatmap.json', 'job_metadata.json']
    : ['summary.json', 'vehicles.json', 'vehicle_events.json', 'job_metadata.json'];
  const note = result.video_url
    ? '구조화된 데이터와 첫 프레임, 그리고 요청하신 결과 영상을 보관합니다.'
    : '이 작업은 결과 영상을 생성하지 않았습니다. 구조화된 데이터와 첫 프레임만 보관합니다.';
  return card('원본 결과 파일', note, `
    <ul class="files">${files.map((f) =>
      `<li><a href="/api/jobs/${esc(JOB_ID)}/result/${esc(f)}" target="_blank" rel="noopener">${esc(f)}</a></li>`).join('')}
      ${result.frame_url ? `<li><a href="${esc(result.frame_url)}" target="_blank" rel="noopener">frame.jpg</a></li>` : ''}
    </ul>
    ${result.video_url ? `<p class="note">result.mp4 · <a class="dl" href="${esc(result.video_url)}" download>내려받기</a></p>` : ''}
    <p class="note">${esc(result.result_dir || '')}</p>
  `);
}

function mountHeatmap(result) {
  const canvas = $('heat-canvas');
  if (!canvas || !result.heatmap || !result.frame_url) return;
  const heat = result.heatmap;
  const flat = [];
  heat.cells.forEach((row) => row.forEach((v) => { if (v > 0) flat.push(v); }));
  flat.sort((a, b) => a - b);
  const vmax = Math.max(1, percentile(flat, 0.98));

  const img = new Image();
  img.onload = () => {
    const opts = { showHeat: true, showRoi: true, vmax };
    const redraw = () => drawHeatmap(canvas, img, heat, opts);
    redraw();
    $('opt-heat').addEventListener('change', (e) => { opts.showHeat = e.target.checked; redraw(); });
    $('opt-roi').addEventListener('change', (e) => {
      opts.showRoi = e.target.checked;
      $('roi-legend').hidden = !e.target.checked;   // never describe what is not drawn
      redraw();
    });
  };
  img.onerror = () => { $('card-heat').innerHTML += '<p class="err">첫 프레임 이미지를 불러오지 못했습니다.</p>'; };
  img.src = result.frame_url;
}

/* -------------------------------------------------------------------- boot */

document.addEventListener('click', (ev) => {
  const btn = ev.target.closest('button[data-table]');
  if (!btn) return;
  const box = $(`table-${btn.dataset.table}`);
  if (!box) return;
  const open = box.hidden;
  box.hidden = !open;
  btn.setAttribute('aria-expanded', String(open));
  btn.textContent = open ? '표 숨기기' : '표로 보기';
});

async function boot() {
  if (!JOB_ID) {
    $('main').innerHTML = '<p class="err">job 파라미터가 없습니다. 예: /result?job=job_000123</p>';
    return;
  }
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(JOB_ID)}/result`);
    const text = await res.text();
    if (!res.ok) throw new Error(text);
    const result = JSON.parse(text);
    const meta = result.metadata || {};
    $('title').textContent = result.filename || meta.source_video || JOB_ID;
    $('subtitle').textContent =
      `${JOB_ID} · ${result.analysis_type === 'person' ? '사람 분석' : '차량 분석'} · ${result.status}`;
    document.title = `${result.filename || JOB_ID} · 분석 결과`;
    if (result.analysis_type === 'person') renderPerson(result);
    else renderVehicle(result);
  } catch (err) {
    $('main').innerHTML = `<p class="err">결과를 불러오지 못했습니다: ${esc(err.message)}</p>`;
  }
}

boot();
window.addEventListener('resize', () => {
  clearTimeout(window.__rz);
  window.__rz = setTimeout(boot, 250);
});
