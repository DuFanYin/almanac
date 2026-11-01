// Chart building blocks shared by Status and Finance: SVG helpers, ticks, axes, bars,
// hover marks and the one floating tooltip. Each page draws its own chart types with these.
// Status imports this module (s/<version>/chart.js); fin/make_web.py inlines it into the
// report. Looks come from ui.css (.ui-chart, .ui-tip, .ui-key).

const NS = "http://www.w3.org/2000/svg";

export function svgEl(tag, attrs, parent, style) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (style) Object.assign(e.style, style);
  if (parent) parent.appendChild(e);
  return e;
}

// An <svg> filling `host` (a .ui-chart), reused across redraws; returns [svg, width].
export function chartSvg(host, height, attrs = {}) {
  let svg = host.querySelector("svg");
  if (!svg) svg = svgEl("svg", { role: "img", ...attrs }, host);
  const W = Math.max(260, host.clientWidth);
  svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
  svg.setAttribute("height", height);
  svg.replaceChildren();
  return [svg, W];
}

// Round tick values covering [lo, hi], about n steps.
export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const t = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
  if (t[t.length - 1] < hi) t.push(+(t[t.length - 1] + step).toFixed(10));
  return t;
}

// Horizontal grid with a label per tick; the zero line (else the lowest) is the baseline.
export function yGrid(svg, ticks, Y, left, right, fmt) {
  const base = ticks.includes(0) ? 0 : ticks[0];
  for (const v of ticks) {
    const y = Y(v);
    svgEl("line", { x1: left, x2: right, y1: y, y2: y, class: v === base ? "baseline" : "gridline" }, svg);
    svgEl("text", { x: left - 6, y: y + 3.5, "text-anchor": "end" }, svg).textContent = fmt(v);
  }
}

// Bar from (x, y) with size w × h; only the data end (`end`: top / bottom / left / right) is rounded.
export function barPath(x, y, w, h, end = "top", r = 4) {
  if (w <= 0 || h <= 0) return "";
  r = Math.min(r, end === "top" || end === "bottom" ? Math.min(w / 2, h) : Math.min(h / 2, w));
  const X = x + w, Y = y + h;
  switch (end) {
    case "top": return `M${x},${Y}V${y + r}Q${x},${y} ${x + r},${y}H${X - r}Q${X},${y} ${X},${y + r}V${Y}Z`;
    case "bottom": return `M${x},${y}V${Y - r}Q${x},${Y} ${x + r},${Y}H${X - r}Q${X},${Y} ${X},${Y - r}V${y}Z`;
    case "right": return `M${x},${y}H${X - r}Q${X},${y} ${X},${y + r}V${Y - r}Q${X},${Y} ${X - r},${Y}H${x}Z`;
    default: return `M${X},${y}H${x + r}Q${x},${y} ${x},${y + r}V${Y - r}Q${x},${Y} ${x + r},${Y}H${X}Z`;
  }
}

// Crosshair at x plus a ringed dot per value; replaces the previous hover marks.
export function hoverMarks(svg, x, top, bottom, dots = []) {
  svg.querySelector(".hover")?.remove();
  const g = svgEl("g", { class: "hover" }, svg);
  svgEl("line", { x1: x, x2: x, y1: top, y2: bottom, class: "cross" }, g);
  for (const d of dots) svgEl("circle", { cx: x, cy: d.y, r: 4, "stroke-width": 2 }, g, { fill: d.color, stroke: "var(--ui-surface)" });
  return g;
}
export const clearHover = (svg) => svg.querySelector(".hover")?.remove();

// Labels at the right end of lines: pushed apart so they never overlap, kept inside `bottom`.
export function spreadLabels(ends, bottom, gap = 13) {
  ends.sort((a, b) => a.y - b.y);
  ends.forEach((e, i) => { e.ly = i ? Math.max(e.y, ends[i - 1].ly + gap) : e.y; });
  const over = ends.length ? ends[ends.length - 1].ly - bottom : 0;
  if (over > 0) ends.forEach((e) => { e.ly -= over; });
  return ends;
}

// Pointer tracking in viewBox units: onMove(px, py, clientX, clientY), onLeave().
export function bindPointer(svg, onMove, onLeave) {
  const move = (e) => {
    const r = svg.getBoundingClientRect(), k = svg.viewBox.baseVal.width / r.width;
    onMove((e.clientX - r.left) * k, (e.clientY - r.top) * k, e.clientX, e.clientY);
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerdown", move);
  svg.addEventListener("pointerleave", () => { onLeave?.(); hideTip(svg); });
}

// Index of the value in sorted `xs` nearest to x.
export function nearest(xs, x) {
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; xs[mid] < x ? (lo = mid) : (hi = mid); }
  return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
}

// The one floating tooltip. rows: [{ name, value, color? }]; text only, never HTML.
let tip;
export function showTip(owner, cx, cy, title, rows) {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "ui-tip";
    tip.hidden = true;
    document.body.appendChild(tip);
  }
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  tip.replaceChildren(el("div", "t", title));
  for (const r of rows) {
    const row = el("div", "r");
    if (r.color) { const k = el("i", "ui-key"); k.style.background = r.color; row.append(k); }
    row.append(el("span", null, r.name), el("b", null, r.value));
    tip.append(row);
  }
  tip.hidden = false;
  tip._owner = owner;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = cx + 14, y = cy - h - 12;
  if (x + w > innerWidth - 8) x = cx - w - 14;
  if (x < 8) x = 8;
  if (y < 8) y = cy + 16;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
export function hideTip(owner) {
  if (tip && (!owner || tip._owner === owner)) tip.hidden = true;
}

// Resolved value of a CSS custom property (for colours that must be fixed at draw time).
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
