// Reference, Indicators: one small chart each, half a year, each on its own scale.
import { svgEl } from "../../chart.js";
import { num, signed } from "./format.js";
import { el } from "./ui.js";

function trend(values) {
  const vs = (values || []).filter(x => x != null);
  const svg = svgEl("svg", { class: "trend", viewBox: "0 0 100 16", preserveAspectRatio: "none", "aria-hidden": "true" });
  if (vs.length > 1) {
    const lo = Math.min(...vs), hi = Math.max(...vs), span = hi - lo || 1;
    svgEl("polyline", { points: vs.map((x, i) => `${(i / (vs.length - 1)) * 100},${14 - ((x - lo) / span) * 12}`).join(" ") }, svg);
  }
  return svg;
}
// a table that scrolls sideways on its own on a narrow screen, not the page

export function smallMultiples(v) {
  const grid = el("div", "multiples");
  for (const i of v.indicators) {
    const cell = el("div", "mini", null, { title: `${i.name}, ${i.day}` }), chg = i.month_ago ? i.value / i.month_ago - 1 : null;
    const head = el("div", "mini-head");
    head.append(el("span", null, i.name), el("b", "num", num(i.value)));
    cell.append(head);
    const vs = (i.spark || []).filter(x => x != null);
    const svg = svgEl("svg", { viewBox: "0 0 100 40", preserveAspectRatio: "none", "aria-hidden": "true" }, cell);
    if (vs.length > 1) {
      const lo = Math.min(...vs), hi = Math.max(...vs), span = hi - lo || 1, Y = x => 37 - ((x - lo) / span) * 34;
      svgEl("polyline", { points: vs.map((x, k) => `${(k / (vs.length - 1)) * 100},${Y(x)}`).join(" ") }, svg);
    }
    cell.append(el("small", "num", chg == null ? "–" : `${signed(chg)} in a month`));
    grid.append(cell);
  }
  const wrap = el("div");
  wrap.append(grid, el("p", "foot", "Daily closes over the last six months, each on its own scale; the figure is the last value."));
  return wrap;
}
