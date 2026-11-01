// The board's own mark: a question's two sides on one track. The simulation (the world of the indicators, with the market)
// is a grey dot with its 90% band, the world with the LLM's scenarios a hollow ring, the line between them the LLM's
// pull, and the final the white tick. Read left to right on a scale from 0.
import { svgEl } from "../../chart.js";
import { pct } from "./format.js";

const scaleTop = xs => Math.min(1, Math.max(0.25, Math.ceil((Math.max(...xs) + 0.04) * 4) / 4));

export function strip(s, { width = 260, height = 30, labels = true } = {}) {
  const xs = [s.final, s.math, s.llm, ...(s.mathIv || []), ...(s.finalIv || [])].filter(x => x != null);
  const svg = svgEl("svg", { class: "strip", viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
    "aria-label": `Simulation ${pct(s.math)}${s.llm != null ? `, with the LLM ${pct(s.llm)}` : ""}, final ${pct(s.final)}` });
  if (!xs.length) return svg;
  const top = scaleTop(xs), L = 4, R = width - (labels ? 34 : 4), mid = 12, X = x => L + (R - L) * (x / top);
  svgEl("line", { x1: L, x2: R, y1: mid, y2: mid, class: "track" }, svg);
  if (s.mathIv) svgEl("rect", { x: X(s.mathIv[0]), y: mid - 4, width: Math.max(1, X(s.mathIv[1]) - X(s.mathIv[0])), height: 8, rx: 2, class: "band" }, svg);
  if (s.llm != null && s.math != null) svgEl("line", { x1: X(s.math), x2: X(s.llm), y1: mid, y2: mid, class: "pull" }, svg);
  if (s.math != null) svgEl("circle", { cx: X(s.math), cy: mid, r: 4, class: "math" }, svg);
  if (s.llm != null) svgEl("circle", { cx: X(s.llm), cy: mid, r: 4.5, class: "llm" }, svg);
  if (s.final != null) svgEl("rect", { x: X(s.final) - 1, y: mid - 8, width: 2, height: 16, class: "final" }, svg);
  if (labels) {
    const t = svgEl("text", { x: width, y: mid + 4, "text-anchor": "end", class: "end" }, svg);
    t.textContent = pct(top);
  }
  return svg;
}

// the strip's key, in words: each mark named as drawn
export const stripKey = s => [
  `simulation ${pct(s.math)}${s.mathIv ? ` (${pct(s.mathIv[0])}–${pct(s.mathIv[1])})` : ""}`,
  s.llm != null ? `with the LLM's scenarios ${pct(s.llm)}` : "no LLM",
].join(" · ");
