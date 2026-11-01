// The System tab: the whole system as one diagram, from src/board/flow.py. Each stage of the day in
// its phase, left to right; what feeds what (a dashed line is read the next day); where the LLM enters, each such
// stage behind a check; and how each stage went today. Choosing a stage opens it in the side panel, with what feeds
// it and what it feeds; on a narrow screen the phases stack and the lines are left out.
import { svgEl } from "../../chart.js";
import { select } from "./inspector.js";
import { page } from "./page.js";
import { el } from "./ui.js";

export const SIDES = { data: "Computed from data", llm: "The LLM writes it, behind a check", you: "Yours", out: "What you read" };

let listening = false;

// the page's own words, each once
const WORDS = [
  ["Indicator", "a reading computed from the data (the VIX, credit spreads, the Fed's moves, a text the LLM reads), green, yellow or red, and what it is expected to say about the factors"],
  ["Factor", "one of seven things that describe what happens: the S&P 500, the 10-year yield, the real yield, inflation, unemployment, the dollar, financial conditions"],
  ["Length", "how far ahead: a week, a month, three months, a year, three years"],
  ["The futures", "every period of that length in history since 1971, each a possible future, with a weight saying how much it looks like today's"],
  ["The simulation", "the futures weighted by today's indicators alone, before the LLM"],
  ["Scenario", "a story one LLM writes about a length: the ranges of one or two factors; a period of history can be in several or none"],
  ["The readers", "four LLMs with different viewpoints; each says how many times as likely a scenario is than the simulation says (×1.2: a fifth likelier), for what the simulation leaves out"],
  ["The reviewer", "an LLM that checks the readers' reasons against the data, giving no numbers; a scenario it finds fault with is read again"],
  ["The world", "the futures with the scenarios' probabilities put in: the one belief every forecast and decision reads"],
  ["Usual", "history's own periods, unweighted"],
  ["Bad case, good case", "the average of the worst, or the best, 10% of futures for your targets"],
  ["Moving your targets", "the decision step shifts a class's target toward what today's world says of it, at most your risk level's limit a class"],
  ["Rebalancing band", "how far a class may drift from its target before a trade is suggested"],
  ["Guardrail", "your limit on the bad case at each length; when it is past, money moves into a shelter (cash, Treasuries, gold)"],
];

export function systemPage(v) {
  const sys = v.system, wrap = el("div", "sys");
  if (!sys) { wrap.append(el("p", "prose muted", "The diagram appears after the next daily run.")); return wrap; }
  const failed = sys.stages.filter(s => s.state === "failed"), llm = sys.stages.filter(s => s.side === "llm");
  wrap.append(el("p", "prose", `${sys.stages.length} stages, run each day after the US close in six phases. In ${llm.length} of them the LLM writes something, and each must pass its check before it counts. ${failed.length ? `Failed today: ${failed.map(s => s.name).join(", ")}.` : "Nothing failed today."}`));
  const words = el("details", "earlier"), dl = el("dl", "facts");
  words.append(el("summary", null, "The words this page uses"));
  for (const [k, x] of WORDS) dl.append(el("dt", null, k), el("dd", null, x));
  words.append(dl);
  wrap.append(words);
  const legend = el("div", "sys-legend");
  for (const [k, label] of Object.entries(SIDES)) { const x = el("span", "sys-key side-" + k); x.append(el("i"), label); legend.append(x); }
  const back = el("span", "sys-key back"); back.append(el("i"), "read the next day");
  legend.append(back);
  wrap.append(legend);

  const stage = el("div", "sys-stage"), grid = el("div", "sys-grid"), cards = {};
  for (const ph of sys.phases) {
    const col = el("div", "sys-col");
    col.append(el("h3", "sys-phase", ph.label));
    for (const s of sys.stages.filter(x => x.phase === ph.id)) {
      const b = el("button", `sys-card side-${s.side} state-${s.state}`, null, { type: "button", "data-id": s.id, title: s.what });
      const top = el("span", "sys-top");
      top.append(el("span", "sys-dot", null, { "aria-label": s.state }), el("b", null, s.name));
      b.append(top, el("span", "sys-when", s.when));
      if (s.note) b.append(el("span", "sys-note", s.note));
      if (s.error) b.append(el("span", "sys-err", s.error));
      b.onclick = () => select("sys", s.id);
      cards[s.id] = b;
      col.append(b);
    }
    grid.append(col);
  }
  stage.append(grid);
  wrap.append(stage);
  requestAnimationFrame(() => draw(stage, cards, sys));
  if (!listening) {
    listening = true;
    document.addEventListener("sc-select", () => mark());
    let t;
    addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { const s = document.querySelector(".sys-stage"); if (s && s.offsetParent) draw(s, null, page.state.view.system); }, 150); });
  }
  return wrap;
}

// the lines between the cards, laid over the grid once it has its size
function draw(stage, cards, sys) {
  cards ??= Object.fromEntries([...stage.querySelectorAll(".sys-card")].map(b => [b.dataset.id, b]));
  stage.querySelector("svg")?.remove();
  const box = stage.getBoundingClientRect();
  if (!box.width || getComputedStyle(stage).getPropertyValue("--sys-lines").trim() === "none") return;
  const svg = svgEl("svg", { width: box.width, height: box.height, viewBox: `0 0 ${box.width} ${box.height}`, "aria-hidden": "true" });
  stage.prepend(svg);
  const order = Object.fromEntries(sys.phases.map((p, i) => [p.id, i])), phase = Object.fromEntries(sys.stages.map(s => [s.id, s.phase]));
  const r = id => { const x = cards[id].getBoundingClientRect(); return { l: x.left - box.left, r: x.right - box.left, y: (x.top + x.bottom) / 2 - box.top, t: x.top - box.top, b: x.bottom - box.top }; };
  for (const [a, b] of sys.edges) {
    if (!cards[a] || !cards[b]) continue;
    const A = r(a), B = r(b), da = order[phase[a]], db = order[phase[b]];
    let d, cls = "sys-edge";
    if (db > da) {
      const k = (B.l - A.r) / 2;
      d = `M${A.r},${A.y} C${A.r + k},${A.y} ${B.l - k},${B.y} ${B.l},${B.y}`;
    } else if (db === da) {  // within a phase: around the cards' right side
      const x = A.r + 10;
      d = `M${A.r},${A.y} C${x},${A.y} ${x},${B.y} ${B.r},${B.y}`;
    } else {  // read the next day: under the cards, back to the left
      cls += " back";
      d = `M${A.l},${A.y} C${A.l - 30},${A.y} ${B.r + 30},${B.y} ${B.r},${B.y}`;
    }
    svgEl("path", { d, class: cls, "data-a": a, "data-b": b }, svg);
  }
  mark();
}

// the chosen stage: its lines and the stages it touches stand out, the rest steps back
function mark() {
  const stage = document.querySelector(".sys-stage");
  if (!stage) return;
  const sel = page.sel?.startsWith("sys:") ? page.sel.slice(4) : null, near = new Set(sel ? [sel] : []);
  for (const p of stage.querySelectorAll("path")) {
    const on = sel && (p.dataset.a === sel || p.dataset.b === sel);
    p.classList.toggle("on", !!on);
    p.classList.toggle("dim", !!sel && !on);
    if (on) { near.add(p.dataset.a); near.add(p.dataset.b); }
  }
  for (const c of stage.querySelectorAll(".sys-card")) {
    c.classList.toggle("sel", c.dataset.id === sel);
    c.classList.toggle("dim", !!sel && !near.has(c.dataset.id));
  }
}
