// A number's lineage as a graph, in the map's layers: evidence, readings, questions, scenarios, actions and you, the
// number in its own layer. A line on top says what it was made from; steps of one kind past a few fold into one card
// that opens on a click. Each card says its step in plain words and is as tall as that; choosing one shows its stored
// values below and lights the path from it to the number. The dialogs close on their button and on a click outside.
import { svgEl } from "../../chart.js";
import { $ } from "../dom.js";
import { day, num, pct } from "./format.js";
import { el, escapeHtml } from "./ui.js";

const probLine = p => (p && typeof p === "object" ? Object.entries(p).filter(([, x]) => typeof x === "number").map(([k, x]) => `${escapeHtml(k.replaceAll("_", " "))} ${pct(x)}`).join(" · ") : "");

const KIND = {
  raw_snapshot: "Data", signpost: "Signpost", base_rate: "History", lr_fit: "Fit", market: "Market", calibration: "Calibration",
  computed: "Data model", llm_panel: "LLM", llm_prior: "LLM (old prior)", research: "Research", data_snapshot: "Data given to the LLM",
  judgment: "Your view", balance: "The LLM's record", probability: "Final", rule_trigger: "Rule fired", exposure: "Holdings",
  resolution: "Resolved", distribution: "Futures", llm_scenario: "A future the LLM wrote", scenario_set: "The LLM's scenarios", proposal: "Proposed forecast",
  decision: "Your decision", action: "Action", you: "You", backtest: "Held through history",
};
const PLURAL = { raw_snapshot: "data series", signpost: "signposts", base_rate: "histories", computed: "data models", market: "markets", probability: "finals", llm_panel: "panel answers" };

// the layer a step belongs to, as on the map; the LLM's scenarios (and its old named futures) belong to the futures
const LAYERS = ["Data", "Indicators", "Forecasts", "Futures", "Actions", "You"];
function layer(n) {
  const futures = /^llm:scenarios-/.test(n.name);
  switch (n.kind) {
    case "raw_snapshot": case "market": case "data_snapshot": case "research": return 0;
    case "signpost": case "lr_fit": case "calibration": return 1;
    case "base_rate": case "computed": case "llm_panel": case "llm_prior": case "judgment": case "probability": case "resolution": case "balance": return futures ? 3 : 2;
    case "distribution": case "llm_scenario": case "scenario_set": case "proposal": return 3;
    case "rule_trigger": case "action": case "decision": case "backtest": return 4;
    case "you": case "exposure": return 5;
    default: return 2;
  }
}

function say(n) {
  const v = n.value, m = n.meta || {};
  switch (n.kind) {
    case "raw_snapshot": return `${num(v.last)} on ${day(v.last_day)}: ${(v.days ?? 0).toLocaleString("en-US")} days since ${(v.first_day ?? "").slice(0, 4)} from ${m.source}${m.symbol ? " (" + m.symbol + ")" : ""}.`;
    case "signpost": return `${v.value == null ? "no value" : num(v.value)}, ${v.state}; ${m.expr}, red ${m.red}.`;
    case "base_rate": return `${probLine(v.p)}: ${(m.windows ?? 0).toLocaleString("en-US")} windows of ${m.series}, ${(m.from ?? "").slice(0, 4)}–${(m.to ?? "").slice(0, 4)}, about ${v.n_eff} independent.`;
    case "lr_fit": return Object.keys(v.lr || {}).length ? `Fitted by state: ${Object.keys(v.lr).join(", ")}.` : "Too few cases to fit: the ratios set by hand.";
    case "market": return `Kalshi: ${probLine(v)}.`;
    case "computed": return `${m.llm_upstream?.length ? `A model upstream: ${m.llm_upstream.join(", ")}. ` : ""}${probLine(v)}; pooled ${Object.entries(m.weights || {}).map(([k, w]) => `${k} ×${w.toFixed(2)}`).join(", ")}${Object.keys(m.applied || {}).length ? `, moved by ${Object.keys(m.applied).join(", ")}` : ""}.`;
    case "llm_panel": return `${probLine(v.p)}; ${v.n} panelists over ${v.rounds} rounds, their spread ${(v.spread || []).map(x => pct(x, 1)).join(" → ")}${m.cited ? `; cited most: ${Object.keys(m.cited).slice(0, 3).map(x => x.split(":")[1] ?? x).join(", ")}` : ""}.`;
    case "llm_prior": return `${probLine(v.p)}; ${v.n} answers (before the LLM).`;
    case "research": return `${probLine(v.p)}; ${v.found} news items, ${v.notes} checked notes.`;
    case "data_snapshot": return `${v.length} numbers, each with its id.`;
    case "judgment": return `${probLine(v.p ?? v)}${v.note ? `, because <q>${escapeHtml(v.note)}</q>` : ""}.`;
    case "balance": return `The LLM's bias ±${pct(v.bias_sd, 1)} from ${v.n} resolved forecasts${v.proven ? "; proven" : "; not proven yet"}.`;
    case "probability": return `${probLine(v)}${m.weights ? `; ${Object.entries(m.weights).map(([k, w]) => `${k} ×${w.toFixed(2)}`).join(", ")}` : ""}.`;
    case "rule_trigger": return `${escapeHtml(m.rule?.explain ?? m.rule?.title ?? "")} Fired on ${day(v.day)}.`;
    case "exposure": return `The holdings as of ${day(m.date)} (kept with the account data).`;
    case "resolution": return v.bucket == null ? `Withdrawn${m.how ? ` (${escapeHtml(m.how.replace(/^withdrawn: /, ""))})` : ""}.` : `Resolved: ${escapeHtml(String(v.bucket))}.`;
    case "distribution": return m.llm_moved != null ? `Every period of history over ${m.days} days, the LLM's scenarios moving ${pct(m.llm_moved)} of the weight.` : `${(m.draws ?? 0).toLocaleString("en-US")} futures: ${pct(m.parts?.history ?? 1)} from history's periods, the rest from ${Object.keys(m.parts || {}).length - 1} named futures the LLM wrote.`;
    case "scenario_set": return `The LLM's scenarios: ${Object.entries(v || {}).map(([k, x]) => `${escapeHtml(k)} ${pct(x)}`).join(", ")}.`;
    case "llm_scenario": return `${escapeHtml(v.name ?? "")}, written by the LLM: ${escapeHtml(m.story ?? "")}`;
    case "proposal": return `${escapeHtml(v.title)}: ${m.checks?.passed ? "passed its checks" : "did not pass"}.`;
    case "decision": return `You ${v.status} it${v.note ? `: <q>${escapeHtml(v.note)}</q>` : ""}.`;
    case "action": return `${escapeHtml(v.title ?? "")}${m.detail ? `: ${escapeHtml(m.detail)}` : ""}`;
    case "you": return `Risk level ${v.profile}, a bad case of at most −${pct(v.limits?.worst_loss)}; ${m.own ? "your targets" : "the risk level's base allocation"} over ${Object.keys(v.policy || {}).length} classes${m.holdings ? "; with holdings" : ""}.`;
    case "backtest": return `${Object.keys(v || {}).length} mixes held through history.`;
    default: return "";
  }
}

const FOLD = 4;  // steps of one kind in one layer past this many fold into one card

export async function showLineage(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) return;
  const nodes = await r.json();
  if (!nodes.length) return;
  const by = Object.fromEntries(nodes.map(n => [n.id, n]));
  const root = nodes.reduce((a, n) => (n.id > a.id ? n : a), nodes[0]);
  const children = {};
  for (const n of nodes) for (const p of n.parents) (children[p.id] ??= []).push(n.id);
  const open = new Set();  // folded groups opened
  const summary = () => {
    const count = {};
    for (const n of nodes) if (n !== root) count[n.kind] = (count[n.kind] ?? 0) + 1;
    const parts = Object.entries(count).sort((a, b) => layer(by[nodes.find(x => x.kind === a[0]).id]) - layer(by[nodes.find(x => x.kind === b[0]).id]))
      .map(([k, c]) => (c > 1 ? `${c} ${PLURAL[k] ?? (KIND[k] ?? k).toLowerCase() + "s"}` : (KIND[k] ?? k).toLowerCase()).replace(/\bllm\b/g, "LLM"));
    return `${KIND[root.kind] ?? root.kind} ${root.name.replace(/^[a-z_]+:/, "")}, made from ${parts.join(", ")}.`;
  };
  $("#lineage [data-raw]").replaceChildren(el("p", "pane-note", "Choose a step for its stored values and the path from it to the number."));
  const draw = () => {
    // the layers present, left to right; within one, steps of a kind past FOLD fold into a card
    const cols = LAYERS.map(() => []);
    const groups = {};
    for (const n of nodes) (groups[`${layer(n)}|${n.kind}`] ??= []).push(n);
    for (const [key, ns] of Object.entries(groups)) {
      const col = +key.split("|")[0];
      if (ns.length > FOLD && !open.has(key) && !ns.includes(root)) cols[col].push({ group: key, nodes: ns });
      else cols[col].push(...ns.map(n => ({ node: n })));
    }
    const box = el("div", "dag"), grid = el("div", "dag-cols"), cards = {}, rep = {};
    const svg = svgEl("svg", { "aria-hidden": "true" }, box);
    box.append(grid);
    LAYERS.forEach((name, i) => {
      if (!cols[i].length) return;
      const c = el("div", "dag-col");
      c.append(el("p", "dag-layer", name));
      for (const item of cols[i]) {
        if (item.group) {
          const [, kind] = item.group.split("|"), b = el("button", "dn group", null, { type: "button" });
          b.append(el("span", "kind", `${item.nodes.length} ${PLURAL[kind] ?? KIND[kind] ?? kind}`), el("span", "say", item.nodes.map(n => n.name.replace(/^[a-z_]+:/, "")).join(", ")), el("span", "more", "Show each"));
          b.onclick = () => { open.add(item.group); draw(); };
          c.append(b);
          for (const n of item.nodes) rep[n.id] = b;
          continue;
        }
        const n = item.node, b = el("button", "dn" + (n === root ? " root" : "") + (n.excluded ? " excluded" : ""), null, { type: "button" });
        const line = el("span", "say");
        line.innerHTML = say(n);
        b.append(el("span", "kind", KIND[n.kind] ?? n.kind.replaceAll("_", " ")), el("b", null, n.name.replace(/^[a-z_]+:/, "")), line);
        if (n.excluded) b.append(el("span", "bad", "marked as bad data"));
        b.onclick = () => choose(n.id);
        cards[n.id] = b;
        rep[n.id] = b;
        c.append(b);
      }
      grid.append(c);
    });
    const wrap = el("div", "dag-scroll");
    wrap.append(box);
    $("#lineage [data-tree]").replaceChildren(el("p", "dag-sum", summary()), wrap);
    // the edges once the cards are laid out: from a step's side toward the number to the next step's facing side
    const edges = [];
    requestAnimationFrame(() => {
      const o = box.getBoundingClientRect();
      svg.setAttribute("width", box.scrollWidth);
      svg.setAttribute("height", box.scrollHeight);
      const seen = new Set();
      for (const n of nodes) for (const p of n.parents) {
        const a = rep[p.id], b = rep[n.id];
        if (!a || !b || a === b || seen.has(`${p.id}>${n.id}`)) continue;
        seen.add(`${p.id}>${n.id}`);
        const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), right = ra.left < rb.left;
        const x1 = (right ? ra.right : ra.left) - o.left, x2 = (right ? rb.left : rb.right) - o.left;
        const y1 = ra.top + ra.height / 2 - o.top, y2 = rb.top + rb.height / 2 - o.top, mx = (x1 + x2) / 2;
        edges.push([p.id, n.id, svgEl("path", { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}` }, svg)]);
      }
    });
    const choose = id => {
      const n = by[id], up = new Set([id]), todo = [id];  // the path from this step to the number
      while (todo.length) for (const c of children[todo.pop()] || []) if (!up.has(c)) { up.add(c); todo.push(c); }
      const lit = new Set([...up].map(x => rep[x]));
      for (const c of new Set(Object.values(rep))) c.classList.toggle("on", lit.has(c));
      for (const [a, b, p] of edges) p.classList.toggle("on", up.has(a) && up.has(b));
      box.classList.add("focus");
      $("#lineage [data-raw]").replaceChildren(el("p", "pane-kind", `${KIND[n.kind] ?? n.kind} · ${n.name} · made ${new Date(n.created_at * 1000).toLocaleString("en-GB", { hour12: false, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}`),
        el("pre", null, JSON.stringify({ value: n.value, meta: n.meta, made_by: n.parents.map(p => `${p.step}, code ${p.code}`) }, null, 1)));
    };
  };
  draw();
  $("#lineage").showModal();
}

export function initDialogs() {
  for (const d of document.querySelectorAll("dialog")) {
    $("[data-close]", d).onclick = () => d.close();
    d.addEventListener("click", e => { if (e.target === d) d.close(); });
  }
}
