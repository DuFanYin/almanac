// The overview: the page's home, read in order. First the verdict (each length better, worse, wider or as usual for
// your targets) and what to do, simple: a sentence and a list, beside one length in full: its spread of returns over
// its figures. Then every length side by side, a column a length and a row a measure (the expected return, the bad
// and good case against usual, the LLMs' biggest move, where the worst and best 10% are), so it reads across. Then
// what is lit. Every row and figure opens its detail in the side panel.
import { $ } from "../dom.js";
import { decide, skip } from "./api.js";
import { day, pct, period, signed, tone } from "./format.js";
import { select } from "./inspector.js";
import { svgEl } from "../../chart.js";
import { button, el } from "./ui.js";
import { sentence } from "./verdict.js";

const go = page => { location.hash = page; };

// what each kind of suggestion is, in a few words
const KIND = { lean: "Your targets, moved by today's world", guard: "A guardrail: the bad case past your limit", trade: "What you hold", new_money: "New money", concentration: "A single stock held in size" };

// What there is to do, as one consistent list: the rules that fired and still hold today (a purchase waits while a
// guardrail moves out of what it buys: the limit wins), a guardrail a row each (one a class), the trades on what
// you hold in one row (where new money goes beneath), the single stocks in one. How today's world leans your targets
// is not a thing to do (the trades carry it once it is past your band): one note at the end. Each item {title, sub,
// open, acts (Done / Skip for a firing), note}.
function todoItems(s) {
  const v = s.view, dec = s.decisions || {}, items = [], sug = suggestions(s);
  const state = a => (a.status === "standing" ? `standing since ${day(a.since)}` : "suggestion");
  const first = t => t[0].toUpperCase() + t.slice(1);
  const buys = new Set((v.rules || []).filter(r => r.buys).map(r => r.id));
  const out = new Set(sug.filter(a => a.kind === "guard" && a.id.startsWith("guard-held:")).map(a => a.class));
  const waits = out.has("bitcoin");  // the rules' purchases are Bitcoin's
  for (const t of v.triggers.filter(t => t.standing !== false && (!dec[t.id] || dec[t.id].status === "open"))) {
    if (waits && buys.has(t.rule)) continue;
    const x = sentence(t, s);
    items.push({ title: x.head, sub: `Your rule triggered · ${period(t.period)} · since ${day(t.day)}`, open: () => select("a", `rule:${t.rule}`), acts: t });
  }
  for (const a of sug.filter(a => a.kind === "guard"))
    items.push({ title: a.title, sub: `${KIND.guard} · ${state(a)}${a.class === "bitcoin" && waits ? " · the Bitcoin purchase waits" : ""}`, open: () => select("a", a.id) });
  const trades = sug.filter(a => a.kind === "trade"), money = sug.find(a => a.kind === "new_money");
  if (trades.length === 1) items.push({ title: trades[0].title, sub: `${KIND.trade} · ${state(trades[0])}${money ? ` · ${moneySay(money)}` : ""}`, open: () => select("a", trades[0].id) });
  else if (trades.length) {
    const sell = trades.filter(a => a.size < 0), buy = trades.filter(a => a.size > 0);
    const names = xs => xs.map(a => a.title.replace(/^(Buy|Sell) [\d.]+% of /, "")).join(", ");
    items.push({ title: first([sell.length && `sell ${names(sell)}`, buy.length && `buy ${names(buy)}`].filter(Boolean).join("; ")) + ": a step each",
      sub: `${KIND.trade} · ${trades.length} trades${money ? ` · ${moneySay(money)}` : ""} · the Holdings tab`, open: () => go("holdings") });
  } else if (money) items.push({ title: money.title, sub: KIND.new_money, open: () => select("a", money.id), note: true });
  const names = sug.filter(a => a.kind === "concentration");
  if (names.length === 1) items.push({ title: names[0].title, sub: `${KIND.concentration} · suggestion`, open: () => select("a", names[0].id) });
  else if (names.length) items.push({ title: `Trim ${names.map(a => a.title.replace(/^Trim (\S+).*/, "$1")).join(" and ")} by a step each`, sub: `${KIND.concentration} · suggestion`, open: () => select("a", names[0].id) });
  const leans = sug.filter(a => a.kind === "lean");
  if (leans.length) {
    const say = a => `${a.title.replace(/^(Raise|Lower) (.*) by ([\d.]+)%$/, (_, w, k, x) => `${k} ${w === "Raise" ? "+" : "−"}${x}`)}`;
    const band = v.you?.limits?.band;
    items.push({ title: "Today's world moves your targets", sub: `${leans.map(say).join(", ")} (points of your mix) · trades follow once a class is ${band ? `more than ${(band * 100).toFixed(1)} points` : "far enough"} off its target`, open: () => select("a", leans[0].id), note: true });
  }
  return items;
}

function todo(s) {
  const rows = todoItems(s).map(x => {
    const li = el("li", "todo" + (x.acts ? " fired" : "") + (x.note ? " note" : "")), b = el("button", "todo-main", null, { type: "button" });
    b.append(el("b", null, x.title), el("span", "todo-sub", x.sub));
    b.onclick = x.open;
    li.append(b);
    if (x.acts) {
      const acts = el("span", "todo-acts");
      acts.append(button("Done", "small", () => decide(x.acts, "done")), button("Skip", "small bare", () => skip(x.acts)));
      li.append(acts);
    }
    return li;
  });
  if (!rows.some(r => !r.classList.contains("note"))) rows.unshift(el("li", "todo none", "Nothing to do: your targets as they are, every length within its limit."));
  return rows;
}

// the decision step's suggestions on your targets and on what you hold, guardrails first, new money last
const ORDER = { guard: 0, trade: 1, concentration: 2, lean: 3, new_money: 4 };
const moneySay = a => `new money to ${a.title.replace(/^New money: (.*) first$/, "$1")} first`;
const suggestions = s => [...(s.view.actions || []), ...(s.account?.actions || [])].filter(a => a.kind in ORDER).sort((a, b) => ORDER[a.kind] - ORDER[b.kind]);

// the lengths acted on, the overview's columns (a week is shown in full below, not acted on)
const LENGTHS = ["month", "short", "long", "longer"];
const WORD = { better: "better than usual", worse: "worse than usual", wider: "wider both ways", narrower: "narrower both ways", usual: "as usual" };

// the length shown in full, remembered
let horizon = (() => { try { return localStorage.getItem("scenarios-horizon") || "long"; } catch { return "long"; } })();

function choose(s, k) {
  horizon = k;
  try { localStorage.setItem("scenarios-horizon", k); } catch {}
  const sec = $("#overview");
  lengths(s, $("[data-lengths]", sec));
  detail(s, $("[data-world]", sec));
}

// every length side by side, a column a length and a row a measure, so each reads across: the word against usual,
// your targets' expected return, bad and good case, each lime or red as it is better or worse than usual (the limit red
// when the bad case is past it); then the scenario the LLMs moved furthest from the simulation, and the scenarios most
// of the worst and the best 10% are in, each with a dot lime or red as it is better or worse than usual for your
// targets. A column's head shows its length in full below
function lengths(s, box) {
  const v = s.view, hs = v.horizons || {}, sd = v.decision?.sides || {}, all = v.scenarios || [];
  const ks = LENGTHS.filter((k) => hs[k] && sd[k]);
  if (!ks.length) { box.replaceChildren(el("p", "muted", "Appears after the next daily run.")); return; }
  const grid = el("div", "len-grid");
  grid.style.setProperty("--n", ks.length);
  const label = (text, cls = "") => el("div", "len-row " + cls, text);
  const cell = (cls, ...kids) => { const c = el("div", "len-cell " + cls); c.append(...kids); return c; };
  const fig = (value, cls, ...sub) => { const s = el("small", "num"); s.append(...sub); return cell("", el("b", "num " + cls, value), s); };
  const scenario = (f, sub, cls = "") => {
    if (!f) return cell(cls, el("small", null, "–"));
    const b = el("button", "len-scen " + (hs[f.horizon]?.scenarios?.[f.id]?.targets?.side || ""), null, { type: "button" });
    b.append(el("span", null, f.name), el("small", "num", sub));
    b.onclick = () => select("f", f.id);
    return cell(cls, b);
  };
  grid.append(label(""));
  for (const k of ks) {
    const b = el("button", "len-head", null, { type: "button", "aria-pressed": String(k === horizon) });
    b.append(el("span", "len-name", `Next ${hs[k].label}`), el("span", "word " + sd[k].word, WORD[sd[k].word]));
    b.onclick = () => choose(s, k);
    grid.append(b);
  }
  const vs = (k, m) => tone(sd[k].now[m], sd[k].usual[m]);
  grid.append(label("Expected"), ...ks.map((k) => fig(signed(sd[k].now.expected, 0), vs(k, "expected"), `usual ${signed(sd[k].usual.expected, 0)}`)));
  grid.append(label("Bad case"), ...ks.map((k) => {
    const x = sd[k], lim = hs[k].limit, past = lim != null && x.now.bad < lim - 1e-9;
    return fig(signed(x.now.bad, 0), vs(k, "bad"), `usual ${signed(x.usual.bad, 0)} · `, el("span", past ? "bad" : "", `limit ${signed(lim, 0)}`));
  }));
  grid.append(label("Good case"), ...ks.map((k) => fig(signed(sd[k].now.good, 0), vs(k, "good"), `usual ${signed(sd[k].usual.good, 0)}`)));
  const moved = (f) => Math.abs(f.p - f.p_data);
  grid.append(label("Moved most by the LLMs", "gap"), ...ks.map((k) => {
    const f = all.filter((x) => x.horizon === k).sort((a, b) => moved(b) - moved(a))[0];
    return scenario(f, f && `${pct(f.p)}, ${f.p >= f.p_data ? "up" : "down"} from the simulation's ${pct(f.p_data)}`, "gap");
  }));
  const mostly = (k, key) => {
    const [id, x] = Object.entries(hs[k].stats?.targets?.[key] || {}).sort((a, b) => b[1] - a[1])[0] || [];
    const f = all.find((y) => y.id === id);
    return scenario(f, f && `${pct(x)} of the ${key === "worst_parts" ? "worst" : "best"} 10% · ${pct(f.p)} likely`);
  };
  grid.append(label("Worst 10% mostly in"), ...ks.map((k) => mostly(k, "worst_parts")));
  grid.append(label("Best 10% mostly in"), ...ks.map((k) => mostly(k, "best_parts")));
  box.replaceChildren(grid, el("p", "len-foot", "Bad and good case: the average of the worst and the best 10% of futures for your targets; usual: history's own periods, unweighted. Lime: better than usual for you, red: worse; the limit red when the bad case is past it. A scenario's dot: better or worse than usual for your targets. The simulation: history's periods weighted by today's indicators, before the LLMs; moved most by the LLMs: the scenario whose probability the LLMs' readers moved furthest from it."));
}

// one length in full: your targets' returns across its futures (with the LLMs' scenarios; without the LLM as a line)
// beside its figures, each against usual and without the LLM, and the deepest fall along the way
function detail(s, box) {
  const v = s.view, hs = v.horizons || {}, keys = Object.keys(hs);
  if (!keys.length) { box.replaceChildren(el("p", "muted", "The futures appear after the next daily run.")); return; }
  if (!hs[horizon]) horizon = keys.includes("long") ? "long" : keys[0];
  const d = hs[horizon], st = d.stats?.targets, label = v.you?.own ? "your targets" : "your base allocation";
  $("[data-world-h]").textContent = `${label[0].toUpperCase()}${label.slice(1)} over the next ${d.label}`;
  const tabs = el("div", "ui-seg small", null, { role: "tablist", "aria-label": "Over the next" });
  for (const k of keys) {
    const b = el("button", null, hs[k].label, { type: "button", role: "tab", "aria-selected": String(k === horizon), "aria-label": `Next ${hs[k].label}` });
    b.onclick = () => choose(s, k);
    tabs.append(b);
  }
  if (!st?.all) { box.replaceChildren(tabs); return; }
  const a = st.all, n = st.no_llm, h = st.history || {}, past = d.limit != null && a.bad < d.limit;
  const figs = el("dl", "detail-figs");
  const fig = (name, value, sub, cls = "") => { const r = el("div"); r.append(el("dt", null, name), el("dd", "num " + cls, value), el("dd", "sub", sub)); figs.append(r); };
  fig("Expected", signed(a.expected, 0), `usual ${signed(h.expected, 0)} · without the LLM ${signed(n.expected, 0)}`, tone(a.expected, h.expected));
  fig("Bad case (worst 10%)", signed(a.bad, 0), `usual ${signed(h.bad, 0)} · without the LLM ${signed(n.bad, 0)} · your limit ${signed(d.limit, 0)}`, past ? "bad" : tone(a.bad, h.bad));
  fig("Good case (best 10%)", signed(a.good, 0), `usual ${signed(h.good, 0)} · without the LLM ${signed(n.good, 0)}`, tone(a.good, h.good));
  if (a.dd) fig("Deepest fall on the way (worst 10%)", signed(a.dd.bad, 0), `typical ${signed(a.dd.median, 0)} · without the LLM ${signed(n.dd?.bad, 0)}`);
  const chart = el("div"), body = el("div", "detail");
  body.append(chart, figs);
  box.replaceChildren(tabs, body);
  const w = chart.clientWidth || (matchMedia("(min-width: 1001px)").matches ? 640 : 360);
  chart.append(histogram(a.hist, n.hist, d.limit, a.expected, w, Math.round(Math.max(150, Math.min(240, w * 0.42)))));
}

function histogram(h, h0, limit, mean, W, H) {
  const B = H - 20, svg = svgEl("svg", { class: "fut-hist", viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img", "aria-label": "Distribution of returns" });
  // the bins come with the data, fitted to this horizon (allocation.py chart_bins), its end bins the tails folded in.
  // The axis is centred on the expected, as far each side as the nearer tail reaches (0 and the limit kept in view);
  // what lies past it, the end bins included, is a share at that end, not a bar
  const edge = i => h.lo + i * h.step, m = h.share.length;
  const far = Math.max(mean - edge(1), edge(m - 1) - mean);
  const d = Math.min(far, Math.max(Math.min(mean - edge(1), edge(m - 1) - mean), ...[0, limit].filter(r => r != null).map(r => Math.abs(mean - r) + h.step / 2)));
  const kept = [...h.share.keys()].filter(i => i > 0 && i < m - 1 && edge(i) >= mean - d - 1e-9 && edge(i + 1) <= mean + d + 1e-9);
  const a = kept[0], b = kept[kept.length - 1], sum = xs => xs.reduce((s, p) => s + p, 0);
  const below = sum(h.share.slice(0, a)), above = sum(h.share.slice(b + 1));
  const lo = mean - d, hi = mean + d, x = r => ((r - lo) / (hi - lo)) * W, bw = (h.step / (hi - lo)) * W;
  const top = Math.max(...kept.map(i => h.share[i]), ...(h0 ? kept.map(i => h0.share[i]) : [0]));
  for (const i of kept) {
    const p = h.share[i], r = edge(i);
    svgEl("rect", { x: x(r) + 0.5, y: B - (p / top) * (B - 8), width: bw - 1, height: (p / top) * (B - 8), class: r < (limit ?? -9) ? "hb bad" : r < 0 ? "hb neg" : "hb pos" }, svg);
  }
  if (h0) svgEl("polyline", { points: kept.map(i => `${x(edge(i)) + bw / 2},${B - (h0.share[i] / top) * (B - 8)}`).join(" "), class: "line0" }, svg);
  // a mark's label beside its line, on the side with room; the limit's to the left, so it does not meet the others
  const mark = (r, cls, text, left) => {
    svgEl("line", { x1: x(r), x2: x(r), y1: 4, y2: B, class: cls }, svg);
    if (!text) return;
    const toLeft = left ? x(r) > 70 : x(r) > W - 70;
    const t = svgEl("text", { x: x(r) + (toLeft ? -3 : 3), y: 12, class: cls, "text-anchor": toLeft ? "end" : "start" }, svg);
    t.textContent = text;
  };
  mark(0, "zero");
  if (limit != null) mark(limit, "lim", "your limit", true);
  mark(mean, "mean", "expected", false);
  // ticks: a round step giving at most seven; the shares past the axis at its two ends
  const tick = [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1].find(s => (hi - lo) / s <= 7) ?? 1;
  const end = (p, left) => {
    if (p < 0.0005) return false;
    const t = svgEl("text", { x: left ? 0 : W, y: H - 4, class: "beyond", "text-anchor": left ? "start" : "end" }, svg);
    t.textContent = left ? `${pct(p, p < 0.01 ? 1 : 0)} below` : `${pct(p, p < 0.01 ? 1 : 0)} above`;
    return true;
  };
  const room = [end(below, true) ? 64 : 14, end(above, false) ? 64 : 14];
  for (let r = Math.ceil(lo / tick - 1e-9) * tick; r <= hi + 1e-9; r += tick) {
    if (x(r) < room[0] || x(r) > W - room[1]) continue;  // a label cut by the edge or meeting an end's share
    const t = svgEl("text", { x: x(r), y: H - 4, class: "tick", "text-anchor": "middle" }, svg);
    t.textContent = Math.abs(r) < 1e-9 ? "0" : `${r > 0 ? "+" : "−"}${Math.round(Math.abs(r) * 100)}%`;
  }
  return svg;
}

// what is lit today, red first, all of them: each a link to its detail, and the map with every indicator
function signals(v) {
  const lit = v.signposts.filter(x => ["red", "yellow"].includes(x.state)).sort((a, b) => (a.state === "red" ? 0 : 1) - (b.state === "red" ? 0 : 1));
  const p = el("p", "signals"), pushing = v.signposts.filter(x => Object.keys(x.push || {}).length).length;
  p.append(el("span", "fig-label", `Red or yellow today · ${lit.length} of ${v.signposts.length} indicators · ${pushing} shifting the odds today`));
  for (const x of lit) {
    const b = el("button", "sig " + x.state, x.name.replace(/\s*\(.*\)$/, ""), { type: "button" });
    b.onclick = () => select("s", x.id);
    p.append(b);
  }
  p.append(button("All indicators on the map", "link", () => go("map")));
  return p;
}

// the one finding, read first: each length better, worse, wider or as usual for your targets, the lengths saying the
// same together
function finding(s) {
  const v = s.view, sd = v.decision?.sides || {}, hs = v.horizons || {}, by = {};
  for (const k of LENGTHS) if (sd[k] && hs[k]) (by[sd[k].word] ||= []).push(hs[k].label);
  const join = (xs) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}` : xs[0]);
  const said = Object.entries(by).map(([w, ls], i) => `${i ? WORD[w] : WORD[w][0].toUpperCase() + WORD[w].slice(1) + " for your targets"} over the next ${join(ls)}`);
  return said.length ? `${said.join("; ")}.` : "The world appears after the next daily run.";
}

export function overview(s) {
  const sec = $("#overview"), v = s.view;
  $("[data-finding]", sec).textContent = finding(s);
  const n = todoItems(s).filter((x) => !x.note).length;  // the rows of the list, the notes apart
  $("[data-todo-h]", sec).textContent = n ? `To do · ${n}` : "Nothing to do";
  $("[data-todo]", sec).replaceChildren(...todo(s));
  lengths(s, $("[data-lengths]", sec));
  detail(s, $("[data-world]", sec));
  $("[data-signals]", sec).replaceChildren(signals(v));
}
