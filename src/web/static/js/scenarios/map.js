// 3, how it is made: the board as a map in its layers. The world from left to right: evidence (the data series and
// the other sources: Kalshi, your view), readings (the signposts), the world at each length it is read at (your
// targets' bad case against its limit, how much the LLM's scenarios moved it), and the forecasts read off it. You on the
// far right: your style, your targets, your rules, your holdings. Between them the actions, where the two meet: arrows
// come in from both sides. A signpost's line into a length is as thick as it pushes it today. Choosing anything opens
// it in the inspector beside the map.
import { svgEl } from "../../chart.js";
import { $ } from "../dom.js";
import { select } from "./inspector.js";
import { day, num, pct, pts, signed } from "./format.js";
import { page } from "./page.js";
import { keyQuestions, keyLabel, shortName, sides, spName, spValue } from "./questions.js";
import { button, el } from "./ui.js";

let mapAll = (() => { try { return localStorage.getItem("scenarios-map") === "all"; } catch { return false; } })();  // today's chains first
let listening = false;
const NAMES = ["Data", "Indicators", "The world", "Forecasts", "Actions", "You"];
const SCEN_ROWS = 6;  // a length's node is as tall as this many rows: its name and three lines
const W_COL = 2, Q_COL = 3;  // the world's column, the forecasts' column

export function lineageMap(v, dec) {
  const wrap = el("div", "map"), g = v.graph;
  if (!g) { wrap.append(el("p", "prose muted", "The map appears after the next daily run.")); return wrap; }
  const main = keyQuestions(v).map(q => q.id), hs = v.horizons || {}, futs = v.scenarios || [];
  const qs = v.questions.filter(q => !q.resolved && q.kind !== "scenario");
  qs.sort((a, b) => (main.indexOf(a.id) + 1 || 99) - (main.indexOf(b.id) + 1 || 99));
  const acts = v.actions || [], me = g.you || [];
  const open = new Set(v.triggers.filter(t => !dec[t.id] || dec[t.id].status === "open").map(t => t.rule));
  const live = a => (a.kind === "rule" ? open.has(a.rule) || a.status === "holds" : true);  // an adjustment or a move is today's
  const all = [g.data.map(d => "d:" + d.id), v.signposts.map(s => "s:" + s.id), Object.keys(hs).map(k => "h:" + k),
    qs.map(q => "q:" + q.id), acts.map(a => "a:" + a.id), me.map(y => "y:" + y.id)];
  const push = g.push || {}, drives = new Set(g.edges.filter(([a, b]) => a.startsWith("q:") && b.startsWith("a:")).map(([a]) => a));
  // today's chains: the lit signposts and those pushing the world today, every length, the four key forecasts, the
  // actions live today and what feeds them, and you
  const today = new Set([...v.signposts.filter(s => ["red", "yellow"].includes(s.state)).map(s => "s:" + s.id),
    ...Object.keys(push).map(e => e.split(">")[0]), ...Object.keys(hs).map(k => "h:" + k), ...main.map(id => "q:" + id),
    ...acts.filter(live).map(a => "a:" + a.id), ...me.map(y => "y:" + y.id)]);
  for (const [a, b] of g.edges) if (b.startsWith("a:") && today.has(b) && !a.startsWith("y:")) today.add(a);
  for (const [a, b] of g.edges) if (a.startsWith("d:") && today.has(b)) today.add(a);
  const keep = mapAll ? null : today;
  const cols = keep ? all.map(c => c.filter(id => keep.has(id))) : all;
  const out = {}, inn = {};
  for (const [a, b] of g.edges) if (!keep || (keep.has(a) && keep.has(b))) { (out[a] ??= []).push(b); (inn[b] ??= []).push(a); }
  const rank = id => { for (const c of cols) { const i = c.indexOf(id); if (i >= 0) return i / Math.max(1, c.length - 1); } return 0.5; };
  const mean = xs => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0.5);
  cols[4].sort((a, b) => mean((inn[a] || []).filter(x => !x.startsWith("y:")).map(rank)) - mean((inn[b] || []).filter(x => !x.startsWith("y:")).map(rank)));
  // the indicators by how much they push today, then the data beside its indicators: shorter lines
  const pushed = id => Object.entries(push).filter(([e]) => e.startsWith(id + ">")).reduce((m, [, x]) => Math.max(m, x), 0);
  cols[1].sort((a, b) => pushed(b) - pushed(a));
  cols[0].sort((a, b) => mean((out[a] || []).map(rank)) - mean((out[b] || []).map(rank)));
  const PITCH = keep ? 24 : 20, TOP = 34, rows = Math.max(SCEN_ROWS, cols[W_COL].length * 5, ...cols.map(c => c.length));
  const H = TOP + rows * PITCH + 8, W = Math.max(980, $("#map [data-map]")?.clientWidth || 980);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "group", "aria-label": "How data, indicators, the world, forecasts and your settings lead to actions" }, wrap);
  // x of each column: [label start, label end / value, edge out]; edges come in at the label start; You's edges
  // leave from its start, leftwards, into the actions' end
  const X = [[0, 0.13 * W, 0.138 * W], [0.162 * W, 0.292 * W, 0.3 * W], [0.35 * W, 0.5 * W, 0.508 * W],
    [0.545 * W, 0.695 * W, 0.703 * W], [0.735 * W, 0.9 * W, 0.908 * W], [0.93 * W, W, W]];
  const pos = {};
  cols.forEach((c, i) => {
    const gap = i === W_COL ? 5 : 1, offset = TOP + ((rows - c.length * gap) * PITCH) / 2 + (i === W_COL ? 2 * PITCH : 0);
    c.forEach((id, j) => { pos[id] = { col: i, y: offset + j * gap * PITCH + PITCH / 2 }; });
  });
  NAMES.forEach((name, i) => {
    const tx = svgEl("text", { x: i === 0 ? X[0][1] : X[i][0], y: 12, class: "col" + (i === 5 ? " you" : ""), "text-anchor": i === 0 ? "end" : "start" }, svg);
    tx.textContent = keep ? `${name} · ${cols[i].length} of ${all[i].length}` : i === W_COL ? name : `${name} · ${all[i].length}`;
  });
  const edgesG = svgEl("g", { class: "edges" }, svg), nodesG = svgEl("g", { class: "nodes" }, svg);
  const paths = [];
  const tip = (p, text) => { p.append(svgEl("title", {})); p.lastChild.textContent = text; };
  for (const [a, b] of g.edges) {
    const pa = pos[a], pb = pos[b];
    if (!pa || !pb) continue;
    const back = pa.col > pb.col, key = `${a}>${b}`;  // from you, leftwards into an action
    const x1 = back ? X[pa.col][0] - 6 : X[pa.col][2], x2 = back ? X[pb.col][2] : X[pb.col][0] - 6, mx = (x1 + x2) / 2;
    const p = svgEl("path", { d: `M${x1},${pa.y} C${mx},${pa.y} ${mx},${pb.y} ${x2},${pb.y}` }, edgesG);
    if (push[key]) {
      p.setAttribute("class", "push");
      p.style.strokeWidth = Math.min(2.5, 0.8 + push[key] * 1.2);
      tip(p, `${spName(v, a.slice(2))} pushes the world over the next ${hs[b.slice(2)]?.label}: ${push[key].toFixed(2)} of a usual move, its strongest factor`);
    } else if (b.startsWith("a:") && acts.find(x => "a:" + x.id === b)?.kind !== "rule") p.setAttribute("class", back ? "tilted you" : "tilted");
    else if (b.startsWith("a:rule:") && open.has(b.slice(7))) p.setAttribute("class", back ? "fired you" : "fired");
    else if (back) p.setAttribute("class", "you");
    else if (a.startsWith("h:") || b.startsWith("h:")) p.setAttribute("class", "feeds");
    else if (b.startsWith("s:") && ["red", "yellow"].includes(v.signposts.find(s => "s:" + s.id === b)?.state)) p.setAttribute("class", "feeds");
    paths.push([a, b, p]);
  }
  const label = (x, y, text, cls, anchor = "start") => {
    const tx = svgEl("text", { x, y: y + 4, class: cls, "text-anchor": anchor }, nodesG);
    tx.textContent = text;
    return tx;
  };
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  const chars = col => Math.floor((X[col][1] - X[col][0]) / (col >= 4 ? 6.1 : 6.6));
  const nodes = {};
  const node = (id, cls, title) => {
    const grp = svgEl("g", { class: "n" + (cls ? " " + cls : ""), tabindex: "0", role: "button", "aria-label": title }, nodesG);
    grp.append(svgEl("title", {}));
    grp.lastChild.textContent = title;
    const [kind, ...rest] = id.split(":");
    grp.onclick = () => select(kind, rest.join(":"));
    nodes[id] = grp;
    return grp;
  };
  for (const d of g.data) {
    const id = "d:" + d.id, p = pos[id];
    if (!p) continue;
    const value = d.kind === "series" ? num(d.value) : "";
    const grp = node(id, d.kind === "source" ? "source" : "", d.kind === "series" ? `${d.name}: ${value} on ${day(d.day)}` : d.name);
    const tx = label(X[0][1], p.y, clip(d.name, chars(0) - value.length - 2), "name" + (d.kind === "source" ? " source" : ""), "end");
    if (value) { const vv = svgEl("tspan", { class: "val", dx: "8" }); vv.textContent = value; tx.append(vv); }
    grp.append(tx);
  }
  for (const s of v.signposts) {
    const id = "s:" + s.id, p = pos[id];
    if (!p) continue;
    const grp = node(id, ["red", "yellow"].includes(s.state) ? "lit" : "", `${s.name}: ${spValue(s)}, ${s.state}${s.since ? " since " + day(s.since) : ""}. ${s.said ?? ""}`);
    grp.append(svgEl("circle", { cx: X[1][0] - 1, cy: p.y, r: 3.5, class: "st " + s.state }));
    grp.append(label(X[1][0] + 9, p.y, clip(s.name, chars(1) - 8), "name"), label(X[1][1], p.y, s.error ? "–" : spValue(s), "val", "end"));
  }
  for (const q of qs) {
    const id = "q:" + q.id, p = pos[id];
    if (!p) continue;
    const s = sides(q), x0 = X[Q_COL][0];
    const grp = node(id, main.includes(q.id) ? "lit" : "", `${q.title}: ${keyLabel(q)} ${pct(s.final)}` + (drives.has(id) ? "" : "; scored when it resolves, no action reads it"));
    grp.append(label(x0, p.y, clip(shortName(q), chars(Q_COL) - 7), "name" + (drives.has(id) ? "" : " scored")), label(X[Q_COL][1], p.y, pct(s.final), "val", "end"));
  }
  for (const [k, d] of Object.entries(hs)) {  // each length: your targets' bad case against its limit, the LLM's move
    const p = pos["h:" + k];
    if (!p) continue;
    const bad = d.stats?.targets?.all?.bad, past = bad != null && d.limit != null && bad < d.limit;
    const top = p.y - PITCH, x0 = X[W_COL][0], span = X[W_COL][1] - X[W_COL][0];
    const grp = node("h:" + k, "lit scen", `The next ${d.label}: your targets' bad case ${signed(bad, 1)}, limit ${signed(d.limit, 1)}; the LLM's scenarios moved ${pct(d.llm_moved ?? 0)} of the world`);
    grp.append(svgEl("rect", { x: x0 - 8, y: top - PITCH / 2, width: span + 14, height: 4.4 * PITCH, rx: 3, class: "scen-box" }));  // where it is picked
    grp.append(label(x0, top, `Next ${d.label}`, "name"), label(X[W_COL][1], top, signed(bad, 1), "val" + (past ? " bad" : ""), "end"));
    // one strip: the bad case as far as it goes toward the limit (past it, red)
    if (bad != null && d.limit) grp.append(svgEl("rect", { x: x0, y: top + 9, width: Math.max(1, span * Math.min(1, bad / d.limit)), height: 3, class: "scen-seg" + (past ? " bad" : "") }));
    [["Limit", signed(d.limit, 1)], ["Without the LLM", signed(d.stats?.targets?.no_llm?.bad, 1)], ["LLM moved", pct(d.llm_moved ?? 0)]].forEach(([n, x], i) => {
      const y = top + (i + 1.15) * PITCH * 0.8 + 10;
      grp.append(label(x0, y, n, "name small dim"), label(X[W_COL][1], y, x, "val small dim", "end"));
    });
  }
  for (const a of acts) {
    const id = "a:" + a.id, p = pos[id];
    if (!p) continue;
    const fired = a.kind === "rule" && open.has(a.rule), on = fired || a.kind !== "rule" || a.status === "holds";
    const tag = a.kind === "rule" ? (fired ? " (fired)" : a.status === "holds" ? " (holds)" : "") : " (suggested)";
    const grp = node(id, (on ? "lit" : "") + (fired ? " fired" : "") + (a.kind === "rule" ? "" : " tilt"), `${a.title}${tag}; from ${a.from.map(f => (f === "world" ? "the world" : "your settings")).join(" and ")}`);
    const short = a.kind === "rule" ? a.title : a.kind === "hedge" ? `Hedge: ${a.title.replace(/^Move /, "move ")}` : a.title;
    grp.append(label(X[4][0], p.y, clip(short, chars(4) - (a.size != null ? 6 : 0)), "name"));
    if (a.size != null && a.kind === "tilt") grp.append(label(X[4][2], p.y, signed(a.size, 1), "val", "end"));
  }
  for (const y of me) {
    const id = "y:" + y.id, p = pos[id];
    if (!p) continue;
    const grp = node(id, "lit you", y.name);
    grp.append(label(X[5][0], p.y, clip(y.name, chars(5)), "name"));
  }
  // on the whole map, what is not in today's chains steps back
  if (!keep) for (const [id, e] of Object.entries(nodes)) e.classList.toggle("quiet", !today.has(id));
  // follow a chain: pointing at a node lights everything upstream and downstream of it
  const reach = (start, next) => { const seen = new Set([start]), todo = [start]; while (todo.length) for (const n of next[todo.pop()] || []) if (!seen.has(n)) { seen.add(n); todo.push(n); } return seen; };
  const focus = id => {
    const on = new Set([...reach(id, out), ...reach(id, inn)]);
    wrap.classList.add("focus");
    for (const [n, e] of Object.entries(nodes)) e.classList.toggle("on", on.has(n));
    for (const [a, b, p] of paths) p.classList.toggle("on", on.has(a) && on.has(b));
  };
  const blur = () => wrap.classList.remove("focus");
  for (const [id, e] of Object.entries(nodes)) {
    e.addEventListener("pointerenter", () => focus(id));
    e.addEventListener("focus", () => focus(id));
    e.addEventListener("pointerleave", blur);
    e.addEventListener("blur", blur);
    e.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); e.onclick?.(); } });
  }
  wrap.mark = sel => { for (const [id, e] of Object.entries(nodes)) e.classList.toggle("sel", id === sel); };
  wrap.mark(page.sel);
  if (!listening) {  // the map is redrawn on each render; the current one is marked
    listening = true;
    document.addEventListener("sc-select", e => $("#map .map")?.mark?.(e.detail));
  }
  const total = all.reduce((a, c) => a + c.length, 0);
  const toggle = button(mapAll ? "Only what is active today" : `Show all ${total}`, "link", () => {
    mapAll = !mapAll;
    try { localStorage.setItem("scenarios-map", mapAll ? "all" : "today"); } catch {}
    $("#map [data-map]").replaceChildren(lineageMap(v, dec));
  });
  const foot = el("p", "foot", (keep ? "Showing what is active today: " : "Showing everything; bright = active today: ") + "red or yellow indicators, every indicator moving the world today, each length of the world, the four key forecasts, today's actions and their inputs. " +
    "A line from an indicator into a length: it pushes the world over it today, as thick as its push (in the factor's usual moves). A forecast hangs off the length nearest the days it has left. The strip under a length: your targets' bad case toward its limit, red past it. White: into an action. ");
  foot.append(toggle);
  wrap.append(foot);
  return wrap;
}
