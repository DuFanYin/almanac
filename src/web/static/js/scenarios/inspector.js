// 3, the inspector beside the map: whatever is chosen on the map (or in the view), in full. A question: how it is
// read off the world, its two sides (the data's world, and with the LLM's scenarios) and the final by outcome, its
// history, the owner's own view; an indicator: today's dial, what it says about the factors and how that measured up,
// what it pushes today; a series, a source; a scenario: its weight against the simulation, against history, for you,
// for and against it now; a length of the world; an action (a lean, a guardrail, a trade, a rule): which side it comes
// from and why; you: your style, targets and rules. Each step's full lineage opens from here.
import { bindPointer, chartSvg, clearHover, hoverMarks, nearest, niceTicks, showTip, svgEl, yGrid } from "../../chart.js";
import { $ } from "../dom.js";
import { decide, post, skip, why } from "./api.js";
import { showLineage } from "./lineage.js";
import { day, num, pct, plain, pts, signed, times, toneClass } from "./format.js";
import { page } from "./page.js";
import { keyBuckets, keyLabel, shortName, sides, spValue } from "./questions.js";
import { strip, stripKey } from "./strip.js";
import { sentence } from "./verdict.js";
import { forAgainst, LENGTHS, rangeText } from "./view.js";
import { button, el, scrolled, table, td } from "./ui.js";

const v = () => page.state.view;
const lineageButton = (label, node) => (node ? button(label, "link", () => showLineage(`api/scenarios/lineage/${node}`)) : null);
const facts = rows => {
  const dl = el("dl", "facts");
  for (const [k, x] of rows.filter(Boolean)) { const dd = el("dd"); dd.append(x); dl.append(el("dt", null, k), dd); }
  return dl;
};
const sub = text => el("h4", "pane-h", text);

// choose a node: the map marks it, the side panel opens on it; ids as on the map (q:spx_2y, s:vix_high, a:hedge:policy)
export function select(kind, id) {
  page.sel = `${kind}:${id}`;
  document.dispatchEvent(new CustomEvent("sc-select", { detail: page.sel }));
  render();
  const host = $("#inspector");
  if (host.hidden) { host.hidden = false; host.scrollTop = 0; }
  $("[data-close]", host)?.focus({ preventScroll: true });
}

export function close() {
  $("#inspector").hidden = true;
  page.sel = null;
  document.dispatchEvent(new CustomEvent("sc-select", { detail: null }));
}

document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("#inspector").hidden && !document.querySelector("dialog[open]")) close(); });

export function render() {
  const host = $("#inspector");
  if (!host || !v() || !page.sel) return;
  const [kind, ...rest] = page.sel.split(":"), id = rest.join(":");
  const make = { q: question, s: signpost, d: datum, p: panel, a: action, y: you, f: future, h: horizonPane, sys: stagePane }[kind];
  const shut = button("Close", "bare small", close);
  shut.setAttribute("data-close", "");
  const top = el("div", "drawer-top");
  top.append(shut);
  host.replaceChildren(top, ...((make && make(id)) || placeholder()));
}

function placeholder() {
  return [el("p", "pane-kind", "Inspector"), el("h3", "pane-title", "Choose anything on the map"),
    el("p", "pane-lede", "Click any forecast, scenario, indicator, data series, action or rule to see its numbers, how they were calculated and where the data came from.")];
}

// ---- a question
function question(qid) {
  const q = v().questions.find(x => x.id === qid);
  if (!q) return null;
  const s = sides(q), out = [];
  const at = (v().graph?.edges || []).find(([a, b]) => b === "q:" + q.id && a.startsWith("h:"))?.[0]?.slice(2), h = at && v().horizons?.[at];
  out.push(el("p", "pane-kind", `Forecast${q.proposed ? ", proposed by the LLM and adopted" : ""}`), el("h3", "pane-title", q.title),
    el("p", "pane-lede", [q.detail, q.closes ? `Resolves ${day(q.closes)}.` : ""].filter(Boolean).join(" ")));
  const head = el("div", "pane-fig");
  head.append(el("b", "num", pct(s.final)), el("span", null, keyLabel(q)), el("small", "num", s.finalIv ? ` ${pct(s.finalIv[0])}–${pct(s.finalIv[1])}` : ""));
  out.push(head, strip(s, { width: 320 }), el("p", "pane-key", stripKey(s)));
  const w = q.weights || {};
  out.push(facts([
    q.sources?.base && ["History", `${pct(sides({ ...q, model: q.sources.base }).final)}: its outcome after every week since 1971, unweighted`],
    ["Simulation", `${pct(s.math)}: the same weeks weighted by today's indicators${w.market ? `, pooled with the market (its weight ${w.market.toFixed(2)})` : ""}`],
    s.llm != null && ["With the LLM's scenarios", `${pct(s.llm)}: the world weighed by the scenarios the LLM wrote${q.scenarios_from ? ` for the next ${v().horizons?.[q.scenarios_from]?.label ?? q.scenarios_from}` : ""}`],
    w.own && ["Your view", `pooled on top, its weight ${w.own.toFixed(2)}`],
    s.finalIv && ["90% band", `${pct(s.finalIv[0])}–${pct(s.finalIv[1])}, from how many separate periods of history its outcome rests on`],
  ]));
  if (h) { const d = el("div", "pane-links"); d.append(button(`Read off the world over ${h.label}: open it`, "link", () => select("h", at))); out.push(d); }
  // the two sides and the final by outcome; the sources behind the computed side, and the owner's own view, folded
  out.push(sub("By outcome"));
  const name = bk => { const c = td(null); c.append(el("i", "key " + toneClass(bk.tone)), bk.label); if (q.resolved?.bucket === bk.id) c.append(el("b", null, " · happened")); return c; };
  out.push(scrolled(table(["", "Simulation", "With the LLM", "Final"], q.buckets.map(bk => {
    const tr = el("tr"), fin = td("main num", pct(q.model?.[bk.id], 1)), iv = q.interval?.[bk.id];
    if (iv) fin.append(el("small", "soft", `${pct(iv[0])}–${pct(iv[1])}`));
    tr.append(name(bk), td("num", pct(q.math?.[bk.id], 1)), td("num", q.sources?.futures ? pct(q.sources.futures[bk.id], 1) : "–"), fin);
    return tr;
  }), "by-outcome")));
  const behind = [["base", "History"], ["market", "Market"]].filter(([k]) => q.sources?.[k]);
  if (behind.length) {
    const d = el("details");
    d.append(el("summary", null, `${behind.map(([, l]) => l).join(", ")} by outcome`), scrolled(table(["", ...behind.map(([, l]) => l)], q.buckets.map(bk => {
      const tr = el("tr");
      tr.append(name(bk), ...behind.map(([k]) => td("num", pct(q.sources[k][bk.id], 1))));
      return tr;
    }))));
    out.push(d);
  }
  if (!q.resolved) {
    const mine = page.state.judgments?.[q.id], d = el("details"), form = el("form", "pane-form");
    d.append(el("summary", null, mine ? `Your view, from ${new Date(mine.at * 1000).toLocaleDateString("en-GB")}` : "Give your own view"));
    form.append(scrolled(table(["", "Yours, %"], q.buckets.map(bk => {
      const input = el("input", null, null, { name: bk.id, type: "number", min: "0", max: "100", step: "1", "aria-label": `Your ${bk.label}` });
      input.value = mine ? Math.round(mine.probs[bk.id] * 100) : "";
      input.placeholder = String(Math.round((q.model?.[bk.id] ?? 0) * 100));
      const tr = el("tr");
      tr.append(name(bk), td(null, input));
      return tr;
    }))));
    const foot = el("div", "pane-row"), note = el("input", null, null, { name: "note", maxlength: "300", autocomplete: "off", placeholder: "Why (kept with your numbers)", "aria-label": "Why" });
    note.value = mine?.note ?? "";
    foot.append(note, el("button", "btn", "Save", { type: "submit" }));
    form.append(foot, el("p", "pane-key", `Pooled on top of the final, its weight ${(q.weights?.own ?? 0.2).toFixed(2)}; blank outcomes take the final's figure.`));
    form.onsubmit = async e => {
      e.preventDefault();
      const probs = Object.fromEntries(q.buckets.map(bk => [bk.id, form[bk.id].value === "" ? +form[bk.id].placeholder : +form[bk.id].value]));
      try { await post("api/scenarios/judgment", { question: q.id, probs, note: form.note.value }); } catch (err) { alert("Not saved: " + err.message); }
    };
    d.append(form);
    out.push(d);
  }
  if ((q.history || []).length > 1) { out.push(sub(`${keyLabel(q)}, day by day`)); out.push(historyChart(q)); }
  out.push(el("div", "pane-links", null));
  out.at(-1).append(...[lineageButton("Sources of the final estimate", q.node), lineageButton("of the simulation", q.math_node)].filter(Boolean));
  return out;
}

const list = items => { const ul = el("ul", "pane-list"); ul.append(...items.map(x => el("li", null, x))); return ul; };

// the key outcome's final, the simulation and with the LLM's scenarios by day: three lines, the final in ink (before the
// world, the LLM's side was its panel's)
function historyChart(q) {
  const host = el("div", "ui-chart pane-chart"), keys = keyBuckets(q), rows = q.history.filter(r => r.model);
  const lk = rows.some(r => r.futures) ? "futures" : "llm", lname = "With the LLM";
  const tot = p => (p ? keys.reduce((a, k) => a + (p[k] ?? 0), 0) : null);
  requestAnimationFrame(() => {
    const H = 150, T = 8, B = H - 20, L = 34;
    const [svg, W] = chartSvg(host, H, { "aria-label": `${keyLabel(q)} by day: final, simulation, with the LLM` });
    const R = W - 4, n = rows.length, xs = rows.map((_, j) => L + (R - L) * (n > 1 ? j / (n - 1) : 0.5));
    const vals = rows.flatMap(r => [tot(r.model), tot(r.math), tot(r[lk])]).filter(x => x != null);
    const ticks = niceTicks(0, Math.max(0.1, ...vals), 4), Y = x => B - (B - T) * (x / ticks.at(-1));
    yGrid(svg, ticks, Y, L, R, x => pct(x));
    const series = [[lk, "llm"], ["math", "math"], ["model", "final"]];
    for (const [k, cls] of series) {
      const pts_ = rows.map((r, j) => [xs[j], tot(r[k])]).filter(([, y]) => y != null);
      if (pts_.length > 1) svgEl("polyline", { points: pts_.map(([x, y]) => `${x},${Y(y)}`).join(" "), class: "h-" + cls }, svg);
    }
    const first = svgEl("text", { x: L, y: H - 4, class: "tick" }, svg), last = svgEl("text", { x: R, y: H - 4, class: "tick", "text-anchor": "end" }, svg);
    first.textContent = day(rows[0].day); last.textContent = day(rows.at(-1).day);
    bindPointer(svg, (x, _y, cx, cy) => {
      const j = nearest(xs, x);
      clearHover(svg);
      hoverMarks(svg, xs[j], T, B);
      showTip(svg, cx, cy, rows[j].day, [["Final", rows[j].model], ["Simulation", rows[j].math], [lname, rows[j][lk]]]
        .filter(([, p]) => p).map(([name, p]) => ({ name, value: pct(tot(p), 1) })));
    }, () => clearHover(svg));
  });
  const wrap = el("div");
  wrap.append(host, el("p", "pane-key", `White: final estimate. Grey: the simulation. Dashed: ${lname.toLowerCase()}.`));
  return wrap;
}

// ---- an indicator (a signpost)
const VERDICT = { agrees: "measured so", flat: "within the noise: nothing", against: "measured the other way: not used", unmeasured: "too little history to measure", assumed: "the LLM's, at half weight, sized as measured ones are, until it can be measured" };
const WAY = { up: "higher", down: "lower", wider: "swings wider", narrower: "swings narrower" };
function signpost(sid) {
  const sp = v().signposts.find(x => x.id === sid);
  if (!sp) return null;
  const facs = v().world?.factors || {}, lab = f => (facs[f]?.label ?? f).replace(/\s*\(.*\)$/, "");
  const span = ([a, b]) => (b <= 31 ? `days ${a}–${b}` : b <= 400 ? `months ${Math.round(a / 30.4)}–${Math.round(b / 30.4)}` : `years ${(a / 365).toFixed(1)}–${(b / 365).toFixed(1)}`);
  const rows = (sp.expect || []).map(e => {
    const tr = el("tr"), name = el("td", "wrapcell");
    name.append(el("span", null, `${lab(e.factor)} ${WAY[e.way] ?? e.way}, ${span(e.days)}`), el("small", "soft", e.why));
    tr.append(name, td(e.verdict === "agrees" ? "num" : "soft", e.verdict === "agrees" || e.verdict === "assumed" ? (e.size ?? 0).toFixed(2) : "–"), td(e.verdict === "against" ? "warn" : "soft", VERDICT[e.verdict] ?? "not yet measured"));
    return tr;
  });
  const pushed = [];
  for (const [kind, by] of Object.entries(sp.push || {}))
    for (const [f, xs] of Object.entries(by)) {
      const tot = xs.reduce((a, x) => a + x, 0);
      if (Math.abs(tot) >= 0.005) pushed.push(`${lab(f)}: ${kind === "spread" ? (tot > 0 ? "wider" : "narrower") : tot > 0 ? "higher" : "lower"} (${tot > 0 ? "+" : ""}${tot.toFixed(2)})`);
    }
  const dial = sp.dial == null ? "no reading today" : `${pct(sp.dial + 0.5)} of the way toward red across its own history (50%: its middle)`;
  const links = el("div", "pane-links");
  if (sp.node) links.append(lineageButton("Sources of today's value", sp.node));
  return [el("p", "pane-kind", `Indicator · ${sp.driver}${sp.proposed ? " · proposed by the LLM" : ""}`), el("h3", "pane-title", sp.name),
    facts([["Today", `${spValue(sp)}, ${sp.state}${sp.since ? ` since ${day(sp.since)}` : ""}`], ["Where it reads", dial], ["What it is", sp.said ?? sp.expr],
      sp.as_of && ["Data as of", day(sp.as_of)], sp.error && ["Error", sp.error]]),
    sub("What it says about the factors"),
    rows.length ? scrolled(table(["", "How strong", "In history"], rows)) : el("p", "pane-note", "Nothing: a rule reads it."),
    el("p", "pane-note", "Each a mechanism written beside it, checked against history: how much more (or more widely) the factor moved than usual over those days, the more the indicator read toward red, shrunk by its noise. Only what measured the way expected moves the world (the weights on history's periods)."),
    sub("What it moves today"), pushed.length ? list(pushed) : el("p", "pane-note", "Nothing: near the middle of its history, or what it says did not measure up."),
    sp.share && sp.share < 1 ? el("p", "pane-note", "Proposed by the LLM, which has read the history it is measured on: it pushes half as much.") : null,
    links].filter(Boolean);
}

// ---- a series or another source
function datum(id) {
  const d = v().graph.data.find(x => x.id === id);
  if (!d) return null;
  if (d.kind === "series") {
    const links = el("div", "pane-links");
    if (d.node) links.append(lineageButton("Sources: where it was fetched", d.node));
    return [el("p", "pane-kind", "Data"), el("h3", "pane-title", d.name), facts([["Last", `${num(d.value)} on ${day(d.day)}`], ["Series code", id]]), links];
  }
  const qs = v().questions.filter(q => !q.resolved);
  const say = {
    kalshi: ["Kalshi's prices, a market in the outcome itself, pooled with the world's figure where both exist.", qs.filter(q => q.sources?.market).map(q => `${shortName(q)}: the market's weight ${(q.weights?.market ?? 0).toFixed(2)}, the world's ${(q.weights?.world ?? 0).toFixed(2)}`)],
    own: ["Your own probabilities, entered on a forecast; mixed into the final estimate.", qs.filter(q => q.sources?.own).map(q => shortName(q))],
  }[id] || ["", []];
  return [el("p", "pane-kind", "Source"), el("h3", "pane-title", d.name), el("p", "pane-lede", say[0]), say[1].length ? list(say[1]) : el("p", "pane-note", "Nothing from it today.")];
}

// ---- the LLM in the world: its scenarios at each length against the simulation
function panel() {
  const hs = v().horizons || {}, all = v().scenarios || [];
  const out = [el("p", "pane-kind", "LLM"), el("h3", "pane-title", "The LLM in the world"),
    el("p", "pane-lede", "The world's last layer. At a month, three months, a year and three years one LLM writes the scenarios worth telling apart now, each in a direction of its own (one or two factors' ranges: rates, inflation, jobs, the dollar, credit, stocks); a period of history may be in several or in none. Four readers (a macroeconomist, a market historian, an optimist, a pessimist) each read one scenario on its own and say how much likelier or less likely it is than the simulation's share, for what the simulation does not include; a reviewer with the board's data checks their reasons, giving no numbers; the readers it found fault with read again. Each scenario should have the simulation's share times the median factor: history's periods are reweighed until each has it, the simulation moved least. Each scenario is scored when its length is over: did it happen. A coach reads the scores for every role.")];
  const anyset = Object.values(v().scenario_sets || {})[0];
  if (anyset) out.push(facts([["Models", [...new Set((anyset.readers || []).map(r => r.model))].join(", ") || "–"], ["The coach", anyset.coach || "no note yet: no set has reached its end"]]));
  for (const k of LENGTHS) {
    const mine = all.filter(f => f.horizon === k).sort((a, b) => b.p / (b.p_data || 1) - a.p / (a.p_data || 1));
    if (!mine.length) continue;
    out.push(sub(`Next ${hs[k]?.label ?? k}: ${pct(hs[k]?.llm_moved ?? 0)} of the weight moved`), scrolled(table(["", "Simulation", "Readers' factor", "Probability"], mine.map(f => {
      const tr = el("tr", "pick", null, { tabindex: "0" });
      tr.append(td(null, f.name), td("num", pct(f.p_data, 1)), td("soft num", times(f.times)), td("main num", pct(f.p, 1)));
      tr.onclick = () => select("f", f.id);
      return tr;
    }))));
  }
  return out;
}

// ---- a scenario: its story, its ranges, its probability (the simulation's share times the readers' factor), what
// each reader said and why, what it means for your targets
function future(fid) {
  const f = (v().scenarios || []).find(x => x.id === fid);
  if (!f) return null;
  const h = v().horizons?.[f.horizon], facs = v().world?.factors || h?.factors || {}, usual = h?.stats?.targets?.history?.expected;
  const was = f.was != null ? `; a week before ${pct(f.was, 1)}` : "";
  const side = f.impact?.side ? `${f.impact.side} than usual: their mean return in it ${signed(f.impact.mean, 1)}, in history at this length ${signed(usual, 1)}` : "–";
  const out = [el("p", "pane-kind", `Scenario · next ${h?.label ?? f.horizon} · written by the LLM ${day(f.written)}`), el("h3", "pane-title", f.name),
    el("p", "pane-lede", plain(f.story)),
    facts([["Its probability", `${pct(f.p, 1)}: the simulation's ${pct(f.p_data, 1)} times the readers' factor ${times(f.times).slice(1)}${f.p_target != null && Math.abs(f.p - f.p_target) >= 0.005 ? `, ${pct(f.p_target, 1)}, as near as the other scenarios allow` : ""}${was}`],
      ["The simulation", `${pct(f.p_data, 1)} of history's periods of ${h?.label ?? f.days + " days"} are in it, weighted by today's indicators (${pct(f.p_history, 1)} unweighted)${f.made ? `. History has too few periods like it (${f.episodes} that do not overlap): the ${f.made} nearest are moved into its ranges, the other factors with them as they moved together` : ""}`],
      ["It says", Object.entries(f.ranges || {}).map(([k, r]) => `${facs[k]?.label ?? k} ${rangeText(k, r)}`).join("; ") + ` over the next ${h?.label ?? f.days + " days"}`],
      ["For your targets", side],
      ["Worst 10% of your targets", h?.stats?.targets?.worst_parts?.[fid] != null ? `${pct(h.stats.targets.worst_parts[fid])} of it is in this scenario` : "none of it"]])];
  const rs = (f.readings || []).filter(r => r.final);
  if (rs.length) {
    out.push(sub("What each reader said"), list(rs.map(r => {
      const again = r.first && r.first.times !== r.final.times ? ` (first ${times(r.first.times)}, before the reviewer)` : "";
      return `${r.view.replace(/ who .*$/, "")}: ${times(r.final.times)}${again}. Missing from the simulation: ${(r.final.missing || "–").replace(/\.$/, "")}. ${r.final.why}`;
    })));
  }
  if (f.history_like_by_factor) {
    const rows = Object.entries(f.history_like_by_factor).sort((x, y) => x[1] - y[1]).map(([k, s]) => `${facs[k]?.label ?? k} ${rangeText(k, f.ranges[k])}: ${pct(s)} of history's periods`);
    out.push(sub("Against history"), list(rows));
  }
  if (f.impact) {
    const acts = (v().actions || []).filter(a => (f.drives || []).includes(a.id));
    out.push(sub("What it means for your targets"), facts([["Their return in it", signed(f.impact.mean, 1)], ["Their worst 10%", `${pct(f.impact.worst)} of it is in this scenario`], ["Its share of the futures", pct(f.impact.share)]]));
    if (acts.length) { const d = el("div", "pane-links"); d.append(...acts.map(a => button(a.title, "link", () => select("a", a.id)))); out.push(el("p", "pane-note", "It drives these suggestions:"), d); }
  }
  if (f.readings_now) {
    const fa = forAgainst(f);
    out.push(sub("For and against it now"), facts([["For", fa.yes.length ? fa.yes.join("; ") : "nothing yet"], ["Against", fa.no.length ? fa.no.join("; ") : "nothing yet"]]));
  }
  if (f.opposite) {
    const o = f.opposite, d = el("div", "pane-links");
    d.append(button(`${o.name} (${pct(o.p)})`, "link", () => select("f", o.id)));
    out.push(sub("The other view"), d);
  }
  const said = (v().scenario_sets?.[f.horizon]?.review?.points || []).filter(p => p.scenario === f.sid);
  if (said.length) out.push(sub("The reviewer said"), list(said.map(p => `${p.says}. ${p.why}`)));
  out.push(sub("Its ranges over the length"), list(Object.entries(f.ranges).map(([k, r]) => `${facs[k]?.label ?? k}: ${rangeText(k, r)}`)));
  out.push(el("div", "pane-links"));
  if (f.node) out.at(-1).append(lineageButton("Sources", f.node));
  return out;
}

// ---- a length of the world: how it is weighted, what the LLM's scenarios move, what it means for your targets
function horizonPane(k) {
  const d = v().horizons?.[k];
  if (!d) return null;
  const st = d.stats?.targets, a = st?.all, n = st?.no_llm, e = v().world?.ess?.[k];
  const live = (v().scenarios || []).filter(f => f.horizon === k).sort((x, y) => y.p - x.p), sim = st?.data || {};
  const from = d.from && d.from !== k ? `; the LLM's scenarios of the next ${v().horizons?.[d.from]?.label ?? d.from} carried over by start day` : "";
  const out = [el("p", "pane-kind", "The world"), el("h3", "pane-title", `The next ${d.label}`),
    el("p", "pane-lede", `Every period of ${d.label} in history since 1971, weighted by today's indicators${e ? ` (as much evidence as ${Math.round(e.data)} of its ${e.of} weeks)` : ""}: the simulation. The LLM's scenarios then weigh its parts${from}: they move ${pct(d.llm_moved ?? 0)} of the weight.`)];
  const sd = v().decision?.sides?.[k], gaps = v().decision?.gaps?.[k] || {}, labels = v().class_labels || {};
  if (a) out.push(facts([sd && ["Against usual", { better: "better than usual", worse: "worse than usual", wider: "wider both ways", narrower: "narrower both ways", usual: "as usual" }[sd.word]],
    ["Good case", `${signed(a.good, 1)} (usual ${signed(st.history?.good, 1)}; the simulation ${signed(sim.good, 1)}; without the LLM ${signed(n.good, 1)})`],
    ["Expected", `${signed(a.expected, 1)} (usual ${signed(st.history?.expected, 1)}; the simulation ${signed(sim.expected, 1)}; without the LLM ${signed(n.expected, 1)})`],
    ["Bad case", `${signed(a.bad, 1)} (usual ${signed(st.history?.bad, 1)}; the simulation ${signed(sim.bad, 1)}; without the LLM ${signed(n.bad, 1)})`],
    ["Your limit", `${signed(d.limit, 1)}: a year's from your risk level, this length's as far past what is usual for your targets${d.days < 30 ? "; shown, not acted on" : ""}`],
    ["Chance of a 10% fall", `${pct(a.p_below["-0.1"])} (without the LLM ${pct(n.p_below["-0.1"])})`]]));
  const apart = Object.entries(gaps).filter(([, g]) => Math.abs(g.gap) >= 0.001).sort((x, y) => Math.abs(y[1].gap) - Math.abs(x[1].gap));
  if (apart.length) out.push(sub("Each class against usual (the decision step reads these)"), scrolled(table(["", "Today − usual", "Chance alone", "From the data / the LLM"], apart.map(([c, g]) => {
    const tr = el("tr");
    tr.append(td(null, labels[c] ?? c), td("num " + (Math.abs(g.gap) > g.se ? "main" : "soft"), signed(g.gap, 1)), td("soft num", `±${pct(g.se, 1)}`), td("soft num", `${signed(g.data, 1)} / ${signed(g.llm, 1)}`));
    return tr;
  }))));
  const facs = v().world?.factors || {}, moved = Object.entries(facs).filter(([, x]) => Math.abs(x.mu?.[k] ?? 0) >= 0.02 || Math.abs((x.var?.[k] ?? 1) - 1) >= 0.02);
  if (moved.length) out.push(sub("Where today's indicators push the factors"), list(moved.map(([f, x]) => `${x.label.replace(/\s*\(.*\)$/, "")}: ${(x.mu[k] ?? 0) >= 0 ? "+" : ""}${(x.mu[k] ?? 0).toFixed(2)} of a usual move${Math.abs((x.var?.[k] ?? 1) - 1) >= 0.02 ? `, swings ${x.var[k].toFixed(2)} times as wide` : ""}`)));
  if (live.length) out.push(sub("The LLM's scenarios"), scrolled(table(["", "Simulation", "Probability"], live.map(f => { const tr = el("tr", "pick"); tr.append(td(null, f.name), td("soft num", pct(f.p_data, 1)), td("main num", pct(f.p, 1))); tr.onclick = () => select("f", f.id); return tr; }))));
  out.push(el("div", "pane-links"));
  if (d.node) out.at(-1).append(lineageButton("Sources", d.node));
  return out;
}

// ---- an action: a rule (its rule trigger), a lean, a guardrail, a trade, a single stock
function action(aid) {
  const a = [...(v().actions || []), ...(page.state.account?.actions || [])].find(x => x.id === aid);
  if (!a) return null;
  const from = el("p", "pane-note", `From ${a.from.map(f => (f === "world" ? "the scenarios" : "your settings (risk level, targets, rules)")).join(" and ")}.`);
  if (a.kind === "rule") {
    // a firing still open: what it asks in one instruction, today's reading behind it, when it stops asking
    const dec = page.state.decisions || {}, t = v().triggers.find(x => x.rule === a.rule && (!dec[x.id] || dec[x.id].status === "open"));
    const x = t ? sentence(t, page.state) : null, said = [];
    if (x) {
      const today = el("p", "pane-lede"), lift = el("p", "pane-note");
      today.innerHTML = x.today || "";
      lift.innerHTML = x.lifts || "";
      said.push(el("p", "pane-kind", "To do"), el("h3", "pane-title", x.head + "."), today, lift);
    }
    return [...said, ...(rule(a.rule) || []).slice(x ? 2 : 0), from];
  }
  const what = { lean: "your targets moved by today's world", guard: "a guardrail", trade: "what you hold", new_money: "new money", concentration: "a single stock held in size" }[a.kind] ?? a.kind;
  const out = [el("p", "pane-kind", `Action · ${what} · ${a.status === "note" ? "note" : a.status}`), el("h3", "pane-title", a.title), el("p", "pane-lede", a.detail ?? ""), from];
  const named = Object.fromEntries((v().scenarios || []).map(f => [f.id, f.name]));
  if (a.kind === "lean") {
    const llm = Object.entries(a.llm || {});
    if (llm.length) { const d = el("div", "pane-links"); d.append(...llm.map(([id, x]) => button(`${named[id] ?? id} ${signed(x, 1)}`, "link", () => select("f", id)))); out.push(sub("The LLM's scenarios behind it"), d); }
    const dv = v().decision;
    out.push(el("p", "pane-note", `The decision step moves a class toward what today's world says of it against usual, both ways alike: at a month, three months, a year and three years, its expected return in today's world less in history's own periods, less the gap a reweighting that large makes by chance, measured in the class's usual swings; the lengths together. The data's part then counts ×${dv?.sense ?? 0} (from ${dv?.walk ?? "the replay through history"}), the LLM's ×${dv?.llm_sense ?? 1} (yours). At most ±${pct(v().you?.limits?.tilt ?? 0)} a class; cash takes the difference; the guardrails win over it.`));
  } else if (a.kind === "guard") {
    out.push(facts([a.bad != null && ["Bad case", `${signed(a.bad, 1)} against your limit ${signed(a.limit, 1)}`], a.after != null && ["After it", signed(a.after, 1)], a.all != null && ["In all", `${pct(a.all, 1)} of everything moved this way would bring it within`]]),
      el("p", "pane-note", "Your limits hold: while a length's bad case is past its limit, the class whose move lifts it most goes into the shelter in your targets (cash, Treasuries or gold) that does it, a point at a time; a step at a time on what you hold."));
  } else if (a.kind === "trade") {
    out.push(facts([["Decided share", pct(a.decided, 1)], ["Held", pct(a.held, 1)], ["Why", a.why === "band" ? "past your rebalancing band" : "the decided share moved half a step"], a.status === "standing" && ["Standing since", day(a.since)]]),
      el("p", "pane-note", "What you hold against the decided mix, category by category: a trade when you are past your band, or when the decided share has moved half a step from where you last were at it; one step at a time; once a step is done, the next one the same way waits four weeks."));
  }
  out.push(el("div", "pane-links"));
  if (a.node) out.at(-1).append(lineageButton("Sources", a.node));
  return out;
}

// ---- you: your style, your targets, your rules, your holdings
function you(part) {
  const y = v().you;
  if (!y) return null;
  const labels = (v().class_labels || {});
  const out = [el("p", "pane-kind", "Your settings"), el("h3", "pane-title", { style: `Risk level ${y.profile} (${y.label})`, policy: y.own ? "Your targets" : `${y.label} base allocation`, rules: "Your rules", holdings: "Your holdings" }[part] ?? "You")];
  if (part === "style") out.push(facts([["Bad case allowed", `−${pct(y.limits.worst_loss)}`], ["Cash at least", pct(y.limits.cash_min)], ["One class at most", `${pct(y.limits.class_cap)} (not counting cash and Treasuries)`],
    ["Today's world moves a class at most", `±${pct(y.limits.tilt)}`], ["Rebalancing band", `a trade once a class is ±${pct(y.limits.band, 1)} off its target`], ["A step", pct(y.limits.step)]]),
    el("p", "pane-note", "Preferences, not beliefs: they change what the system suggests doing, never the probabilities."));
  if (part === "policy") out.push(scrolled(table(["", "Share"], Object.entries(y.policy).sort((a, b) => b[1] - a[1]).map(([k, w]) => { const tr = el("tr"); tr.append(td(null, labels[k] ?? k), td("num", pct(w))); return tr; }))),
    el("p", "pane-note", y.own ? "Your category targets, grouped by asset class (local/rules.toml): the mix in a usual world. The decision step moves it only within your risk level's limit." : "You have no targets set, so your risk level's base allocation is used."));
  if (part === "rules") {
    const ul = el("ul", "pane-list");
    for (const a of (v().actions || []).filter(x => x.kind === "rule")) {
      const li = el("li"), b = button(a.title, "link", () => select("a", a.id));
      li.append(b, el("small", "soft", ` ${a.status}`));
      ul.append(li);
    }
    out.push(el("p", "pane-lede", "Rules you wrote in advance, for what the decision step does not do: Bitcoin's regular purchase, its cap, its pause, its review. When a rule's conditions hold, it triggers and asks for an action. A rule reads values, indicators, dates and shares, never a forecast."), ul);
  }
  if (part === "holdings") out.push(el("p", "pane-lede", "What you hold, read from your positions file. Used by the decision step's trades and guardrails and by the rules that read a share. The numbers are on the Holdings tab."));
  return out;
}

// ---- a rule
function rule(rid) {
  const r = v().rules.find(x => x.id === rid);
  if (!r) return null;
  const dec = page.state.decisions || {}, t = v().triggers.find(x => x.rule === rid && (!dec[x.id] || dec[x.id].status === "open"));
  const flat = xs => (xs || []).flatMap(x => (x.group ? x.group.flatMap(flat) : [x]));
  const rows = flat(r.readings).map(x => {
    const val = x.value == null ? "–" : typeof x.value === "number" ? num(x.value) : String(x.value);
    const state = x.holds === true ? "holds" : x.holds === false ? "no" : "cannot say";
    const tr = el("tr");
    tr.append(td(null, x.said || x.expr || x.name), td("num", val), td("soft num", x.line_said ?? (x.than == null ? "" : `${x.op ?? ""} ${typeof x.than === "number" ? num(x.than) : x.than}`)), td(x.holds ? "main" : "soft", state));
    return tr;
  });
  const out = [el("p", "pane-kind", t ? "Rule · fired" : r.holds ? "Rule · holds" : "Rule"), el("h3", "pane-title", r.title), el("p", "pane-lede", r.explain ?? ""),
    scrolled(table(["Condition", "Now", "Line", ""], rows))];
  if (t) {
    const acts = el("div", "pane-links");
    acts.append(button("Mark done", "primary", () => decide(t, "done")), button("Skip", "", () => skip(t)), button("Trace it", "bare", () => why(t)));
    out.push(acts);
  }
  return out;
}

// ---- a stage of the system (the System tab): what it does, how often, how it went today, what its LLM work must pass,
// its code, and what feeds it and what it feeds
const SIDE_WORDS = { data: "computed from data", llm: "the LLM writes it", you: "yours", out: "what you read" };
function stagePane(id) {
  const sys = v().system, s = sys?.stages.find(x => x.id === id);
  if (!s) return null;
  const name = Object.fromEntries(sys.stages.map(x => [x.id, x.name])), phase = sys.phases.find(p => p.id === s.phase)?.label;
  const order = Object.fromEntries(sys.phases.map((p, i) => [p.id, i])), at = Object.fromEntries(sys.stages.map(x => [x.id, order[x.phase]]));
  const link = (x, later) => button(name[x] + (later ? " (the next day)" : ""), "link", () => select("sys", x));
  const from = sys.edges.filter(([, b]) => b === id).map(([a]) => link(a, at[a] > at[id]));
  const to = sys.edges.filter(([a]) => a === id).map(([, b]) => link(b, at[b] < at[id]));
  const today = { ran: "ran", failed: "failed", off: "not asked in this run (the LLM was off)" }[s.state] ?? s.state;
  const out = [el("p", "pane-kind", `Stage · ${phase} · ${SIDE_WORDS[s.side] ?? s.side}`), el("h3", "pane-title", s.name), el("p", "pane-lede", s.what),
    facts([["How often", s.when], ["Today", [today, s.note].filter(Boolean).join(": ")], s.error ? ["Error", s.error] : null, s.checks ? ["Its check", s.checks] : null])];
  const list = xs => { const d = el("div", "pane-links"); d.append(...xs); return d; };
  if (from.length) out.push(sub("Reads from"), list(from));
  if (to.length) out.push(sub("Feeds"), list(to));
  out.push(sub("Code (src/board/)"), el("p", "pane-code", s.modules.join(" · ")));
  return out;
}
