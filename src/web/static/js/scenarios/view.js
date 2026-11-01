// The scenarios page and the forecasts page. Scenarios: the LLM's scenarios of one length at a time (a tab each), the
// world's last layer: those better than usual for your targets first, each with its probability (the simulation's
// share times the readers' factor) against the simulation's share and history's, and what it means for your targets; then each asset class's figures at that
// length. Forecasts: the factors at every length, then every other open forecast's two sides (the data's world, and
// with the LLM's scenarios) in one table, the LLM's largest moves first. Anything chosen opens in the side panel.
import { $ } from "../dom.js";
import { day, pct, plain, pts, signed, times, tone, toneClass } from "./format.js";
import { select } from "./inspector.js";
import { badWhenUp, bar, keyLabel, keyQuestions, shortName, sides, weekAgo } from "./questions.js";
import { strip, stripKey } from "./strip.js";
import { el, scrolled, table, td } from "./ui.js";

const open = q => select("q", q.id);

function figure(q) {
  const d = el("button", "driver", null, { type: "button" }), s = sides(q), H = s.final;
  const by = q.template === "fed" ? `on ${day(q.closes)}` : q.headline?.by ?? (q.closes ? `by ${day(q.closes)}` : "");
  d.append(el("span", "label", q.headline ? `${q.headline.label} ${by}` : `${shortName(q)}: ${keyLabel(q)} ${by}`));
  const fig = el("span", "fig");
  fig.append(el("b", "num", pct(H)));
  if (s.finalIv) fig.append(el("small", null, `${pct(s.finalIv[0])}–${pct(s.finalIv[1])}`));
  d.append(fig, strip(s, { width: 240 }), el("span", "strip-key", stripKey(s)));
  const w = weekAgo(q), foot = el("span", "delta");
  if (w) {
    const dv = H - sides({ ...q, model: w.model }).final, worse = (dv > 0) === badWhenUp(q);
    foot.append(el("span", "num " + (Math.abs(dv) < 0.01 ? "" : worse ? "bad" : "good"), `${pts(dv)} in a week`));
  }
  if (s.apart) foot.append(el("span", "warn", "the two estimates disagree"));
  d.append(foot);
  d.onclick = () => open(q);
  return d;
}

// ---- a scenario as a view: against history, for you, for and against it now (the page's own words for the fields
// steps/scenarios/page.py explained adds). A range may be open at one end (null).
const endText = (k, x) => (/(stocks|infl|dollar)/.test(k) ? signed(x, 1) : `${x > 0 ? "+" : ""}${x.toFixed(2)} pts`);
export const rangeText = (k, [a, b]) => (a == null ? `below ${endText(k, b)}` : b == null ? `above ${endText(k, a)}` : `${endText(k, a)} to ${endText(k, b)}`);

export function rarestText(f, factors) {
  const r = f.rarest;
  if (!r) return null;
  return `${factors?.[r.factor]?.label ?? r.factor} ${rangeText(r.factor, r.range)}: ${pct(r.share)} of history's periods`;
}

// what speaks for and against it today: the texts read into numbers
export function forAgainst(f, readingName = x => x.name.replace(/\s*\(LLM[^)]*\)$/, "")) {
  const yes = [], no = [];
  for (const r of f.readings_now || []) (r.says === "for" ? yes : no).push(`${readingName(r)} ${r.value > 0 ? "+" : ""}${r.value.toFixed(1)}`);
  return { yes, no };
}

// the scenarios page: one length at a time; a sentence read off its paths; its scenarios a row each, the worst for
// your targets first (what it is and says, how likely: the simulation's share times the readers' factor, what it
// means for your targets, how the four readers read it); then each asset class in each scenario, a row a scenario
export const LENGTHS = ["month", "short", "long", "longer"];

// one length at a time, the one chosen remembered
let horizon = (() => { try { return localStorage.getItem("scenarios-horizon"); } catch { return null; } })();

const reader = r => r.view.replace(/^an? /, "").replace(/ who .*$/, "");
const SIDE = { better: "good", worse: "bad" };
// the asset classes' names in a column head (the full name on hover)
const SHORT = { us_large: "S&P 500", us_equal: "Equal wt", us_tech: "Nasdaq-100", tech_sector: "Tech", semis: "Semis", intl: "Intl", treasuries: "10y Tsy", cash: "Cash", gold: "Gold", bitcoin: "Bitcoin" };

function futuresPage(v, sec) {
  const hs = v.horizons || {}, all = v.scenarios || [], ks = LENGTHS.filter(k => hs[k]);
  const box = $("[data-futures]", sec);
  if (!ks.length) { box.replaceChildren(el("p", "muted", "Appears after the next daily run.")); return; }
  if (!ks.includes(horizon)) horizon = ks[0];
  const d = hs[horizon], mean = f => d.scenarios?.[f.id]?.targets?.mean ?? 0;
  const mine = all.filter(f => f.horizon === horizon).sort((a, b) => mean(a) - mean(b));
  $("[data-htabs]", sec).replaceChildren(...ks.map(k => {
    const b = el("button", null, `Next ${hs[k].label}`, { type: "button", role: "tab", "aria-selected": String(k === horizon) });
    b.onclick = () => { horizon = k; try { localStorage.setItem("scenarios-horizon", k); } catch {} futuresPage(v, sec); };
    return b;
  }));
  // the length in a sentence, read off the paths (the scenarios overlap and some paths are in none, so their
  // probabilities are not added up): how likely your targets end below usual, against the simulation; how much of the
  // world is in none of the scenarios
  const set = v.scenario_sets?.[horizon], rv = set?.review || {}, worse = d.sides?.worse, none = d.sides?.none;
  const usual = d.stats?.targets?.history?.expected;
  $("[data-hsub]", sec).replaceChildren(...(mine.length ? [
    `Your targets end the next ${d.label} below their usual ${signed(usual, 0)} with a chance of `,
    el("b", worse && worse.p > worse.data + 0.005 ? "bad" : worse && worse.p < worse.data - 0.005 ? "good" : "", pct(worse?.p)),
    ` (the simulation ${pct(worse?.data)}). The LLM wrote ${mine.length} scenarios${mine[0].written ? ` on ${day(mine[0].written)}` : ""}, each a direction of its own: a period of history can be in several, and ${pct(none?.p)} of the futures are in none of them (the simulation ${pct(none?.data)}). The readers' factors shifted ${pct(d.llm_moved ?? 0)} of the futures' weight.`,
  ] : [`No scenarios yet for the next ${d.label}: the world is the simulation alone.`]));
  if (!mine.length) { box.replaceChildren(); return; }

  // the scenarios, a row each, the worst for your targets first: what it is and says, how likely, what it means for
  // your targets, how the four readers read it. As many rows as there are scenarios, so nothing scrolls sideways
  const facs = v.world?.factors || d.factors || {}, readers = set?.readers || [];
  const ret = x => (x == null ? "–" : signed(x, Math.abs(x) < 0.095 ? 1 : 0));  // a small return with a decimal
  const side = f => d.scenarios?.[f.id]?.targets?.side ?? f.impact?.side;
  const top = Math.max(0.4, ...mine.map(f => Math.max(f.p, f.p_data)));
  const fig = (value, cls, ...sub) => { const c = el("div", "scn-fig"); const s = el("small", "num"); s.append(...sub); c.append(el("b", "num " + cls, value), s); return c; };
  const list = el("div", "scn-list");
  list.append(...["Scenario: what it says", "Probability", "Your targets", `The ${readers.length || 4} readers: times as likely`].map((x, i) => el("div", "scn-h" + (i ? " fig" : ""), x)));
  for (const f of mine) {
    const who = el("button", "scn-what " + (side(f) ?? ""), null, { type: "button" });
    who.append(el("span", "scn-name", f.name), el("small", "scn-says", Object.entries(f.ranges || {}).map(([k, r]) => `${facs[k]?.label ?? k} ${rangeText(k, r)}`).join("; ")), el("span", "scn-story", plain(f.story)));
    who.onclick = () => select("f", f.id);
    const meter = el("span", "meter " + (SIDE[side(f)] ?? ""));
    meter.append(el("i", null, null, { style: `width:${(f.p / top) * 100}%` }), el("u", null, null, { style: `left:${(f.p_data / top) * 100}%` }));
    const prob = fig(pct(f.p), "", `simulation ${pct(f.p_data)}`, f.was != null && Math.abs(f.p - f.was) >= 0.005 ? ` · was ${pct(f.was)}` : "");
    prob.insertBefore(meter, prob.lastChild);
    const you = fig(ret(f.impact?.mean), SIDE[side(f)] ?? "", el("span", (f.impact?.worst ?? 0) >= 0.3 ? "bad" : "", `usual ${ret(usual)} · ${pct(f.impact?.worst)} of your worst 10%`));
    const rs = readers.map(r => (f.readings || []).find(y => y.view === r.view));
    const last = rs.map(x => x?.final?.times).filter(x => x != null), first = rs.map(x => x?.first?.times).filter(x => x != null);
    const span = xs => (Math.min(...xs) === Math.max(...xs) ? times(xs[0]) : `${times(Math.min(...xs))}–${times(Math.max(...xs))}`);
    const moved = first.some((x, i) => Math.abs(x - last[i]) > 1e-9);
    const read = el("div", "scn-fig reads");
    read.append(el("span", "num", last.length ? last.map(times).join(" ") : "–"), el("small", "num", moved ? `first ${span(first)}, before the reviewer` : ""));
    list.append(who, prob, you, read);
  }
  // each asset class in each scenario: a scenario a row in the same order, a class a column, its usual on top; how far
  // off, in the class's own usual spread there (usual less its 10th percentile), sets the colour: grey within a
  // quarter of it, dim lime or red within three quarters, bright past; a small return with a decimal
  const labels = v.class_labels || {}, cs = Object.keys(d.classes || {});
  const mat = el("div", "scn-mat");
  mat.style.setProperty("--k", cs.length);
  mat.append(el("div", "mat-h"), ...cs.map(c => el("div", "mat-h num", SHORT[c] ?? labels[c] ?? c, { title: labels[c] ?? c })));
  const matRow = (name, cls, value) => {
    mat.append(el("div", "mat-name " + cls, name), ...cs.map(c => {
      const [x, how] = value(c), cell = el("div", "mat-cell num " + how);
      cell.append(el("small", "mat-k", SHORT[c] ?? labels[c] ?? c), ret(x));
      return cell;
    }));
  };
  matRow("Usual", "usual", c => [d.classes_history?.[c]?.expected, "usual"]);
  for (const f of mine) {
    matRow(f.name, side(f) ?? "", c => {
      const h0 = d.classes_history?.[c], r = d.scenarios?.[f.id]?.classes?.[c];
      const z = r == null || !h0 ? 0 : (r - h0.expected) / Math.max(h0.expected - h0.p10, 1e-6);
      return [r, Math.abs(z) < 0.25 ? "" : `${z > 0 ? "good" : "bad"}${Math.abs(z) < 0.75 ? " dim" : ""}`];
    });
  }
  const classes = el("section", "scn-classes");
  classes.append(el("h3", "ov-h", `Each asset class in each scenario, over the next ${d.label}`), mat);
  const more = [];
  if ((rv.points || []).length) {
    const det = el("details", "earlier"), ul = el("ul");
    det.append(el("summary", null, `What the reviewer said · ${rv.points.length}`));
    ul.append(...rv.points.map(p => el("li", null, `${mine.find(f => f.sid === p.scenario)?.name ?? p.scenario}: ${p.says}. ${p.why}`)));
    det.append(ul);
    more.push(det);
  }
  if ((rv.asked || []).length) {
    const det = el("details", "earlier"), ul = el("ul");
    det.append(el("summary", null, `What it asked the data · ${rv.asked.length}`));
    ul.append(...rv.asked.map(a => el("li", null, `${a.tool} ${JSON.stringify(a.args)}: ${a.answer}`)));
    det.append(ul);
    more.push(det);
  }
  if (set?.coach) more.push(el("p", "foot", `The coach: ${set.coach}`));
  box.replaceChildren(list, classes, ...more, el("p", "len-foot", `One LLM writes the scenarios worth telling apart over the next ${d.label}, each in a direction of its own (under each name, the ranges of the factors it is about); a period of history may be in several or in none. Probability: the simulation's (the tick on the bar) times the readers' factor, history's periods reweighed until each scenario has it (as near as the others allow where they are at odds). The readers: ${readers.length || "four"} LLMs, each reading one scenario on its own and saying how many times as likely it is than the simulation says, for what the simulation leaves out; the factor used is their median. A reviewer checks their reasons, and a scenario it found fault with is read again: below the factors, the range first given, when it changed. Your targets: their mean return in it, lime or red as it is better or worse than usual. Each asset class: its mean return in each scenario, its usual at this length on top; lime or red as it is far above or below it for that class (bright past three quarters of the class's usual spread, dim past a quarter, grey nearer). Click a scenario for all of it.`));
}

// the factors: for each, at each length, the chance of history's lowest and highest quarter
function factorsGrid(v, host) {
  const hs = v.horizons || {}, ks = Object.keys(hs), labels = v.world?.factors || {};
  const names = Object.keys(labels).length ? Object.keys(labels) : Object.keys(hs[ks.at(-1)]?.factors || {});
  if (!names.length) { host.replaceChildren(el("p", "muted", "Appears after the next daily run.")); return; }
  const cell = (f, k) => {
    const x = hs[k].factors?.[f], c = td("num");
    if (!x?.final) return td("soft", "–");
    c.append(el("span", null, `${pct(x.final[0])} / ${pct(x.final.at(-1))}`), el("small", "soft", ` simulation ${pct(x.data?.[0])} / ${pct(x.data?.at(-1))}`));
    c.title = `history's quartiles over ${hs[k].label}: ${x.edges.map(e => e.toPrecision(3)).join(", ")}`;
    return c;
  };
  host.replaceChildren(scrolled(table(["Factor (lowest / highest quarter)", ...ks.map(k => `Next ${hs[k].label}`)], names.map(f => {
    const tr = el("tr");
    tr.append(td(null, labels[f]?.label ?? hs[ks.at(-1)].factors[f]?.label ?? f), ...ks.map(k => cell(f, k)));
    return tr;
  }), "factors")), el("p", "foot", "Each cell: the chance the factor ends in the lowest quarter of its history over that length / in the highest, with the LLM's scenarios; simulation: today's indicators alone, before the LLM's scenarios. History alone gives each 25%. A factor published monthly says nothing over a week (–)."));
}

export function viewSection(v) {
  factorsGrid(v, $("#view [data-factors]"));
  futuresPage(v, $("#scenarios"));
  const main = keyQuestions(v), sec = $("#view");
  $("[data-figures]", sec).replaceChildren(...main.map(figure));
  const rest = v.questions.filter(q => !q.resolved && q.model && q.kind !== "factor");
  rest.sort((a, b) => Math.abs((sides(b).llm ?? sides(b).math) - sides(b).math) - Math.abs((sides(a).llm ?? sides(a).math) - sides(a).math));
  const rows = rest.map(q => {
    const s = sides(q), tr = el("tr", "pick" + (s.apart ? " apart" : ""), null, { tabindex: "0" }), name = el("td");
    name.append(el("span", null, shortName(q)), el("small", "soft", ` ${keyLabel(q)}`));
    const gap = s.llm == null ? null : s.llm - s.math;
    tr.append(name, td("bar-cell opt", bar(q)), td("main num", pct(s.final)), td("soft num opt", s.finalIv ? `${pct(s.finalIv[0])}–${pct(s.finalIv[1])}` : "–"),
      td("num", pct(s.math)), td("num", s.llm == null ? "–" : pct(s.llm)), td("num opt " + (s.apart ? "warn" : "soft"), gap == null ? "–" : pts(gap).replace(" pts", "")),
      td("soft opt", q.scenarios_from ? `next ${v.horizons?.[q.scenarios_from]?.label ?? q.scenarios_from}` : "–"), td("strip-cell opt", strip(s, { width: 150, height: 24, labels: false })), td("soft opt", q.closes ? day(q.closes) : "–"));
    tr.onclick = () => open(q);
    tr.onkeydown = e => { if (e.key === "Enter") open(q); };
    return tr;
  });
  const moves = rest.map(q => sides(q)).filter(s => s.llm != null).map(s => Math.abs(s.llm - s.math));
  const most = moves.length ? Math.max(...moves) : null;
  $("[data-finding]", sec).textContent = most == null
    ? `${rest.length} forecasts open, each read off the simulation (today's indicators); no LLM scenarios in it yet.`
    : `${rest.length} forecasts open, each read off the world: its outcome after every week of history since 1971, weighted as the world weighs that week. The LLM's scenarios move them by ${(100 * most).toFixed(1)} points at most.`;
  $("[data-table]", sec).replaceChildren(scrolled(table(["Forecast: the outcome shown", "", "Final", "90% range", "Simulation", "With the LLM", "LLM − simulation", "Moved by the scenarios of", "", "Resolves"], rows, "sides")));
}
