// Reference, Record: every source's score, the LLM panel's record against the computed side, what the decisions rest
// on, the decision layer walked through history against simple ways of holding the same things, how often the LLM's
// writing passed its checks, the panel's method measured, the versions tried.
import { day, pct, pts, signed, times, tone } from "./format.js";
import { el, scrolled, table, td } from "./ui.js";

export function recordPanel(v) {
  const sc = Object.entries(v.scores || {}).sort((a, b) => a[1].brier - b[1].brier);
  const label = {
    base: "History", market: "Market", math: "Simulation (history weighted by the indicators, with the market)", llm: "LLM", research: "Research's own forecast",
    research_first: "Research before its debate (the old app)", own: "Yours", prior: "Prior (before the LLM)", model: "Final",
    no_llm: "Final without the LLM", no_research: "Without research (before the LLM)", uniform: "Even guess",
  };
  const versioned = k => (k.startsWith("llm ") ? `LLM, method ${k.slice(4)}` : k.startsWith("research ") ? `Research, method ${k.slice(9)}` : k);
  const wrap = el("div");
  wrap.append(sc.length ? scrolled(table(["Source", "Brier (lower is better)", "Log score (lower is better)", "Forecasts"], sc.map(([k, x]) => {
    const tr = el("tr"); tr.append(td(null, label[k] ?? versioned(k)), td("main", x.brier.toFixed(3)), td(null, x.log == null ? "–" : x.log.toFixed(2)), td("soft", String(x.questions))); return tr;
  }))) : el("p", "prose muted", "No forecast has resolved yet. The first are the S&P 500 and Bitcoin this month, on 1 December; then every source is scored on every day it gave a probability."));
  // the LLM's scenarios: how much they move each length, and how the sets whose length is over turned out: the weight
  // each scenario of each scored set, happened or not, against its probability, the simulation's share and history's
  const hz = Object.values(v.horizons || {}), fr = v.scenarios_record;
  if (hz.length) {
    wrap.append(el("h4", "sub-h", "The LLM's scenarios"));
    wrap.append(scrolled(table(["Length", "Weight the LLM moved", "Your bad case, with it", "The simulation", "Without the LLM"], hz.map(h => {
      const tr = el("tr"), st = h.stats?.targets;
      tr.append(td(null, h.label), td("main num", pct(h.llm_moved ?? 0)), td("num " + tone(st?.all?.bad, st?.data?.bad, 1), signed(st?.all?.bad, 1)), td("soft num", signed(st?.data?.bad, 1)), td("soft num", signed(st?.no_llm?.bad, 1)));
      return tr;
    }))));
    wrap.append(el("p", "foot", "The simulation: history's periods weighted by today's indicators. Each of the LLM's scenarios should have its probability in the simulation times the readers' factor; history's periods are reweighed until it does, and the weight moved is how much of the futures changed place. Without the LLM: the simulation without the indicators on texts the LLM read or that it proposed. Your bad case lime or red: better or worse than the simulation's."));
    if (fr?.n) {
      const lg = x => (x == null ? "–" : x.toFixed(2));
      wrap.append(el("p", "prose", `${fr.n} scenario${fr.n > 1 ? "s" : ""} scored. The mean log of the probability given to what came about, that it happened or that it did not (higher is better): with the LLM ${lg(fr.llm)}, the simulation ${lg(fr.share)}, history ${lg(fr.history)}.`),
        scrolled(table(["Written", "Length", "Scenario", "Happened", "Readers' factor", "With the LLM", "Simulation", "History"], fr.sets.map(x => {
          const tr = el("tr");
          tr.append(td("soft", day(x.written)), td(null, v.horizons?.[x.horizon]?.label ?? x.horizon), td(null, x.name), td(null, x.hit ? "yes" : "no"), td("soft num", times(x.times)), td("main num", pct(x.weight, 1)), td("num", pct(x.share, 1)), td("soft num", pct(x.history, 1)));
          return tr;
        }))));
    } else wrap.append(el("p", "prose", "No set of scenarios has reached its end yet: the first, a month's, a month after it was written. The score goes back to the LLMs when they write and read; it never changes a factor."));
  }
  const bt = v.backtest;
  if (bt?.base) {
    // the decision step replayed through history: on the data's world as it could be built then, against simpler ways
    const row = (name, x, cls) => {
      const tr = el("tr", cls || ""), m = x.all;
      tr.append(td(null, name), td("main num", signed(m.cagr, 1)), td("num", signed(x.first.cagr, 1)), td("num", signed(x.second.cagr, 1)),
        td("num bad", signed(m.max_dd, 0)), td("num soft", m.worst_2y == null ? "–" : signed(m.worst_2y, 0)), td("num soft", pct(x.turnover, 0)));
      return tr;
    };
    const name = v.you?.own ? "Your targets" : "The base mix";
    const rows = [row(`The decision step (${bt.sense ? `the data counted ×${bt.sense}, ${bt.weights_name}` : "the data not counted"})`, bt.chosen, "pick"), row(`${name}, rebalanced past your band`, bt.base),
      ...Object.entries(bt.others || {}).map(([n, x]) => row(n, x))];
    wrap.append(el("h4", "sub-h", `The decision step through history, ${bt.from.slice(0, 4)} to today`));
    wrap.append(scrolled(table(["", "A year", `To ${bt.middle.slice(0, 4)}`, "After", "Deepest fall", "Worst two years", "Traded a year"], rows, "backtest")));
    const tried = (bt.grid || []).filter(g => g.sense > 0).sort((a, b) => b.first.cagr - a.first.cagr)[0];
    wrap.append(el("p", "foot", `Every four weeks from ${day(bt.from)}, the data's world as it could be built that day (the indicators measured on the years before, the values known then; the LLM's scenarios and market prices cannot be rebuilt and are left out), each length's futures the periods already over, cash at that day's bill rate; the decision step's mix, then its trades on what the replay holds (your band, a step at a time), the holdings drifting between. Total returns with dividends and interest. How much the data counts (its sensitivity) and how much each length counts are chosen on the years to ${bt.middle.slice(0, 4)} and judged on those after: a sensitivity is kept only if it adds at least ${pct(bt.rules.gain_min, 1)} a year there without a fall ${pct(bt.rules.dd_slack, 0)} deeper or much more trading. ${bt.sense ? "" : `None did${tried ? ` (the best, ×${tried.sense} with the lengths ${tried.weights}, ${signed(tried.first.cagr, 1)} a year to ${bt.middle.slice(0, 4)} against ${signed(bt.base.first.cagr, 1)})` : ""}, so the data does not move your targets: your targets, the guardrails and your band. `}The decision step includes the guardrails: each length's bad case within its limit.${bt.left_out?.length ? ` Not included, too little history: ${bt.left_out.map(k => v.class_labels?.[k] ?? k).join(", ")}.` : ""} Before 1994 the tech classes use the Nasdaq Composite.`));
  }
  const tc = v.text_checks;
  if (tc) {
    const share = x => (x.passed == null ? "–" : `${pct(x.passed)} of ${x.n}`);
    wrap.append(el("h4", "sub-h", "How often the LLM's work passed its checks"));
    wrap.append(scrolled(table(["", "Passed", "The check"], [
      ["The weekly brief", tc.brief, "every number one of the data's, cited by its id"],
      ["The case against your holdings", tc.redteam || { passed: null, n: 0 }, "every number one of the data's, cited by its id"],
      ["Research's notes", tc.notes, "every number the cited news item's, and the item real; a failing note is dropped"],
      ["The LLM's answers", tc.panel, "a usable probability for every outcome; a failed answer keeps that panelist's last"],
    ].map(([k, x, why]) => { const tr = el("tr"); tr.append(td(null, k), td("main num", share(x)), td("wrapcell", why)); return tr; }))));
  }
  // the indicators the LLM proposed: checked on the data, then judged on history and on what came after; none moves a
  // forecast before both
  const sp = v.signposts_proposed || [], L = v.labels || {}, fac = t => L.factors?.[t.replace(/_short$/, "")] ?? t.replace(/_short$/, "");
  wrap.append(el("h4", "sub-h", "Indicators on trial"));
  if (!sp.length) wrap.append(el("p", "prose muted", "None yet. The LLM proposes up to three every four weeks."));
  else {
    const STATE = { "waiting for the calibration": "waiting for Saturday's check on history", "failed in history": "did not beat history", "on new data": "beat history; now on the data after it was proposed", active: "beat history and the data after", "failed on new data": "did not hold on the data after" };
    const STATUS = { trial: "on trial", accumulating: "collecting readings", failed: "turned down", active: "in use", retired: "retired" };
    wrap.append(scrolled(table(["Proposed", "Indicator", "Meant to tell about", "Where it stands", "Status"], sp.map(p => {
      const tr = el("tr"), name = el("td", "wrapcell");
      name.append(el("span", null, p.name), el("small", null, `${p.said ?? `${p.value}, red ${p.red}`} · ${p.reading ? "on a text the LLM reads" : "proposed by the LLM"}`));
      const judged = Object.entries(p.judged || {}), same = new Set(judged.map(([, j]) => j.state)).size === 1 && !judged.some(([, j]) => j.windows != null);
      const where = p.status === "accumulating" ? "collecting a year of readings before it is checked" : p.status === "failed" ? (p.checks || []).filter(c => !c.ok).map(c => `${c.name}: ${c.say}`).join("; ")
        : !judged.length ? STATE["waiting for the calibration"] : same ? STATE[judged[0][1].state] ?? judged[0][1].state
        : judged.map(([t, j]) => `${fac(t)}: ${STATE[j.state] ?? j.state}${j.windows != null ? ` (${j.windows} periods)` : ""}`).join("; ");
      tr.append(td("soft", day(p.day)), name, td("soft", p.try_on.map(fac).join(", ")), td("wrapcell soft", where), td(p.status === "active" ? "main" : "soft", STATUS[p.status] ?? p.status));
      return tr;
    }), "futures")));
  }
  wrap.append(el("p", "foot", "Paused since the world was built: an indicator moves it only through what it says about the factors, a mechanism measured against history. Those the LLM proposes will come back drafting that, measured like the others and used at half their size at first (the LLM has read the history they are measured on). Those on the texts the LLM reads wait for years of readings."));
  // texts read into numbers: the latest, each with the phrase it rests on
  const rs = v.readings || [];
  wrap.append(el("h4", "sub-h", "Texts read into numbers"));
  if (!rs.length) wrap.append(el("p", "prose muted", "Nothing read yet."));
  else wrap.append(scrolled(table(["Day", "What is read", "Text", "Reading", "Rests on"], rs.map(r => {
    const tr = el("tr"), t = td("wrapcell");
    const title = r.title.replace(/(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})\S*/, (_, d, hm) => `${day(d)} ${hm}`);
    if (r.url) t.append(el("a", null, title, { href: r.url, rel: "noopener noreferrer", target: "_blank" })); else t.append(title);
    tr.append(td("soft", day(r.day)), td("soft", (L.series?.[r.series] ?? r.series).replace(/\s*\(LLM[^)]*\)$/, "")), t, td("main num", r.value == null ? "not relevant" : `${r.value > 0 ? "+" : ""}${r.value.toFixed(1)}${r.spread ? ` ±${r.spread.toFixed(1)}` : ""}`), td("wrapcell soft", r.quote ? `“${r.quote}”` : "–"));
    return tr;
  }), "futures")));
  wrap.append(el("p", "foot", "Each text is read three times by the LLM onto a scale from −2 to +2 (Fed: toward cutting … toward raising; AI investment, in the news and in the earnings releases of Microsoft, Alphabet, Amazon, Meta, NVIDIA and Oracle: cut … sharply raised; news: alarm … euphoria), and an answer counts only if the phrases it quotes are in the text word for word. Each day's value is the mean of its readings, from 30 October 2026 on: texts from before are never read, since the model knows how they turned out. The indicators on these readings wait a year before they are checked and tested like the others."));
  const au = v.audit;
  if (au?.classes?.length) {
    // what the decision step rests on, class by class: how far today's world stands from usual at each length, what
    // chance alone would give, the score and the lean
    wrap.append(el("h4", "sub-h", "What the decision rests on"));
    const H = v.horizons || {}, ks = [...new Set(au.classes.flatMap(c => Object.keys(c.lengths)))];
    const rows = au.classes.map(c => {
      const tr = el("tr");
      tr.append(td(null, v.class_labels?.[c.class] ?? c.class), ...ks.map(k => { const x = c.lengths[k]; return x ? td("num " + (Math.abs(x.shrunk) >= 0.001 ? "main " + tone(x.gap, 0, 1) : "soft"), signed(x.gap, 1), el("small", "soft", ` ±${pct(x.se, 1)}`)) : td("soft", "–"); }),
        td("num", c.score.toFixed(3)), td(c.lean ? "main num" : "soft num", c.lean ? pts(c.lean) : "–"), td("soft", c.status));
      return tr;
    });
    wrap.append(scrolled(table(["Asset class", ...ks.map(k => H[k]?.label ?? k), "Score", "Target moved", ""], rows)));
    const n = au.counts || {};
    wrap.append(el("p", "foot", `Each class's expected return in today's world less in history's own periods, at each length the decision step acts at; ± what a reweighting as large would give by chance; lime or red where it is past that, better or worse than usual. The score: what is left after that noise, measured in the class's usual swings, the lengths together (${Object.entries(au.weights || {}).map(([k, x]) => `${H[k]?.label ?? k} ${pct(x)}`).join(", ")}), the data's part counted ×${au.sense ?? 0} and the LLM's ×${v.decision?.llm_sense ?? 1}. The target moved: the score as points of your mix, at most ±${pct(v.you?.limits?.tilt ?? 0)} a class; only the classes in your targets move, and cash takes the difference. ${au.guard?.length ? `The guardrails moved ${au.guard.map(g => `${pct(g.size, 1)} from ${v.class_labels?.[g.from] ?? g.from} into ${v.class_labels?.[g.to] ?? g.to}`).join(", ")}. ` : "No guardrail moved anything. "}Today: ${Object.entries(n).map(([k, x]) => `${x} ${k}`).join(", ")}.`));
  }
  const evs = (v.evals || []).filter(e => "spread_first" in e.overall);
  if (evs.length) {
    wrap.append(el("h4", "sub-h", "The LLM's method, measured before any outcome"));
    wrap.append(scrolled(table(["Version", "Measured", "Runs disagree, first round", "Runs disagree, last round", "Reversed order moves it", "Viewpoints disagree", "Cost"], evs.map(e => {
      const tr = el("tr"), o = e.overall;
      tr.append(td(null, e.method), td("soft", day(e.day)), td("num", pct(o.spread_first, 1)), td("num", pct(o.spread_last, 1)), td("num", pct(o.order_gap, 1)),
        td("num", pct(o.view_gap, 1)), td("soft num", e.cost_usd == null ? "–" : "$" + e.cost_usd.toFixed(2)));
      return tr;
    }))));
    wrap.append(el("p", "foot", "The LLM is re-tested on the same forecasts and the same first three data sets whenever its prompts, code or model change. Lower is better in every column. Runs disagree: how much six runs of the same question differ, in the first and the last round. Reversed order: how much the answers change when the outcomes are listed in reverse. Viewpoints: how far the three viewpoints differ at the end."));
  }
  const ver = v.versions || {};
  const tried = [["rules", "the rules"], ["panel", "the LLM's method"], ["forecast", "the old forecast method"], ["research", "research"], ["proposal", "proposed forecasts"]].filter(([k]) => ver[k]).map(([k, name]) => `${name} ${ver[k].n} version${ver[k].n > 1 ? "s" : ""} since ${day(ver[k].first)}`);
  wrap.append(el("p", "foot", (tried.length ? `Tried: ${tried.join("; ")}. Every version counts: a good record after many tries is worth less than after one. ` : "")
    + `Code version ${v.code} · settings version ${v.config}. Each probability keeps its sources: open one and follow it to the raw data and the model's own words.`));
  return wrap;
}
