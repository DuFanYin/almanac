// Proposals: forecasts the LLM proposes (weekly) and new factors (every four weeks), each checked on the data first;
// adopted or not by you, or, left three days, by the LLM (three readings, a majority); anything adopted you can
// withdraw.
import { $ } from "../dom.js";
import { post } from "./api.js";
import { day, named, pct } from "./format.js";
import { showLineage } from "./lineage.js";
import { button, el, scrolled, table, td } from "./ui.js";

const vote = (p, status) => {
  const note = status === "rejected" ? prompt("Why not? (kept with the record)") : "";
  if (note === null) return;
  post("../api/scenarios/proposal", { id: p.id, status, note }).catch(err => alert("Not saved: " + err.message));
};

// a new factor: what it measures, and how much more of the asset classes' returns the fit explains with it
function factorBody(p) {
  const s = p.spec, g = p.checks.gains || {}, labels = { short: "3 months", long: "1 year", longer: "3 years" };
  return [el("p", "pane-key", `${s.series}, measured as its ${{ ret: "return", diff: "change", ann: "yearly rate" }[s.measure] ?? s.measure} over each horizon · drafted ${day(p.day)}`),
    scrolled(table(["Horizon", "Explained in addition"], Object.entries(g).map(([h, x]) => { const tr = el("tr"); tr.append(td(null, labels[h] ?? h), td("num " + (x == null ? "" : x >= 0.02 ? "good" : x < 0 ? "bad" : ""), x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(1)} points`)); return tr; })))];
}

function proposal(p, votes, labels) {
  const s = p.spec, box = el("article", "prop"), mine = votes[p.id];
  box.append(el("h4", null, s.title), el("p", "prop-why", named(s.why, labels)));
  if (s.type === "factor") {
    box.append(...factorBody(p));
    return finish(p, box, mine);
  }
  const horizon = s.kind === "path" ? `over ${s.days} days` : `each ${s.every}`;
  box.append(el("p", "pane-key", `${s.purpose === "score" ? "To score the forecasters sooner" : "Would change a decision"} · ${labels.series?.[s.series] ?? s.series} ${horizon} · drafted ${day(p.day)}`));
  const base = p.checks.base || {};
  box.append(scrolled(table(["Outcome", "How often in history"], s.buckets.map(b => {
    const tr = el("tr");
    tr.append(td(null, b.label), td("num", pct(base[b.id], 0)));
    return tr;
  }))));
  return finish(p, box, mine);
}

// the checks, and adopt or reject
function finish(p, box, mine) {
  const checks = el("ul", "checks");
  for (const c of p.checks.checks) {
    const li = el("li", c.ok ? "ok" : "no");
    li.append(el("b", null, `${c.ok ? "✓" : "✗"} ${c.name[0].toUpperCase()}${c.name.slice(1)}`), `: ${c.say}`);
    checks.append(li);
  }
  box.append(checks);
  const acts = el("div", "pane-links");
  const done = { question: "Adopted: registered at the next run.", factor: "Adopted: written to local/factors.toml at the next run, in the futures from then on." }[p.spec.type || "question"];
  if (p.status === "declined by the LLM") box.append(el("p", "foot", `The LLM declined it${p.why ? `: ${p.why}` : ""}. You can still adopt it.`));
  if (mine) acts.append(el("span", "soft", mine.status === "adopted" ? done : "Rejected: kept with the record at the next run."), button("Undo", "bare", () => post("../api/scenarios/proposal", { id: p.id, status: "open" })));
  else acts.append(button("Adopt", "primary", () => vote(p, "adopted")), button("Reject", "", () => vote(p, "rejected")));
  if (p.node) acts.append(button("Lineage", "bare", () => showLineage(`../api/scenarios/lineage/${p.node}`)));
  box.append(acts);
  return box;
}

export function proposalsSection(s) {
  const v = s.view, sec = $("#proposals"), all = v.proposals || [], votes = s.votes || {};
  const waiting = all.filter(p => p.status === "open" || p.status === "declined by the LLM"), failed = all.filter(p => p.status === "failed");
  const decided = all.filter(p => ["adopted", "rejected", "withdrawn", "retired"].includes(p.status));
  const grid = $("[data-list]", sec);
  grid.replaceChildren(...waiting.filter(p => (p.spec.type || "question") !== "rule").map(p => proposal(p, votes, v.labels || {})));
  if (!waiting.length) grid.append(el("p", "empty-line", "Nothing waiting. The LLM drafts up to three forecasts a week and up to two factors every four weeks."));

  // beside an odd one (or alone): how drafts are judged, those turned down, those decided
  const more = el("div", "props-aside");
  more.append(el("p", "prose", "Each draft is checked before you see it. A forecast: history has enough independent cases and no outcome is near 0% or 100%; it does not repeat a forecast the system already makes; a forecast for decisions must change how your targets would do between its outcomes, by more than the uncertainty; a forecast for scoring must resolve within three months. A factor: its data go back to 1971, it is not a near copy of a factor, and with it the asset classes' returns are explained better on stretches of history held out, by at least 2 points at some horizon. One that passes and that you leave for three days is decided by the LLM: asked three times on its own, a majority adopts, at most two forecasts and one factor a month; a forecast it adopts that opens every period stops after a year. You can adopt one it declined, and withdraw anything adopted."));
  if (failed.length) {
    const d = el("details", "stand-ins");
    d.append(el("summary", null, `${failed.length} draft${failed.length > 1 ? "s" : ""} the data turned down`), el("ul", null));
    d.lastChild.append(...failed.map(p => el("li", null, `${day(p.day)}, ${p.spec.title}: ${p.checks.checks.filter(c => !c.ok).map(c => `${c.name}, ${c.say}`).join("; ")}`)));
    more.append(d);
  }
  if (decided.length) {
    const ul = el("ul", "decided");
    for (const p of decided) {
      const li = el("li");
      li.append(el("span", null, `${p.spec.title}: ${p.status}${p.by === "llm" ? " by the LLM" : ""} ${day(p.decided)}`));
      if (p.why) li.append(el("small", "soft", ` · ${p.why}`));
      const mine = votes[p.id];
      if (p.status === "adopted" && mine?.status !== "withdrawn") li.append(" ", button("Withdraw", "bare small", () => vote(p, "withdrawn")));
      else if (mine?.status === "withdrawn" && p.status === "adopted") li.append(el("small", "soft", " · withdrawn at the next run"));
      ul.append(li);
    }
    more.append(ul);
  }
  if (waiting.length % 2) grid.append(more);
  else $("[data-more]", sec).replaceChildren(more);
  if (waiting.length % 2) $("[data-more]", sec).replaceChildren();
}
