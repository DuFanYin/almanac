// Scenarios page: a board, one view at a time. The overview answers whether there is anything to do and how the plan
// stands against the world, in one screen; the rest is a tab away (the scenarios in full, every forecast, the map of
// how it is made, the system's stages, the holdings, the proposed forecasts, the record), and anything chosen opens in full in the side
// panel. The view is in the address (#map), so a link or the back button returns to it.
// Data: api/scenarios, written by the daily job (src/board/run.py). No inline styles (CSP).
import { $ } from "../dom.js";
import { day } from "./format.js";
import { against, horizonBars, plan } from "./holdings.js";
import { close, render as inspect } from "./inspector.js";
import { initDialogs } from "./lineage.js";
import { lineageMap } from "./map.js";
import { overview } from "./overview.js";
import { page } from "./page.js";
import { proposalsSection } from "./proposals.js";
import { reference } from "./reference.js";
import { systemPage } from "./system.js";
import { el } from "./ui.js";
import { viewSection } from "./view.js";

const PAGES = [["overview", "Overview"], ["scenarios", "Scenarios"], ["forecasts", "Forecasts"], ["map", "Map"],
  ["system", "System"], ["holdings", "Holdings"], ["proposals", "Proposals"], ["record", "Record"]];

const current = () => { const h = location.hash.slice(1); return PAGES.some(([id]) => id === h) ? h : "overview"; };

function show() {
  const id = current(), s = page.state;
  for (const sec of document.querySelectorAll("#app [data-page]")) sec.hidden = sec.dataset.page !== id;
  for (const b of document.querySelectorAll("[data-pages] button")) b.setAttribute("aria-selected", String(b.dataset.id === id));
  // the map is laid out to its width, which a hidden section does not have
  if (id === "map" && s?.view) $("#map [data-map]").replaceChildren(lineageMap(s.view, s.decisions || {}));
  if (id === "system" && s?.view) $("#system [data-system]").replaceChildren(systemPage(s.view));
}

function nav(s) {
  const waiting = (s.view.proposals || []).filter(p => p.status === "open" && !s.votes?.[p.id]).length;
  $("[data-pages]").replaceChildren(...PAGES.map(([id, label]) => {
    const b = el("button", null, label, { type: "button", role: "tab", "data-id": id });
    if (id === "proposals" && waiting) b.append(el("span", "badge", String(waiting)));
    b.onclick = () => { location.hash = id; };
    return b;
  }));
}

function render(s) {
  page.state = s;
  const v = s.view;
  $("#empty").hidden = !!v;
  if (!v) return;
  const errors = Object.keys(v.errors || {});
  const sub = $("#sub");
  sub.replaceChildren(`Data to ${day(v.day)} · updated ${new Date(v.updated * 1000).toLocaleString("en-GB", { hour12: false, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}`);
  if (errors.length || v.stale.length) sub.append(" · ", el("span", "warn", errors.length ? `not updating: ${errors.join(", ")}` : `stale: ${v.stale.join(", ")}`));
  nav(s);
  overview(s);
  viewSection(v);
  inspect();
  const held = !!s.account?.categories;
  $("#holdings [data-none]").hidden = held;
  for (const x of document.querySelectorAll("#holdings [data-held]")) x.hidden = !held;
  if (held) {
    plan(s.account);
    $("#under [data-bars]").replaceChildren(horizonBars(v, s.account));
    against(v, s.account?.against);
  }
  proposalsSection(s);
  reference(s);
  show();
  $("#foot").textContent = "Sources: Yahoo, Cboe, FRED, CNN, DefiLlama, Kalshi, multpl, Coin Metrics (Bitcoin before 2015), Google News · LLM: DeepSeek · your rules and targets: local/ in Almanac's folder";
}

async function load() {
  const r = await fetch("api/scenarios", { cache: "no-store" });
  if (r.ok) render(await r.json());
}

page.reload = load;
initDialogs();
// a view is a tab, not a place on the page: its section's id in the address would scroll it under the sticky header
addEventListener("hashchange", () => { close(); show(); scrollTo(0, 0); });
addEventListener("load", () => { if (location.hash) scrollTo(0, 0); });
const initial = JSON.parse($("#initial").textContent);
if (initial) render(initial);
else load();
