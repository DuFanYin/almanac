// Holdings: by category against your targets and the decided shares (your targets leaned by today's world within the
// guardrails), on each length's futures now and after the suggested trades, and by asset class against the decision.
import { $ } from "../dom.js";
import { day, pct, pts, signed, usd } from "./format.js";
import { el, scrolled, table, td } from "./ui.js";

export function plan(acc) {
  const c = $("#plan");
  if (!acc?.categories) { $("[data-sub]", c).textContent = "No holdings yet: fin/data/exposure.json comes with the IBKR refresh."; return; }
  const s = acc.shares, t = acc.targets, cats = acc.categories;
  $("[data-sub]", c).textContent = `The IBKR account, ${usd(acc.nav)} on ${day(acc.date)}, by category: now and at its target`;
  const shade = i => "c" + Math.min(i, 8);
  const stack = rows => {
    const b = el("div", "stack");
    for (const [cls, label, share] of rows) {
      if (share < 0.004) continue;
      // the name where it fits, the share alone where it does not
      const seg = el("i", cls, share >= 0.2 ? `${label} ${pct(share)}` : share >= 0.045 ? pct(share) : "", { title: `${label}: ${pct(share, 1)}` });
      seg.style.flexGrow = share;
      b.append(seg);
    }
    return b;
  };
  const now = cats.map((k, i) => [shade(i), k.label, s[k.id] ?? 0]);
  if ((s.unclassified ?? 0) > 0.004) now.push(["cx", "Unclassified", s.unclassified]);
  $("[data-mix]", c).replaceChildren(el("span", null, "Now"), stack(now), el("span", null, "Target"), stack(cats.map((k, i) => [shade(i), k.label, t[k.id] ?? 0])));
  const moves = $("[data-moves]", c), head = el("tr"), dec = acc.decided || {};
  head.append(...["", "Now", "Target", "Decided", "To reach it"].map(x => el("th", null, x)));
  const body = [head];
  // the category's symbols, after its name: "NVDA 24% · GOOGL 9%"
  const symbols = cat => {
    const rows = (acc.rows || []).filter(r => r.category === cat);
    return el("span", "syms", rows.map(r => `${r.symbol}${r.options ? " with options" : ""} ${pct(r.value / acc.nav)}`).join(" · "),
      { title: rows.map(r => `${r.symbol}: ${usd(r.value)}`).join("\n") });
  };
  cats.forEach((k, i) => {
    const gap = acc.gaps?.[k.id] ?? 0, share = s[k.id] ?? 0;
    if (share < 0.0005 && !(t[k.id] > 0)) return;
    const tr = el("tr", "cat"), name = el("td");
    name.append(el("i", "sw " + shade(i)), k.label + (k.cap ? ` (cap ${pct(k.cap)})` : ""), symbols(k.id));
    const d = dec[k.id], leaned = d != null && Math.abs(d - (t[k.id] ?? 0)) >= 0.0005;
    tr.append(name, el("td", null, pct(share)), el("td", null, pct(t[k.id] ?? 0)), el("td", leaned ? "main" : "soft", d == null ? "–" : pct(d, 1)),
      el("td", "move", Math.abs(gap) < 25 ? "–" : `${gap < 0 ? "sell" : "buy"} ${usd(Math.abs(gap))}`));
    body.push(tr);
  });
  if (acc.unclassified?.length) {
    const tr = el("tr", "cat warn");
    const name = el("td", null, "Not in a category yet");
    name.append(symbols("unclassified"));
    tr.append(name, el("td", null, pct(s.unclassified)), el("td"), el("td"), el("td", "move", "classify under [symbols] in local/rules.toml"));
    body.push(tr);
  }
  const btc = el("tr", "cat");
  const btcName = el("td");
  btcName.append(el("i", "sw"), "Bitcoin, of everything");
  btc.append(btcName, el("td", null, pct(s.btc)), el("td", null, `cap ${pct(t.btc_max)}`), el("td"),
    el("td", "move", s.btc >= t.btc_max ? "hold purchases" : `room for ${usd((t.btc_max - s.btc) * acc.total)}`));
  body.push(btc);
  moves.replaceChildren(...body);
  const missing = acc.missing?.trades || acc.missing?.cash;
  $("[data-foot]", c).textContent = [`Tech together: ${pct(s.tech)} against ${pct(t.tech)}.`,
    missing ? `The exported CSVs end ${day(acc.csv_end)}: ${acc.missing.trades} trade${acc.missing.trades === 1 ? "" : "s"} since are not in these numbers.` : "",
    "Decided: your targets leaned by today's world within the guardrails (the decision step); to reach it, the whole gap to the decided share, which the suggestions close a step at a time.",
    "Categories, their targets and each symbol's category are set by hand in local/rules.toml."].filter(Boolean).join(" ");
}

// which of the LLM's scenarios the worst 10% of a horizon's futures is in
export function worstParts(v, parts, n = 2) {
  const names = Object.fromEntries((v.scenarios || []).map(f => [f.id, f.name]));
  return Object.entries(parts || {}).slice(0, n).map(([k, x]) => `${names[k] ?? k} ${pct(x)}`).join(", ");
}

// the holdings on each horizon's futures: the expected return, and the bad case (a bar) against its limit (a tick)
export function horizonBars(v, acc) {
  const a = acc?.against, wrap = el("div", "outbars");
  if (!a?.horizons) { wrap.append(el("p", "prose muted", "Needs the holdings and the futures.")); return wrap; }
  const rows = Object.values(a.horizons).filter(h => h.holding);
  const max = Math.max(0.1, ...rows.flatMap(h => [Math.abs(h.holding.bad), Math.abs(h.limit ?? 0)]));
  for (const h of rows) {
    const st = h.holding, past = h.limit != null && st.bad < h.limit - 1e-9;
    const row = el("div", "ob"), name = el("span", "ob-name", `Next ${h.label}`);
    const from = worstParts(v, h.worst_parts);
    if (from) name.append(el("small", null, `Worst 10% mostly from: ${from}`));
    // losses only: 0 at the right end, the bar reaching left to the bad case
    const axis = el("div", "ob-axis loss"), barEl = el("i", "neg"), tick = el("u");
    const at = x => 100 + (Math.min(x, 0) / max) * 100;
    barEl.style.left = at(st.bad) + "%";
    barEl.style.width = 100 - at(st.bad) + "%";
    axis.append(barEl);
    if (h.limit != null) { tick.style.left = at(h.limit) + "%"; tick.title = `your limit ${signed(h.limit, 0)}`; axis.append(tick); }
    row.append(name, el("span", "ob-p num", signed(st.expected, 0)), axis, el("b", "num " + (past ? "bad" : ""), signed(st.bad, 0)));
    wrap.append(row);
  }
  wrap.append(el("p", "foot", `Left: the expected return of what you hold now. Bar and right: the bad case, the average of the worst 10% of futures. White tick: your limit for that horizon. Futures: history's periods reweighted to today's forecasts, plus the futures the LLM wrote.${a.outside?.length ? ` Not included: ${a.outside.join(", ")}.` : ""}`));
  return wrap;
}

// the holdings by asset class against your targets with today's adjustments: the classes on the left (now, the target,
// a bar for now with a tick at the target), the expected return and bad case of both on each horizon on the right
export function against(v, a) {
  const sec = $("#against");
  sec.hidden = !a;
  if (!a) return;
  const labels = { ...(v.class_labels || {}), options: "Options", unclassified: "Unclassified" };
  $("#against-h").textContent = "Against your targets";
  $("[data-sub]", sec).textContent = `Risk level ${a.profile} (${a.label}). The decided mix: your targets moved by today's world, within the guardrails.`;
  const keys = [...new Set([...Object.keys(a.mix), ...Object.keys(a.now)])].sort((x, y) => (a.mix[y] ?? 0) + (a.now[y] ?? 0) - (a.mix[x] ?? 0) - (a.now[x] ?? 0));
  const outside = new Set([...(a.outside || []), "options", "unclassified"]);
  // what lies outside the targets' classes stays as it is; the targets rearrange the rest, scaled to its share
  const rest = 1 - Object.entries(a.now).filter(([k]) => outside.has(k) && !(k in a.mix)).reduce((x, [, w]) => x + w, 0);
  const want = k => (outside.has(k) && !(k in a.mix) ? null : (a.mix[k] ?? 0) * rest);
  const max = Math.max(0.01, ...keys.flatMap(k => [a.now[k] ?? 0, want(k) ?? 0]));
  const byClass = table(["Asset class", "Now", "Decided", "", "Move"], keys.map(k => {
    const now = a.now[k] ?? 0, w = want(k), tr = el("tr"), track = el("div", "track"), tilt = (a.mix[k] ?? 0) - (a.policy[k] ?? 0);
    const fill = el("i");
    fill.style.width = (now / max) * 100 + "%";
    track.append(fill);
    if (w != null) { const tick = el("u"); tick.style.left = (w / max) * 100 + "%"; track.append(tick); }
    const tg = td("soft num", w == null ? "outside" : pct(w));
    if (Math.abs(tilt) >= 0.0005) tg.append(el("small", tilt > 0 ? "good" : "bad", ` ${tilt > 0 ? "+" : "−"}${pct(Math.abs(tilt))} today`));
    tr.append(td(null, labels[k] ?? k), td("num", pct(now)), tg, td("bar-cell", track),
      td("main num", w == null ? "–" : pts(w - now).replace(" pts", "")));
    return tr;
  }));
  $("[data-classes]", sec).replaceChildren(scrolled(byClass));
  const hs = Object.values(a.horizons || {});
  const fig = (x, limit, cls = "") => td(`num ${cls} ${limit != null && x != null && x < limit - 1e-9 ? "bad" : ""}`, x == null ? "–" : signed(x, 0));
  const row = (label, cells) => { const tr = el("tr"); tr.append(td(null, label), ...cells); return tr; };
  const rows = [
    row("Holdings, expected", hs.map(h => fig(h.holding?.expected, null, "main"))),
    row("Holdings, bad case", hs.map(h => fig(h.holding?.bad, h.limit, "main"))),
    row("Holdings, deepest fall on the way", hs.map(h => fig(h.holding?.dd?.bad, null, "main"))),
    row("After the suggested trades, expected", hs.map(h => fig(h.after?.expected ?? h.holding?.expected))),
    row("After the suggested trades, bad case", hs.map(h => fig(h.after?.bad ?? h.holding?.bad, h.limit))),
    row("Targets, expected", hs.map(h => fig(h.targets?.expected))),
    row("Targets, bad case", hs.map(h => fig(h.targets?.bad, h.limit))),
    row("Targets, deepest fall on the way", hs.map(h => fig(h.targets?.dd?.bad))),
    row("Your limit for the bad case", hs.map(h => td("soft num", signed(h.limit, 0)))),
  ];
  $("[data-figures]", sec).replaceChildren(scrolled(table(["Next", ...hs.map(h => h.label)], rows)),
    el("p", "foot", `Bad case: the average of the worst 10% of futures, at the end of the horizon. Deepest fall on the way: the average of the worst 10% of the largest drops from a high before the end, held without rebalancing. Red: past your limit for that horizon. Targets: your categories' targets (local/rules.toml) grouped by asset class, without today's leans. After the suggested trades: what you hold once the trades suggested today are done (the same as now when there are none). ${[...outside].some(k => a.now[k]) ? `Not included: ${[...outside].filter(k => a.now[k]).map(k => labels[k] ?? k).join(", ")}; the rest is ${pct(rest)} of the total. ` : ""}Bar: now. White tick: target.`));
}
