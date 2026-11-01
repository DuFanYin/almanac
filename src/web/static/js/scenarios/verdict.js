// What a fired rule asks, in one instruction and today's reading behind it, and when it stops asking: the overview's
// to-do rows and the detail panel read these.
import { pct, usd } from "./format.js";

// what each fired rule asks, in one instruction, and today's reading behind it
export function sentence(t, s) {
  const acc = s.account, sh = acc?.shares, tg = acc?.targets, a = acc?.amounts?.[t.id];
  const btcValue = sh && sh.btc * acc.total;
  const S = {
    btc_dca: [`Buy ${usd(a?.usd)} of Bitcoin`, ["The regular purchase."]],
    btc_dca_double: [`Buy another ${usd(a?.usd)} of Bitcoin`, ["AHR999 is in its buy zone, or the price is under the doubling line."]],
    btc_cap: ["Hold the Bitcoin purchases", sh && [`Bitcoin is <b>${pct(sh.btc)}</b> of everything (${usd(btcValue)} of ${usd(acc.total)}); the cap is <b>${pct(tg.btc_max)}</b>.`]],
    btc_thesis: ["Pause Bitcoin and review the thesis", ["One of the conditions you set for the Bitcoin plan being wrong is now true."]],
    halving_review: ["Review the Bitcoin plan", ["The 2028 halving is near."]],
  };
  const r = s.view.rules.find(x => x.id === t.rule);
  const [head, today] = S[t.rule] ?? [r?.title ?? t.rule, null];
  return { head, rule: r?.explain ?? r?.title, today: (today || []).filter(Boolean).join(" "), lifts: lifts(t, s) };
}

// when the rule stops asking, in today's numbers
export function lifts(t, s) {
  const acc = s.account, sh = acc?.shares, tg = acc?.targets;
  if (t.rule === "btc_cap" && sh) {
    const B = sh.btc * acc.total, T = acc.total, c = tg.btc_max;
    const fall = B - (c * (T - B)) / (1 - c), more = B / c - T;
    return `When Bitcoin is back under ${pct(c)} of everything: a fall of about <b>${usd(fall)}</b> in its value, or <b>${usd(more)}</b> more in everything else.`;
  }
  return null;
}
