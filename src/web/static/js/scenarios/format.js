// Numbers and days as the page writes them.

export const pct = (v, d = 0) => (v == null ? "–" : (v * 100).toFixed(d) + "%");

export const usd = v => (v == null ? "–" : (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 }));

// better or worse for you than a baseline, as the page colours it: "good" (lime), "bad" (red), or "" when the two read
// the same at the precision shown (d decimals of a percent); lower: the lower the better
export const tone = (x, base, d = 0, lower = false) =>
  x == null || base == null || (x * 100).toFixed(d) === (base * 100).toFixed(d) ? "" : (x > base) !== lower ? "good" : "bad";

export const signed = (v, d = 1) => (v == null ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(d) + "%");

// a factor on a probability: ×2.0, ×0.50, ×12
export const times = v => (v == null ? "–" : "×" + (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2)));

export const pts = v => (Math.abs(v) < 0.0005 ? "±0 pts" : (v > 0 ? "+" : "−") + Math.abs(v * 100).toFixed(1) + " pts");

export const num = v => (v == null ? "–" : Math.abs(v) >= 1e9 ? (v / 1e9).toFixed(1) + "B" : Math.abs(v) >= 1000
  ? v.toLocaleString("en-US", { maximumFractionDigits: 0 }) : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));

export const toneClass = t => (t == null ? "t0" : t <= -1.5 ? "t-2" : t <= -0.75 ? "t-1" : t < 0 ? "t-h" : t === 0 ? "t0" : t < 1.5 ? "t1" : "t2");

export const period = p => (/W\d+$/.test(p) ? "this week" : /^\d{4}-\d{2}$/.test(p) ? "this month" : /Q\d$/.test(p) ? "this quarter" : "once");
// a day as "29 Oct", with its year when it is not this year's

export const day = s => new Date(s + "T12:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", ...(s.slice(0, 4) === String(new Date().getFullYear()) ? {} : { year: "numeric" }) });

// a text the LLM wrote for a person, its ids made names: "[D3]" goes, "[cpi_next]" becomes the forecast's title in
// brackets when it is known, and goes otherwise
export const named = (t, labels = {}) => (t || "").replace(/\s*\[([A-Za-z0-9_-]+)\]/g, (_, id) => {
  const name = labels.questions?.[id] ?? labels.signposts?.[id] ?? labels.factors?.[id];
  return name ? ` (${name})` : "";
});

// a text the LLM wrote, without its data ids: "[D3]" goes, "[D26 core PCE 3.03%]" keeps what it says, in brackets
export const plain = t => (t || "").replace(/\s*\[([^\]]*)\]/g, (_, x) => {
  const y = x.replace(/\bD\d+\b:?\s*/g, "").replace(/^[\s,;]+|[\s,;]+$/g, "");
  return y ? ` (${y})` : "";
});
