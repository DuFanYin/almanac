// Reading a question and a signpost: the headline figure, the bad outcomes, the two sides (the data's world and the
// world with the LLM's scenarios) on the buckets that matter, its line.
import { num, pct, signed, toneClass } from "./format.js";
import { el } from "./ui.js";

export const headlineOf = (q, p) => (q.headline && p ? q.headline.buckets.reduce((a, b) => a + (p[b] ?? 0), 0) : null);

export const top = q => q.buckets.reduce((a, b) => ((q.model?.[b.id] ?? 0) > (q.model?.[a.id] ?? 0) ? b : a), q.buckets[0]);

// the buckets a question is read on: its headline, or its bad outcomes (tone < 0), or its likeliest
const bad = q => q.buckets.filter(b => (b.tone ?? 0) < 0);
export const keyBuckets = q => (q.headline ? q.headline.buckets : bad(q).length ? bad(q).map(b => b.id) : [top(q).id]);
export const keyLabel = q => (q.headline ? q.headline.label : bad(q).length ? bad(q).map(b => b.label).join(" or ") : top(q).label);
const sum = (p, keys) => (p ? keys.reduce((a, k) => a + (p[k] ?? 0), 0) : null);

// the two sides and the final on the question's key buckets: a band is shown only on one bucket, as the draws'
// intervals of several buckets do not add up
export function sides(q) {
  const keys = keyBuckets(q), one = band => (keys.length === 1 ? band?.[keys[0]] ?? null : null);
  return {
    keys,
    final: sum(q.model, keys),
    finalIv: one(q.interval),
    math: sum(q.math, keys),
    mathIv: one(q.math_interval),
    llm: q.sources?.futures ? sum(q.sources.futures, keys) : null,
    w: q.weights?.llm ?? 0,
    apart: false,
  };
}

// one outcome's two sides and the final (a scenario's, each on its own)
export const bucketSides = (q, b) => ({
  keys: [b],
  final: q.model?.[b] ?? null,
  finalIv: q.interval?.[b] ?? null,
  math: q.math?.[b] ?? null,
  mathIv: q.math_interval?.[b] ?? null,
  llm: q.sources?.futures ? q.sources.futures[b] : null,
  w: q.weights?.llm ?? 0,
  apart: false,
});

// the view: the scenario question, open

export const badWhenUp = q => keyBuckets(q).reduce((a, b) => a + (q.buckets.find(x => x.id === b)?.tone ?? 0), 0) < 0;

export const weekAgo = q => (q.history || []).filter(r => r.model && r.day <= new Date(Date.now() - 6.5 * 864e5).toISOString().slice(0, 10)).at(-1);

// the week's key forecasts: those whose probability stands furthest from history's rate (the board's "key")
export const keyQuestions = v => (v.key || []).map(k => v.questions.find(q => q.id === k.id)).filter(Boolean);

export const shortName = q => q.headline?.short ?? q.title.replace(/^FOMC decision: /, "FOMC ");

export const spName = (v, id) => v.signposts.find(s => s.id === id)?.name ?? id.replaceAll("_", " ");

// good for the holdings on the left, bad on the right, on every bar of the page
export const byTone = buckets => [...buckets].sort((a, b) => (b.tone ?? 0) - (a.tone ?? 0));

export function bar(q, cls = "bar") {
  const b = el("div", cls);
  for (const x of byTone(q.buckets)) {
    const seg = el("i", toneClass(x.tone), null, { title: `${x.label}: ${pct(q.model?.[x.id], 1)}` });
    seg.style.flexGrow = Math.max(0.002, q.model?.[x.id] ?? 0);
    b.append(seg);
  }
  return b;
}

export const isShare = e => /\b(ret|dd|gap|yoy|ann|rank)\(/.test(e);

export const spValue = s => (s.value == null ? "–" : isShare(s.expr) ? signed(s.value) : num(s.value));
// a threshold ("< -0.15") in the same form as the value it is compared with

export const line = (s, c) => { const [op, x] = c.split(" "); return `${op} ${isShare(s.expr) ? signed(+x) : num(+x)}`; };
