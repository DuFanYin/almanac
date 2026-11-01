// Reference, This week: the brief, the case against the holdings, what is coming up, the rules nearest to changing
// and their lines in history.
import { day, num, pts, signed } from "./format.js";
import { isShare } from "./questions.js";
import { cited, el, scrolled, table, td } from "./ui.js";

const prose = (body) => { const x = el("p", "prose"); x.append(body); return x; };

export function weekPanel(v, acc) {
  const cols = el("div", "cols"), a = el("div"), b = el("div"), br = v.brief, red = acc?.redteam?.output?.points;
  a.append(el("p", "foot", br ? `Brief of ${day(br.day)}${br.valid ? "" : " (it used numbers not in its data)"}` : "The brief is written on Saturdays."),
    br ? prose(cited(br.text, br.cites)) : el("p", "prose", "What moved during the week and why, every number taken from the data."));
  a.append(el("p", "foot", "Against the current holdings"));
  const ul = el("ul", "points");
  ul.append(...(red?.length ? red.map(p => { const li = el("li"); li.append(cited(String(p), acc.redteam.cites)); return li; }) : [el("li", null, "Written with the brief.")]));
  a.append(ul);
  b.append(table(["Coming up", ""], v.calendar.slice(0, 10).map(c => { const tr = el("tr"); tr.append(td(null, c.what), td("soft", day(c.day))); return tr; })));
  // the rules nearest to changing: each one's numeric condition closest to its line, nearest first
  const near = [];
  const walk = (r, xs) => { for (const x of xs || []) { if (x.group) x.group.forEach(alt => walk(r, alt)); else if (x.gap != null) near.push([r, x]); } };
  for (const r of v.rules) walk(r, r.readings);
  const rel = x => Math.abs(x.gap) / Math.max(Math.abs(x.than), 0.05);
  const best = new Map();
  for (const [r, x] of near.sort((p, q) => rel(p[1]) - rel(q[1]))) if (!best.has(r.id)) best.set(r.id, [r, x]);
  const fmt = x => (isShare(x.expr || "") ? signed(x.value) : num(x.value));
  const line = x => (isShare(x.expr || "") ? signed(x.than) : num(x.than));
  const gap = x => (isShare(x.expr || "") ? pts(x.gap) : (x.gap > 0 ? "+" : "−") + num(Math.abs(x.gap)));
  if (best.size) {
    b.append(el("p", "foot", "For each rule, the condition closest to changing: its value now and the value where it changes."));
    b.append(table(["Rule", "Now", "Flips at", "To go"], [...best.values()].slice(0, 6).map(([r, x]) => {
      const tr = el("tr"), name = el("td");
      const what = x.said || x.expr || x.name;
      name.append(r.title, el("small", "soft", " " + what));
      tr.append(name, td("num", fmt(x)), td("soft num", line(x)),
        td(r.holds ? "num" : "main num", r.holds ? "holds" : gap(x)));
      return tr;
    })));
  }
  cols.append(a, b);
  const whole = el("div");
  whole.append(cols);
  const lines = Object.entries(v.rule_lines || {});
  if (lines.length) {
    // each rule's line on a value, replayed through history with lines moved either side
    const users = name => v.rules.filter(r => JSON.stringify(r.readings).includes(`"name":"${name}"`)).map(r => r.title);
    const say = { steady: "steady", cliff: "a cliff", unclear: "too noisy to say", rare: "too few times" };
    whole.append(el("h4", "sub-h", "The rules' lines in history"));
    whole.append(el("p", "foot", "How often each condition would have started to hold in the past, and the median return one year later, at the line and with the line moved 10% and 20% either way."));
    const t = table(["Line", "Times", "A year after", "−20%", "−10%", "+10%", "+20%", ""], lines.map(([name, x]) => {
      const at = k => x.rows.find(r => r.shift === k), c = at(0), tr = el("tr"), nm = el("td", "wrapcell");
      nm.append(el("span", null, x.said ?? `${x.expr} ${x.op} ${num(c.line)}`), el("small", "soft", ` ${v.labels?.series?.[x.subject] ?? x.subject} followed; read by ${users(name).join("; ")}`));
      const med = r => (r && r.n ? signed(r.median, 0) : "–");
      tr.append(nm, td("num", String(c.n)), td("main num", med(c)), ...[-0.2, -0.1, 0.1, 0.2].map(k => td("num soft", med(at(k)))), td("verdict " + x.verdict, say[x.verdict]));
      return tr;
    }), "lines");
    whole.append(scrolled(t), el("p", "foot", "Steady: moving the line changes the result by less than 10 points. Sensitive: moving it changes the result by more than twice the noise, so the exact line matters too much. Too noisy: the result changes, but within the noise (few cases). A new case is counted after 20 days without one."));
  }
  return whole;
}
