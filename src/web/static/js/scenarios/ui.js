// The page's small DOM pieces: an element, a button, a table that scrolls on its own on a narrow screen.
import { h } from "../dom.js";

export function el(tag, cls, text, attrs = {}) {
  const e = h(tag, cls, text);
  // a style through the CSSOM: the page's CSP (style-src 'self') drops a style attribute set as markup
  for (const k in attrs) k === "style" ? (e.style.cssText = attrs[k]) : e.setAttribute(k, attrs[k]);
  return e;
}

// a text written with its data ids ([D12] after each number): each id a small superscript number, the data item it
// stands for (its name and value) on hover
export function cited(text, cites = {}) {
  const out = document.createDocumentFragment();
  let at = 0;
  for (const m of text.matchAll(/\s*\[(D\d+)\]/g)) {
    out.append(text.slice(at, m.index));
    const sup = el("sup", "cite", m[1].slice(1));
    if (cites[m[1]]) sup.title = cites[m[1]];
    out.append(sup);
    at = m.index + m[0].length;
  }
  out.append(text.slice(at));
  return out;
}

export const button = (label, cls, fn) => { const b = el("button", "btn " + cls, label, { type: "button" }); b.onclick = fn; return b; };

export const escapeHtml = s => s.replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

export const scrolled = t => { const d = el("div", "tbl-scroll"); d.append(t); return d; };

export const td = (cls, ...xs) => { const e = el("td", cls); e.append(...xs); return e; };

// a table of a header row and rows; cls adds the table's own kind (backtest, lines)
export function table(heads, rows, cls) {
  const t = el("table", cls ? "tbl " + cls : "tbl"), hr = el("tr");
  hr.append(...heads.map(x => { const h = el("th"); h.append(x); return h; }));
  t.append(hr, ...rows);
  return t;
}
