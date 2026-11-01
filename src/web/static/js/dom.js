// Small DOM helpers shared by the cards of both pages.
export const $ = (sel, root = document) => root.querySelector(sel);

export function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
// status = icon + label, never colour alone (.st in style.css)
export function status(level, text) {
  const s = h("span", "st " + level);
  s.append(h("span", null, text));
  return s;
}
export function setValue(el, num, unit) {
  el.replaceChildren(num);
  if (unit) el.append(h("small", null, unit));
}
export function facts(dl, items) {
  dl.replaceChildren(...items.filter(Boolean).map(([k, v]) => {
    const d = h("div");
    const dd = h("dd");
    dd.append(v);
    d.append(h("dt", null, k), dd);
    return d;
  }));
}
export function legend(ul, items) {
  ul.replaceChildren(...items.map(({ color, name, value, box }) => {
    const li = h("li");
    const key = h("span", box ? "ui-key box" : "ui-key");
    key.style.background = color;
    li.append(key, name + " ", h("b", null, value));
    return li;
  }));
}
export function meter(el, pct, level) {
  el.classList.remove("warning", "critical");
  if (level) el.classList.add(level);
  $("i", el).style.width = Math.max(0, Math.min(100, pct)) + "%";
}
