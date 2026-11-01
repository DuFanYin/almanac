// The record page's views, one at a time (the asset classes are on the scenarios page); the chosen one is remembered.
import { $ } from "../dom.js";
import { smallMultiples } from "./indicators.js";
import { recordPanel } from "./record.js";
import { el } from "./ui.js";
import { weekPanel } from "./week.js";

const TABS = [["record", "Scores and decisions"], ["week", "This week"], ["indicators", "Market data"]];

let tab = (() => { try { return TABS.some(([id]) => id === localStorage.getItem("scenarios-tab")) ? localStorage.getItem("scenarios-tab") : "record"; } catch { return "record"; } })();

const PANELS = { indicators: smallMultiples, week: weekPanel, record: recordPanel };

export function reference(s) {
  const tabs = $("#reference [data-tabs]");
  tabs.replaceChildren(...TABS.map(([id, label]) => {
    const b = el("button", null, label, { type: "button", role: "tab", "aria-selected": String(id === tab) });
    b.onclick = () => { tab = id; try { localStorage.setItem("scenarios-tab", id); } catch {} reference(s); };
    return b;
  }));
  $("#reference [data-panel]").replaceChildren(PANELS[tab](s.view, s.account, s.decisions || {}));
}
