// What the page sends back: a firing marked done or skipped, the owner's own probabilities, a proposed question
// adopted or rejected.
import { showLineage } from "./lineage.js";
import { page } from "./page.js";

export async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error("HTTP " + r.status);
  await page.reload();
}

export const decide = (t, status, note = "") => post("api/scenarios/decision", { trigger: t.id, status, note });

export const skip = t => { const note = prompt("Why skip it? (kept with the record)"); if (note !== null) decide(t, "skipped", note); };

export const why = t => showLineage(`api/scenarios/lineage-by-name?name=${encodeURIComponent("trigger:" + t.id)}`);
