// The page's service worker (served at the app root as sw.js, so its scope is the page). The page crosses the tunnel,
// a few hundred milliseconds by way of Cloudflare's Tokyo edge, so it opens at once from the browser's copy of it and
// is fetched again behind it; its data is only in the page, so it is told to show itself again when the new copy
// differs, which it does from the copy just stored, so at once too;
// - a fetch that ends at the sign-in page (signed out, or the session ran out) drops every copy and sends the page
//   there.
// Nothing but the page's own address is touched: not the sign-in, the API or the assets.
const CACHE = "invest-1";
const PAGES = [""];
const RELOAD = new Set([""]);

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil((async () => {
  // its own older copies only: other pages on the same site keep theirs
  for (const k of await caches.keys()) if (k.startsWith("invest-") && k !== CACHE) await caches.delete(k);
  await self.clients.claim();
})()));

self.addEventListener("fetch", (e) => {
  const req = e.request, root = self.registration.scope;
  if (req.method !== "GET" || req.mode !== "navigate") return;
  const url = new URL(req.url);
  url.hash = "";
  if (url.search || !url.href.startsWith(root) || !PAGES.includes(url.href.slice(root.length))) return;
  e.respondWith(open(e, url.href, url.href.slice(root.length)));
});

async function open(e, key, page) {
  const cache = await caches.open(CACHE), copy = await cache.match(key);
  if (!copy) {
    const res = await fetch(e.request);
    if (res.ok && !res.redirected) e.waitUntil(cache.put(key, res.clone()));
    return res;
  }
  const before = copy.clone();
  e.waitUntil(refresh(cache, key, page, before));
  return copy;
}

async function refresh(cache, key, page, before) {
  let res;
  try { res = await fetch(key, { cache: "no-store" }); } catch { return; }  // offline: the copy stays
  if (res.redirected || res.status === 401) {  // signed out
    await caches.delete(CACHE);
    return tell(key);
  }
  if (!res.ok) return;
  const text = await res.clone().text();
  await cache.put(key, res);
  if (RELOAD.has(page) && text !== await before.text()) tell(key);
}

// the open windows of this page: show it again (from the new copy, or the sign-in page)
async function tell(key) {
  for (const c of await self.clients.matchAll({ type: "window" }))
    if (c.url.split("#")[0] === key) c.postMessage("reload");
}
