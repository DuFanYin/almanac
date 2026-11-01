// The page loads this: it registers its service worker (sw.js at the app root, from this file's
// versioned address s/<version>/js/fresh.js) and shows the page again when the worker says its copy was out of date.
if ("serviceWorker" in navigator) {
  const root = new URL("../../../", import.meta.url);
  navigator.serviceWorker.register(new URL("sw.js", root), { scope: root.pathname }).catch(() => {});
  navigator.serviceWorker.addEventListener("message", (e) => { if (e.data === "reload") location.reload(); });
}
