import { getJSON, el, isDark } from "./util.js";

const pages = {
  experiment: () => import("./pages/experiment.js"),
  catalogs: () => import("./pages/catalogs.js"),
  forecasts: () => import("./pages/forecasts.js"),
  results: () => import("./pages/results.js"),
};

const ctx = { manifest: null, grid: null, catalog: null, page: null, current: null };
const root = document.getElementById("page");

// theme
const themeSeg = document.getElementById("theme-seg");
function applyTheme(mode) {
  const html = document.documentElement;
  if (mode === "system") html.removeAttribute("data-theme");
  else html.setAttribute("data-theme", mode);
  for (const b of themeSeg.children) b.setAttribute("aria-pressed", String(b.dataset.theme === mode));
  try { localStorage.setItem("fc-theme", mode); } catch (e) {}
  window.dispatchEvent(new CustomEvent("fc-theme", { detail: { dark: isDark() } }));
}
themeSeg.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) applyTheme(b.dataset.theme);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!document.documentElement.getAttribute("data-theme")) window.dispatchEvent(new CustomEvent("fc-theme", { detail: { dark: isDark() } }));
});
(() => {
  let t = "system";
  try { t = localStorage.getItem("fc-theme") || "system"; } catch (e) {}
  for (const b of themeSeg.children) b.setAttribute("aria-pressed", String(b.dataset.theme === t));
})();

async function route() {
  const hash = location.hash.replace(/^#\/?/, "") || "experiment";
  const [name, query] = hash.split("?");
  const loader = pages[name] || pages.experiment;
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("active", a.dataset.page === (pages[name] ? name : "experiment"));
  if (ctx.page && ctx.page.destroy) ctx.page.destroy();
  root.replaceChildren(el("div", { class: "empty span-12" }, "Loading…"));
  const mod = await loader();
  ctx.params = Object.fromEntries(new URLSearchParams(query || ""));
  ctx.page = mod;
  root.replaceChildren();
  try {
    await mod.render(root, ctx);
  } catch (err) {
    console.error(err);
    root.replaceChildren(el("div", { class: "empty span-12" }, `This page could not be built: ${err.message}`));
  }
}

async function main() {
  try {
    ctx.manifest = await getJSON("manifest.json");
  } catch (err) {
    root.replaceChildren(el("div", { class: "empty span-12" }, "manifest.json was not found. Run `floatcsep export <config>` and serve the dashboard folder over HTTP."));
    return;
  }
  const m = ctx.manifest;
  document.title = `floatCSEP: ${m.experiment.name}`;
  document.getElementById("exp-name").textContent = m.experiment.name;
  document.getElementById("footer-version").textContent = `floatCSEP ${m.experiment.floatcsep_version || ""}`.trim() + (m.experiment.pycsep_version ? ` · pyCSEP ${m.experiment.pycsep_version}` : "");
  const counts = { forecasts: m.models.length, results: m.tests.length };
  for (const [k, v] of Object.entries(counts)) {
    const c = document.querySelector(`[data-count="${k}"]`);
    if (c) c.textContent = String(v);
  }
  if (m.grid && m.grid.file) ctx.gridPromise = getJSON(m.grid.file);
  window.addEventListener("hashchange", route);
  route();
}

main();
