import { el, panel, seg, fmt, doiBadge, makeMap, outlineStyle, seriesColor, css } from "../util.js";
import { makeChart, disposeChart, chartNode, chrome, axis } from "../charts.js";
import { GridLayer } from "../gridlayer.js";

let map = null, chart = null, onTheme = null;

const TEST_TYPES = { consistency: "Consistency", comparative: "Comparative", sequential: "Sequential", sequential_comparative: "Sequential comparative", batch: "Batch" };

function props(pairs) {
  const dl = el("dl", { class: "props" });
  for (const [k, v] of pairs) {
    if (v === null || v === undefined || v === "" || v === "None") continue;
    dl.append(el("dt", {}, k), el("dd", {}, v));
  }
  return dl;
}

function code(v) {
  return v === null || v === undefined ? null : el("code", {}, typeof v === "object" ? JSON.stringify(v) : String(v));
}

function key(k) {
  return el("span", { class: "cfg-key" }, k);
}

function header(m) {
  const e = m.experiment;
  const badges = el("div", { class: "badges" });
  const add = (label, node) => { if (node) badges.append(el("span", { class: "badge-label" }, label), node); };
  add("Experiment", doiBadge(e.doi));
  add("Manuscript", doiBadge(e.manuscript_doi));
  add("Catalog", doiBadge(e.catalog_doi));
  if (e.license) badges.append(el("span", { class: "chip" }, el("b", {}, e.license)));
  badges.append(el("span", { class: "chip accent" }, el("b", {}, e.exp_class)));

  const right = el("div", { class: "stack", style: { gap: "8px", alignItems: "flex-end" } });
  if (e.fair) {
    const tone = (v) => (v >= 75 ? css("--good") : v >= 40 ? css("--warn") : css("--bad"));
    right.append(el("span", { class: "note" }, "FAIR score, F-UJI"),
      el("div", { class: "fair" }, ["F", "A", "I", "R", "FAIR"].filter((k) => e.fair[k] !== undefined).map((k) =>
        el("span", { class: "pill", style: { borderColor: tone(e.fair[k]) } }, el("b", { style: { color: tone(e.fair[k]) } }, `${Math.round(e.fair[k])}`), el("span", {}, k)))));
  }
  return el("section", { class: "exp-head" },
    el("div", {},
      el("h1", {}, e.name),
      e.authors ? el("p", { class: "authors" }, e.authors) : null,
      e.journal ? el("p", { class: "note", style: { margin: "-6px 0 10px" } }, e.journal) : null,
      badges),
    right);
}

function flow(m) {
  const e = m.experiment, r = m.region || {};
  const lines = (...xs) => el("ul", { class: "lines" }, xs.filter(Boolean).map((x) => el("li", {}, x)));
  const cell = (title, n, unit, href, body) => el("a", { class: "flow-cell", href },
    el("h3", {}, title), el("div", { class: "n" }, n, unit ? el("span", { class: "unit" }, unit) : null), body);
  const nw = m.time_windows.length;
  const mag = (v) => (v === null || v === undefined ? "?" : Number(v).toFixed(1));
  return el("section", { class: "panel span-12" }, el("div", { class: "flow" },
    cell("Time Configuration", String(nw), nw === 1 ? "window" : "windows", "#/catalogs", lines(
      `${fmt.iso(e.start_date)} – ${fmt.iso(e.end_date)}`,
      e.horizon ? `Horizon ${e.horizon}` : null,
      e.growth ? `${e.growth[0].toUpperCase()}${e.growth.slice(1)} windows` : null)),
    cell("Region", fmt.int(r.n_cells), "cells", "#/catalogs", lines(
      r.name, r.dh ? `${r.dh}° grid` : null, `M ${mag(r.mag_min)} – ${mag(r.mag_max)}`)),
    cell("Models", String(m.models.length), "", "#/forecasts", lines(...m.models.map((x) => x.name))),
    cell("Tests", String(m.tests.length), "", "#/results", lines(...m.tests.map((x) => x.name))),
  ));
}

function modelsList(m) {
  return m.models.map((mod, i) =>
    el("div", { class: "model-card" },
      el("h4", {}, el("span", { class: "swatch", style: { background: seriesColor(i) } }), mod.name, doiBadge(mod.doi)),
      props([
        ["Authors", mod.authors],
        ["Type", mod.forecast_class === "CatalogForecastRepository" ? "Catalog-based" : "Gridded"],
        ["Forecasts", `${Object.keys(mod.forecasts || {}).length} / ${m.time_windows.length} windows`],
        ["Forecast unit", mod.forecast_unit ? `${mod.forecast_unit} years` : null],
        ["Source", mod.giturl ? el("a", { href: mod.giturl, target: "_blank", rel: "noopener" }, mod.giturl.replace(/^https?:\/\//, "")) : null],
        ["Commit", code(mod.git_hash)],
        ["Zenodo", mod.zenodo_id ? el("a", { href: `https://zenodo.org/records/${mod.zenodo_id}`, target: "_blank", rel: "noopener" }, String(mod.zenodo_id)) : null],
        ["Path", code(mod.path)],
        ["Function", code(mod.func)],
        ["Arguments", mod.func_kwargs && Object.keys(mod.func_kwargs).length ? code(mod.func_kwargs) : null],
        ["Format", mod.fmt],
        ["Note", mod.notes && mod.notes.length ? mod.notes.join("; ") : null],
      ])));
}

function testsList(m) {
  return m.tests.map((t) =>
    el("div", { class: "model-card" },
      el("h4", {}, t.name, el("span", { class: "note" }, TEST_TYPES[t.type] || t.type || "")),
      props([
        ["Function", code(t.func)],
        ["Arguments", t.func_kwargs && Object.keys(t.func_kwargs).length ? code(t.func_kwargs) : null],
        ["Reference", t.ref_model],
        ["Plot", t.plot_func && t.plot_func.length ? code(t.plot_func.join(", ")) : null],
      ])));
}

function citeBlock(e) {
  const c = e.citation;
  if (!c) return null;
  const box = el("textarea", { class: "cite-box", readonly: true, "aria-label": "Citation" });
  box.value = c.apa;
  const copy = el("button", { class: "btn small", onclick: async () => {
    try { await navigator.clipboard.writeText(box.value); copy.textContent = "Copied"; } catch (err) { box.select(); copy.textContent = "Press Ctrl+C"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1500);
  } }, "Copy");
  const fmtSeg = seg([["apa", "APA"], ["bibtex", "BibTeX"], ["ris", "RIS"]], "apa", (v) => { box.value = c[v] || ""; });
  return panel("Citation", el("div", { class: "stack", style: { gap: "8px" } },
    el("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, fmtSeg, copy, el("span", { class: "note", style: { marginLeft: "auto" } }, `Source: ${c.source === "DataCite" ? "DataCite" : "experiment metadata"}`)),
    box));
}

function windowsChart(node, m) {
  chart = makeChart(node);
  const c = chrome();
  const wins = m.time_windows;
  const data = wins.map((w, i) => [new Date(w.start).getTime(), new Date(w.end).getTime(), i, w.label]);
  const accent = css("--accent");
  chart.setOption({
    ...c,
    grid: { left: 10, right: 18, top: 10, bottom: 30, containLabel: true },
    tooltip: { ...c.tooltip, trigger: "item", formatter: (p) => `Window ${p.data[2] + 1}<br>${p.data[3].replace(" to ", " – ")}` },
    xAxis: { type: "time", ...axis() },
    yAxis: { type: "category", data: wins.map((_, i) => `${i + 1}`), inverse: true, ...axis(), axisLine: { show: false }, splitLine: { show: false }, axisLabel: { show: wins.length <= 30, color: css("--ink-3") } },
    series: [{
      type: "custom",
      renderItem: (params, api) => {
        const a = api.coord([api.value(0), api.value(2)]), b = api.coord([api.value(1), api.value(2)]);
        const h = Math.max(2, Math.min(14, api.size([0, 1])[1] * 0.6));
        return { type: "rect", shape: { x: a[0], y: a[1] - h / 2, width: Math.max(2, b[0] - a[0]), height: h, r: 2 }, style: { fill: accent, opacity: 0.85 } };
      },
      encode: { x: [0, 1], y: 2 },
      data,
    }],
  });
}

export async function render(root, ctx) {
  const m = ctx.manifest, e = m.experiment, r = m.region || {};
  root.append(header(m), flow(m));
  const left = el("div", { class: "span-7 stack" });
  const right = el("div", { class: "span-5 stack" });
  root.append(right, left);

  const mapNode = el("div", { class: "map short" });
  left.append(panel("Testing Region", el("div", {}, mapNode, el("div", { style: { padding: "12px 16px" } }, props([
    ["Name", r.name],
    ["Cell size", r.dh ? `${r.dh}°` : null],
    ["Magnitudes", `${Number(r.mag_min).toFixed(1)} – ${Number(r.mag_max).toFixed(1)}`],
    ["Bin width", r.mag_bin ? `ΔM ${r.mag_bin}` : null],
    ["Depth", r.depth_min !== null && r.depth_min !== undefined ? `${r.depth_min} – ${r.depth_max} km` : null],
  ]))), { flush: true }));

  const winNode = chartNode("short");
  left.append(panel("Time Windows", el("div", {},
    winNode,
    el("div", { style: { padding: "4px 8px 8px" } }, props([
      ["Class", e.exp_class],
      ["Start", fmt.iso(e.start_date)],
      ["End", fmt.iso(e.end_date)],
      ["Horizon", e.horizon],
      ["Offset", e.offset],
      ["Growth", e.growth ? `${e.growth[0].toUpperCase()}${e.growth.slice(1)}` : null],
    ]))), { tight: true }));

  right.append(panel("Models", el("div", {}, modelsList(m))));
  right.append(panel("Tests", el("div", {}, testsList(m))));
  const cite = citeBlock(e);
  if (cite) right.append(cite);
  right.append(panel("Run", props([
    ["floatCSEP", e.floatcsep_version],
    ["pyCSEP", e.pycsep_version],
    ["Last run", e.last_run],
    ["Run mode", e.run_mode ? `${e.run_mode[0].toUpperCase()}${e.run_mode.slice(1)}` : null],
    ["Results", code(e.run_dir)],
    ["Configuration", code(e.config_file)],
    ["Models file", code(e.model_config)],
    ["Tests file", code(e.test_config)],
    ["Catalog file", code(m.catalog && m.catalog.source)],
    ["Exported", fmt.iso(m.generated)],
  ])));

  windowsChart(winNode, m);

  map = makeMap(mapNode);
  if (r.bbox) map.fitBounds([[r.bbox[1], r.bbox[0]], [r.bbox[3], r.bbox[2]]], { padding: [10, 10] });
  else map.setView([0, 0], 2);
  let outline = null, cells = null;
  const draw = async () => {
    if (!ctx.gridPromise) return;
    const g = await ctx.gridPromise;
    if (cells) cells.remove();
    if (outline) outline.remove();
    cells = new GridLayer({ fill: css("--accent-soft") }).addTo(map);
    cells.setData({ origins: g.origins, dh: g.dh, values: new Float32Array(g.origins.length).fill(1) });
    if (g.outline) outline = L.geoJSON(g.outline, { style: outlineStyle() }).addTo(map);
  };
  await draw();

  onTheme = () => { draw(); disposeChart(chart); windowsChart(winNode, m); };
  window.addEventListener("fc-theme", onTheme);
}

export function destroy() {
  window.removeEventListener("fc-theme", onTheme);
  if (map) { map.remove(); map = null; }
  disposeChart(chart);
  chart = null;
}
