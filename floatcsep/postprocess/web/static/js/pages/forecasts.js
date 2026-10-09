import { el, panel, select, field, fmt, getJSON, makeMap, outlineStyle, css, decodeU16, cellValue, doiBadge, seriesColor, debounce } from "../util.js";
import { makeChart, disposeChart, chartNode, chrome, axis } from "../charts.js";
import { GridLayer, drawColorbar } from "../gridlayer.js";

let map = null, chart = null, onTheme = null, layer = null, events = null, outline = null;

async function loadForecast(m, modelId, winId) {
  const mod = m.models.find((x) => x.id === modelId);
  const rel = mod && mod.forecasts[winId];
  if (!rel) return null;
  const raw = await getJSON(rel);
  if (!raw._q) {
    raw._q = decodeU16(raw.data || "");
    raw.data = "";
  }
  return { q: raw._q, nm: raw.mags.length, vmin: raw.vmin, vmax: raw.vmax, mags: raw.mags, totals: raw.totals, n: raw.n_cells, grid: raw.grid };
}

function valuesAt(fc, k) {
  const out = new Float32Array(fc.n);
  const finite = [];
  for (let i = 0; i < fc.n; i++) {
    const v = cellValue(fc, i, k);
    out[i] = v;
    if (Number.isFinite(v)) finite.push(v);
  }
  finite.sort((a, b) => a - b);
  const count = finite.length;
  return { values: out, lo: count ? finite[Math.floor(count * 0.01)] : NaN, hi: count ? finite[count - 1] : NaN, count };
}

export async function render(root, ctx) {
  const m = ctx.manifest, r = m.region || {};
  const models = m.models.filter((x) => Object.keys(x.forecasts || {}).length);
  if (!models.length) {
    root.append(el("div", { class: "empty span-12" }, "No forecasts were exported for this experiment."));
    return;
  }
  const grid = ctx.gridPromise ? await ctx.gridPromise : null;
  const mags = m.magnitudes || [];
  const state = { model: models[0].id, win: Object.keys(models[0].forecasts)[0], k: 0, range: null, lock: false, showCat: true };
  let fc = null, cur = null, catalog = null;
  const thr = () => (fc ? fc.mags : mags)[state.k];
  const windows = m.time_windows;

  // controls
  const modelSel = select(models.map((x) => [x.id, x.name]), state.model, (v) => { state.model = v; refresh(); });
  const winSel = select(windows.map((w) => [w.id, w.label]), state.win, (v) => { state.win = v; refresh(); });
  const step = (d) => {
    const i = windows.findIndex((w) => w.id === state.win) + d;
    if (i >= 0 && i < windows.length) { state.win = windows[i].id; winSel.value = state.win; refresh(); }
  };
  const winRow = el("div", { class: "stepper" },
    el("button", { class: "btn", title: "Previous window", "aria-label": "Previous window", onclick: () => step(-1) }, "‹"), winSel,
    el("button", { class: "btn", title: "Next window", "aria-label": "Next window", onclick: () => step(1) }, "›"));
  const magVal = el("span", { class: "val" }, `M ≥ ${fmt.num(mags[0], 1)}`);
  const magSlider = el("input", { type: "range", min: 0, max: Math.max(0, mags.length - 1), step: 1, value: 0, "aria-label": "Magnitude threshold", oninput: (e) => { state.k = +e.target.value; magVal.textContent = `M ≥ ${fmt.num(thr(), 1)}`; draw(); } });
  const rangeLo = el("input", { type: "range", min: -10, max: 3, step: 0.05, "aria-label": "Colour scale minimum" });
  const rangeHi = el("input", { type: "range", min: -10, max: 3, step: 0.05, "aria-label": "Colour scale maximum" });
  const rangeLab = el("span", { class: "val", style: { minWidth: "7em" } });
  const onRange = debounce(() => {
    let a = +rangeLo.value, b = +rangeHi.value;
    if (a >= b) { if (document.activeElement === rangeLo) { b = a + 0.05; rangeHi.value = b; } else { a = b - 0.05; rangeLo.value = a; } }
    state.range = [a, b];
    applyRange();
  }, 30);
  rangeLo.addEventListener("input", onRange);
  rangeHi.addEventListener("input", onRange);
  const rangeBox = el("div", { class: "drange" }, el("div", { class: "track" }), el("div", { class: "fill" }), rangeLo, rangeHi);
  const fitBtn = el("button", { class: "btn small", onclick: () => { state.range = null; draw(); } }, "Reset scale");
  const lockCb = el("label", { class: "checkbox" }, el("input", { type: "checkbox", onchange: (e) => { state.lock = e.target.checked; } }), "Lock scale");
  const catCb = el("label", { class: "checkbox" }, el("input", { type: "checkbox", checked: true, onchange: (e) => { state.showCat = e.target.checked; drawEvents(); } }), "Show observed events");
  const left = el("div", { class: "span-4 stack" });
  const right = el("div", { class: "span-8 stack" });
  root.append(left, right);
  const summary = el("div", { class: "readout", "aria-live": "polite" });
  left.append(panel("Forecast", el("div", { class: "stack", style: { gap: "16px" } },
    el("div", { class: "group" }, field("Model", modelSel), field("Time window", winRow)),
    el("div", { class: "group" },
      el("div", { class: "field" }, el("span", {}, "Minimum magnitude"), el("div", { class: "range" }, magSlider, magVal)),
      el("div", { class: "field" }, el("span", {}, "Colour scale, log₁₀ rate per cell"), el("div", { class: "range" }, rangeBox, rangeLab)),
      el("div", { class: "row" }, fitBtn, lockCb)),
    m.catalog ? el("div", { class: "group" }, catCb) : null,
    el("div", { class: "group" }, summary),
  ), { class: "rail" }));
  const kpis = el("dl", { class: "props" });
  const modelMeta = el("dl", { class: "props" });
  left.append(panel("Forecast Details", el("div", { class: "stack", style: { gap: "12px" } }, modelMeta, kpis)));

  // map
  const mapNode = el("div", { class: "map", role: "region", "aria-label": "Forecast map" });
  const tip = el("div", { class: "cell-tip", role: "status" });
  const mapWrap = el("div", { class: "map-wrap" }, mapNode, tip);
  const cb = el("canvas", { "aria-hidden": "true" });
  const cbLo = el("span", {}), cbHi = el("span", {});
  const cbBar = el("div", { class: "colorbar" }, el("span", { class: "cb-label" }, "log₁₀ rate per cell"), cbLo, cb, cbHi, el("span", { class: "note", id: "cb-note" }));
  right.append(panel("Forecast Map", el("div", {}, mapWrap, cbBar), { flush: true }));
  const cmpNode = chartNode();
  right.append(panel("Expected N(≥M) per Model", cmpNode, { tight: true }));

  map = makeMap(mapNode);
  if (r.bbox) map.fitBounds([[r.bbox[1], r.bbox[0]], [r.bbox[3], r.bbox[2]]], { padding: [10, 10] });
  layer = new GridLayer().addTo(map);
  const drawOutline = () => {
    if (outline) outline.remove();
    if (grid && grid.outline) outline = L.geoJSON(grid.outline, { style: { ...outlineStyle(), weight: 1, opacity: 0.6 } }).addTo(map);
  };
  drawOutline();
  map.on("mousemove", (e) => {
    const hit = layer.valueAt(e.latlng);
    if (!hit) { tip.style.display = "none"; return; }
    const p = map.latLngToContainerPoint(e.latlng);
    tip.style.display = "block";
    tip.style.left = p.x + 14 + "px";
    tip.style.top = p.y + 14 + "px";
    tip.innerHTML = `Rate <b>${fmt.sci(Math.pow(10, hit.value))}</b><br>log₁₀ ${fmt.num(hit.value, 2)}<br><span class="note">Cell ${fmt.lon(hit.origin[0])} ${fmt.lat(hit.origin[1])}</span>`;
  });
  map.on("mouseout", () => { tip.style.display = "none"; });

  chart = makeChart(cmpNode);

  function setRangeUI(lo, hi) {
    const min = Math.floor(Math.min(lo, fc ? fc.vmin : lo) - 0.5), max = Math.ceil(Math.max(hi, fc ? fc.vmax : hi) + 0.5);
    rangeLo.min = rangeHi.min = min;
    rangeLo.max = rangeHi.max = max;
    rangeLo.value = lo;
    rangeHi.value = hi;
    const fill = rangeBox.querySelector(".fill");
    fill.style.left = ((lo - min) / (max - min)) * 100 + "%";
    fill.style.width = ((hi - lo) / (max - min)) * 100 + "%";
    rangeLab.textContent = `${fmt.num(lo, 2)} – ${fmt.num(hi, 2)}`;
    cbLo.textContent = fmt.num(lo, 1);
    cbHi.textContent = fmt.num(hi, 1);
    drawColorbar(cb, lo, hi);
  }

  function applyRange() {
    if (!state.range) return;
    layer.setRange(state.range[0], state.range[1]);
    setRangeUI(state.range[0], state.range[1]);
  }

  function draw() {
    if (!fc) return;
    cur = valuesAt(fc, state.k);
    const g = fc.grid === "region" ? grid : fc.grid;
    layer.setData({ origins: g.origins, dh: g.dh, values: cur.values });
    if (!state.range || !state.lock) state.range = cur.count ? [cur.lo, cur.hi] : [0, 1];
    if (state.range && state.lock) { /* keep */ }
    applyRange();
    const total = fc.totals[state.k];
    const mod = m.models.find((x) => x.id === state.model);
    const w = windows.find((x) => x.id === state.win);
    const obs = catalog ? countObserved() : null;
    summary.replaceChildren(el("table", { class: "mini" },
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, `M ≥ ${fmt.num(thr(), 1)}`), el("th", { scope: "col" }, "Events"))),
      el("tbody", {},
        el("tr", {}, el("td", {}, "Expected"), el("td", {}, fmt.num(total, total >= 100 ? 0 : 2))),
        obs !== null ? el("tr", {}, el("td", {}, "Observed"), el("td", {}, String(obs))) : null)));
    kpis.replaceChildren();
    for (const [k, v] of [
      ["Active cells", `${fmt.int(cur.count)} / ${fmt.int(fc.n)}`],
      ["Max cell rate", cur.count ? fmt.sci(Math.pow(10, cur.hi)) : "–"],
      ["Magnitude bins", `${fc.mags.length}`],
      ["Forecast unit", mod.forecast_unit ? `${mod.forecast_unit} years` : null],
    ]) if (v) kpis.append(el("dt", {}, k), el("dd", {}, v));
    drawEvents();
  }

  function countObserved() {
    const wi = windows.findIndex((x) => x.id === state.win), th = thr();
    let n = 0;
    for (let i = 0; i < catalog.n; i++) if (catalog.w[i].includes(wi) && catalog.mag[i] >= th - 1e-9) n++;
    return n;
  }

  function drawEvents() {
    if (events) { events.remove(); events = null; }
    if (!state.showCat || !catalog) return;
    const wi = windows.findIndex((x) => x.id === state.win), th = thr();
    const canvas = L.canvas({ padding: 0.3 });
    events = L.layerGroup();
    for (let i = 0; i < catalog.n; i++) {
      if (!catalog.w[i].includes(wi) || catalog.mag[i] < th - 1e-9) continue;
      L.circleMarker([catalog.lat[i], catalog.lon[i]], { renderer: canvas, radius: 3 + Math.max(0, catalog.mag[i] - th) * 1.8, color: "#ffffff", weight: 1.2, fillColor: css("--accent"), fillOpacity: 0.95 })
        .bindPopup(`<b>M ${fmt.num(catalog.mag[i], 1)}</b><br>${fmt.datetime(catalog.t[i])} UTC`).addTo(events);
    }
    events.addTo(map);
  }

  async function drawComparison() {
    const c = chrome();
    const series = [];
    for (const [i, mod] of m.models.entries()) {
      const f = await loadForecast(m, mod.id, state.win).catch(() => null);
      if (!f) continue;
      series.push({ name: mod.name, type: "line", data: f.mags.map((mg, k) => [mg, f.totals[k] > 0 ? f.totals[k] : null]), symbol: "circle", symbolSize: 5, lineStyle: { width: 2, color: seriesColor(i) }, itemStyle: { color: seriesColor(i) }, emphasis: { focus: "series" } });
    }
    chart.setOption({
      ...c,
      legend: { ...c.legend, data: series.map((s) => s.name) },
      tooltip: { ...c.tooltip, trigger: "axis", valueFormatter: (v) => (v === null ? "0" : fmt.sci(v)) },
      xAxis: { type: "value", name: "Magnitude", nameLocation: "middle", nameGap: 24, min: mags[0], max: mags[mags.length - 1], ...axis() },
      yAxis: { type: "log", logBase: 10, name: "Expected events", ...axis(), minorSplitLine: { show: false } },
      series,
    }, true);
  }

  function drawModelMeta() {
    const mod = m.models.find((x) => x.id === state.model);
    modelMeta.replaceChildren();
    const rows = [["Name", mod.name], ["Authors", mod.authors], ["DOI", doiBadge(mod.doi)], ["Type", mod.forecast_class === "CatalogForecastRepository" ? "Catalog-based" : "Gridded"], ["Path", mod.path ? el("code", {}, mod.path) : null], ["Source", mod.giturl ? el("a", { href: mod.giturl, target: "_blank", rel: "noopener" }, mod.giturl.replace(/^https?:\/\//, "")) : null], ["Commit", mod.git_hash ? el("code", {}, mod.git_hash) : null]];
    for (const [k, v] of rows) if (v) modelMeta.append(el("dt", {}, k), el("dd", {}, v));
  }

  async function refresh() {
    drawModelMeta();
    document.getElementById("cb-note").textContent = "loading…";
    fc = await loadForecast(m, state.model, state.win).catch((e) => { console.error(e); return null; });
    document.getElementById("cb-note").textContent = "";
    if (!fc) {
      layer.setData(null);
      summary.replaceChildren("No forecast for this model and window."); kpis.replaceChildren();
      return;
    }
    if (fc.mags.length !== mags.length) {
      magSlider.max = fc.mags.length - 1;
      if (state.k > fc.mags.length - 1) state.k = 0;
    }
    draw();
    drawComparison();
  }

  if (m.catalog && m.catalog.file) getJSON(m.catalog.file).then((c) => { catalog = c.testing; draw(); });
  await refresh();

  onTheme = () => { map.retheme(); drawOutline(); drawEvents(); drawComparison(); };
  window.addEventListener("fc-theme", onTheme);
}

export function destroy() {
  window.removeEventListener("fc-theme", onTheme);
  if (map) { map.remove(); map = null; }
  disposeChart(chart);
  chart = null;
  layer = events = outline = null;
}
