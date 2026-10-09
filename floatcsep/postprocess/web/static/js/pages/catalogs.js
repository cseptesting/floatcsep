import { el, panel, seg, select, field, fmt, getJSON, makeMap, outlineStyle, css, downloadCSV, debounce } from "../util.js";
import { makeChart, disposeChart, chartNode, chrome, axis } from "../charts.js";

let map = null, charts = [], onTheme = null, markers = null;

function dualRange(lo, hi, step, value, onChange) {
  const wrap = el("div", { class: "range" });
  const box = el("div", { class: "drange" });
  const fill = el("div", { class: "fill" });
  const a = el("input", { type: "range", min: lo, max: hi, step, value: value[0], "aria-label": "Minimum magnitude" });
  const b = el("input", { type: "range", min: lo, max: hi, step, value: value[1], "aria-label": "Maximum magnitude" });
  const lab = el("span", { class: "val", style: { minWidth: "6.5em" } });
  const update = (emit) => {
    let v0 = +a.value, v1 = +b.value;
    if (v0 > v1) { if (document.activeElement === a) { v1 = v0; b.value = v0; } else { v0 = v1; a.value = v1; } }
    const p0 = ((v0 - lo) / (hi - lo || 1)) * 100, p1 = ((v1 - lo) / (hi - lo || 1)) * 100;
    fill.style.left = p0 + "%";
    fill.style.width = p1 - p0 + "%";
    lab.textContent = `M ${v0.toFixed(1)} – ${v1.toFixed(1)}`;
    if (emit) onChange([v0, v1]);
  };
  a.addEventListener("input", () => update(true));
  b.addEventListener("input", () => update(true));
  box.append(el("div", { class: "track" }), fill, a, b);
  wrap.append(box, lab);
  wrap.reset = (v) => { a.min = b.min = lo = v[0]; a.max = b.max = hi = v[1]; a.value = v[0]; b.value = v[1]; update(false); };
  update(false);
  return wrap;
}

function gr(mags, bin) {
  if (!mags.length) return { x: [], inc: [], cum: [] };
  const lo = Math.floor(Math.min(...mags) / bin + 1e-6) * bin;
  const n = Math.round((Math.max(...mags) - lo) / bin) + 1;
  const inc = new Array(n).fill(0);
  for (const m of mags) inc[Math.min(n - 1, Math.max(0, Math.round((m - lo) / bin - 1e-6)))]++;
  const cum = [];
  let c = 0;
  for (let i = n - 1; i >= 0; i--) { c += inc[i]; cum[i] = c; }
  return { x: inc.map((_, i) => +(lo + i * bin).toFixed(2)), inc, cum };
}

function filterList(f, stored) {
  if (!f) return null;
  const row = (k, v) => (v === null || v === undefined ? null : el("li", {}, el("span", {}, k), el("span", {}, String(v))));
  const num = (v) => (v === null || v === undefined ? null : Number(v));
  const range = (a, b, u = "", d = 1) => {
    a = num(a); b = num(b);
    if (a === null && b === null) return "None";
    const f = (v) => (v === null ? "–" : d ? v.toFixed(d) : String(v));
    return `${f(a)} – ${f(b)}${u}`;
  };
  return el("ul", { class: "filters" },
    row("Region", f.region || "None"),
    row("Magnitude", range(f.mag_min, f.mag_max)),
    row("Depth", range(f.depth_min, f.depth_max, " km", 0)),
    f.start_date !== undefined ? row("Start", f.start_date ? fmt.iso(f.start_date) : "None") : null,
    row("Time", f.time),
    row("Stored in", stored));
}

export async function render(root, ctx) {
  const m = ctx.manifest, r = m.region || {};
  if (!m.catalog || !m.catalog.file) {
    root.append(el("div", { class: "empty span-12" }, "No catalog was exported for this experiment."));
    return;
  }
  const [data, grid] = await Promise.all([getJSON(m.catalog.file), ctx.gridPromise || Promise.resolve(null)]);
  const windows = m.time_windows.map((w, i) => ({ ...w, i, s: new Date(w.start).getTime(), e: new Date(w.end).getTime() }));
  const cats = { input: data.input, testing: data.testing };
  const hasInput = !!(cats.input && cats.input.n);
  const bin = r.mag_bin || 0.1;
  const LABEL = { input: "Input", testing: "Testing" };
  const colorOf = (k) => (k === "input" ? css("--series-1") : css("--accent"));

  // one magnitude scale for every view
  const allMags = [...(cats.testing ? cats.testing.mag : []), ...(hasInput ? cats.input.mag : [])];
  const magLo = allMags.length ? Math.floor(Math.min(...allMags) * 10) / 10 : 0;
  const magHi = allMags.length ? Math.ceil(Math.max(...allMags) * 10) / 10 : 1;
  const radius = (mag) => 3 + Math.max(0, mag - magLo) * 3;

  const state = { kind: hasInput ? "both" : "testing", win: "all", mag: [magLo, magHi] };
  const kindsShown = () => (state.kind === "both" ? ["input", "testing"] : [state.kind]);

  /** Events in view as {k, i} pairs, input first so testing is drawn on top. */
  function selection() {
    const [m0, m1] = state.mag;
    const w = state.win === "all" ? null : +state.win;
    const out = [];
    for (const k of kindsShown()) {
      const cat = cats[k];
      if (!cat) continue;
      for (let i = 0; i < cat.n; i++) {
        if (cat.mag[i] < m0 - 1e-9 || cat.mag[i] > m1 + 1e-9) continue;
        if (k === "testing") {
          if (w !== null && !cat.w[i].includes(w)) continue;
        } else {
          const end = windows[w === null ? 0 : w].s;
          if (cat.t[i] >= end) continue;
        }
        out.push({ k, i });
      }
    }
    return out;
  }

  // controls
  const kinds = hasInput ? [["both", "Both"], ["input", "Input"], ["testing", "Testing"]] : [["testing", "Testing"]];
  const winSel = el("select", { onchange: () => { state.win = winSel.value; update(); } });
  const winLabel = el("span", {}, "Time window");
  function fillWindows() {
    winSel.replaceChildren();
    const inputOnly = state.kind === "input";
    if (!inputOnly) winSel.append(el("option", { value: "all" }, "All windows"));
    for (const w of windows) {
      const lab = inputOnly ? `Window ${w.i + 1}: before ${fmt.iso(w.start)}` : `Window ${w.i + 1}: ${w.label.replace(" to ", " – ")}`;
      winSel.append(el("option", { value: String(w.i) }, lab));
    }
    if (inputOnly && state.win === "all") {
      const first = windows.find((w) => cats.input.t.some((t) => t < w.s));
      state.win = String(first ? first.i : windows.length - 1);
    }
    winSel.value = state.win;
    winLabel.textContent = "Time window";
  }
  const magCtl = dualRange(magLo, magHi, 0.1, state.mag, debounce((v) => { state.mag = v; update(); }, 60));
  const hint = el("span", { class: "hint" });
  const HINTS = {
    both: "Input events are drawn below testing events. Both use the same magnitude scale.",
    input: "Events available to the models before the window starts.",
    testing: "Events used to evaluate the forecasts.",
  };
  const kindSeg = seg(kinds, state.kind, (v) => {
    state.kind = v;
    fillWindows();
    update();
  }, false);
  fillWindows();
  const readout = el("div", { class: "readout", "aria-live": "polite" });
  const dl = el("button", { class: "btn", onclick: () => {
    const sel = selection();
    downloadCSV(`${m.experiment.name.replace(/\s+/g, "_")}_${state.kind}_catalog.csv`, ["catalog", "id", "origin_time", "latitude", "longitude", "depth", "magnitude"],
      sel.map(({ k, i }) => { const c = cats[k]; return [k, c.id[i], new Date(c.t[i]).toISOString(), c.lat[i], c.lon[i], c.depth[i], c.mag[i]]; }));
  } }, "Download CSV");

  const left = el("div", { class: "span-4 stack" });
  const right = el("div", { class: "span-8 stack" });
  root.append(left, right);
  left.append(panel("Catalogs", el("div", { class: "stack", style: { gap: "16px" } },
    el("div", { class: "group" }, field("Show", kindSeg), hint),
    el("div", { class: "group" }, el("label", { class: "field" }, winLabel, winSel), el("div", { class: "field" }, el("span", {}, "Magnitude"), magCtl)),
    el("div", { class: "group" }, readout, el("div", { class: "row" }, dl)),
  ), { class: "rail" }));
  left.lastChild.querySelector(".panel-body").style.display = "block";
  const filtersBox = el("div", { class: "stack", style: { gap: "12px" } });
  left.append(panel("Catalog Properties", filtersBox));
  const tableHead = el("tr");
  const tableBody = el("tbody");
  left.append(panel("Largest Events", el("div", { class: "table-scroll" }, el("table", { class: "data" },
    el("caption", { class: "visually-hidden" }, "Largest events in view. Select a row to show it on the map."),
    el("thead", {}, tableHead), tableBody)), { flush: true }));

  const mapNode = el("div", { class: "map", role: "region", "aria-label": "Map of catalog events" });
  const legend = el("div", { class: "map-legend", "aria-label": "Map legend" });
  const mapWrap = el("div", { class: "map-wrap" }, mapNode, legend);
  right.append(panel("Event Map", mapWrap, { flush: true }));
  const mtNode = chartNode(), grNode = chartNode();
  const mtPanel = panel("Magnitude–Time", mtNode, { tight: true });
  const grPanel = panel("Frequency–Magnitude", grNode, { tight: true });
  right.append(el("div", { class: "two" }, mtPanel, grPanel));
  const perNode = chartNode("short");
  const perPanel = panel("Events per Time Window", perNode, { tight: true });
  right.append(perPanel);

  // map
  map = makeMap(mapNode);
  if (r.bbox) map.fitBounds([[r.bbox[1], r.bbox[0]], [r.bbox[3], r.bbox[2]]], { padding: [10, 10] });
  else map.setView([0, 0], 2);
  let outline = null;
  const drawOutline = () => {
    if (outline) outline.remove();
    if (grid && grid.outline) outline = L.geoJSON(grid.outline, { style: { ...outlineStyle(), weight: 1.2, opacity: 0.8 } }).addTo(map);
  };
  drawOutline();
  const layerOf = new Map();

  function drawMarkers(sel) {
    if (markers) markers.remove();
    markers = L.layerGroup();
    layerOf.clear();
    for (const k of kindsShown()) {
      const cat = cats[k];
      const canvas = L.canvas({ padding: 0.3 });
      const items = sel.filter((x) => x.k === k).sort((a, b) => cat.mag[a.i] - cat.mag[b.i]);
      for (const { i } of items) {
        const mk = L.circleMarker([cat.lat[i], cat.lon[i]], {
          renderer: canvas, radius: radius(cat.mag[i]), color: k === "testing" ? "#ffffff" : css("--surface"), weight: k === "testing" ? 1 : 0.6,
          fillColor: colorOf(k), fillOpacity: k === "testing" ? 0.92 : 0.7,
        });
        const wins = k === "testing" ? `<br>Window ${cat.w[i].map((x) => x + 1).join(" ")}` : "";
        mk.bindPopup(`<b>M ${fmt.num(cat.mag[i], 1)}</b> ${LABEL[k]} catalog<br>${fmt.datetime(cat.t[i])} UTC<br>${fmt.lat(cat.lat[i])} ${fmt.lon(cat.lon[i])}<br>Depth ${fmt.num(cat.depth[i], 1)} km${wins}<br><span class="note">ID ${cat.id[i]}</span>`);
        mk.addTo(markers);
        layerOf.set(`${k}:${i}`, mk);
      }
    }
    markers.addTo(map);
    drawLegend();
  }

  function drawLegend() {
    const span = magHi - magLo;
    const step = span > 2.5 ? 1 : 0.5;
    let ticks = [];
    for (let v = Math.ceil(magLo / step - 1e-9) * step; v <= magHi + 1e-9; v += step) ticks.push(+v.toFixed(1));
    if (!ticks.length || ticks[0] - magLo > step / 2) ticks.unshift(magLo);
    while (ticks.length > 4) ticks = ticks.filter((_, j) => j % 2 === 0 || j === ticks.length - 1);
    const rmax = radius(ticks[ticks.length - 1]);
    const h = 2 * rmax + 4;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    let x = 0;
    const parts = [];
    for (const t of ticks) {
      const rr = radius(t), wd = Math.max(2 * rr + 8, 46);
      parts.push({ t, cx: x + wd / 2, rr });
      x += wd;
    }
    svg.setAttribute("width", String(x));
    svg.setAttribute("height", String(h + 16));
    svg.setAttribute("aria-hidden", "true");
    for (const { t, cx, rr } of parts) {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", cx); c.setAttribute("cy", h - rr); c.setAttribute("r", rr);
      c.setAttribute("fill", css("--accent")); c.setAttribute("fill-opacity", "0.25"); c.setAttribute("stroke", css("--ink-2")); c.setAttribute("stroke-width", "1");
      const tx = document.createElementNS(ns, "text");
      tx.setAttribute("x", cx); tx.setAttribute("y", h + 13); tx.setAttribute("text-anchor", "middle");
      tx.setAttribute("font-size", "11"); tx.setAttribute("fill", css("--ink-2"));
      tx.textContent = `M ${t.toFixed(1)}`;
      svg.append(c, tx);
    }
    const sw = (k) => el("span", {}, el("i", { class: "swatch", style: { background: colorOf(k), borderRadius: "50%", opacity: k === "input" ? 0.75 : 1 } }), `${LABEL[k]} catalog`);
    legend.replaceChildren(
      el("div", { class: "legend-kinds" }, ...kindsShown().slice().reverse().map(sw)),
      el("div", { class: "legend-sizes" }, el("span", { class: "note" }, "Magnitude"), svg),
      el("span", { class: "visually-hidden" }, `Marker size shows magnitude from M ${magLo.toFixed(1)} to M ${magHi.toFixed(1)}.`),
    );
  }

  const grc = makeChart(grNode), mt = makeChart(mtNode), per = makeChart(perNode);
  charts = [grc, mt, per];

  function drawCharts(sel) {
    const c = chrome();
    const ks = kindsShown();
    const nz = (v) => (v > 0 ? v : null);
    const grSeries = [];
    let x0 = magLo, x1 = magHi;
    for (const k of ks) {
      const d = gr(sel.filter((x) => x.k === k).map((x) => cats[k].mag[x.i]), bin);
      if (!d.x.length) continue;
      grSeries.push({ name: `${LABEL[k]} N(≥M)`, type: "line", step: "end", data: d.x.map((x, i) => [x, nz(d.cum[i])]), symbol: "none", lineStyle: { width: 2, color: colorOf(k) }, itemStyle: { color: colorOf(k) }, z: k === "testing" ? 3 : 2 });
      grSeries.push({ name: `${LABEL[k]} N(M)`, type: "scatter", data: d.x.map((x, i) => [x, nz(d.inc[i])]), symbolSize: 6, itemStyle: { color: colorOf(k), opacity: 0.55 }, z: k === "testing" ? 3 : 2 });
    }
    grc.setOption({
      ...c,
      legend: { ...c.legend, data: grSeries.filter((s) => s.type === "line").map((s) => s.name), left: 10, right: "auto" },
      tooltip: { ...c.tooltip, trigger: "axis", valueFormatter: (v) => (v === null ? "0" : v) },
      grid: { ...c.grid, left: 44, top: 34 },
      xAxis: { type: "value", name: "Magnitude", nameLocation: "middle", nameGap: 24, min: x0, max: x1, ...axis() },
      yAxis: { type: "log", logBase: 10, min: 0.8, ...axis(), minorSplitLine: { show: false } },
      series: grSeries,
    }, true);

    const w = state.win === "all" ? null : +state.win;
    const band = [];
    if (state.kind !== "input") {
      for (const win of w === null ? windows : [windows[w]]) band.push([{ xAxis: win.s }, { xAxis: win.e }]);
    } else {
      band.push([{ xAxis: windows[w === null ? 0 : w].s }, { xAxis: windows[windows.length - 1].e }]);
    }
    mt.setOption({
      ...c,
      legend: { show: false },
      grid: { ...c.grid, left: 40 },
      tooltip: { ...c.tooltip, trigger: "item", formatter: (p) => `${p.seriesName}<br>M ${fmt.num(p.value[1], 1)}<br>${fmt.datetime(p.value[0])}` },
      xAxis: { type: "time", ...axis() },
      yAxis: { type: "value", name: "M", min: Math.floor(magLo * 2) / 2, max: Math.ceil(magHi * 2) / 2, ...axis() },
      series: ks.map((k) => ({
        name: LABEL[k], type: "scatter", z: k === "testing" ? 3 : 2,
        data: sel.filter((x) => x.k === k).map(({ i }) => [cats[k].t[i], cats[k].mag[i]]),
        symbolSize: (v) => 2 * radius(v[1]) * 0.7,
        itemStyle: { color: colorOf(k), opacity: k === "testing" ? 0.9 : 0.55, borderColor: css("--surface"), borderWidth: 0.5 },
        markArea: k === ks[ks.length - 1] && !(state.kind !== "input" && w === null && windows.length > 1) ? { silent: true, itemStyle: { color: css("--accent-soft") }, data: band } : undefined,
      })),
    }, true);

    const many = windows.length > 30;
    const series = [];
    if (ks.includes("input")) {
      const ci = cats.input;
      series.push({ name: LABEL.input, counts: windows.map((win) => { let n = 0; for (let i = 0; i < ci.n; i++) if (ci.t[i] < win.s) n++; return n; }) });
    }
    if (ks.includes("testing")) {
      const ct = cats.testing;
      series.push({ name: LABEL.testing, counts: windows.map((win) => { let n = 0; for (let i = 0; i < ct.n; i++) if (ct.w[i].includes(win.i)) n++; return n; }) });
    }
    per.setOption({
      ...c,
      legend: { ...c.legend, data: series.map((s) => s.name), left: 10, right: "auto" },
      grid: { ...c.grid, left: 20, right: 30, top: 34, bottom: 10, containLabel: true },
      tooltip: { ...c.tooltip, trigger: "axis", formatter: (ps) => `${windows[ps[0].dataIndex].label}<br>` + ps.map((p) => `${p.seriesName}: <b>${p.value}</b>`).join("<br>") },
      xAxis: many
        ? { type: "category", data: windows.map((x) => x.end.slice(0, 10)), ...axis(), axisLabel: { ...axis().axisLabel, interval: Math.ceil(windows.length / 12) } }
        : { type: "category", data: windows.map((x) => x.label), ...axis(), axisLabel: { ...axis().axisLabel, rotate: windows.length > 6 ? 30 : 0, fontSize: 11 } },
      yAxis: { type: ks.length > 1 ? "log" : "value", minInterval: 1, min: ks.length > 1 ? 1 : 0, ...axis(), minorSplitLine: { show: false } },
      series: series.map((s) => {
        const k = s.name === LABEL.input ? "input" : "testing";
        return {
          name: s.name, type: many ? "line" : "bar", barMaxWidth: 26, symbol: "none", lineStyle: { width: 2, color: colorOf(k) }, itemStyle: { color: colorOf(k) },
          data: s.counts.map((v, i) => ({ value: ks.length > 1 && v === 0 ? null : v, itemStyle: { color: colorOf(k), opacity: w === null || w === i ? 1 : 0.35, borderRadius: [2, 2, 0, 0] } })),
        };
      }),
    }, true);
    per.off("click");
    per.on("click", (p) => {
      if (state.kind === "input") state.win = String(p.dataIndex);
      else state.win = state.win === String(p.dataIndex) ? "all" : String(p.dataIndex);
      winSel.value = state.win;
      update();
    });
  }

  function drawTable(sel) {
    const both = state.kind === "both";
    tableHead.replaceChildren(...[
      el("th", { scope: "col" }, "Origin time UTC"), el("th", { scope: "col", class: "num" }, "M"), el("th", { scope: "col", class: "num" }, "Depth km"),
      both ? el("th", { scope: "col" }, "Catalog") : null].filter(Boolean));
    const top = [...sel].sort((a, b) => cats[b.k].mag[b.i] - cats[a.k].mag[a.i]).slice(0, 12);
    tableBody.replaceChildren(...top.map(({ k, i }) => {
      const cat = cats[k];
      return el("tr", { class: "clickable", tabindex: "0", title: "Show on the map", onclick: () => focusEvent(k, i), onkeydown: (e) => { if (e.key === "Enter") focusEvent(k, i); } },
        el("td", {}, fmt.datetime(cat.t[i])), el("td", { class: "num" }, fmt.num(cat.mag[i], 1)), el("td", { class: "num" }, fmt.num(cat.depth[i], 0)),
        both ? el("td", {}, el("span", { class: "swatch", style: { background: colorOf(k), borderRadius: "50%", marginRight: "6px" } }), LABEL[k]) : null);
    }));
    if (!top.length) tableBody.append(el("tr", {}, el("td", { colspan: both ? 4 : 3, class: "empty" }, "No events match these filters.")));
  }

  function focusEvent(k, i) {
    const cat = cats[k];
    map.setView([cat.lat[i], cat.lon[i]], Math.max(map.getZoom(), 8));
    const mk = layerOf.get(`${k}:${i}`);
    if (mk) mk.openPopup();
  }

  function drawReadout(sel) {
    const rows = kindsShown().slice().reverse().map((k) => {
      const ms = sel.filter((x) => x.k === k).map((x) => cats[k].mag[x.i]);
      return el("tr", {},
        el("td", {}, el("span", { class: "swatch", style: { background: colorOf(k) } }), LABEL[k]),
        el("td", {}, fmt.int(ms.length)),
        el("td", {}, ms.length ? fmt.num(Math.max(...ms), 1) : "–"));
    });
    readout.replaceChildren(el("table", { class: "mini" },
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Catalog"), el("th", { scope: "col" }, "Events"), el("th", { scope: "col" }, "M max"))),
      el("tbody", {}, rows)));
  }

  function drawFilters() {
    filtersBox.replaceChildren(...kindsShown().slice().reverse().map((k) => el("div", {},
      el("h4", { style: { margin: "0 0 4px", fontSize: "13px", display: "flex", alignItems: "center", gap: "6px" } }, el("span", { class: "swatch", style: { background: colorOf(k), borderRadius: "50%" } }), `${LABEL[k]} catalog`),
      filterList(cats[k].filters, cats[k].stored) || el("p", { class: "note" }, "No filter information."))));
  }

  function update() {
    hint.textContent = HINTS[state.kind];
    const sel = selection();
    drawFilters();
    drawMarkers(sel);
    drawCharts(sel);
    drawTable(sel);
    drawReadout(sel);
  }
  update();

  onTheme = () => { drawOutline(); update(); };
  window.addEventListener("fc-theme", onTheme);
}

export function destroy() {
  window.removeEventListener("fc-theme", onTheme);
  if (map) { map.remove(); map = null; }
  charts.forEach(disposeChart);
  charts = [];
  markers = null;
}
