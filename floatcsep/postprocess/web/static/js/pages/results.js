import { el, panel, seg, select, field, fmt, getJSON, css, seriesColor, poissonPmf, poissonQuantiles, percentile } from "../util.js";
import { makeChart, disposeChart, chartNode, chrome, axis } from "../charts.js";

let charts = [], onTheme = null;

const TYPE_LABEL = { consistency: "Consistency", comparative: "Comparative", sequential: "Sequential", sequential_comparative: "Sequential comparative", batch: "Batch" };

/** Pass/fail of a consistency result at level alpha; null when it cannot be decided. */
function verdict(rec, alpha) {
  const q = rec.quantile;
  if (Array.isArray(q)) {
    const v = q.filter((x) => typeof x === "number" && Number.isFinite(x));
    if (!v.length) return null;
    return Math.min(...v) >= alpha / 2;
  }
  if (typeof q === "number" && Number.isFinite(q)) return q >= alpha;
  return null;
}

/** [low, q25, median, q75, high] of the test distribution, for the range bars. */
function rangeOf(rec, alpha) {
  if (rec.dist && rec.dist.type === "poisson") {
    const lam = rec.dist.params[0];
    const [lo, q25, med, q75, hi] = poissonQuantiles(lam, [alpha / 2, 0.25, 0.5, 0.75, 1 - alpha / 2]);
    return { lo, q25, med, q75, hi, twoSided: true, lambda: lam };
  }
  if (rec.dist_q) {
    const [p2, p5, q25, med, q75, p95, p97] = rec.dist_q;
    const twoSided = Array.isArray(rec.quantile);
    return twoSided ? { lo: p2, q25, med, q75, hi: p97, twoSided } : { lo: p5, q25, med, q75, hi: Math.max(p97, med), twoSided, oneSided: true };
  }
  return null;
}

export async function render(root, ctx) {
  const m = ctx.manifest;
  if (!m.results || !m.results.index) {
    root.append(el("div", { class: "empty span-12" }, "No results were exported for this experiment."));
    return;
  }
  const index = await getJSON(m.results.index);
  const windows = m.time_windows, models = m.models;
  const mName = Object.fromEntries(models.map((x) => [x.id, x.name]));
  const mIdx = Object.fromEntries(models.map((x, i) => [x.id, i]));
  const wIdx = Object.fromEntries(windows.map((w, i) => [w.id, i]));
  const tests = m.tests.filter((t) => index.some((r) => r.test === t.id));
  if (!tests.length) {
    root.append(el("div", { class: "empty span-12" }, "The results folder has no evaluation files yet."));
    return;
  }
  const state = { test: tests[0].id, win: windows[windows.length - 1].id, alpha: 0.05, view: "window", figure: false, pick: null };
  const testOf = () => tests.find((t) => t.id === state.test);
  const recs = () => index.filter((r) => r.test === state.test);
  const winRecs = () => recs().filter((r) => r.window === state.win);

  // controls
  const testSel = select(tests.map((t) => [t.id, t.name]), state.test, (v) => { state.test = v; state.pick = null; update(); });
  const winSel = select(windows.map((w) => [w.id, w.label]), state.win, (v) => { state.win = v; update(); });
  const step = (d) => { const i = wIdx[state.win] + d; if (i >= 0 && i < windows.length) { state.win = windows[i].id; winSel.value = state.win; update(); } };
  const winRow = el("div", { class: "stepper" }, el("button", { class: "btn", onclick: () => step(-1), title: "Previous window", "aria-label": "Previous window" }, "‹"), winSel, el("button", { class: "btn", onclick: () => step(1), title: "Next window", "aria-label": "Next window" }, "›"));
  const viewSeg = seg([["window", "Single window"], ["time", "All windows"]], state.view, (v) => { state.view = v; update(); }, false);
  const alphaSel = select([["0.01", "0.01"], ["0.05", "0.05"], ["0.1", "0.10"]], "0.05", (v) => { state.alpha = +v; update(); });
  const figCb = el("label", { class: "checkbox" }, el("input", { type: "checkbox", onchange: (e) => { state.figure = e.target.checked; update(); } }), "Show pyCSEP figure");
  const winField = field("Time window", winRow);
  const left = el("div", { class: "span-4 stack" });
  const right = el("div", { class: "span-8 stack" });
  root.append(left, right);
  left.append(panel("Tests", el("div", { class: "stack", style: { gap: "16px" } },
    el("div", { class: "group" }, field("Test", testSel), windows.length > 1 ? field("Show", viewSeg) : null, winField),
    el("div", { class: "group" }, field("Significance level α", alphaSel), m.figures ? figCb : null),
  ), { class: "rail" }));
  const tableWrap = el("div", { class: "table-scroll" });
  left.append(panel("Scores", tableWrap, { flush: true }));
  const testInfo = el("dl", { class: "props" });
  left.append(panel("Test Definition", testInfo));

  const mainNode = chartNode("tall");
  const mainPanel = panel("", mainNode, { tight: true });
  const mainTitle = mainPanel.querySelector("h2");
  const mainSub = el("span", { class: "sub" });
  mainPanel.querySelector(".panel-head").append(mainSub);
  right.append(mainPanel);
  const detailNode = chartNode();
  const detailPanel = panel("Test Distribution", detailNode, { tight: true });
  const detailSub = el("span", { class: "sub" });
  detailPanel.querySelector(".panel-head").append(detailSub);
  right.append(detailPanel);
  const figPanel = panel("pyCSEP Figure", el("div", { class: "fig-body" }), {});
  right.append(figPanel);
  const legend = el("div", { class: "legend", style: { padding: "0 14px 10px" } });
  mainPanel.append(legend);

  const main = makeChart(mainNode), detail = makeChart(detailNode);
  charts = [main, detail];

  function testInfoDraw() {
    const t = testOf();
    testInfo.replaceChildren();
    for (const [k, v] of [["Type", TYPE_LABEL[t.type] || t.type], ["Function", t.func ? el("code", {}, t.func) : null], ["Reference", t.ref_model], ["Arguments", t.func_kwargs && Object.keys(t.func_kwargs).length ? el("code", {}, JSON.stringify(t.func_kwargs)) : null]]) {
      if (v) testInfo.append(el("dt", {}, k), el("dd", {}, v));
    }
  }

  function statusEl(ok) {
    return el("span", { class: "status " + (ok === null ? "na" : ok ? "pass" : "fail") }, ok === null ? "–" : ok ? "Pass" : "Fail");
  }

  function consistencyWindow(t) {
    const rows = winRecs().sort((a, b) => mIdx[a.model] - mIdx[b.model]);
    const c = chrome();
    const cats = rows.map((r) => mName[r.model]);
    const ranges = rows.map((r) => rangeOf(r, state.alpha));
    const ok = rows.map((r) => verdict(r, state.alpha));
    const good = css("--good"), bad = css("--bad"), ink2 = css("--ink-2"), ink3 = css("--ink-3");
    const whisker = rows.map((r, i) => ranges[i] ? [ranges[i].lo, ranges[i].hi, i] : null).filter(Boolean);
    const box = rows.map((r, i) => ranges[i] ? [ranges[i].q25, ranges[i].q75, i] : null).filter(Boolean);
    const obs = rows.map((r, i) => ({ value: [r.observed_statistic, i], itemStyle: { color: ok[i] === null ? ink3 : ok[i] ? good : bad } }));
    const oneSided = ranges.some((x) => x && x.oneSided);
    mainTitle.textContent = t.name;
    mainSub.textContent = `Window ${wIdx[state.win] + 1}: ${windows[wIdx[state.win]].label.replace(" to ", " – ")}`;
    main.setOption({
      ...c,
      legend: { show: false },
      grid: { left: 10, right: 24, top: 16, bottom: 40, containLabel: true },
      tooltip: { ...c.tooltip, trigger: "item", formatter: (p) => {
        const i = p.data.value ? p.data.value[1] : p.data[2];
        const r = rows[i], g = ranges[i];
        const q = Array.isArray(r.quantile) ? r.quantile.map((x) => fmt.num(x, 3)).join(" / ") : fmt.num(r.quantile, 3);
        return `<b>${mName[r.model]}</b><br>Observed ${fmt.num(r.observed_statistic, 2)}<br>Quantile ${q}` + (g ? `<br>${Math.round(100 * (1 - state.alpha))}% interval ${fmt.num(g.lo, 2)} – ${fmt.num(g.hi, 2)}` : "");
      } },
      xAxis: { type: "value", ...axis(), scale: true },
      yAxis: { type: "category", data: cats, inverse: true, ...axis(), axisLine: { show: false }, splitLine: { show: false }, axisLabel: { ...axis().axisLabel, color: css("--ink") } },
      series: [
        { type: "custom", name: "range", data: whisker, encode: { x: [0, 1], y: 2 }, renderItem: (p, api) => {
          const a = api.coord([api.value(0), api.value(2)]), b = api.coord([api.value(1), api.value(2)]);
          return { type: "group", children: [
            { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: ink2, lineWidth: 1.5 } },
            { type: "line", shape: { x1: a[0], y1: a[1] - 6, x2: a[0], y2: a[1] + 6 }, style: { stroke: ink2, lineWidth: 1.5 } },
            { type: "line", shape: { x1: b[0], y1: b[1] - 6, x2: b[0], y2: b[1] + 6 }, style: { stroke: ink2, lineWidth: 1.5 } },
          ] };
        } },
        { type: "custom", name: "iqr", data: box, encode: { x: [0, 1], y: 2 }, renderItem: (p, api) => {
          const a = api.coord([api.value(0), api.value(2)]), b = api.coord([api.value(1), api.value(2)]);
          return { type: "rect", shape: { x: a[0], y: a[1] - 4, width: Math.max(1, b[0] - a[0]), height: 8, r: 2 }, style: { fill: ink2, opacity: 0.45 } };
        } },
        { type: "scatter", name: "observed", data: obs, symbolSize: 12, itemStyle: { borderColor: css("--surface"), borderWidth: 1.5 }, z: 10 },
      ],
    }, true);
    legend.replaceChildren(
      el("span", {}, el("i", { class: "swatch", style: { background: good, borderRadius: "50%" } }), "Pass"),
      el("span", {}, el("i", { class: "swatch", style: { background: bad, borderRadius: "50%" } }), "Fail"),
      el("span", {}, el("i", { class: "swatch", style: { background: ink2, opacity: 0.45, width: "16px" } }), "50% interval"),
      el("span", {}, el("i", { class: "swatch", style: { background: ink2, height: "2px", width: "16px" } }), oneSided ? `${Math.round(100 * (1 - state.alpha))}% interval, one-sided` : `${Math.round(100 * (1 - state.alpha))}% interval`),
    );
    main.off("click");
    main.on("click", (p) => { const i = p.data.value ? p.data.value[1] : p.data[2]; state.pick = rows[i] && rows[i].model; detailDraw(); });
    table(rows.map((r, i) => [mName[r.model], fmt.num(r.observed_statistic, 2), Array.isArray(r.quantile) ? r.quantile.map((x) => fmt.num(x, 3)).join(" / ") : fmt.num(r.quantile, 3), statusEl(ok[i])]), ["Model", "Statistic", "Quantile", "Result"], rows.map((r) => r.model));
    if (!state.pick || !rows.some((r) => r.model === state.pick)) state.pick = rows.length ? rows[0].model : null;
  }

  async function detailDraw() {
    const t = testOf();
    const rec = winRecs().find((r) => r.model === state.pick);
    if (!rec || t.type !== "consistency" || state.view !== "window") { detailPanel.style.display = "none"; return; }
    detailPanel.style.display = "";
    const c = chrome();
    const ok = verdict(rec, state.alpha);
    const obsColor = ok === null ? css("--ink-3") : ok ? css("--good") : css("--bad");
    detailSub.textContent = mName[rec.model];
    let xs = [], ys = [], xType = "category", yName = "";
    if (rec.dist && rec.dist.type === "poisson") {
      const lam = rec.dist.params[0];
      const kmax = Math.ceil(Math.max(lam + 4 * Math.sqrt(lam) + 5, rec.observed_statistic + 2));
      const pmf = poissonPmf(lam, kmax);
      xs = pmf.map((_, k) => k);
      ys = pmf;
      yName = "Probability";
    } else if (rec.dist_file) {
      const d = await getJSON(rec.dist_file);
      const vals = d.test_distribution.filter((v) => typeof v === "number" && Number.isFinite(v)).sort((a, b) => a - b);
      const lo = Math.min(vals[0], rec.observed_statistic), hi = Math.max(vals[vals.length - 1], rec.observed_statistic);
      const nb = Math.min(60, Math.max(15, Math.round(Math.sqrt(vals.length))));
      const w = (hi - lo) / nb || 1;
      const counts = new Array(nb).fill(0);
      for (const v of vals) counts[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
      xs = counts.map((_, i) => +(lo + (i + 0.5) * w).toFixed(4));
      ys = counts;
      yName = "Simulations";
    } else { detailPanel.style.display = "none"; return; }
    const obsIdx = xs.reduce((best, x, i) => (Math.abs(x - rec.observed_statistic) < Math.abs(xs[best] - rec.observed_statistic) ? i : best), 0);
    detail.setOption({
      ...c,
      legend: { show: false },
      grid: { ...c.grid, left: 56 },
      tooltip: { ...c.tooltip, trigger: "axis", valueFormatter: (v) => fmt.num(v, yName === "Probability" ? 4 : 0) },
      xAxis: { type: "category", data: xs.map((x) => (typeof x === "number" ? fmt.num(x, yName === "Probability" ? 0 : 1) : x)), name: "Statistic", nameLocation: "middle", nameGap: 24, ...axis(), axisLabel: { ...axis().axisLabel, interval: Math.max(0, Math.ceil(xs.length / 12) - 1) } },
      yAxis: { type: "value", name: yName, ...axis() },
      series: [{
        type: "bar", data: ys, barCategoryGap: "10%", itemStyle: { color: css("--series-1"), borderRadius: [2, 2, 0, 0] },
        markLine: { silent: true, symbol: "none", lineStyle: { color: obsColor, width: 2 }, label: { color: obsColor, formatter: `Observed ${fmt.num(rec.observed_statistic, 2)}`, position: "insideEndTop", rotate: 0, distance: [0, 4] }, data: [{ xAxis: obsIdx }] },
      }],
    }, true);
  }

  function consistencyTime(t) {
    const c = chrome();
    const rows = recs();
    const series = models.map((mod, i) => {
      const data = windows.map((w) => {
        const r = rows.find((x) => x.model === mod.id && x.window === w.id);
        if (!r) return null;
        const q = Array.isArray(r.quantile) ? Math.min(...r.quantile.filter((x) => Number.isFinite(x))) : r.quantile;
        return Number.isFinite(q) ? q : null;
      });
      return data.some((v) => v !== null) ? { name: mod.name, type: "line", data, symbol: "circle", symbolSize: 7, lineStyle: { width: 2, color: seriesColor(i) }, itemStyle: { color: seriesColor(i) }, connectNulls: false, emphasis: { focus: "series" } } : null;
    }).filter(Boolean);
    mainTitle.textContent = `${t.name} Quantile Score`;
    const twoSided = rows.some((r) => Array.isArray(r.quantile));
    const thr = twoSided ? state.alpha / 2 : state.alpha;
    mainSub.textContent = twoSided ? "Lower of the two quantiles. Dashed line at α/2" : "Dashed line at α";
    main.setOption({
      ...c,
      legend: { ...c.legend, data: series.map((s) => s.name) },
      grid: { ...c.grid, top: 36, bottom: windows.length > 6 ? 70 : 40 },
      tooltip: { ...c.tooltip, trigger: "axis", valueFormatter: (v) => (v === null ? "–" : fmt.num(v, 3)) },
      xAxis: { type: "category", data: windows.map((w) => (windows.length > 12 ? w.end.slice(0, 10) : w.label)), ...axis(), axisLabel: { ...axis().axisLabel, rotate: windows.length > 6 ? 30 : 0, interval: windows.length > 30 ? Math.ceil(windows.length / 15) : 0 } },
      yAxis: { type: "value", name: "Quantile", min: 0, max: 1, ...axis() },
      series: [...series, { type: "line", name: "α", data: [], markLine: { silent: true, symbol: "none", lineStyle: { color: css("--bad"), type: "dashed" }, label: { color: css("--bad"), formatter: twoSided ? `α/2 = ${thr}` : `α = ${thr}` }, data: [{ yAxis: thr }] } }],
    }, true);
    legend.replaceChildren();
    main.off("click");
    table(windows.map((w, j) => [`${j + 1}`, ...models.map((mod) => { const r = rows.find((x) => x.model === mod.id && x.window === w.id); return r ? statusEl(verdict(r, state.alpha)) : "–"; })]), ["Window", ...models.map((x) => x.name)]);
  }

  function comparativeWindow(t) {
    const rows = winRecs().sort((a, b) => mIdx[a.model] - mIdx[b.model]);
    const c = chrome();
    const good = css("--good"), bad = css("--bad"), ink2 = css("--ink-2"), ink3 = css("--ink-3");
    const sig = (r) => (r.ci && Number.isFinite(r.ci[0]) && Number.isFinite(r.ci[1]) ? (r.ci[0] > 0 ? 1 : r.ci[1] < 0 ? -1 : 0) : null);
    mainTitle.textContent = t.name;
    mainSub.textContent = `Window ${wIdx[state.win] + 1}. Reference ${t.ref_model || "–"}`;
    main.setOption({
      ...c,
      legend: { show: false },
      grid: { left: 10, right: 24, top: 16, bottom: 40, containLabel: true },
      tooltip: { ...c.tooltip, trigger: "item", formatter: (p) => { const r = rows[p.data.value ? p.data.value[1] : p.data[2]]; return `<b>${mName[r.model]}</b><br>IG ${fmt.num(r.observed_statistic, 3)}` + (r.ci ? `<br>${Math.round(100 * (1 - state.alpha))}% CI ${fmt.num(r.ci[0], 3)} – ${fmt.num(r.ci[1], 3)}` : ""); } },
      xAxis: { type: "value", name: "Information gain per event", nameLocation: "middle", nameGap: 26, ...axis(), scale: true },
      yAxis: { type: "category", data: rows.map((r) => mName[r.model] + (mName[r.model] === t.ref_model ? " (ref)" : "")), inverse: true, ...axis(), axisLine: { show: false }, splitLine: { show: false }, axisLabel: { ...axis().axisLabel, color: css("--ink") } },
      series: [
        { type: "custom", data: rows.map((r, i) => (r.ci && Number.isFinite(r.ci[0]) && Number.isFinite(r.ci[1]) ? [r.ci[0], r.ci[1], i] : null)).filter(Boolean), encode: { x: [0, 1], y: 2 }, renderItem: (p, api) => {
          const a = api.coord([api.value(0), api.value(2)]), b = api.coord([api.value(1), api.value(2)]);
          return { type: "group", children: [
            { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: ink2, lineWidth: 1.5 } },
            { type: "line", shape: { x1: a[0], y1: a[1] - 6, x2: a[0], y2: a[1] + 6 }, style: { stroke: ink2, lineWidth: 1.5 } },
            { type: "line", shape: { x1: b[0], y1: b[1] - 6, x2: b[0], y2: b[1] + 6 }, style: { stroke: ink2, lineWidth: 1.5 } },
          ] };
        } },
        { type: "scatter", data: rows.map((r, i) => ({ value: [Number.isFinite(r.observed_statistic) ? r.observed_statistic : null, i], itemStyle: { color: sig(r) === 1 ? good : sig(r) === -1 ? bad : sig(r) === 0 ? ink2 : ink3 } })), symbolSize: 12, itemStyle: { borderColor: css("--surface"), borderWidth: 1.5 }, z: 10,
          markLine: { silent: true, symbol: "none", lineStyle: { color: ink3, type: "dashed" }, data: [{ xAxis: 0 }] } },
      ],
    }, true);
    legend.replaceChildren(
      el("span", {}, el("i", { class: "swatch", style: { background: good, borderRadius: "50%" } }), "Better than reference"),
      el("span", {}, el("i", { class: "swatch", style: { background: bad, borderRadius: "50%" } }), "Worse than reference"),
      el("span", {}, el("i", { class: "swatch", style: { background: ink2, borderRadius: "50%" } }), "No significant difference"),
      el("span", {}, el("i", { class: "swatch", style: { background: ink2, height: "2px", width: "16px" } }), `${Math.round(100 * (1 - state.alpha))}% confidence interval`),
    );
    main.off("click");
    table(rows.map((r) => [mName[r.model], fmt.num(r.observed_statistic, 3), r.ci ? `${fmt.num(r.ci[0], 3)} – ${fmt.num(r.ci[1], 3)}` : "–", sig(r) === 1 ? el("span", { class: "status pass" }, "Better") : sig(r) === -1 ? el("span", { class: "status fail" }, "Worse") : el("span", { class: "status na" }, sig(r) === 0 ? "Equal" : "–")]), ["Model", "IG", "CI", "Result"]);
  }

  function linesOverWindows(t, title, sub, valueOf, yName) {
    const c = chrome();
    const series = models.map((mod, i) => {
      const data = windows.map((w) => valueOf(mod.id, w.id));
      return data.some((v) => v !== null && v !== undefined) ? { name: mod.name, type: "line", data, symbol: "circle", symbolSize: 6, lineStyle: { width: 2, color: seriesColor(i) }, itemStyle: { color: seriesColor(i) }, emphasis: { focus: "series" } } : null;
    }).filter(Boolean);
    mainTitle.textContent = title;
    mainSub.textContent = sub;
    main.setOption({
      ...c,
      legend: { ...c.legend, data: series.map((s) => s.name) },
      grid: { ...c.grid, top: 36, bottom: windows.length > 6 ? 70 : 40, left: 64 },
      tooltip: { ...c.tooltip, trigger: "axis", valueFormatter: (v) => (v === null || v === undefined ? "–" : fmt.num(v, 3)) },
      xAxis: { type: "category", data: windows.map((w) => (windows.length > 12 ? w.end.slice(0, 10) : w.label)), ...axis(), axisLabel: { ...axis().axisLabel, rotate: windows.length > 6 ? 30 : 0, interval: windows.length > 30 ? Math.ceil(windows.length / 15) : 0 } },
      yAxis: { type: "value", name: yName, scale: true, ...axis() },
      series: [...series, { type: "line", data: [], markLine: { silent: true, symbol: "none", lineStyle: { color: css("--ink-3"), type: "dashed" }, data: [{ yAxis: 0 }] } }],
    }, true);
    legend.replaceChildren();
    main.off("click");
    table(windows.map((w, j) => [`${j + 1}`, ...models.map((mod) => fmt.num(valueOf(mod.id, w.id), 2))]), ["Window", ...models.map((x) => x.name)]);
  }

  function sequential(t) {
    const rows = recs();
    const best = {};
    for (const r of rows) {
      const n = Array.isArray(r.observed_statistic) ? r.observed_statistic.length : 0;
      if (!best[r.model] || n > best[r.model].n) best[r.model] = { n, r };
    }
    const valueOf = (mid, wid) => { const b = best[mid]; const i = wIdx[wid]; const v = b && b.r.observed_statistic[i]; return Number.isFinite(v) ? v : null; };
    linesOverWindows(t, t.type === "sequential_comparative" ? "Cumulative Information Gain" : "Cumulative Log-Likelihood", t.ref_model ? `Reference ${t.ref_model}` : t.name, valueOf, t.type === "sequential_comparative" ? "Information gain" : "Log-likelihood");
  }

  function batch(t) {
    const rows = winRecs().filter((r) => Array.isArray(r.observed_statistic));
    const refs = t.ref_models && t.ref_models.length ? t.ref_models : models.map((x) => x.name);
    if (!rows.length) { generic(t); return; }
    const score = (r) => r.observed_statistic.filter(Number.isFinite).reduce((a, b) => a + b, 0) / r.observed_statistic.length;
    rows.sort((a, b) => score(b) - score(a));
    const yNames = rows.map((r) => mName[r.model]);
    const cells = [];
    let amax = 0;
    rows.forEach((r, yi) => {
      const tq = Array.isArray(r.quantile) ? r.quantile[0] || [] : [];
      const wq = Array.isArray(r.quantile) ? r.quantile[1] || [] : [];
      r.observed_statistic.forEach((v, xi) => {
        if (!Number.isFinite(v) || mName[r.model] === refs[xi]) return;
        amax = Math.max(amax, Math.abs(v));
        cells.push({ xi, yi, v, t: Number.isFinite(tq[xi]) && tq[xi] > 0, w: Number.isFinite(wq[xi]) && wq[xi] < state.alpha });
      });
    });
    amax = amax || 1;
    const pos = css("--series-1"), neg = css("--series-8"), mid = css("--surface-2"), ink = css("--ink");
    const mix = (v) => {
      const f = Math.min(1, Math.abs(v) / amax);
      return { color: v >= 0 ? pos : neg, opacity: 0.15 + 0.85 * f };
    };
    mainTitle.textContent = t.name;
    mainSub.textContent = `Window ${wIdx[state.win] + 1}. Information gain of each model (row) over each reference (column)`;
    const c = chrome();
    main.setOption({
      ...c,
      legend: { show: false },
      grid: { left: 10, right: 20, top: 10, bottom: 60, containLabel: true },
      tooltip: { ...c.tooltip, trigger: "item", formatter: (p) => {
        const d = cells[p.dataIndex];
        return `<b>${yNames[d.yi]}</b> over <b>${refs[d.xi]}</b><br>IG ${fmt.num(d.v, 3)}<br>T-test ${d.t ? "significant" : "not significant"}<br>W-test ${d.w ? "significant" : "not significant"}`;
      } },
      xAxis: { type: "category", data: refs, ...axis(), splitLine: { show: false }, axisLabel: { ...axis().axisLabel, rotate: refs.length > 5 ? 30 : 0, interval: 0 } },
      yAxis: { type: "category", data: yNames, inverse: true, ...axis(), splitLine: { show: false }, axisLabel: { ...axis().axisLabel, color: ink } },
      series: [{
        type: "custom",
        data: cells.map((d) => [d.xi, d.yi, d.v]),
        renderItem: (params, api) => {
          const d = cells[params.dataIndex];
          const p = api.coord([d.xi, d.yi]);
          const sz = api.size([1, 1]);
          const w = sz[0] - 3, h = sz[1] - 3;
          const st = mix(d.v);
          const kids = [{ type: "rect", shape: { x: p[0] - w / 2, y: p[1] - h / 2, width: w, height: h, r: 3 }, style: { fill: st.color, opacity: st.opacity } }];
          if (d.t) kids.push({ type: "rect", shape: { x: p[0] - w / 2 + 1, y: p[1] - h / 2 + 1, width: w - 2, height: h - 2, r: 3 }, style: { fill: "none", stroke: ink, lineWidth: 2 } });
          kids.push({ type: "text", style: { x: p[0], y: p[1], text: fmt.num(d.v, 2) + (d.w ? "" : "*"), fill: ink, align: "center", verticalAlign: "middle", fontSize: 12, fontWeight: d.t ? 700 : 400 } });
          return { type: "group", children: kids };
        },
      }],
    }, true);
    legend.replaceChildren(
      el("span", {}, el("i", { class: "swatch", style: { background: pos } }), "Row model better"),
      el("span", {}, el("i", { class: "swatch", style: { background: neg } }), "Reference better"),
      el("span", {}, el("i", { class: "swatch", style: { background: "transparent", border: `2px solid ${ink}` } }), "T-test significant"),
      el("span", {}, "* W-test not significant"),
    );
    main.off("click");
    table(rows.map((r) => [mName[r.model], fmt.num(score(r), 3)]), ["Model", "Mean IG"]);
  }

  function generic(t) {
    const rows = winRecs();
    mainTitle.textContent = t.name;
    mainSub.textContent = "No interactive plot for this test type";
    main.clear();
    legend.replaceChildren();
    table(rows.map((r) => [mName[r.model], Array.isArray(r.observed_statistic) ? r.observed_statistic.map((v) => fmt.num(v, 2)).join(", ") : fmt.num(r.observed_statistic, 3), Array.isArray(r.quantile) ? r.quantile.map((v) => fmt.num(v, 3)).join(", ") : fmt.num(r.quantile, 3), r.status || ""]), ["Model", "Statistic", "Quantile", "Status"]);
  }

  function table(rows, header, keys) {
    const body = el("tbody", {}, rows.map((r, i) => el("tr", { class: keys ? "clickable" + (keys[i] === state.pick ? " selected" : "") : "", onclick: keys ? () => { state.pick = keys[i]; detailDraw(); table(rows, header, keys); } : null },
      r.map((v, j) => el("td", { class: j > 0 && typeof v === "string" && /^[-–\d]/.test(v) ? "num" : "" }, v)))));
    tableWrap.replaceChildren(el("table", { class: "data" }, el("thead", {}, el("tr", {}, header.map((h, j) => el("th", { class: j > 0 ? "num" : "" }, h)))), body));
  }

  function figureDraw() {
    const body = figPanel.querySelector(".fig-body");
    body.replaceChildren();
    if (!state.figure || !m.figures) { figPanel.style.display = "none"; return; }
    const wf = (m.figures.windows || {})[state.win] || {};
    const srcs = [];
    if (wf.tests && wf.tests[state.test]) srcs.push([`${testOf().name}`, wf.tests[state.test]]);
    for (const mod of models) if (wf.tests && wf.tests[`${state.test}:${mod.id}`]) srcs.push([mod.name, wf.tests[`${state.test}:${mod.id}`]]);
    if (!srcs.length) { body.append(el("div", { class: "empty" }, "No figure for this test and window.")); figPanel.style.display = ""; return; }
    body.append(...srcs.map(([label, src]) => el("figure", { style: { margin: "0 0 10px" } }, el("img", { class: "figure", src, alt: label }), el("figcaption", { class: "note", style: { textAlign: "center" } }, label))));
    figPanel.style.display = "";
  }

  function update() {
    const t = testOf();
    testInfoDraw();
    const seqLike = t.type === "sequential" || t.type === "sequential_comparative";
    const single = seqLike || t.type === "batch";
    winField.style.display = seqLike || (state.view === "time" && t.type !== "batch") ? "none" : "";
    if (viewSeg.parentElement) viewSeg.parentElement.style.display = single ? "none" : "";
    if (seqLike) sequential(t);
    else if (t.type === "consistency") (state.view === "time" && windows.length > 1 ? consistencyTime : consistencyWindow)(t);
    else if (t.type === "comparative") {
      if (state.view === "time" && windows.length > 1) linesOverWindows(t, "Information Gain per Window", `Reference ${t.ref_model || "–"}`, (mid, wid) => { const r = recs().find((x) => x.model === mid && x.window === wid); return r && Number.isFinite(r.observed_statistic) ? r.observed_statistic : null; }, "Information gain");
      else comparativeWindow(t);
    } else if (t.type === "batch") batch(t);
    else generic(t);
    detailDraw();
    figureDraw();
  }
  update();

  onTheme = () => update();
  window.addEventListener("fc-theme", onTheme);
}

export function destroy() {
  window.removeEventListener("fc-theme", onTheme);
  charts.forEach(disposeChart);
  charts = [];
}
