import { css, el } from "./util.js";

export function chrome() {
  const ink = css("--ink"), ink2 = css("--ink-2"), grid = css("--grid"), line = css("--line-2");
  return {
    textStyle: { fontFamily: "Noto Sans, Segoe UI, system-ui, sans-serif", color: ink2, fontSize: 12 },
    animationDuration: 250,
    useUTC: true,
    grid: { left: 52, right: 18, top: 28, bottom: 36, containLabel: false },
    tooltip: {
      backgroundColor: css("--surface"), borderColor: css("--line"), textStyle: { color: ink, fontSize: 12 },
      extraCssText: "box-shadow: 0 2px 10px rgba(0,0,0,0.25); border-radius: 4px;",
    },
    axis: {
      axisLine: { lineStyle: { color: line } },
      axisTick: { show: false },
      axisLabel: { color: ink2 },
      splitLine: { lineStyle: { color: grid } },
      nameTextStyle: { color: ink2, fontSize: 12 },
    },
    legend: { type: "scroll", textStyle: { color: ink2 }, icon: "roundRect", itemWidth: 12, itemHeight: 8, top: 0, left: "center", width: "72%", pageIconColor: ink2, pageTextStyle: { color: ink2 } },
  };
}

export function axis(extra = {}) {
  return { ...chrome().axis, ...extra };
}

/** Creates an ECharts instance on a node and keeps it sized to its container. */
export function makeChart(node) {
  const inst = echarts.init(node, null, { renderer: "canvas" });
  const ro = new ResizeObserver(() => inst.resize());
  ro.observe(node);
  inst._ro = ro;
  return inst;
}

export function disposeChart(inst) {
  if (!inst) return;
  inst._ro && inst._ro.disconnect();
  inst.dispose();
}

export function chartNode(cls = "") {
  return el("div", { class: "chart " + cls });
}
