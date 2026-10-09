const cache = new Map();

export async function getJSON(url) {
  if (cache.has(url)) return cache.get(url);
  const p = fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  });
  cache.set(url, p);
  try {
    return await p;
  } catch (e) {
    cache.delete(url);
    throw e;
  }
}

// DOM
export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return n;
}

export function panel(title, body, opts = {}) {
  const head = el("div", { class: "panel-head" }, el("h2", {}, title));
  if (opts.sub) head.append(el("span", { class: "sub" }, opts.sub));
  if (opts.right) head.append(el("div", { class: "right" }, opts.right));
  const bodyEl = el("div", { class: "panel-body" + (opts.flush ? " flush" : "") + (opts.tight ? " tight" : "") }, body);
  return el("section", { class: "panel " + (opts.class || "") }, head, bodyEl);
}

export function seg(options, value, onChange, small = true) {
  const box = el("div", { class: "seg" + (small ? " small" : ""), role: "group" });
  const set = (v) => {
    for (const b of box.children) b.setAttribute("aria-pressed", String(b.dataset.v === String(v)));
  };
  for (const o of options) {
    const [v, label] = Array.isArray(o) ? o : [o, o];
    box.append(
      el("button", { "data-v": String(v), type: "button", "aria-pressed": String(v === value), onclick: () => { set(v); onChange(v); } }, label)
    );
  }
  box.set = set;
  return box;
}

export function select(options, value, onChange) {
  const s = el("select", { onchange: () => onChange(s.value) });
  for (const o of options) {
    const [v, label] = Array.isArray(o) ? o : [o, o];
    s.append(el("option", { value: v, selected: String(v) === String(value) }, label));
  }
  return s;
}

export function field(label, control) {
  return el("label", { class: "field" }, el("span", {}, label), control);
}

// theme
export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export function isDark() {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
export function seriesColor(i) {
  return css(`--series-${(i % 10) + 1}`);
}

// formatting
export const fmt = {
  int: (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Math.round(v).toLocaleString("en")),
  num: (v, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : Number(v).toFixed(d)),
  sci: (v) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1000 || a < 0.01) return v.toExponential(2);
    return v.toFixed(a < 1 ? 3 : 2);
  },
  date: (ms) => new Date(ms).toISOString().slice(0, 10),
  lat: (v) => `${Math.abs(v).toFixed(3)}°${v >= 0 ? "N" : "S"}`,
  lon: (v) => `${Math.abs(v).toFixed(3)}°${v >= 0 ? "E" : "W"}`,
  datetime: (ms) => new Date(ms).toISOString().slice(0, 16).replace("T", " "),
  iso: (s) => (s ? String(s).slice(0, 10) : "–"),
};

export function doiBadge(doi, text = false) {
  if (!doi) return null;
  doi = String(doi).replace(/^https?:\/\/(dx\.)?doi\.org\//, "").trim();
  const a = el("a", { class: text ? "doi" : "", href: `https://doi.org/${doi}`, target: "_blank", rel: "noopener", title: `doi:${doi}` });
  if (text) { a.textContent = doi; return a; }
  const img = el("img", { src: `https://zenodo.org/badge/DOI/${doi}.svg`, alt: `DOI ${doi}`, style: { height: "20px", verticalAlign: "middle" } });
  img.addEventListener("error", () => { a.className = "doi"; a.replaceChildren(doi); });
  a.append(img);
  return a;
}

// stats
export function percentile(sorted, p) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}
export function poissonPmf(lambda, kmax) {
  const out = new Array(kmax + 1);
  let p = Math.exp(-lambda);
  out[0] = p;
  for (let k = 1; k <= kmax; k++) {
    p *= lambda / k;
    out[k] = p;
  }
  return out;
}
export function poissonQuantiles(lambda, probs) {
  const kmax = Math.ceil(lambda + 10 * Math.sqrt(lambda) + 20);
  const pmf = poissonPmf(lambda, kmax);
  const out = probs.map(() => kmax);
  let c = 0;
  for (let k = 0; k <= kmax; k++) {
    c += pmf[k];
    probs.forEach((p, i) => { if (out[i] === kmax && c >= p) out[i] = k; });
  }
  return out;
}

// forecast decoding
export function decodeU16(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Uint16Array(buf);
}
export function cellValue(fc, i, k) {
  const q = fc.q[i * fc.nm + k];
  return q ? fc.vmin + ((q - 1) / 65534) * (fc.vmax - fc.vmin) : NaN;
}

// grid index: lon/lat -> cell index
export function gridIndex(origins, dh) {
  const map = new Map();
  const key = (x, y) => `${Math.round(x / dh)}|${Math.round(y / dh)}`;
  origins.forEach(([x, y], i) => map.set(key(x, y), i));
  return (lon, lat) => {
    const i = map.get(key(Math.floor(lon / dh + 1e-9) * dh, Math.floor(lat / dh + 1e-9) * dh));
    return i === undefined ? -1 : i;
  };
}

export function downloadCSV(name, header, rows) {
  const esc = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  const text = [header.join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n");
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: name });
  document.body.append(a);
  a.click();
  a.remove();
}

export function debounce(fn, ms = 80) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// basemaps
export const BASEMAPS = {
  terrain: ["Terrain", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}", "Tiles © Esri, USGS, NOAA", 13],
  satellite: ["Satellite", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Tiles © Esri, Maxar, Earthstar Geographics", 18],
};

let lastBase = "terrain";
try { lastBase = localStorage.getItem("fc-basemap") || "terrain"; } catch (e) {}

/** Leaflet map with a Terrain/Satellite switch stacked under the zoom buttons. */
export function makeMap(node, opts = {}) {
  const map = L.map(node, { zoomControl: true, attributionControl: true, preferCanvas: true, ...opts });
  let layer = null;
  const buttons = {};
  const setBase = (name) => {
    if (!BASEMAPS[name]) name = "terrain";
    const [, url, attr, maxZoom] = BASEMAPS[name];
    if (layer) layer.remove();
    layer = L.tileLayer(url, { attribution: attr, maxNativeZoom: maxZoom, maxZoom: 18 }).addTo(map);
    for (const [k, b] of Object.entries(buttons)) b.classList.toggle("on", k === name);
    lastBase = name;
    try { localStorage.setItem("fc-basemap", name); } catch (e) {}
  };
  const Switch = L.Control.extend({
    options: { position: "topleft" },
    onAdd() {
      const box = L.DomUtil.create("div", "leaflet-bar basemap-switch");
      for (const [k, [label]] of Object.entries(BASEMAPS)) {
        const a = L.DomUtil.create("a", "", box);
        a.href = "#";
        a.textContent = label;
        a.title = `${label} basemap`;
        a.setAttribute("role", "button");
        L.DomEvent.on(a, "click", (e) => { L.DomEvent.preventDefault(e); setBase(k); });
        buttons[k] = a;
      }
      L.DomEvent.disableClickPropagation(box);
      return box;
    },
  });
  new Switch().addTo(map);
  setBase(lastBase);
  map.setBase = setBase;
  map.retheme = () => {};
  return map;
}

export function outlineStyle() {
  return { color: css("--accent"), weight: 1.5, fill: false, opacity: 0.9, interactive: false };
}
