import { MAGMA } from "./magma.js";

/**
 * Leaflet layer that paints gridded cell values on a canvas.
 *
 * `setData({origins, dh, values})` sets the cells (values aligned with origins, NaN = empty);
 * `setRange(lo, hi)` sets the colour scale; `valueAt(latlng)` looks a cell up for tooltips.
 */
export const GridLayer = L.Layer.extend({
  initialize(options) {
    L.setOptions(this, { alphaMin: 0.15, alphaMax: 0.9, pane: "overlayPane", ...options });
    this._cells = null;
    this._lo = 0;
    this._hi = 1;
    this._alphaRamp = this._makeAlpha();
  },

  onAdd(map) {
    this._map = map;
    if (!map.getPane("gridPane")) map.createPane("gridPane").style.zIndex = 350;
    this.options.pane = "gridPane";
    this._canvas = L.DomUtil.create("canvas", "leaflet-zoom-animated leaflet-grid-layer");
    this._canvas.style.pointerEvents = "none";
    this.getPane().appendChild(this._canvas);
    map.on("moveend zoomend resize viewreset", this._reset, this);
    if (map.options.zoomAnimation) map.on("zoomanim", this._animateZoom, this);
    this._reset();
  },

  onRemove(map) {
    L.DomUtil.remove(this._canvas);
    map.off("moveend zoomend resize viewreset", this._reset, this);
    map.off("zoomanim", this._animateZoom, this);
  },

  setData(cells) {
    this._cells = cells;
    this._index = null;
    if (cells) {
      const dh = cells.dh;
      const idx = new Map();
      cells.origins.forEach(([x, y], i) => idx.set(`${Math.round(x / dh)}|${Math.round(y / dh)}`, i));
      this._index = (lon, lat) => {
        const k = `${Math.round(Math.floor(lon / dh + 1e-9))}|${Math.round(Math.floor(lat / dh + 1e-9))}`;
        const i = idx.get(k);
        return i === undefined ? -1 : i;
      };
    }
    this._draw();
  },

  setRange(lo, hi) {
    this._lo = lo;
    this._hi = hi > lo ? hi : lo + 1e-9;
    this._draw();
  },

  valueAt(latlng) {
    if (!this._cells || !this._index) return null;
    const i = this._index(latlng.lng, latlng.lat);
    if (i < 0) return null;
    const v = this._cells.values[i];
    return Number.isFinite(v) ? { index: i, value: v, origin: this._cells.origins[i] } : null;
  },

  _makeAlpha() {
    const { alphaMin, alphaMax } = this.options;
    const out = new Float32Array(256);
    for (let i = 0; i < 256; i++) out[i] = alphaMin + (alphaMax - alphaMin) * Math.pow(i / 255, 0.8);
    return out;
  },

  _animateZoom(e) {
    const scale = this._map.getZoomScale(e.zoom);
    const offset = this._map._latLngBoundsToNewLayerBounds(this._map.getBounds(), e.zoom, e.center).min;
    L.DomUtil.setTransform(this._canvas, offset, scale);
  },

  _reset() {
    const size = this._map.getSize();
    const topLeft = this._map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this._canvas, topLeft);
    const dpr = window.devicePixelRatio || 1;
    this._canvas.width = size.x * dpr;
    this._canvas.height = size.y * dpr;
    this._canvas.style.width = size.x + "px";
    this._canvas.style.height = size.y + "px";
    this._dpr = dpr;
    this._draw();
  },

  _draw() {
    if (!this._map || !this._canvas) return;
    const ctx = this._canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    if (!this._cells) return;
    ctx.scale(this._dpr, this._dpr);
    const { origins, dh, values } = this._cells;
    const map = this._map;
    const b = map.getBounds().pad(0.05);
    const span = this._hi - this._lo;
    const n = origins.length;
    const rgb = this._rgb || (this._rgb = MAGMA.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]));
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      const [x, y] = origins[i];
      if (x + dh < b.getWest() || x > b.getEast() || y + dh < b.getSouth() || y > b.getNorth()) continue;
      const p0 = map.latLngToContainerPoint([y + dh, x]);
      const p1 = map.latLngToContainerPoint([y, x + dh]);
      if (this.options.fill) {
        ctx.fillStyle = this.options.fill;
      } else {
        let t = (v - this._lo) / span;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const ci = Math.round(t * 255);
        const c = rgb[ci];
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${this._alphaRamp[ci]})`;
      }
      ctx.fillRect(p0.x, p0.y, Math.max(p1.x - p0.x, 1), Math.max(p1.y - p0.y, 1));
    }
  },
});

export function drawColorbar(canvas, lo, hi) {
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = 220), h = (canvas.height = 10);
  const g = ctx.createLinearGradient(0, 0, w, 0);
  MAGMA.forEach((c, i) => g.addColorStop(i / 255, c));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  return [lo, hi];
}
