'use client';

import { useMemo, useEffect, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, useMap, LayersControl } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icons
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png').default.src,
    iconUrl: require('leaflet/dist/images/marker-icon.png').default.src,
    shadowUrl: require('leaflet/dist/images/marker-shadow.png').default.src,
});

// Viridis color palette (better visibility than Magma)
const VIRIDIS_COLORS = [
    '#440154', '#440256', '#450457', '#450559', '#46075a', '#46085c', '#460a5d', '#460b5e', '#470d60', '#470e61', '#471063', '#471164', '#471365', '#481467', '#481668', '#481769', '#48186a', '#481a6c', '#481b6d', '#481c6e', '#481d6f', '#481f70', '#482071', '#482173', '#482374', '#482475', '#482576', '#482677', '#482878', '#482979', '#472a7a', '#472c7a', '#472d7b', '#472e7c', '#472f7d', '#46307e', '#46327e', '#46337f', '#463480', '#453581', '#453781', '#453882', '#443983', '#443a83', '#443b84', '#433d84', '#433e85', '#423f85', '#424086', '#424186', '#414287', '#414487', '#404588', '#404688', '#3f4788', '#3f4889', '#3e4989', '#3e4a89', '#3e4c8a', '#3d4d8a', '#3d4e8a', '#3c4f8a', '#3c508b', '#3b518b', '#3b528b', '#3a538b', '#3a548c', '#39558c', '#39568c', '#38588c', '#38598c', '#375a8c', '#375b8d', '#365c8d', '#365d8d', '#355e8d', '#355f8d', '#34608d', '#34618d', '#33628d', '#33638d', '#32648e', '#32658e', '#31668e', '#31678e', '#31688e', '#30698e', '#306a8e', '#2f6b8e', '#2f6c8e', '#2e6d8e', '#2e6e8e', '#2e6f8e', '#2d708e', '#2d718e', '#2c718e', '#2c728e', '#2c738e', '#2b748e', '#2b758e', '#2a768e', '#2a778e', '#2a788e', '#29798e', '#297a8e', '#297b8e', '#287c8e', '#287d8e', '#277e8e', '#277f8e', '#27808e', '#26818e', '#26828e', '#26828e', '#25838e', '#25848e', '#25858e', '#24868e', '#24878e', '#23888e', '#23898e', '#238a8d', '#228b8d', '#228c8d', '#228d8d', '#218e8d', '#218f8d', '#21908d', '#21918c', '#20928c', '#20928c', '#20938c', '#1f948c', '#1f958b', '#1f968b', '#1f978b', '#1f988b', '#1f998a', '#1f9a8a', '#1e9b8a', '#1e9c89', '#1e9d89', '#1f9e89', '#1f9f88', '#1fa088', '#1fa188', '#1fa187', '#1fa287', '#20a386', '#20a486', '#21a585', '#21a685', '#22a785', '#22a884', '#23a983', '#24aa83', '#25ab82', '#25ac82', '#26ad81', '#27ad81', '#28ae80', '#29af7f', '#2ab07f', '#2cb17e', '#2db27d', '#2eb37c', '#2fb47c', '#31b57b', '#32b67a', '#34b679', '#35b779', '#37b878', '#38b977', '#3aba76', '#3bbb75', '#3dbc74', '#3fbc73', '#40bd72', '#42be71', '#44bf70', '#46c06f', '#48c16e', '#4ac16d', '#4cc26c', '#4ec36b', '#50c46a', '#52c569', '#54c568', '#56c667', '#58c765', '#5ac864', '#5cc863', '#5ec962', '#60ca60', '#63cb5f', '#65cb5e', '#67cc5c', '#69cd5b', '#6ccd5a', '#6ece58', '#70cf57', '#73d056', '#75d054', '#77d153', '#7ad151', '#7cd250', '#7fd34e', '#81d34d', '#84d44b', '#86d549', '#89d548', '#8bd646', '#8ed645', '#90d743', '#93d741', '#95d840', '#98d83e', '#9bd93c', '#9dd93b', '#a0da39', '#a2da37', '#a5db36', '#a8db34', '#aadc32', '#addc30', '#b0dd2f', '#b2dd2d', '#b5de2b', '#b8de29', '#bade28', '#bddf26', '#c0df25', '#c2df23', '#c5e021', '#c8e020', '#cae11f', '#cde11d', '#d0e11c', '#d2e21b', '#d5e21a', '#d8e219', '#dae319', '#dde318', '#dfe318', '#e2e418', '#e5e419', '#e7e419', '#eae51a', '#ece51b', '#efe51c', '#f1e51d', '#f4e61e', '#f6e620', '#f8e621', '#fbe723', '#fde725'
];

interface ForecastCell {
    lon: number;
    lat: number;
    rate: number;
}

interface ForecastMapProps {
    cells: ForecastCell[];
    bbox?: [number, number, number, number];
    vmin: number;
    vmax: number;
    colorbarMin?: number;
    colorbarMax?: number;
}

// Pre-computed cell data structure for performance
interface ProcessedCellData {
    lons: Float32Array;
    lats: Float32Array;
    colorIndices: Uint16Array;
    count: number;
    dLon: number;
    dLat: number;
    minLon: number;
    minLat: number;
    // Lookup map for tooltip hit testing
    lookup: Map<string, { lon: number; lat: number; rate: number }>;
}

// Optimized Canvas Layer for rendering thousands of grid cells
function ForecastLayer({ cells, vmin, vmax, colorbarMin, colorbarMax }: {
    cells: ForecastCell[];
    vmin: number;
    vmax: number;
    colorbarMin?: number;
    colorbarMax?: number;
}) {
    const map = useMap();
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const tooltipRef = useRef<L.Tooltip | null>(null);
    const rafIdRef = useRef<number | null>(null);
    const lastZoomRef = useRef<number | null>(null);
    const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const offscreenBoundsRef = useRef<L.LatLngBounds | null>(null);

    const effectiveMin = colorbarMin !== undefined ? colorbarMin : vmin;
    const effectiveMax = colorbarMax !== undefined ? colorbarMax : vmax;

    // Pre-process cell data: compute color indices and use typed arrays
    const processedData = useMemo<ProcessedCellData | null>(() => {
        if (cells.length === 0) return null;

        const count = cells.length;
        const lons = new Float32Array(count);
        const lats = new Float32Array(count);
        const colorIndices = new Uint16Array(count);

        // Find unique sorted coords to determine step size
        const uniqueLons = [...new Set(cells.map(c => c.lon))].sort((a, b) => a - b);
        const uniqueLats = [...new Set(cells.map(c => c.lat))].sort((a, b) => a - b);

        const minLon = uniqueLons[0];
        const minLat = uniqueLats[0];
        const dLon = uniqueLons.length > 1 ? uniqueLons[1] - uniqueLons[0] : 0.1;
        const dLat = uniqueLats.length > 1 ? uniqueLats[1] - uniqueLats[0] : 0.1;

        // Create lookup map for hit testing
        const lookup = new Map<string, { lon: number; lat: number; rate: number }>();

        const range = effectiveMax - effectiveMin;
        const colorCount = VIRIDIS_COLORS.length - 1;

        for (let i = 0; i < count; i++) {
            const cell = cells[i];
            lons[i] = cell.lon;
            lats[i] = cell.lat;

            // Pre-compute color index
            const logRate = Math.log10(cell.rate);
            const normalized = range > 0 ? (logRate - effectiveMin) / range : 0.5;
            const clamped = Math.max(0, Math.min(1, normalized));
            colorIndices[i] = Math.floor(clamped * colorCount);

            // Add to lookup
            const lonIdx = Math.round((cell.lon - minLon) / dLon);
            const latIdx = Math.round((cell.lat - minLat) / dLat);
            lookup.set(`${latIdx},${lonIdx}`, { lon: cell.lon, lat: cell.lat, rate: cell.rate });
        }

        return { lons, lats, colorIndices, count, dLon, dLat, minLon, minLat, lookup };
    }, [cells, effectiveMin, effectiveMax]);

    // Throttled draw function using requestAnimationFrame
    const scheduleDraw = useCallback(() => {
        if (rafIdRef.current) {
            cancelAnimationFrame(rafIdRef.current);
        }
        rafIdRef.current = requestAnimationFrame(() => {
            rafIdRef.current = null;
            drawCells();
        });
    }, []);

    // Main draw function - renders cells to canvas
    const drawCells = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !map || !processedData) return;

        const { lons, lats, colorIndices, count, dLon, dLat } = processedData;

        const size = map.getSize();
        const bounds = map.getBounds();
        const currentZoom = map.getZoom();

        // Handle high DPI
        const dpr = window.devicePixelRatio || 1;
        canvas.width = size.x * dpr;
        canvas.height = size.y * dpr;
        canvas.style.width = size.x + 'px';
        canvas.style.height = size.y + 'px';

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.scale(dpr, dpr);

        // Transform origin to match leaflet's overlay pane position
        const topLeft = map.containerPointToLayerPoint([0, 0]);
        L.DomUtil.setPosition(canvas, topLeft);

        // Clear canvas
        ctx.clearRect(0, 0, size.x, size.y);

        const hw = dLon / 2;
        const hh = dLat / 2;

        // Get bounds for culling
        const west = bounds.getWest();
        const east = bounds.getEast();
        const south = bounds.getSouth();
        const north = bounds.getNorth();

        // Batch rendering - iterate through typed arrays
        for (let i = 0; i < count; i++) {
            const lon = lons[i];
            const lat = lats[i];

            // Quick bounds check - skip if outside viewport
            if (lon + hw < west || lon - hw > east ||
                lat + hh < south || lat - hh > north) {
                continue;
            }

            // Project corners to pixel coords
            const p1 = map.latLngToContainerPoint([lat + hh, lon - hw]);
            const p2 = map.latLngToContainerPoint([lat - hh, lon + hw]);

            const w = Math.ceil(p2.x - p1.x); // Ceil to avoid gaps
            const h = Math.ceil(p2.y - p1.y);

            ctx.fillStyle = VIRIDIS_COLORS[colorIndices[i]];
            ctx.fillRect(Math.floor(p1.x), Math.floor(p1.y), w, h);
        }

        lastZoomRef.current = currentZoom;
    }, [map, processedData]);

    useEffect(() => {
        if (!processedData) return;

        const canvas = L.DomUtil.create('canvas', 'leaflet-zoom-animated') as HTMLCanvasElement;
        canvas.style.zIndex = '100'; // Above tiles
        canvas.style.pointerEvents = 'auto'; // allow mouse events

        // Add to overlay pane
        map.getPanes().overlayPane.appendChild(canvas);
        canvasRef.current = canvas;

        // Tooltip instance
        const tooltip = L.tooltip({
            direction: 'top',
            sticky: true,
            className: 'leaflet-tooltip-own',
            opacity: 0.9,
        });
        tooltipRef.current = tooltip;

        function handleMouseMove(e: MouseEvent) {
            if (!processedData || !map) return;

            // Convert mouse event to container point
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const latlng = map.containerPointToLatLng([x, y]);

            // Hit test using lookup map
            const { minLon, minLat, dLon, dLat, lookup } = processedData;

            const lonIdx = Math.round((latlng.lng - minLon) / dLon);
            const latIdx = Math.round((latlng.lat - minLat) / dLat);

            const cell = lookup.get(`${latIdx},${lonIdx}`);

            if (cell) {
                const hw = dLon / 2;
                const hh = dLat / 2;
                if (latlng.lng >= cell.lon - hw && latlng.lng <= cell.lon + hw &&
                    latlng.lat >= cell.lat - hh && latlng.lat <= cell.lat + hh) {

                    tooltip.setLatLng(latlng)
                        .setContent(`<div style="font-size: 11px;">log<sub>10</sub> λ: ${Math.log10(cell.rate).toFixed(2)}</div>`)
                        .addTo(map);
                    canvas.style.cursor = 'crosshair';
                    return;
                }
            }

            map.closeTooltip(tooltip);
            canvas.style.cursor = 'grab';
        }

        // Use throttled draw for map events
        const onMapUpdate = () => scheduleDraw();

        map.on('move', onMapUpdate);
        map.on('zoom', onMapUpdate);
        map.on('viewreset', onMapUpdate);

        // Mouse interaction on the canvas itself
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseleave', () => map.closeTooltip(tooltip));

        // Initial draw
        drawCells();

        return () => {
            // Cancel any pending animation frame
            if (rafIdRef.current) {
                cancelAnimationFrame(rafIdRef.current);
            }

            map.off('move', onMapUpdate);
            map.off('zoom', onMapUpdate);
            map.off('viewreset', onMapUpdate);
            if (canvasRef.current) {
                canvasRef.current.removeEventListener('mousemove', handleMouseMove);
                canvasRef.current.parentNode?.removeChild(canvasRef.current);
            }
            if (tooltipRef.current) {
                map.removeLayer(tooltipRef.current);
            }
        };
    }, [map, processedData, scheduleDraw, drawCells]);

    // Redraw when color range changes
    useEffect(() => {
        if (canvasRef.current && processedData) {
            drawCells();
        }
    }, [effectiveMin, effectiveMax, drawCells, processedData]);

    return null;
}

const LeafletForecastMap = ({ cells, bbox, vmin, vmax, colorbarMin, colorbarMax }: ForecastMapProps) => {
    const bounds = useMemo(() => {
        if (bbox) {
            const [west, south, east, north] = bbox;
            return L.latLngBounds([[south, west], [north, east]]);
        }
        if (cells.length > 0) {
            const lats = cells.map(c => c.lat);
            const lons = cells.map(c => c.lon);
            return L.latLngBounds([
                [Math.min(...lats), Math.min(...lons)],
                [Math.max(...lats), Math.max(...lons)]
            ]);
        }
        return null;
    }, [bbox, cells]);

    if (cells.length === 0) {
        return (
            <div className="w-full h-[500px] rounded-lg border border-border flex items-center justify-center bg-background">
                <p className="text-gray-400">No forecast data available</p>
            </div>
        );
    }

    const defaultBounds = L.latLngBounds([[-47, 165], [-34, 179]]);

    return (
        <div className="relative w-full">
            <div className="w-full h-[500px] rounded-lg border border-border overflow-hidden relative z-0">
                <MapContainer
                    bounds={bounds || defaultBounds}
                    style={{ height: '100%', width: '100%' }}
                    preferCanvas={true}
                    scrollWheelZoom={true}
                >
                    <LayersControl position="topright">
                        <LayersControl.BaseLayer name="OpenStreetMap">
                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            />
                        </LayersControl.BaseLayer>
                        <LayersControl.BaseLayer name="Dark Mode">
                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                            />
                        </LayersControl.BaseLayer>
                        <LayersControl.BaseLayer checked name="Light Mode">
                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                                url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                            />
                        </LayersControl.BaseLayer>
                        <LayersControl.BaseLayer name="Esri WorldImagery">
                            <TileLayer
                                attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                            />
                        </LayersControl.BaseLayer>
                    </LayersControl>
                    <ForecastLayer
                        cells={cells}
                        vmin={vmin}
                        vmax={vmax}
                        colorbarMin={colorbarMin}
                        colorbarMax={colorbarMax}
                    />
                </MapContainer>
            </div>
        </div >
    );
};

export default LeafletForecastMap;
