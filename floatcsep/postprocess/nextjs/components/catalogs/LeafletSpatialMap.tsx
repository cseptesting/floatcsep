'use client';

import { useMemo, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMap, LayersControl } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { CatalogEvent } from '@/lib/types';

// Fix for default marker icons in Next.js/Leaflet
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png').default.src,
    iconUrl: require('leaflet/dist/images/marker-icon.png').default.src,
    shadowUrl: require('leaflet/dist/images/marker-shadow.png').default.src,
});

interface SpatialMapProps {
    events: CatalogEvent[];
    bbox?: [number, number, number, number] | null;
    startDate?: string;
}

/**
 * Detect if the region crosses the antimeridian (180° longitude).
 * This happens when west longitude > east longitude in the bbox.
 * Example: bbox [165, -47, -175, -34] where west=165 and east=-175
 */
function crossesAntimeridian(bbox: [number, number, number, number] | null | undefined): boolean {
    if (!bbox) return false;
    const [west, , east] = bbox;
    // If west > east, the region spans across 180°
    return west > east;
}

/**
 * Normalize longitude for antimeridian crossing regions.
 * When crossing the antimeridian, we shift all negative longitudes to positive
 * by adding 360. This puts everything in the [0, 360] range instead of [-180, 180].
 * 
 * Example: -175° becomes 185°, so points at 178° and -175° (now 185°) are adjacent.
 */
function normalizeLon(lon: number, crossingAntimeridian: boolean): number {
    if (crossingAntimeridian && lon < 0) {
        return lon + 360;
    }
    return lon;
}

// Component that renders markers using native Leaflet API
function CatalogMarkers({ events, startDate, minMag, maxMag, crossingAntimeridian }: {
    events: CatalogEvent[];
    startDate?: string;
    minMag: number;
    maxMag: number;
    crossingAntimeridian: boolean;
}) {
    const map = useMap();
    const layerRef = useRef<L.LayerGroup | null>(null);

    const getMarkerRadius = (mag: number) => {
        const normalized = maxMag > minMag ? (mag - minMag) / (maxMag - minMag) : 0.5;
        return 4 + Math.pow(normalized, 2) * 8;
    };

    useEffect(() => {
        // Remove existing layer
        if (layerRef.current) {
            map.removeLayer(layerRef.current);
        }

        const layerGroup = L.layerGroup();

        events.forEach(event => {
            const eventDate = new Date(event.time);
            const isInput = startDate ? eventDate < new Date(startDate) : false;

            const color = isInput ? '#38bdf8' : '#ef4444';
            const opacity = isInput ? 0.6 : 0.8;

            // Normalize longitude if crossing antimeridian
            const normalizedLon = normalizeLon(event.lon, crossingAntimeridian);

            const marker = L.circleMarker([event.lat, normalizedLon], {
                radius: getMarkerRadius(event.magnitude),
                color: color,
                fillColor: color,
                fillOpacity: opacity,
                weight: 1
            });

            marker.bindTooltip(
                `<div style="font-size: 11px;">
          <strong>${event.event_id}</strong><br/>
          Time: ${new Date(event.time).toLocaleString()}<br/>
          Mag: ${event.magnitude.toFixed(2)}<br/>
          Loc: ${event.lat.toFixed(3)}°, ${event.lon.toFixed(3)}°
        </div>`,
                { direction: 'top', offset: [0, -5] }
            );

            marker.addTo(layerGroup);
        });

        layerGroup.addTo(map);
        layerRef.current = layerGroup;

        return () => {
            if (layerRef.current) {
                map.removeLayer(layerRef.current);
            }
        };
    }, [map, events, startDate, minMag, maxMag, crossingAntimeridian]);

    return null;
}

// Component to draw the bounding box
function BoundaryBox({ bbox, crossingAntimeridian }: {
    bbox: [number, number, number, number];
    crossingAntimeridian: boolean;
}) {
    const map = useMap();
    const layerRef = useRef<L.Rectangle | null>(null);

    useEffect(() => {
        if (layerRef.current) {
            map.removeLayer(layerRef.current);
        }

        const [west, south, east, north] = bbox;

        // Normalize longitudes for antimeridian crossing
        const normWest = normalizeLon(west, crossingAntimeridian);
        const normEast = normalizeLon(east, crossingAntimeridian);

        const rectangle = L.rectangle([[south, normWest], [north, normEast]], {
            color: '#9ca3af',
            weight: 2,
            fill: false,
            dashArray: '5, 10'
        });

        rectangle.addTo(map);
        layerRef.current = rectangle;

        return () => {
            if (layerRef.current) {
                map.removeLayer(layerRef.current);
            }
        };
    }, [map, bbox, crossingAntimeridian]);

    return null;
}

const LeafletSpatialMap = ({ events, bbox, startDate }: SpatialMapProps) => {
    // Smart detection of antimeridian crossing based on events
    // This is more robust than relying on backend bbox which might be stale or standard format
    const crossingAntimeridian = useMemo(() => {
        // If we have events, check their spread
        if (events.length > 0) {
            const lons = events.map(e => e.lon);
            const minLon = Math.min(...lons);
            const maxLon = Math.max(...lons);
            const rawSpread = maxLon - minLon;

            // Check spread if we shift negative longitudes by +360
            // e.g. -179 becomes 181. 179 stays 179. Spread is 2.
            const shiftedLons = events.map(e => e.lon < 0 ? e.lon + 360 : e.lon);
            const minShifted = Math.min(...shiftedLons);
            const maxShifted = Math.max(...shiftedLons);
            const shiftedSpread = maxShifted - minShifted;

            // If raw spread is huge (e.g. > 180) but shifted spread is compact,
            // we definitely have an antimeridian crossing region.
            if (rawSpread > 180 && shiftedSpread < 180) {
                return true;
            }
        }

        // Fallback to bbox check if available
        return crossesAntimeridian(bbox);
    }, [events, bbox]);

    // Calculate map bounds with normalized coordinates
    const bounds = useMemo(() => {
        // Prioritize calculating bounds from events if we have them and detected crossing
        // This avoids issues where backend bbox is "standard" ([-180, 180]) but events are localized
        if (events.length > 0) {
            const lats = events.map(e => e.lat);
            const lons = events.map(e => normalizeLon(e.lon, crossingAntimeridian));

            return L.latLngBounds([
                [Math.min(...lats), Math.min(...lons)],
                [Math.max(...lats), Math.max(...lons)]
            ]);
        }

        if (bbox) {
            const [west, south, east, north] = bbox;
            // If we detected crossing but bbox is standard format (west < east), 
            // we might need to force normalization on bbox too if it looks global
            // But usually calculating from events is safer for display.
            const normWest = normalizeLon(west, crossingAntimeridian);
            const normEast = normalizeLon(east, crossingAntimeridian);
            return L.latLngBounds([[south, normWest], [north, normEast]]);
        }

        return null;
    }, [bbox, events, crossingAntimeridian]);

    const minMag = useMemo(() =>
        events.length > 0 ? Math.min(...events.map(e => e.magnitude)) : 0,
        [events]
    );
    const maxMag = useMemo(() =>
        events.length > 0 ? Math.max(...events.map(e => e.magnitude)) : 5,
        [events]
    );

    // Default bounds centered on New Zealand
    const defaultBounds = L.latLngBounds([[-47, 165], [-34, 179]]);

    // Calculate visual bounding box from events to ensure it matches the displayed markers
    // This overrides potentially incorrect "global" metadata bbox from backend
    const visualBbox = useMemo(() => {
        if (events.length > 0) {
            const lats = events.map(e => e.lat);
            const lons = events.map(e => normalizeLon(e.lon, crossingAntimeridian));
            return [
                Math.min(...lons), // west (normalized)
                Math.min(...lats), // south
                Math.max(...lons), // east (normalized)
                Math.max(...lats)  // north
            ] as [number, number, number, number];
        }
        return bbox;
    }, [events, bbox, crossingAntimeridian]);

    if (events.length === 0) {
        return (
            <div className="w-full h-[450px] rounded-lg border border-border flex items-center justify-center bg-background">
                <p className="text-gray-400">No events to display</p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            <div className="w-full h-[450px] rounded-lg border border-border overflow-hidden relative z-0">
                <MapContainer
                    bounds={bounds || defaultBounds}
                    style={{ height: '100%', width: '100%' }}
                    preferCanvas={true}
                    scrollWheelZoom={true}
                    worldCopyJump={true}
                >
                    <LayersControl position="topright">
                        <LayersControl.BaseLayer checked name="OpenStreetMap">
                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            />
                        </LayersControl.BaseLayer>
                        <LayersControl.BaseLayer name="Esri WorldImagery">
                            <TileLayer
                                attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                            />
                        </LayersControl.BaseLayer>
                    </LayersControl>
                    <CatalogMarkers
                        events={events}
                        startDate={startDate}
                        minMag={minMag}
                        maxMag={maxMag}
                        crossingAntimeridian={crossingAntimeridian}
                    />
                    {visualBbox && <BoundaryBox bbox={visualBbox} crossingAntimeridian={crossingAntimeridian} />}
                </MapContainer>
            </div>
            <div className="text-xs text-gray-400 space-y-1">
                <p className="flex items-center gap-2">
                    <span className="inline-block w-3 h-3 rounded-full bg-[#38bdf8] opacity-60"></span>
                    <span>Input Catalog (t &lt; start)</span>
                    <span className="ml-4 inline-block w-3 h-3 rounded-full bg-[#ef4444] opacity-80"></span>
                    <span>Test Catalog (start ≤ t)</span>
                </p>
                <p><span className="font-semibold">Note:</span> Marker size represents magnitude</p>
            </div>
        </div>
    );
};

export default LeafletSpatialMap;
