'use client';

import { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON, Rectangle, LayersControl, LayerGroup, Pane } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icons in Next.js/Leaflet
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png').default.src,
    iconUrl: require('leaflet/dist/images/marker-icon.png').default.src,
    shadowUrl: require('leaflet/dist/images/marker-shadow.png').default.src,
});

// Fix for default paths of marker icons
const iconUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png';
const iconRetinaUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png';
const shadowUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png';

interface RegionMapProps {
    bbox?: [number, number, number, number]; // [west, south, east, north]
    regionName?: string | null;
    dh?: number | null;
    origins?: [number, number][] | null;
}

const LeafletMap = ({ bbox, regionName, dh, origins }: RegionMapProps) => {
    const bounds = useMemo(() => {
        if (!bbox) return null;
        const [west, south, east, north] = bbox;
        // Leaflet expects [lat, lon] tuples
        return L.latLngBounds([
            [south, west],
            [north, east]
        ]);
    }, [bbox]);

    // Generate GeoJSON for grid cells
    const gridGeoJSON = useMemo(() => {
        if (!origins || !dh) return null;

        const features = origins.map(origin => {
            const [lon, lat] = origin;
            return {
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'Polygon',
                    coordinates: [[
                        [lon, lat],
                        [lon + dh, lat],
                        [lon + dh, lat + dh],
                        [lon, lat + dh],
                        [lon, lat]
                    ]]
                }
            };
        });

        return {
            type: 'FeatureCollection',
            features: features
        };
    }, [origins, dh]);

    if (!bounds) {
        return (
            <div className="w-full h-[500px] rounded-lg border border-border flex items-center justify-center bg-background">
                <p className="text-gray-400">No map region defined</p>
            </div>
        );
    }

    return (
        <div className="space-y-2">
            <div className="w-full h-[500px] rounded-lg border border-border overflow-hidden relative z-0">
                <MapContainer
                    bounds={bounds}
                    style={{ height: '100%', width: '100%' }}
                    preferCanvas={true} // Use Canvas for performance with many layers
                    scrollWheelZoom={true}
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

                        <LayersControl.Overlay checked name="Region Grid">
                            <LayerGroup>
                                {gridGeoJSON && (
                                    <GeoJSON
                                        data={gridGeoJSON as any}
                                        style={{
                                            color: '#0077b6',
                                            weight: 0.5,
                                            fillColor: '#0077b6',
                                            fillOpacity: 0.1,
                                        }}
                                    />
                                )}
                            </LayerGroup>
                        </LayersControl.Overlay>

                        <LayersControl.Overlay checked name="Boundaries">
                            <Rectangle
                                bounds={bounds}
                                pathOptions={{
                                    color: '#ef4444',
                                    weight: 2,
                                    fill: false
                                }}
                            />
                        </LayersControl.Overlay>
                    </LayersControl>
                </MapContainer>
            </div>
            {(regionName || bbox) && (
                <div className="flex justify-between text-xs text-gray-400">
                    {regionName && <p><span className="font-semibold">Region:</span> {regionName}</p>}
                    {bbox && <p><span className="font-semibold">Bounds:</span> [{bbox[0].toFixed(2)}, {bbox[1].toFixed(2)}] to [{bbox[2].toFixed(2)}, {bbox[3].toFixed(2)}]</p>}
                </div>
            )}
        </div>
    );
};

export default LeafletMap;
