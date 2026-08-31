'use client';

import dynamic from 'next/dynamic';

interface RegionMapProps {
  bbox?: [number, number, number, number]; // [west, south, east, north]
  regionName?: string | null;
  dh?: number | null;
  origins?: [number, number][] | null;
}

// Dynamically import LeafletMap with SSR disabled since Leaflet relies on window
const LeafletMap = dynamic(
  () => import('./LeafletMap'),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[500px] rounded-lg border border-border flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Loading map...</p>
        </div>
      </div>
    )
  }
);

export default function RegionMap(props: RegionMapProps) {
  return <LeafletMap {...props} />;
}
