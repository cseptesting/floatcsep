'use client';

import dynamic from 'next/dynamic';

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

// Dynamically import LeafletMap with SSR disabled
const LeafletForecastMap = dynamic(
  () => import('./LeafletForecastMap'),
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

export default function ForecastMap(props: ForecastMapProps) {
  return <LeafletForecastMap {...props} />;
}
