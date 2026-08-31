'use client';

import dynamic from 'next/dynamic';
import { CatalogEvent } from '@/lib/types';

interface SpatialMapProps {
  events: CatalogEvent[];
  bbox?: [number, number, number, number] | null;
  startDate?: string;
}

// Dynamically import LeafletMap with SSR disabled
const LeafletSpatialMap = dynamic(
  () => import('./LeafletSpatialMap'),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[450px] rounded-lg border border-border flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Loading map...</p>
        </div>
      </div>
    )
  }
);

export default function SpatialMap(props: SpatialMapProps) {
  return <LeafletSpatialMap {...props} />;
}
