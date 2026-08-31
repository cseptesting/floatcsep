'use client';

import { useManifest } from '@/lib/contexts/ManifestContext';
import { useCatalogData } from '@/lib/api-client';
import dynamic from 'next/dynamic';

const SpatialMap = dynamic(() => import('@/components/catalogs/SpatialMap'), {
  ssr: false,
  loading: () => <div className="w-full h-[450px] bg-surface rounded-lg border border-border animate-pulse" />,
});

const MagnitudeTimePlot = dynamic(() => import('@/components/catalogs/MagnitudeTimePlot'), {
  ssr: false,
  loading: () => <div className="w-full h-[350px] bg-surface rounded-lg border border-border animate-pulse" />,
});

export default function CatalogsPage() {
  const { manifest, isLoading: manifestLoading } = useManifest();
  const catalogPath = manifest?.catalog?.path || null;
  const { data: catalogData, error: catalogError, isLoading: catalogDataLoading } = useCatalogData(catalogPath);

  if (manifestLoading || catalogDataLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Loading catalog data...</p>
        </div>
      </div>
    );
  }

  if (catalogError) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center text-red-400">
          <p className="text-xl font-semibold mb-2">Error loading catalog</p>
          <p className="text-sm">{catalogError.message}</p>
        </div>
      </div>
    );
  }

  if (!manifest || !catalogData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-gray-400">No catalog data available</p>
      </div>
    );
  }

  const events = catalogData.events || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left Column: Metadata */}
      <div className="lg:col-span-1 space-y-6">
        <div>
          <h1 className="text-2xl font-bold mb-1">Catalogs</h1>
          <p className="text-sm text-gray-400">Event catalog visualization</p>
        </div>

        <div className="bg-surface p-6 rounded-lg border border-border space-y-4">
          <h2 className="text-lg font-semibold">Catalog Metadata</h2>

          <div className="space-y-2 text-sm">
            {catalogPath && (
              <p>
                <span className="text-gray-400">Path:</span>{' '}
                <code className="text-xs bg-background px-1 py-0.5 rounded">{catalogPath}</code>
              </p>
            )}

            <p>
              <span className="text-gray-400">Event count:</span>{' '}
              <span className="font-semibold text-primary">{catalogData.count}</span>
            </p>

            {manifest.start_date && manifest.end_date && (
              <>
                <p>
                  <span className="text-gray-400">Time span:</span>{' '}
                  {manifest.start_date} → {manifest.end_date}
                </p>
              </>
            )}

            {manifest.mag_min !== null && manifest.mag_max !== null && (
              <p>
                <span className="text-gray-400">Magnitude range:</span>{' '}
                [{manifest.mag_min.toFixed(2)}, {manifest.mag_max.toFixed(2)}]
              </p>
            )}

            {catalogData.bbox && (
              <p>
                <span className="text-gray-400">Catalog extent:</span>{' '}
                lon [{catalogData.bbox[0].toFixed(2)}, {catalogData.bbox[2].toFixed(2)}],{' '}
                lat [{catalogData.bbox[1].toFixed(2)}, {catalogData.bbox[3].toFixed(2)}]
              </p>
            )}

            {manifest.catalog_doi && (
              <p>
                <span className="text-gray-400">DOI:</span>{' '}
                <code className="text-xs bg-background px-1 py-0.5 rounded">{manifest.catalog_doi}</code>
              </p>
            )}
          </div>
        </div>

        <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
          <h3 className="text-sm font-semibold">Event Categories</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full bg-[#38bdf8] opacity-60"></span>
              <span className="text-gray-400">Input Catalog</span>
              <span className="ml-auto text-xs">t &lt; {manifest.start_date}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full bg-[#ef4444] opacity-80"></span>
              <span className="text-gray-400">Test Catalog</span>
              <span className="ml-auto text-xs">t ≥ {manifest.start_date}</span>
            </div>
          </div>
        </div>

        {events.length > 0 && (
          <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
            <h3 className="text-sm font-semibold">Statistics</h3>
            <div className="space-y-2 text-sm">
              <p>
                <span className="text-gray-400">Total events:</span>{' '}
                <span className="font-semibold">{events.length}</span>
              </p>
              <p>
                <span className="text-gray-400">Min magnitude:</span>{' '}
                {Math.min(...events.map((e) => e.magnitude)).toFixed(2)}
              </p>
              <p>
                <span className="text-gray-400">Max magnitude:</span>{' '}
                {Math.max(...events.map((e) => e.magnitude)).toFixed(2)}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Right Column: Visualizations */}
      <div className="lg:col-span-2 space-y-6">
        {/* Spatial Map */}
        <div className="bg-surface p-6 rounded-lg border border-border">
          <h2 className="text-lg font-semibold mb-4">Spatial Distribution</h2>
          <SpatialMap
            events={events}
            bbox={catalogData.bbox}
            startDate={manifest.start_date}
          />
        </div>

        {/* Magnitude-Time Plot */}
        <div className="bg-surface p-6 rounded-lg border border-border">
          <h2 className="text-lg font-semibold mb-4">Magnitude vs Time</h2>
          <MagnitudeTimePlot
            events={events}
            timeWindows={manifest.time_windows}
            startDate={manifest.start_date}
          />
          <p className="text-xs text-gray-400 mt-3">
            <span className="font-semibold">Note:</span> Light blue bands indicate forecast time windows
          </p>
        </div>
      </div>
    </div>
  );
}
