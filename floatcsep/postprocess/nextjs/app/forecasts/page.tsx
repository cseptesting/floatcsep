'use client';

import { useManifest } from '@/lib/contexts/ManifestContext';
import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import useSWR from 'swr';
import dynamic from 'next/dynamic';

const ForecastMap = dynamic(() => import('@/components/forecasts/ForecastMap'), {
  ssr: false,
  loading: () => <div className="w-full h-[500px] bg-surface rounded-lg border border-border animate-pulse" />,
});

const ColorbarLegend = dynamic(() => import('@/components/forecasts/ColorbarLegend'), {
  ssr: false,
  loading: () => <div className="w-full h-16 bg-surface rounded-lg border border-border animate-pulse" />,
});

const fetcher = (params: [string, any]) => {
  const [url, body] = params;
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((res) => {
    if (!res.ok) {
      throw new Error('Failed to fetch forecast data');
    }
    return res.json();
  });
};

export default function ForecastsPage() {
  const { manifest, isLoading: manifestLoading } = useManifest();
  const [selectedModelIndex, setSelectedModelIndex] = useState<number>(0);
  const [selectedTimeWindowIndex, setSelectedTimeWindowIndex] = useState<number>(0);
  const [colorbarMin, setColorbarMin] = useState<number | undefined>(undefined);
  const [colorbarMax, setColorbarMax] = useState<number | undefined>(undefined);

  // Prefetch state
  const [prefetchProgress, setPrefetchProgress] = useState<{ current: number; total: number } | null>(null);
  const prefetchedRef = useRef<Set<string>>(new Set());

  const selectedModel = manifest?.models?.[selectedModelIndex] || null;
  const selectedTimeWindow = manifest?.time_windows?.[selectedTimeWindowIndex] || null;

  // Build forecast path
  const forecastPath = useMemo(() => {
    if (!selectedModel || !selectedTimeWindow) return null;
    return selectedModel.forecast_paths?.[selectedTimeWindowIndex] || null;
  }, [selectedModel, selectedTimeWindow, selectedTimeWindowIndex]);

  const isCatalogFc = selectedModel?.is_catalog_forecast || false;

  // Fetch forecast data
  const { data: forecastData, error: forecastError, isLoading: forecastLoading } = useSWR(
    forecastPath && manifest
      ? ['/api/forecasts/data', {
        path: forecastPath,
        modelIndex: selectedModelIndex,
        timeWindow: selectedTimeWindowIndex,
        isCatalogFc,
        region: manifest.region
      }]
      : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 300000, // 5 minutes
    }
  );

  // Background prefetch all time windows for current model
  const prefetchAllTimeWindows = useCallback(async () => {
    if (!manifest || !selectedModel) return;

    const timeWindows = manifest.time_windows || [];
    const forecastPaths = selectedModel.forecast_paths || [];
    const prefetchKey = `model_${selectedModelIndex}`;

    // Skip if already prefetched for this model
    if (prefetchedRef.current.has(prefetchKey)) return;

    const toPrefetch: number[] = [];
    for (let i = 0; i < timeWindows.length; i++) {
      if (i !== selectedTimeWindowIndex && forecastPaths[i]) {
        toPrefetch.push(i);
      }
    }

    if (toPrefetch.length === 0) {
      prefetchedRef.current.add(prefetchKey);
      return;
    }

    setPrefetchProgress({ current: 0, total: toPrefetch.length });

    // Prefetch sequentially to avoid overwhelming the server
    for (let idx = 0; idx < toPrefetch.length; idx++) {
      const twIndex = toPrefetch[idx];
      try {
        await fetch('/api/forecasts/data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: forecastPaths[twIndex],
            modelIndex: selectedModelIndex,
            timeWindow: twIndex,
            isCatalogFc: selectedModel.is_catalog_forecast || false,
            region: manifest.region
          }),
        });
        setPrefetchProgress({ current: idx + 1, total: toPrefetch.length });
      } catch (err) {
        console.error(`Failed to prefetch time window ${twIndex}:`, err);
      }
    }

    prefetchedRef.current.add(prefetchKey);
    setPrefetchProgress(null);
  }, [manifest, selectedModel, selectedModelIndex, selectedTimeWindowIndex]);

  // Start prefetching after initial data loads
  useEffect(() => {
    if (forecastData && !forecastLoading) {
      // Delay prefetch to not block the initial render
      const timer = setTimeout(() => {
        prefetchAllTimeWindows();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [forecastData, forecastLoading, prefetchAllTimeWindows]);

  // Reset colorbar range when forecast changes
  useMemo(() => {
    if (forecastData) {
      setColorbarMin(undefined);
      setColorbarMax(undefined);
    }
  }, [forecastData]);

  if (manifestLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Loading manifest...</p>
        </div>
      </div>
    );
  }

  if (!manifest || !manifest.models || manifest.models.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-gray-400">No forecast models available</p>
      </div>
    );
  }

  const effectiveMin = colorbarMin !== undefined ? colorbarMin : (forecastData?.vmin || 0);
  const effectiveMax = colorbarMax !== undefined ? colorbarMax : (forecastData?.vmax || 0);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Left Column: Controls */}
      <div className="lg:col-span-1 space-y-6">
        <div>
          <h1 className="text-2xl font-bold mb-1">Forecasts</h1>
          <p className="text-sm text-gray-400">Interactive forecast visualization</p>
        </div>

        {/* Model Selector */}
        <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
          <h2 className="text-lg font-semibold">Model</h2>
          <select
            value={selectedModelIndex}
            onChange={(e) => setSelectedModelIndex(Number(e.target.value))}
            className="w-full bg-background border border-border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {manifest.models.map((model, idx) => (
              <option key={idx} value={idx}>
                {model.name}
              </option>
            ))}
          </select>

          {selectedModel && (
            <div className="text-xs text-gray-400 space-y-1 pt-2">
              <p>
                <span className="font-semibold">Type:</span>{' '}
                {selectedModel.is_catalog_forecast ? 'Catalog-based' : 'Gridded'}
              </p>
              {selectedModel.zenodo_id && (
                <p>
                  <span className="font-semibold">Zenodo:</span>{' '}
                  <code className="bg-background px-1 py-0.5 rounded">{selectedModel.zenodo_id}</code>
                </p>
              )}
              {selectedModel.doi && (
                <p>
                  <span className="font-semibold">DOI:</span>{' '}
                  <a
                    href={`https://doi.org/${selectedModel.doi}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    {selectedModel.doi}
                  </a>
                </p>
              )}
            </div>
          )}
        </div>

        {/* Time Window Selector */}
        <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
          <h2 className="text-lg font-semibold">Time Window</h2>
          <select
            value={selectedTimeWindowIndex}
            onChange={(e) => setSelectedTimeWindowIndex(Number(e.target.value))}
            className="w-full bg-background border border-border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {manifest.time_windows.map((tw, idx) => (
              <option key={idx} value={idx}>
                T{idx + 1}: {tw}
              </option>
            ))}
          </select>

          {/* Prefetch progress indicator */}
          {prefetchProgress && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-gray-400">
                <span>Caching other time windows...</span>
                <span>{prefetchProgress.current}/{prefetchProgress.total}</span>
              </div>
              <div className="w-full bg-background rounded-full h-1.5">
                <div
                  className="bg-primary h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${(prefetchProgress.current / prefetchProgress.total) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>


        {/* Colorbar Range Controls */}
        {/* Colorbar Range Controls removed - moved to legend slider */}
        {forecastData && (
          <div className="bg-surface p-6 rounded-lg border border-border space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-semibold">Color Range</h2>
              <button
                onClick={() => {
                  setColorbarMin(undefined);
                  setColorbarMax(undefined);
                }}
                className="text-xs text-primary hover:underline"
              >
                Reset
              </button>
            </div>
            <p className="text-xs text-gray-400">
              Use the slider below the map to adjust the color scale.
            </p>
          </div>
        )}

        {/* Statistics */}
        {forecastData && (
          <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
            <h3 className="text-sm font-semibold">Statistics</h3>
            <div className="space-y-2 text-sm">
              <p>
                <span className="text-gray-400">Grid cells:</span>{' '}
                <span className="font-semibold">{forecastData.cells?.length || 0}</span>
              </p>
              <p>
                <span className="text-gray-400">Rate range:</span>{' '}
                {Math.pow(10, forecastData.vmin).toExponential(2)} -{' '}
                {Math.pow(10, forecastData.vmax).toExponential(2)}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Right Column: Visualization */}
      <div className="lg:col-span-3 space-y-6">
        {forecastLoading && (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-gray-400">Loading forecast data...</p>
            </div>
          </div>
        )}

        {forecastError && (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center text-red-400">
              <p className="text-xl font-semibold mb-2">Error loading forecast</p>
              <p className="text-sm">{forecastError.message}</p>
            </div>
          </div>
        )}

        {forecastData && forecastData.cells && (
          <>
            <div className="bg-surface p-6 rounded-lg border border-border">
              <h2 className="text-lg font-semibold mb-4">Forecast Map</h2>
              <ForecastMap
                cells={forecastData.cells}
                bbox={manifest.region?.bbox ?? undefined}
                vmin={forecastData.vmin}
                vmax={forecastData.vmax}
                colorbarMin={colorbarMin}
                colorbarMax={colorbarMax}
              />
            </div>

            <ColorbarLegend
              vmin={effectiveMin}
              vmax={effectiveMax}
              dataMin={forecastData.vmin}
              dataMax={forecastData.vmax}
              title="log10 λ"
              onRangeChange={([min, max]) => {
                setColorbarMin(min);
                setColorbarMax(max);
              }}
            />

            <div className="text-xs text-gray-400">
              <p>
                <span className="font-semibold">Note:</span> Forecast rates are displayed in log10 scale.
                Hover over cells for detailed information.
              </p>
            </div>
          </>
        )}

        {!forecastLoading && !forecastError && !forecastData && (
          <div className="flex items-center justify-center min-h-[400px]">
            <p className="text-gray-400">Select a model and time window to view forecast</p>
          </div>
        )}
      </div>
    </div>
  );
}
