'use client';

import { useManifest } from '@/lib/contexts/ManifestContext';
import { useState, useMemo } from 'react';

import { safeRender } from '@/lib/utils';

export default function ResultsPage() {
  const { manifest, isLoading: manifestLoading } = useManifest();
  const [selectedTimeWindowIndex, setSelectedTimeWindowIndex] = useState<number>(0);
  const [selectedTestIndex, setSelectedTestIndex] = useState<number>(0);
  const [selectedModelIndex, setSelectedModelIndex] = useState<number>(0);
  const [imageError, setImageError] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState<boolean>(false);

  // Get selected items
  const selectedTest = manifest?.tests?.[selectedTestIndex] || null;
  const selectedModel = manifest?.models?.[selectedModelIndex] || null;
  const selectedTimeWindow = manifest?.time_windows?.[selectedTimeWindowIndex] || null;

  // Look up image path from manifest's results_model dictionary
  // The manifest maps (time_window|test|model) -> actual file path
  const imagePath = useMemo(() => {
    if (!manifest || !selectedTest || !selectedModel || selectedTimeWindow === null) return null;

    const testName = selectedTest.name;
    const modelName = selectedModel.name;

    // Build the lookup key: "time_window|test_name|model_name"
    const lookupKey = `${selectedTimeWindow}|${testName}|${modelName}`;

    // Check if we have a per-model result
    if (manifest.results_model && manifest.results_model[lookupKey]) {
      const relativePath = manifest.results_model[lookupKey];
      return `/api/results/${relativePath}`;
    }

    // Fall back to main results (test-level, not per-model)
    const mainKey = `${selectedTimeWindow}|${testName}`;
    if (manifest.results_main && manifest.results_main[mainKey]) {
      const relativePath = manifest.results_main[mainKey];
      return `/api/results/${relativePath}`;
    }

    return null;
  }, [manifest, selectedTest, selectedModel, selectedTimeWindow]);

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

  if (!manifest || !manifest.tests || manifest.tests.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-gray-400">No test results available</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Left Column: Selectors */}
      <div className="lg:col-span-1 space-y-6">
        <div>
          <h1 className="text-2xl font-bold mb-1">Results</h1>
          <p className="text-sm text-gray-400">Evaluation test results</p>
        </div>

        {/* Time Window Selector */}
        <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
          <h2 className="text-lg font-semibold">Time Window</h2>
          <select
            value={selectedTimeWindowIndex}
            onChange={(e) => {
              setSelectedTimeWindowIndex(Number(e.target.value));
              setImageError(null);
            }}
            className="w-full bg-background border border-border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {manifest.time_windows.map((tw, idx) => (
              <option key={idx} value={idx}>
                T{idx + 1}: {tw}
              </option>
            ))}
          </select>
        </div>

        {/* Test Selector */}
        <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
          <h2 className="text-lg font-semibold">Test</h2>
          <select
            value={selectedTestIndex}
            onChange={(e) => {
              setSelectedTestIndex(Number(e.target.value));
              setImageError(null);
            }}
            className="w-full bg-background border border-border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {manifest.tests.map((test, idx) => (
              <option key={idx} value={idx}>
                {test.name}
              </option>
            ))}
          </select>

          {selectedTest && (
            <div className="text-xs text-gray-400 space-y-1 pt-2">
              {selectedTest.type && (
                <p>
                  <span className="font-semibold">Type:</span> {safeRender(selectedTest.type)}
                </p>
              )}
              {selectedTest.percentile && (
                <p>
                  <span className="font-semibold">Percentile:</span> {safeRender(selectedTest.percentile)}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Model Selector */}
        <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
          <h2 className="text-lg font-semibold">Model</h2>
          <select
            value={selectedModelIndex}
            onChange={(e) => {
              setSelectedModelIndex(Number(e.target.value));
              setImageError(null);
            }}
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

        {/* Test Metadata */}
        {selectedTest && (
          <div className="bg-surface p-6 rounded-lg border border-border space-y-3">
            <h3 className="text-sm font-semibold">Test Details</h3>
            <div className="space-y-2 text-xs text-gray-400">
              <p>
                <span className="font-semibold">Name:</span>{' '}
                {safeRender(selectedTest.name)}
              </p>
              {selectedTest.type && (
                <p>
                  <span className="font-semibold">Type:</span>{' '}
                  {safeRender(selectedTest.type)}
                </p>
              )}
              {selectedTest.func && safeRender(selectedTest.func) && (
                <p>
                  <span className="font-semibold">Function:</span>{' '}
                  <code className="bg-background px-1 py-0.5 rounded text-xs">
                    {safeRender(selectedTest.func)}
                  </code>
                </p>
              )}
              {selectedTest.plot_func && safeRender(selectedTest.plot_func) && (
                <p>
                  <span className="font-semibold">Plot function:</span>{' '}
                  <code className="bg-background px-1 py-0.5 rounded text-xs">
                    {safeRender(selectedTest.plot_func)}
                  </code>
                </p>
              )}
              {selectedTest.percentile && (
                <p>
                  <span className="font-semibold">Percentile:</span>{' '}
                  {safeRender(selectedTest.percentile)}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Right Column: Result Image */}
      <div className="lg:col-span-3 space-y-6">
        <div className="bg-surface p-6 rounded-lg border border-border">
          <h2 className="text-lg font-semibold mb-4">
            {selectedTest?.name} - {selectedModel?.name}
          </h2>

          {imagePath && (
            <div className="relative w-full bg-background rounded-lg overflow-hidden">
              {imageLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-background z-10">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading result image...</p>
                  </div>
                </div>
              )}

              {imageError && (
                <div className="flex items-center justify-center min-h-[400px] p-8">
                  <div className="text-center text-red-400">
                    <p className="text-xl font-semibold mb-2">Error loading result</p>
                    <p className="text-sm">{imageError}</p>
                    <p className="text-xs text-gray-500 mt-4">
                      Expected path: {imagePath}
                    </p>
                  </div>
                </div>
              )}

              {!imageError && (
                <img
                  src={imagePath}
                  alt={`${selectedTest?.name} - ${selectedModel?.name}`}
                  className="w-full h-auto"
                  onLoadStart={() => setImageLoading(true)}
                  onLoad={() => {
                    setImageLoading(false);
                    setImageError(null);
                  }}
                  onError={() => {
                    setImageLoading(false);
                    setImageError('Result image not found. The test may not have been run yet.');
                  }}
                />
              )}
            </div>
          )}

          {!imagePath && (
            <div className="flex items-center justify-center min-h-[400px]">
              <div className="text-center text-gray-400">
                <p className="mb-2">No result image available for this combination</p>
                <p className="text-xs">
                  This test may not have been run for the selected model and time window.
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="text-xs text-gray-400">
          <p>
            <span className="font-semibold">Note:</span> Result images are generated by floatCSEP evaluation tests.
            If an image is not found, the test may not have been run for this combination.
          </p>
        </div>
      </div>
    </div>
  );
}
