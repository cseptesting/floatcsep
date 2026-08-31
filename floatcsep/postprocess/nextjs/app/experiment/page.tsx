'use client';

import { Manifest, Model, Test } from '@/lib/types';
import { useManifest } from '@/lib/contexts/ManifestContext';
import RegionMap from '@/components/experiment/RegionMap';
import { safeRender } from '@/lib/utils';
import MetadataCard from '@/components/ui/MetadataCard';

export default function ExperimentPage() {
  const { manifest, isLoading, error } = useManifest();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Loading experiment data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center text-red-400">
          <p className="text-xl font-semibold mb-2">Error loading experiment</p>
          <p className="text-sm">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!manifest) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-gray-400">No experiment data available</p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">{safeRender(manifest.name)}</h1>
        <p className="text-gray-400">{safeRender(manifest.date_range)}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">

          {/* Metadata */}
          <MetadataCard
            title="Metadata"
            data={[
              { label: 'Authors', value: manifest.authors },
              { label: 'Class', value: manifest.exp_class },
              { label: 'Start', value: manifest.start_date },
              { label: 'End', value: manifest.end_date },
            ]}
          />

          {/* Region */}
          <MetadataCard
            title="Region"
            data={[
              { label: 'Name', value: manifest.region?.name },
              { label: 'Magnitude', value: `[${safeRender(manifest.mag_min)}, ${safeRender(manifest.mag_max)}]` },
              { label: 'Depth', value: manifest.depth_min !== null ? `[${safeRender(manifest.depth_min)}, ${safeRender(manifest.depth_max)}] km` : null },
            ]}
          />

          {/* Models */}
          <div className="bg-surface p-6 rounded-lg border border-border">
            <h2 className="text-xl font-semibold mb-4">Models ({manifest.models?.length || 0})</h2>
            <div className="space-y-3">
              {manifest.models?.filter(m => m && m.name).map((model: Model, idx: number) => (
                <div key={idx} className="border-l-2 border-primary pl-3">
                  <h3 className="font-semibold text-sm">{safeRender(model.name)}</h3>
                  {model.doi && (
                    <p className="text-xs text-gray-400">DOI: {safeRender(model.doi)}</p>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Tests */}
          <div className="bg-surface p-6 rounded-lg border border-border">
            <h2 className="text-xl font-semibold mb-4">Tests ({manifest.tests?.length || 0})</h2>
            <div className="space-y-3">
              {manifest.tests?.filter(t => t && t.name).map((test: Test, idx: number) => (
                <div key={idx} className="border-l-2 border-secondary pl-3">
                  <h3 className="font-semibold text-sm">{safeRender(test.name)}</h3>
                  {test.func && (
                    <p className="text-xs text-gray-400 font-mono">{safeRender(test.func)}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column - Region Map */}
        <div className="space-y-6">
          <div className="bg-surface p-6 rounded-lg border border-border">
            <h2 className="text-xl font-semibold mb-4">Test Region</h2>
            <RegionMap
              bbox={manifest.region?.bbox || undefined}
              regionName={manifest.region?.name || null}
              dh={manifest.region?.dh || undefined}
              origins={manifest.region?.origins || undefined}
            />
          </div>
        </div>
      </div>

      {/* Time Windows */}
      <div className="bg-surface p-6 rounded-lg border border-border">
        <h2 className="text-xl font-semibold mb-4">Time Windows ({manifest.time_windows?.length || 0})</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm">
          {manifest.time_windows?.filter((tw: any) => tw && typeof tw === 'string').map((tw: string, idx: number) => (
            <div key={idx} className="bg-background p-2 rounded">
              <span className="text-primary font-semibold">T{idx + 1}:</span> {safeRender(tw)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
