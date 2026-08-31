import { NextResponse } from 'next/server';
import fs from 'fs/promises';

export async function GET() {
  try {
    const manifestPath = process.env.MANIFEST_PATH;

    if (!manifestPath) {
      return NextResponse.json(
        { error: 'Manifest path not configured. Set MANIFEST_PATH environment variable.' },
        { status: 500 }
      );
    }

    const data = await fs.readFile(manifestPath, 'utf-8');
    const manifest = JSON.parse(data);

    // Fix region bbox format: convert [minLon, maxLon, minLat, maxLat] to [west, south, east, north]
    if (manifest.region && manifest.region.bbox && Array.isArray(manifest.region.bbox)) {
      const [minLon, maxLon, minLat, maxLat] = manifest.region.bbox;
      manifest.region.bbox = [minLon, minLat, maxLon, maxLat]; // [west, south, east, north]
    }

    // Transform forecasts dictionary to forecast_paths array for each model
    if (manifest.models && manifest.time_windows) {
      manifest.models = manifest.models.map((model: any) => {
        if (model.forecasts && typeof model.forecasts === 'object') {
          // Convert forecasts dict to array matching time_windows order
          const forecastPaths = manifest.time_windows.map((tw: string) => {
            return model.forecasts[tw] || null;
          });

          return {
            ...model,
            forecast_paths: forecastPaths,
            is_catalog_forecast: model.forecast_class === 'CatalogForecastRepository',
          };
        }
        return model;
      });
    }

    return NextResponse.json(manifest, {
      headers: {
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error) {
    console.error('Error loading manifest:', error);
    return NextResponse.json(
      { error: 'Failed to load manifest', details: String(error) },
      { status: 500 }
    );
  }
}
