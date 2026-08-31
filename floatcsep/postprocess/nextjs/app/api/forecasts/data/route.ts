import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { promises as fs } from 'fs';
import os from 'os';
import crypto from 'crypto';

export const dynamic = 'force-dynamic';

const execAsync = promisify(exec);

// File-based cache directory (survives hot-reload)
const CACHE_DIR = path.join(process.cwd(), '.cache', 'forecast_api_cache');

async function ensureCacheDir() {
  try {
    await fs.mkdir(CACHE_DIR, { recursive: true });
  } catch {
    // Directory exists
  }
}

function getCachePath(modelIndex: number, timeWindow: number): string {
  return path.join(CACHE_DIR, `forecast_${modelIndex}_${timeWindow}.json`);
}

async function readCache(cachePath: string): Promise<any | null> {
  try {
    const data = await fs.readFile(cachePath, 'utf-8');
    return JSON.parse(data);
  } catch {
    return null;
  }
}

async function writeCache(cachePath: string, data: any): Promise<void> {
  try {
    await ensureCacheDir();
    await fs.writeFile(cachePath, JSON.stringify(data));
  } catch (err) {
    console.error('Failed to write cache:', err);
  }
}

export async function POST(request: NextRequest) {
  let tempFilePath: string | null = null;
  try {
    const body = await request.json();
    const { path: forecastPath, modelIndex, timeWindow, isCatalogFc, region: regionData } = body;

    if (!forecastPath) {
      return NextResponse.json(
        { error: 'Forecast path required' },
        { status: 400 }
      );
    }

    // Check file-based cache (survives hot-reload)
    const cachePath = getCachePath(modelIndex, timeWindow);
    const cachedData = await readCache(cachePath);
    if (cachedData) {
      return NextResponse.json(cachedData, {
        headers: {
          'Cache-Control': 'public, max-age=3600',
          'X-Cache': 'HIT',
        },
      });
    }

    // Call Python subprocess
    const appRoot = process.env.APP_ROOT || '.';
    const pythonScript = path.join(process.cwd(), 'manifest_api.py');
    const regionStr = JSON.stringify(regionData || {});

    // Write region data to temp file to avoid command line length limits
    const tempDir = os.tmpdir();
    const tempFileName = `region-${crypto.randomUUID()}.json`;
    tempFilePath = path.join(tempDir, tempFileName);
    await fs.writeFile(tempFilePath, regionStr);

    const command = `python "${pythonScript}" load_forecast "${forecastPath}" "${appRoot}" "${tempFilePath}" "${isCatalogFc}"`;

    const { stdout, stderr } = await execAsync(command, { maxBuffer: 1024 * 1024 * 50 });

    if (stderr) {
      console.error('Python stderr:', stderr);
    }

    const data = JSON.parse(stdout);

    if (data.error) {
      return NextResponse.json(
        { error: data.error },
        { status: 500 }
      );
    }

    // Write to file-based cache
    await writeCache(cachePath, data);

    return NextResponse.json(data, {
      headers: {
        'Cache-Control': 'public, max-age=3600',
        'X-Cache': 'MISS',
      },
    });
  } catch (error) {
    console.error('Error loading forecast:', error);
    return NextResponse.json(
      { error: 'Failed to load forecast data', details: String(error) },
      { status: 500 }
    );
  } finally {
    // Clean up temp file
    if (tempFilePath) {
      try {
        await fs.unlink(tempFilePath);
      } catch (err) {
        console.error('Failed to delete temp file:', err);
      }
    }
  }
}
