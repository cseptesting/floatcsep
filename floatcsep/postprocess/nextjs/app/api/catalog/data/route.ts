import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

export const dynamic = 'force-dynamic';

const execAsync = promisify(exec);

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const catalogPath = searchParams.get('path');
    const appRoot = process.env.APP_ROOT || '.';

    if (!catalogPath) {
      return NextResponse.json(
        { error: 'Catalog path required' },
        { status: 400 }
      );
    }

    // Call Python subprocess
    const pythonScript = path.join(process.cwd(), 'manifest_api.py');
    const command = `python "${pythonScript}" load_catalog "${catalogPath}" "${appRoot}"`;

    const { stdout, stderr } = await execAsync(command);

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

    return NextResponse.json(data, {
      headers: {
        'Cache-Control': 'no-store, no-cache, must-revalidate',
      },
    });
  } catch (error) {
    console.error('Error loading catalog:', error);
    return NextResponse.json(
      { error: 'Failed to load catalog data', details: String(error) },
      { status: 500 }
    );
  }
}
