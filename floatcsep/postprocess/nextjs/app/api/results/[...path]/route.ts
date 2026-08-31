import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import path from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const appRoot = process.env.APP_ROOT || '.';

    // Reconstruct file path from params
    // Note: paths from manifest already include 'results/', so don't add it again
    const filePath = path.join(appRoot, ...params.path);

    // Security check: ensure path is within app root
    const resolvedPath = path.resolve(filePath);
    const resolvedRoot = path.resolve(appRoot);

    if (!resolvedPath.startsWith(resolvedRoot)) {
      return NextResponse.json(
        { error: 'Invalid file path' },
        { status: 403 }
      );
    }

    // Read file
    const fileBuffer = await readFile(resolvedPath);

    // Determine content type based on extension
    const ext = path.extname(resolvedPath).toLowerCase();
    let contentType = 'application/octet-stream';

    if (ext === '.png') {
      contentType = 'image/png';
    } else if (ext === '.jpg' || ext === '.jpeg') {
      contentType = 'image/jpeg';
    } else if (ext === '.svg') {
      contentType = 'image/svg+xml';
    } else if (ext === '.pdf') {
      contentType = 'application/pdf';
    }

    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error: any) {
    console.error('Error reading result file:', error);

    if (error.code === 'ENOENT') {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to read result file', details: String(error) },
      { status: 500 }
    );
  }
}
