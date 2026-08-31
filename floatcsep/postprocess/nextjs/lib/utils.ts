import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function lonLatToMercator(lon: number, lat: number): [number, number] {
  const k = 6378137.0; // Earth radius in meters
  const x = k * (lon * Math.PI / 180);
  const y = k * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI / 180) / 2));
  return [x, y];
}

export function safeRender(value: any): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    // Filter out empty objects and render the rest
    const filtered = value.filter(v => {
      if (typeof v === 'object' && v !== null) {
        return Object.keys(v).length > 0;
      }
      return v !== null && v !== undefined;
    });
    if (filtered.length === 0) return '';
    return filtered.map(v => safeRender(v)).join(', ');
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value);
    if (keys.length === 0) return '';
    return JSON.stringify(value);
  }
  return String(value);
}
