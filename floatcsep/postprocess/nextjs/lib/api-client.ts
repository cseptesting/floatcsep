import useSWR from 'swr';
import { CatalogData, ForecastData } from './types';

export const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error('Failed to fetch');
  }
  return res.json();
};

export function useCatalogData(catalogPath: string | null) {
  return useSWR<CatalogData>(
    catalogPath ? `/api/catalog/data?path=${encodeURIComponent(catalogPath)}` : null,
    fetcher,
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      dedupingInterval: 0, // No deduping - always fetch fresh
      refreshInterval: 0,
    }
  );
}

export function useForecastData(
  forecastPath: string | null,
  modelIndex: number | null,
  timeWindow: string | null,
  isCatalogFc: boolean
) {
  const params = new URLSearchParams();
  if (forecastPath) params.set('path', forecastPath);
  if (modelIndex !== null) params.set('modelIndex', String(modelIndex));
  if (timeWindow) params.set('timeWindow', timeWindow);
  params.set('isCatalogFc', String(isCatalogFc));

  return useSWR<ForecastData>(
    forecastPath && modelIndex !== null && timeWindow
      ? `/api/forecasts/data?${params.toString()}`
      : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 300000, // 5 minutes
    }
  );
}
