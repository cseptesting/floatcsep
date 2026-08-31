'use client';

import { createContext, useContext, ReactNode } from 'react';
import useSWR from 'swr';
import { Manifest } from '../types';

interface ManifestContextType {
  manifest: Manifest | null;
  isLoading: boolean;
  error: any;
}

const ManifestContext = createContext<ManifestContextType | undefined>(undefined);

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch manifest: ${res.status} ${res.statusText}`);
  }
  return res.json();
};

export function ManifestProvider({ children }: { children: ReactNode }) {
  const { data, error, isLoading } = useSWR<Manifest>('/api/manifest', fetcher, {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
  });

  return (
    <ManifestContext.Provider value={{ manifest: data || null, isLoading, error }}>
      {children}
    </ManifestContext.Provider>
  );
}

export function useManifest() {
  const context = useContext(ManifestContext);
  if (!context) {
    throw new Error('useManifest must be used within ManifestProvider');
  }
  return context;
}
