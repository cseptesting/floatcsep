'use client';

import { useManifest } from '@/lib/contexts/ManifestContext';

export default function Footer() {
  const { manifest } = useManifest();

  return (
    <footer className="bg-surface border-t border-border py-4 px-6">
      <div className="container mx-auto flex items-center justify-between text-xs text-gray-400">
        <div>
          {manifest?.floatcsep_version && (
            <span className="font-semibold">floatCSEP</span>
          )}
          {manifest?.floatcsep_version && ` v${manifest.floatcsep_version}`}
        </div>
        <div className="flex items-center space-x-4">
          <a
            href="https://github.com/cseptesting/floatcsep"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-primary transition-colors"
          >
            GitHub
          </a>
          <a
            href="https://floatcsep.readthedocs.io"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-primary transition-colors"
          >
            Documentation
          </a>
          <a
            href="https://cseptesting.org"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-primary transition-colors"
          >
            CSEP
          </a>
        </div>
      </div>
    </footer>
  );
}
