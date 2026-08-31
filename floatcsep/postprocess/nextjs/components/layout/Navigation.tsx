'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const tabs = [
  { name: 'Experiment', href: '/experiment' },
  { name: 'Catalogs', href: '/catalogs' },
  { name: 'Forecasts', href: '/forecasts' },
  { name: 'Results', href: '/results' },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-surface border-b border-border" aria-label="Main navigation">
      <div className="container mx-auto px-6">
        <div className="flex space-x-8" role="tablist">
          {tabs.map((tab) => {
            const isActive = pathname === tab.href;
            return (
              <Link
                key={tab.href}
                href={tab.href}
                role="tab"
                aria-selected={isActive}
                aria-current={isActive ? 'page' : undefined}
                className={cn(
                  'py-4 px-2 border-b-2 transition-colors text-sm font-medium focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-surface',
                  isActive
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-400 hover:text-foreground hover:border-gray-600'
                )}
              >
                {tab.name}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
