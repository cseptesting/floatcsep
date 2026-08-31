'use client';

import Link from 'next/link';
import { ThemeToggle } from '../ThemeToggle';

export default function Header() {
  return (
    <header className="bg-surface border-b border-border">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link href="/" className="flex items-center space-x-3">
            <div className="text-2xl font-bold text-primary">floatCSEP</div>
          </Link>
          <span className="text-sm text-gray-400">Experiment Dashboard</span>
        </div>
        <div className="flex items-center space-x-4">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
