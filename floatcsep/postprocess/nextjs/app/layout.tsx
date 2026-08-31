import type { Metadata } from 'next';
import './globals.css';
import { ManifestProvider } from '@/lib/contexts/ManifestContext';
import { ThemeProvider } from '@/components/ThemeProvider';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import Navigation from '@/components/layout/Navigation';

export const metadata: Metadata = {
  title: 'floatCSEP Dashboard',
  description: 'Interactive earthquake forecasting experiment dashboard',
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <ManifestProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="light"
            enableSystem
            disableTransitionOnChange
          >
            <div className="min-h-screen flex flex-col">
              <a
                href="#main-content"
                className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:bg-primary focus:text-background focus:px-4 focus:py-2 focus:rounded"
              >
                Skip to main content
              </a>
              <Header />
              <Navigation />
              <main id="main-content" className="flex-1 container mx-auto px-6 py-8" role="main">
                {children}
              </main>
              <Footer />
            </div>
          </ThemeProvider>
        </ManifestProvider>
      </body>
    </html>
  );
}
