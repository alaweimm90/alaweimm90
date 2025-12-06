import { useLocation } from "react-router-dom";
import { useEffect, useMemo } from "react";
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { useSEO } from '@/hooks/use-seo';

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      '404 Error: User attempted to access non-existent route:',
      location.pathname
    );
  }, [location.pathname]);

  useSEO({ title: '404 – Page Not Found | SimCore', description: 'The page you are looking for does not exist. Return to SimCore home.' });

  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name: '404 Not Found',
    description: 'The page you are looking for does not exist.',
    inLanguage: 'en',
    isPartOf: { '@type': 'WebSite', name: 'SimCore' },
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);

  useEffect(() => {
    let meta = document.querySelector('meta[name="robots"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.setAttribute('name', 'robots');
      document.head.appendChild(meta);
    }
    meta.setAttribute('content', 'noindex, nofollow');
  }, []);

return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">404</h1>
          <p className="text-xl text-muted-foreground mb-6">Oops! Physics module not found</p>
          <a href="/" className="inline-flex items-center px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
            Return to Home
          </a>
        </div>
      </div>
    </PhysicsModuleLayout>
  );
};

export default NotFound;
