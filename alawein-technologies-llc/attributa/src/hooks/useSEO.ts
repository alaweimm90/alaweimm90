import { useEffect } from 'react';

/**
 * useSEO - sets document title, meta description, and canonical link.
 * Ensures a single H1 per page is handled in the page component itself.
 */
export function useSEO({
  title,
  description,
  canonical,
}: {
  title: string;
  description?: string;
  canonical?: string;
}) {
  useEffect(() => {
    if (title) {
      document.title = title;
    }

    if (description) {
      let meta = document.querySelector('meta[name="description"]') as HTMLMetaElement | null;
      if (!meta) {
        meta = document.createElement('meta');
        meta.name = 'description';
        document.head.appendChild(meta);
      }
      meta.content = description;
    }

    const canonicalHref = canonical || window.location.href;
    let link = document.querySelector('link[rel="canonical"]') as HTMLLinkElement | null;
    if (!link) {
      link = document.createElement('link');
      link.rel = 'canonical';
      document.head.appendChild(link);
    }
    link.href = canonicalHref;
  }, [title, description, canonical]);
}
