import { useEffect } from 'react';

interface UseSEOOptions {
  title?: string;
  description?: string;
  canonical?: string;
  image?: string;
}

export function useSEO({ title, description, canonical, image }: UseSEOOptions) {
  useEffect(() => {
    if (title) {
      document.title = title;
    }

    const ensureMeta = (selector: string, create: () => HTMLElement) => {
      let el = document.querySelector(selector) as HTMLElement | null;
      if (!el) {
        el = create();
        document.head.appendChild(el);
      }
      return el;
    };

    if (description) {
      const metaDesc = ensureMeta('meta[name="description"]', () => {
        const m = document.createElement('meta');
        m.setAttribute('name', 'description');
        return m;
      });
      metaDesc.setAttribute('content', description);
    }

    const url = canonical || window.location.href;

    if (url) {
      const link = ensureMeta("link[rel='canonical']", () => {
        const l = document.createElement('link');
        l.setAttribute('rel', 'canonical');
        return l;
      }) as HTMLLinkElement;
      link.setAttribute('href', url);
    }

    if (title) {
      const ogTitle = ensureMeta('meta[property="og:title"]', () => {
        const m = document.createElement('meta');
        m.setAttribute('property', 'og:title');
        return m;
      });
      ogTitle.setAttribute('content', title);
    }
    if (description) {
      const ogDesc = ensureMeta('meta[property="og:description"]', () => {
        const m = document.createElement('meta');
        m.setAttribute('property', 'og:description');
        return m;
      });
      ogDesc.setAttribute('content', description);
    }
    if (url) {
      const ogUrl = ensureMeta('meta[property="og:url"]', () => {
        const m = document.createElement('meta');
        m.setAttribute('property', 'og:url');
        return m;
      });
      ogUrl.setAttribute('content', url);
    }

    const imageUrl = image || `${window.location.origin}/placeholder.svg`;
    const ogImg = ensureMeta('meta[property="og:image"]', () => {
      const m = document.createElement('meta');
      m.setAttribute('property', 'og:image');
      return m;
    });
    ogImg.setAttribute('content', imageUrl);

    const twImg = ensureMeta('meta[name="twitter:image"]', () => {
      const m = document.createElement('meta');
      m.setAttribute('name', 'twitter:image');
      return m;
    });
    twImg.setAttribute('content', imageUrl);

    const ogType = ensureMeta('meta[property="og:type"]', () => {
      const m = document.createElement('meta');
      m.setAttribute('property', 'og:type');
      return m;
    });
    ogType.setAttribute('content', 'website');

    const twCard = ensureMeta('meta[name="twitter:card"]', () => {
      const m = document.createElement('meta');
      m.setAttribute('name', 'twitter:card');
      return m;
    });
    twCard.setAttribute('content', 'summary_large_image');

    if (title) {
      const twTitle = ensureMeta('meta[name="twitter:title"]', () => {
        const m = document.createElement('meta');
        m.setAttribute('name', 'twitter:title');
        return m;
      });
      twTitle.setAttribute('content', title);
    }
    if (description) {
      const twDesc = ensureMeta('meta[name="twitter:description"]', () => {
        const m = document.createElement('meta');
        m.setAttribute('name', 'twitter:description');
        return m;
      });
      twDesc.setAttribute('content', description);
    }

    // Auto Breadcrumbs JSON-LD
    try {
      const origin = window.location.origin;
      const path = window.location.pathname;
      const segments = path.split('/').filter(Boolean);
      const items = [
        { '@type': 'ListItem', position: 1, name: 'Home', item: `${origin}/` },
        ...segments.map((seg, idx) => {
          const name = decodeURIComponent(seg)
            .replace(/-/g, ' ')
            .replace(/\b\w/g, (m) => m.toUpperCase());
          const itemUrl = `${origin}/${segments.slice(0, idx + 1).join('/')}`;
          return { '@type': 'ListItem', position: idx + 2, name: idx === segments.length - 1 && title ? title : name, item: itemUrl };
        })
      ];
      const breadcrumbs = {
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: items
      };

      let script = document.getElementById('breadcrumbs-json-ld') as HTMLScriptElement | null;
      if (!script) {
        script = document.createElement('script');
        script.id = 'breadcrumbs-json-ld';
        script.type = 'application/ld+json';
        document.head.appendChild(script);
      }
      script.textContent = JSON.stringify(breadcrumbs);
    } catch (e) {
      // noop
    }
  }, [title, description, canonical, image]);
}
