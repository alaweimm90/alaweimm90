/**
 * Enhanced SEO Hook
 * 
 * Advanced SEO utilities building on the existing use-seo hook with
 * structured data, OpenGraph, and Twitter Card support.
 */

import { useEffect } from 'react';
import { useSEO } from '@/hooks/use-seo';

interface EnhancedSEOOptions {
  title: string;
  description: string;
  keywords?: string[];
  canonicalUrl?: string;
  openGraph?: {
    type?: 'website' | 'article' | 'profile';
    image?: string;
    imageAlt?: string;
    siteName?: string;
  };
  twitterCard?: {
    card?: 'summary' | 'summary_large_image' | 'app' | 'player';
    site?: string;
    creator?: string;
    image?: string;
    imageAlt?: string;
  };
  structuredData?: Record<string, any>;
  language?: string;
  author?: string;
  publishedTime?: string;
  modifiedTime?: string;
}

export function useEnhancedSEO(options: EnhancedSEOOptions) {
  const {
    title,
    description,
    keywords = [],
    canonicalUrl,
    openGraph = {},
    twitterCard = {},
    structuredData,
    language = 'en',
    author,
    publishedTime,
    modifiedTime
  } = options;

  // Use base SEO hook
  useSEO({ title, description });

  useEffect(() => {
    const head = document.head;
    const metaTags: HTMLMetaElement[] = [];
    const linkTags: HTMLLinkElement[] = [];
    const scriptTags: HTMLScriptElement[] = [];

    // Helper function to create meta tag
    const createMetaTag = (property: string, content: string, isProperty = false) => {
      const meta = document.createElement('meta');
      if (isProperty) {
        meta.setAttribute('property', property);
      } else {
        meta.setAttribute('name', property);
      }
      meta.setAttribute('content', content);
      head.appendChild(meta);
      metaTags.push(meta);
    };

    // Keywords
    if (keywords.length > 0) {
      createMetaTag('keywords', keywords.join(', '));
    }

    // Language
    document.documentElement.lang = language;

    // Author
    if (author) {
      createMetaTag('author', author);
    }

    // Canonical URL
    if (canonicalUrl) {
      const link = document.createElement('link');
      link.rel = 'canonical';
      link.href = canonicalUrl;
      head.appendChild(link);
      linkTags.push(link);
    }

    // OpenGraph tags
    const ogDefaults = {
      type: 'website',
      siteName: 'SimCore - Interactive Scientific Computing',
      ...openGraph
    };

    createMetaTag('og:title', title, true);
    createMetaTag('og:description', description, true);
    createMetaTag('og:type', ogDefaults.type, true);
    createMetaTag('og:site_name', ogDefaults.siteName, true);

    if (ogDefaults.image) {
      createMetaTag('og:image', ogDefaults.image, true);
      if (ogDefaults.imageAlt) {
        createMetaTag('og:image:alt', ogDefaults.imageAlt, true);
      }
    }

    if (canonicalUrl) {
      createMetaTag('og:url', canonicalUrl, true);
    }

    // Twitter Card tags
    const twitterDefaults = {
      card: 'summary_large_image',
      site: '@SimCore',
      ...twitterCard
    };

    createMetaTag('twitter:card', twitterDefaults.card);
    createMetaTag('twitter:title', title);
    createMetaTag('twitter:description', description);

    if (twitterDefaults.site) {
      createMetaTag('twitter:site', twitterDefaults.site);
    }

    if (twitterDefaults.creator) {
      createMetaTag('twitter:creator', twitterDefaults.creator);
    }

    if (twitterDefaults.image) {
      createMetaTag('twitter:image', twitterDefaults.image);
      if (twitterDefaults.imageAlt) {
        createMetaTag('twitter:image:alt', twitterDefaults.imageAlt);
      }
    }

    // Article-specific meta tags
    if (publishedTime) {
      createMetaTag('article:published_time', publishedTime, true);
    }

    if (modifiedTime) {
      createMetaTag('article:modified_time', modifiedTime, true);
    }

    // Structured Data (JSON-LD)
    if (structuredData) {
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.text = JSON.stringify(structuredData);
      head.appendChild(script);
      scriptTags.push(script);
    }

    // Cleanup function
    return () => {
      metaTags.forEach(tag => head.removeChild(tag));
      linkTags.forEach(tag => head.removeChild(tag));
      scriptTags.forEach(tag => head.removeChild(tag));
    };
  }, [
    title,
    description,
    keywords,
    canonicalUrl,
    openGraph,
    twitterCard,
    structuredData,
    language,
    author,
    publishedTime,
    modifiedTime
  ]);
}

export function usePhysicsModuleSEO(
  moduleName: string,
  description: string,
  category: string,
  difficulty: string,
  equation?: string
) {
  const keywords = [
    'physics simulation',
    'interactive physics',
    'computational physics',
    'scientific computing',
    moduleName.toLowerCase(),
    category.toLowerCase(),
    difficulty.toLowerCase(),
    'WebGL',
    'scientific visualization'
  ];

  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: `${moduleName} - SimCore`,
    description,
    applicationCategory: 'EducationalApplication',
    operatingSystem: 'Web Browser',
    offers: {
      '@type': 'Offer',
      price: '0',
      priceCurrency: 'USD'
    },
    creator: {
      '@type': 'Organization',
      name: 'SimCore Development Team',
      url: 'https://simcore.dev'
    },
    educationalUse: 'Research, Education, Learning',
    learningResourceType: 'Interactive Simulation',
    typicalAgeRange: '18-',
    interactivityType: 'active',
    educationalLevel: difficulty === 'Beginner' ? 'undergraduate' : 'graduate'
  };

  useEnhancedSEO({
    title: `${moduleName} – SimCore`,
    description: `Interactive ${moduleName.toLowerCase()} simulation. ${description}`,
    keywords,
    openGraph: {
      type: 'article',
      image: '/og-physics-simulation.png',
      imageAlt: `${moduleName} physics simulation interface`
    },
    twitterCard: {
      card: 'summary_large_image',
      image: '/twitter-physics-card.png',
      imageAlt: `${moduleName} simulation preview`
    },
    structuredData,
    publishedTime: new Date().toISOString(),
    modifiedTime: new Date().toISOString()
  });
}

export function useHomepageSEO() {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'SimCore',
    url: typeof window !== 'undefined' ? window.location.origin : '',
    description: 'Interactive scientific computing platform for physics, mathematics, and machine learning simulations',
    inLanguage: 'en',
    creator: {
      '@type': 'Organization',
      name: 'SimCore Development Team'
    },
    potentialAction: {
      '@type': 'SearchAction',
      target: {
        '@type': 'EntryPoint',
        urlTemplate: `${typeof window !== 'undefined' ? window.location.origin : ''}?search={search_term_string}`
      },
      'query-input': 'required name=search_term_string'
    }
  };

  useEnhancedSEO({
    title: 'SimCore – Interactive Scientific Computing Platform',
    description: 'Explore interactive physics, mathematics, and ML simulations in your browser. Build intuition through hands-on computational experiments.',
    keywords: [
      'interactive physics',
      'scientific computing',
      'physics simulations',
      'computational physics',
      'scientific visualization',
      'WebGL physics',
      'quantum mechanics',
      'statistical physics',
      'machine learning physics',
      'browser simulations'
    ],
    openGraph: {
      type: 'website',
      image: '/og-homepage.png',
      imageAlt: 'SimCore - Interactive Scientific Computing Platform'
    },
    twitterCard: {
      card: 'summary_large_image',
      image: '/twitter-homepage.png',
      imageAlt: 'SimCore physics simulations overview'
    },
    structuredData
  });
}