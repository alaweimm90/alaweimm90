import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

interface SEOProps {
  title?: string;
  description?: string;
  keywords?: string;
  image?: string;
  noIndex?: boolean;
}

const DEFAULT_SEO = {
  title: 'REPZ Coach Pro - AI-Powered Personal Training Platform',
  description: 'Transform your fitness journey with AI-powered coaching, personalized training plans, and real-time form analysis. Join thousands of users achieving their health goals.',
  keywords: 'personal training, AI fitness coach, workout plans, form analysis, nutrition tracking, fitness app',
  image: '/og-image.jpg'
};

export const SEOHead: React.FC<SEOProps> = ({
  title,
  description = DEFAULT_SEO.description,
  keywords = DEFAULT_SEO.keywords,
  image = DEFAULT_SEO.image,
  noIndex = false
}) => {
  const location = useLocation();
  const fullTitle = title ? `${title} | REPZ Coach Pro` : DEFAULT_SEO.title;
  const canonicalUrl = `${window.location.origin}${location.pathname}`;

  useEffect(() => {
    // Update document title
    document.title = fullTitle;

    // Update meta tags
    const updateMetaTag = (name: string, content: string, isProperty = false) => {
      const selector = isProperty ? `meta[property="${name}"]` : `meta[name="${name}"]`;
      let element = document.querySelector(selector) as HTMLMetaElement;
      
      if (!element) {
        element = document.createElement('meta');
        if (isProperty) {
          element.setAttribute('property', name);
        } else {
          element.setAttribute('name', name);
        }
        document.head.appendChild(element);
      }
      
      element.setAttribute('content', content);
    };

    // Update link tags
    const updateLinkTag = (rel: string, href: string) => {
      let element = document.querySelector(`link[rel="${rel}"]`) as HTMLLinkElement;
      
      if (!element) {
        element = document.createElement('link');
        element.setAttribute('rel', rel);
        document.head.appendChild(element);
      }
      
      element.setAttribute('href', href);
    };

    // Basic SEO
    updateMetaTag('description', description);
    updateMetaTag('keywords', keywords);
    
    if (noIndex) {
      updateMetaTag('robots', 'noindex, nofollow');
    } else {
      updateMetaTag('robots', 'index, follow');
    }

    // Open Graph
    updateMetaTag('og:title', fullTitle, true);
    updateMetaTag('og:description', description, true);
    updateMetaTag('og:image', image, true);
    updateMetaTag('og:url', canonicalUrl, true);
    updateMetaTag('og:type', 'website', true);
    updateMetaTag('og:site_name', 'REPZ Coach Pro', true);

    // Twitter Card
    updateMetaTag('twitter:card', 'summary_large_image');
    updateMetaTag('twitter:title', fullTitle);
    updateMetaTag('twitter:description', description);
    updateMetaTag('twitter:image', image);

    // Canonical URL
    updateLinkTag('canonical', canonicalUrl);

    // JSON-LD Structured Data
    const structuredData = {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "REPZ Coach Pro",
      "description": description,
      "url": canonicalUrl,
      "applicationCategory": "HealthApplication",
      "operatingSystem": "Web Browser",
      "offers": {
        "@type": "Offer",
        "category": "Subscription",
        "availability": "https://schema.org/InStock"
      },
      "featureList": [
        "AI-Powered Coaching",
        "Real-time Form Analysis",
        "Personalized Training Plans",
        "Progress Tracking",
        "Nutrition Guidance"
      ]
    };

    let scriptElement = document.querySelector('script[type="application/ld+json"]');
    if (!scriptElement) {
      scriptElement = document.createElement('script');
      scriptElement.setAttribute('type', 'application/ld+json');
      document.head.appendChild(scriptElement);
    }
    scriptElement.textContent = JSON.stringify(structuredData);

  }, [fullTitle, description, keywords, image, canonicalUrl, noIndex]);

  return null;
};