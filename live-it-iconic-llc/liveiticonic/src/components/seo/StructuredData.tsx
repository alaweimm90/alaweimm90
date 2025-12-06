import React from 'react';
import { Helmet } from 'react-helmet-async';

interface StructuredDataProps {
  type: 'organization' | 'product' | 'breadcrumb' | 'localbusiness' | 'webpage';
  data: Record<string, unknown>;
}

/**
 * StructuredData component for rendering JSON-LD schemas
 *
 * Supports multiple schema types for SEO:
 * - organization: Company/brand information
 * - product: Product details with pricing and availability
 * - breadcrumb: Navigation path breadcrumbs
 * - localbusiness: Physical location information
 * - webpage: Generic webpage metadata
 *
 * @component
 * @param {StructuredDataProps} props - Component props
 * @param {'organization' | 'product' | 'breadcrumb' | 'localbusiness' | 'webpage'} props.type - Schema type
 * @param {Record<string, unknown>} props.data - Schema data object
 *
 * @example
 * <StructuredData
 *   type="product"
 *   data={{
 *     "@context": "https://schema.org",
 *     "@type": "Product",
 *     "name": "Black T-Shirt",
 *     "price": "89.00"
 *   }}
 * />
 */
const StructuredData: React.FC<StructuredDataProps> = ({ type, data }) => {
  return (
    <Helmet>
      <script type="application/ld+json">{JSON.stringify(data)}</script>
    </Helmet>
  );
};

/**
 * Generate Organization/Brand Schema
 * Used on homepage and site-wide
 */
export const generateOrganizationSchema = () => ({
  '@context': 'https://schema.org',
  '@type': 'ClothingStore',
  '@id': 'https://liveiticonic.com/#organization',
  name: 'Live It Iconic',
  url: 'https://liveiticonic.com',
  logo: 'https://liveiticonic.com/logo.png',
  image: 'https://liveiticonic.com/og-home.jpg',
  description:
    'Precision-cut apparel inspired by automotive discipline. Statement pieces for bold days—luxury apparel with clean lines, durable materials, and precise fits.',
  sameAs: [
    'https://instagram.com/liveiticonic',
    'https://twitter.com/liveiticonic',
    'https://linkedin.com/company/liveiticonic',
    'https://tiktok.com/@liveiticonic',
  ],
  contactPoint: {
    '@type': 'ContactPoint',
    contactType: 'Customer Service',
    email: 'hello@liveiticonic.com',
    telephone: '+1-800-ICONIC-1',
  },
  foundingDate: '2024',
  knowsAbout: ['Automotive Fashion', 'Luxury Apparel', 'Motorsport Style', 'Premium Streetwear'],
});

/**
 * Generate Product Schema
 * Used on product detail pages
 */
export const generateProductSchema = (product: {
  id: string;
  name: string;
  description: string;
  price: number;
  sku: string;
  images: string[];
  rating?: number;
  reviewCount?: number;
  inStock: boolean;
  category?: string;
  url?: string;
}) => ({
  '@context': 'https://schema.org',
  '@type': 'Product',
  '@id': `https://liveiticonic.com/product/${product.id}`,
  name: product.name,
  description: product.description,
  image: product.images,
  sku: product.sku,
  brand: {
    '@type': 'Brand',
    name: 'Live It Iconic',
    url: 'https://liveiticonic.com',
    logo: 'https://liveiticonic.com/logo.png',
  },
  category: product.category || 'Apparel',
  offers: {
    '@type': 'Offer',
    url: product.url || `https://liveiticonic.com/product/${product.id}`,
    priceCurrency: 'USD',
    price: product.price.toFixed(2),
    availability: product.inStock
      ? 'https://schema.org/InStock'
      : 'https://schema.org/OutOfStock',
    seller: {
      '@type': 'Organization',
      name: 'Live It Iconic',
      url: 'https://liveiticonic.com',
    },
  },
  ...(product.rating && product.reviewCount && {
    aggregateRating: {
      '@type': 'AggregateRating',
      ratingValue: product.rating.toFixed(1),
      reviewCount: product.reviewCount,
      bestRating: '5',
      worstRating: '1',
    },
  }),
});

/**
 * Generate BreadcrumbList Schema
 * Used for navigation paths
 */
export const generateBreadcrumbSchema = (
  breadcrumbs: Array<{ name: string; url: string }>
) => ({
  '@context': 'https://schema.org',
  '@type': 'BreadcrumbList',
  itemListElement: breadcrumbs.map((item, index) => ({
    '@type': 'ListItem',
    position: index + 1,
    name: item.name,
    item: item.url,
  })),
});

/**
 * Generate LocalBusiness Schema
 * Used if you have physical locations
 */
export const generateLocalBusinessSchema = (business: {
  name?: string;
  address: {
    streetAddress: string;
    addressLocality: string;
    addressRegion: string;
    postalCode: string;
    addressCountry: string;
  };
  phone?: string;
  latitude?: number;
  longitude?: number;
  openingHours?: Array<{
    dayOfWeek: string;
    opens: string;
    closes: string;
  }>;
}) => ({
  '@context': 'https://schema.org',
  '@type': 'ClothingStore',
  '@id': 'https://liveiticonic.com/#localbusiness',
  name: business.name || 'Live It Iconic Store',
  url: 'https://liveiticonic.com',
  address: {
    '@type': 'PostalAddress',
    streetAddress: business.address.streetAddress,
    addressLocality: business.address.addressLocality,
    addressRegion: business.address.addressRegion,
    postalCode: business.address.postalCode,
    addressCountry: business.address.addressCountry,
  },
  ...(business.phone && { telephone: business.phone }),
  ...(business.latitude &&
    business.longitude && {
      geo: {
        '@type': 'GeoCoordinates',
        latitude: business.latitude,
        longitude: business.longitude,
      },
    }),
  ...(business.openingHours && {
    openingHoursSpecification: business.openingHours.map(hours => ({
      '@type': 'OpeningHoursSpecification',
      dayOfWeek: hours.dayOfWeek,
      opens: hours.opens,
      closes: hours.closes,
    })),
  }),
});

/**
 * Generate WebPage Schema
 * Used for general page metadata
 */
export const generateWebPageSchema = (page: {
  title: string;
  description: string;
  url: string;
  image?: string;
  datePublished?: string;
  dateModified?: string;
  author?: string;
}) => ({
  '@context': 'https://schema.org',
  '@type': 'WebPage',
  '@id': page.url,
  name: page.title,
  description: page.description,
  url: page.url,
  ...(page.image && { image: page.image }),
  ...(page.datePublished && { datePublished: page.datePublished }),
  ...(page.dateModified && { dateModified: page.dateModified }),
  ...(page.author && { author: { '@type': 'Organization', name: page.author } }),
  isPartOf: {
    '@id': 'https://liveiticonic.com/#website',
  },
  publisher: {
    '@type': 'Organization',
    '@id': 'https://liveiticonic.com/#organization',
    name: 'Live It Iconic',
  },
});

/**
 * Generate Website Schema
 * Root schema for the entire site
 */
export const generateWebsiteSchema = () => ({
  '@context': 'https://schema.org',
  '@type': 'Website',
  '@id': 'https://liveiticonic.com/#website',
  url: 'https://liveiticonic.com',
  name: 'Live It Iconic',
  description:
    'Premium automotive-inspired apparel and lifestyle merchandise for discerning individuals',
  publisher: {
    '@id': 'https://liveiticonic.com/#organization',
  },
  potentialAction: {
    '@type': 'SearchAction',
    target: {
      '@type': 'EntryPoint',
      urlTemplate: 'https://liveiticonic.com/search?q={search_term_string}',
    },
    'query-input': 'required name=search_term_string',
  },
});

export default StructuredData;
