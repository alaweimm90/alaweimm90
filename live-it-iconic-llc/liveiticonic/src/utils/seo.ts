import { Product } from '@/types/product';
import { Order } from '@/types/order';

interface StructuredData {
  '@context': string;
  '@type': string;
  [key: string]: unknown;
}

/**
 * Generate Product Schema for rich snippets
 * Supports star ratings, pricing, availability, and more
 */
export const generateProductStructuredData = (product: Product): StructuredData => {
  return {
    '@context': 'https://schema.org',
    '@type': 'Product',
    '@id': `https://liveiticonic.com/product/${product.id}`,
    name: product.name,
    description: product.description,
    image: product.images.map(img => img.url),
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
      price: String(product.price),
      priceCurrency: 'USD',
      availability:
        product.inventory.quantity > 0
          ? 'https://schema.org/InStock'
          : 'https://schema.org/OutOfStock',
      url: `https://liveiticonic.com/product/${product.id}`,
      seller: {
        '@type': 'Organization',
        name: 'Live It Iconic',
        url: 'https://liveiticonic.com',
      },
    },
    // Add rating if available
    ...(product.rating && product.reviewCount && {
      aggregateRating: {
        '@type': 'AggregateRating',
        ratingValue: String(product.rating),
        reviewCount: product.reviewCount,
        bestRating: '5',
        worstRating: '1',
      },
    }),
  };
};

/**
 * Generate Breadcrumb Schema for navigation paths
 */
export const generateBreadcrumbs = (
  items: Array<{ name: string; url: string }>
): StructuredData => {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: item.name,
      item: item.url,
    })),
  };
};

/**
 * Generate Organization Schema
 * Provides brand/company information to search engines
 */
export const generateOrganizationSchema = (): StructuredData => {
  return {
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
    ],
    contactPoint: {
      '@type': 'ContactPoint',
      contactType: 'Customer Service',
      email: 'hello@liveiticonic.com',
      telephone: '+1-800-ICONIC-1',
    },
    foundingDate: '2024',
    knowsAbout: ['Automotive Fashion', 'Luxury Apparel', 'Motorsport Style', 'Premium Streetwear'],
  };
};

/**
 * Generate Order Schema
 * Used on order confirmation pages
 */
export const generateOrderStructuredData = (order: Order): StructuredData => {
  return {
    '@context': 'https://schema.org',
    '@type': 'Order',
    '@id': `https://liveiticonic.com/order/${order.orderNumber}`,
    orderNumber: order.orderNumber,
    orderStatus: `https://schema.org/OrderStatus/${order.status}`,
    price: String(order.total),
    priceCurrency: 'USD',
    orderDate: order.createdAt.toISOString(),
    orderedItem: order.items.map(item => ({
      '@type': 'Product',
      name: item.name,
      quantity: item.quantity,
      price: String(item.price),
    })),
    seller: {
      '@type': 'Organization',
      name: 'Live It Iconic',
      url: 'https://liveiticonic.com',
    },
  };
};

/**
 * Generate Website Schema
 * Root schema for entire site
 */
export const generateWebsiteSchema = (): StructuredData => {
  return {
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
  };
};
