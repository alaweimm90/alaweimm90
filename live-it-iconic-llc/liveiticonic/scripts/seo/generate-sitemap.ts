#!/usr/bin/env node

/**
 * Sitemap Generator
 * Generates XML sitemaps for static pages and products
 * Supports multiple sitemap files with proper indexing
 */

import fs from 'fs';
import path from 'path';

interface SitemapEntry {
  url: string;
  lastmod?: string;
  changefreq?: 'always' | 'hourly' | 'daily' | 'weekly' | 'monthly' | 'yearly' | 'never';
  priority?: number;
  images?: Array<{
    url: string;
    caption?: string;
    title?: string;
  }>;
}

interface SitemapConfig {
  baseUrl: string;
  outputDir: string;
  maxUrlsPerSitemap?: number;
}

const CURRENT_DATE = new Date().toISOString().split('T')[0];

// Default configuration
const config: SitemapConfig = {
  baseUrl: 'https://liveiticonic.com',
  outputDir: path.resolve(process.cwd(), 'public'),
  maxUrlsPerSitemap: 50000,
};

/**
 * Escape XML special characters
 */
function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * Generate single URL entry XML
 */
function generateUrlEntry(entry: SitemapEntry): string {
  let xml = '  <url>\n';
  xml += `    <loc>${escapeXml(entry.url)}</loc>\n`;

  if (entry.lastmod) {
    xml += `    <lastmod>${entry.lastmod}</lastmod>\n`;
  }

  if (entry.changefreq) {
    xml += `    <changefreq>${entry.changefreq}</changefreq>\n`;
  }

  if (entry.priority !== undefined) {
    xml += `    <priority>${entry.priority.toFixed(1)}</priority>\n`;
  }

  // Add images if provided
  if (entry.images && entry.images.length > 0) {
    entry.images.forEach(img => {
      xml += '    <image:image>\n';
      xml += `      <image:loc>${escapeXml(img.url)}</image:loc>\n`;
      if (img.caption) {
        xml += `      <image:caption>${escapeXml(img.caption)}</image:caption>\n`;
      }
      if (img.title) {
        xml += `      <image:title>${escapeXml(img.title)}</image:title>\n`;
      }
      xml += '    </image:image>\n';
    });
  }

  xml += '  </url>\n';
  return xml;
}

/**
 * Generate sitemap XML file
 */
function generateSitemap(
  entries: SitemapEntry[],
  filename: string,
  outputDir: string
): void {
  let xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
`;

  entries.forEach(entry => {
    xml += generateUrlEntry(entry);
  });

  xml += '</urlset>';

  const filepath = path.join(outputDir, filename);
  fs.writeFileSync(filepath, xml, 'utf-8');
  console.log(`✓ Generated ${filename} (${entries.length} URLs)`);
}

/**
 * Generate sitemap index for multiple sitemaps
 */
function generateSitemapIndex(
  sitemaps: Array<{ loc: string; lastmod?: string }>,
  outputDir: string
): void {
  let xml = `<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
`;

  sitemaps.forEach(sitemap => {
    xml += '  <sitemap>\n';
    xml += `    <loc>${escapeXml(sitemap.loc)}</loc>\n`;
    if (sitemap.lastmod) {
      xml += `    <lastmod>${sitemap.lastmod}</lastmod>\n`;
    }
    xml += '  </sitemap>\n';
  });

  xml += '</sitemapindex>';

  const filepath = path.join(outputDir, 'sitemap-index.xml');
  fs.writeFileSync(filepath, xml, 'utf-8');
  console.log(`✓ Generated sitemap-index.xml (${sitemaps.length} sitemaps)`);
}

/**
 * Main sitemap generation logic
 */
function generateSitemaps(): void {
  console.log('Generating sitemaps...\n');

  // === STATIC PAGES SITEMAP ===
  const staticPages: SitemapEntry[] = [
    // Homepage
    {
      url: `${config.baseUrl}/`,
      lastmod: CURRENT_DATE,
      changefreq: 'daily',
      priority: 1.0,
    },

    // Main sections
    {
      url: `${config.baseUrl}/shop`,
      lastmod: CURRENT_DATE,
      changefreq: 'daily',
      priority: 0.95,
    },
    {
      url: `${config.baseUrl}/collection`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.9,
    },
    {
      url: `${config.baseUrl}/lifestyle`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.8,
    },

    // About & Brand
    {
      url: `${config.baseUrl}/about`,
      lastmod: CURRENT_DATE,
      changefreq: 'monthly',
      priority: 0.7,
    },
    {
      url: `${config.baseUrl}/brand`,
      lastmod: CURRENT_DATE,
      changefreq: 'monthly',
      priority: 0.7,
    },

    // Customer Service
    {
      url: `${config.baseUrl}/contact`,
      lastmod: CURRENT_DATE,
      changefreq: 'monthly',
      priority: 0.6,
    },
    {
      url: `${config.baseUrl}/policies`,
      lastmod: CURRENT_DATE,
      changefreq: 'monthly',
      priority: 0.5,
    },
  ];

  generateSitemap(staticPages, 'sitemap-pages.xml', config.outputDir);

  // === PRODUCTS SITEMAP (Example - would be dynamic in production) ===
  const products: SitemapEntry[] = [
    {
      url: `${config.baseUrl}/product/black-tshirt`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.8,
      images: [
        {
          url: `${config.baseUrl}/images/black-tshirt-front.jpg`,
          title: 'Black T-Shirt Front View',
          caption: 'Premium black t-shirt with automotive-inspired design',
        },
      ],
    },
    {
      url: `${config.baseUrl}/product/premium-hoodie`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.8,
      images: [
        {
          url: `${config.baseUrl}/images/hoodie-front.jpg`,
          title: 'Premium Hoodie',
          caption: 'Luxury hoodie with precision-cut design',
        },
      ],
    },
    {
      url: `${config.baseUrl}/product/iconic-cap`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.8,
      images: [
        {
          url: `${config.baseUrl}/images/cap-front.jpg`,
          title: 'Iconic Cap',
          caption: 'Premium automotive-inspired cap',
        },
      ],
    },
  ];

  generateSitemap(products, 'sitemap-products.xml', config.outputDir);

  // === CATEGORIES SITEMAP ===
  const categories: SitemapEntry[] = [
    {
      url: `${config.baseUrl}/category/apparel`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.7,
    },
    {
      url: `${config.baseUrl}/category/accessories`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.7,
    },
    {
      url: `${config.baseUrl}/category/limited-edition`,
      lastmod: CURRENT_DATE,
      changefreq: 'weekly',
      priority: 0.7,
    },
  ];

  generateSitemap(categories, 'sitemap-categories.xml', config.outputDir);

  // === GENERATE SITEMAP INDEX ===
  const sitemapIndex = [
    { loc: `${config.baseUrl}/sitemap-pages.xml`, lastmod: CURRENT_DATE },
    { loc: `${config.baseUrl}/sitemap-products.xml`, lastmod: CURRENT_DATE },
    { loc: `${config.baseUrl}/sitemap-categories.xml`, lastmod: CURRENT_DATE },
  ];

  generateSitemapIndex(sitemapIndex, config.outputDir);

  console.log('\n✓ Sitemap generation completed successfully!');
  console.log('\nSitemaps created:');
  console.log(`  - sitemap-pages.xml (${staticPages.length} URLs)`);
  console.log(`  - sitemap-products.xml (${products.length} URLs)`);
  console.log(`  - sitemap-categories.xml (${categories.length} URLs)`);
  console.log(`  - sitemap-index.xml (3 sitemaps)`);
  console.log(`\nNext steps:`);
  console.log(`  1. Submit sitemap-index.xml to Google Search Console`);
  console.log(`  2. Submit sitemap-index.xml to Bing Webmaster Tools`);
  console.log(`  3. Monitor crawl stats in Search Console`);
}

// Run sitemap generation
generateSitemaps();
