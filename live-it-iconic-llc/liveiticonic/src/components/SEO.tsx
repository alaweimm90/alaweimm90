import { Helmet } from 'react-helmet-async';

interface SEOProps {
  title: string;
  description: string;
  canonical?: string;
  ogImage?: string;
  ogType?: 'website' | 'article' | 'product';
  ogImageAlt?: string;
  ogImageWidth?: number;
  ogImageHeight?: number;
  twitterHandle?: string;
  keywords?: string;
  robots?: string;
  author?: string;
  datePublished?: string;
  dateModified?: string;
  structuredData?: object;
}

/**
 * SEO component manages comprehensive page SEO including meta tags, Open Graph, Twitter Card, and structured data
 *
 * Uses react-helmet-async to inject SEO meta tags into the document head.
 * Handles:
 * - Title and meta description
 * - Canonical URLs for duplicate prevention
 * - Open Graph tags for social media sharing
 * - Twitter Card tags for Twitter/X
 * - Additional meta tags (keywords, robots, author)
 * - Publishing dates for content
 * - JSON-LD structured data for rich snippets
 *
 * @component
 * @param {SEOProps} props - Component props
 * @param {string} props.title - Page title (prepended with "| Live It Iconic")
 * @param {string} props.description - Meta description for search results (150-160 chars)
 * @param {string} [props.canonical] - Canonical URL path to prevent duplicate content
 * @param {string} [props.ogImage] - Open Graph image URL (1200x630px recommended)
 * @param {'website' | 'article' | 'product'} [props.ogType] - Open Graph content type
 * @param {string} [props.ogImageAlt] - Alt text for OG image
 * @param {number} [props.ogImageWidth] - OG image width
 * @param {number} [props.ogImageHeight] - OG image height
 * @param {string} [props.twitterHandle] - Twitter/X handle (without @)
 * @param {string} [props.keywords] - Comma-separated keywords
 * @param {string} [props.robots] - Robots meta directive
 * @param {string} [props.author] - Content author
 * @param {string} [props.datePublished] - ISO 8601 publication date
 * @param {string} [props.dateModified] - ISO 8601 modification date
 * @param {object} [props.structuredData] - JSON-LD structured data object
 *
 * @example
 * <SEO
 *   title="Premium Black T-Shirt"
 *   description="Precision-cut black t-shirt with automotive-inspired design. Premium quality apparel."
 *   canonical="/product/black-tshirt"
 *   ogType="product"
 *   ogImage="/og-black-tshirt.jpg"
 *   keywords="black t-shirt, premium apparel, automotive fashion"
 *   structuredData={productSchema}
 * />
 *
 * @remarks
 * - Site URL hardcoded as https://liveiticonic.com
 * - Title automatically appended with " | Live It Iconic"
 * - Description should be 150-160 characters for optimal display in search results
 * - Keywords: up to 5-7 primary keywords
 * - OG Image minimum: 200x200px, recommended: 1200x630px
 * - Canonical URLs prevent SEO penalties for duplicate content
 */
const SEO = ({
  title,
  description,
  canonical,
  ogImage = 'https://liveiticonic.com/og-image.jpg',
  ogType = 'website',
  ogImageAlt,
  ogImageWidth = 1200,
  ogImageHeight = 630,
  twitterHandle = 'liveiconic',
  keywords,
  robots = 'index, follow',
  author = 'Live It Iconic',
  datePublished,
  dateModified,
  structuredData,
}: SEOProps) => {
  const siteUrl = 'https://liveiticonic.com';
  const fullTitle = `${title} | Live It Iconic`;
  const canonicalUrl = canonical ? `${siteUrl}${canonical}` : siteUrl;

  return (
    <Helmet>
      {/* ===== BASIC META TAGS ===== */}
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
      <meta name="theme-color" content="#0B0B0C" />
      <meta name="charset" content="utf-8" />
      <meta httpEquiv="X-UA-Compatible" content="IE=edge" />

      {/* ===== CANONICAL URL ===== */}
      {canonical && <link rel="canonical" href={canonicalUrl} />}

      {/* ===== SEARCHBOT META TAGS ===== */}
      <meta name="robots" content={robots} />
      <meta name="googlebot" content={robots} />
      <meta name="bingbot" content="index, follow" />
      {keywords && <meta name="keywords" content={keywords} />}

      {/* ===== CONTENT META TAGS ===== */}
      <meta name="author" content={author} />
      {datePublished && <meta property="article:published_time" content={datePublished} />}
      {dateModified && <meta property="article:modified_time" content={dateModified} />}

      {/* ===== OPEN GRAPH (SOCIAL SHARING) ===== */}
      <meta property="og:type" content={ogType} />
      <meta property="og:url" content={canonicalUrl} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:image" content={ogImage} />
      <meta property="og:image:width" content={String(ogImageWidth)} />
      <meta property="og:image:height" content={String(ogImageHeight)} />
      {ogImageAlt && <meta property="og:image:alt" content={ogImageAlt} />}
      <meta property="og:site_name" content="Live It Iconic" />
      <meta property="og:locale" content="en_US" />

      {/* ===== TWITTER CARD ===== */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:site" content={`@${twitterHandle}`} />
      <meta name="twitter:creator" content={`@${twitterHandle}`} />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={ogImage} />
      <meta name="twitter:image:alt" content={ogImageAlt || title} />

      {/* ===== ADDITIONAL SEO TAGS ===== */}
      <link rel="icon" href="/favicon.ico" />
      <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
      <meta name="apple-mobile-web-app-capable" content="yes" />
      <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />

      {/* ===== JSON-LD STRUCTURED DATA ===== */}
      {structuredData && (
        <script type="application/ld+json">{JSON.stringify(structuredData)}</script>
      )}
    </Helmet>
  );
};

export default SEO;
