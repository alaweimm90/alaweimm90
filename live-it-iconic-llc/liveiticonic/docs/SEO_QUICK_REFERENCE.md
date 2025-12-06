# Live It Iconic - SEO Quick Reference Guide

## Getting Started with SEO on Live It Iconic

This quick reference covers the essential SEO implementation details for the project.

---

## File Locations

### Configuration Files
- **robots.txt**: `/public/robots.txt`
- **Sitemaps**: `/public/sitemap-*.xml`
- **Sitemap Generator Script**: `/scripts/seo/generate-sitemap.ts`

### Component Files
- **SEO Component**: `/src/components/SEO.tsx` (Meta tags, Open Graph, Twitter Card)
- **Structured Data Component**: `/src/components/seo/StructuredData.tsx` (JSON-LD schemas)
- **SEO Utilities**: `/src/utils/seo.ts` (Schema generation functions)

### Documentation
- **Implementation Guide**: `/docs/SEO_IMPLEMENTATION.md`
- **Content Guidelines**: `/docs/SEO_CONTENT_GUIDELINES.md`
- **Audit Checklist**: `/docs/SEO_AUDIT_CHECKLIST.md`
- **Strategy Document**: `/docs/SEO_STRATEGY.md`
- **This Guide**: `/docs/SEO_QUICK_REFERENCE.md`

---

## SEO Component Usage

### Basic Usage
```tsx
import SEO from '@/components/SEO';

export default function MyPage() {
  return (
    <>
      <SEO
        title="Page Title"
        description="150-160 character meta description"
        canonical="/my-page"
      />
      {/* Page content */}
    </>
  );
}
```

### Full Example with All Options
```tsx
<SEO
  title="Premium Black T-Shirt - Shop Now"
  description="Premium black t-shirt with automotive-inspired design. Quality apparel for discerning individuals."
  canonical="/product/black-tshirt"
  ogImage="/og-black-tshirt.jpg"
  ogImageAlt="Premium black t-shirt product image"
  ogImageWidth={1200}
  ogImageHeight={630}
  keywords="black t-shirt, premium apparel, automotive fashion"
  ogType="product"
  twitterHandle="liveiconic"
  author="Live It Iconic"
  structuredData={productSchema}
/>
```

### Available Props
```typescript
interface SEOProps {
  title: string;                    // Required: 50-60 chars
  description: string;              // Required: 150-160 chars
  canonical?: string;               // Optional: /path format
  ogImage?: string;                 // Optional: full URL or /path
  ogType?: 'website' | 'article' | 'product'; // Optional: defaults to 'website'
  ogImageAlt?: string;              // Optional: image description
  ogImageWidth?: number;            // Optional: defaults to 1200
  ogImageHeight?: number;           // Optional: defaults to 630
  twitterHandle?: string;           // Optional: defaults to 'liveiconic'
  keywords?: string;                // Optional: comma-separated
  robots?: string;                  // Optional: "index, follow"
  author?: string;                  // Optional: defaults to 'Live It Iconic'
  datePublished?: string;           // Optional: ISO 8601 format
  dateModified?: string;            // Optional: ISO 8601 format
  structuredData?: object;          // Optional: JSON-LD schema
}
```

---

## Structured Data Implementation

### Using StructuredData Component
```tsx
import StructuredData from '@/components/seo/StructuredData';
import { generateProductSchema } from '@/utils/seo';

export default function ProductPage() {
  const productData = {
    id: '123',
    name: 'Black T-Shirt',
    description: 'Premium black t-shirt...',
    price: 89.99,
    sku: 'SKU-001',
    images: ['image1.jpg', 'image2.jpg'],
    inStock: true,
  };

  const schema = generateProductSchema(productData);

  return (
    <>
      <StructuredData type="product" data={schema} />
      {/* Page content */}
    </>
  );
}
```

### Available Schema Types

#### 1. Organization Schema
```tsx
import { generateOrganizationSchema } from '@/utils/seo';

const schema = generateOrganizationSchema();
// Returns: Organization schema with brand info, social links, contact
```

**Used on**: Homepage
**Includes**: Brand name, logo, social media, contact info

#### 2. Product Schema
```tsx
import { generateProductStructuredData } from '@/utils/seo';

const schema = generateProductStructuredData(product);
// Returns: Product schema with pricing, availability, rating
```

**Used on**: Product pages
**Includes**: Price, availability, brand, rating, reviews

#### 3. Breadcrumb Schema
```tsx
import { generateBreadcrumbs } from '@/utils/seo';

const schema = generateBreadcrumbs([
  { name: 'Home', url: '/' },
  { name: 'Shop', url: '/shop' },
  { name: 'Product', url: '/product/123' }
]);
```

**Used on**: All pages with navigation
**Includes**: Navigation path, URLs

#### 4. Website Schema
```tsx
import { generateWebsiteSchema } from '@/utils/seo';

const schema = generateWebsiteSchema();
// Returns: Website schema with search action
```

**Used on**: Homepage (optional)
**Includes**: Site name, search functionality

---

## Page Optimization Checklist

### Before Publishing Any Page

- [ ] **Title Tag**
  - [ ] 50-60 characters
  - [ ] Includes primary keyword
  - [ ] Unique and compelling
  - [ ] Includes brand name (after |)

- [ ] **Meta Description**
  - [ ] 150-160 characters
  - [ ] Starts with benefit/hook
  - [ ] Includes soft CTA
  - [ ] Includes keywords (naturally)

- [ ] **Content**
  - [ ] Meets minimum word count (see guidelines)
  - [ ] H1 present and unique
  - [ ] H2/H3 hierarchy logical
  - [ ] Primary keyword in first 100 words
  - [ ] Internal links (3-5 minimum)

- [ ] **Images**
  - [ ] Descriptive filenames
  - [ ] Alt text (10-14 words)
  - [ ] Proper dimensions
  - [ ] Compressed/optimized
  - [ ] WebP format with JPEG fallback

- [ ] **Technical**
  - [ ] Canonical URL set
  - [ ] Schema markup implemented
  - [ ] Mobile-friendly
  - [ ] Page loads < 2.5s
  - [ ] No broken links

- [ ] **SEO Component**
  - [ ] SEO component imported
  - [ ] All required props filled
  - [ ] Structured data passed
  - [ ] Keywords included

---

## Keyword Research & Selection

### Keyword Types & Targets

#### Brand Keywords (Target: #1)
- "Live It Iconic"
- "Live It Iconic shop"
- "liveiconic"
- Difficulty: Very Low
- Traffic: 50-200/month

#### Primary Keywords (Target: Top 5)
- "Premium apparel"
- "Automotive inspired clothing"
- "Motorsport fashion"
- Difficulty: Medium-High
- Traffic: 100-500/month

#### Long-Tail Keywords (Target: Top 1-3)
- "Automotive inspired t-shirts for men"
- "Best premium black t-shirt"
- "Where to buy luxury apparel online"
- Difficulty: Low-Medium
- Traffic: 10-100/month

### Tools for Research
- Google Keyword Planner (free)
- SEMrush (paid)
- Ahrefs (paid)
- Moz (paid)
- AnswerThePublic (free/paid)

---

## Internal Linking Best Practices

### Link Target Recommendations
| From | To | Anchor Text | Quantity |
|------|-----|-----------|----------|
| Homepage | Shop | "Shop Collection" | 1 |
| Homepage | About | "Our Story" | 1 |
| Homepage | Blog | "Resources" | 1 |
| Category | Products | Product name | 3-5 |
| Product | Category | Category name | 1 |
| Product | Related | "Shop Similar" | 2-3 |
| Blog | Product | Product name | 1-2 |
| Blog | Category | Category name | 1-2 |

### Best Practices
- Use descriptive anchor text (not "click here")
- Link to relevant content only
- Keep links natural (don't over-optimize)
- Distribute link equity across site
- Update internal links quarterly

---

## Image Optimization Checklist

### Filename Requirements
- Use descriptive names
- Separate words with hyphens
- Include primary keyword if relevant
- Lowercase only
- No spaces or special characters

**Examples**:
✓ `premium-black-tshirt-front.jpg`
✓ `automotive-inspired-hoodie.jpg`
✗ `IMG_1234.jpg`
✗ `image-2.jpg`

### Alt Text Requirements
- 10-14 words
- Describe image content
- Include keywords (naturally)
- Specific, not generic
- Accessible for screen readers

**Formula**: `[Adjective] [Product] [Material/Design] [Context]`

**Examples**:
✓ "Premium black t-shirt with precision-cut automotive design"
✓ "Luxury motorsport-inspired hoodie in durable fabric"
✗ "Black shirt"
✗ "Product image"

### Image Technical Requirements
- **Dimensions**: Proper aspect ratio, no stretching
- **File Size**: < 200KB (use compression)
- **Format**: WebP primary, JPEG fallback
- **Responsive**: Multiple sizes for different screens
- **Lazy Loading**: Off-screen images should be lazy-loaded

### Compression Tools
- TinyPNG: https://tinypng.com
- ImageOptim: https://imageoptim.com
- Squoosh: https://squoosh.app
- GTmetrix: https://gtmetrix.com

---

## Monitoring & Maintenance Schedule

### Daily Tasks (Automated)
- Sitemap generation (if dynamic)
- Monitoring crawl health (GSC alerts)

### Weekly Tasks (Manual)
- [ ] Check for broken links
- [ ] Monitor 404 errors
- [ ] Review new content indexation

### Monthly Tasks (1st of month)
- [ ] Review GSC coverage report
- [ ] Check Core Web Vitals
- [ ] Monitor top 10 keyword rankings
- [ ] Analyze organic traffic trends
- [ ] Review page speed metrics

### Quarterly Tasks
- [ ] Content audit and refresh
- [ ] Keyword opportunity analysis
- [ ] Competitor benchmarking
- [ ] Technical SEO full audit
- [ ] Link profile review

### Annual Tasks
- [ ] Strategy review and reset
- [ ] Goal setting for next year
- [ ] Comprehensive competitive analysis
- [ ] Platform/tool evaluation

---

## Search Console Setup Checklist

### Initial Setup (Week 1)
- [ ] Verify domain ownership
- [ ] Set target domain (www or non-www)
- [ ] Submit sitemap-index.xml
- [ ] Check mobile usability
- [ ] Review indexation coverage

### Ongoing Monitoring
- [ ] Coverage report (weekly)
- [ ] Crawl errors (weekly)
- [ ] Performance report (monthly)
- [ ] Mobile usability (monthly)
- [ ] Security issues (as needed)
- [ ] Manual actions (as needed)

### Appearance Settings
- [ ] Enable breadcrumbs
- [ ] Review search appearance
- [ ] Monitor SERP features
- [ ] Enable sitelinks
- [ ] Review rich results

---

## Common SEO Mistakes to Avoid

### Content Mistakes
- ✗ Duplicate content across pages
- ✗ Thin content (< 200 words)
- ✗ Keyword stuffing
- ✗ Poor grammar/spelling
- ✗ Outdated information

### Technical Mistakes
- ✗ Missing canonical URLs
- ✗ No robots.txt
- ✗ Broken 404 page
- ✗ Mobile not optimized
- ✗ Slow page speed (> 3s)

### On-Page Mistakes
- ✗ No meta description
- ✗ Multiple H1 tags
- ✗ Poor title tags
- ✗ Missing alt text
- ✗ No internal links

### Backlink Mistakes
- ✗ Buying low-quality links
- ✗ Exact match anchor text overuse
- ✗ Irrelevant backlinks
- ✗ Unnatural link profile
- ✗ Bad neighborhood links

---

## Tools & Resources

### Essential Tools
- **Google Search Console**: https://search.google.com/search-console
- **Google Analytics 4**: https://analytics.google.com
- **Google PageSpeed Insights**: https://pagespeed.web.dev
- **Schema.org Validator**: https://validator.schema.org

### Recommended Premium Tools
1. **SEMrush** ($120+/month)
2. **Ahrefs** ($99+/month)
3. **Moz Pro** ($99+/month)
4. **Surfer SEO** ($99+/month)

### Free Tools
- Ubersuggest
- AnswerThePublic
- GTmetrix
- WebPageTest
- Lighthouse (Chrome)

### Learning Resources
- Google Search Central: https://developers.google.com/search
- Moz SEO Guide: https://moz.com/beginners-guide-to-seo
- Neil Patel: https://neilpatel.com
- Search Engine Journal: https://searchenginejournal.com

---

## Quick Wins (Easy Improvements)

### Do Today (< 1 hour)
1. [ ] Submit sitemaps to GSC and Bing
2. [ ] Check robots.txt configuration
3. [ ] Verify canonical URLs on top 5 pages
4. [ ] Fix any broken image links
5. [ ] Add keywords to 3 product page titles

### Do This Week (< 5 hours)
1. [ ] Optimize 10 title tags
2. [ ] Improve 10 meta descriptions
3. [ ] Add/improve alt text on 20 images
4. [ ] Compress and optimize product images
5. [ ] Create internal linking plan

### Do This Month (< 20 hours)
1. [ ] Expand homepage content (to 700+ words)
2. [ ] Add FAQ section
3. [ ] Create related products sections
4. [ ] Set up GA4 custom events
5. [ ] Link GSC to Analytics
6. [ ] Create blog post outline

---

## Performance Targets

### Search Visibility
- 3 months: Indexed pages, search appearance
- 6 months: 10-15 keywords in top 20
- 12 months: 50+ keywords in top 20

### Traffic Goals
- Current: ~1,000 sessions/month
- 3 months: 3,000 sessions/month
- 6 months: 7,500 sessions/month
- 12 months: 15,000 sessions/month

### Conversion Goals
- Conversion rate: 2-3%
- Monthly transactions: 150-300
- Annual revenue: $50,000-100,000
- ROI: 3:1 minimum

### Page Speed Goals
- LCP: < 2.5 seconds
- FID: < 100 milliseconds
- CLS: < 0.1
- Total page size: < 2MB

---

## Troubleshooting

### Pages Not Indexed
1. Submit URL to GSC manually
2. Verify robots.txt allows crawling
3. Check for noindex meta tags
4. Verify canonical URLs correct
5. Wait 7-14 days, resubmit

### Low Organic Traffic
1. Check keyword rankings
2. Verify content quality
3. Review SERP appearance
4. Analyze page speed
5. Check mobile-friendliness

### Drop in Rankings
1. Check GSC for manual actions
2. Review recent changes
3. Analyze backlink changes
4. Test Core Web Vitals
5. Review competitor activity

### High Bounce Rate
1. Improve page speed
2. Improve content relevance
3. Enhance UX (mobile, layout)
4. Add internal links
5. Test different formats

---

## Contact & Support

**Questions about SEO?**
- Email: hello@liveiticonic.com
- Check documentation in `/docs` folder
- Create GitHub issues for technical questions

**Need SEO audit?**
- Review: `/docs/SEO_AUDIT_CHECKLIST.md`
- Schedule monthly reviews
- Quarterly strategy planning

---

**Last Updated**: November 12, 2025
**Next Update**: February 2026
