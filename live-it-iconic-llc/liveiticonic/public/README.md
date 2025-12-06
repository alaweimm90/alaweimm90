# Public Assets Directory

This directory contains static assets that are served directly by the web server for the **Live It Iconic** e-commerce platform.

## Directory Structure

```
public/
├── images/                # Product images and brand assets
│   ├── products/         # Product photography
│   ├── brand/            # Logo variations
│   └── content/          # YouTube thumbnails, social media
├── icons/                # App icons and favicons
├── fonts/                # Web fonts (DM Sans, Inter)
├── sitemap.xml           # SEO sitemap
├── robots.txt            # Search engine crawler instructions
└── manifest.json         # PWA manifest (optional future enhancement)
```

## SEO Files

### sitemap.xml

Helps search engines discover and index pages on the Live It Iconic e-commerce site.

**Key Pages Included:**
- Homepage (`/`)
- Shop catalog (`/shop`)
- Individual product pages (`/products/*`)
- Brand showcase (`/brand-showcase`)
- About page (`/about`)
- Contact page (`/contact`)

**Update Frequency:** Weekly (when new products are added)

### robots.txt

Controls search engine crawler access and behavior.

**Current Configuration:**
```txt
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /checkout/
Disallow: /api/

Sitemap: https://liveiticonic.com/sitemap.xml
```

**Key Rules:**
- Allow all public pages
- Block admin dashboard from indexing
- Block checkout pages (privacy)
- Block API endpoints
- Point to sitemap location

## Brand Assets

### Product Images

Location: `images/products/`

**Guidelines:**
- Format: WebP primary, PNG/JPG fallback
- Dimensions: 1200x1200px (square)
- File size: < 200 KB (compressed)
- Naming: `{sku}-{variant}.webp` (e.g., `cap-black-01.webp`)
- Multiple angles: Front, back, detail, lifestyle

**Current Products:**
- Black Cap ($29) - `cap-black-*.webp`
- Black Hoodie ($79) - `hoodie-black-*.webp`
- Black T-Shirt ($49) - `tshirt-black-*.webp`

### Logo Files

Location: `images/brand/`

**Available Variants:**
1. **Wordmark** - `logo-wordmark.svg` (full "LIVE IT ICONIC")
2. **Stacked** - `logo-stacked.svg` (two-line version)
3. **Monogram** - `logo-monogram.svg` (LII)
4. **Icon** - `logo-icon.svg` (stylized "I")

**Color Versions:**
- Black (`*-black.svg`) - Primary
- White (`*-white.svg`) - On dark backgrounds
- Gold (`*-gold.svg`) - Accent use

### Icons

Location: `icons/`

**PWA Icons:**
- `icon-192x192.png` - Android home screen
- `icon-512x512.png` - Android splash screen
- `favicon.ico` - Browser tab icon
- `apple-touch-icon.png` - iOS home screen

**Brand Colors:**
- Iconic Black: #0A0A0A
- Iconic Gold: #D4AF37

## Web Fonts

Location: `fonts/`

**Primary Font - DM Sans:**
- `DMSans-Regular.woff2`
- `DMSans-Medium.woff2`
- `DMSans-Bold.woff2`

Used for: Headings, UI elements, CTAs

**Secondary Font - Inter:**
- `Inter-Regular.woff2`
- `Inter-Medium.woff2`

Used for: Body text, descriptions

**Logo Font - Montserrat:**
- `Montserrat-Bold.woff2`

Used for: Logo (all caps)

**Font Loading:**
```css
@font-face {
  font-family: 'DM Sans';
  src: url('/fonts/DMSans-Regular.woff2') format('woff2');
  font-display: swap; /* Prevent FOIT */
}
```

## Image Optimization

### Best Practices

1. **Use WebP Format**
   - 25-35% smaller than PNG/JPG
   - Excellent quality
   - Fallback to PNG/JPG for older browsers

2. **Compress Images**
   - Tools: Squoosh, ImageOptim, TinyPNG
   - Target: < 200 KB for product images
   - Maintain quality > 85%

3. **Responsive Images**
   ```html
   <picture>
     <source srcset="/images/products/cap-black-01.webp" type="image/webp">
     <img src="/images/products/cap-black-01.jpg" alt="Black Cap">
   </picture>
   ```

4. **Lazy Loading**
   ```html
   <img src="product.jpg" loading="lazy" alt="Product">
   ```

## File Naming Conventions

### Product Images
```
{sku}-{color}-{angle}.{ext}
Examples:
- cap-black-front.webp
- hoodie-black-back.webp
- tshirt-black-detail.webp
```

### Brand Assets
```
logo-{variant}-{color}.svg
Examples:
- logo-wordmark-black.svg
- logo-stacked-white.svg
- logo-monogram-gold.svg
```

### Content Images
```
{content-type}-{id}-{description}.{ext}
Examples:
- youtube-thumbnail-001-interview.jpg
- social-post-instagram-001.jpg
```

## Performance Optimization

### Server Configuration

**Recommended Headers:**
```
# Cache static assets for 1 year
Cache-Control: public, max-age=31536000, immutable

# Enable compression
Content-Encoding: gzip / brotli

# Security headers
X-Content-Type-Options: nosniff
```

### CDN Recommendations

For production, use a CDN to serve static assets:
- **Vercel** - Automatic CDN for all public files
- **Cloudflare** - Additional caching layer
- **Image CDN** - Cloudinary, Imgix for product images

### Image Sizes

**Product Catalog:**
- Thumbnail: 300x300px (< 50 KB)
- Large: 1200x1200px (< 200 KB)

**Brand Assets:**
- Social media: 1200x630px (Open Graph)
- Email: 600px wide (< 100 KB)

## SEO Best Practices

### Meta Tags

All pages include:
```html
<meta name="description" content="Premium lifestyle merchandise...">
<meta property="og:image" content="/images/brand/og-image.jpg">
<meta property="og:title" content="Live It Iconic">
<meta name="twitter:card" content="summary_large_image">
```

### Structured Data

Product pages include JSON-LD:
```json
{
  "@context": "https://schema.org/",
  "@type": "Product",
  "name": "Black Hoodie",
  "image": "/images/products/hoodie-black-front.webp",
  "description": "Premium quality hoodie...",
  "sku": "HOODIE-BLK",
  "offers": {
    "@type": "Offer",
    "priceCurrency": "USD",
    "price": "79.00",
    "availability": "https://schema.org/InStock"
  }
}
```

## Security Considerations

### File Validation

- Only accept specific image types (WebP, PNG, JPG, SVG)
- Validate file sizes (max 5 MB)
- Scan for malware in uploads
- Use secure S3 bucket or Supabase Storage for user uploads

### CORS Configuration

```javascript
// Only allow requests from own domain
Access-Control-Allow-Origin: https://liveiticonic.com
```

### Content Security Policy

```
img-src 'self' https://supabase.co https://stripe.com;
font-src 'self';
```

## Adding New Assets

### Product Images

1. **Photograph product** with consistent lighting and background
2. **Edit photos** - Remove background, adjust colors
3. **Export multiple formats**:
   ```bash
   # Convert to WebP
   cwebp input.png -o output.webp -q 85
   ```
4. **Upload to** `public/images/products/`
5. **Update product database** with image URLs

### Brand Assets

1. Create asset following brand guidelines
2. Export as SVG (vector) or high-res PNG
3. Optimize:
   ```bash
   # SVG
   svgo input.svg -o output.svg

   # PNG
   optipng input.png
   ```
4. Upload to `public/images/brand/`
5. Update brand documentation

## Accessibility

### Alt Text

Always provide descriptive alt text:
```html
<!-- ❌ Bad -->
<img src="product.jpg" alt="Product">

<!-- ✅ Good -->
<img src="hoodie-black.webp" alt="Black hoodie with Live It Iconic logo, front view">
```

### File Naming

Use descriptive, human-readable names:
- `black-hoodie-front-view.webp` ✅
- `IMG_12345.jpg` ❌

## Maintenance

### Weekly Tasks
- [ ] Update sitemap.xml with new products
- [ ] Compress and optimize new images
- [ ] Verify broken image links
- [ ] Check CDN cache invalidation

### Monthly Tasks
- [ ] Audit unused assets (remove)
- [ ] Review image file sizes (optimize)
- [ ] Update brand assets if needed
- [ ] Verify SEO meta tags

### Quarterly Tasks
- [ ] Full asset audit
- [ ] Performance testing (Lighthouse)
- [ ] Update web fonts if needed
- [ ] Review CDN costs

## Resources

- [Brand Identity Guidelines](../.brand/identity/BRAND_IDENTITY_SYSTEM.md)
- [WebP Conversion](https://developers.google.com/speed/webp)
- [Image Optimization](https://web.dev/fast/)
- [SEO Best Practices](https://developers.google.com/search/docs)

---

For more details, see:
- [PROJECT.MD](../PROJECT.md) - Project overview
- [SEO Strategy](../docs/SEO_STRATEGY.md) - Complete SEO guide
- [Brand Assets](../.brand/README.md) - Brand asset management
