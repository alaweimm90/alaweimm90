# Favicon & App Icon Generation Guide

## Required Assets for LLM Works

### Core Files Needed
```
/public/
├── favicon.svg          ✅ Created (base SVG)
├── favicon.ico          ❌ Generate from SVG 
├── favicon-16x16.png    ❌ Generate 16×16 PNG
├── favicon-32x32.png    ❌ Generate 32×32 PNG  
├── apple-touch-icon.png ⚠️ Placeholder created (need 180×180 PNG)
└── manifest.json        ❌ Create web app manifest
```

## Generation Instructions

### 1. From favicon.svg to ICO/PNG
```bash
# Using ImageMagick or similar tool
convert favicon.svg -resize 16x16 favicon-16x16.png
convert favicon.svg -resize 32x32 favicon-32x32.png
convert favicon.svg -resize 48x48 favicon-48x48.png

# For favicon.ico (multi-resolution)
convert favicon-16x16.png favicon-32x32.png favicon-48x48.png favicon.ico
```

### 2. Apple Touch Icon (180×180)
- Base: favicon.svg 
- Size: 180×180 pixels
- Style: iOS rounded corners, no transparency
- Background: Solid primary blue (#4F83F0)
- Icon: White analytical prism with orange accent

### 3. Web App Manifest
Create `/public/manifest.json`:
```json
{
  "name": "LLM Works",
  "short_name": "LLM Works",
  "description": "Open-source LLM evaluation platform",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0F1419",
  "theme_color": "#4F83F0",
  "icons": [
    {
      "src": "/favicon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/favicon-512x512.png", 
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

## HTML Updates Required

Add to `<head>` in index.html:
```html
<!-- Standard favicon -->
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">

<!-- Apple touch icon -->
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">

<!-- Web app manifest -->
<link rel="manifest" href="/manifest.json">

<!-- Additional mobile meta -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="LLM Works">
```

## Design Specifications

### Analytical Prism Icon
- **Shape**: Geometric hexagonal prism (isometric view)
- **Colors**: 
  - Primary: #4F83F0 (analytical blue)
  - Accent: #FF7A2A (insight orange)  
  - Light ray: White with 40% opacity
- **Style**: Minimal, clean, technical but approachable
- **Metaphor**: Light/data passing through analytical lens, creating spectrum

### Size Guidelines
- **Minimum readable size**: 16×16 pixels
- **Optimal sizes**: 32×32, 48×48, 128×128, 192×192, 512×512
- **Monochrome fallback**: Single color version for system contexts

## Tools & Services

### Online Generators
- [Favicon.io](https://favicon.io) - Generate from SVG
- [RealFaviconGenerator](https://realfavicongenerator.net) - Comprehensive
- [Figma Favicons Plugin](https://figma.com) - If using Figma design

### Command Line Tools  
```bash
# Install ImageMagick
brew install imagemagick  # macOS
sudo apt install imagemagick  # Ubuntu

# Generate all sizes
./scripts/generate-favicons.sh
```

## Testing Checklist

### Browser Testing
- [ ] Chrome: Displays correctly in tabs and bookmarks
- [ ] Firefox: Icon loads without console errors  
- [ ] Safari: Apple touch icon works on iOS
- [ ] Edge: Favicon appears in browser UI

### Size Testing
- [ ] 16×16: Legible when zoomed out
- [ ] 32×32: Clear details visible
- [ ] 180×180: iOS home screen looks good
- [ ] 512×512: Sharp on high-DPI displays

### Context Testing
- [ ] Browser tabs
- [ ] Bookmark menu
- [ ] iOS home screen  
- [ ] Android home screen
- [ ] Windows taskbar
- [ ] macOS dock

---
*Created: January 12, 2025*  
*Priority: High - impacts brand recognition across all platforms*