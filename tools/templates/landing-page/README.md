# Landing Page Template

A modern, responsive landing page template using Vite + React + Tailwind CSS.

## Features

- ✅ Vite for fast development
- ✅ React 18 with TypeScript
- ✅ Tailwind CSS with custom theming
- ✅ Framer Motion animations
- ✅ SEO-optimized meta tags
- ✅ Mobile-first responsive design
- ✅ Dark mode support
- ✅ Contact form with validation
- ✅ GitHub Pages deployment ready

## Usage

```bash
# Copy template
cp -r templates/landing-page my-product-landing
cd my-product-landing

# Replace placeholders
find . -type f -exec sed -i 's/{{PRODUCT_NAME}}/My Product/g' {} \;
find . -type f -exec sed -i 's/{{TAGLINE}}/The best product ever/g' {} \;

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## Structure

```
landing-page/
├── src/
│   ├── components/
│   │   ├── Hero.tsx
│   │   ├── Features.tsx
│   │   ├── Pricing.tsx
│   │   ├── Testimonials.tsx
│   │   ├── CTA.tsx
│   │   └── Footer.tsx
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
│   └── favicon.svg
├── index.html
├── package.json
├── tailwind.config.ts
├── vite.config.ts
└── tsconfig.json
```

## Customization

### Colors

Edit `tailwind.config.ts` to change the color scheme:

```typescript
theme: {
  extend: {
    colors: {
      primary: '#YOUR_PRIMARY_COLOR',
      secondary: '#YOUR_SECONDARY_COLOR',
    }
  }
}
```

### Sections

Enable/disable sections in `App.tsx`:

```tsx
<Hero />
<Features />        {/* Comment out to remove */}
<Pricing />         {/* Comment out to remove */}
<Testimonials />    {/* Comment out to remove */}
<CTA />
<Footer />
```

