# Modern Blog Template

A clean, modern, and responsive blog website template built with HTML, CSS, and JavaScript. Features excellent typography, accessibility, SEO optimization, and mobile-first design.

## Features

### 🎨 Design & UX

- **Clean Typography**: Uses Inter font for excellent readability
- **Modern Design**: Minimalist layout with subtle shadows and smooth transitions
- **Mobile-Responsive**: Optimized for all screen sizes with mobile-first approach
- **Dark/Light Mode Ready**: CSS custom properties for easy theming

### 📱 Pages & Components

- **Homepage** (`index.html`): Featured posts, article grid, categories sidebar
- **Article Page** (`article.html`): Individual post with reading progress, social sharing, comments
- **Author Profile** (`author.html`): Author bio, article list, expertise sections
- **Search Page** (`search.html`): Advanced search with filters and results

### ⚡ Functionality

- **Reading Progress Bar**: Visual indicator of article reading progress
- **Social Sharing**: Twitter, LinkedIn, Facebook, and copy link buttons
- **Newsletter Signup**: Email subscription form with validation
- **Comment System**: Static comment display with form submission
- **Search & Filters**: Simulated search with category and date filters
- **Mobile Menu**: Responsive navigation with hamburger menu
- **Load More**: Pagination simulation for article loading

### ♿ Accessibility

- **WCAG Compliant**: Proper ARIA labels, semantic HTML, keyboard navigation
- **Screen Reader Support**: Alt texts, skip links, live regions
- **High Contrast Support**: CSS for high contrast mode
- **Reduced Motion**: Respects user motion preferences

### 🔍 SEO & Performance

- **Schema Markup**: Article and profile structured data
- **Meta Tags**: Open Graph, Twitter Cards, canonical URLs
- **Fast Loading**: Minimal CSS/JS, efficient selectors
- **Print Styles**: Optimized for printing articles

## File Structure

```
templates/blog/
├── index.html          # Homepage with article listings
├── article.html        # Individual article page
├── author.html         # Author profile page
├── search.html         # Search and filter page
├── styles.css          # Main stylesheet
├── script.js           # JavaScript functionality
├── assets/             # Images and static assets
└── README.md           # This documentation
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Usage

1. **Local Development**: Open any HTML file in a modern web browser
2. **Static Hosting**: Upload files to any static web host (Netlify, Vercel, etc.)
3. **Customization**:
   - Replace placeholder images with your own
   - Update content in HTML files
   - Modify colors in CSS custom properties
   - Extend JavaScript for dynamic functionality

## Customization

### Colors

Update CSS custom properties in `:root`:

```css
:root {
  --primary-color: #your-color;
  --text-primary: #your-text-color;
  /* ... */
}
```

### Content

- Replace Lorem ipsum text with your content
- Update author information and bios
- Add your own articles and images
- Modify navigation links

### Functionality

- Connect forms to your backend API
- Implement real search with your CMS
- Add analytics tracking
- Integrate with social media APIs

## JavaScript Features

### Reading Progress

Automatically tracks and displays reading progress on article pages.

### Social Sharing

Pre-configured sharing buttons with fallbacks for mobile devices.

### Form Validation

Client-side validation for newsletter and comment forms.

### Search Simulation

Mock search functionality - replace with real API calls.

### Mobile Interactions

Touch-friendly buttons and responsive navigation.

## Performance Optimizations

- **CSS**: Efficient selectors, minimal nesting
- **JavaScript**: Event delegation, debounced inputs
- **Images**: Placeholder URLs (replace with optimized images)
- **Loading**: Minimal initial JavaScript execution

## Accessibility Features

- Semantic HTML structure
- ARIA labels and roles
- Keyboard navigation support
- Focus management
- Screen reader announcements
- High contrast mode support
- Reduced motion preferences

## SEO Features

- Article schema markup
- Open Graph meta tags
- Twitter Card support
- Canonical URLs
- Structured data for search engines
- Fast loading times

## Development Notes

- Uses modern CSS Grid and Flexbox
- Progressive enhancement approach
- No external dependencies
- Modular JavaScript architecture
- CSS custom properties for theming

## License

This template is provided as-is for educational and commercial use. Modify and distribute as needed.

## Contributing

Feel free to improve the template and submit pull requests with enhancements.
