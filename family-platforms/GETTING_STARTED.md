# **🚀 GETTING STARTED GUIDE**

## **Quick Setup for Family Platforms Development**

This guide will help you get the family platforms development environment
set up and running in minutes.

---

## **📋 Prerequisites**

### **Required Software**

- **Node.js** 18.0.0 or higher
- **npm** 9.0.0 or higher  
- **Git** for version control
- **VS Code** (recommended) with these extensions:
  - TypeScript and JavaScript Language Features
  - Tailwind CSS IntelliSense
  - ES7+ React/Redux/React-Native snippets
  - Prettier - Code formatter
  - ESLint

### **Optional Tools**

- **Docker** for containerized development
- **PostgreSQL** for local database (DrMAlowein)
- **Shopify CLI** for e-commerce development (Rounaq)

---

## **⚡ 5-Minute Quick Start**

### **1. Clone & Install**

```bash
# Clone the repository
git clone https://github.com/alaweimm90/family-platforms.git
cd family-platforms

# Install all dependencies (this takes 2-3 minutes)
npm install
npm run install:all
```

### **2. Environment Setup**

```bash
# Copy environment templates
cp apps/drmalowein/.env.example apps/drmalowein/.env
cp apps/rounaq/.env.example apps/rounaq/.env

# Edit the files with your local settings
# (For now, defaults work fine for development)
```

### **3. Start Development**

```bash
# Start both applications simultaneously
npm run dev

# Or start individually:
npm run dev:drmalowein    # http://localhost:3000
npm run dev:rounaq       # http://localhost:3001
```

**🎉 That's it! Both platforms are now running locally.**

---

## **🏗️ Project Structure Overview**

```text
family-platforms/
├── apps/
│   ├── drmalowein/          # Father's academic portfolio
│   │   ├── src/components/  # React components
│   │   ├── src/pages/       # Page routes
│   │   ├── src/types/       # TypeScript definitions
│   │   └── src/styles/      # Academic theme styles
│   └── rounaq/              # Mother's fashion platform
│       ├── src/components/  # React components
│       ├── src/pages/       # Page routes
│       ├── src/types/       # TypeScript definitions
│       └── src/styles/      # Fashion theme styles
├── packages/                # Shared code
└── docs/                    # Documentation
```

---

## **🎯 Development Workflows**

### **Daily Development**

```bash
# 1. Start development servers
npm run dev

# 2. Make your changes
# Edit files in apps/drmalowein/ or apps/rounaq/

# 3. Run tests
npm run test

# 4. Check code quality
npm run lint
npm run type-check

# 5. Format code
npm run format
```

### **Building for Deployment**

```bash
# Build for different environments
npm run build:development    # Development build
npm run build:staging       # Staging build  
npm run build:production    # Production build
```

### **Testing**

```bash
# Run all tests
npm run test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

---

## **📱 Application-Specific Setup**

### **DrMAlowein (Academic Portfolio)**

**Purpose**: Professional academic presence with publications, research, and teaching.

**Key Features**:

- Publication database with citation tracking
- Research project showcase  
- Teaching portfolio
- CV download functionality
- Academic search and filtering

**Development**:

```bash
cd apps/drmalowein
npm run dev

# View at: http://localhost:3000
```

**Key Files to Edit**:

- `src/data/publications.json` - Add/update publications
- `src/data/research.json` - Add research projects
- `src/data/profile.json` - Update academic profile
- `src/pages/` - Add new pages
- `src/components/academic/` - Academic-specific components

### **Rounaq (Fashion Platform)**

**Purpose**: Luxury fashion e-commerce for mother's design business.

**Key Features**:

- Product catalog with filtering
- Shopping cart and checkout
- Customer accounts
- Wishlist functionality
- Style recommendations

**Development**:

```bash
cd apps/rounaq
npm run dev

# View at: http://localhost:3001
```

**Key Files to Edit**:

- `src/data/products.json` - Add/update products
- `src/data/collections.json` - Manage collections
- `src/pages/` - Add new pages
- `src/components/commerce/` - E-commerce components
- `src/components/fashion/` - Fashion-specific components

---

## **🎨 Styling & Theming**

### **Design Systems**

**DrMAlowein Academic Theme**:

- Colors: Academic blues, professional grays
- Typography: Georgia (headings), Inter (body)
- Focus: Professional, scholarly, clean

**Rounaq Fashion Theme**:

- Colors: Fashion pinks, purples, luxury golds
- Typography: Playfair Display (headings), Source Sans Pro (body)
- Focus: Elegant, modern, fashion-forward

### **Customizing Styles**

```bash
# Edit Tailwind configs
apps/drmalowein/tailwind.config.ts
apps/rounaq/tailwind.config.ts

# Edit CSS variables
apps/drmalowein/src/styles/globals.css
apps/rounaq/src/styles/globals.css
```

---

## **🔧 Common Development Tasks**

### **Adding a New Page**

1. Create the page component:

```typescript
// apps/drmalowein/src/pages/NewPage.tsx
import React from 'react';

export const NewPage: React.FC = () => {
  return (
    <div className="academic-container">
      <h1 className="text-3xl font-heading text-academic-blue">
        New Page
      </h1>
      <p>Your content here</p>
    </div>
  );
};
```

1. Add to router in `App.tsx`:

```typescript
import { NewPage } from './pages/NewPage';

// Add route:
<Route path="/new-page" element={<NewPage />} />
```

### **Adding a New Component**

1. Create component:

```typescript
// apps/drmalowein/src/components/NewComponent.tsx
import React from 'react';

interface NewComponentProps {
  title: string;
  children: React.ReactNode;
}

export const NewComponent: React.FC<NewComponentProps> = ({ 
  title, 
  children 
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <h2 className="text-xl font-heading text-academic-blue mb-4">
        {title}
      </h2>
      {children}
    </div>
  );
};
```

1. Export from index:

```typescript
// apps/drmalowein/src/components/index.ts
export { NewComponent } from './NewComponent';
```

### **Adding New Data**

1. Update TypeScript types in `types/`
2. Add data to `data/` directory
3. Create components to display data
4. Add pages/routes as needed

---

## **🐛 Troubleshooting**

### **Common Issues**

**Port already in use**:

```bash
# Kill processes on ports 3000-3001
lsof -ti:3000 | xargs kill -9
lsof -ti:3001 | xargs kill -9
```

**Dependency issues**:

```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
npm run install:all
```

**TypeScript errors**:

```bash
# Check types
npm run type-check

# Update types
npm run type-check -- --force
```

**Build failures**:

```bash
# Clean build
npm run clean
npm run build
```

### **Getting Help**

1. Check the console for error messages
2. Look at the GitHub Issues page
3. Check the documentation in `docs/`
4. Contact Meshal at <meshal.alawein@berkeley.edu>

---

## **📚 Next Steps**

### **For New Developers**

1. **Explore the codebase** - Look through existing components and pages
2. **Read the TypeScript types** - Understand the data structures
3. **Run the tests** - See how testing works
4. **Make a small change** - Try adding a new page or component
5. **Submit a PR** - Learn the contribution process

### **For Content Updates**

1. **DrMAlowein**: Edit files in `src/data/` for publications, research, profile
2. **Rounaq**: Edit files in `src/data/` for products, collections
3. **Images**: Add to `public/` directories
4. **Styles**: Modify Tailwind configs and CSS files

### **For Advanced Features**

1. **API Integration**: Add to `src/services/` or `src/hooks/`
2. **Database**: Set up PostgreSQL for DrMAlowein
3. **E-commerce**: Configure Shopify for Rounaq
4. **Analytics**: Add Google Analytics or Segment
5. **SEO**: Implement meta tags and structured data

---

## **🎯 Development Best Practices**

### **Code Quality**

- Follow TypeScript strict mode
- Use ESLint and Prettier
- Write tests for new features
- Document complex components

### **Performance**

- Optimize images and assets
- Use lazy loading for heavy content
- Monitor bundle sizes
- Test on mobile devices

### **Accessibility**

- Use semantic HTML
- Add ARIA labels
- Test with screen readers
- Ensure keyboard navigation

### **Security**

- Validate all inputs
- Use HTTPS in production
- Keep dependencies updated
- Follow OWASP guidelines

---

## **🎉 Happy coding! 🚀**

This is a family project built with love and professional standards.
If you need help or have questions, don't hesitate to reach out.

**Last Updated**: December 6, 2025  
**Version**: 1.0.0
