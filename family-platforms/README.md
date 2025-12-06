# **Family Platforms Monorepo**

A comprehensive monorepo containing the family's digital presence platforms:
**DrMAlowein** (academic portfolio) and **Rounaq** (fashion e-commerce platform).

## **🏗️ Project Structure**

```text
family-platforms/
├── apps/
│   ├── drmalowein/              # Academic portfolio platform
│   │   ├── src/
│   │   │   ├── components/      # React components
│   │   │   │   ├── layout/      # Header, Footer, Navigation
│   │   │   │   ├── academic/    # Academic-specific components
│   │   │   │   └── ui/          # Reusable UI components
│   │   │   ├── pages/           # Route pages
│   │   │   ├── hooks/           # Custom React hooks
│   │   │   ├── types/           # TypeScript definitions
│   │   │   ├── utils/           # Utility functions
│   │   │   ├── styles/          # CSS/Tailwind styles
│   │   │   └── data/            # Static data and content
│   │   ├── public/              # Static assets
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── tailwind.config.ts
│   └── rounaq/                  # Fashion e-commerce platform
│       ├── src/
│       │   ├── components/
│       │   │   ├── layout/      # Header, Footer, Navigation
│       │   │   ├── commerce/    # E-commerce components
│       │   │   ├── fashion/     # Fashion-specific components
│       │   │   └── ui/          # Reusable UI components
│       │   ├── pages/           # Route pages
│       │   ├── hooks/           # Custom React hooks
│       │   ├── services/        # API services
│       │   ├── types/           # TypeScript definitions
│       │   ├── utils/           # Utility functions
│       │   └── styles/          # CSS/Tailwind styles
│       ├── public/              # Static assets
│       ├── package.json
│       ├── vite.config.ts
│       └── tailwind.config.ts
├── packages/
│   ├── shared/                  # Shared utilities and types
│   └── ui-components/           # Shared React components
├── docs/                        # Documentation
├── scripts/                     # Build and deployment scripts
├── .github/workflows/           # CI/CD workflows
├── package.json                 # Root package.json (workspaces)
├── tsconfig.json               # TypeScript configuration
└── README.md
```

## **🚀 Quick Start**

### **Prerequisites**

- Node.js 18+
- npm 9+
- Git

### **Installation**

```bash
# Clone the repository
git clone https://github.com/alaweimm90/family-platforms.git
cd family-platforms

# Install dependencies
npm install

# Install workspace dependencies
npm run install:all
```

### **Development**

```bash
# Start both applications in development mode
npm run dev

# Start individual applications
npm run dev:drmalowein    # http://localhost:3000
npm run dev:rounaq       # http://localhost:3001
```

### **Building**

```bash
# Build all applications
npm run build

# Build individual applications
npm run build:drmalowein
npm run build:rounaq

# Build for production
npm run build:production
```

### **Testing**

```bash
# Run all tests
npm run test

# Run tests for specific app
npm run test:drmalowein
npm run test:rounaq

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

## **📱 Applications**

### **DrMAlowein - Academic Portfolio**

**Purpose**: Professional academic presence showcasing research, publications,
teaching, and expertise.

**Features**:

- 📚 Publication database with citation tracking
- 🔬 Research project showcase
- 🎓 Teaching portfolio and course materials
- 📊 Academic metrics and impact visualization
- 📄 CV download and generation
- 🔍 Advanced search and filtering
- 📱 Responsive design optimized for academic content

**Tech Stack**:

- React 18 + TypeScript
- Tailwind CSS (Academic theme)
- Strapi CMS for content management
- PostgreSQL for publications database
- Netlify for static hosting
- Google Scholar and ORCID integration

**Development**:

```bash
cd apps/drmalowein
npm run dev
```

### **Rounaq - Fashion Platform**

**Purpose**: Luxury fashion e-commerce platform for mother's design business.

**Features**:

- 🛍️ Product catalog with advanced filtering
- 🛒 Shopping cart and secure checkout
- 👤 Customer accounts and order tracking
- 💝 Wishlist and saved items
- 🎨 Virtual try-on and style recommendations
- 📸 Lookbook and fashion showcases
- 📊 Inventory management and analytics
- 💳 Stripe payment integration

**Tech Stack**:

- React 18 + TypeScript
- Tailwind CSS (Fashion theme)
- Shopify Plus for e-commerce
- Stripe for payment processing
- Vercel for dynamic hosting
- Advanced fashion AI features

**Development**:

```bash
cd apps/rounaq
npm run dev
```

## **🛠️ Development Tools**

### **Code Quality**

```bash
# Lint all code
npm run lint

# Fix linting issues
npm run lint:fix

# Type checking
npm run type-check

# Format code
npm run format
```

### **Workspace Management**

```bash
# Add dependency to specific workspace
npm install <package> --workspace=apps/drmalowein
npm install <package> --workspace=apps/rounaq

# Add dev dependency to all workspaces
npm install <package> --workspaces --save-dev

# Remove dependency
npm uninstall <package> --workspace=apps/drmalowein
```

## **📦 Deployment**

### **Development Deployment**

```bash
# Deploy to development environments
npm run deploy:dev
```

### **Staging**

```bash
# Deploy to staging environments
npm run deploy:staging
```

### **Production**

```bash
# Deploy to production
npm run deploy:production

# Individual deployments
npm run deploy:drmalowein
npm run deploy:rounaq
```

### **Environment Variables**

Create `.env` files in each app directory:

**DrMAlowein (.env)**:

```env
VITE_API_BASE_URL=http://localhost:3001/api
VITE_ENVIRONMENT=development
VITE_ENABLE_ANALYTICS=false
```

**Rounaq (.env)**:

```env
VITE_API_BASE_URL=http://localhost:3002/api
VITE_SHOPIFY_STOREFRONT_TOKEN=your_token
VITE_SHOPIFY_DOMAIN=your-store.myshopify.com
VITE_STRIPE_PUBLISHABLE_KEY=your_key
```

## **🧪 Testing Strategy**

### **Unit Tests**

- Component testing with Vitest
- Utility function testing
- Type safety validation

### **Integration Tests**

- API integration testing
- Cross-component interactions
- Data flow validation

### **E2E Tests**

- User journey testing with Playwright
- Cross-browser compatibility
- Mobile responsiveness

### **Performance Testing**

- Bundle size optimization
- Load time monitoring
- Core Web Vitals tracking

## **📊 Monitoring & Analytics**

### **Application Monitoring**

- Sentry for error tracking
- Custom performance metrics
- User behavior analytics

### **Business Intelligence**

- Google Analytics 4
- Custom conversion tracking
- A/B testing framework

## **🔧 Configuration**

### **TypeScript**

- Shared TypeScript configuration
- Strict type checking
- Path mapping for clean imports

### **Tailwind CSS**

- Custom design systems for each brand
- Responsive breakpoints
- Dark mode support

### **ESLint & Prettier**

- Consistent code formatting
- TypeScript-specific rules
- React best practices

## **🤝 Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `npm run test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## **📄 License**

This project is licensed under the MIT License - see the LICENSE file
for details.

## **👨‍👩‍👧‍👦 Family Project**

This monorepo represents the digital presence of the Alawein family:

- **DrMAlowein**: Father's academic and professional portfolio
- **Rounaq**: Mother's fashion design and e-commerce platform
- **Built with ❤️ by Meshal Alawein**

## **📞 Support**

For questions or support:

- Email: <meshal.alawein@berkeley.edu>
- GitHub Issues: [Create an issue](https://github.com/alaweimm90/family-platforms/issues)

---

**Last Updated**: December 6, 2025  
**Version**: 1.0.0  
**Status**: In Development
