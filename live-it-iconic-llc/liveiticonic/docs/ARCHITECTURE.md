# Live It Iconic - Architecture Documentation

**Version:** 1.0.0
**Last Updated:** 2025-11-11
**Status:** Production

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture Principles](#architecture-principles)
4. [Directory Structure](#directory-structure)
5. [Component Architecture](#component-architecture)
6. [Data Flow](#data-flow)
7. [State Management](#state-management)
8. [API Integration Layer](#api-integration-layer)
9. [Launch Platform](#launch-platform)
10. [Security Architecture](#security-architecture)
11. [Performance Optimizations](#performance-optimizations)
12. [Deployment Architecture](#deployment-architecture)
13. [Scalability Considerations](#scalability-considerations)

---

## System Overview

Live It Iconic is a **high-performance e-commerce platform** for a luxury automotive lifestyle brand, featuring a sophisticated AI-powered launch orchestration system.

### Key Components

1. **Frontend Application** - React 18 + TypeScript SPA
2. **Backend Services** - Supabase (PostgreSQL, Auth, Storage, Edge Functions)
3. **Payment Processing** - Stripe integration
4. **AI Launch Platform** - 26 specialized AI agents
5. **Content Delivery** - Optimized assets with lazy loading
6. **Analytics** - Event tracking and user behavior monitoring

### Architecture Style

- **Monorepo:** Single repository with clear module boundaries
- **Component-Based:** Reusable React components with composition
- **Service-Oriented:** Business logic separated into services
- **Event-Driven:** Launch platform uses event bus for agent communication
- **API-First:** Backend communicates through well-defined APIs

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3.1 | UI framework |
| **TypeScript** | 5.8.3 | Type safety |
| **Vite** | 7.2.2 | Build tool & dev server |
| **React Router** | 6.30.1 | Client-side routing |
| **TanStack Query** | 5.83.0 | Server state management |
| **Tailwind CSS** | 3.4.17 | Utility-first CSS |
| **Radix UI** | Various | Accessible component primitives |
| **Lucide React** | 0.462.0 | Icon library |

### Backend

| Technology | Purpose |
|------------|---------|
| **Supabase** | Backend-as-a-Service |
| **PostgreSQL** | Primary database |
| **Supabase Auth** | Authentication & authorization |
| **Supabase Storage** | File storage |
| **Edge Functions** | Serverless functions |

### Payment

| Technology | Purpose |
|------------|---------|
| **Stripe** | Payment processing |
| **Stripe Elements** | Secure payment forms |

### Testing

| Technology | Purpose |
|------------|---------|
| **Vitest** | Unit & integration testing |
| **Playwright** | End-to-end testing |
| **Testing Library** | React component testing |

### Build & DevOps

| Technology | Purpose |
|------------|---------|
| **Vite** | Build & bundling |
| **ESLint** | Code linting |
| **Prettier** | Code formatting |
| **GitHub Actions** | CI/CD |

---

## Architecture Principles

### 1. Single Responsibility
- Each component/service has one clear purpose
- Separation of concerns between presentation and business logic

### 2. DRY (Don't Repeat Yourself)
- Reusable components and utilities
- Shared types across application
- Consolidated business logic in services

### 3. Composition Over Inheritance
- React components composed from smaller components
- Radix UI primitives for consistent behavior

### 4. Type Safety
- TypeScript throughout
- Strict type checking enabled
- Shared type definitions

### 5. Progressive Enhancement
- Core functionality works without JavaScript
- Enhanced experiences for modern browsers
- Graceful degradation

### 6. Performance First
- Code splitting and lazy loading
- Optimized bundle sizes
- Image optimization
- Caching strategies

### 7. Accessibility
- WCAG 2.1 AA compliance
- Semantic HTML
- Keyboard navigation
- Screen reader support

---

## Directory Structure

```
src/
├── components/              # React components
│   ├── ui/                 # Radix UI base components
│   ├── product/            # Product-specific components
│   ├── checkout/           # Checkout flow components
│   ├── admin/              # Admin dashboard components
│   ├── logo/               # Brand logo components
│   ├── brandmarks/         # Brand mark components
│   └── icons/              # Icon components
│
├── pages/                  # Route pages
│   ├── admin/              # Admin pages
│   ├── Index.tsx           # Homepage
│   ├── Shop.tsx            # Product listing
│   ├── ProductDetail.tsx   # Product detail
│   ├── Checkout.tsx        # Checkout flow
│   └── ...                 # Other pages
│
├── launch-platform/        # AI Agent System (54 files)
│   ├── core/               # Core infrastructure
│   │   ├── BaseAgent.ts    # Abstract base agent
│   │   ├── EventBus.ts     # Event-driven communication
│   │   ├── StateManager.ts # State management
│   │   └── LaunchOrchestrator.ts # Orchestration engine
│   │
│   ├── agents/             # 26 specialized agents
│   │   ├── market/         # Market intelligence (5 agents)
│   │   ├── creative/       # Creative & branding (5 agents)
│   │   ├── execution/      # Launch execution (6 agents)
│   │   ├── optimization/   # Optimization (5 agents)
│   │   └── supporting/     # Supporting (5 agents)
│   │
│   ├── types/              # 80+ TypeScript types
│   ├── cli/                # CLI tools
│   └── examples/           # Demo & examples
│
├── api/                    # API integration layer
├── services/               # Business logic services
│   ├── productService.ts   # Product operations
│   ├── orderService.ts     # Order management
│   ├── paymentService.ts   # Payment processing
│   ├── authService.ts      # Authentication
│   ├── emailService.ts     # Email notifications
│   └── ...
│
├── contexts/               # React contexts
│   ├── AuthContext.tsx     # Authentication state
│   └── CartContext.tsx     # Shopping cart state
│
├── hooks/                  # Custom React hooks
│   ├── use-cart.ts         # Cart operations
│   ├── useAnalytics.ts     # Analytics tracking
│   ├── use-mobile.tsx      # Mobile detection
│   └── ...
│
├── types/                  # TypeScript type definitions
│   ├── product.ts          # Product types
│   ├── order.ts            # Order types
│   ├── cart.ts             # Cart types
│   └── ...
│
├── utils/                  # Utility functions
│   ├── analytics.ts        # Analytics utilities
│   └── seo.ts              # SEO utilities
│
├── integrations/           # External integrations
│   └── supabase/           # Supabase client & types
│
├── theme/                  # Design system
│   └── tokens.ts           # Design tokens
│
├── assets/                 # Static assets
└── styles/                 # Global styles
```

---

## Component Architecture

### Component Hierarchy

```
App
├── Navigation (global)
├── Routes
│   ├── Index (Homepage)
│   │   ├── Hero
│   │   ├── ProductShowcase
│   │   ├── FeaturedCollections
│   │   ├── AthleteShowcase
│   │   ├── SupercarGallery
│   │   └── EmailCapture
│   │
│   ├── Shop (Product Listing)
│   │   ├── ProductGrid
│   │   ├── ProductFilters
│   │   └── Pagination
│   │
│   ├── ProductDetail
│   │   ├── ProductImages
│   │   ├── ProductInfo
│   │   ├── ProductVariants
│   │   ├── AddToCartButton
│   │   └── RelatedProducts
│   │
│   ├── Checkout
│   │   ├── CheckoutForm
│   │   ├── OrderSummary
│   │   └── PaymentForm (Stripe)
│   │
│   └── Admin
│       ├── Dashboard
│       ├── OrdersTable
│       └── ProductsTable
│
├── CartDrawer (global)
└── Footer (global)
```

### Component Patterns

#### 1. Compound Components
```tsx
<Accordion>
  <AccordionItem value="item-1">
    <AccordionTrigger>Title</AccordionTrigger>
    <AccordionContent>Content</AccordionContent>
  </AccordionItem>
</Accordion>
```

#### 2. Render Props
```tsx
<DataFetcher url="/api/products">
  {({ data, loading, error }) => (
    loading ? <Loader /> : <ProductGrid products={data} />
  )}
</DataFetcher>
```

#### 3. Hooks Pattern
```tsx
const { items, addItem, removeItem } = useCart();
const { user, signIn, signOut } = useAuth();
```

#### 4. Provider Pattern
```tsx
<AuthProvider>
  <CartProvider>
    <App />
  </CartProvider>
</AuthProvider>
```

---

## Data Flow

### Request Flow

```
User Action
    ↓
Component Event Handler
    ↓
Service Layer (Business Logic)
    ↓
API Integration Layer
    ↓
Supabase / External API
    ↓
Response Processing
    ↓
State Update (Context/TanStack Query)
    ↓
Component Re-render
```

### Example: Add to Cart Flow

```
1. User clicks "Add to Cart" button
2. ProductDetail component calls addToCart()
3. CartContext updates local state
4. API call to /api/cart/add (persisted to backend)
5. Success: Update cart count in Navigation
6. Show success toast notification
7. Analytics event tracked
```

---

## State Management

### State Categories

| State Type | Solution | Example |
|------------|----------|---------|
| **Server State** | TanStack Query | Products, orders, user data |
| **Global Client State** | React Context | Auth, cart |
| **Local Component State** | useState | Form inputs, UI toggles |
| **URL State** | React Router | Filters, pagination, search |
| **Form State** | React Hook Form | Checkout form, search form |

### Context Architecture

#### AuthContext
```tsx
interface AuthContextType {
  user: User | null;
  loading: boolean;
  signIn: (credentials) => Promise<void>;
  signUp: (data) => Promise<void>;
  signOut: () => Promise<void>;
}
```

#### CartContext
```tsx
interface CartContextType {
  items: CartItem[];
  itemCount: number;
  total: number;
  addItem: (product, quantity) => void;
  removeItem: (itemId) => void;
  updateQuantity: (itemId, quantity) => void;
  clearCart: () => void;
}
```

---

## API Integration Layer

### Service Pattern

```typescript
// services/productService.ts
export const productService = {
  async getProducts(filters?: ProductFilters): Promise<Product[]> {
    const { data, error } = await supabase
      .from('products')
      .select('*')
      .match(filters);

    if (error) throw error;
    return data;
  },

  async getProduct(id: string): Promise<Product | null> {
    const { data, error } = await supabase
      .from('products')
      .select('*')
      .eq('id', id)
      .single();

    if (error) throw error;
    return data;
  }
};
```

### TanStack Query Integration

```typescript
// Component usage
const { data: products, isLoading } = useQuery({
  queryKey: ['products', filters],
  queryFn: () => productService.getProducts(filters),
  staleTime: 5 * 60 * 1000, // 5 minutes
});
```

---

## Launch Platform

### Agent Architecture

The Launch Platform is an **event-driven, multi-agent system** for product launch orchestration.

#### Core Components

1. **BaseAgent** - Abstract base class for all agents
2. **EventBus** - Pub/sub communication system
3. **StateManager** - Centralized state management
4. **LaunchOrchestrator** - Coordinates agent workflows

#### Agent Categories (26 Total)

| Category | Count | Examples |
|----------|-------|----------|
| **Market Intelligence** | 5 | CompetitorAnalyst, TrendDetector |
| **Creative & Branding** | 5 | BrandArchitect, CopyWriter |
| **Launch Execution** | 6 | CampaignManager, SocialMediaStrategist |
| **Optimization** | 5 | AnalyticsInterpreter, SEOSpecialist |
| **Supporting** | 5 | DataCollector, QualityController |

#### Event Flow

```
LaunchOrchestrator
    ↓ (broadcasts)
EventBus
    ↓ (distributes)
Agents (subscribe to relevant events)
    ↓ (execute tasks)
StateManager (updates launch state)
    ↓ (notifies)
Orchestrator (coordinates next phase)
```

#### Agent Communication

```typescript
// Agent A broadcasts event
await this.eventBus.broadcast('competitor-analysis-complete', {
  competitors: [...],
  insights: [...]
});

// Agent B subscribes and responds
this.eventBus.subscribe('competitor-analysis-complete', async (event) => {
  // Use competitor data for pricing strategy
  await this.calculatePricing(event.data.competitors);
});
```

---

## Security Architecture

### Authentication & Authorization

- **Method:** Supabase Auth (JWT-based)
- **Storage:** HTTP-only cookies + localStorage (refresh tokens)
- **Roles:** User, Admin
- **Row-Level Security:** PostgreSQL RLS policies

### API Security

- **HTTPS Only:** All traffic encrypted
- **CORS:** Configured for allowed origins
- **Rate Limiting:** 100-1000 req/min depending on endpoint type
- **Input Validation:** Zod schemas for all forms
- **SQL Injection:** Prevented by Supabase query builder
- **XSS Protection:** React's built-in escaping + CSP headers

### Payment Security

- **PCI Compliance:** Stripe handles all card data
- **No Card Storage:** Card details never touch our servers
- **Stripe Elements:** Tokenization before transmission
- **3D Secure:** Supported for additional security

---

## Performance Optimizations

### Code Splitting

```typescript
// Lazy load routes
const ProductDetail = lazy(() => import('./pages/ProductDetail'));
const Checkout = lazy(() => import('./pages/Checkout'));
```

### Bundle Optimization

- **React Vendor:** 272.27 KB (optimized)
- **Main Vendor:** 122.37 KB
- **CSS:** 115.35 KB
- **Code Splitting:** Feature-based chunks (cart, checkout)

### Image Optimization

- Responsive images (multiple sizes)
- Lazy loading with Intersection Observer
- WebP format (future)
- CDN delivery (future)

### Caching Strategy

| Resource | Strategy | TTL |
|----------|----------|-----|
| Products | stale-while-revalidate | 5 min |
| User data | fresh on mount | - |
| Static assets | Cache-first | 1 year |
| API responses | Network-first | - |

---

## Deployment Architecture

### Hosting

- **Frontend:** Lovable Cloud / Vercel / Netlify
- **Backend:** Supabase (managed PostgreSQL + Edge Functions)
- **CDN:** Cloudflare / CloudFront (future)

### Environments

| Environment | Purpose | URL |
|------------|---------|-----|
| **Development** | Local development | localhost:8080 |
| **Staging** | Pre-production testing | staging.liveiticon.com |
| **Production** | Live application | liveiticon.com |

### CI/CD Pipeline

```
git push
    ↓
GitHub Actions
    ↓
├── Lint (ESLint)
├── Type Check (TypeScript)
├── Test (Vitest + Playwright)
├── Build (Vite)
└── Security Scan (npm audit)
    ↓
Deploy to Staging
    ↓
Manual Approval
    ↓
Deploy to Production
```

---

## Scalability Considerations

### Current Capacity

- **Concurrent Users:** 1,000-10,000
- **Products:** 100-1,000
- **Orders/day:** 100-1,000

### Scaling Strategy

#### Vertical Scaling (Phase 1)
- Increase Supabase tier
- Upgrade database resources

#### Horizontal Scaling (Phase 2)
- Read replicas for database
- CDN for static assets
- Edge functions for global distribution

#### Optimization (Ongoing)
- Database indexing
- Query optimization
- Caching improvements
- Image optimization

---

## Monitoring & Observability

### Metrics

- **Performance:** Core Web Vitals (LCP, FID, CLS)
- **Availability:** Uptime monitoring
- **Errors:** Error tracking and aggregation
- **Business:** Conversion rates, cart abandonment

### Tools (Future)

- **APM:** Datadog / New Relic
- **Logging:** Supabase logs / LogRocket
- **Error Tracking:** Sentry
- **Analytics:** Segment / Google Analytics

---

## Future Enhancements

### Short-term (Q1 2026)
- Consolidate src/ and platform/ directories
- Achieve 80%+ test coverage
- Implement pre-commit hooks
- Add comprehensive documentation

### Medium-term (Q2-Q3 2026)
- Real-time inventory updates (WebSockets)
- Advanced search (Algolia)
- Personalization engine
- Mobile app (React Native)

### Long-term (Q4 2026+)
- Multi-region deployment
- Microservices architecture
- GraphQL API
- AI-powered recommendations

---

## References

- [Phase 1 Discovery Summary](../reports/PHASE1-DISCOVERY-SUMMARY.md)
- [ADR-001: Consolidation Decision](./adr/ADR-001-consolidate-src-platform.md)
- [API Documentation](./API.md)
- [Hybrid Approach Strategy](../reports/HYBRID_APPROACH.md)

---

**Maintained by:** Engineering Team
**Last Review:** 2025-11-11
**Next Review:** 2026-01-11
