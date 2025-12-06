# Source Code Directory

This directory contains the main source code for **Live It Iconic**, a lifestyle brand e-commerce platform combining merchandise sales, content creation, and community building.

## Directory Structure

```
src/
├── pages/                 # Page components (routes)
│   ├── Shop.tsx          # Product catalog
│   ├── ProductDetail.tsx # Single product view
│   ├── Checkout.tsx      # Payment flow
│   ├── OrderConfirmation.tsx
│   ├── BrandShowcase.tsx # YouTube/Twitch integration
│   ├── Lifestyle.tsx     # Content hub
│   └── admin/            # Admin pages
├── components/           # React components
│   ├── checkout/        # Payment components
│   ├── product/         # Product display
│   ├── admin/           # Admin controls
│   ├── ui/              # Shared UI (shadcn)
│   ├── logo/            # Logo variants
│   └── seo/             # SEO components
├── services/            # Business logic services
│   ├── stripeService.ts      # Payment processing
│   ├── productService.ts     # Product queries
│   ├── orderService.ts       # Order management
│   ├── inventoryService.ts   # Stock tracking
│   ├── emailService.ts       # Email notifications
│   ├── paymentService.ts     # Payment operations
│   ├── authService.ts        # Authentication
│   └── adminService.ts       # Admin operations
├── types/               # TypeScript type definitions
│   ├── product.ts       # Product interfaces
│   ├── order.ts         # Order interfaces
│   ├── cart.ts          # Shopping cart
│   └── api.ts           # API types
├── middleware/          # Request/response middleware
│   ├── auth.ts          # Authentication middleware
│   ├── rateLimit.ts     # API rate limiting
│   └── validation.ts    # Input validation
├── utils/               # Utility functions
├── hooks/               # Custom React hooks
├── contexts/            # React Context providers
├── api/                 # API integration layer
├── emails/              # Email templates (React Email)
├── launch-platform/     # Content launch automation
└── design-tokens/       # Brand colors, typography
```

## Architecture Overview

### E-Commerce Platform

Live It Iconic is a **lifestyle brand e-commerce platform** targeting automotive enthusiasts and luxury lifestyle followers (25-45 year old demographic).

**Core Business Model:**
- 60% Merchandise sales (hoodies, caps, t-shirts)
- 20% YouTube ad revenue
- 15% Sponsorships
- 5% Memberships/community

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18.3+, TypeScript 5.8, Vite 7.1 |
| **Styling** | TailwindCSS 3.4, shadcn/ui |
| **State** | React Query, Context API |
| **Routing** | React Router v6 |
| **Backend** | Supabase (PostgreSQL 15) |
| **Payments** | Stripe (checkout, webhooks) |
| **Email** | React Email + Resend |
| **Testing** | Vitest, Playwright |
| **Deployment** | Vercel (free tier) |

## Key Features

### 1. E-Commerce Foundation

**Product Catalog** (`pages/Shop.tsx`)
- Display merchandise with images, descriptions, pricing
- Filtering and sorting
- Stock level indicators
- Currency selection

**Product Details** (`pages/ProductDetail.tsx`)
- Individual product pages
- Image galleries
- Reviews and ratings
- Add to cart functionality

**Shopping Cart** (`components/CartDrawer.tsx`)
- Add/remove items
- Quantity adjustment
- Persistent storage (localStorage)
- Real-time price calculation

**Inventory Management** (`services/inventoryService.ts`)
- Track stock levels
- Prevent overselling
- Low stock warnings
- Admin inventory controls

### 2. Payment Processing

**Stripe Integration** (`services/stripeService.ts`)
- Secure checkout with Stripe Elements
- Payment Intent creation
- Multiple payment methods (cards, digital wallets)
- Test card support for development

**Checkout Flow** (`pages/Checkout.tsx`)
- Shipping address collection
- Payment method selection
- Order summary
- Loading states and error handling

**Order Confirmation** (`pages/OrderConfirmation.tsx`)
- Order number display
- Order details summary
- Email confirmation sent
- Thank you message

**Webhook Handling** (`api/webhooks/stripe.ts`)
- Process payment success/failure
- Handle refunds
- Update order status
- Send confirmation emails

### 3. Content Integration

**YouTube Integration** (`pages/BrandShowcase.tsx`)
- Embed latest videos
- Playlist display
- View counts and engagement
- Call-to-action for subscriptions

**Twitch Integration**
- Live streaming indicator
- Embed streams when live
- Schedule display
- Community engagement

**Social Media Links**
- Instagram feed integration
- TikTok content
- Twitter updates
- Multi-channel presence

### 4. Brand Identity System

**Logo System** (`.brand/assets/`)
- 4 variants: Wordmark, Stacked, Monogram, Icon
- Black (#0A0A0A) and Gold (#D4AF37) color palette
- Usage guidelines for all contexts

**Typography**
- Primary: DM Sans (headings, UI)
- Secondary: Inter (body text)
- Montserrat Bold (logo, all caps)

**Design Tokens** (`design-tokens/`)
- Color palette
- Spacing scale
- Typography scale
- Shadow/elevation system

### 5. Order Management

**Customer Orders** (`pages/OrderDetails.tsx`)
- View order history
- Track shipping status
- Download invoices
- Contact support

**Admin Dashboard** (`pages/admin/`)
- View all orders
- Update order status
- Add tracking numbers
- Manage inventory
- View analytics

### 6. Email System

**React Email Templates** (`emails/`)
- Order confirmation
- Shipping notification
- Newsletter subscription
- Marketing campaigns

**Email Service** (`services/emailService.ts`)
- Integration with Resend API
- Template rendering
- Delivery tracking
- Error handling

### 7. Content Launch Platform

**AI-Powered Launch System** (`launch-platform/`)
- Content calendar generation
- Social media post creation
- Email campaign automation
- Multi-channel distribution

**Agent System**
- StoryTeller - Narrative creation
- VideoProducer - YouTube content planning
- SocialMediaStrategist - Cross-platform campaigns
- EmailMarketer - Newsletter automation

### 8. Admin Features

**Analytics Dashboard**
- Sales metrics (revenue, orders, conversion)
- Product analytics (top sellers, inventory)
- Customer insights (growth, repeat purchases)
- Content engagement (video views, social metrics)

**Product Management**
- Create/edit products
- Upload product images
- Set pricing and stock
- Manage categories

**Order Processing**
- Fulfill orders
- Print shipping labels
- Update tracking
- Process refunds

## Service Layer Architecture

All business logic is organized into service modules using ES6 classes:

```typescript
// Example: Order Service
export class OrderService {
  async createOrder(items: CartItem[], shippingAddress: Address) {
    // Business logic for order creation
  }

  async getOrderById(orderId: string) {
    // Fetch order details
  }

  async updateOrderStatus(orderId: string, status: OrderStatus) {
    // Update order status
  }
}
```

**Key Services:**
- `stripeService.ts` - Payment processing with Stripe API
- `productService.ts` - Product CRUD operations
- `orderService.ts` - Order management and fulfillment
- `inventoryService.ts` - Stock tracking and updates
- `emailService.ts` - Transactional and marketing emails
- `authService.ts` - User authentication with Supabase Auth

## Type Safety

Comprehensive TypeScript interfaces ensure type safety:

**`types/product.ts`**
```typescript
export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  cost: number;
  sku: string;
  category: string;
  imageUrls: string[];
  stockQuantity: number;
  isActive: boolean;
}
```

**`types/order.ts`**
```typescript
export interface Order {
  id: string;
  orderNumber: string;
  customerId: string;
  items: OrderItem[];
  totalAmount: number;
  status: 'pending' | 'paid' | 'shipped' | 'delivered';
  shippingAddress: Address;
  trackingNumber?: string;
  createdAt: Date;
}
```

## State Management

- **Server State** - React Query for API data caching and synchronization
- **Client State** - React Context API for cart, user session
- **Local Storage** - Persistent cart data
- **URL State** - React Router for navigation state

## Middleware

**Authentication** (`middleware/auth.ts`)
- JWT token validation
- Protected route guards
- User session management

**Rate Limiting** (`middleware/rateLimit.ts`)
- API request throttling
- Prevent abuse
- Per-endpoint limits

**Validation** (`middleware/validation.ts`)
- Input sanitization
- Form validation
- Error handling

## Getting Started

### Install Dependencies
```bash
npm install
```

### Start Development Server
```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### Build for Production
```bash
npm run build
```

### Run Tests
```bash
npm test
```

## User Flows

### Customer Purchase Flow
```
1. Browse Shop page → Filter/search products
2. Click product → View details
3. Add to cart → Cart drawer opens
4. Proceed to checkout → Enter shipping address
5. Payment → Stripe Elements form
6. Submit payment → Processing
7. Order confirmation → Receive order number + email
```

### Admin Order Management
```
1. Login to admin dashboard
2. View pending orders list
3. Click order → View details
4. Mark as shipped → Add tracking number
5. Customer receives shipping notification email
```

## Testing

See `../tests/README.md` for testing guidelines.

**Test Coverage:**
- Unit tests for services (Vitest)
- Component tests (React Testing Library)
- E2E tests for checkout flow (Playwright)

## Performance Considerations

- Route-based code splitting
- Image optimization (WebP format)
- Lazy loading for below-fold content
- React Query caching for API responses
- Vite's production optimizations

## Security

- Stripe handles all sensitive payment data (PCI compliant)
- Environment variables for API keys (never committed)
- CORS configuration limits to Vercel domains
- Supabase Row Level Security (RLS) on database
- Rate limiting prevents API abuse

## Brand Guidelines

See `.brand/identity/BRAND_IDENTITY_SYSTEM.md` for:
- Logo usage guidelines
- Color palette specifications
- Typography system
- Brand voice and tone
- Photography guidelines

## Revenue Model

**Year 1 Target**: $56K revenue, break-even Month 6

| Revenue Stream | % of Total | Month 6 | Month 12 |
|---------------|-----------|---------|----------|
| Merchandise | 60% | $3,000 | $6,000 |
| YouTube Ads | 20% | $180 | $900 |
| Sponsorships | 15% | $0 | $3,500 |
| Memberships | 5% | $0 | $500 |

**Gross Margins**: 70%+ on merchandise

## Resources

- [Project Overview](../PROJECT.md)
- [Directory Structure](../STRUCTURE.md)
- [Business Plan](../docs/planning/business-plan.md)
- [Launch Strategy](../docs/deployment/launch-guide.md)
- [Brand Identity](../.brand/identity/BRAND_IDENTITY_SYSTEM.md)
- [YouTube Strategy](../docs/guides/youtube-strategy.md)

## Adding New Features

1. Define types in `types/`
2. Create service module in `services/`
3. Build React components in `components/`
4. Add page routes in `pages/`
5. Write tests in `../tests/`
6. Update documentation

---

For complete documentation, see:
- [PROJECT.md](../PROJECT.md) - Complete project overview
- [STRUCTURE.md](../STRUCTURE.md) - Full directory structure
- [docs/README.md](../docs/README.md) - Documentation hub
