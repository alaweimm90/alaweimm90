# Supabase Directory

This directory contains Supabase configuration, database migrations, and backend logic for the **Live It Iconic** e-commerce platform.

## Directory Structure

```
supabase/
├── migrations/               # Database schema migrations
│   ├── 001_initial_schema.sql
│   ├── 002_products_and_inventory.sql
│   ├── 003_orders_and_payments.sql
│   ├── 004_reviews_and_ratings.sql
│   └── ...
├── functions/               # Supabase Edge Functions (Deno runtime)
│   ├── stripe-webhook/
│   ├── order-confirmation/
│   └── analytics/
├── seed.sql                 # Initial data for development
├── config.toml              # Supabase project configuration
└── .env.local              # Local environment variables (gitignored)
```

## Database Schema

### Core Tables

#### Users & Authentication
- `auth.users` - User authentication (managed by Supabase Auth)
- `public.profiles` - Extended user profiles
- `public.addresses` - Shipping/billing addresses

#### E-Commerce

**Products**
- `products` - Product catalog
- `product_images` - Product image URLs
- `product_variants` - Size/color variations (future)
- `inventory` - Stock levels and tracking

**Orders**
- `orders` - Customer orders
- `order_items` - Individual line items in orders
- `order_status_history` - Order status tracking

**Payments**
- `payments` - Payment transactions
- `payment_intents` - Stripe Payment Intents
- `refunds` - Refund records

**Reviews**
- `reviews` - Product reviews
- `review_images` - Review photos (optional)

**Cart**
- `carts` - Shopping cart sessions
- `cart_items` - Items in cart

**Discounts**
- `discount_codes` - Promotional codes
- `discount_redemptions` - Usage tracking

## Detailed Schema

### Products Table

```sql
CREATE TABLE products (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  price DECIMAL(10, 2) NOT NULL,
  cost DECIMAL(10, 2), -- Cost of goods (for margin calc)
  sku VARCHAR(100) UNIQUE NOT NULL,
  category VARCHAR(50) NOT NULL, -- 'hoodies', 'caps', 'tshirts'
  image_urls TEXT[] DEFAULT '{}',
  stock_quantity INT NOT NULL DEFAULT 0,
  low_stock_threshold INT DEFAULT 10,
  is_active BOOLEAN DEFAULT TRUE,
  featured BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = TRUE;
```

### Orders Table

```sql
CREATE TABLE orders (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  order_number VARCHAR(50) UNIQUE NOT NULL, -- LII-001, LII-002, etc.
  customer_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,

  -- Order totals
  subtotal DECIMAL(10, 2) NOT NULL,
  discount_amount DECIMAL(10, 2) DEFAULT 0,
  tax_amount DECIMAL(10, 2) DEFAULT 0,
  shipping_amount DECIMAL(10, 2) DEFAULT 0,
  total_amount DECIMAL(10, 2) NOT NULL,

  -- Order status
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- 'pending', 'paid', 'processing', 'shipped', 'delivered', 'cancelled'

  -- Shipping information
  shipping_address JSONB NOT NULL,
    /* {
      "name": "John Doe",
      "email": "john@example.com",
      "street": "123 Main St",
      "street2": "Apt 4",
      "city": "Los Angeles",
      "state": "CA",
      "zip": "90001",
      "country": "US",
      "phone": "+1234567890"
    } */

  shipping_method VARCHAR(50),
  tracking_number VARCHAR(100),
  tracking_url TEXT,

  -- Discount code applied
  discount_code VARCHAR(50),

  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  paid_at TIMESTAMPTZ,
  shipped_at TIMESTAMPTZ,
  delivered_at TIMESTAMPTZ
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created ON orders(created_at DESC);
CREATE INDEX idx_orders_number ON orders(order_number);
```

### Order Items Table

```sql
CREATE TABLE order_items (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
  product_id UUID NOT NULL REFERENCES products(id) ON DELETE RESTRICT,

  product_name VARCHAR(255) NOT NULL, -- Snapshot at time of purchase
  product_sku VARCHAR(100) NOT NULL,
  product_image_url TEXT,

  quantity INT NOT NULL CHECK (quantity > 0),
  unit_price DECIMAL(10, 2) NOT NULL,
  line_total DECIMAL(10, 2) NOT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
```

### Payments Table

```sql
CREATE TABLE payments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,

  -- Stripe identifiers
  stripe_payment_intent_id VARCHAR(255) UNIQUE NOT NULL,
  stripe_charge_id VARCHAR(255),

  -- Payment details
  amount DECIMAL(10, 2) NOT NULL,
  currency VARCHAR(3) DEFAULT 'USD',
  status VARCHAR(20) NOT NULL,
    -- 'pending', 'processing', 'succeeded', 'failed', 'canceled', 'refunded'

  payment_method_type VARCHAR(50), -- 'card', 'apple_pay', 'google_pay'
  payment_method_brand VARCHAR(50), -- 'visa', 'mastercard', 'amex'
  payment_method_last4 VARCHAR(4),

  -- Failure information
  failure_code VARCHAR(100),
  failure_message TEXT,

  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  succeeded_at TIMESTAMPTZ
);

CREATE INDEX idx_payments_order ON payments(order_id);
CREATE INDEX idx_payments_stripe_pi ON payments(stripe_payment_intent_id);
CREATE INDEX idx_payments_status ON payments(status);
```

### Reviews Table

```sql
CREATE TABLE reviews (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
  customer_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  order_id UUID REFERENCES orders(id) ON DELETE SET NULL, -- Verified purchase

  rating INT NOT NULL CHECK (rating >= 1 AND rating <= 5),
  title VARCHAR(255),
  text TEXT,
  images TEXT[] DEFAULT '{}',

  verified_purchase BOOLEAN DEFAULT FALSE,
  is_featured BOOLEAN DEFAULT FALSE,

  -- Moderation
  is_approved BOOLEAN DEFAULT FALSE,
  moderated_at TIMESTAMPTZ,
  moderated_by UUID REFERENCES auth.users(id),

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_reviews_customer ON reviews(customer_id);
CREATE INDEX idx_reviews_approved ON reviews(is_approved) WHERE is_approved = TRUE;
```

## Row Level Security (RLS)

All tables use Row Level Security to ensure proper data access.

### Products (Public Read)

```sql
-- Anyone can view active products
CREATE POLICY "Public can view active products"
  ON products
  FOR SELECT
  USING (is_active = TRUE);

-- Only admins can modify products
CREATE POLICY "Admins can manage products"
  ON products
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM profiles
      WHERE id = auth.uid()
      AND role = 'admin'
    )
  );
```

### Orders (Customer + Admin Access)

```sql
-- Customers can view their own orders
CREATE POLICY "Customers can view own orders"
  ON orders
  FOR SELECT
  USING (auth.uid() = customer_id);

-- Customers can create orders
CREATE POLICY "Customers can create orders"
  ON orders
  FOR INSERT
  WITH CHECK (auth.uid() = customer_id);

-- Admins can view all orders
CREATE POLICY "Admins can view all orders"
  ON orders
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM profiles
      WHERE id = auth.uid()
      AND role = 'admin'
    )
  );

-- Admins can update orders
CREATE POLICY "Admins can update orders"
  ON orders
  FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM profiles
      WHERE id = auth.uid()
      AND role = 'admin'
    )
  );
```

### Payments (Restricted Access)

```sql
-- Customers can only view their own payments
CREATE POLICY "Customers can view own payments"
  ON payments
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM orders
      WHERE orders.id = payments.order_id
      AND orders.customer_id = auth.uid()
    )
  );

-- Only system (service role) can insert payments
CREATE POLICY "Service role can insert payments"
  ON payments
  FOR INSERT
  WITH CHECK (auth.jwt() ->> 'role' = 'service_role');
```

### Reviews

```sql
-- Anyone can view approved reviews
CREATE POLICY "Public can view approved reviews"
  ON reviews
  FOR SELECT
  USING (is_approved = TRUE);

-- Customers can create reviews for products they purchased
CREATE POLICY "Customers can create reviews"
  ON reviews
  FOR INSERT
  WITH CHECK (
    auth.uid() = customer_id
    AND EXISTS (
      SELECT 1 FROM orders o
      JOIN order_items oi ON oi.order_id = o.id
      WHERE o.customer_id = auth.uid()
      AND oi.product_id = reviews.product_id
      AND o.status = 'delivered'
    )
  );

-- Customers can update their own reviews
CREATE POLICY "Customers can update own reviews"
  ON reviews
  FOR UPDATE
  USING (auth.uid() = customer_id);
```

## Migrations

### Creating a Migration

```bash
supabase migration new add_discount_codes
```

This creates: `supabase/migrations/YYYYMMDDHHMMSS_add_discount_codes.sql`

### Writing Migrations

```sql
-- supabase/migrations/005_add_discount_codes.sql

-- Create discount_codes table
CREATE TABLE discount_codes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code VARCHAR(50) UNIQUE NOT NULL,
  type VARCHAR(20) NOT NULL, -- 'percentage' or 'fixed_amount'
  value DECIMAL(10, 2) NOT NULL,
  min_order_amount DECIMAL(10, 2),
  max_uses INT,
  times_used INT DEFAULT 0,
  valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  valid_until TIMESTAMPTZ,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_discount_codes_code ON discount_codes(code);
CREATE INDEX idx_discount_codes_active ON discount_codes(is_active) WHERE is_active = TRUE;

-- Enable RLS
ALTER TABLE discount_codes ENABLE ROW LEVEL SECURITY;

-- Public can view active codes
CREATE POLICY "Public can view active discount codes"
  ON discount_codes
  FOR SELECT
  USING (is_active = TRUE AND NOW() BETWEEN valid_from AND COALESCE(valid_until, 'infinity'));
```

### Running Migrations

**Local Development:**
```bash
supabase db reset  # Reset and run all migrations
```

**Production:**
```bash
supabase db push  # Push migrations to production
```

## Edge Functions

Supabase Edge Functions run on Deno runtime at the edge.

### stripe-webhook

Handles Stripe webhook events for payment processing.

```bash
supabase functions serve stripe-webhook
```

**Trigger**: Stripe webhook (payment events)
**Purpose**: Update order status, send confirmation emails

### order-confirmation

Sends order confirmation emails.

```bash
supabase functions serve order-confirmation
```

**Trigger**: After successful payment
**Purpose**: Send email with order details and tracking

### Creating a New Function

```bash
supabase functions new inventory-alert
```

**Example Function:**
```typescript
// supabase/functions/inventory-alert/index.ts
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

serve(async (req) => {
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  );

  // Check for low stock products
  const { data: lowStockProducts } = await supabase
    .from('products')
    .select('*')
    .lt('stock_quantity', 10);

  // Send alert email to admin
  // ... email logic

  return new Response(
    JSON.stringify({ alertsSent: lowStockProducts?.length }),
    { headers: { 'Content-Type': 'application/json' } }
  );
});
```

### Deploying Functions

```bash
supabase functions deploy stripe-webhook --no-verify-jwt
```

## Seeding Data

`seed.sql` populates the database with initial data for development.

```sql
-- supabase/seed.sql

-- Insert test users
INSERT INTO auth.users (id, email) VALUES
  ('user-1-uuid', 'customer@example.com'),
  ('admin-1-uuid', 'admin@liveiticonic.com');

-- Insert profiles
INSERT INTO profiles (id, email, full_name, role) VALUES
  ('user-1-uuid', 'customer@example.com', 'Test Customer', 'customer'),
  ('admin-1-uuid', 'admin@liveiticonic.com', 'Admin User', 'admin');

-- Insert products
INSERT INTO products (name, description, price, cost, sku, category, stock_quantity, image_urls) VALUES
  ('Black Hoodie', 'Premium quality hoodie with Live It Iconic branding', 79.00, 20.00, 'HOODIE-BLK', 'hoodies', 50, ARRAY['/images/products/hoodie-black-front.webp']),
  ('Black Cap', 'Adjustable cap with embroidered logo', 29.00, 8.00, 'CAP-BLK', 'caps', 100, ARRAY['/images/products/cap-black-front.webp']),
  ('Black T-Shirt', 'Soft cotton t-shirt with printed logo', 49.00, 12.00, 'TSHIRT-BLK', 'tshirts', 75, ARRAY['/images/products/tshirt-black-front.webp']);

-- Insert sample order
INSERT INTO orders (id, order_number, customer_id, subtotal, total_amount, status, shipping_address) VALUES
  ('order-1-uuid', 'LII-001', 'user-1-uuid', 79.00, 79.00, 'delivered',
   '{"name": "Test Customer", "email": "customer@example.com", "street": "123 Main St", "city": "Los Angeles", "state": "CA", "zip": "90001"}');

-- Insert order items
INSERT INTO order_items (order_id, product_id, product_name, product_sku, quantity, unit_price, line_total)
SELECT 'order-1-uuid', id, name, sku, 1, price, price
FROM products WHERE sku = 'HOODIE-BLK';
```

Run seed:
```bash
supabase db seed
```

## Real-time Subscriptions

Supabase supports real-time updates via WebSockets.

**Client-side example:**
```typescript
import { supabase } from '@/lib/supabase';

// Subscribe to order status changes
const channel = supabase
  .channel('orders')
  .on(
    'postgres_changes',
    {
      event: 'UPDATE',
      schema: 'public',
      table: 'orders',
      filter: `customer_id=eq.${userId}`,
    },
    (payload) => {
      console.log('Order updated:', payload);
      // Update UI with new order status
    }
  )
  .subscribe();

// Cleanup
channel.unsubscribe();
```

## Storage

Supabase Storage for user uploads (review photos, etc.).

### Buckets
- `product-images` - Product photography
- `review-images` - Customer review photos
- `brand-assets` - Logo and brand files

### Storage Policies

```sql
-- Allow public read access to product images
CREATE POLICY "Public can view product images"
  ON storage.objects
  FOR SELECT
  USING (bucket_id = 'product-images');

-- Customers can upload review images
CREATE POLICY "Customers can upload review images"
  ON storage.objects
  FOR INSERT
  WITH CHECK (
    bucket_id = 'review-images' AND
    auth.role() = 'authenticated'
  );
```

## Environment Variables

Required environment variables:

```bash
# .env.local
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key  # Server-side only
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

## Performance Optimization

### Indexes

Add indexes for frequently queried columns:

```sql
CREATE INDEX idx_orders_status_created ON orders(status, created_at DESC);
CREATE INDEX idx_products_category_active ON products(category, is_active);
CREATE INDEX idx_order_items_product ON order_items(product_id) WHERE created_at > NOW() - INTERVAL '90 days';
```

### Materialized Views

For complex queries, use materialized views:

```sql
CREATE MATERIALIZED VIEW product_stats AS
SELECT
  p.id,
  p.name,
  p.stock_quantity,
  COUNT(DISTINCT oi.order_id) as total_orders,
  SUM(oi.quantity) as units_sold,
  SUM(oi.line_total) as revenue,
  AVG(r.rating) as avg_rating,
  COUNT(DISTINCT r.id) as review_count
FROM products p
LEFT JOIN order_items oi ON oi.product_id = p.id
LEFT JOIN reviews r ON r.product_id = p.id
GROUP BY p.id;

-- Refresh daily
REFRESH MATERIALIZED VIEW product_stats;
```

### Connection Pooling

Use connection pooling in production for better performance.

## Security

### Secrets Management

Never commit:
- Service role keys
- Stripe secret keys
- Webhook secrets

Use Supabase Vault for secrets:

```sql
SELECT vault.create_secret('stripe_webhook_secret', 'whsec_...');
```

### Rate Limiting

Implemented in `src/middleware/rateLimit.ts`:
- API calls: 100 requests / 15 minutes
- Auth: 5 requests / 15 minutes
- Checkout: 10 requests / hour per user

### Data Encryption

- TLS for all connections
- Sensitive payment data handled by Stripe (PCI compliant)
- Customer addresses encrypted in JSONB columns

## Monitoring

### Query Performance

```sql
-- Check slow queries
SELECT
  query,
  mean_exec_time,
  calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Database Size

```sql
SELECT
  pg_size_pretty(pg_database_size(current_database())) as size;
```

### Table Sizes

```sql
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Backup & Recovery

### Automated Backups

Supabase provides automated daily backups (retention based on plan).

### Manual Backup

```bash
supabase db dump > backup.sql
```

### Restore from Backup

```bash
supabase db reset
psql -h db.your-project.supabase.co -U postgres -d postgres -f backup.sql
```

## Resources

- [Supabase Documentation](https://supabase.com/docs)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Supabase Edge Functions](https://supabase.com/docs/guides/functions)
- [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)
- [Stripe Webhooks](https://stripe.com/docs/webhooks)

---

For more details, see:
- [Business Plan](../docs/planning/business-plan.md)
- [Architecture Guide](../docs/ARCHITECTURE.md)
- [Deployment Guide](../DEPLOYMENT_INFRASTRUCTURE_PLAN.md)
- [Stripe Integration](../docs/guides/stripe-integration.md)
