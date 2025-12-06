-- =====================================================
-- Migration 003: Orders and Payments
-- =====================================================
-- Creates tables for order management and payment processing
-- =====================================================

-- =====================================================
-- ORDERS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_number TEXT NOT NULL UNIQUE,
    customer_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    email TEXT NOT NULL,

    -- Order status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled', 'refunded')
    ),
    payment_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        payment_status IN ('pending', 'paid', 'failed', 'refunded', 'partially_refunded')
    ),
    fulfillment_status TEXT DEFAULT 'unfulfilled' CHECK (
        fulfillment_status IN ('unfulfilled', 'partially_fulfilled', 'fulfilled')
    ),

    -- Pricing (all in cents)
    subtotal INTEGER NOT NULL CHECK (subtotal >= 0),
    shipping_cost INTEGER NOT NULL DEFAULT 0 CHECK (shipping_cost >= 0),
    tax INTEGER NOT NULL DEFAULT 0 CHECK (tax >= 0),
    discount_amount INTEGER DEFAULT 0 CHECK (discount_amount >= 0),
    total INTEGER NOT NULL CHECK (total >= 0),

    -- Shipping information
    shipping_first_name TEXT NOT NULL,
    shipping_last_name TEXT NOT NULL,
    shipping_address_line1 TEXT NOT NULL,
    shipping_address_line2 TEXT,
    shipping_city TEXT NOT NULL,
    shipping_state TEXT NOT NULL,
    shipping_postal_code TEXT NOT NULL,
    shipping_country TEXT NOT NULL DEFAULT 'US',
    shipping_phone TEXT,

    -- Billing information (optional, defaults to shipping)
    billing_first_name TEXT,
    billing_last_name TEXT,
    billing_address_line1 TEXT,
    billing_address_line2 TEXT,
    billing_city TEXT,
    billing_state TEXT,
    billing_postal_code TEXT,
    billing_country TEXT,

    -- Fulfillment
    tracking_number TEXT,
    tracking_url TEXT,
    carrier TEXT,
    shipped_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,

    -- Payment
    stripe_payment_intent_id TEXT UNIQUE,
    discount_code TEXT,

    -- Additional
    notes TEXT,
    customer_notes TEXT,
    ip_address INET,
    user_agent TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cancelled_at TIMESTAMP WITH TIME ZONE,
    refunded_at TIMESTAMP WITH TIME ZONE
);

-- Enable RLS
ALTER TABLE public.orders ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own orders"
    ON public.orders
    FOR SELECT
    USING (
        auth.uid() = customer_id OR
        email = auth.jwt()->>'email'
    );

-- =====================================================
-- ORDER ITEMS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.order_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES public.orders(id) ON DELETE CASCADE,
    product_id UUID REFERENCES public.products(id) ON DELETE SET NULL,
    variant_id UUID REFERENCES public.product_variants(id) ON DELETE SET NULL,

    -- Product snapshot (in case product is deleted/changed)
    product_name TEXT NOT NULL,
    variant_name TEXT,
    sku TEXT,

    -- Pricing (in cents)
    price INTEGER NOT NULL CHECK (price >= 0), -- Price per unit
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    subtotal INTEGER NOT NULL CHECK (subtotal >= 0), -- price * quantity
    discount INTEGER DEFAULT 0 CHECK (discount >= 0),
    total INTEGER NOT NULL CHECK (total >= 0),

    -- Product details snapshot
    image_url TEXT,
    weight DECIMAL(10, 2),
    requires_shipping BOOLEAN DEFAULT TRUE,

    -- Fulfillment
    fulfillment_status TEXT DEFAULT 'unfulfilled' CHECK (
        fulfillment_status IN ('unfulfilled', 'fulfilled', 'cancelled')
    ),
    fulfilled_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.order_items ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own order items"
    ON public.order_items
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.orders
            WHERE orders.id = order_items.order_id
            AND (orders.customer_id = auth.uid() OR orders.email = auth.jwt()->>'email')
        )
    );

-- =====================================================
-- PAYMENTS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES public.orders(id) ON DELETE CASCADE,

    -- Payment details
    stripe_payment_intent_id TEXT NOT NULL,
    stripe_charge_id TEXT,
    amount INTEGER NOT NULL CHECK (amount >= 0), -- Amount in cents
    currency TEXT NOT NULL DEFAULT 'usd',
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'processing', 'succeeded', 'failed', 'cancelled', 'refunded')
    ),

    -- Payment method
    payment_method_type TEXT, -- 'card', 'bank_transfer', etc.
    card_brand TEXT, -- 'visa', 'mastercard', etc.
    card_last4 TEXT,
    card_exp_month INTEGER,
    card_exp_year INTEGER,

    -- Metadata
    error_message TEXT,
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    succeeded_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE
);

-- Enable RLS
ALTER TABLE public.payments ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own payments"
    ON public.payments
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.orders
            WHERE orders.id = payments.order_id
            AND (orders.customer_id = auth.uid() OR orders.email = auth.jwt()->>'email')
        )
    );

-- =====================================================
-- DISCOUNT CODES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.discount_codes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL CHECK (type IN ('percentage', 'fixed_amount', 'free_shipping')),
    value INTEGER NOT NULL CHECK (value > 0), -- Percentage or amount in cents

    -- Usage limits
    max_uses INTEGER,
    times_used INTEGER DEFAULT 0 CHECK (times_used >= 0),
    max_uses_per_customer INTEGER DEFAULT 1,

    -- Restrictions
    minimum_order_amount INTEGER DEFAULT 0 CHECK (minimum_order_amount >= 0),
    applies_to_product_ids UUID[],
    applies_to_category_ids UUID[],

    -- Validity period
    valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    -- Metadata
    description TEXT,
    created_by UUID REFERENCES public.profiles(id),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.discount_codes ENABLE ROW LEVEL SECURITY;

-- RLS Policies (public can check valid codes)
CREATE POLICY "Active discount codes are viewable by everyone"
    ON public.discount_codes
    FOR SELECT
    USING (
        is_active = TRUE AND
        valid_from <= NOW() AND
        (valid_until IS NULL OR valid_until >= NOW()) AND
        (max_uses IS NULL OR times_used < max_uses)
    );

-- =====================================================
-- DISCOUNT CODE REDEMPTIONS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.discount_code_redemptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    discount_code_id UUID NOT NULL REFERENCES public.discount_codes(id) ON DELETE CASCADE,
    order_id UUID NOT NULL REFERENCES public.orders(id) ON DELETE CASCADE,
    customer_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    discount_amount INTEGER NOT NULL CHECK (discount_amount >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(discount_code_id, order_id)
);

-- Enable RLS
ALTER TABLE public.discount_code_redemptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own redemptions"
    ON public.discount_code_redemptions
    FOR SELECT
    USING (auth.uid() = customer_id);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_orders_customer ON public.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_email ON public.orders(email);
CREATE INDEX IF NOT EXISTS idx_orders_status ON public.orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_payment_status ON public.orders(payment_status);
CREATE INDEX IF NOT EXISTS idx_orders_stripe_payment_intent ON public.orders(stripe_payment_intent_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON public.orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_number ON public.orders(order_number);

CREATE INDEX IF NOT EXISTS idx_order_items_order ON public.order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON public.order_items(product_id);

CREATE INDEX IF NOT EXISTS idx_payments_order ON public.payments(order_id);
CREATE INDEX IF NOT EXISTS idx_payments_stripe_payment_intent ON public.payments(stripe_payment_intent_id);
CREATE INDEX IF NOT EXISTS idx_payments_status ON public.payments(status);

CREATE INDEX IF NOT EXISTS idx_discount_codes_code ON public.discount_codes(code);
CREATE INDEX IF NOT EXISTS idx_discount_codes_active ON public.discount_codes(is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_discount_redemptions_code ON public.discount_code_redemptions(discount_code_id);
CREATE INDEX IF NOT EXISTS idx_discount_redemptions_order ON public.discount_code_redemptions(order_id);
CREATE INDEX IF NOT EXISTS idx_discount_redemptions_customer ON public.discount_code_redemptions(customer_id);

-- =====================================================
-- TRIGGERS
-- =====================================================

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON public.orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payments_updated_at
    BEFORE UPDATE ON public.payments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_discount_codes_updated_at
    BEFORE UPDATE ON public.discount_codes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Generate unique order number
CREATE OR REPLACE FUNCTION generate_order_number()
RETURNS TEXT AS $$
DECLARE
    new_number TEXT;
    counter INTEGER;
BEGIN
    -- Get current count of orders today
    SELECT COUNT(*) INTO counter
    FROM public.orders
    WHERE DATE(created_at) = CURRENT_DATE;

    -- Format: LII-YYYYMMDD-NNNN
    new_number := 'LII-' || TO_CHAR(CURRENT_DATE, 'YYYYMMDD') || '-' || LPAD((counter + 1)::TEXT, 4, '0');

    RETURN new_number;
END;
$$ LANGUAGE plpgsql;

-- Auto-generate order number on insert
CREATE OR REPLACE FUNCTION set_order_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.order_number IS NULL THEN
        NEW.order_number := generate_order_number();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_order_number_trigger
    BEFORE INSERT ON public.orders
    FOR EACH ROW
    EXECUTE FUNCTION set_order_number();

-- Function to increment discount code usage
CREATE OR REPLACE FUNCTION increment_discount_usage(discount_code TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE public.discount_codes
    SET times_used = times_used + 1
    WHERE code = UPPER(discount_code);
END;
$$ LANGUAGE plpgsql;

-- RPC function for clients to call
CREATE OR REPLACE FUNCTION public.increment_discount_usage_rpc(p_code TEXT)
RETURNS VOID AS $$
BEGIN
    PERFORM increment_discount_usage(p_code);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
