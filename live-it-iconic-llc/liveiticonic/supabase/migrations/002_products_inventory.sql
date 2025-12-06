-- =====================================================
-- Migration 002: Products and Inventory
-- =====================================================
-- Creates tables for product catalog and inventory management
-- =====================================================

-- =====================================================
-- CATEGORIES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    slug TEXT NOT NULL UNIQUE,
    description TEXT,
    image_url TEXT,
    parent_id UUID REFERENCES public.categories(id) ON DELETE SET NULL,
    sort_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS (public read)
ALTER TABLE public.categories ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Categories are viewable by everyone"
    ON public.categories
    FOR SELECT
    USING (is_active = TRUE);

-- =====================================================
-- PRODUCTS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    description TEXT,
    short_description TEXT,
    price INTEGER NOT NULL CHECK (price >= 0), -- Price in cents
    compare_at_price INTEGER CHECK (compare_at_price >= 0),
    cost INTEGER CHECK (cost >= 0), -- Cost of goods in cents
    category_id UUID REFERENCES public.categories(id) ON DELETE SET NULL,
    sku TEXT UNIQUE,
    barcode TEXT,
    track_inventory BOOLEAN DEFAULT TRUE,
    continue_selling_when_out_of_stock BOOLEAN DEFAULT FALSE,
    requires_shipping BOOLEAN DEFAULT TRUE,
    weight DECIMAL(10, 2), -- Weight in grams
    dimensions JSONB, -- {length, width, height} in cm
    is_active BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    seo_title TEXT,
    seo_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS (public read for active products)
ALTER TABLE public.products ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Active products are viewable by everyone"
    ON public.products
    FOR SELECT
    USING (is_active = TRUE);

-- =====================================================
-- PRODUCT IMAGES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.product_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES public.products(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    alt_text TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.product_images ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Product images are viewable by everyone"
    ON public.product_images
    FOR SELECT
    USING (TRUE);

-- =====================================================
-- PRODUCT VARIANTS TABLE
-- =====================================================
-- For products with size/color options

CREATE TABLE IF NOT EXISTS public.product_variants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES public.products(id) ON DELETE CASCADE,
    name TEXT NOT NULL, -- e.g., "Large / Black"
    sku TEXT UNIQUE,
    price INTEGER, -- Override product price if set
    option1_name TEXT, -- e.g., "Size"
    option1_value TEXT, -- e.g., "Large"
    option2_name TEXT, -- e.g., "Color"
    option2_value TEXT, -- e.g., "Black"
    option3_name TEXT,
    option3_value TEXT,
    image_url TEXT,
    sort_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.product_variants ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Product variants are viewable by everyone"
    ON public.product_variants
    FOR SELECT
    USING (is_active = TRUE);

-- =====================================================
-- INVENTORY TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.inventory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID REFERENCES public.products(id) ON DELETE CASCADE,
    variant_id UUID REFERENCES public.product_variants(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL DEFAULT 0 CHECK (quantity >= 0),
    reserved_quantity INTEGER NOT NULL DEFAULT 0 CHECK (reserved_quantity >= 0),
    low_stock_threshold INTEGER DEFAULT 10,
    location TEXT, -- Warehouse location
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure either product_id or variant_id is set, not both
    CONSTRAINT inventory_product_or_variant CHECK (
        (product_id IS NOT NULL AND variant_id IS NULL) OR
        (product_id IS NULL AND variant_id IS NOT NULL)
    ),
    -- Unique constraint
    UNIQUE(product_id, variant_id, location)
);

-- Enable RLS
ALTER TABLE public.inventory ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Inventory is viewable by everyone"
    ON public.inventory
    FOR SELECT
    USING (TRUE);

-- =====================================================
-- INVENTORY TRANSACTIONS TABLE
-- =====================================================
-- Audit log for inventory changes

CREATE TABLE IF NOT EXISTS public.inventory_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    inventory_id UUID NOT NULL REFERENCES public.inventory(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('sale', 'restock', 'adjustment', 'reservation', 'release')),
    quantity INTEGER NOT NULL, -- Positive for additions, negative for subtractions
    reference_id UUID, -- Order ID or adjustment ID
    notes TEXT,
    created_by UUID REFERENCES public.profiles(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.inventory_transactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Inventory transactions viewable by authenticated users"
    ON public.inventory_transactions
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_categories_parent ON public.categories(parent_id);
CREATE INDEX IF NOT EXISTS idx_categories_slug ON public.categories(slug);

CREATE INDEX IF NOT EXISTS idx_products_category ON public.products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_slug ON public.products(slug);
CREATE INDEX IF NOT EXISTS idx_products_sku ON public.products(sku);
CREATE INDEX IF NOT EXISTS idx_products_active ON public.products(is_active);
CREATE INDEX IF NOT EXISTS idx_products_featured ON public.products(is_featured) WHERE is_featured = TRUE;

CREATE INDEX IF NOT EXISTS idx_product_images_product ON public.product_images(product_id);

CREATE INDEX IF NOT EXISTS idx_product_variants_product ON public.product_variants(product_id);
CREATE INDEX IF NOT EXISTS idx_product_variants_sku ON public.product_variants(sku);

CREATE INDEX IF NOT EXISTS idx_inventory_product ON public.inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_variant ON public.inventory(variant_id);
CREATE INDEX IF NOT EXISTS idx_inventory_low_stock ON public.inventory(quantity) WHERE quantity <= low_stock_threshold;

CREATE INDEX IF NOT EXISTS idx_inventory_transactions_inventory ON public.inventory_transactions(inventory_id);
CREATE INDEX IF NOT EXISTS idx_inventory_transactions_reference ON public.inventory_transactions(reference_id);

-- =====================================================
-- TRIGGERS
-- =====================================================

CREATE TRIGGER update_categories_updated_at
    BEFORE UPDATE ON public.categories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at
    BEFORE UPDATE ON public.products
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_product_variants_updated_at
    BEFORE UPDATE ON public.product_variants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_inventory_updated_at
    BEFORE UPDATE ON public.inventory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Function to get available inventory
CREATE OR REPLACE FUNCTION get_available_inventory(
    p_product_id UUID DEFAULT NULL,
    p_variant_id UUID DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    available INTEGER;
BEGIN
    SELECT (quantity - reserved_quantity)
    INTO available
    FROM public.inventory
    WHERE (product_id = p_product_id OR p_product_id IS NULL)
      AND (variant_id = p_variant_id OR p_variant_id IS NULL);

    RETURN COALESCE(available, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to reserve inventory
CREATE OR REPLACE FUNCTION reserve_inventory(
    p_product_id UUID DEFAULT NULL,
    p_variant_id UUID DEFAULT NULL,
    p_quantity INTEGER DEFAULT 1
)
RETURNS BOOLEAN AS $$
DECLARE
    available INTEGER;
BEGIN
    -- Get available quantity
    available := get_available_inventory(p_product_id, p_variant_id);

    -- Check if enough inventory
    IF available < p_quantity THEN
        RETURN FALSE;
    END IF;

    -- Reserve the inventory
    UPDATE public.inventory
    SET reserved_quantity = reserved_quantity + p_quantity
    WHERE (product_id = p_product_id OR p_product_id IS NULL)
      AND (variant_id = p_variant_id OR p_variant_id IS NULL);

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to release reserved inventory
CREATE OR REPLACE FUNCTION release_inventory(
    p_product_id UUID DEFAULT NULL,
    p_variant_id UUID DEFAULT NULL,
    p_quantity INTEGER DEFAULT 1
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE public.inventory
    SET reserved_quantity = GREATEST(reserved_quantity - p_quantity, 0)
    WHERE (product_id = p_product_id OR p_product_id IS NULL)
      AND (variant_id = p_variant_id OR p_variant_id IS NULL);

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
