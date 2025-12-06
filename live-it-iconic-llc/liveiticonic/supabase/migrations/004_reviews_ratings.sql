-- =====================================================
-- Migration 004: Reviews and Ratings
-- =====================================================
-- Creates tables for product reviews and ratings
-- =====================================================

-- =====================================================
-- REVIEWS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES public.products(id) ON DELETE CASCADE,
    customer_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    order_id UUID REFERENCES public.orders(id) ON DELETE SET NULL,

    -- Review content
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    title TEXT,
    text TEXT NOT NULL,

    -- Verification
    is_verified_purchase BOOLEAN DEFAULT FALSE,
    is_approved BOOLEAN DEFAULT FALSE, -- Requires moderation
    is_featured BOOLEAN DEFAULT FALSE,

    -- Customer info (snapshot in case profile is deleted)
    customer_name TEXT,
    customer_email TEXT,

    -- Helpfulness
    helpful_count INTEGER DEFAULT 0 CHECK (helpful_count >= 0),
    not_helpful_count INTEGER DEFAULT 0 CHECK (not_helpful_count >= 0),

    -- Response from merchant
    merchant_response TEXT,
    merchant_responded_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    approved_at TIMESTAMP WITH TIME ZONE,

    -- Prevent duplicate reviews per product per customer
    UNIQUE(product_id, customer_id)
);

-- Enable RLS
ALTER TABLE public.reviews ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Approved reviews are viewable by everyone"
    ON public.reviews
    FOR SELECT
    USING (is_approved = TRUE);

CREATE POLICY "Users can view own reviews"
    ON public.reviews
    FOR SELECT
    USING (auth.uid() = customer_id);

CREATE POLICY "Users can create reviews"
    ON public.reviews
    FOR INSERT
    WITH CHECK (auth.uid() = customer_id);

CREATE POLICY "Users can update own reviews"
    ON public.reviews
    FOR UPDATE
    USING (auth.uid() = customer_id);

CREATE POLICY "Users can delete own reviews"
    ON public.reviews
    FOR DELETE
    USING (auth.uid() = customer_id);

-- =====================================================
-- REVIEW IMAGES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS public.review_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id UUID NOT NULL REFERENCES public.reviews(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    alt_text TEXT,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.review_images ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Review images are viewable with approved reviews"
    ON public.review_images
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.reviews
            WHERE reviews.id = review_images.review_id
            AND reviews.is_approved = TRUE
        )
    );

CREATE POLICY "Users can add images to own reviews"
    ON public.review_images
    FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.reviews
            WHERE reviews.id = review_images.review_id
            AND reviews.customer_id = auth.uid()
        )
    );

-- =====================================================
-- REVIEW VOTES TABLE
-- =====================================================
-- Track which users found reviews helpful

CREATE TABLE IF NOT EXISTS public.review_votes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id UUID NOT NULL REFERENCES public.reviews(id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    is_helpful BOOLEAN NOT NULL, -- TRUE for helpful, FALSE for not helpful
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- One vote per customer per review
    UNIQUE(review_id, customer_id)
);

-- Enable RLS
ALTER TABLE public.review_votes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can create review votes"
    ON public.review_votes
    FOR INSERT
    WITH CHECK (auth.uid() = customer_id);

CREATE POLICY "Users can view own votes"
    ON public.review_votes
    FOR SELECT
    USING (auth.uid() = customer_id);

-- =====================================================
-- PRODUCT RATING STATISTICS TABLE
-- =====================================================
-- Materialized view for performance

CREATE TABLE IF NOT EXISTS public.product_rating_stats (
    product_id UUID PRIMARY KEY REFERENCES public.products(id) ON DELETE CASCADE,
    total_reviews INTEGER DEFAULT 0,
    average_rating DECIMAL(3, 2) DEFAULT 0.00,
    rating_1_count INTEGER DEFAULT 0,
    rating_2_count INTEGER DEFAULT 0,
    rating_3_count INTEGER DEFAULT 0,
    rating_4_count INTEGER DEFAULT 0,
    rating_5_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS (public read)
ALTER TABLE public.product_rating_stats ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Rating stats are viewable by everyone"
    ON public.product_rating_stats
    FOR SELECT
    USING (TRUE);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_reviews_product ON public.reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_customer ON public.reviews(customer_id);
CREATE INDEX IF NOT EXISTS idx_reviews_approved ON public.reviews(is_approved) WHERE is_approved = TRUE;
CREATE INDEX IF NOT EXISTS idx_reviews_featured ON public.reviews(is_featured) WHERE is_featured = TRUE;
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON public.reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON public.reviews(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_review_images_review ON public.review_images(review_id);

CREATE INDEX IF NOT EXISTS idx_review_votes_review ON public.review_votes(review_id);
CREATE INDEX IF NOT EXISTS idx_review_votes_customer ON public.review_votes(customer_id);

-- =====================================================
-- TRIGGERS
-- =====================================================

CREATE TRIGGER update_reviews_updated_at
    BEFORE UPDATE ON public.reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Function to update product rating statistics
CREATE OR REPLACE FUNCTION update_product_rating_stats(p_product_id UUID)
RETURNS VOID AS $$
DECLARE
    v_total_reviews INTEGER;
    v_average_rating DECIMAL(3, 2);
    v_rating_1 INTEGER;
    v_rating_2 INTEGER;
    v_rating_3 INTEGER;
    v_rating_4 INTEGER;
    v_rating_5 INTEGER;
BEGIN
    -- Calculate statistics from approved reviews only
    SELECT
        COUNT(*),
        COALESCE(AVG(rating), 0),
        COUNT(*) FILTER (WHERE rating = 1),
        COUNT(*) FILTER (WHERE rating = 2),
        COUNT(*) FILTER (WHERE rating = 3),
        COUNT(*) FILTER (WHERE rating = 4),
        COUNT(*) FILTER (WHERE rating = 5)
    INTO
        v_total_reviews,
        v_average_rating,
        v_rating_1,
        v_rating_2,
        v_rating_3,
        v_rating_4,
        v_rating_5
    FROM public.reviews
    WHERE product_id = p_product_id
      AND is_approved = TRUE;

    -- Upsert statistics
    INSERT INTO public.product_rating_stats (
        product_id,
        total_reviews,
        average_rating,
        rating_1_count,
        rating_2_count,
        rating_3_count,
        rating_4_count,
        rating_5_count,
        updated_at
    ) VALUES (
        p_product_id,
        v_total_reviews,
        v_average_rating,
        v_rating_1,
        v_rating_2,
        v_rating_3,
        v_rating_4,
        v_rating_5,
        NOW()
    )
    ON CONFLICT (product_id) DO UPDATE SET
        total_reviews = v_total_reviews,
        average_rating = v_average_rating,
        rating_1_count = v_rating_1,
        rating_2_count = v_rating_2,
        rating_3_count = v_rating_3,
        rating_4_count = v_rating_4,
        rating_5_count = v_rating_5,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Trigger to update rating stats when review is created/updated/deleted
CREATE OR REPLACE FUNCTION trigger_update_rating_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM update_product_rating_stats(OLD.product_id);
        RETURN OLD;
    ELSE
        PERFORM update_product_rating_stats(NEW.product_id);
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rating_stats_on_review_change
    AFTER INSERT OR UPDATE OR DELETE ON public.reviews
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_rating_stats();

-- Function to update review helpful counts
CREATE OR REPLACE FUNCTION update_review_helpful_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.is_helpful THEN
            UPDATE public.reviews SET helpful_count = helpful_count + 1 WHERE id = NEW.review_id;
        ELSE
            UPDATE public.reviews SET not_helpful_count = not_helpful_count + 1 WHERE id = NEW.review_id;
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.is_helpful AND NOT NEW.is_helpful THEN
            UPDATE public.reviews
            SET helpful_count = helpful_count - 1, not_helpful_count = not_helpful_count + 1
            WHERE id = NEW.review_id;
        ELSIF NOT OLD.is_helpful AND NEW.is_helpful THEN
            UPDATE public.reviews
            SET helpful_count = helpful_count + 1, not_helpful_count = not_helpful_count - 1
            WHERE id = NEW.review_id;
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.is_helpful THEN
            UPDATE public.reviews SET helpful_count = helpful_count - 1 WHERE id = OLD.review_id;
        ELSE
            UPDATE public.reviews SET not_helpful_count = not_helpful_count - 1 WHERE id = OLD.review_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_review_helpful_count_trigger
    AFTER INSERT OR UPDATE OR DELETE ON public.review_votes
    FOR EACH ROW
    EXECUTE FUNCTION update_review_helpful_count();

-- Function to check if customer has purchased product
CREATE OR REPLACE FUNCTION has_purchased_product(p_customer_id UUID, p_product_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    purchased BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM public.order_items oi
        INNER JOIN public.orders o ON oi.order_id = o.id
        WHERE o.customer_id = p_customer_id
          AND oi.product_id = p_product_id
          AND o.payment_status = 'paid'
    ) INTO purchased;

    RETURN purchased;
END;
$$ LANGUAGE plpgsql;
