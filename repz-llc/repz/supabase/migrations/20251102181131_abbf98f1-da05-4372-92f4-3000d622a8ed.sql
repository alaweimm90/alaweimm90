-- REPZ Member Portal Database Schema
-- Following Supabase best practices with proper security

-- 1. Create role enum (roles stored separately for security)
CREATE TYPE public.app_role AS ENUM ('member', 'admin');

-- 2. Create subscription tier enum
CREATE TYPE public.subscription_tier AS ENUM ('core', 'adaptive', 'performance', 'longevity');

-- 3. Create subscription status enum
CREATE TYPE public.subscription_status AS ENUM ('active', 'canceled', 'paused', 'expired');

-- 4. Create content type enum
CREATE TYPE public.content_type AS ENUM ('article', 'video', 'guide', 'protocol');

-- 5. Create profiles table (extends auth.users)
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL UNIQUE,
  subscription_tier subscription_tier NOT NULL DEFAULT 'core',
  subscription_status subscription_status NOT NULL DEFAULT 'active',
  stripe_customer_id TEXT,
  last_login TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Create user_roles table (CRITICAL: separate table for security)
CREATE TABLE public.user_roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  role app_role NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(user_id, role)
);

-- 7. Create content table
CREATE TABLE public.content (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  content_type content_type NOT NULL,
  required_tier subscription_tier NOT NULL DEFAULT 'core',
  is_published BOOLEAN NOT NULL DEFAULT false,
  published_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 8. Create indexes for performance
CREATE INDEX idx_profiles_email ON public.profiles(email);
CREATE INDEX idx_profiles_subscription_status ON public.profiles(subscription_status);
CREATE INDEX idx_profiles_subscription_tier ON public.profiles(subscription_tier);
CREATE INDEX idx_user_roles_user_id ON public.user_roles(user_id);
CREATE INDEX idx_content_slug ON public.content(slug);
CREATE INDEX idx_content_type ON public.content(content_type);
CREATE INDEX idx_content_required_tier ON public.content(required_tier);
CREATE INDEX idx_content_is_published ON public.content(is_published);

-- 9. Create security definer function to check user roles
CREATE OR REPLACE FUNCTION public.has_role(_user_id UUID, _role app_role)
RETURNS BOOLEAN
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.user_roles
    WHERE user_id = _user_id
      AND role = _role
  )
$$;

-- 10. Create function to get user's subscription tier
CREATE OR REPLACE FUNCTION public.get_user_tier(_user_id UUID)
RETURNS subscription_tier
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT subscription_tier
  FROM public.profiles
  WHERE id = _user_id
$$;

-- 11. Create function to check tier access
CREATE OR REPLACE FUNCTION public.can_access_tier(_user_tier subscription_tier, _required_tier subscription_tier)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
AS $$
  SELECT CASE _user_tier
    WHEN 'longevity' THEN true
    WHEN 'performance' THEN _required_tier IN ('core', 'adaptive', 'performance')
    WHEN 'adaptive' THEN _required_tier IN ('core', 'adaptive')
    WHEN 'core' THEN _required_tier = 'core'
    ELSE false
  END
$$;

-- 12. Create trigger function for updated_at
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- 13. Create trigger function to auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email)
  VALUES (NEW.id, NEW.email);
  
  -- Assign default 'member' role
  INSERT INTO public.user_roles (user_id, role)
  VALUES (NEW.id, 'member');
  
  RETURN NEW;
END;
$$;

-- 14. Create triggers
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

CREATE TRIGGER handle_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_content_updated_at
  BEFORE UPDATE ON public.content
  FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- 15. Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.content ENABLE ROW LEVEL SECURITY;

-- 16. RLS Policies for profiles
CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "Admins can view all profiles"
  ON public.profiles FOR SELECT
  USING (public.has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can update all profiles"
  ON public.profiles FOR UPDATE
  USING (public.has_role(auth.uid(), 'admin'));

-- 17. RLS Policies for user_roles
CREATE POLICY "Users can view own roles"
  ON public.user_roles FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Admins can manage all roles"
  ON public.user_roles FOR ALL
  USING (public.has_role(auth.uid(), 'admin'));

-- 18. RLS Policies for content
CREATE POLICY "Users can view published content matching their tier"
  ON public.content FOR SELECT
  USING (
    is_published = true 
    AND public.can_access_tier(
      public.get_user_tier(auth.uid()),
      required_tier
    )
  );

CREATE POLICY "Admins can view all content"
  ON public.content FOR SELECT
  USING (public.has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can manage all content"
  ON public.content FOR ALL
  USING (public.has_role(auth.uid(), 'admin'));