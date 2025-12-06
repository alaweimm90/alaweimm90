export interface Product {
  id: string;
  slug: string;
  name: string;
  tagline: string;
  description: string;
  category: 'tees' | 'hoodies' | 'caps' | 'accessories' | 'outerwear';
  collection: 'heritage' | 'performance' | 'urban';
  price: number;
  compareAtPrice?: number;
  currency: 'USD';
  colors: ProductColor[];
  sizes: ProductSize[];
  images: ProductImage[];
  materials: string[];
  features: string[];
  careInstructions: string[];
  sku: string;
  inStock: boolean;
  isNewArrival: boolean;
  isFeatured: boolean;
  availableQuantity: number;
  seoTitle: string;
  seoDescription: string;
  inventory: {
    quantity: number;
    trackInventory: boolean;
  };
  variants?: ProductVariant[];
  metadata?: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

export interface ProductColor {
  name: string;
  hex: string;
  images: string[];
}

export interface ProductSize {
  label: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL' | 'ONE';
  inStock: boolean;
  measurements?: {
    chest?: number;
    length?: number;
    sleeve?: number;
    circumference?: number;
    [key: string]: number | undefined;
  };
}

export interface ProductImage {
  id: string;
  url: string;
  altText: string;
  position: number;
}

export interface ProductVariant {
  id: string;
  name: string;
  values: {
    size?: string;
    color?: string;
    material?: string;
    [key: string]: string | undefined;
  };
  sku: string;
  price: number;
  inventory: number;
  images?: ProductImage[];
}

export interface ProductCategory {
  id: string;
  name: string;
  slug: string;
  description: string;
  image?: string;
  parent?: string;
}

export interface ProductFilter {
  category?: string;
  collection?: string;
  color?: string;
  size?: string;
  priceRange?: [number, number];
  search?: string;
  inStockOnly?: boolean;
  newArrivalsOnly?: boolean;
  featuredOnly?: boolean;
  sortBy?: 'newest' | 'price-low' | 'price-high' | 'popular' | 'featured';
  page?: number;
  limit?: number;
}
