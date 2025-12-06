import { Product, ProductFilter, ProductCategory } from '@/types/product';
import { products } from '@/data/products';

// Product categories
const mockCategories: ProductCategory[] = [
  {
    id: '1',
    name: 'Tees',
    slug: 'tees',
    description: 'Premium crew neck and athletic fit tees',
  },
  {
    id: '2',
    name: 'Hoodies',
    slug: 'hoodies',
    description: 'Elevated hoodies and sweatshirts',
  },
  {
    id: '3',
    name: 'Caps',
    slug: 'caps',
    description: 'Structured and unstructured caps',
  },
  {
    id: '4',
    name: 'Outerwear',
    slug: 'outerwear',
    description: 'Jackets and outerwear pieces',
  },
  {
    id: '5',
    name: 'Accessories',
    slug: 'accessories',
    description: 'Bags, scarves, and other accessories',
  },
];

export const productService = {
  async getProducts(filters: ProductFilter): Promise<Product[]> {
    // Use products from data/products.ts with comprehensive filtering
    let filtered = [...products];

    // Category filter
    if (filters.category) {
      const categoryLower = filters.category.toLowerCase();
      filtered = filtered.filter(
        p => p.category.toLowerCase() === categoryLower || p.category === categoryLower
      );
    }

    // Collection filter
    if (filters.collection) {
      filtered = filtered.filter(p => p.collection === filters.collection);
    }

    // Color filter
    if (filters.color) {
      const colorLower = filters.color.toLowerCase();
      filtered = filtered.filter(p =>
        p.colors.some(c => c.name.toLowerCase().includes(colorLower))
      );
    }

    // Size filter
    if (filters.size) {
      filtered = filtered.filter(p => p.sizes.some(s => s.label === filters.size));
    }

    // Stock filter
    if (filters.inStockOnly) {
      filtered = filtered.filter(p => p.inStock && p.availableQuantity > 0);
    }

    // New arrivals filter
    if (filters.newArrivalsOnly) {
      filtered = filtered.filter(p => p.isNewArrival);
    }

    // Featured filter
    if (filters.featuredOnly) {
      filtered = filtered.filter(p => p.isFeatured);
    }

    // Search filter (fuzzy match)
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(
        p =>
          p.name.toLowerCase().includes(searchLower) ||
          p.description.toLowerCase().includes(searchLower) ||
          p.tagline.toLowerCase().includes(searchLower) ||
          p.materials.some(m => m.toLowerCase().includes(searchLower)) ||
          p.features.some(f => f.toLowerCase().includes(searchLower))
      );
    }

    // Price range filter
    if (filters.priceRange) {
      filtered = filtered.filter(
        p => p.price >= filters.priceRange![0] && p.price <= filters.priceRange![1]
      );
    }

    // Sort
    switch (filters.sortBy) {
      case 'price-low':
        filtered.sort((a, b) => a.price - b.price);
        break;
      case 'price-high':
        filtered.sort((a, b) => b.price - a.price);
        break;
      case 'popular':
        filtered.sort((a, b) => b.availableQuantity - a.availableQuantity);
        break;
      case 'featured':
        filtered.sort((a, b) => {
          if (a.isFeatured === b.isFeatured) return 0;
          return a.isFeatured ? -1 : 1;
        });
        break;
      case 'newest':
      default:
        filtered.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
        break;
    }

    // Pagination
    const page = filters.page || 1;
    const limit = filters.limit || 12;
    const start = (page - 1) * limit;
    const end = start + limit;

    return filtered.slice(start, end);
  },

  async getProduct(id: string): Promise<Product | null> {
    return products.find(p => p.id === id) || null;
  },

  async getProductBySlug(slug: string): Promise<Product | null> {
    return products.find(p => p.slug === slug) || null;
  },

  async getCategories(): Promise<ProductCategory[]> {
    return mockCategories;
  },

  async getProductsByCollection(collection: string): Promise<Product[]> {
    return products.filter(p => p.collection === collection);
  },

  async getRelatedProducts(productId: string, limit: number = 4): Promise<Product[]> {
    const product = products.find(p => p.id === productId);
    if (!product) return [];

    return products
      .filter(
        p =>
          p.id !== productId &&
          (p.collection === product.collection || p.category === product.category)
      )
      .slice(0, limit);
  },

  async getFeaturedProducts(limit: number = 6): Promise<Product[]> {
    return products.filter(p => p.isFeatured).slice(0, limit);
  },

  async getNewArrivals(limit: number = 6): Promise<Product[]> {
    return products.filter(p => p.isNewArrival).slice(0, limit);
  },
};
