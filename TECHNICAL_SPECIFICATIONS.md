# **TECHNICAL SPECIFICATIONS**
## **DrMAlowein & Rounaq Platform Implementation Details**

---

## **🏗️ DRMALOWEIN TECHNICAL SPECIFICATIONS**

### **System Architecture**

```typescript
// src/config/drmalowein-config.ts
export const drmaloweinConfig = {
  application: {
    name: "DrMAlowein",
    version: "1.0.0",
    framework: "React 18.3.1",
    language: "TypeScript 5.5.2",
    bundler: "Vite 5.4.0",
    renderer: "React DOM"
  },
  
  styling: {
    system: "Tailwind CSS 3.4.0",
    theme: "Academic Professional Theme",
    colors: {
      primary: "#1e3a8a",      // Academic blue
      secondary: "#dc2626",     // Knowledge red
      accent: "#059669",        // Success green
      neutral: "#6b7280",       // Professional gray
      background: "#ffffff",    // Clean white
      text: "#111827"          // Professional text
    },
    typography: {
      heading: ["Georgia", "serif"],
      body: ["Inter", "sans-serif"],
      mono: ["JetBrains Mono", "monospace"]
    }
  },
  
  cms: {
    platform: "Strapi 4.24.0",
    database: "PostgreSQL 15.0",
    hosting: "Netlify",
    api: "RESTful API with GraphQL support"
  },
  
  features: {
    publications: {
      type: "Dynamic content management",
      search: "Full-text search with filters",
      export: "PDF download capability",
      metrics: "Real-time citation tracking"
    },
    research: {
      showcase: "Interactive project displays",
      collaboration: "Research network integration",
      funding: "Grant and funding tracking"
    },
    teaching: {
      portfolio: "Course management system",
      materials: "Lecture notes and resources",
      evaluation: "Student feedback integration"
    }
  },
  
  performance: {
    target: {
      lighthouse: 95+,
      loadTime: "<2 seconds",
      mobileScore: 95+,
      seoScore: 100
    },
    optimization: {
      bundleSize: "<250KB gzipped",
      imageOptimization: "WebP with fallbacks",
      codeSplitting: "Route-based splitting",
      caching: "Aggressive browser caching"
    }
  }
};
```

### **Component Architecture**

```typescript
// src/types/academic.types.ts
export interface AcademicProfile {
  id: string;
  name: string;
  title: string;
  institution: string;
  email: string;
  bio: string;
  photo: string;
  socialLinks: SocialLink[];
  expertise: string[];
}

export interface Publication {
  id: string;
  title: string;
  abstract: string;
  authors: Author[];
  journal: string;
  date: Date;
  doi: string;
  url: string;
  pdfUrl: string;
  citations: number;
  type: PublicationType;
  status: PublicationStatus;
}

export interface ResearchProject {
  id: string;
  title: string;
  description: string;
  category: ResearchCategory;
  startDate: Date;
  endDate?: Date;
  funding?: FundingInfo;
  collaborators: Collaborator[];
  publications: string[];
  images: string[];
  outcomes: string[];
}

export interface Course {
  id: string;
  title: string;
  code: string;
  level: CourseLevel;
  institution: string;
  semester: string;
  description: string;
  syllabus: string;
  materials: CourseMaterial[];
  enrollment: number;
}

// src/components/academic/PublicationCard.tsx
import React from 'react';
import { Publication } from '../types/academic.types';

interface PublicationCardProps {
  publication: Publication;
  onViewDetails: (id: string) => void;
  onDownloadPDF: (url: string) => void;
}

export const PublicationCard: React.FC<PublicationCardProps> = ({
  publication,
  onViewDetails,
  onDownloadPDF
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-serif text-gray-900 leading-tight">
          {publication.title}
        </h3>
        <span className="text-sm text-gray-500 whitespace-nowrap ml-4">
          {publication.date.getFullYear()}
        </span>
      </div>
      
      <p className="text-gray-600 mb-4 line-clamp-3">
        {publication.abstract}
      </p>
      
      <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
        <span>{publication.journal}</span>
        <span className="font-medium">{publication.citations} citations</span>
      </div>
      
      <div className="flex items-center justify-between">
        <div className="flex flex-wrap gap-2">
          {publication.authors.slice(0, 3).map((author, index) => (
            <span key={index} className="text-xs bg-gray-100 px-2 py-1 rounded">
              {author.name}
            </span>
          ))}
          {publication.authors.length > 3 && (
            <span className="text-xs text-gray-500">+{publication.authors.length - 3} more</span>
          )}
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={() => onViewDetails(publication.id)}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
          >
            View Details
          </button>
          {publication.pdfUrl && (
            <button
              onClick={() => onDownloadPDF(publication.pdfUrl)}
              className="text-green-600 hover:text-green-800 text-sm font-medium"
            >
              Download PDF
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// src/components/academic/ResearchShowcase.tsx
import React from 'react';
import { ResearchProject } from '../types/academic.types';

interface ResearchShowcaseProps {
  projects: ResearchProject[];
  onProjectSelect: (project: ResearchProject) => void;
}

export const ResearchShowcase: React.FC<ResearchShowcaseProps> = ({
  projects,
  onProjectSelect
}) => {
  const categories = [...new Set(projects.map(p => p.category))];
  
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-4">
        <h2 className="text-3xl font-serif text-center mb-12">
          Research Excellence
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project) => (
            <div
              key={project.id}
              className="bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
              onClick={() => onProjectSelect(project)}
            >
              {project.images.length > 0 && (
                <div className="h-48 bg-gray-100">
                  <img
                    src={project.images[0]}
                    alt={project.title}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}
              
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-sm font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">
                    {project.category}
                  </span>
                  <span className="text-sm text-gray-500">
                    {project.startDate.getFullYear()} - {project.endDate?.getFullYear() || 'Present'}
                  </span>
                </div>
                
                <h3 className="text-xl font-serif text-gray-900 mb-3">
                  {project.title}
                </h3>
                
                <p className="text-gray-600 mb-4 line-clamp-3">
                  {project.description}
                </p>
                
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">
                    {project.collaborators.length} collaborators
                  </span>
                  <span className="text-gray-500">
                    {project.publications.length} publications
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
```

### **Data Management**

```typescript
// src/services/academic-api.ts
import { Publication, ResearchProject, Course } from '../types/academic.types';

class AcademicAPI {
  private baseURL = process.env.REACT_APP_API_URL || '/api';
  
  async getPublications(filters?: PublicationFilters): Promise<Publication[]> {
    const params = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, value.toString());
      });
    }
    
    const response = await fetch(`${this.baseURL}/publications?${params}`);
    return response.json();
  }
  
  async getPublication(id: string): Promise<Publication> {
    const response = await fetch(`${this.baseURL}/publications/${id}`);
    return response.json();
  }
  
  async getResearchProjects(): Promise<ResearchProject[]> {
    const response = await fetch(`${this.baseURL}/research`);
    return response.json();
  }
  
  async getCourses(): Promise<Course[]> {
    const response = await fetch(`${this.baseURL}/courses`);
    return response.json();
  }
  
  async getCitationMetrics(authorId: string): Promise<CitationMetrics> {
    const response = await fetch(`${this.baseURL}/citations/${authorId}`);
    return response.json();
  }
  
  async downloadCV(template: CVTemplate): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/cv/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ template })
    });
    return response.blob();
  }
}

export const academicAPI = new AcademicAPI();

// src/hooks/useAcademicData.ts
import { useState, useEffect } from 'react';
import { academicAPI } from '../services/academic-api';
import { Publication, ResearchProject, Course } from '../types/academic.types';

export const usePublications = (filters?: PublicationFilters) => {
  const [publications, setPublications] = useState<Publication[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchPublications = async () => {
      try {
        setLoading(true);
        const data = await academicAPI.getPublications(filters);
        setPublications(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch publications');
      } finally {
        setLoading(false);
      }
    };
    
    fetchPublications();
  }, [filters]);
  
  return { publications, loading, error };
};

export const useResearchProjects = () => {
  const [projects, setProjects] = useState<ResearchProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        setLoading(true);
        const data = await academicAPI.getResearchProjects();
        setProjects(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch projects');
      } finally {
        setLoading(false);
      }
    };
    
    fetchProjects();
  }, []);
  
  return { projects, loading, error };
};
```

---

## **👗 ROUNAQ TECHNICAL SPECIFICATIONS**

### **E-commerce Architecture**

```typescript
// src/config/rounaq-config.ts
export const rounaqConfig = {
  application: {
    name: "Rounaq",
    version: "1.0.0",
    framework: "React 18.3.1",
    language: "TypeScript 5.5.2",
    bundler: "Vite 5.4.0",
    renderer: "React DOM"
  },
  
  styling: {
    system: "Tailwind CSS 3.4.0",
    theme: "Fashion Luxury Theme",
    colors: {
      primary: "#ec4899",      // Fashion pink
      secondary: "#8b5cf6",     // Creative purple
      accent: "#f59e0b",        // Luxury gold
      neutral: "#374151",       // Sophisticated gray
      background: "#fef7f0",    // Warm cream
      text: "#1f2937"          // Elegant text
    },
    typography: {
      heading: ["Playfair Display", "serif"],
      body: ["Source Sans Pro", "sans-serif"],
      accent: ["Dancing Script", "cursive"]
    }
  },
  
  ecommerce: {
    platform: "Shopify Plus",
    payment: "Stripe Payment Processing",
    inventory: "Real-time stock management",
    shipping: "Multiple carrier integration",
    taxes: "Automated tax calculation"
  },
  
  features: {
    catalog: {
      browsing: "Advanced filtering and search",
      visualization: "360° product views",
      customization: "Color and variant selection",
      recommendations: "AI-powered style suggestions"
    },
    shopping: {
      cart: "Persistent shopping cart",
      checkout: "One-page optimized checkout",
      payments: "Multiple payment methods",
      guest_checkout: "Guest purchase option"
    },
    experience: {
      try_on: "AR virtual try-on",
      styling: "Outfit builder and recommendations",
      social: "Social sharing and wishlist",
      loyalty: "Customer rewards program"
    }
  },
  
  performance: {
    target: {
      lighthouse: 90+,
      loadTime: "<3 seconds",
      mobileScore: 90+,
      conversionRate: "2.5%+"
    },
    optimization: {
      bundleSize: "<350KB gzipped",
      imageOptimization: "WebP with responsive images",
      codeSplitting: "Route and component splitting",
      caching: "E-commerce optimized caching"
    }
  }
};
```

### **Product Management**

```typescript
// src/types/fashion.types.ts
export interface Product {
  id: string;
  title: string;
  description: string;
  handle: string;
  vendor: string;
  productType: ProductType;
  category: ProductCategory;
  tags: string[];
  price: Money;
  compareAtPrice?: Money;
  cost?: Money;
  variants: ProductVariant[];
  images: ProductImage[];
  options: ProductOption[];
  metafields?: ProductMetafields;
  seo: SEO;
  createdAt: Date;
  updatedAt: Date;
}

export interface ProductVariant {
  id: string;
  title: string;
  price: Money;
  sku: string;
  barcode?: string;
  inventory: InventoryInfo;
  weight?: number;
  requiresShipping: boolean;
  taxable: boolean;
  image?: ProductImage;
  selectedOptions: SelectedOption[];
}

export interface ProductImage {
  id: string;
  url: string;
  altText?: string;
  width: number;
  height: number;
  position: number;
}

export interface Outfit {
  id: string;
  name: string;
  description: string;
  products: string[];
  occasion: OutfitOccasion;
  season: Season;
  style: StyleCategory;
  image: string;
  createdAt: Date;
}

export interface StyleProfile {
  id: string;
  customerId: string;
  preferences: StylePreferences;
  sizes: SizeProfile;
  colors: ColorPreferences;
  occasions: PreferredOccasions[];
  budget: BudgetRange;
  createdAt: Date;
  updatedAt: Date;
}

// src/components/fashion/ProductCard.tsx
import React, { useState } from 'react';
import { Product, ProductVariant } from '../types/fashion.types';

interface ProductCardProps {
  product: Product;
  onQuickView: (product: Product) => void;
  onAddToCart: (variant: ProductVariant) => void;
  onAddToWishlist: (product: Product) => void;
}

export const ProductCard: React.FC<ProductCardProps> = ({
  product,
  onQuickView,
  onAddToCart,
  onAddToWishlist
}) => {
  const [selectedVariant, setSelectedVariant] = useState<ProductVariant>(product.variants[0]);
  const [isHovered, setIsHovered] = useState(false);
  
  const handleAddToCart = () => {
    onAddToCart(selectedVariant);
  };
  
  return (
    <div 
      className="bg-white group border border-gray-200 rounded-lg overflow-hidden hover:shadow-xl transition-all duration-300"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Product Image */}
      <div className="relative h-80 bg-gray-50 overflow-hidden">
        <img
          src={product.images[0]?.url}
          alt={product.images[0]?.altText || product.title}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
        />
        
        {/* Quick Actions */}
        <div className={`absolute top-4 right-4 space-y-2 transition-opacity duration-300 ${
          isHovered ? 'opacity-100' : 'opacity-0'
        }`}>
          <button
            onClick={() => onQuickView(product)}
            className="p-2 bg-white rounded-full shadow-md hover:shadow-lg transition-shadow"
            aria-label="Quick view"
          >
            <EyeIcon className="w-5 h-5 text-gray-700" />
          </button>
          <button
            onClick={() => onAddToWishlist(product)}
            className="p-2 bg-white rounded-full shadow-md hover:shadow-lg transition-shadow"
            aria-label="Add to wishlist"
          >
            <HeartIcon className="w-5 h-5 text-gray-700" />
          </button>
        </div>
        
        {/* Badge */}
        {product.compareAtPrice && (
          <div className="absolute top-4 left-4">
            <span className="bg-red-500 text-white px-2 py-1 text-sm font-medium rounded">
              Sale
            </span>
          </div>
        )}
      </div>
      
      {/* Product Info */}
      <div className="p-6">
        <div className="mb-2">
          <h3 className="text-lg font-serif text-gray-900 mb-1">
            {product.title}
          </h3>
          <p className="text-sm text-gray-600">{product.vendor}</p>
        </div>
        
        {/* Price */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <span className="text-xl font-bold text-gray-900">
              ${selectedVariant.price.amount}
            </span>
            {product.compareAtPrice && (
              <span className="text-sm text-gray-500 line-through">
                ${product.compareAtPrice.amount}
              </span>
            )}
          </div>
          
          {/* Rating */}
          <div className="flex items-center">
            <div className="flex text-yellow-400">
              {[...Array(5)].map((_, i) => (
                <StarIcon key={i} className="w-4 h-4 fill-current" />
              ))}
            </div>
            <span className="text-sm text-gray-500 ml-1">(24)</span>
          </div>
        </div>
        
        {/* Color Options */}
        {product.options.find(opt => opt.name === 'Color') && (
          <div className="mb-4">
            <div className="flex items-center space-x-2">
              {product.options
                .find(opt => opt.name === 'Color')
                ?.values.map((color, index) => (
                  <button
                    key={index}
                    className="w-6 h-6 rounded-full border-2 border-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500"
                    style={{ backgroundColor: color.toLowerCase() }}
                    aria-label={`Select ${color}`}
                  />
                ))}
            </div>
          </div>
        )}
        
        {/* Add to Cart Button */}
        <button
          onClick={handleAddToCart}
          disabled={!selectedVariant.available}
          className="w-full bg-pink-600 text-white py-3 px-4 rounded-md font-medium hover:bg-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {selectedVariant.available ? 'Add to Cart' : 'Out of Stock'}
        </button>
      </div>
    </div>
  );
};

// src/components/fashion/OutfitBuilder.tsx
import React, { useState } from 'react';
import { Product, Outfit } from '../types/fashion.types';

interface OutfitBuilderProps {
  products: Product[];
  onSaveOutfit: (outfit: Partial<Outfit>) => void;
}

export const OutfitBuilder: React.FC<OutfitBuilderProps> = ({
  products,
  onSaveOutfit
}) => {
  const [selectedProducts, setSelectedProducts] = useState<Product[]>([]);
  const [outfitName, setOutfitName] = useState('');
  const [occasion, setOccasion] = useState<OutfitOccasion>('casual');
  
  const handleProductSelect = (product: Product) => {
    setSelectedProducts(prev => {
      const exists = prev.find(p => p.id === product.id);
      if (exists) {
        return prev.filter(p => p.id !== product.id);
      } else {
        return [...prev, product];
      }
    });
  };
  
  const handleSaveOutfit = () => {
    if (selectedProducts.length > 0 && outfitName) {
      onSaveOutfit({
        name: outfitName,
        products: selectedProducts.map(p => p.id),
        occasion,
        season: 'all-season',
        style: 'contemporary'
      });
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-serif text-gray-900 mb-6">
        Build Your Perfect Outfit
      </h2>
      
      {/* Outfit Details */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Outfit Name
          </label>
          <input
            type="text"
            value={outfitName}
            onChange={(e) => setOutfitName(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-pink-500"
            placeholder="My Perfect Outfit"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Occasion
          </label>
          <select
            value={occasion}
            onChange={(e) => setOccasion(e.target.value as OutfitOccasion)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-pink-500"
          >
            <option value="casual">Casual</option>
            <option value="business">Business</option>
            <option value="formal">Formal</option>
            <option value="party">Party</option>
            <option value="date">Date Night</option>
          </select>
        </div>
      </div>
      
      {/* Selected Products */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Selected Items ({selectedProducts.length})
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {selectedProducts.map((product) => (
            <div key={product.id} className="relative">
              <img
                src={product.images[0]?.url}
                alt={product.title}
                className="w-full h-32 object-cover rounded-md"
              />
              <button
                onClick={() => handleProductSelect(product)}
                className="absolute top-2 right-2 p-1 bg-white rounded-full shadow-md hover:shadow-lg"
              >
                <XIcon className="w-4 h-4 text-gray-700" />
              </button>
              <p className="text-sm text-gray-700 mt-2 truncate">{product.title}</p>
            </div>
          ))}
        </div>
      </div>
      
      {/* Product Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Add More Items
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
          {products
            .filter(product => !selectedProducts.find(p => p.id === product.id))
            .map((product) => (
              <div
                key={product.id}
                onClick={() => handleProductSelect(product)}
                className="cursor-pointer border border-gray-200 rounded-lg p-3 hover:border-pink-500 transition-colors"
              >
                <img
                  src={product.images[0]?.url}
                  alt={product.title}
                  className="w-full h-24 object-cover rounded-md mb-2"
                />
                <p className="text-sm text-gray-700 truncate">{product.title}</p>
                <p className="text-sm font-medium text-gray-900">
                  ${product.variants[0].price.amount}
                </p>
              </div>
            ))}
        </div>
      </div>
      
      {/* Save Button */}
      <button
        onClick={handleSaveOutfit}
        disabled={selectedProducts.length === 0 || !outfitName}
        className="w-full bg-pink-600 text-white py-3 px-4 rounded-md font-medium hover:bg-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Save Outfit
      </button>
    </div>
  );
};
```

### **Shopping Cart & Checkout**

```typescript
// src/services/shopify-api.ts
import { Product, Cart, Checkout } from '../types/fashion.types';

class ShopifyAPI {
  private storefrontAccessToken: string;
  private domain: string;
  
  constructor() {
    this.storefrontAccessToken = process.env.REACT_APP_SHOPIFY_TOKEN!;
    this.domain = process.env.REACT_APP_SHOPIFY_DOMAIN!;
  }
  
  async createCheckout(cartItems: CartItem[]): Promise<Checkout> {
    const mutation = `
      mutation checkoutCreate($input: CheckoutCreateInput!) {
        checkoutCreate(input: $input) {
          checkout {
            id
            webUrl
            subtotalPriceV2 {
              amount
              currencyCode
            }
            totalTaxV2 {
              amount
              currencyCode
            }
            totalPriceV2 {
              amount
              currencyCode
            }
            lineItems(first: 250) {
              edges {
                node {
                  id
                  title
                  quantity
                  variant {
                    id
                    title
                    priceV2 {
                      amount
                      currencyCode
                    }
                  }
                }
              }
            }
          }
          checkoutUserErrors {
            code
            field
            message
          }
        }
      }
    `;
    
    const variables = {
      input: {
        lineItems: cartItems.map(item => ({
          variantId: item.variantId,
          quantity: item.quantity
        }))
      }
    };
    
    const response = await this.fetchGraphQL(mutation, variables);
    return response.data.checkoutCreate.checkout;
  }
  
  async updateCheckout(checkoutId: string, lineItems: CartItem[]): Promise<Checkout> {
    const mutation = `
      mutation checkoutLineItemsReplace($checkoutId: ID!, $lineItems: [CheckoutLineItemInput!]!) {
        checkoutLineItemsReplace(checkoutId: $checkoutId, lineItems: $lineItems) {
          checkout {
            id
            webUrl
            subtotalPriceV2 {
              amount
              currencyCode
            }
            totalTaxV2 {
              amount
              currencyCode
            }
            totalPriceV2 {
              amount
              currencyCode
            }
            lineItems(first: 250) {
              edges {
                node {
                  id
                  title
                  quantity
                  variant {
                    id
                    title
                    priceV2 {
                      amount
                      currencyCode
                    }
                  }
                }
              }
            }
          }
          checkoutUserErrors {
            code
            field
            message
          }
        }
      }
    `;
    
    const variables = {
      checkoutId,
      lineItems: lineItems.map(item => ({
        variantId: item.variantId,
        quantity: item.quantity
      }))
    };
    
    const response = await this.fetchGraphQL(mutation, variables);
    return response.data.checkoutLineItemsReplace.checkout;
  }
  
  async searchProducts(query: string, first: number = 20): Promise<Product[]> {
    const searchQuery = `
      query searchProducts($query: String!, $first: Int!) {
        products(first: $first, query: $query) {
          edges {
            node {
              id
              title
              handle
              description
              vendor
              productType
              tags
              priceRangeV2 {
                minVariantPrice {
                  amount
                  currencyCode
                }
                maxVariantPrice {
                  amount
                  currencyCode
                }
              }
              images(first: 5) {
                edges {
                  node {
                    id
                    url
                    altText
                    width
                    height
                  }
                }
              }
              variants(first: 10) {
                edges {
                  node {
                    id
                    title
                    sku
                    priceV2 {
                      amount
                      currencyCode
                    }
                    availableForSale
                    selectedOptions {
                      name
                      value
                    }
                    image {
                      id
                      url
                      altText
                      width
                      height
                    }
                  }
                }
              }
            }
          }
        }
      }
    `;
    
    const variables = { query, first };
    const response = await this.fetchGraphQL(searchQuery, variables);
    return response.data.products.edges.map((edge: any) => this.transformProduct(edge.node));
  }
  
  private async fetchGraphQL(query: string, variables: any) {
    const response = await fetch(`https://${this.domain}/api/2023-10/graphql.json`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Shopify-Storefront-Access-Token': this.storefrontAccessToken
      },
      body: JSON.stringify({ query, variables })
    });
    
    if (!response.ok) {
      throw new Error('Shopify API request failed');
    }
    
    return response.json();
  }
  
  private transformProduct(shopifyProduct: any): Product {
    return {
      id: shopifyProduct.id,
      title: shopifyProduct.title,
      description: shopifyProduct.description,
      handle: shopifyProduct.handle,
      vendor: shopifyProduct.vendor,
      productType: shopifyProduct.productType,
      category: this.categorizeProduct(shopifyProduct.productType, shopifyProduct.tags),
      tags: shopifyProduct.tags,
      price: shopifyProduct.priceRangeV2.minVariantPrice,
      compareAtPrice: shopifyProduct.priceRangeV2.maxVariantPrice,
      variants: shopifyProduct.variants.edges.map((edge: any) => this.transformVariant(edge.node)),
      images: shopifyProduct.images.edges.map((edge: any) => this.transformImage(edge.node)),
      options: this.extractOptions(shopifyProduct.variants.edges),
      seo: {
        title: shopifyProduct.title,
        description: shopifyProduct.description
      },
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }
  
  private transformVariant(shopifyVariant: any): ProductVariant {
    return {
      id: shopifyVariant.id,
      title: shopifyVariant.title,
      price: shopifyVariant.priceV2,
      sku: shopifyVariant.sku,
      inventory: {
        available: shopifyVariant.availableForSale,
        quantity: 0 // Would need admin API for actual quantity
      },
      requiresShipping: true,
      taxable: true,
      selectedOptions: shopifyVariant.selectedOptions,
      image: shopifyVariant.image ? this.transformImage(shopifyVariant.image) : undefined
    };
  }
  
  private transformImage(shopifyImage: any): ProductImage {
    return {
      id: shopifyImage.id,
      url: shopifyImage.url,
      altText: shopifyImage.altText,
      width: shopifyImage.width,
      height: shopifyImage.height,
      position: 0
    };
  }
  
  private categorizeProduct(productType: string, tags: string[]): ProductCategory {
    // Logic to categorize products based on type and tags
    if (tags.includes('clothing') || productType.includes('Clothing')) {
      return 'clothing';
    }
    if (tags.includes('accessories') || productType.includes('Accessory')) {
      return 'accessories';
    }
    if (tags.includes('footwear') || productType.includes('Shoes')) {
      return 'footwear';
    }
    return 'clothing'; // Default
  }
  
  private extractOptions(variants: any[]): ProductOption[] {
    const options: ProductOption[] = [];
    const optionMap = new Map<string, Set<string>>();
    
    variants.forEach(({ node: variant }) => {
      variant.selectedOptions.forEach((option: any) => {
        if (!optionMap.has(option.name)) {
          optionMap.set(option.name, new Set());
        }
        optionMap.get(option.name)!.add(option.value);
      });
    });
    
    optionMap.forEach((values, name) => {
      options.push({
        name,
        values: Array.from(values)
      });
    });
    
    return options;
  }
}

export const shopifyAPI = new ShopifyAPI();

// src/hooks/useShoppingCart.ts
import { useState, useEffect, useCallback } from 'react';
import { shopifyAPI } from '../services/shopify-api';
import { Cart, CartItem, Product, ProductVariant } from '../types/fashion.types';

export const useShoppingCart = () => {
  const [cart, setCart] = useState<Cart>({
    id: '',
    items: [],
    subtotal: { amount: '0', currencyCode: 'USD' },
    totalTax: { amount: '0', currencyCode: 'USD' },
    totalPrice: { amount: '0', currencyCode: 'USD' },
    checkoutUrl: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const addItem = useCallback(async (variant: ProductVariant, quantity: number = 1) => {
    try {
      setLoading(true);
      setError(null);
      
      const existingItem = cart.items.find(item => item.variantId === variant.id);
      let newItems: CartItem[];
      
      if (existingItem) {
        newItems = cart.items.map(item =>
          item.variantId === variant.id
            ? { ...item, quantity: item.quantity + quantity }
            : item
        );
      } else {
        newItems = [...cart.items, {
          id: `${variant.id}-${Date.now()}`,
          variantId: variant.id,
          quantity,
          title: variant.title,
          price: variant.price,
          image: variant.image
        }];
      }
      
      const checkout = await shopifyAPI.updateCheckout(cart.id, newItems);
      setCart({
        id: checkout.id,
        items: checkout.lineItems.edges.map((edge: any) => ({
          id: edge.node.id,
          variantId: edge.node.variant.id,
          quantity: edge.node.quantity,
          title: edge.node.title,
          price: edge.node.variant.priceV2,
          image: edge.node.variant.image
        })),
        subtotal: checkout.subtotalPriceV2,
        totalTax: checkout.totalTaxV2,
        totalPrice: checkout.totalPriceV2,
        checkoutUrl: checkout.webUrl
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add item to cart');
    } finally {
      setLoading(false);
    }
  }, [cart]);
  
  const removeItem = useCallback(async (variantId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const newItems = cart.items.filter(item => item.variantId !== variantId);
      const checkout = await shopifyAPI.updateCheckout(cart.id, newItems);
      
      setCart({
        id: checkout.id,
        items: checkout.lineItems.edges.map((edge: any) => ({
          id: edge.node.id,
          variantId: edge.node.variant.id,
          quantity: edge.node.quantity,
          title: edge.node.title,
          price: edge.node.variant.priceV2,
          image: edge.node.variant.image
        })),
        subtotal: checkout.subtotalPriceV2,
        totalTax: checkout.totalTaxV2,
        totalPrice: checkout.totalPriceV2,
        checkoutUrl: checkout.webUrl
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove item from cart');
    } finally {
      setLoading(false);
    }
  }, [cart]);
  
  const updateQuantity = useCallback(async (variantId: string, quantity: number) => {
    if (quantity === 0) {
      removeItem(variantId);
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const newItems = cart.items.map(item =>
        item.variantId === variantId ? { ...item, quantity } : item
      );
      
      const checkout = await shopifyAPI.updateCheckout(cart.id, newItems);
      setCart({
        id: checkout.id,
        items: checkout.lineItems.edges.map((edge: any) => ({
          id: edge.node.id,
          variantId: edge.node.variant.id,
          quantity: edge.node.quantity,
          title: edge.node.title,
          price: edge.node.variant.priceV2,
          image: edge.node.variant.image
        })),
        subtotal: checkout.subtotalPriceV2,
        totalTax: checkout.totalTaxV2,
        totalPrice: checkout.totalPriceV2,
        checkoutUrl: checkout.webUrl
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update quantity');
    } finally {
      setLoading(false);
    }
  }, [cart, removeItem]);
  
  const clearCart = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const checkout = await shopifyAPI.updateCheckout(cart.id, []);
      setCart({
        id: checkout.id,
        items: [],
        subtotal: checkout.subtotalPriceV2,
        totalTax: checkout.totalTaxV2,
        totalPrice: checkout.totalPriceV2,
        checkoutUrl: checkout.webUrl
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear cart');
    } finally {
      setLoading(false);
    }
  }, [cart.id]);
  
  return {
    cart,
    loading,
    error,
    addItem,
    removeItem,
    updateQuantity,
    clearCart
  };
};
```

---

## **🔧 SHARED INFRASTRUCTURE**

### **Monitoring & Analytics**

```typescript
// src/services/analytics.ts
import { AnalyticsBrowser } from '@segment/analytics-next';

class AnalyticsService {
  private analytics: AnalyticsBrowser;
  
  constructor() {
    this.analytics = new AnalyticsBrowser();
    this.initialize();
  }
  
  private async initialize() {
    await this.analytics.load({ writeKey: process.env.REACT_APP_SEGMENT_WRITE_KEY! });
  }
  
  trackPageView(path: string, title: string) {
    this.analytics.page({
      path,
      title,
      url: window.location.href
    });
  }
  
  trackEvent(event: string, properties: Record<string, any>) {
    this.analytics.track(event, properties);
  }
  
  trackUser(userId: string, traits: Record<string, any>) {
    this.analytics.identify(userId, traits);
  }
  
  // Academic specific events
  trackPublicationView(publicationId: string, title: string) {
    this.trackEvent('Publication Viewed', {
      publicationId,
      title,
      timestamp: new Date().toISOString()
    });
  }
  
  trackCitationClick(publicationId: string, citationUrl: string) {
    this.trackEvent('Citation Clicked', {
      publicationId,
      citationUrl,
      timestamp: new Date().toISOString()
    });
  }
  
  trackCVDownload(template: string) {
    this.trackEvent('CV Downloaded', {
      template,
      timestamp: new Date().toISOString()
    });
  }
  
  // Fashion specific events
  trackProductView(productId: string, title: string, price: number) {
    this.trackEvent('Product Viewed', {
      productId,
      title,
      price,
      timestamp: new Date().toISOString()
    });
  }
  
  trackAddToCart(productId: string, variantId: string, price: number, quantity: number) {
    this.trackEvent('Product Added', {
      productId,
      variantId,
      price,
      quantity,
      timestamp: new Date().toISOString()
    });
  }
  
  trackPurchase(orderId: string, total: number, items: any[]) {
    this.trackEvent('Order Completed', {
      orderId,
      total,
      items,
      timestamp: new Date().toISOString()
    });
  }
  
  trackStyleQuizCompletion(profile: any) {
    this.trackEvent('Style Quiz Completed', {
      profile,
      timestamp: new Date().toISOString()
    });
  }
}

export const analyticsService = new AnalyticsService();

// src/services/performance.ts
class PerformanceService {
  private metrics: PerformanceMetric[] = [];
  
  startMeasure(name: string) {
    performance.mark(`${name}-start`);
  }
  
  endMeasure(name: string): number {
    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);
    
    const measure = performance.getEntriesByName(name, 'measure')[0];
    const duration = measure.duration;
    
    this.recordMetric(name, duration);
    return duration;
  }
  
  recordMetric(name: string, value: number) {
    this.metrics.push({
      name,
      value,
      timestamp: Date.now(),
      url: window.location.pathname
    });
    
    // Send to analytics service
    analyticsService.trackEvent('Performance Metric', {
      metricName: name,
      value,
      url: window.location.pathname
    });
    
    // Keep only last 100 metrics
    if (this.metrics.length > 100) {
      this.metrics = this.metrics.slice(-100);
    }
  }
  
  getMetrics(): PerformanceMetric[] {
    return this.metrics;
  }
  
  getAverageMetric(name: string): number {
    const filteredMetrics = this.metrics.filter(m => m.name === name);
    if (filteredMetrics.length === 0) return 0;
    
    const sum = filteredMetrics.reduce((acc, m) => acc + m.value, 0);
    return sum / filteredMetrics.length;
  }
  
  // Core Web Vitals
  observeCoreWebVitals() {
    // Largest Contentful Paint (LCP)
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.recordMetric('LCP', lastEntry.startTime);
    }).observe({ entryTypes: ['largest-contentful-paint'] });
    
    // First Input Delay (FID)
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry: any) => {
        this.recordMetric('FID', entry.processingStart - entry.startTime);
      });
    }).observe({ entryTypes: ['first-input'] });
    
    // Cumulative Layout Shift (CLS)
    let clsValue = 0;
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry: any) => {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      });
      this.recordMetric('CLS', clsValue);
    }).observe({ entryTypes: ['layout-shift'] });
  }
}

export const performanceService = new PerformanceService();

// src/services/error-tracking.ts
import * as Sentry from '@sentry/react';

class ErrorTrackingService {
  constructor() {
    this.initializeSentry();
  }
  
  private initializeSentry() {
    Sentry.init({
      dsn: process.env.REACT_APP_SENTRY_DSN,
      environment: process.env.NODE_ENV,
      tracesSampleRate: 1.0,
      beforeSend(event) {
        // Filter out certain errors if needed
        return event;
      }
    });
  }
  
  captureError(error: Error, context?: Record<string, any>) {
    Sentry.captureException(error, {
      contexts: { custom: context }
    });
  }
  
  captureMessage(message: string, level: Sentry.SeverityLevel = 'info') {
    Sentry.captureMessage(message, level);
  }
  
  setUser(user: { id: string; email: string; username: string }) {
    Sentry.setUser(user);
  }
  
  clearUser() {
    Sentry.setUser(null);
  }
  
  addBreadcrumb(breadcrumb: Sentry.Breadcrumb) {
    Sentry.addBreadcrumb(breadcrumb);
  }
}

export const errorTrackingService = new ErrorTrackingService();
```

### **Security & Authentication**

```typescript
// src/services/auth.ts
import { jwtDecode } from 'jwt-decode';

interface AuthUser {
  id: string;
  email: string;
  role: 'admin' | 'customer' | 'academic';
  permissions: string[];
}

class AuthService {
  private token: string | null = null;
  private user: AuthUser | null = null;
  
  async login(email: string, password: string): Promise<AuthUser> {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        throw new Error('Login failed');
      }
      
      const { token, user } = await response.json();
      this.setToken(token);
      this.setUser(user);
      return user;
    } catch (error) {
      errorTrackingService.captureError(error as Error, { email });
      throw error;
    }
  }
  
  async register(userData: {
    email: string;
    password: string;
    name: string;
    role: 'customer' | 'academic';
  }): Promise<AuthUser> {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      });
      
      if (!response.ok) {
        throw new Error('Registration failed');
      }
      
      const { token, user } = await response.json();
      this.setToken(token);
      this.setUser(user);
      return user;
    } catch (error) {
      errorTrackingService.captureError(error as Error, { email: userData.email });
      throw error;
    }
  }
  
  async logout() {
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.token}`
        }
      });
    } catch (error) {
      errorTrackingService.captureError(error as Error);
    } finally {
      this.clearAuth();
    }
  }
  
  private setToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
    
    // Set user from token
    const decoded = jwtDecode<AuthUser>(token);
    this.setUser(decoded);
  }
  
  private setUser(user: AuthUser) {
    this.user = user;
    errorTrackingService.setUser({
      id: user.id,
      email: user.email,
      username: user.email
    });
  }
  
  private clearAuth() {
    this.token = null;
    this.user = null;
    localStorage.removeItem('auth_token');
    errorTrackingService.clearUser();
  }
  
  getToken(): string | null {
    if (!this.token) {
      const stored = localStorage.getItem('auth_token');
      if (stored) {
        this.setToken(stored);
      }
    }
    return this.token;
  }
  
  getUser(): AuthUser | null {
    return this.user;
  }
  
  isAuthenticated(): boolean {
    return !!this.getToken();
  }
  
  hasPermission(permission: string): boolean {
    return this.user?.permissions.includes(permission) || false;
  }
  
  isAdmin(): boolean {
    return this.user?.role === 'admin';
  }
  
  // Refresh token before expiry
  async refreshToken(): Promise<boolean> {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Token refresh failed');
      }
      
      const { token } = await response.json();
      this.setToken(token);
      return true;
    } catch (error) {
      errorTrackingService.captureError(error as Error);
      this.clearAuth();
      return false;
    }
  }
}

export const authService = new AuthService();

// src/hooks/useAuth.ts
import { useState, useEffect, useCallback } from 'react';
import { authService } from '../services/auth';

export const useAuth = () => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const initAuth = async () => {
      try {
        if (authService.isAuthenticated()) {
          const currentUser = authService.getUser();
          setUser(currentUser);
          
          // Check if token needs refresh
          const token = authService.getToken();
          if (token) {
            const decoded = jwtDecode(token);
            const now = Date.now() / 1000;
            if (decoded.exp! - now < 300) { // Less than 5 minutes
              await authService.refreshToken();
              setUser(authService.getUser());
            }
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Auth initialization failed');
      } finally {
        setLoading(false);
      }
    };
    
    initAuth();
  }, []);
  
  const login = useCallback(async (email: string, password: string) => {
    try {
      setLoading(true);
      setError(null);
      const user = await authService.login(email, password);
      setUser(user);
      
      analyticsService.trackUser(user.id, {
        email: user.email,
        role: user.role
      });
      
      analyticsService.trackEvent('User Logged In', {
        role: user.role,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  const register = useCallback(async (userData: {
    email: string;
    password: string;
    name: string;
    role: 'customer' | 'academic';
  }) => {
    try {
      setLoading(true);
      setError(null);
      const user = await authService.register(userData);
      setUser(user);
      
      analyticsService.trackUser(user.id, {
        email: user.email,
        role: user.role
      });
      
      analyticsService.trackEvent('User Registered', {
        role: user.role,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  const logout = useCallback(async () => {
    try {
      setLoading(true);
      await authService.logout();
      setUser(null);
      
      analyticsService.trackEvent('User Logged Out', {
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Logout failed');
    } finally {
      setLoading(false);
    }
  }, []);
  
  return {
    user,
    loading,
    error,
    isAuthenticated: authService.isAuthenticated(),
    login,
    register,
    logout,
    hasPermission: authService.hasPermission.bind(authService),
    isAdmin: authService.isAdmin()
  };
};
```

---

## **🚀 DEPLOYMENT CONFIGURATION**

### **CI/CD Pipeline**

```yaml
# .github/workflows/deploy.yml
name: Deploy Family Websites

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '18'
  CACHE_VERSION: 'v1'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Run linting
        run: npm run lint
        
      - name: Run type checking
        run: npm run type-check
        
      - name: Run unit tests
        run: npm run test:unit
        
      - name: Run integration tests
        run: npm run test:integration
        
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info

  build-drmalowein:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Build DrMAlowein
        run: |
          cd apps/drmalowein
          npm run build:production
          
      - name: Deploy to Netlify
        uses: nwtgck/actions-netlify@v2
        with:
          publish-dir: 'apps/drmalowein/dist'
          production-deploy: true
        env:
          NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
          NETLIFY_SITE_ID: ${{ secrets.NETLIFY_DRMALOWEIN_SITE_ID }}

  build-rounaq:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Build Rounaq
        run: |
          cd apps/rounaq
          npm run build:production
          
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_ROUNAQ_PROJECT_ID }}
          working-directory: ./apps/rounaq

  security-scan:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Run security audit
        run: npm audit --audit-level=moderate
        
      - name: Run Snyk security scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

  performance-audit:
    needs: [build-drmalowein, build-rounaq]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Run Lighthouse CI
        uses: treosh/lighthouse-ci-action@v9
        with:
          urls: |
            https://drmalowein.com
            https://rounaq.com
          configPath: './lighthouserc.json'
          uploadArtifacts: true
          temporaryPublicStorage: true
```

### **Environment Configuration**

```typescript
// config/environments.ts
export interface EnvironmentConfig {
  api: {
    baseUrl: string;
    timeout: number;
  };
  auth: {
    tokenRefreshThreshold: number;
    sessionTimeout: number;
  };
  analytics: {
    segmentWriteKey: string;
    googleAnalyticsId: string;
  };
  monitoring: {
    sentryDsn: string;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
  };
  features: {
    enableAnalytics: boolean;
    enableErrorTracking: boolean;
    enablePerformanceMonitoring: boolean;
    enableDebugMode: boolean;
  };
}

const development: EnvironmentConfig = {
  api: {
    baseUrl: 'http://localhost:3001/api',
    timeout: 10000
  },
  auth: {
    tokenRefreshThreshold: 300,
    sessionTimeout: 3600
  },
  analytics: {
    segmentWriteKey: 'dev_segment_key',
    googleAnalyticsId: 'GA_MEASUREMENT_ID_DEV'
  },
  monitoring: {
    sentryDsn: 'development_sentry_dsn',
    logLevel: 'debug'
  },
  features: {
    enableAnalytics: false,
    enableErrorTracking: true,
    enablePerformanceMonitoring: true,
    enableDebugMode: true
  }
};

const staging: EnvironmentConfig = {
  ...development,
  api: {
    baseUrl: 'https://staging-api.family-platforms.com/api',
    timeout: 10000
  },
  analytics: {
    segmentWriteKey: 'staging_segment_key',
    googleAnalyticsId: 'GA_MEASUREMENT_ID_STAGING'
  },
  monitoring: {
    sentryDsn: 'staging_sentry_dsn',
    logLevel: 'info'
  },
  features: {
    enableAnalytics: true,
    enableErrorTracking: true,
    enablePerformanceMonitoring: true,
    enableDebugMode: false
  }
};

const production: EnvironmentConfig = {
  ...staging,
  api: {
    baseUrl: 'https://api.family-platforms.com/api',
    timeout: 10000
  },
  analytics: {
    segmentWriteKey: process.env.REACT_APP_SEGMENT_WRITE_KEY!,
    googleAnalyticsId: process.env.REACT_APP_GA_MEASUREMENT_ID!
  },
  monitoring: {
    sentryDsn: process.env.REACT_APP_SENTRY_DSN!,
    logLevel: 'warn'
  },
  features: {
    enableAnalytics: true,
    enableErrorTracking: true,
    enablePerformanceMonitoring: true,
    enableDebugMode: false
  }
};

export const environments = {
  development,
  staging,
  production
};

export const getCurrentEnvironment = (): EnvironmentConfig => {
  const env = process.env.NODE_ENV as keyof typeof environments;
  return environments[env] || development;
};
```

---

**Technical specifications complete.** This provides comprehensive implementation details for both platforms with production-ready architecture, security, monitoring, and deployment configurations.
