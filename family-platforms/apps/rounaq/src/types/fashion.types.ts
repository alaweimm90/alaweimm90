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
  availableForSale: boolean;
  requiresShipping: boolean;
  taxable: boolean;
}

export interface ProductVariant {
  id: string;
  title: string;
  price: Money;
  sku: string;
  barcode?: string;
  inventory: InventoryInfo;
  weight?: number;
  weightUnit?: 'KG' | 'G' | 'LB' | 'OZ';
  requiresShipping: boolean;
  taxable: boolean;
  image?: ProductImage;
  selectedOptions: SelectedOption[];
  position: number;
  availableForSale: boolean;
}

export interface ProductImage {
  id: string;
  url: string;
  altText?: string;
  width: number;
  height: number;
  position: number;
}

export interface ProductOption {
  id: string;
  name: string;
  values: string[];
  position: number;
}

export interface SelectedOption {
  name: string;
  value: string;
}

export interface InventoryInfo {
  available: number;
  location?: string;
  requiresTracking: boolean;
  policy: 'deny' | 'continue';
}

export interface Money {
  amount: string;
  currencyCode: string;
}

export interface ProductMetafields {
  materials?: string[];
  careInstructions?: string;
  sizingGuide?: string;
  sustainability?: string;
  designerNotes?: string;
}

export interface SEO {
  title?: string;
  description?: string;
  keywords?: string[];
}

export interface Collection {
  id: string;
  title: string;
  description: string;
  handle: string;
  image?: CollectionImage;
  products: string[];
  rules?: CollectionRule[];
  sortOrder: 'MANUAL' | 'BEST_SELLING' | 'ALPHABETICAL_ASC' | 'ALPHABETICAL_DESC' | 'PRICE_DESC' | 'PRICE_ASC' | 'CREATED_DESC' | 'CREATED';
  publishedAt: Date;
  updatedAt: Date;
}

export interface CollectionImage {
  id: string;
  url: string;
  altText?: string;
  width: number;
  height: number;
}

export interface CollectionRule {
  column: 'TAG' | 'TYPE' | 'VENDOR' | 'PRODUCT_TYPE' | 'TITLE';
  relation: 'EQUALS' | 'NOT_EQUALS' | 'CONTAINS' | 'NOT_CONTAINS' | 'STARTS_WITH' | 'ENDS_WITH';
  condition: string;
}

export interface Cart {
  id: string;
  items: CartItem[];
  subtotal: Money;
  totalTax: Money;
  totalPrice: Money;
  currencyCode: string;
  checkoutUrl: string;
  estimatedCost: {
    subtotalPrice: Money;
    totalTax: Money;
    totalShippingPrice: Money;
    totalDuties: Money;
    totalDiscount: Money;
  };
}

export interface CartItem {
  id: string;
  variantId: string;
  quantity: number;
  title: string;
  price: Money;
  compareAtPrice?: Money;
  image?: ProductImage;
  product: {
    id: string;
    handle: string;
    title: string;
    availableForSale: boolean;
  };
  customAttributes?: CartAttribute[];
}

export interface CartAttribute {
  key: string;
  value: string;
}

export interface Customer {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  phone?: string;
  acceptsMarketing: boolean;
  orders: string[];
  addresses: CustomerAddress[];
  defaultAddress?: CustomerAddress;
  createdAt: Date;
  updatedAt: Date;
}

export interface CustomerAddress {
  id: string;
  address1: string;
  address2?: string;
  city: string;
  province?: string;
  country: string;
  zip: string;
  phone?: string;
  firstName: string;
  lastName: string;
  company?: string;
}

export interface Order {
  id: string;
  orderNumber: number;
  email: string;
  createdAt: Date;
  processedAt: Date;
  currencyCode: string;
  totalPrice: Money;
  subtotalPrice: Money;
  totalTax: Money;
  totalShippingPrice: Money;
  totalDiscount: Money;
  lineItems: OrderLineItem[];
  shippingAddress?: CustomerAddress;
  billingAddress?: CustomerAddress;
  status: OrderStatus;
  financialStatus: FinancialStatus;
  fulfillmentStatus: FulfillmentStatus;
}

export interface OrderLineItem {
  id: string;
  title: string;
  quantity: number;
  variant?: ProductVariant;
  product: Product;
  price: Money;
  totalDiscount: Money;
  customAttributes?: CartAttribute[];
}

export interface Wishlist {
  id: string;
  customerId: string;
  items: WishlistItem[];
  createdAt: Date;
  updatedAt: Date;
  name: string;
  isPublic: boolean;
}

export interface WishlistItem {
  id: string;
  productId: string;
  variantId?: string;
  addedAt: Date;
  notes?: string;
}

export interface StyleRecommendation {
  id: string;
  customerId: string;
  products: string[];
  occasion: StyleOccasion;
  season: Season;
  styleProfile: StyleProfile;
  confidence: number;
  createdAt: Date;
}

export interface StyleProfile {
  preferences: {
    colors: string[];
    styles: string[];
    materials: string[];
    priceRange: {
      min: number;
      max: number;
    };
  };
  measurements?: {
    size: string;
    fit: 'slim' | 'regular' | 'relaxed' | 'oversized';
    height: string;
    weight?: string;
  };
  occasions: string[];
}

export interface VirtualTryOn {
  id: string;
  customerId: string;
  productId: string;
  variantId: string;
  imageUrl: string;
  resultUrl?: string;
  status: 'processing' | 'completed' | 'failed';
  createdAt: Date;
  completedAt?: Date;
}

export interface FashionShowcase {
  id: string;
  title: string;
  description: string;
  images: ShowcaseImage[];
  products: string[];
  season: Season;
  year: number;
  designer?: string;
  location?: string;
  publishedAt: Date;
}

export interface ShowcaseImage {
  id: string;
  url: string;
  altText?: string;
  width: number;
  height: number;
  position: number;
  isHero?: boolean;
}

// Type Definitions
export type ProductType =
  | 'clothing'
  | 'accessories'
  | 'footwear'
  | 'handbags'
  | 'jewelry'
  | 'scarves'
  | 'belts'
  | 'hats'
  | 'gloves';

export type ProductCategory =
  | 'clothing'
  | 'accessories'
  | 'footwear'
  | 'handbags'
  | 'jewelry'
  | 'scarves'
  | 'belts'
  | 'hats'
  | 'gloves';

export type OrderStatus =
  | 'open'
  | 'closed'
  | 'cancelled'
  | 'archived';

export type FinancialStatus =
  | 'pending'
  | 'authorized'
  | 'partially_paid'
  | 'paid'
  | 'partially_refunded'
  | 'refunded'
  | 'voided';

export type FulfillmentStatus =
  | 'fulfilled'
  | 'null'
  | 'partial'
  | 'restocked';

export type StyleOccasion =
  | 'casual'
  | 'business'
  | 'formal'
  | 'party'
  | 'wedding'
  | 'vacation'
  | 'sport'
  | 'date-night';

export type Season =
  | 'spring'
  | 'summer'
  | 'fall'
  | 'winter'
  | 'year-round';

// API Response Types
export interface ProductsResponse {
  products: Product[];
  total: number;
  page: number;
  pageSize: number;
}

export interface CollectionsResponse {
  collections: Collection[];
  total: number;
}

export interface CartResponse {
  cart: Cart;
}

export interface CheckoutResponse {
  checkout: {
    id: string;
    webUrl: string;
  };
}

// Search and Filter Types
export interface ProductFilters {
  category?: ProductCategory[];
  priceRange?: {
    min: number;
    max: number;
  };
  colors?: string[];
  sizes?: string[];
  materials?: string[];
  tags?: string[];
  available?: boolean;
  onSale?: boolean;
}

export interface SortOptions {
  field: 'price' | 'title' | 'createdAt' | 'bestSelling' | 'featured';
  direction: 'asc' | 'desc';
}

// Component Props Types
export interface ProductCardProps {
  product: Product;
  onQuickView: (id: string) => void;
  onAddToCart: (variantId: string, quantity: number) => void;
  onAddToWishlist: (productId: string) => void;
  compact?: boolean;
  showQuickAdd?: boolean;
}

export interface CollectionCardProps {
  collection: Collection;
  onViewCollection: (id: string) => void;
}

export interface CartItemProps {
  item: CartItem;
  onQuantityChange: (id: string, quantity: number) => void;
  onRemove: (id: string) => void;
}

export interface WishlistItemProps {
  item: WishlistItem;
  product: Product;
  onAddToCart: (variantId: string) => void;
  onRemove: (id: string) => void;
}

// Form Types
export interface CheckoutForm {
  email: string;
  shippingAddress: CustomerAddress;
  billingAddress: CustomerAddress;
  paymentMethod: {
    type: 'credit_card' | 'paypal' | 'apple_pay' | 'google_pay';
    token?: string;
  };
  notes?: string;
}

export interface CustomerForm {
  firstName: string;
  lastName: string;
  email: string;
  phone?: string;
  acceptsMarketing: boolean;
}

export interface ReviewForm {
  rating: number;
  title: string;
  content: string;
  name: string;
  email: string;
}
