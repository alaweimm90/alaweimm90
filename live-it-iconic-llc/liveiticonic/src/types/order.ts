export interface OrderItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
  image: string;
  variant?: string;
}

export interface ShippingAddress {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  state: string;
  zipCode: string;
  country: string;
}

export type OrderStatus =
  | 'pending'
  | 'confirmed'
  | 'processing'
  | 'shipped'
  | 'delivered'
  | 'cancelled';
export type PaymentStatus = 'pending' | 'paid' | 'failed' | 'refunded';

export interface Order {
  id: string;
  orderNumber: string;
  customerId?: string;
  items: OrderItem[];
  subtotal: number;
  shipping: number;
  tax: number;
  total: number;
  shippingAddress: ShippingAddress;
  billingAddress?: ShippingAddress;
  status: OrderStatus;
  paymentStatus: PaymentStatus;
  trackingNumber?: string;
  notes?: string;
  createdAt: Date;
  updatedAt?: Date;
  completedAt?: Date;
  stripePaymentIntentId?: string;
}

export interface CheckoutFormData extends ShippingAddress {
  cardNumber: string;
  expiryDate: string;
  cvv: string;
  billingAddressSame: boolean;
  billingAddress?: Omit<ShippingAddress, 'firstName' | 'lastName' | 'email' | 'phone'>;
}
