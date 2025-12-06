/**
 * Mock data for testing
 */

export const mockProduct = {
  id: 'prod-1',
  name: 'Classic T-Shirt',
  category: 'Performance' as const,
  price: 65,
  description: 'High-quality performance t-shirt for active individuals',
  image: 'https://example.com/tshirt.jpg',
};

export const mockProduct2 = {
  id: 'prod-2',
  name: 'Premium Hoodie',
  category: 'Lifestyle' as const,
  price: 125,
  description: 'Comfortable hoodie for everyday wear',
  image: 'https://example.com/hoodie.jpg',
};

export const mockProduct3 = {
  id: 'prod-3',
  name: 'Sports Cap',
  category: 'Training' as const,
  price: 45,
  description: 'Breathable cap for outdoor activities',
  image: 'https://example.com/cap.jpg',
};

export const mockProducts = [mockProduct, mockProduct2, mockProduct3];

export const mockCartItem = {
  id: 'prod-1',
  name: 'Classic T-Shirt',
  price: 65,
  quantity: 1,
  image: 'https://example.com/tshirt.jpg',
};

export const mockCartItem2 = {
  id: 'prod-2',
  name: 'Premium Hoodie',
  price: 125,
  quantity: 2,
  image: 'https://example.com/hoodie.jpg',
};

export const mockCartItems = [mockCartItem, mockCartItem2];

export const mockUser = {
  id: 'user-1',
  email: 'test@example.com',
  name: 'Test User',
  createdAt: new Date('2024-01-01'),
};

export const mockAddress = {
  id: 'addr-1',
  street: '123 Main St',
  city: 'San Francisco',
  state: 'CA',
  zipCode: '94105',
  country: 'US',
};

export const mockOrder = {
  id: 'order-1',
  customerId: 'user-1',
  items: mockCartItems,
  total: 315,
  status: 'pending' as const,
  createdAt: new Date('2024-01-15'),
  shippingAddress: mockAddress,
};
