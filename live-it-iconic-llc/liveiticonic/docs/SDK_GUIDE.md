# Live It Iconic API Client SDK Guide

Complete guide to using the Live It Iconic API Client library for type-safe API requests.

## Table of Contents

1. [Installation](#installation)
2. [Basic Setup](#basic-setup)
3. [Authentication](#authentication)
4. [Making Requests](#making-requests)
5. [Error Handling](#error-handling)
6. [Advanced Features](#advanced-features)
7. [Examples](#examples)
8. [Best Practices](#best-practices)

---

## Installation

The API Client is built into the Live It Iconic application. You can import it from:

```typescript
import { apiClient } from '@/lib/apiClient';
```

### Dependencies

The client requires:
- TypeScript 5.8+
- Node 18+ (for non-browser usage)
- Fetch API support

---

## Basic Setup

### Initialization

The API client is automatically initialized with the correct base URL based on your environment:

```typescript
import { apiClient } from '@/lib/apiClient';

// The client is ready to use immediately
const products = await apiClient.getProducts();
```

### Configuration

By default, the client uses the VITE_API_BASE_URL environment variable:

```env
VITE_API_BASE_URL=http://localhost:5173
```

For production:

```env
VITE_API_BASE_URL=https://api.liveiconic.com
```

---

## Authentication

### Signup

Register a new user account:

```typescript
const response = await apiClient.signup({
  email: 'user@example.com',
  password: 'SecurePass123!',
  name: 'John Doe'
});

const { user, accessToken } = response;
console.log(`Welcome ${user.name}!`);

// Token is automatically set by the client
```

### Signin

Authenticate an existing user:

```typescript
const response = await apiClient.signin({
  email: 'user@example.com',
  password: 'SecurePass123!'
});

const { user, accessToken, refreshToken } = response;

// Tokens are automatically managed
```

### Manual Token Management

Manually set tokens (e.g., after receiving from backend):

```typescript
apiClient.setToken(accessToken, refreshToken);

// Or with just access token
apiClient.setToken(accessToken);
```

### Clear Tokens (Logout)

```typescript
apiClient.clearToken();

// Or after signin/signup, tokens are cleared before new auth
```

### Auto Token Refresh

Tokens are automatically refreshed when expired:

```typescript
// This will automatically refresh if token is expired
const user = await apiClient.getCurrentUser();
```

---

## Making Requests

### Authentication Endpoints

#### Get Current User

```typescript
const user = await apiClient.getCurrentUser();
console.log(user.email);
```

**Returns:** `User` object

**Throws:** Error if not authenticated

---

### Product Endpoints

#### Get All Products

```typescript
// Get all products
const products = await apiClient.getProducts();

// With filters
const tees = await apiClient.getProducts({
  category: 'tees',
  maxPrice: 60,
  limit: 20,
  offset: 0
});

// Search
const results = await apiClient.getProducts({
  search: 'motorsport',
  limit: 10
});
```

**Parameters:**
```typescript
interface ProductFilters {
  category?: 'tees' | 'hoodies' | 'caps' | 'accessories';
  minPrice?: number;
  maxPrice?: number;
  search?: string;
  limit?: number;
  offset?: number;
}
```

**Returns:** `Product[]`

#### Get Single Product

```typescript
const product = await apiClient.getProduct('prod_123456789');

console.log(`Product: ${product.name}`);
console.log(`Price: $${product.price}`);
console.log(`In Stock: ${product.inStock}`);
```

**Returns:** `Product` object

---

### Order Endpoints

#### Create Order

```typescript
const order = await apiClient.createOrder({
  customerId: user.id,
  items: [
    {
      productId: 'prod_1',
      quantity: 2,
      price: 49.99
    },
    {
      productId: 'prod_2',
      quantity: 1,
      price: 79.99
    }
  ],
  subtotal: 179.97,
  shipping: 10.00,
  tax: 15.20,
  total: 205.17,
  shippingAddress: {
    street: '123 Main St',
    city: 'New York',
    state: 'NY',
    zip: '10001',
    country: 'US'
  }
});

console.log(`Order created: ${order.id}`);
console.log(`Order Number: ${order.orderNumber}`);
```

**Parameters:** `CreateOrderDTO`

**Returns:** `Order` object

#### Get Order

```typescript
const order = await apiClient.getOrder('ORD-1234567890');

console.log(`Status: ${order.status}`);
console.log(`Total: $${order.total}`);
```

**Returns:** `Order` object

#### Get User Orders

```typescript
const orders = await apiClient.getUserOrders();

orders.forEach(order => {
  console.log(`Order ${order.id}: ${order.status}`);
});
```

**Returns:** `Order[]`

#### Update Order Status

```typescript
const updated = await apiClient.updateOrderStatus(
  'ORD-1234567890',
  'shipped',
  'TRACK123456789'
);

console.log(`Tracking: ${updated.trackingNumber}`);
```

**Returns:** `Order` object

---

### Inventory Endpoints

#### Check Inventory

```typescript
const inventory = await apiClient.checkInventory('prod_123456789');

console.log(`Available: ${inventory.available}`);
console.log(`Reserved: ${inventory.reserved}`);
```

**Returns:** Object with `available` and `reserved` properties

#### Reserve Inventory

```typescript
const reserved = await apiClient.reserveInventory([
  { productId: 'prod_1', quantity: 2 },
  { productId: 'prod_2', quantity: 1 }
]);

if (reserved) {
  console.log('Inventory successfully reserved');
}
```

**Returns:** `boolean`

---

### Payment Endpoints

#### Create Payment Intent

```typescript
const paymentIntent = await apiClient.createPaymentIntent(
  'ORD-1234567890',
  205.17
);

console.log(`Intent ID: ${paymentIntent.id}`);
console.log(`Client Secret: ${paymentIntent.clientSecret}`);

// Use with Stripe.js
const stripe = Stripe('pk_test_...');
const result = await stripe.confirmCardPayment(
  paymentIntent.clientSecret
);
```

#### Confirm Payment

```typescript
const payment = await apiClient.confirmPayment(
  'pi_1234567890',
  'pm_stripe_token'
);

console.log(`Payment Status: ${payment.status}`);
```

---

## Error Handling

### Catching Errors

All methods throw errors when something goes wrong:

```typescript
try {
  const user = await apiClient.signin({
    email: 'user@example.com',
    password: 'wrong_password'
  });
} catch (error) {
  console.error('Signin failed:', error.message);
}
```

### Error Types

The client throws standard JavaScript `Error` objects with descriptive messages:

```typescript
try {
  await apiClient.getProduct('invalid_id');
} catch (error) {
  if (error instanceof Error) {
    console.error(error.message); // API returned error message
  }
}
```

### Handling API Errors

API errors include details from the server:

```typescript
try {
  await apiClient.createOrder({ /* ... */ });
} catch (error) {
  if (error instanceof Error) {
    // Message contains API error details
    if (error.message.includes('insufficient inventory')) {
      // Handle inventory error
    } else if (error.message.includes('invalid payment')) {
      // Handle payment error
    }
  }
}
```

### Automatic Retry

The client automatically retries token refresh on 401 errors:

```typescript
// If token is expired, it will automatically refresh
// No need for manual retry logic
const user = await apiClient.getCurrentUser();
```

---

## Advanced Features

### Request Timeout

Default timeout is 30 seconds (inherited from fetch):

```typescript
// Use AbortController for custom timeout
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 5000);

try {
  // Custom implementation would go here
  // Current client doesn't expose this directly
} finally {
  clearTimeout(timeoutId);
}
```

### Custom Headers

The client automatically includes:
- `Content-Type: application/json`
- `Authorization: Bearer <token>` (when authenticated)

Additional headers can't be added without modifying the client.

### Token Persistence

Tokens are automatically saved to localStorage (browser only):

```typescript
// After signup/signin
apiClient.setToken(accessToken);

// Token is saved and will be loaded on page reload
const user = await apiClient.getCurrentUser();
```

### Custom Base URL

For custom implementations, you can extend the client:

```typescript
// Not recommended - use environment variables instead
// const customClient = new ApiClient('https://custom-api.com');
```

---

## Examples

### Complete Login Flow

```typescript
// User signs in
const response = await apiClient.signin({
  email: 'user@example.com',
  password: 'Password123!'
});

// Get user profile
const user = await apiClient.getCurrentUser();
console.log(`Logged in as: ${user.name}`);

// Get user's orders
const orders = await apiClient.getUserOrders();
console.log(`You have ${orders.length} orders`);
```

### Complete Purchase Flow

```typescript
// 1. Get products
const products = await apiClient.getProducts({
  category: 'tees',
  maxPrice: 60
});

const selectedProduct = products[0];

// 2. Check inventory
const inventory = await apiClient.checkInventory(selectedProduct.id);
if (inventory.available < 1) {
  throw new Error('Product out of stock');
}

// 3. Reserve inventory
const reserved = await apiClient.reserveInventory([
  { productId: selectedProduct.id, quantity: 1 }
]);

if (!reserved) {
  throw new Error('Failed to reserve inventory');
}

// 4. Create order
const user = await apiClient.getCurrentUser();
const order = await apiClient.createOrder({
  customerId: user.id,
  items: [
    {
      productId: selectedProduct.id,
      quantity: 1,
      price: selectedProduct.price
    }
  ],
  total: selectedProduct.price,
  shippingAddress: {
    /* ... */
  }
});

console.log(`Order created: ${order.id}`);

// 5. Create payment intent
const paymentIntent = await apiClient.createPaymentIntent(
  order.id,
  order.total
);

// 6. Process payment (with Stripe)
// ... Stripe integration ...

// 7. Confirm payment
const payment = await apiClient.confirmPayment(
  paymentIntent.id,
  stripePaymentMethodId
);

if (payment.status === 'succeeded') {
  console.log('Payment successful!');

  // 8. Update order status
  await apiClient.updateOrderStatus(order.id, 'processing');
}
```

### Product Search with Filters

```typescript
async function searchProducts(searchTerm, filters) {
  try {
    const results = await apiClient.getProducts({
      search: searchTerm,
      category: filters.category,
      minPrice: filters.minPrice,
      maxPrice: filters.maxPrice,
      limit: 20
    });

    return results;
  } catch (error) {
    console.error('Search failed:', error);
    return [];
  }
}

// Usage
const hoodies = await searchProducts('winter', {
  category: 'hoodies',
  minPrice: 50,
  maxPrice: 150
});
```

### Handling User Authentication State

```typescript
// Check if user is authenticated
async function isAuthenticated() {
  try {
    await apiClient.getCurrentUser();
    return true;
  } catch (error) {
    return false;
  }
}

// Logout
function logout() {
  apiClient.clearToken();
  // Redirect to login page
}

// Auto-login from stored token
async function initializeAuth() {
  const isAuth = await isAuthenticated();
  if (isAuth) {
    const user = await apiClient.getCurrentUser();
    return user;
  }
  return null;
}
```

---

## Best Practices

### 1. Always Handle Errors

```typescript
try {
  const user = await apiClient.getCurrentUser();
} catch (error) {
  console.error('Failed to get user:', error);
  // Show error to user or redirect to login
}
```

### 2. Check Authentication Before Protected Operations

```typescript
async function viewUserOrders() {
  try {
    const orders = await apiClient.getUserOrders();
    return orders;
  } catch (error) {
    // User is not authenticated
    redirectToLogin();
  }
}
```

### 3. Validate Input Before Sending

```typescript
// Validate before calling API
if (!email.includes('@')) {
  throw new Error('Invalid email');
}

if (password.length < 8) {
  throw new Error('Password too short');
}

const response = await apiClient.signup({
  email,
  password,
  name
});
```

### 4. Use Filters to Reduce Data Transfer

```typescript
// Good: Only get relevant products
const products = await apiClient.getProducts({
  category: 'tees',
  maxPrice: 100,
  limit: 20
});

// Bad: Get all products
const allProducts = await apiClient.getProducts();
const filtered = allProducts.filter(/* ... */);
```

### 5. Handle Network Errors

```typescript
try {
  const products = await apiClient.getProducts();
} catch (error) {
  if (error instanceof TypeError && error.message.includes('fetch')) {
    console.error('Network error - check your connection');
  } else {
    console.error('API error:', error);
  }
}
```

### 6. Don't Expose Tokens

```typescript
// Bad
console.log(accessToken);
localStorage.setItem('token', token); // Already handled by client

// Good
apiClient.setToken(token); // Client handles storage
```

### 7. Implement Retry Logic for Critical Operations

```typescript
async function createOrderWithRetry(
  orderData,
  maxRetries = 3
) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await apiClient.createOrder(orderData);
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

---

## Troubleshooting

### 401 Unauthorized

**Problem:** Getting 401 errors on authenticated endpoints

**Solution:**
1. Ensure you're calling signup/signin first
2. Check that tokens are being set: `apiClient.setToken(token)`
3. Verify token isn't expired (should be auto-refreshed)

### Network Errors

**Problem:** `TypeError: fetch failed`

**Solution:**
1. Check API base URL is correct
2. Ensure API server is running
3. Check CORS configuration
4. Look at browser console for details

### Type Errors

**Problem:** TypeScript errors with API types

**Solution:**
1. Make sure types are imported: `import { Product, Order } from '@/lib/apiClient'`
2. Check you're using correct interfaces
3. Enable strict mode in TypeScript

---

## API Reference Quick Links

- [API Documentation](./API_DOCUMENTATION.md)
- [OpenAPI Spec](../src/api/openapi.ts)
- [Postman Collection](../postman/Live-It-Iconic-API.postman_collection.json)

---

## Support

For issues or questions about the SDK:

- Check the [API Documentation](./API_DOCUMENTATION.md)
- Review [code examples](../tests/api/auth.test.ts)
- Contact: api-support@liveiconic.com
