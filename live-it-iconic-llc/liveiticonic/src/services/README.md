# Services Documentation

Comprehensive documentation for the LiveItIconic application services layer. Services provide abstracted access to backend APIs and external integrations.

## Overview

Services act as data access and business logic layers that abstract API interactions from React components. They provide a consistent interface for making API calls, handling errors, and managing data transformation.

### Service Architecture

```
Services
├── authService - User authentication and session management
├── productService - Product catalog and details
├── cartService - Shopping cart operations
├── orderService - Order creation and management
├── paymentService - Payment processing integration
├── emailService - Email notifications
├── inventoryService - Stock management
├── adminService - Admin operations
└── recommendationService - Product recommendations
```

## Core Services

### authService

**Purpose**: Handle user authentication, registration, and session management

**Location**: `/src/services/authService.ts`

**Key Methods**:
- `signUp(params)` - Register new user with email and password
- `signIn(params)` - Authenticate existing user
- `signOut()` - End user session
- `getCurrentUser()` - Retrieve authenticated user info

**Features**:
- Email/password authentication
- Token-based session management
- localStorage persistence
- Error handling and validation

**Example Usage**:
```typescript
import { authService } from '@/services/authService';

// Sign up
const { user, token } = await authService.signUp({
  email: 'user@example.com',
  password: 'SecurePass123',
  name: 'John Doe'
});

// Sign in
const auth = await authService.signIn({
  email: 'user@example.com',
  password: 'SecurePass123'
});

// Get current user
const user = await authService.getCurrentUser();

// Sign out
await authService.signOut();
```

**Error Handling**:
```typescript
try {
  await authService.signIn({ email, password });
} catch (error) {
  if (error.message.includes('Invalid')) {
    console.error('Wrong email or password');
  } else if (error.message.includes('not found')) {
    console.error('User account does not exist');
  }
}
```

**API Endpoints Used**:
- `POST /api/auth/signup` - Create new account
- `POST /api/auth/signin` - Authenticate user
- `POST /api/auth/signout` - End session
- `GET /api/auth/me` - Get current user

---

### productService

**Purpose**: Manage product catalog, retrieval, and filtering

**Location**: `/src/services/productService.ts`

**Key Methods**:
- `getAllProducts()` - Fetch all products
- `getProduct(id)` - Get single product details
- `searchProducts(query)` - Search products by name/description
- `getProductsByCategory(category)` - Filter by category
- `getRelatedProducts(productId)` - Get recommended products

**Features**:
- Product catalog access
- Search and filtering
- Related product recommendations
- Inventory status
- Pricing and discounts

**Example Usage**:
```typescript
import { productService } from '@/services/productService';

// Get all products
const products = await productService.getAllProducts();

// Get single product
const product = await productService.getProduct('prod-123');

// Search products
const results = await productService.searchProducts('navy tee');

// Filter by category
const tees = await productService.getProductsByCategory('Tees');

// Get related products
const related = await productService.getRelatedProducts('prod-123');
```

**API Endpoints Used**:
- `GET /api/products` - List all products
- `GET /api/products/:id` - Get product details
- `GET /api/products/search` - Search products
- `GET /api/products/category/:category` - Filter by category

---

### orderService

**Purpose**: Handle order creation, retrieval, and management

**Location**: `/src/services/orderService.ts`

**Key Methods**:
- `createOrder(params)` - Create new order from cart
- `getOrder(id)` - Retrieve order details
- `getOrderHistory(userId)` - Get user's orders
- `updateOrderStatus(id, status)` - Update order status
- `cancelOrder(id)` - Cancel pending order

**Features**:
- Order creation and validation
- Payment processing integration
- Inventory reservation
- Order tracking
- Status management

**Example Usage**:
```typescript
import { orderService } from '@/services/orderService';

// Create order
const orderId = await orderService.createOrder({
  shippingData: {
    firstName: 'John',
    lastName: 'Doe',
    email: 'john@example.com',
    address: '123 Main St',
    city: 'New York',
    state: 'NY',
    postalCode: '10001',
    country: 'US'
  },
  amount: 15999, // in cents
  paymentMethod: 'card',
  cardLast4: '4242'
});

// Get order details
const order = await orderService.getOrder(orderId);

// Get order history
const orders = await orderService.getOrderHistory(userId);

// Update status
await orderService.updateOrderStatus(orderId, 'shipped');
```

**API Endpoints Used**:
- `POST /api/orders/create` - Create order
- `GET /api/orders/:id` - Get order details
- `GET /api/orders/customer/:customerId` - Get user orders

---

### paymentService

**Purpose**: Handle payment processing and payment method management

**Location**: `/src/services/paymentService.ts`

**Key Methods**:
- `createPaymentIntent(amount)` - Create Stripe payment intent
- `confirmPayment(intentId, paymentMethod)` - Confirm payment
- `validateCard(cardData)` - Validate card details
- `getPaymentMethods(userId)` - Retrieve saved payment methods

**Features**:
- Stripe payment integration
- Payment intent creation
- Card validation
- Secure payment processing
- PCI compliance

**Example Usage**:
```typescript
import { paymentService } from '@/services/paymentService';

// Create payment intent
const intent = await paymentService.createPaymentIntent(15999);

// Confirm payment
const result = await paymentService.confirmPayment(
  intent.id,
  'pm_stripe_card_token'
);

// Get saved cards
const methods = await paymentService.getPaymentMethods(userId);
```

**Security Notes**:
- Never send raw card details to backend
- Always use Stripe's secure token handling
- Cards stored in Stripe, not in database
- PCI compliance handled by Stripe

---

### inventoryService

**Purpose**: Manage product inventory and stock levels

**Location**: `/src/services/inventoryService.ts`

**Key Methods**:
- `checkInventory(productId, quantity)` - Check stock availability
- `reserveInventory(productId, quantity)` - Reserve stock for order
- `confirmInventory(reservationId)` - Confirm reservation
- `getLowStockItems()` - Get items below threshold

**Features**:
- Real-time inventory checking
- Stock reservation during checkout
- Automatic release on order cancellation
- Low stock alerts

**Example Usage**:
```typescript
import { inventoryService } from '@/services/inventoryService';

// Check if in stock
const available = await inventoryService.checkInventory('prod-123', 2);

// Reserve inventory during checkout
const reservation = await inventoryService.reserveInventory('prod-123', 2);

// Confirm after payment
await inventoryService.confirmInventory(reservation.id);
```

---

### emailService

**Purpose**: Handle transactional emails and notifications

**Location**: `/src/services/emailService.ts`

**Key Methods**:
- `sendOrderConfirmation(email, orderData)` - Send order confirmation
- `sendShippingNotification(email, trackingData)` - Send shipping update
- `sendAbandonedCartEmail(email, cartData)` - Send cart reminder
- `sendNewsletterEmail(recipients, content)` - Send newsletter

**Features**:
- Transactional email sending
- Email templating
- Batch sending
- Delivery tracking
- HTML and text versions

**Example Usage**:
```typescript
import { emailService } from '@/services/emailService';

// Send order confirmation
await emailService.sendOrderConfirmation('user@example.com', {
  orderNumber: 'ORD-123456',
  items: [...]
  total: 159.99
});

// Send shipping notification
await emailService.sendShippingNotification('user@example.com', {
  carrier: 'FedEx',
  trackingNumber: 'FDX123456789'
});
```

---

### recommendationService

**Purpose**: Generate personalized product recommendations

**Location**: `/src/services/recommendationService.ts`

**Key Methods**:
- `getRecommendations(userId)` - Get personalized recommendations
- `getFrequentlyBought(productId)` - Get often-bought-with products
- `getTrendingProducts()` - Get trending items
- `getNewArrivals()` - Get new products

**Features**:
- Collaborative filtering
- Content-based recommendations
- Trending analysis
- Purchase pattern analysis

**Example Usage**:
```typescript
import { recommendationService } from '@/services/recommendationService';

// Get personalized recommendations
const recommendations = await recommendationService.getRecommendations(userId);

// Get frequently bought together
const bundled = await recommendationService.getFrequentlyBought('prod-123');

// Get trending products
const trending = await recommendationService.getTrendingProducts();
```

---

### adminService

**Purpose**: Admin operations and management functions

**Location**: `/src/services/adminService.ts`

**Key Methods**:
- `getDashboardMetrics()` - Get key business metrics
- `getOrderMetrics()` - Get order statistics
- `getCustomerMetrics()` - Get customer analytics
- `getRevenueReport()` - Generate revenue reports

**Features**:
- Analytics and reporting
- Admin operations
- User management
- Content management
- System monitoring

**Example Usage**:
```typescript
import { adminService } from '@/services/adminService';

// Get dashboard metrics
const metrics = await adminService.getDashboardMetrics();

// Get revenue report
const report = await adminService.getRevenueReport({
  startDate: '2024-01-01',
  endDate: '2024-01-31'
});
```

---

## Service Patterns

### Error Handling Pattern

All services follow consistent error handling:

```typescript
try {
  const result = await service.method(params);
  return result;
} catch (error) {
  console.error('Operation failed:', error);
  throw error; // Re-throw for component handling
}
```

### Request/Response Pattern

All service methods follow this pattern:

```typescript
// Request: plain object with data
const result = await service.method({
  email: 'user@example.com',
  password: 'password123'
});

// Response: structured data object
// {
//   user: { id, email, name },
//   token: 'jwt_token_here'
// }
```

### Authentication Pattern

Services automatically include authentication token from localStorage:

```typescript
const token = localStorage.getItem('auth_token');
const response = await fetch('/api/endpoint', {
  headers: {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
});
```

## HTTP Status Codes

Services throw errors for non-2xx responses:

| Status | Meaning | Action |
|--------|---------|--------|
| 200-299 | Success | Return response data |
| 400 | Bad request | Check parameters |
| 401 | Unauthorized | Redirect to login |
| 403 | Forbidden | Check permissions |
| 404 | Not found | Handle gracefully |
| 500 | Server error | Retry or fallback |

## Best Practices

### 1. Always Use Services from Components

```typescript
// Good - use service
const products = await productService.getAllProducts();

// Bad - direct API call
const products = await fetch('/api/products').then(r => r.json());
```

### 2. Handle Errors Properly

```typescript
try {
  const result = await authService.signIn({ email, password });
  setUser(result.user);
} catch (error) {
  setError(error.message);
  // Don't re-throw unless necessary
}
```

### 3. Validate Input Before Calling Service

```typescript
if (!email || !password) {
  setError('Email and password required');
  return;
}

const result = await authService.signIn({ email, password });
```

### 4. Cache Service Results When Appropriate

```typescript
const [products, setProducts] = useState<Product[]>([]);
const [loading, setLoading] = useState(true);

useEffect(() => {
  if (products.length === 0) {
    productService.getAllProducts()
      .then(setProducts)
      .finally(() => setLoading(false));
  }
}, []);
```

### 5. Don't Expose Implementation Details

```typescript
// Service handles API details internally
export const authService = {
  async signIn(params) {
    // All API interaction hidden here
    // Components don't need to know about endpoints
  }
};

// Component just calls the service
const result = await authService.signIn({ email, password });
```

## Testing Services

### Unit Testing

```typescript
describe('authService', () => {
  it('should sign up user', async () => {
    const user = await authService.signUp({
      email: 'test@example.com',
      password: 'password123'
    });

    expect(user.email).toBe('test@example.com');
    expect(localStorage.getItem('auth_token')).toBeDefined();
  });
});
```

### Mocking Services

```typescript
// Mock service for testing
const mockAuthService = {
  signIn: jest.fn().mockResolvedValue({
    user: { id: '1', email: 'test@example.com' },
    token: 'mock_token'
  })
};
```

## Related Documentation

- [API Documentation](../docs/API.md)
- [Authentication Guide](../docs/ARCHITECTURE.md#authentication)
- [Error Handling](../docs/DEVELOPMENT.md#error-handling)
- [Testing Guide](../docs/DEVELOPMENT.md#testing)

---

For detailed method signatures and examples, see individual service files in this directory.
