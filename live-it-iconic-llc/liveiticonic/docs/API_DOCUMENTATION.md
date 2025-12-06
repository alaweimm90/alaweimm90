# Live It Iconic API Documentation

**Version:** 1.0.0
**Last Updated:** November 2024
**Base URL:** `https://api.liveiconic.com`
**Development URL:** `http://localhost:5173`

## Table of Contents

1. [Authentication](#authentication)
2. [Products](#products)
3. [Orders](#orders)
4. [Inventory](#inventory)
5. [Payments](#payments)
6. [Email Services](#email-services)
7. [Recommendations](#recommendations)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [Security](#security)

---

## Overview

The Live It Iconic API is a RESTful API that provides comprehensive e-commerce functionality including user authentication, product management, order processing, inventory tracking, and payment integration.

### Key Features

- **JWT-based Authentication:** Secure token-based authentication with access and refresh tokens
- **Type-Safe Endpoints:** All endpoints are fully documented with request/response schemas
- **Rate Limiting:** Protection against abuse with configurable rate limits
- **Error Handling:** Consistent error response format across all endpoints
- **Versioning:** API follows semantic versioning

### Supported Media Types

- **Request:** `application/json`
- **Response:** `application/json`

---

## Authentication

### Authentication Scheme

The API uses JWT (JSON Web Token) authentication. Include the token in the `Authorization` header using the Bearer scheme:

```
Authorization: Bearer <access_token>
```

### Token Expiration

- **Access Token:** 15 minutes
- **Refresh Token:** 7 days

### Obtaining Tokens

Tokens are obtained through the signup or signin endpoints and are automatically managed by the API client.

---

## Endpoints

### POST /api/auth/signup

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "name": "John Doe"
}
```

**Response (200 OK):**
```json
{
  "user": {
    "id": "user_1234567890_abc123",
    "email": "user@example.com",
    "name": "John Doe",
    "createdAt": "2024-01-15T10:30:00.000Z"
  },
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Error Responses:**

- **400 Bad Request:** Validation error
  ```json
  {
    "error": "Validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
  ```

- **429 Too Many Requests:** Rate limit exceeded
  ```json
  {
    "error": "Too many requests",
    "message": "Please try again later."
  }
  ```

**Password Requirements:**
- Minimum 8 characters
- Must contain uppercase letter
- Must contain lowercase letter
- Must contain number
- Must contain special character

**Example Using API Client:**
```typescript
const response = await apiClient.signup({
  email: 'newuser@example.com',
  password: 'SecurePass123!',
  name: 'John Doe'
});

const { user, accessToken } = response;
```

---

### POST /api/auth/signin

Authenticate a user with email and password.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response (200 OK):**
```json
{
  "user": {
    "id": "user_1234567890_abc123",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Error Responses:**

- **401 Unauthorized:** Invalid credentials
  ```json
  {
    "error": "Invalid credentials",
    "message": "Email or password is incorrect"
  }
  ```

**Example Using API Client:**
```typescript
const response = await apiClient.signin({
  email: 'user@example.com',
  password: 'SecurePass123!'
});

apiClient.setToken(response.accessToken, response.refreshToken);
```

---

### GET /api/auth/me

Get the current authenticated user's profile.

**Authentication Required:** Yes (Bearer Token)

**Response (200 OK):**
```json
{
  "user": {
    "id": "user_1234567890_abc123",
    "email": "user@example.com",
    "name": "John Doe",
    "createdAt": "2024-01-15T10:30:00.000Z"
  }
}
```

**Example Using API Client:**
```typescript
const user = await apiClient.getCurrentUser();
console.log(user.email); // user@example.com
```

---

### GET /api/products

Get all products with optional filtering.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| category | string | No | Product category: `tees`, `hoodies`, `caps`, `accessories` |
| minPrice | number | No | Minimum price in USD |
| maxPrice | number | No | Maximum price in USD |
| search | string | No | Search term for product name/description |
| limit | number | No | Number of results (default: 20, max: 100) |
| offset | number | No | Number of results to skip (for pagination) |

**Response (200 OK):**
```json
[
  {
    "id": "prod_1234567890",
    "name": "Premium Motorsport T-Shirt",
    "description": "High-quality cotton t-shirt with motorsport design",
    "price": 49.99,
    "category": "tees",
    "image": "https://cdn.liveiconic.com/products/prod_1.jpg",
    "inStock": true,
    "stock": 150
  }
]
```

**Example Using API Client:**
```typescript
// Get all products
const products = await apiClient.getProducts();

// Get tees under $60
const tees = await apiClient.getProducts({
  category: 'tees',
  maxPrice: 60,
  limit: 10
});
```

---

### GET /api/products/{id}

Get details for a specific product.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Product ID |

**Response (200 OK):**
```json
{
  "id": "prod_1234567890",
  "name": "Premium Motorsport T-Shirt",
  "description": "High-quality cotton t-shirt with motorsport design",
  "price": 49.99,
  "category": "tees",
  "image": "https://cdn.liveiconic.com/products/prod_1.jpg",
  "inStock": true,
  "stock": 150
}
```

**Example Using API Client:**
```typescript
const product = await apiClient.getProduct('prod_1234567890');
console.log(product.name);
```

---

### POST /api/orders/create

Create a new order.

**Authentication Required:** Yes (Bearer Token)

**Request Body:**
```json
{
  "customerId": "user_1234567890_abc123",
  "items": [
    {
      "productId": "prod_1234567890",
      "quantity": 2,
      "price": 49.99
    },
    {
      "productId": "prod_2345678901",
      "quantity": 1,
      "price": 79.99
    }
  ],
  "subtotal": 179.97,
  "shipping": 10.00,
  "tax": 15.20,
  "total": 205.17,
  "shippingAddress": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zip": "10001",
    "country": "US"
  }
}
```

**Response (200 OK):**
```json
{
  "id": "ORD-1234567890",
  "orderNumber": "LII-1234567890",
  "userId": "user_1234567890_abc123",
  "items": [
    {
      "productId": "prod_1234567890",
      "quantity": 2,
      "price": 49.99,
      "subtotal": 99.98
    }
  ],
  "subtotal": 179.97,
  "shipping": 10.00,
  "tax": 15.20,
  "total": 205.17,
  "status": "pending",
  "paymentStatus": "pending",
  "createdAt": "2024-01-15T10:30:00.000Z"
}
```

**Example Using API Client:**
```typescript
const order = await apiClient.createOrder({
  customerId: user.id,
  items: [
    { productId: 'prod_1', quantity: 2, price: 49.99 }
  ],
  total: 99.98
});

console.log(`Order created: ${order.id}`);
```

---

### GET /api/orders/{id}

Get order details.

**Authentication Required:** Yes (Bearer Token)

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Order ID |

**Response (200 OK):**
```json
{
  "id": "ORD-1234567890",
  "orderNumber": "LII-1234567890",
  "userId": "user_1234567890_abc123",
  "items": [...],
  "total": 205.17,
  "status": "pending",
  "paymentStatus": "pending",
  "createdAt": "2024-01-15T10:30:00.000Z"
}
```

**Example Using API Client:**
```typescript
const order = await apiClient.getOrder('ORD-1234567890');
console.log(`Order Status: ${order.status}`);
```

---

### PATCH /api/orders/{id}

Update order status.

**Authentication Required:** Yes (Bearer Token)

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Order ID |

**Request Body:**
```json
{
  "status": "shipped",
  "trackingNumber": "TRACK123456789"
}
```

**Response (200 OK):**
```json
{
  "id": "ORD-1234567890",
  "status": "shipped",
  "trackingNumber": "TRACK123456789",
  "updatedAt": "2024-01-15T11:00:00.000Z"
}
```

**Status Values:**
- `pending` - Order awaiting payment
- `processing` - Order being prepared
- `shipped` - Order has been shipped
- `delivered` - Order delivered to customer
- `cancelled` - Order has been cancelled

---

### GET /api/inventory/check

Check product inventory.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| productId | string | Yes | Product ID to check |

**Response (200 OK):**
```json
{
  "productId": "prod_1234567890",
  "available": 150,
  "reserved": 10,
  "total": 160
}
```

**Example Using API Client:**
```typescript
const inventory = await apiClient.checkInventory('prod_1234567890');
console.log(`Available: ${inventory.available}`);
```

---

### POST /api/inventory/reserve

Reserve inventory for an order.

**Authentication Required:** Yes (Bearer Token)

**Request Body:**
```json
{
  "items": [
    {
      "productId": "prod_1234567890",
      "quantity": 2
    },
    {
      "productId": "prod_2345678901",
      "quantity": 1
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "reserved": true,
  "reservationId": "RES-1234567890",
  "expiresAt": "2024-01-15T11:00:00.000Z"
}
```

**Example Using API Client:**
```typescript
const reserved = await apiClient.reserveInventory([
  { productId: 'prod_1', quantity: 2 }
]);

if (reserved) {
  console.log('Inventory reserved');
}
```

---

### POST /api/payments/create-intent

Create a payment intent for Stripe.

**Authentication Required:** Yes (Bearer Token)

**Request Body:**
```json
{
  "orderId": "ORD-1234567890",
  "amount": 205.17
}
```

**Response (200 OK):**
```json
{
  "id": "pi_1234567890",
  "clientSecret": "pi_1234567890_secret_xxxxxxxxxxxx",
  "amount": 20517,
  "currency": "usd",
  "status": "requires_payment_method"
}
```

**Example Using API Client:**
```typescript
const intent = await apiClient.createPaymentIntent(
  'ORD-1234567890',
  205.17
);

// Use clientSecret with Stripe.js
```

---

### POST /api/payments/confirm-payment

Confirm a payment.

**Authentication Required:** Yes (Bearer Token)

**Request Body:**
```json
{
  "paymentIntentId": "pi_1234567890",
  "paymentMethodId": "pm_1234567890"
}
```

**Response (200 OK):**
```json
{
  "id": "pi_1234567890",
  "status": "succeeded",
  "amount": 20517,
  "currency": "usd"
}
```

---

## Error Handling

All errors follow a consistent response format:

```json
{
  "error": "Error code or type",
  "message": "Human-readable error message",
  "details": []
}
```

### Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | User lacks required permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Request conflicts with existing data |
| `RATE_LIMITED` | 429 | Too many requests |
| `SERVER_ERROR` | 500 | Internal server error |

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": "Validation failed",
  "details": [
    {
      "field": "email",
      "message": "Invalid email format"
    }
  ]
}
```

**401 Unauthorized:**
```json
{
  "error": "Unauthorized",
  "message": "Missing or invalid authentication token"
}
```

**404 Not Found:**
```json
{
  "error": "Not found",
  "message": "Product not found"
}
```

**429 Too Many Requests:**
```json
{
  "error": "Too many requests",
  "message": "Rate limit exceeded. Please try again later."
}
```

---

## Rate Limiting

The API implements rate limiting to prevent abuse. Limits vary by endpoint:

### Rate Limit Headers

All responses include rate limit information in headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705335000
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/auth/signup` | 5 | 15 minutes |
| `/api/auth/signin` | 5 | 15 minutes |
| `/api/products` | 100 | 1 hour |
| `/api/orders/*` | 50 | 1 hour |
| Other endpoints | 100 | 1 hour |

### Handling Rate Limits

When rate limited, the API returns a 429 status code. Retry after the time specified in the `Retry-After` header:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

---

## Security

### Best Practices

1. **Always use HTTPS** in production
2. **Never expose tokens** in client-side code or version control
3. **Rotate tokens** regularly
4. **Use strong passwords** (8+ characters, mixed case, numbers, special chars)
5. **Implement CORS** appropriately on the frontend

### CORS

The API supports Cross-Origin Resource Sharing (CORS) for requests from authorized domains.

**Allowed Origins:**
- `https://liveiconic.com`
- `https://www.liveiconic.com`
- `http://localhost:3000` (development)
- `http://localhost:5173` (development)

### HTTPS

- All production endpoints use HTTPS
- TLS 1.2 or higher required
- Certificates from trusted Certificate Authorities

---

## Versioning Strategy

The API uses semantic versioning:

- **Major (1.x.x):** Breaking changes
- **Minor (x.1.x):** New features (backward compatible)
- **Patch (x.x.1):** Bug fixes

Current version: **1.0.0**

### Deprecation

Deprecated endpoints will:
1. Be marked as deprecated in documentation
2. Return `Deprecation: true` header
3. Continue functioning for at least 6 months
4. Be removed with major version bump

---

## Examples

### Complete Signup and Order Flow

```typescript
import { apiClient } from '@/lib/apiClient';

// 1. Sign up a user
const signupResponse = await apiClient.signup({
  email: 'customer@example.com',
  password: 'SecurePass123!',
  name: 'Jane Doe'
});

const { user, accessToken } = signupResponse;
apiClient.setToken(accessToken);

// 2. Get products
const products = await apiClient.getProducts({
  category: 'tees',
  maxPrice: 60
});

// 3. Reserve inventory
await apiClient.reserveInventory([
  { productId: products[0].id, quantity: 2 }
]);

// 4. Create order
const order = await apiClient.createOrder({
  customerId: user.id,
  items: [
    {
      productId: products[0].id,
      quantity: 2,
      price: products[0].price
    }
  ],
  total: products[0].price * 2
});

// 5. Create payment intent
const paymentIntent = await apiClient.createPaymentIntent(
  order.id,
  order.total
);

// 6. Confirm payment (with Stripe payment method)
const payment = await apiClient.confirmPayment(
  paymentIntent.id,
  'pm_stripe_token'
);

console.log(`Order ${order.id} payment confirmed!`);
```

---

## SDK/Client Libraries

### JavaScript/TypeScript

```typescript
npm install @liveiconic/api-client
```

```typescript
import { apiClient } from '@/lib/apiClient';

// Automatically handles authentication and tokens
const user = await apiClient.getCurrentUser();
```

---

## Support

For API support, issues, or feature requests:

- Email: `api-support@liveiconic.com`
- Documentation: `https://docs.liveiconic.com`
- Status Page: `https://status.liveiconic.com`

---

## Changelog

### v1.0.0 (November 2024)

- Initial release
- Authentication endpoints (signup, signin, profile)
- Product browsing and search
- Order management
- Inventory tracking
- Payment processing with Stripe integration
- Rate limiting and security features

---

## License

The API is provided under a proprietary license. See LICENSE file for details.
