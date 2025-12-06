# Live It Iconic - API Documentation

**Version:** 1.0.0
**Last Updated:** 2025-11-11
**Base URL:** `/api`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Products API](#products-api)
3. [Cart API](#cart-api)
4. [Orders API](#orders-api)
5. [Payments API](#payments-api)
6. [Email API](#email-api)
7. [Admin API](#admin-api)
8. [Inventory API](#inventory-api)
9. [Recommendations API](#recommendations-api)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)

---

## Authentication

### Authentication Methods

**Current:** Session-based (Supabase Auth)
**Future:** JWT tokens, API keys for third-party integrations

### Headers

```http
Authorization: Bearer <token>
Content-Type: application/json
```

### Auth Endpoints

#### Sign Up
```http
POST /api/auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123",
  "firstName": "John",
  "lastName": "Doe"
}
```

**Response:**
```json
{
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe"
  },
  "session": {
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "expires_in": 3600
  }
}
```

#### Sign In
```http
POST /api/auth/signin
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

#### Sign Out
```http
POST /api/auth/signout
Authorization: Bearer <token>
```

#### Get Current User
```http
GET /api/auth/me
Authorization: Bearer <token>
```

---

## Products API

### Get All Products

```http
GET /api/products?category=apparel&limit=20&offset=0
```

**Query Parameters:**
- `category` (optional): Filter by category (apparel, accessories, etc.)
- `limit` (optional): Number of products per page (default: 20)
- `offset` (optional): Pagination offset (default: 0)
- `search` (optional): Search query
- `sortBy` (optional): Sort field (price, name, createdAt)
- `order` (optional): Sort order (asc, desc)

**Response:**
```json
{
  "products": [
    {
      "id": "prod_123",
      "name": "Performance Hoodie",
      "description": "Premium athletic hoodie",
      "price": 89.99,
      "category": "apparel",
      "images": ["url1", "url2"],
      "stock": 50,
      "variants": [
        {
          "id": "var_123",
          "name": "Small",
          "stock": 10
        }
      ]
    }
  ],
  "total": 100,
  "limit": 20,
  "offset": 0
}
```

### Get Product by ID

```http
GET /api/products/:productId
```

**Response:**
```json
{
  "id": "prod_123",
  "name": "Performance Hoodie",
  "description": "Premium athletic hoodie designed for athletes",
  "price": 89.99,
  "compareAtPrice": 120.00,
  "category": "apparel",
  "images": ["url1", "url2", "url3"],
  "stock": 50,
  "variants": [...],
  "specifications": {},
  "reviews": [],
  "createdAt": "2025-01-01T00:00:00Z"
}
```

---

## Cart API

### Add to Cart

```http
POST /api/cart/add
Authorization: Bearer <token>
Content-Type: application/json

{
  "productId": "prod_123",
  "quantity": 2,
  "variantId": "var_123"
}
```

### Get Cart

```http
GET /api/cart
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "cart_123",
  "items": [
    {
      "id": "item_123",
      "productId": "prod_123",
      "name": "Performance Hoodie",
      "price": 89.99,
      "quantity": 2,
      "variant": "Small",
      "image": "url"
    }
  ],
  "subtotal": 179.98,
  "shipping": 10.00,
  "tax": 18.00,
  "total": 207.98
}
```

### Update Cart Item

```http
PUT /api/cart/items/:itemId
Content-Type: application/json

{
  "quantity": 3
}
```

### Remove from Cart

```http
DELETE /api/cart/items/:itemId
```

---

## Orders API

### Create Order

```http
POST /api/orders/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "items": [...],
  "shippingAddress": {
    "firstName": "John",
    "lastName": "Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "address": "123 Main St",
    "city": "San Francisco",
    "state": "CA",
    "zipCode": "94102",
    "country": "US"
  },
  "paymentMethodId": "pm_123"
}
```

**Response:**
```json
{
  "order": {
    "id": "ord_123",
    "orderNumber": "LII-1699999999",
    "status": "pending",
    "paymentStatus": "pending",
    "total": 207.98,
    "createdAt": "2025-11-11T00:00:00Z"
  }
}
```

### Get Order by ID

```http
GET /api/orders/:orderId
Authorization: Bearer <token>
```

### Get Customer Orders

```http
GET /api/orders/customer/:customerId
Authorization: Bearer <token>
```

### Cancel Order

```http
PUT /api/orders/:orderId/cancel
Authorization: Bearer <token>
```

---

## Payments API

### Create Payment Intent

```http
POST /api/payments/create-intent
Authorization: Bearer <token>
Content-Type: application/json

{
  "amount": 207.98,
  "currency": "usd",
  "orderData": {
    "items": [...],
    "shippingAddress": {...}
  }
}
```

**Response:**
```json
{
  "clientSecret": "pi_123_secret_456",
  "paymentIntentId": "pi_123"
}
```

### Confirm Payment

```http
POST /api/payments/confirm-payment
Content-Type: application/json

{
  "paymentIntentId": "pi_123",
  "paymentMethodId": "pm_123"
}
```

**Response:**
```json
{
  "success": true,
  "orderId": "ord_123"
}
```

---

## Email API

### Send Order Confirmation

```http
POST /api/email/order-confirmation
Content-Type: application/json

{
  "email": "customer@example.com",
  "order": {...}
}
```

### Send Shipping Notification

```http
POST /api/email/shipping-notification
Content-Type: application/json

{
  "email": "customer@example.com",
  "order": {...},
  "trackingNumber": "TRACK123"
}
```

### Subscribe to Newsletter

```http
POST /api/email/newsletter
Content-Type: application/json

{
  "email": "subscriber@example.com"
}
```

### Send Abandoned Cart Email

```http
POST /api/email/abandoned-cart
Content-Type: application/json

{
  "email": "customer@example.com",
  "cartItems": [...]
}
```

---

## Admin API

**Note:** All admin endpoints require admin role authentication.

### Get Dashboard Stats

```http
GET /api/admin/dashboard
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "totalOrders": 1234,
  "totalRevenue": 123456.78,
  "totalProducts": 56,
  "lowStockCount": 3,
  "recentOrders": [...]
}
```

### Manage Products

```http
POST /api/admin/products
PUT /api/admin/products/:productId
DELETE /api/admin/products/:productId
```

### Manage Orders

```http
GET /api/admin/orders
PUT /api/admin/orders/:orderId/status
```

---

## Inventory API

### Check Inventory

```http
GET /api/inventory/check?productId=prod_123&variantId=var_123
```

**Response:**
```json
{
  "productId": "prod_123",
  "variantId": "var_123",
  "available": 50,
  "reserved": 5,
  "total": 55
}
```

### Reserve Inventory

```http
POST /api/inventory/reserve
Content-Type: application/json

{
  "items": [
    {
      "productId": "prod_123",
      "variantId": "var_123",
      "quantity": 2
    }
  ]
}
```

### Confirm Reservation

```http
POST /api/inventory/confirm
Content-Type: application/json

{
  "reservationId": "res_123"
}
```

### Get Low Stock Products

```http
GET /api/inventory/low-stock?threshold=10
```

---

## Recommendations API

### Get User Recommendations

```http
GET /api/recommendations/:userId
Authorization: Bearer <token>
```

**Response:**
```json
{
  "recommendations": [
    {
      "id": "prod_456",
      "name": "Matching Cap",
      "reason": "Based on your purchase history"
    }
  ]
}
```

### Get Frequently Bought Together

```http
GET /api/recommendations/frequently-bought/:productId
```

**Response:**
```json
{
  "product": {...},
  "frequentlyBoughtWith": [...]
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "PRODUCT_NOT_FOUND",
    "message": "Product with ID prod_123 not found",
    "details": {},
    "timestamp": "2025-11-11T00:00:00Z"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., out of stock) |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Code | Description |
|------|-------------|
| `AUTH_REQUIRED` | Authentication required |
| `INVALID_CREDENTIALS` | Invalid username or password |
| `PRODUCT_NOT_FOUND` | Product doesn't exist |
| `OUT_OF_STOCK` | Product out of stock |
| `INVALID_QUANTITY` | Invalid quantity requested |
| `PAYMENT_FAILED` | Payment processing failed |
| `ORDER_NOT_FOUND` | Order doesn't exist |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |

---

## Rate Limiting

### Limits

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Public APIs | 100 requests | Per minute |
| Authenticated APIs | 500 requests | Per minute |
| Admin APIs | 1000 requests | Per minute |
| Payment APIs | 50 requests | Per minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699999999
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Please try again later.",
    "retryAfter": 60
  }
}
```

---

## Pagination

### Query Parameters

- `limit`: Items per page (default: 20, max: 100)
- `offset`: Number of items to skip (default: 0)
- `page`: Page number (alternative to offset)

### Response Format

```json
{
  "data": [...],
  "pagination": {
    "total": 100,
    "limit": 20,
    "offset": 0,
    "page": 1,
    "totalPages": 5
  }
}
```

---

## Webhooks

### Available Webhooks

- `order.created`
- `order.completed`
- `order.cancelled`
- `payment.succeeded`
- `payment.failed`
- `product.out_of_stock`

### Webhook Payload

```json
{
  "id": "evt_123",
  "type": "order.created",
  "data": {...},
  "timestamp": "2025-11-11T00:00:00Z"
}
```

---

## SDK & Libraries

### JavaScript/TypeScript

```bash
npm install @live-it-iconic/sdk
```

```typescript
import { LiveItIconic } from '@live-it-iconic/sdk';

const client = new LiveItIconic({
  apiKey: 'your_api_key',
  environment: 'production'
});

const products = await client.products.list();
```

---

## Support

- **Documentation:** https://docs.liveiticon.com
- **API Status:** https://status.liveiticon.com
- **Support:** support@liveiticon.com
- **Community:** https://community.liveiticon.com

---

## Changelog

### v1.0.0 (2025-11-11)
- Initial API documentation
- All core endpoints documented
- Authentication, products, orders, payments
- Error handling and rate limiting
- Admin API endpoints

---

**Note:** This API is under active development. Breaking changes will be versioned and communicated in advance.
