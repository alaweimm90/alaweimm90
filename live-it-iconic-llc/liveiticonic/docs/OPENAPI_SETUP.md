# OpenAPI/Swagger Setup Guide

Complete setup and deployment guide for the Live It Iconic OpenAPI/Swagger documentation.

## Overview

The Live It Iconic API documentation is built using OpenAPI 3.0 specification with Swagger UI for interactive exploration. All endpoints are fully documented with request/response schemas, examples, and error codes.

## Files Structure

```
project/
├── src/
│   ├── api/
│   │   ├── openapi.ts              # OpenAPI specification definition
│   │   ├── auth/
│   │   │   ├── signup.ts           # Documented with @swagger
│   │   │   ├── signin.ts           # Documented with @swagger
│   │   │   └── me.ts               # Documented with @swagger
│   │   ├── orders/
│   │   │   ├── create.ts           # Documented with @swagger
│   │   │   └── [id].ts             # Documented with @swagger
│   │   ├── inventory/
│   │   │   └── check.ts            # Documented with @swagger
│   │   ├── payments/
│   │   │   └── create-intent.ts    # Documented with @swagger
│   │   └── ... (other endpoints)
│   ├── pages/
│   │   └── ApiDocs.tsx             # Swagger UI React page
│   └── lib/
│       └── apiClient.ts            # Type-safe API client library
├── docs/
│   ├── API_DOCUMENTATION.md        # Comprehensive API reference (1500+ lines)
│   ├── SDK_GUIDE.md                # API Client SDK usage guide
│   └── OPENAPI_SETUP.md            # This file
├── postman/
│   ├── Live-It-Iconic-API.postman_collection.json
│   └── Live-It-Iconic-API-Environment.json
├── tests/
│   └── api/
│       └── auth.test.ts            # API testing suite
└── package.json
```

## Installation

### 1. Installed Dependencies

```bash
npm install swagger-jsdoc swagger-ui-react
npm install -D @types/swagger-jsdoc @types/swagger-ui-express
```

**Packages:**
- `swagger-jsdoc`: Extracts OpenAPI spec from JSDoc comments
- `swagger-ui-react`: Interactive Swagger UI component
- Type definitions for development

### 2. Environment Configuration

Add to `.env` or `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:5173
VITE_API_DOCS_ENABLED=true
```

## OpenAPI Specification

### Location

```
src/api/openapi.ts
```

### Structure

The specification includes:

```typescript
const options: swaggerJsdoc.Options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Live It Iconic API',
      version: '1.0.0',
      description: 'E-commerce API for Live It Iconic',
      contact: { ... },
      license: { ... }
    },
    servers: [
      { url: 'http://localhost:5173', description: 'Development' },
      { url: 'https://api.liveiconic.com', description: 'Production' }
    ],
    components: {
      securitySchemes: {
        bearerAuth: { type: 'http', scheme: 'bearer' }
      },
      schemas: {
        User: { ... },
        Product: { ... },
        Order: { ... },
        // ... more schemas
      }
    }
  },
  apis: ['./src/api/**/*.ts'] // Scans for @swagger comments
};

export const swaggerSpec = swaggerJsdoc(options);
```

### Updating the Spec

To add or update the specification:

1. Modify `src/api/openapi.ts` for global definitions
2. Add `@swagger` comments to endpoint files
3. Spec automatically regenerates on page load

## Documenting Endpoints

### Using @swagger Comments

Every API endpoint should include JSDoc comments with `@swagger` tag:

```typescript
/**
 * @swagger
 * /api/auth/signup:
 *   post:
 *     summary: Register a new user
 *     description: Creates a new user account
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - password
 *             properties:
 *               email: { type: string, format: email }
 *               password: { type: string }
 *     responses:
 *       200:
 *         description: User created successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/AuthResponse'
 *       400:
 *         description: Validation error
 */
export default async function handler(req: Request) {
  // Implementation
}
```

### Schema References

Reuse defined schemas:

```yaml
schema:
  $ref: '#/components/schemas/User'
```

### Common Schema Types

See `src/api/openapi.ts` for defined schemas:

- `User`: User profile with id, email, name
- `Product`: Product with price, category, stock
- `Order`: Order with items, total, status
- `OrderItem`: Individual order line item
- `AuthResponse`: Response from auth endpoints
- `Error`: Standard error response

## Swagger UI Page

### Location

```
src/pages/ApiDocs.tsx
```

### Usage

The component renders interactive Swagger UI:

```typescript
import { ApiDocs } from '@/pages/ApiDocs';

// In your router
<Route path="/api-docs" element={<ApiDocs />} />
```

### Features

- Browse all endpoints
- See request/response schemas
- View examples for each endpoint
- Try endpoints directly from browser
- Download OpenAPI spec as JSON/YAML
- Authentication with token
- Response/request logging

### Styling

The component uses Tailwind CSS classes:

```tsx
<div className="min-h-screen bg-lii-bg">
  <div className="container mx-auto py-8 px-4">
    <h1 className="text-4xl font-display text-lii-bg">API Documentation</h1>
    {/* ... */}
  </div>
</div>
```

## API Client SDK

### Location

```
src/lib/apiClient.ts
```

### Features

- Type-safe endpoints
- Automatic JWT token management
- Built-in error handling
- Token refresh on expiration
- localStorage persistence
- Request/response validation

### Exports

```typescript
export const apiClient: ApiClient;

// Type exports
export type {
  User,
  Product,
  Order,
  OrderItem,
  AuthResponse,
  ApiError,
  ProductFilters,
  SignupDTO,
  SigninDTO,
  CreateOrderDTO
};
```

### Usage

```typescript
import { apiClient, type Product } from '@/lib/apiClient';

const products: Product[] = await apiClient.getProducts();
```

## Postman Integration

### Files

- `postman/Live-It-Iconic-API.postman_collection.json` - Complete API requests
- `postman/Live-It-Iconic-API-Environment.json` - Variables and configuration

### Import into Postman

1. Open Postman
2. Click "Import" button
3. Select both JSON files
4. Collection and environment are imported

### Available Requests

- Authentication (signup, signin, profile)
- Products (list, detail)
- Orders (create, list, update)
- Inventory (check, reserve)
- Payments (create intent, confirm)

### Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| baseUrl | API base URL | http://localhost:5173 |
| accessToken | JWT token | (empty) |
| userId | Current user ID | (empty) |
| orderId | Order ID | (empty) |
| productId | Product ID | prod_123456789 |

## Testing

### Test Suite Location

```
tests/api/auth.test.ts
```

### Running Tests

```bash
npm run test
npm run test:ui
npm run test:coverage
```

### Test Coverage

- Authentication endpoints
- Input validation
- Error handling
- Token management
- Rate limiting
- Edge cases

### Adding More Tests

Create test files following the pattern:

```typescript
// tests/api/products.test.ts
import { describe, it, expect } from 'vitest';
import { apiClient } from '@/lib/apiClient';

describe('Products API', () => {
  it('should get all products', async () => {
    const products = await apiClient.getProducts();
    expect(products).toBeInstanceOf(Array);
  });

  // More tests...
});
```

## Documentation Files

### API_DOCUMENTATION.md

Comprehensive API reference (1500+ lines):

- Overview and features
- Complete endpoint reference
- Request/response examples
- Error codes and handling
- Rate limiting
- Security best practices
- Versioning strategy
- Usage examples
- Changelog

### SDK_GUIDE.md

API Client SDK usage guide:

- Installation and setup
- Authentication flows
- Making requests to each endpoint type
- Error handling
- Advanced features
- Complete examples
- Best practices
- Troubleshooting

## Deployment

### Development

```bash
npm run dev
# Visit http://localhost:5173/api-docs
```

### Building for Production

```bash
npm run build
```

The OpenAPI spec is bundled with the application. Update at build time.

### Production Deployment

1. Build the application
2. Serve static files
3. API docs available at `/api-docs`

### Environment Variables

```env
# Production
VITE_API_BASE_URL=https://api.liveiconic.com
VITE_API_DOCS_ENABLED=true
```

## Updating Documentation

### Add New Endpoint

1. Create endpoint handler in `src/api/`
2. Add `@swagger` JSDoc comment
3. Refresh Swagger UI page
4. Update `docs/API_DOCUMENTATION.md`
5. Add to Postman collection

### Update Existing Endpoint

1. Modify `@swagger` comment in endpoint file
2. Update schema in `src/api/openapi.ts` if needed
3. Refresh Swagger UI
4. Update documentation files

### Schema Changes

1. Update schema in `src/api/openapi.ts`
2. Update endpoint `@swagger` comments that use it
3. Update `src/lib/apiClient.ts` types
4. Update tests to match new schema

## Verification

### Checklist

- [ ] All 21+ endpoints documented with @swagger
- [ ] OpenAPI spec loads without errors
- [ ] Swagger UI page renders at /api-docs
- [ ] Postman collection imports successfully
- [ ] API Client has types for all endpoints
- [ ] Tests pass for documented endpoints
- [ ] API_DOCUMENTATION.md is comprehensive (1500+ lines)
- [ ] SDK_GUIDE.md covers all client methods
- [ ] Environment files configured
- [ ] Examples in documentation work

### Validate OpenAPI Spec

Use the OpenAPI Validator:

```bash
npm install -D @apidevtools/swagger-cli
npx swagger-cli validate src/api/openapi.ts
```

## Endpoints Documented

### Authentication (3)
- POST /api/auth/signup
- POST /api/auth/signin
- GET /api/auth/me

### Products (2)
- GET /api/products
- GET /api/products/{id}

### Orders (4)
- POST /api/orders/create
- GET /api/orders/{id}
- PATCH /api/orders/{id}
- GET /api/orders/customer/{customerId}

### Inventory (2)
- GET /api/inventory/check
- POST /api/inventory/reserve

### Payments (2)
- POST /api/payments/create-intent
- POST /api/payments/confirm-payment

### Additional Endpoints (8+)
- Email services
- Recommendations
- Admin dashboard
- Low stock alerts
- Order confirmation
- Newsletter
- Shipping notifications
- Abandoned cart

**Total: 21+ endpoints documented**

## Documentation Pages

1. `docs/API_DOCUMENTATION.md` - 1500+ lines comprehensive reference
2. `docs/SDK_GUIDE.md` - Complete SDK usage guide
3. `docs/OPENAPI_SETUP.md` - This file, setup guide
4. Swagger UI page at `/api-docs`
5. Postman collection with examples

## Support

For help with:
- **API usage**: See `docs/API_DOCUMENTATION.md`
- **SDK integration**: See `docs/SDK_GUIDE.md`
- **Setup issues**: Check this file
- **Testing**: See `tests/api/`

Contact: api-support@liveiconic.com

---

## Quick Links

- [Full API Documentation](./API_DOCUMENTATION.md)
- [SDK Usage Guide](./SDK_GUIDE.md)
- [OpenAPI Spec](../src/api/openapi.ts)
- [Swagger UI Component](../src/pages/ApiDocs.tsx)
- [API Client](../src/lib/apiClient.ts)
- [Postman Collection](../postman/Live-It-Iconic-API.postman_collection.json)
- [Test Suite](../tests/api/auth.test.ts)
