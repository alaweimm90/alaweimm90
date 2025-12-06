# API Development & Integration Standards

## Purpose

RESTful/GraphQL API design and implementation with enterprise-grade quality.

---

## Design Principles

### RESTful Conventions

```yaml
resources:
  /users:
    GET: List users (paginated)
    POST: Create user
  /users/{id}:
    GET: Get user by ID
    PUT: Update user (full)
    PATCH: Update user (partial)
    DELETE: Remove user

naming:
  - Use plural nouns for collections
  - Use kebab-case for multi-word resources
  - Nest related resources: /users/{id}/orders
```

### GraphQL Schema

```graphql
type Query {
  user(id: ID!): User
  users(filter: UserFilter, pagination: Pagination): UserConnection!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
}

type User {
  id: ID!
  email: String!
  profile: Profile
  orders: [Order!]!
}
```

### Versioning Strategy

```yaml
strategies:
  url_versioning:
    pattern: /api/v1/resource
    pros: Clear, cacheable
    cons: URL pollution

  header_versioning:
    header: Accept-Version: v1
    pros: Clean URLs
    cons: Less discoverable

  recommended: url_versioning for public APIs
```

---

## Authentication & Authorization

### OAuth2 / JWT

```yaml
authentication:
  type: oauth2
  flows:
    - authorization_code # Web apps
    - client_credentials # Service-to-service
    - refresh_token # Token renewal

jwt_structure:
  header:
    alg: RS256
    typ: JWT
  payload:
    sub: user_id
    iat: issued_at
    exp: expiration
    scope: [read, write]
  signature: [private_key_signed]
```

### API Keys

```yaml
api_keys:
  generation: cryptographically_secure_random
  storage: hashed_in_database
  rotation: every_90_days
  scoping: per_environment
  rate_limiting: per_key
```

### Rate Limiting

```yaml
rate_limits:
  default:
    requests: 1000
    window: 1_hour
  authenticated:
    requests: 10000
    window: 1_hour
  burst:
    requests: 100
    window: 1_minute

headers:
  X-RateLimit-Limit: [max requests]
  X-RateLimit-Remaining: [remaining]
  X-RateLimit-Reset: [reset timestamp]
```

---

## Implementation Standards

### OpenAPI/Swagger Specification

```yaml
openapi: 3.0.3
info:
  title: API Name
  version: 1.0.0
  description: API description

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://api-staging.example.com/v1
    description: Staging

paths:
  /resource:
    get:
      summary: List resources
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
      responses:
        200:
          description: Success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ResourceList'
```

### Input Validation

```typescript
// Zod schema example
const CreateUserSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
  name: z.string().min(1).max(100),
  role: z.enum(['user', 'admin']).default('user'),
});

// Validation middleware
app.post('/users', validate(CreateUserSchema), createUser);
```

### Error Handling

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ],
    "requestId": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Standard HTTP Status Codes

```yaml
success:
  200: OK (GET, PUT, PATCH)
  201: Created (POST)
  204: No Content (DELETE)

client_errors:
  400: Bad Request (validation)
  401: Unauthorized (auth required)
  403: Forbidden (insufficient permissions)
  404: Not Found
  409: Conflict (duplicate)
  422: Unprocessable Entity
  429: Too Many Requests

server_errors:
  500: Internal Server Error
  502: Bad Gateway
  503: Service Unavailable
  504: Gateway Timeout
```

### Logging and Monitoring

```yaml
logging:
  format: json
  fields:
    - timestamp
    - request_id
    - method
    - path
    - status_code
    - duration_ms
    - user_id
    - ip_address

monitoring:
  metrics:
    - request_count
    - request_duration_histogram
    - error_rate
    - active_connections

  alerts:
    - error_rate > 1%
    - p99_latency > 500ms
    - availability < 99.9%
```

---

## Testing Requirements

### Unit Tests

```typescript
describe('UserController', () => {
  it('should create user with valid input', async () => {
    const response = await request(app)
      .post('/users')
      .send({ email: 'test@example.com', password: 'secure123' });

    expect(response.status).toBe(201);
    expect(response.body.email).toBe('test@example.com');
  });
});
```

### Integration Tests

```yaml
integration_tests:
  - database_connectivity
  - external_service_mocking
  - authentication_flow
  - rate_limiting_behavior
```

### Load Testing

```yaml
load_testing:
  tool: k6 # or artillery, locust
  scenarios:
    smoke:
      vus: 1
      duration: 1m
    load:
      vus: 100
      duration: 10m
    stress:
      vus: 500
      duration: 5m
    spike:
      vus: 1000
      duration: 1m
```

### Security Testing (OWASP)

```yaml
security_checks:
  - sql_injection
  - xss_prevention
  - csrf_protection
  - authentication_bypass
  - authorization_flaws
  - sensitive_data_exposure
  - rate_limit_bypass
```

---

## Documentation Requirements

- [ ] OpenAPI specification complete
- [ ] All endpoints documented
- [ ] Request/response examples provided
- [ ] Authentication flows explained
- [ ] Rate limits documented
- [ ] Error codes catalogued
- [ ] SDK/client libraries documented
- [ ] Changelog maintained
