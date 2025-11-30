# ATLAS API Gateway Service
FROM atlas-base:latest

# Copy API and CLI code
COPY tools/atlas/cli/ ./cli/
COPY docs/atlas/api/ ./api/
COPY tools/atlas/services/ ./services/
COPY tools/atlas/types/ ./types/
COPY tools/atlas/utils/ ./utils/

# Install additional dependencies for API Gateway
RUN npm install --save \
    express \
    cors \
    helmet \
    express-rate-limit \
    express-jwt \
    bcryptjs \
    jsonwebtoken \
    joi \
    swagger-jsdoc \
    swagger-ui-express \
    prom-client \
    winston \
    redis

# Create service directory
RUN mkdir -p /app/api-gateway

# Copy service files
COPY tools/atlas/cli/index.ts /app/api-gateway/
COPY docs/atlas/api/rest-api.md /app/api-gateway/
COPY docs/atlas/api/python-sdk.md /app/api-gateway/
COPY docs/atlas/api/typescript-sdk.md /app/api-gateway/

# Create API routes
RUN mkdir -p /app/api-gateway/routes
COPY tools/atlas/services/api/routes/ /app/api-gateway/routes/

# Set environment variables
ENV SERVICE_NAME=api-gateway
ENV SERVICE_PORT=3000
ENV METRICS_PORT=9090

# API configuration
ENV API_RATE_LIMIT_REQUESTS=1000
ENV API_RATE_LIMIT_WINDOW=900000
ENV JWT_SECRET=""
ENV JWT_EXPIRATION=3600

# Redis for rate limiting
ENV REDIS_URL=redis://redis:6379

# CORS configuration
ENV CORS_ORIGIN="*"
ENV CORS_CREDENTIALS=true

# Swagger documentation
ENV SWAGGER_ENABLED=true
ENV SWAGGER_PATH=/api-docs

# Expose ports
EXPOSE 3000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start API Gateway service
CMD ["node", "api-gateway/index.js"]</instructions>