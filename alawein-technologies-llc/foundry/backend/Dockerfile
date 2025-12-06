# Multi-stage Dockerfile for API Backend
# Production image size target: <150MB

# Stage 1: Dependencies
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat python3 make g++
WORKDIR /app

# Copy dependency files
COPY package*.json ./
# Install production dependencies
RUN npm ci --only=production
# Copy production dependencies aside
RUN cp -R node_modules /prod_node_modules
# Install all dependencies (including dev)
RUN npm ci

# Stage 2: Builder
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the application (if TypeScript)
RUN npm run build || npm run compile || echo "No build step required"

# Stage 3: Production image
FROM node:20-alpine AS runner
WORKDIR /app

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Security: Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 api

# Set production environment
ENV NODE_ENV production

# Copy production dependencies
COPY --from=deps /prod_node_modules ./node_modules

# Copy built application
COPY --from=builder /app/dist ./dist 2>/dev/null || true
COPY --from=builder /app/src ./src 2>/dev/null || true
COPY --from=builder /app/package*.json ./

# Security: Change ownership
RUN chown -R api:nodejs /app

USER api

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8000/health', (r) => {r.statusCode === 200 ? process.exit(0) : process.exit(1)})"

EXPOSE 808000

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]

# Development stage
FROM node:20-alpine AS development
WORKDIR /app

# Install additional dev tools
RUN apk add --no-cache git python3 make g++

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 808000

CMD ["npm", "run", "dev"]