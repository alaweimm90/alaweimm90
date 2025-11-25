# Multi-stage Dockerfile for AI Agent Demo
# Node.js Express application (monorepo)

# Stage 1: Base
FROM node:20-alpine AS base
WORKDIR /app

# Stage 2: Dependencies
FROM base AS deps
# Copy root package files for monorepo dependencies
COPY package.json package-lock.json* pnpm-lock.yaml* ./
# Install all dependencies
RUN \
  if [ -f pnpm-lock.yaml ]; then \
    corepack enable pnpm && pnpm install --frozen-lockfile; \
  elif [ -f package-lock.json ]; then \
    npm install --prefer-offline --no-audit; \
  else \
    npm install --prefer-offline --no-audit; \
  fi

# Stage 3: Production
FROM base AS production
WORKDIR /app

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy app code only
COPY apps/ai-agent-demo ./apps/ai-agent-demo

# Set environment
ENV NODE_ENV=production
ENV PORT=3000

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nodejs && \
    chown -R nodejs:nodejs /app

USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000', (r) => {if(r.statusCode !== 200) throw new Error(r.statusCode)})"

# Start the application
WORKDIR /app/apps/ai-agent-demo
CMD ["node", "server.js"]
