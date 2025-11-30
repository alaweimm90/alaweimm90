# ATLAS Execution Service
FROM atlas-base:latest

# Copy execution service code
COPY tools/atlas/agents/ ./agents/
COPY tools/atlas/services/ ./services/
COPY tools/atlas/types/ ./types/
COPY tools/atlas/utils/ ./utils/

# Install additional dependencies for execution
RUN npm install --save \
    @anthropic-ai/sdk \
    openai \
    @google/generative-ai \
    axios \
    prom-client \
    winston \
    joi \
    rate-limiter-flexible

# Create service directory
RUN mkdir -p /app/execution

# Copy service files
COPY tools/atlas/agents/index.js /app/execution/
COPY tools/atlas/agents/claude-adapter.js /app/execution/
COPY tools/atlas/agents/gpt4-adapter.js /app/execution/
COPY tools/atlas/agents/gemini-adapter.js /app/execution/
COPY tools/atlas/agents/local-adapter.js /app/execution/

# Set environment variables
ENV SERVICE_NAME=execution
ENV SERVICE_PORT=3002
ENV REDIS_URL=redis://redis:6379
ENV METRICS_PORT=9092

# AI Provider API Keys (should be set via secrets)
ENV ANTHROPIC_API_KEY=""
ENV OPENAI_API_KEY=""
ENV GOOGLE_API_KEY=""

# Rate limiting
ENV RATE_LIMIT_REQUESTS=100
ENV RATE_LIMIT_WINDOW=900000

# Expose ports
EXPOSE 3002 9092

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3002/health || exit 1

# Start execution service
CMD ["node", "execution/index.js"]</instructions>