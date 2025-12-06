// GraphQL Federation Gateway for TalAI
// Distributed GraphQL API with edge optimization

import { ApolloServer } from '@apollo/server';
import { ApolloGateway, IntrospectAndCompose, RemoteGraphQLDataSource } from '@apollo/gateway';
import { ApolloServerPluginLandingPageLocalDefault } from '@apollo/server/plugin/landingPage/default';
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { expressMiddleware } from '@apollo/server/express4';
import express from 'express';
import http from 'http';
import cors from 'cors';
import { readFileSync } from 'fs';
import { buildSubgraphSchema } from '@apollo/subgraph';
import gql from 'graphql-tag';
import DataLoader from 'dataloader';
import Redis from 'ioredis';
import { RateLimiterRedis } from 'rate-limiter-flexible';

// Configuration
const CONFIG = {
  services: [
    { name: 'research', url: process.env.RESEARCH_SERVICE_URL || 'http://research-service:4001/graphql' },
    { name: 'agents', url: process.env.AGENTS_SERVICE_URL || 'http://agents-service:4002/graphql' },
    { name: 'users', url: process.env.USERS_SERVICE_URL || 'http://users-service:4003/graphql' },
    { name: 'prompts', url: process.env.PROMPTS_SERVICE_URL || 'http://prompts-service:4004/graphql' },
    { name: 'analytics', url: process.env.ANALYTICS_SERVICE_URL || 'http://analytics-service:4005/graphql' },
    { name: 'billing', url: process.env.BILLING_SERVICE_URL || 'http://billing-service:4006/graphql' }
  ],
  redis: {
    host: process.env.REDIS_HOST || 'redis-master',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD
  },
  rateLimits: {
    anonymous: { points: 10, duration: 60 },
    authenticated: { points: 100, duration: 60 },
    premium: { points: 1000, duration: 60 }
  },
  cache: {
    defaultTTL: 300,
    maxSize: 1000
  }
};

// Initialize Redis clients
const redis = new Redis(CONFIG.redis);
const pubClient = new Redis(CONFIG.redis);
const subClient = new Redis(CONFIG.redis);

// Rate limiters
const rateLimiters = {
  anonymous: new RateLimiterRedis({
    storeClient: redis,
    keyPrefix: 'rl:anon:',
    points: CONFIG.rateLimits.anonymous.points,
    duration: CONFIG.rateLimits.anonymous.duration
  }),
  authenticated: new RateLimiterRedis({
    storeClient: redis,
    keyPrefix: 'rl:auth:',
    points: CONFIG.rateLimits.authenticated.points,
    duration: CONFIG.rateLimits.authenticated.duration
  }),
  premium: new RateLimiterRedis({
    storeClient: redis,
    keyPrefix: 'rl:prem:',
    points: CONFIG.rateLimits.premium.points,
    duration: CONFIG.rateLimits.premium.duration
  })
};

// Custom RemoteGraphQLDataSource with caching and monitoring
class CachedDataSource extends RemoteGraphQLDataSource {
  constructor(config) {
    super(config);
    this.cache = new Map();
    this.loader = new DataLoader(
      keys => this.batchLoad(keys),
      { cache: false }
    );
  }

  async willSendRequest({ request, context }) {
    // Add authentication headers
    if (context.token) {
      request.http.headers.set('authorization', context.token);
    }

    // Add tracing headers
    if (context.traceId) {
      request.http.headers.set('x-trace-id', context.traceId);
      request.http.headers.set('x-span-id', context.spanId);
    }

    // Add service mesh headers
    request.http.headers.set('x-forwarded-for', context.ip || 'unknown');
    request.http.headers.set('x-request-id', context.requestId);
  }

  async didReceiveResponse({ response, request, context }) {
    // Cache successful responses
    if (response.http.status === 200 && request.http.method === 'GET') {
      const cacheKey = this.getCacheKey(request);
      const ttl = this.getCacheTTL(request);

      await redis.setex(
        `gql:${cacheKey}`,
        ttl,
        JSON.stringify(response.data)
      );
    }

    // Collect metrics
    const duration = Date.now() - context.startTime;
    await redis.hincrby('metrics:graphql:requests', this.name, 1);
    await redis.hincrbyfloat('metrics:graphql:duration', this.name, duration);

    return response;
  }

  async didEncounterError(error) {
    console.error(`[${this.name}] Error:`, error);
    await redis.hincrby('metrics:graphql:errors', this.name, 1);
    throw error;
  }

  getCacheKey(request) {
    const { query, variables } = request.body;
    return `${this.name}:${query}:${JSON.stringify(variables || {})}`;
  }

  getCacheTTL(request) {
    // Different TTL based on operation type
    const query = request.body.query;
    if (query.includes('mutation')) return 0;
    if (query.includes('subscription')) return 0;
    if (query.includes('research')) return 3600; // 1 hour for research
    if (query.includes('prompt')) return 1800; // 30 minutes for prompts
    return CONFIG.cache.defaultTTL;
  }

  async batchLoad(keys) {
    // Batch load implementation for DataLoader
    const results = await Promise.all(
      keys.map(key => this.loadFromCache(key))
    );
    return results;
  }

  async loadFromCache(key) {
    const cached = await redis.get(`gql:${key}`);
    if (cached) {
      return JSON.parse(cached);
    }
    return null;
  }
}

// Apollo Gateway with federation
const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs: CONFIG.services,
    pollIntervalInMs: 10000 // Poll for schema changes every 10 seconds
  }),
  buildService({ name, url }) {
    return new CachedDataSource({
      url,
      name,
      willSendRequest: ({ request, context }) => {
        request.http.timeout = 30000; // 30 second timeout
      }
    });
  },
  // Custom error handling
  didEncounterError({ error, service }) {
    console.error(`Gateway error from ${service}:`, error);
  }
});

// Create Apollo Server
async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);

  const server = new ApolloServer({
    gateway,
    csrfPrevention: true,
    cache: 'bounded',
    plugins: [
      ApolloServerPluginDrainHttpServer({ httpServer }),
      ApolloServerPluginLandingPageLocalDefault({ embed: true }),
      // Custom plugin for metrics and logging
      {
        async requestDidStart() {
          const startTime = Date.now();
          return {
            async willSendResponse(requestContext) {
              const duration = Date.now() - startTime;
              console.log(`Query completed in ${duration}ms`);

              // Update metrics
              await redis.hincrby('metrics:gateway:requests', 'total', 1);
              await redis.hincrbyfloat('metrics:gateway:duration', 'total', duration);

              // Log slow queries
              if (duration > 1000) {
                console.warn('Slow query detected:', {
                  duration,
                  query: requestContext.request.query,
                  variables: requestContext.request.variables
                });
              }
            },
            async didEncounterErrors(requestContext) {
              console.error('GraphQL errors:', requestContext.errors);
              await redis.hincrby('metrics:gateway:errors', 'total', 1);
            }
          };
        }
      }
    ],
    formatError: (err) => {
      // Remove sensitive information from errors in production
      if (process.env.NODE_ENV === 'production') {
        delete err.extensions.exception;
        delete err.extensions.stacktrace;
      }
      return err;
    }
  });

  await server.start();

  // Apply middleware
  app.use(
    '/graphql',
    cors({
      origin: true,
      credentials: true,
      methods: ['GET', 'POST', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key']
    }),
    express.json({ limit: '10mb' }),
    // Rate limiting middleware
    async (req, res, next) => {
      const apiKey = req.headers['x-api-key'];
      const token = req.headers.authorization;

      let rateLimiter = rateLimiters.anonymous;
      let identifier = req.ip;

      if (apiKey && apiKey.startsWith('pk_')) {
        rateLimiter = rateLimiters.premium;
        identifier = apiKey;
      } else if (token) {
        rateLimiter = rateLimiters.authenticated;
        identifier = token;
      }

      try {
        await rateLimiter.consume(identifier);
        next();
      } catch (rejRes) {
        res.status(429).json({
          error: 'Too Many Requests',
          retryAfter: Math.round(rejRes.msBeforeNext / 1000) || 60
        });
      }
    },
    // Context middleware
    async (req, res, next) => {
      req.context = {
        token: req.headers.authorization,
        apiKey: req.headers['x-api-key'],
        ip: req.ip,
        requestId: req.headers['x-request-id'] || generateRequestId(),
        traceId: req.headers['x-trace-id'] || generateTraceId(),
        spanId: generateSpanId(),
        startTime: Date.now(),
        dataSources: {}
      };
      next();
    },
    expressMiddleware(server, {
      context: async ({ req }) => req.context
    })
  );

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
  });

  // Metrics endpoint
  app.get('/metrics', async (req, res) => {
    const metrics = await gatherMetrics();
    res.type('text/plain').send(metrics);
  });

  // WebSocket support for subscriptions
  const { WebSocketServer } = await import('ws');
  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/graphql'
  });

  const { useServer } = await import('graphql-ws/lib/use/ws');
  const { execute, subscribe } = await import('graphql');

  useServer(
    {
      schema: gateway.schema,
      execute,
      subscribe,
      context: async (ctx) => {
        return {
          ...ctx,
          dataSources: {}
        };
      },
      onConnect: async (ctx) => {
        console.log('WebSocket connected');
        return true;
      },
      onDisconnect: async (ctx) => {
        console.log('WebSocket disconnected');
      }
    },
    wsServer
  );

  await new Promise((resolve) => httpServer.listen({ port: 4000 }, resolve));
  console.log(`🚀 GraphQL Gateway ready at http://localhost:4000/graphql`);
}

// Metrics collection
async function gatherMetrics() {
  const requests = await redis.hgetall('metrics:gateway:requests');
  const duration = await redis.hgetall('metrics:gateway:duration');
  const errors = await redis.hgetall('metrics:gateway:errors');

  let metrics = '# TYPE gateway_requests_total counter\n';
  for (const [service, count] of Object.entries(requests)) {
    metrics += `gateway_requests_total{service="${service}"} ${count}\n`;
  }

  metrics += '\n# TYPE gateway_duration_seconds histogram\n';
  for (const [service, total] of Object.entries(duration)) {
    const avgDuration = parseFloat(total) / parseInt(requests[service] || 1);
    metrics += `gateway_duration_seconds{service="${service}"} ${avgDuration / 1000}\n`;
  }

  metrics += '\n# TYPE gateway_errors_total counter\n';
  for (const [service, count] of Object.entries(errors)) {
    metrics += `gateway_errors_total{service="${service}"} ${count}\n`;
  }

  return metrics;
}

// Helper functions
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function generateTraceId() {
  return `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function generateSpanId() {
  return `span_${Math.random().toString(36).substr(2, 9)}`;
}

// Schema stitching for specific services
const researchSchema = gql`
  extend type Query {
    searchResearch(query: String!, limit: Int = 10): [ResearchPaper!]!
    getResearchById(id: ID!): ResearchPaper
    getCitations(paperId: ID!): [Citation!]!
  }

  type ResearchPaper @key(fields: "id") {
    id: ID!
    title: String!
    abstract: String!
    authors: [Author!]!
    publishedDate: String!
    citations: Int!
    keywords: [String!]!
  }

  type Author @key(fields: "id") {
    id: ID!
    name: String!
    affiliation: String
    papers: [ResearchPaper!]!
  }

  type Citation {
    id: ID!
    fromPaper: ResearchPaper!
    toPaper: ResearchPaper!
    context: String
  }
`;

const agentsSchema = gql`
  extend type Query {
    getAgents: [Agent!]!
    getAgent(id: ID!): Agent
    getAgentMetrics(agentId: ID!, timeRange: TimeRange!): AgentMetrics!
  }

  extend type Mutation {
    createAgent(input: CreateAgentInput!): Agent!
    updateAgent(id: ID!, input: UpdateAgentInput!): Agent!
    executeAgent(id: ID!, input: ExecuteAgentInput!): ExecutionResult!
  }

  type Agent @key(fields: "id") {
    id: ID!
    name: String!
    type: AgentType!
    status: AgentStatus!
    capabilities: [String!]!
    configuration: JSON!
    metrics: AgentMetrics
  }

  type AgentMetrics {
    executionCount: Int!
    averageLatency: Float!
    successRate: Float!
    tokenUsage: TokenUsage!
  }

  type TokenUsage {
    total: Int!
    input: Int!
    output: Int!
    cost: Float!
  }

  enum AgentType {
    RESEARCH
    ANALYSIS
    GENERATION
    REVIEW
    ORCHESTRATION
  }

  enum AgentStatus {
    IDLE
    RUNNING
    SUSPENDED
    ERROR
  }
`;

// Start the server
startServer().catch((err) => {
  console.error('Failed to start GraphQL Gateway:', err);
  process.exit(1);
});