# TRAE Code Generation Prompts

## Generate Routing Components from Templates

**Purpose**: Transform TRAE routing templates into production-ready code components for any
programming language and framework.

**Prerequisites**:

- Access to TRAE routing templates
- Target programming language selected
- Framework preferences identified
- Existing AI integration patterns understood

---

## CODE GENERATION FRAMEWORK

You are a senior full-stack engineer specializing in AI system integration. Your task is to generate
production-ready routing components by adapting TRAE templates to any programming language and
framework.

### Language Adaptation Strategy

**Template Transformation Rules**:

1. **TypeScript → Target Language**: Convert TypeScript interfaces to native types
2. **Class Structure Preservation**: Maintain component architecture
3. **Error Handling Patterns**: Adapt to language-specific patterns
4. **Async/Await Patterns**: Convert to language-appropriate async patterns
5. **Configuration Management**: Use language-native config patterns

---

## COMPONENT 1: INTELLIGENT ROUTER GENERATION

### TypeScript Template Reference

```typescript
export class IntelligentRouter {
  constructor(config: RoutingConfig, aiEngine: AIEngineIntegration) {
    // Implementation
  }

  async routeRequest(prompt: string, context?: RoutingContext): Promise<RoutingResult> {
    // 7-step routing pipeline
  }
}
```

### Python Generation Prompt

**Target**: Generate Python IntelligentRouter with FastAPI/Flask integration

```
You are a Python AI routing specialist. Generate a complete IntelligentRouter class that:

LANGUAGE ADAPTATION:
- Convert TypeScript interfaces to Python dataclasses or TypedDict
- Use asyncio for async operations
- Implement proper Python error handling with custom exceptions
- Use Python logging instead of console.log
- Convert JSON configuration to Python dictionaries

FRAMEWORK INTEGRATION:
- Add FastAPI route decorators for HTTP endpoints
- Implement dependency injection with FastAPI Depends
- Add Pydantic models for request/response validation
- Include middleware for request timing and logging

PRODUCTION FEATURES:
- Add comprehensive logging with structured JSON output
- Implement circuit breaker pattern for reliability
- Add metrics collection with Prometheus integration
- Include health check endpoints
- Add graceful shutdown handling

CONFIGURATION MANAGEMENT:
- Use Pydantic settings for configuration
- Support environment variable overrides
- Add configuration validation
- Include default configurations for common scenarios

ERROR HANDLING:
- Create custom exception hierarchy
- Implement retry logic with exponential backoff
- Add error classification and reporting
- Include fallback error responses

Generate the complete IntelligentRouter implementation with:
1. Full class definition with all methods
2. Type hints for all parameters and return values
3. Comprehensive docstrings
4. Unit test examples
5. Integration examples with FastAPI
6. Configuration examples
7. Deployment considerations
```

### Java Generation Prompt

**Target**: Generate Java IntelligentRouter with Spring Boot integration

```
You are a Java enterprise architect. Generate a complete IntelligentRouter service that:

LANGUAGE ADAPTATION:
- Convert TypeScript interfaces to Java records or POJOs
- Use CompletableFuture for async operations
- Implement proper Java exception handling
- Use SLF4J logging with Logback
- Convert JSON configuration to application.yml/properties

FRAMEWORK INTEGRATION:
- Create Spring Boot service with @Service annotation
- Add REST controller with @RestController
- Implement dependency injection with @Autowired
- Add validation with Bean Validation annotations
- Include Spring Actuator for health checks

ENTERPRISE FEATURES:
- Add Spring Cloud Config integration
- Implement Hystrix circuit breaker
- Add Micrometer metrics collection
- Include distributed tracing with Sleuth
- Add database integration for audit logging

CONFIGURATION MANAGEMENT:
- Use @ConfigurationProperties for type-safe config
- Support multiple profiles (dev/staging/prod)
- Add configuration validation
- Include externalized configuration

ERROR HANDLING:
- Create custom exception classes extending RuntimeException
- Implement @ControllerAdvice for global error handling
- Add error classification and HTTP status mapping
- Include structured error responses

Generate the complete implementation with:
1. Service class with all business logic
2. REST controller for API endpoints
3. Configuration classes
4. Exception handling classes
5. Test classes with JUnit and Mockito
6. Application configuration files
7. Docker configuration for deployment
```

### Go Generation Prompt

**Target**: Generate Go IntelligentRouter with Gin/gRPC integration

```
You are a Go systems engineer. Generate a complete IntelligentRouter that:

LANGUAGE ADAPTATION:
- Convert TypeScript interfaces to Go structs
- Use goroutines and channels for concurrency
- Implement proper Go error handling with error types
- Use structured logging with zap or logrus
- Convert JSON configuration to YAML with viper

FRAMEWORK INTEGRATION:
- Create Gin HTTP handlers for REST API
- Add gRPC service definitions and implementations
- Implement dependency injection with fx or wire
- Add middleware for logging and metrics
- Include health check endpoints

CLOUD-NATIVE FEATURES:
- Add Kubernetes readiness/liveness probes
- Implement Prometheus metrics
- Add distributed tracing with OpenTelemetry
- Include service mesh integration (Istio/Linkerd)
- Add configuration hot-reloading

CONFIGURATION MANAGEMENT:
- Use viper for configuration management
- Support environment variables and config files
- Add configuration validation
- Include default configurations

ERROR HANDLING:
- Create custom error types with error wrapping
- Implement error classification and handling
- Add structured error responses
- Include panic recovery middleware

Generate the complete implementation with:
1. Service struct with all methods
2. HTTP/gRPC handlers
3. Configuration management
4. Error handling types
5. Test files with Go testing
6. Dockerfile and Kubernetes manifests
7. CI/CD pipeline configuration
```

---

## COMPONENT 2: MODEL SELECTOR GENERATION

### Universal Generation Prompt

**Target**: Generate ModelSelector for any language with task complexity analysis

```
You are an AI model selection expert. Generate a complete ModelSelector that:

CORE FUNCTIONALITY:
- Implement 8-indicator task complexity analysis
- Create 4-tier complexity classification (simple/moderate/complex/critical)
- Build multi-criteria model ranking system
- Add automatic fallback chain generation

LANGUAGE AGNOSTIC FEATURES:
- Convert complexity indicators to language-native patterns
- Implement capability matching algorithms
- Add model scoring and ranking logic
- Include geographic preference handling

INTELLIGENT SELECTION:
- Analyze prompt content for code, math, reasoning patterns
- Determine required capabilities based on complexity
- Rank models by quality, cost, reliability, and capability match
- Generate optimal fallback chains

CONFIGURATION INTEGRATION:
- Load model registry from configuration
- Support dynamic model updates
- Include capability override mechanisms
- Add model health and performance tracking

Generate with:
1. Complexity analysis engine
2. Model ranking and selection logic
3. Fallback chain generation
4. Configuration management
5. Comprehensive test cases
6. Performance benchmarks
```

---

## COMPONENT 3: COST CONTROLLER GENERATION

### Universal Generation Prompt

**Target**: Generate CostController with 7-10x cost reduction capabilities

```
You are a cost optimization architect. Generate a complete CostController that:

COST MANAGEMENT:
- Implement multi-layer budget controls (daily/monthly/per-request)
- Add real-time cost evaluation and automatic optimization
- Create emergency cost reduction measures (up to 90% savings)
- Build predictive cost analysis

OPTIMIZATION STRATEGIES:
- Automatic model tier downgrades based on budget
- Geographic cost optimization
- Time-based routing for cheaper off-peak usage
- Request batching and caching mechanisms

BUDGET ENFORCEMENT:
- Real-time budget monitoring and alerts
- Automatic cost control triggers
- Emergency mode activation with circuit breakers
- Cost prediction and forecasting

MONITORING & ANALYTICS:
- Comprehensive cost tracking and reporting
- Savings analysis and optimization recommendations
- Budget utilization dashboards
- Cost anomaly detection

Generate with:
1. Budget management system
2. Cost optimization engine
3. Real-time monitoring
4. Emergency controls
5. Analytics and reporting
6. Integration with external systems
```

---

## COMPONENT 4: FALLBACK MANAGER GENERATION

### Universal Generation Prompt

**Target**: Generate FallbackManager with geographic failover capabilities

```
You are a reliability engineer. Generate a complete FallbackManager that:

GEOGRAPHIC FAILOVER:
- Implement 6-region geographic support
- Create intelligent failover chains with latency optimization
- Add health monitoring with consecutive failure tracking
- Build provider diversity and redundancy

RELIABILITY FEATURES:
- Real-time health checks and status monitoring
- Smart fallback chain optimization by latency and health
- Automatic recovery and circuit breaker patterns
- Geographic load balancing and distribution

MONITORING & ALERTING:
- Comprehensive health status tracking
- Failure detection and automatic failover
- Performance monitoring and alerting
- Geographic performance analytics

CONFIGURATION MANAGEMENT:
- Dynamic fallback chain configuration
- Geographic preference and restriction handling
- Health check interval and threshold configuration
- Alert and notification setup

Generate with:
1. Geographic failover engine
2. Health monitoring system
3. Load balancing logic
4. Configuration management
5. Alerting and notification
6. Comprehensive testing scenarios
```

---

## COMPONENT 5: PROMPT ADAPTER GENERATION

### Universal Generation Prompt

**Target**: Generate PromptAdapter with token optimization

```
You are a prompt engineering specialist. Generate a complete PromptAdapter that:

PROMPT OPTIMIZATION:
- Implement provider-specific prompt formatting
- Add token usage optimization (70% compression target)
- Create model-specific prompt constraints and validation
- Build quality enhancement algorithms

TOKEN EFFICIENCY:
- Automatic prompt compression while maintaining meaning
- Token counting and estimation for different models
- Context window optimization
- Cost-effective prompt restructuring

MODEL COMPATIBILITY:
- Provider-specific template management
- Model capability adaptation
- Format conversion and normalization
- Validation and error handling

PERFORMANCE FEATURES:
- Caching for repeated prompt patterns
- Batch processing optimization
- Real-time token counting
- Performance monitoring and analytics

Generate with:
1. Prompt optimization engine
2. Token management system
3. Model compatibility layer
4. Validation and testing
5. Performance optimization
6. Configuration management
```

---

## INTEGRATION ADAPTERS GENERATION

### AIEngineAdapter Generation

```
You are an AI integration specialist. Generate AIEngineAdapter that:

PROVIDER AGNOSTIC INTERFACE:
- Unified interface for multiple AI providers
- Dynamic model registry management
- Real token counting and cost estimation
- Compliance validation and audit trails

PROVIDER SUPPORT:
- OpenAI GPT models integration
- Anthropic Claude models support
- Google Gemini/PaLM integration
- Local LLM and open-source model support
- OpenRouter and Together AI integration

RELIABILITY FEATURES:
- Rate limiting and quota management
- Automatic retry with exponential backoff
- Health monitoring and failover detection
- Error classification and handling

AUDIT & COMPLIANCE:
- Request/response logging and audit trails
- Compliance validation per request
- Data handling and privacy controls
- Geographic compliance enforcement

Generate with:
1. Provider abstraction layer
2. Model management system
3. Rate limiting and quotas
4. Error handling and retries
5. Audit and compliance
6. Testing and validation
```

### EventSystem Generation

```
You are an event-driven architecture specialist. Generate EventSystem that:

EVENT MANAGEMENT:
- Multiple transport layers (memory/Redis/Kafka/webhooks)
- Real-time alerting with configurable rules
- Analytics aggregation and reporting
- Buffered event processing with backpressure

MONITORING & ANALYTICS:
- Comprehensive event collection and processing
- Real-time dashboards and alerting
- Historical analytics and reporting
- Performance monitoring and optimization

COMPLIANCE TRACKING:
- Compliance event logging and audit trails
- Violation detection and alerting
- Regulatory reporting capabilities
- Data retention and privacy controls

SCALABILITY FEATURES:
- Horizontal scaling support
- Event batching and compression
- Queue management and backpressure
- High availability and fault tolerance

Generate with:
1. Event processing engine
2. Transport layer abstraction
3. Analytics and reporting
4. Compliance monitoring
5. Scalability and performance
6. Configuration and management
```

---

## SCAFFOLDING GENERATION

### Types & Error Handling Generation

```
You are a type system architect. Generate comprehensive scaffolding that:

TYPE DEFINITIONS:
- Framework-agnostic type system
- Comprehensive interface definitions
- Enum and constant declarations
- Generic type support for extensibility

ERROR MANAGEMENT:
- Hierarchical error classification system
- Configurable retry strategies with circuit breakers
- Compliance-aware error reporting
- Geographic failover on errors

VALIDATION & SAFETY:
- Input validation and sanitization
- Type safety enforcement
- Security controls and checks
- Audit logging and monitoring

EXTENSIBILITY:
- Plugin architecture support
- Custom type registration
- Extension points for customization
- Backward compatibility management

Generate with:
1. Complete type system
2. Error handling framework
3. Validation system
4. Extension mechanisms
5. Testing utilities
6. Documentation generation
```

---

## CONFIGURATION GENERATION

### Configuration Generator Prompt

```
You are a configuration management expert. Generate configuration system that:

CONFIGURATION MANAGEMENT:
- Multi-format support (JSON/YAML/TOML)
- Environment-specific configurations
- Hierarchical configuration merging
- Runtime configuration updates

VALIDATION & SAFETY:
- Schema validation for configurations
- Type checking and conversion
- Security validation for sensitive data
- Configuration integrity checks

DEPLOYMENT SUPPORT:
- Configuration templating and substitution
- Secret management integration
- Configuration distribution and synchronization
- Rollback and versioning support

MONITORING & AUDIT:
- Configuration change tracking
- Audit logging for configuration access
- Compliance monitoring for configuration changes
- Performance impact assessment

Generate with:
1. Configuration loader and parser
2. Validation and type checking
3. Environment management
4. Security and compliance
5. Monitoring and auditing
6. Deployment and operations
```

---

## TESTING FRAMEWORK GENERATION

### Comprehensive Testing Suite

```
You are a testing specialist. Generate testing framework that:

UNIT TESTING:
- Component-level unit tests
- Mock implementations for dependencies
- Edge case and error condition testing
- Performance benchmarking utilities

INTEGRATION TESTING:
- End-to-end routing pipeline testing
- Multi-provider integration testing
- Geographic failover scenario testing
- Cost optimization validation

COMPLIANCE TESTING:
- Compliance framework validation
- Data handling and privacy testing
- Audit trail verification
- Security control testing

PERFORMANCE TESTING:
- Load testing utilities
- Latency and throughput measurement
- Memory usage profiling
- Scalability testing tools

Generate with:
1. Unit test suites for all components
2. Integration test scenarios
3. Compliance test frameworks
4. Performance testing tools
5. Test data generation
6. CI/CD integration
```

---

## DEPLOYMENT & OPERATIONS

### Docker & Kubernetes Generation

```
You are a DevOps engineer. Generate deployment configurations that:

CONTAINERIZATION:
- Multi-stage Dockerfile optimization
- Security hardening and vulnerability scanning
- Resource limit configuration
- Health check implementation

ORCHESTRATION:
- Kubernetes deployment manifests
- Service mesh integration (Istio/Linkerd)
- Horizontal Pod Autoscaling configuration
- ConfigMap and Secret management

OBSERVABILITY:
- Prometheus metrics collection
- Grafana dashboard configurations
- Distributed tracing setup
- Log aggregation and analysis

SCALABILITY:
- Auto-scaling policies and triggers
- Load balancing configuration
- Database connection pooling
- Caching layer integration

Generate with:
1. Container configurations
2. Kubernetes manifests
3. Monitoring and observability
4. Scaling and performance
5. Security and compliance
6. Backup and disaster recovery
```

---

**These code generation prompts transform TRAE routing templates into production-ready
implementations for any programming language and framework. Each prompt includes specific adaptation
strategies, integration patterns, and production requirements.**</content>
