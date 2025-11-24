# TRAE Repository Analysis Prompt

## Identify Current AI Usage Patterns & Transformation Readiness

**Purpose**: Analyze any repository to identify existing AI integrations, usage patterns, and
transformation requirements for implementing TRAE-level routing capabilities.

**Prerequisites**:

- Access to repository source code
- Understanding of current AI provider integrations
- Knowledge of existing cost tracking (if any)

---

## ANALYSIS FRAMEWORK

You are an expert AI systems analyst specializing in LLM routing and cost optimization. Your task is
to perform a comprehensive analysis of the target repository to identify:

1. **Current AI Integration Patterns**
2. **Cost and Usage Analysis**
3. **Architecture Assessment**
4. **Transformation Complexity**
5. **Compliance Requirements**

### Step 1: Codebase Analysis

Examine the repository using these search patterns:

**AI Provider Detection**:

```
# OpenAI
openai|OpenAI|gpt-|GPT-
# Anthropic
anthropic|Anthropic|claude|Claude
# Google AI
gemini|Gemini|palm|PaLM|vertex|Vertex
# Local/Other
ollama|Ollama|local.*llm|together|Together|openrouter|OpenRouter
```

**API Call Patterns**:

```
# Direct API calls
\.create\(|\.generate\(|\.complete\(
# SDK usage
import.*openai|import.*anthropic|import.*google
# HTTP requests to AI endpoints
api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com
```

**Cost Tracking Indicators**:

```
# Cost calculation
cost.*=|price.*=|billing|usage.*token
# Budget limits
budget|limit|threshold|max.*cost
# Monitoring
metrics|analytics|tracking|monitoring
```

### Step 2: Integration Assessment

For each identified AI integration, document:

**Provider Details**:

- Provider name and SDK version
- Authentication method (API key, OAuth, etc.)
- Rate limits and quotas
- Geographic regions used

**Usage Patterns**:

- Function names making AI calls
- Request frequency (per minute/hour/day)
- Average request size (tokens)
- Response handling patterns

**Cost Structure**:

- Current estimated monthly spend
- Cost per request/token
- Budget constraints or alerts
- Cost optimization attempts

### Step 3: Architecture Evaluation

Assess current architecture against TRAE requirements:

**Modularity**:

- Are AI calls centralized or scattered?
- Dependency injection patterns?
- Configuration management?

**Error Handling**:

- Retry logic implementation?
- Fallback mechanisms?
- Error classification?

**Monitoring**:

- Logging of AI requests?
- Performance metrics?
- Failure tracking?

### Step 4: Compliance & Security Analysis

Identify compliance requirements:

**Data Classification**:

```
# PII detection
email|phone|address|ssn|social|personal
# Health data
medical|health|patient|diagnosis|treatment
# Financial data
credit|card|payment|transaction|bank
```

**Geographic Requirements**:

- Data residency requirements
- Cross-border transfer restrictions
- Regional compliance (GDPR, HIPAA, etc.)

**Security Controls**:

- Encryption of API keys
- Audit logging requirements
- Access control patterns

### Step 5: Transformation Complexity Assessment

Rate transformation complexity on a 1-5 scale:

**Low Complexity (1-2)**:

- Single AI provider
- Centralized AI calls
- Simple request patterns
- No compliance requirements

**Medium Complexity (3)**:

- Multiple providers
- Scattered AI calls
- Basic cost tracking
- Standard compliance needs

**High Complexity (4-5)**:

- Complex multi-provider setup
- Enterprise compliance requirements
- Distributed architecture
- Real-time cost optimization needed

---

## ANALYSIS OUTPUT FORMAT

Provide your analysis in this structured format:

### Executive Summary

- **Repository**: [Name/URL]
- **Analysis Date**: [Date]
- **Current AI Spend**: $[estimated monthly]
- **Transformation Complexity**: [1-5 scale]
- **Estimated Implementation Time**: [weeks/months]

### Current AI Landscape

#### Primary Providers

| Provider  | Usage Frequency | Est. Monthly Cost | Integration Pattern |
| --------- | --------------- | ----------------- | ------------------- |
| OpenAI    | High            | $X,XXX            | Direct API calls    |
| Anthropic | Medium          | $XXX              | SDK integration     |

#### Usage Patterns

- **Total AI Functions**: X identified
- **Request Volume**: X requests/day estimated
- **Average Request Size**: X tokens
- **Peak Usage Times**: [time periods]

### Architecture Assessment

#### Strengths

- [List current architectural strengths]

#### Gaps for TRAE Implementation

- [List areas needing transformation]

### Compliance Requirements

#### Identified Frameworks

- [ ] GDPR (European data protection)
- [ ] HIPAA (Healthcare data)
- [ ] SOC 2 (Security/compliance)
- [ ] PCI DSS (Payment data)

#### Geographic Restrictions

- [List any data residency requirements]

### Transformation Roadmap

#### Phase 1: Foundation (Week 1-2)

- [Specific implementation steps]

#### Phase 2: Core Routing (Week 3-4)

- [Routing implementation steps]

#### Phase 3: Optimization (Week 5-6)

- [Cost and performance optimization]

#### Phase 4: Production (Week 7-8)

- [Production deployment and monitoring]

### Risk Assessment

#### High Risk Items

- [List potential blockers]

#### Mitigation Strategies

- [How to address risks]

### Success Metrics

#### Cost Reduction Target

- **Current Spend**: $X,XXX/month
- **TRAE Target**: $XXX/month (7-10x reduction)
- **Timeline**: [timeframe for achievement]

#### Quality Maintenance

- **Current Success Rate**: X%
- **Target**: >95% with TRAE routing

---

## VALIDATION CHECKLIST

Before completing analysis:

- [ ] All AI provider integrations identified
- [ ] Cost estimates validated against actual usage
- [ ] Compliance requirements documented
- [ ] Architecture gaps identified
- [ ] Transformation complexity accurately assessed
- [ ] Implementation timeline realistic
- [ ] Risk mitigation strategies defined

---

## NEXT STEPS

After completing this analysis, proceed to:

1. **Implementation Roadmap Prompt** - Create detailed transformation plan
2. **Code Generation Prompts** - Generate TRAE routing components
3. **Configuration Prompts** - Set up compliance and cost controls
4. **Testing Prompts** - Validate routing implementation

---

**Note**: This analysis provides the foundation for transforming any repository into having
TRAE-level routing capabilities. The framework-agnostic approach ensures compatibility with any
programming language and AI provider ecosystem.</content>
