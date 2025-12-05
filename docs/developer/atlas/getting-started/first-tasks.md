# First Tasks with ORCHEX

Learn how to perform your first tasks with ORCHEX, from registering agents to submitting various types of development tasks.

---

## Prerequisites

Before starting, ensure you have:

- [ORCHEX CLI installed](../installation.md)
- [ORCHEX initialized](quick-start.md#step-2-initialize-ORCHEX) in your project
- API keys for at least one AI provider

---

## Registering Agents

### Understanding Agents

ORCHEX agents are AI models configured for specific tasks. Each agent has:

- **Provider**: Anthropic (Claude), OpenAI (GPT), Google (Gemini), or Local
- **Capabilities**: What the agent can do (code generation, review, debugging, etc.)
- **Constraints**: Rate limits, cost limits, token limits

### Registering Your First Agent

#### Claude (Anthropic) - Recommended for Code Tasks

```bash
ORCHEX agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --model claude-sonnet-4.5 \
  --capabilities code_generation,code_review,refactoring,debugging,documentation \
  --max-tokens 200000 \
  --cost-per-1k-tokens 0.003
```

#### GPT-4 (OpenAI) - Good for Complex Reasoning

```bash
ORCHEX agent register gpt-4-turbo \
  --name "GPT-4 Turbo" \
  --provider openai \
  --model gpt-4-turbo \
  --capabilities code_generation,code_review,architecture,security_analysis \
  --max-tokens 128000 \
  --cost-per-1k-tokens 0.01
```

#### Gemini (Google) - Fast and Cost-Effective

```bash
ORCHEX agent register gemini-pro \
  --name "Gemini Pro" \
  --provider google \
  --model gemini-pro \
  --capabilities code_generation,code_review,debugging \
  --max-tokens 32000 \
  --cost-per-1k-tokens 0.00025
```

### Multi-Agent Setup

For optimal performance, register multiple agents:

```bash
# Register primary agent (Claude for code generation)
ORCHEX agent register claude-sonnet-4 \
  --name "Claude Sonnet 4" \
  --provider anthropic \
  --capabilities code_generation,refactoring \
  --priority primary

# Register secondary agent (GPT-4 for complex tasks)
ORCHEX agent register gpt-4-turbo \
  --name "GPT-4 Turbo" \
  --provider openai \
  --capabilities architecture,security_analysis \
  --priority secondary

# Register tertiary agent (Gemini for fast tasks)
ORCHEX agent register gemini-pro \
  --name "Gemini Pro" \
  --provider google \
  --capabilities code_review,debugging \
  --priority tertiary
```

### Verifying Agent Registration

```bash
# List all registered agents
ORCHEX agent list

# Check agent health
ORCHEX agent health claude-sonnet-4

# View agent details
ORCHEX agent show claude-sonnet-4
```

---

## Submitting Tasks

### Task Types

ORCHEX supports various task types:

| Task Type           | Description                       | Best For                       |
| ------------------- | --------------------------------- | ------------------------------ |
| `code_generation`   | Generate new code                 | New features, boilerplate      |
| `code_review`       | Review existing code              | Quality assurance, bug finding |
| `refactoring`       | Improve code structure            | Technical debt reduction       |
| `debugging`         | Find and fix bugs                 | Error resolution               |
| `documentation`     | Generate documentation            | API docs, READMEs              |
| `testing`           | Write tests                       | Test coverage improvement      |
| `architecture`      | Design system architecture        | System planning                |
| `security_analysis` | Security vulnerability assessment | Security reviews               |

### Basic Task Submission

#### Code Generation Task

```bash
ORCHEX task submit \
  --type code_generation \
  --description "Create a REST API endpoint for user authentication in Express.js" \
  --context language=javascript,framework=express,database=mongodb \
  --priority high
```

#### Code Review Task

```bash
ORCHEX task submit \
  --type code_review \
  --description "Review this authentication module for security vulnerabilities and best practices" \
  --file-path src/auth.js \
  --priority medium
```

#### Debugging Task

```bash
ORCHEX task submit \
  --type debugging \
  --description "Debug this error: 'TypeError: Cannot read property 'map' of undefined'" \
  --file-path src/components/UserList.tsx \
  --error-log "error.log" \
  --priority high
```

### Advanced Task Options

#### With File Context

```bash
ORCHEX task submit \
  --type refactoring \
  --description "Refactor this function to reduce complexity and improve readability" \
  --file-path src/utils/helpers.js \
  --line-range 15-45 \
  --context "This function handles data validation and transformation"
```

#### With Repository Context

```bash
ORCHEX task submit \
  --type architecture \
  --description "Design a microservices architecture for this e-commerce platform" \
  --repo-path . \
  --requirements "scalability,high-availability,cost-efficiency" \
  --constraints "kubernetes-deployment,react-frontend"
```

#### With Cost and Time Limits

```bash
ORCHEX task submit \
  --type code_generation \
  --description "Create a complete user management system" \
  --max-cost 2.50 \
  --timeout 600 \
  --priority medium
```

### Monitoring Task Progress

```bash
# Check task status
ORCHEX task status task_abc123

# List running tasks
ORCHEX task list --status running

# List completed tasks
ORCHEX task list --status completed --limit 5

# Get task result
ORCHEX task result task_abc123
```

---

## Task Examples

### Example 1: Building a REST API

```bash
# 1. Generate the main API structure
ORCHEX task submit \
  --type code_generation \
  --description "Create Express.js API with basic CRUD operations for users" \
  --context language=javascript,framework=express,database=mongodb

# 2. Add authentication middleware
ORCHEX task submit \
  --type code_generation \
  --description "Add JWT authentication middleware to the API" \
  --file-path src/middleware/auth.js

# 3. Create user model
ORCHEX task submit \
  --type code_generation \
  --description "Create Mongoose user model with validation" \
  --file-path src/models/User.js

# 4. Add input validation
ORCHEX task submit \
  --type code_generation \
  --description "Add input validation using Joi for all API endpoints" \
  --file-path src/validation/userValidation.js
```

### Example 2: Code Review Workflow

```bash
# Review a pull request
ORCHEX task submit \
  --type code_review \
  --description "Review this PR for code quality, security, and best practices" \
  --pr-url https://github.com/org/repo/pull/123 \
  --focus "security,performance,maintainability"

# Review specific files
ORCHEX task submit \
  --type code_review \
  --description "Review authentication logic for security vulnerabilities" \
  --files src/auth.js,src/middleware/auth.js,test/auth.test.js

# Review entire module
ORCHEX task submit \
  --type code_review \
  --description "Comprehensive review of the payment processing module" \
  --directory src/payment/
```

### Example 3: Refactoring Session

```bash
# Analyze code complexity first
ORCHEX analyze file src/utils/helpers.js --metrics complexity

# Refactor complex function
ORCHEX task submit \
  --type refactoring \
  --description "Break down this 50-line function into smaller, focused functions" \
  --file-path src/utils/helpers.js \
  --function-name processUserData

# Extract constants
ORCHEX task submit \
  --type refactoring \
  --description "Extract magic numbers and strings into named constants" \
  --file-path src/config/constants.js

# Improve error handling
ORCHEX task submit \
  --type refactoring \
  --description "Add proper error handling and logging throughout the module" \
  --file-path src/services/apiService.js
```

### Example 4: Testing Workflow

```bash
# Generate unit tests
ORCHEX task submit \
  --type testing \
  --description "Write comprehensive unit tests for the user service" \
  --file-path src/services/userService.js \
  --test-framework jest

# Generate integration tests
ORCHEX task submit \
  --type testing \
  --description "Create integration tests for user registration flow" \
  --files src/routes/auth.js,src/services/userService.js \
  --test-type integration

# Generate API tests
ORCHEX task submit \
  --type testing \
  --description "Write API tests for all user endpoints" \
  --api-spec src/routes/users.js \
  --test-framework supertest
```

---

## Working with Results

### Viewing Task Results

```bash
# Get detailed results
ORCHEX task result task_abc123 --format detailed

# Export results to file
ORCHEX task result task_abc123 --output result.json

# View results in browser (if available)
ORCHEX task result task_abc123 --web
```

### Understanding Result Structure

Task results typically include:

```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "agent_id": "claude-sonnet-4",
  "result": {
    "code": "...generated code...",
    "explanation": "Explanation of the implementation",
    "best_practices": ["List of recommendations"],
    "potential_issues": ["Any concerns or warnings"],
    "test_suggestions": ["Suggested test cases"]
  },
  "metadata": {
    "tokens_used": 1250,
    "cost_usd": 0.004,
    "duration_ms": 3200
  }
}
```

### Applying Results

```bash
# Apply generated code to file
ORCHEX task apply task_abc123 --file-path src/new-feature.js

# Create a new file with results
ORCHEX task apply task_abc123 --create-file src/components/NewComponent.tsx

# Apply with backup
ORCHEX task apply task_abc123 --file-path src/api.js --backup
```

---

## Error Handling and Retries

### Handling Failed Tasks

```bash
# Check failure reason
ORCHEX task result task_failed_123

# Retry with same agent
ORCHEX task retry task_failed_123

# Retry with different agent
ORCHEX task retry task_failed_123 --agent gpt-4-turbo

# Force retry ignoring previous failures
ORCHEX task retry task_failed_123 --force
```

### Common Failure Scenarios

**Rate Limit Exceeded**

```bash
# Wait and retry
sleep 60
ORCHEX task retry task_abc123
```

**Token Limit Exceeded**

```bash
# Split into smaller tasks
ORCHEX task submit --type code_generation --description "Part 1 of user management" --max-tokens 1000
ORCHEX task submit --type code_generation --description "Part 2 of user management" --max-tokens 1000
```

**Network Issues**

```bash
# Retry with longer timeout
ORCHEX task retry task_abc123 --timeout 600
```

---

## Best Practices for First Tasks

### 1. Start Small

Begin with simple, focused tasks:

- Generate a single function
- Review a small module
- Fix a specific bug

### 2. Provide Clear Context

The more context you provide, the better the results:

- Specify programming language and framework
- Include relevant file paths
- Mention existing patterns or conventions
- Describe the expected behavior

### 3. Use Appropriate Task Types

Match the task type to your goal:

- `code_generation` for new features
- `code_review` for quality assurance
- `debugging` for error resolution
- `refactoring` for code improvement

### 4. Monitor Costs

Keep track of API costs:

```bash
ORCHEX metrics costs --period 24h
ORCHEX config set cost.max_per_task 1.00
```

### 5. Review and Iterate

Always review AI-generated code:

- Check for correctness
- Verify it follows your patterns
- Test the implementation
- Make adjustments as needed

---

## Next Steps

Now that you've completed your first tasks:

1. **[Configuration](configuration.md)** - Learn advanced configuration options
2. **[Multi-Agent Orchestration](../architecture/components.md#task-router)** - Understand how ORCHEX selects agents
3. **[Repository Analysis](../cli/analysis.md)** - Analyze entire codebases
4. **[Integration Guides](../integration/)** - Integrate ORCHEX into your workflow

---

## Getting Help

- **Task Status**: `ORCHEX task status <task-id>`
- **Agent Health**: `ORCHEX agent health <agent-id>`
- **System Logs**: `ORCHEX logs --tail`
- **Help**: `ORCHEX task --help` or `ORCHEX --help`

For issues or questions:

- [Troubleshooting Guide](../troubleshooting/common-issues.md)
- [Community Forum](https://community.orchex-platform.com)
- [GitHub Issues](https://github.com/ORCHEX-platform/ORCHEX/issues)</instructions>
