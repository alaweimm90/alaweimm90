# Technology Stack & Dependencies

## Programming Languages & Versions

### Primary Languages

- **TypeScript/JavaScript** - Node.js >=20, ES modules, modern async/await patterns
- **Python** - 3.14+ with scientific computing stack (NumPy, JAX, PyTorch)
- **C++** - High-performance computing components and GPU kernels
- **Rust** - Systems programming and performance-critical components
- **Shell/Bash** - Automation scripts and CI/CD workflows

### Domain-Specific Languages

- **YAML** - Configuration management and CI/CD pipelines
- **JSON** - API schemas and configuration files
- **Rego** - Open Policy Agent (OPA) policy definitions
- **HCL** - Terraform infrastructure as code

## Core Frameworks & Libraries

### Scientific Computing

- **JAX** - GPU-accelerated numerical computing and automatic differentiation
- **PyTorch** - Deep learning and neural network frameworks
- **NumPy** - Fundamental array computing and linear algebra
- **SciPy** - Scientific computing algorithms and optimization
- **CUDA** - GPU programming and parallel computation

### Web & API Development

- **React** - Frontend user interfaces and interactive visualizations
- **Next.js** - Full-stack web applications with SSR/SSG
- **FastAPI** - High-performance Python web APIs
- **Three.js/WebGL** - 3D graphics and physics visualizations
- **Express.js** - Node.js web server framework

### AI & Machine Learning

- **Transformers** - Hugging Face transformer models and pipelines
- **LangChain** - LLM application development framework
- **OpenAI API** - GPT model integration and AI workflows
- **MCP (Model Context Protocol)** - Standardized AI tool communication

## Build Systems & Package Management

### Node.js Ecosystem

```json
{
  "engines": { "node": ">=20" },
  "type": "module",
  "packageManager": "npm"
}
```

### Python Environment

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
requires-python = ">=3.14"
```

### Development Tools

- **TypeScript** - Static type checking and modern JavaScript features
- **ESLint** - Code linting and style enforcement
- **Prettier** - Code formatting and consistency
- **Vitest** - Fast unit testing framework
- **Husky** - Git hooks and pre-commit validation

## Infrastructure & Deployment

### Containerization

- **Docker** - Application containerization and multi-stage builds
- **Docker Compose** - Local development environment orchestration
- **Kubernetes** - Production container orchestration and scaling

### Cloud & DevOps

- **Terraform** - Infrastructure as code and cloud resource management
- **Helm** - Kubernetes package management and templating
- **GitHub Actions** - CI/CD pipelines and workflow automation
- **Prometheus/Grafana** - Monitoring, metrics, and observability

### Security & Compliance

- **OPA (Open Policy Agent)** - Policy enforcement and governance
- **Trivy** - Container and dependency vulnerability scanning
- **SAST Tools** - Static application security testing
- **Secrets Management** - Secure credential handling and rotation

## Development Commands & Scripts

### Core Development

```bash
# TypeScript development
npm run type-check          # Type checking
npm run lint                # Code linting
npm run format              # Code formatting
npm run test                # Unit testing

# DevOps automation
npm run devops              # CLI tool access
npm run governance          # Policy enforcement
npm run orchestrate         # AI workflow management

# Scientific computing
npm run ORCHEX               # ORCHEX research agent
npm run ai:start            # AI orchestration
npm run codemap             # Code analysis
```

### Specialized Workflows

```bash
# Template management
npm run devops:list         # List available templates
npm run devops:builder      # Apply templates
npm run devops:bootstrap    # Initialize new projects

# AI orchestration
npm run ai:complete         # Complete AI tasks
npm run ai:context          # Context management
npm run ai:metrics          # Performance metrics
npm run ai:dashboard        # Monitoring dashboard
```

## Configuration Management

### Environment Configuration

- **`.env.example`** - Environment variable templates
- **`tsconfig.json`** - TypeScript compiler configuration
- **`eslint.config.js`** - ESLint rules and plugins
- **`.prettierrc`** - Code formatting preferences

### AI Tool Integration

- **`.ai/`** directory contains configurations for:
  - Aider, Claude, Cursor, Continue, Copilot
  - Amazon Q, Gemini, Windsurf, Cline
  - MCP servers and workflow definitions

### Governance & Policies

- **`.github/workflows/`** - Automated CI/CD and governance
- **`.metaHub/policies/`** - OPA policy definitions
- **`.pre-commit-config.yaml`** - Git hook configurations

## Performance Optimization

### GPU Acceleration

- **CUDA Toolkit** - NVIDIA GPU programming
- **JAX** - XLA compilation for TPU/GPU optimization
- **Memory Management** - Efficient array operations and caching

### Build Optimization

- **Tree Shaking** - Dead code elimination
- **Code Splitting** - Lazy loading and bundle optimization
- **Caching Strategies** - Build artifact and dependency caching

### Monitoring & Profiling

- **Performance Metrics** - Real-time application monitoring
- **Resource Usage** - CPU, memory, and GPU utilization tracking
- **Bottleneck Analysis** - Profiling and optimization recommendations
