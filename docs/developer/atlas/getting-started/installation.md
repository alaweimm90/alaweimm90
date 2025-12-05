# Installation Guide

Complete installation instructions for ORCHEX across all supported platforms and environments.

---

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS (10.15+), Windows (10+)
- **Node.js**: Version 16.0.0 or higher
- **Memory**: 4GB RAM
- **Storage**: 1GB free disk space
- **Network**: Internet connection for AI API access

### Recommended Requirements

- **Operating System**: Linux or macOS
- **Node.js**: Version 18.0.0 or higher
- **Memory**: 8GB RAM
- **Storage**: 5GB free disk space
- **Network**: High-speed internet (100Mbps+) for optimal performance

---

## Installation Methods

### Method 1: NPM Global Install (Recommended)

Install ORCHEX CLI globally using npm:

```bash
npm install -g @ORCHEX/cli
```

Verify installation:

```bash
ORCHEX --version
# Output: ORCHEX CLI v1.0.0
```

### Method 2: Yarn Global Install

If you prefer yarn:

```bash
yarn global add @ORCHEX/cli
```

### Method 3: NPX (No Global Install)

Use ORCHEX without global installation:

```bash
npx @ORCHEX/cli --version
# Or create an alias
alias ORCHEX="npx @ORCHEX/cli"
```

### Method 4: Docker

Run ORCHEX in a Docker container:

```bash
# Pull the official image
docker pull atlasplatform/ORCHEX:latest

# Run ORCHEX commands
docker run --rm atlasplatform/ORCHEX:latest --version
```

### Method 5: From Source

For development or custom builds:

```bash
# Clone the repository
git clone https://github.com/ORCHEX-platform/ORCHEX.git
cd ORCHEX

# Install dependencies
npm install

# Build the CLI
npm run build

# Link globally (optional)
npm link
```

---

## Platform-Specific Instructions

### Linux Installation

#### Ubuntu/Debian

```bash
# Install Node.js (if not already installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install ORCHEX
sudo npm install -g @ORCHEX/cli

# Verify
ORCHEX --version
```

#### CentOS/RHEL/Fedora

```bash
# Install Node.js
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs

# Install ORCHEX
sudo npm install -g @ORCHEX/cli
```

#### Arch Linux

```bash
# Install Node.js
sudo pacman -S nodejs npm

# Install ORCHEX
sudo npm install -g @ORCHEX/cli
```

### macOS Installation

#### Using Homebrew (Recommended)

```bash
# Install Node.js
brew install node

# Install ORCHEX
npm install -g @ORCHEX/cli
```

#### Using MacPorts

```bash
# Install Node.js
sudo port install nodejs18

# Install ORCHEX
npm install -g @ORCHEX/cli
```

### Windows Installation

#### Using Chocolatey

```bash
# Install Node.js
choco install nodejs

# Install ORCHEX
npm install -g @ORCHEX/cli
```

#### Using Scoop

```bash
# Install Node.js
scoop install nodejs

# Install ORCHEX
npm install -g @ORCHEX/cli
```

#### Manual Installation

1. Download Node.js from [nodejs.org](https://nodejs.org/)
2. Install Node.js
3. Open Command Prompt or PowerShell as Administrator
4. Run: `npm install -g @ORCHEX/cli`

---

## Post-Installation Setup

### 1. Verify Installation

```bash
# Check version
ORCHEX --version

# Check help
ORCHEX --help

# Check system compatibility
ORCHEX doctor
```

### 2. Configure Shell Auto-Completion

#### Bash

Add to `~/.bashrc`:

```bash
# ORCHEX CLI completion
if command -v ORCHEX &> /dev/null; then
  eval "$(ORCHEX completion bash)"
fi
```

#### Zsh

Add to `~/.zshrc`:

```bash
# ORCHEX CLI completion
if command -v ORCHEX &> /dev/null; then
  eval "$(ORCHEX completion zsh)"
fi
```

#### Fish

```bash
# ORCHEX CLI completion
ORCHEX completion fish > ~/.config/fish/completions/ORCHEX.fish
```

#### PowerShell

```powershell
# Add to PowerShell profile
ORCHEX completion powershell >> $PROFILE
```

### 3. Environment Variables

Set up environment variables for API keys:

```bash
# Create .env file in your project
cat > .env << EOF
# AI Provider API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# ORCHEX Configuration
ATLAS_LOG_LEVEL=info
ATLAS_CONFIG_DIR=./.orchex
EOF
```

### 4. Initialize ORCHEX in Your Project

```bash
# Navigate to your project
cd your-project-directory

# Initialize ORCHEX
ORCHEX init

# This creates:
# - .orchex/ directory
# - .orchex/config.json
# - .orchex/agents/ directory
# - .orchex/tasks/ directory
```

---

## Enterprise Installation

### Docker Compose Setup

For enterprise deployments, use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  ORCHEX:
    image: atlasplatform/ORCHEX:latest
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - NODE_ENV=production
      - ATLAS_CONFIG_DIR=/app/config
    ports:
      - '3000:3000'
    restart: unless-stopped
```

```bash
# Start ORCHEX
docker-compose up -d
```

### Kubernetes Deployment

For Kubernetes deployments:

```yaml
# ORCHEX-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ORCHEX
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ORCHEX
  template:
    metadata:
      labels:
        app: ORCHEX
    spec:
      containers:
        - name: ORCHEX
          image: atlasplatform/ORCHEX:latest
          ports:
            - containerPort: 3000
          env:
            - name: NODE_ENV
              value: 'production'
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
      volumes:
        - name: config-volume
          configMap:
            name: ORCHEX-config
```

### CI/CD Integration

Integrate ORCHEX into your CI/CD pipeline:

```yaml
# .github/workflows/ORCHEX.yml
name: ORCHEX Code Analysis
on: [push, pull_request]

jobs:
  ORCHEX-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install -g @ORCHEX/cli
      - run: ORCHEX init
      - run: ORCHEX agent register claude-sonnet-4 --api-key ${{ secrets.ANTHROPIC_API_KEY }}
      - run: ORCHEX analyze repo . --format json
```

---

## Configuration

### Global Configuration

Configure ORCHEX globally:

```bash
# Set default log level
ORCHEX config set log.level info

# Set default timeout
ORCHEX config set task.timeout 300

# Set cost limits
ORCHEX config set cost.max_per_task 1.00
ORCHEX config set cost.max_per_day 50.00
```

### Project Configuration

Configure ORCHEX for a specific project:

```bash
# Initialize project config
ORCHEX init

# Set project-specific settings
ORCHEX config set project.name "My Project"
ORCHEX config set project.language typescript
ORCHEX config set project.framework nextjs
```

### Configuration File

The `.orchex/config.json` file contains:

```json
{
  "version": "1.0.0",
  "project": {
    "name": "My Project",
    "language": "typescript",
    "framework": "nextjs"
  },
  "agents": {
    "default_provider": "anthropic",
    "fallback_enabled": true
  },
  "tasks": {
    "timeout": 300,
    "max_retries": 3
  },
  "cost": {
    "max_per_task": 1.0,
    "max_per_day": 50.0
  },
  "logging": {
    "level": "info",
    "file": "./.orchex/logs/ORCHEX.log"
  }
}
```

---

## Troubleshooting Installation

### Common Issues

**"npm ERR! code EACCES"**

```bash
# Fix permissions
sudo chown -R $(whoami) ~/.npm
# Or use nvm
```

**"ORCHEX: command not found"**

```bash
# Check PATH
echo $PATH
# Add npm global bin to PATH
export PATH="$(npm config get prefix)/bin:$PATH"
```

**"Node.js version too old"**

```bash
# Update Node.js
npm install -g n
sudo n latest
```

**"Permission denied" on Linux/macOS**

```bash
# Fix npm permissions
sudo chown -R $(whoami) $(npm config get prefix)/{lib/node_modules,bin,share}
```

### Verification Script

Run this script to verify your installation:

```bash
#!/bin/bash
echo "🔍 ORCHEX Installation Verification"
echo "=================================="

# Check Node.js
echo "Node.js version: $(node --version)"
echo "NPM version: $(npm --version)"

# Check ORCHEX
if command -v ORCHEX &> /dev/null; then
    echo "✅ ORCHEX CLI installed: $(ORCHEX --version)"
else
    echo "❌ ORCHEX CLI not found"
    exit 1
fi

# Check permissions
if ORCHEX --help &> /dev/null; then
    echo "✅ ORCHEX CLI functional"
else
    echo "❌ ORCHEX CLI not functional"
    exit 1
fi

echo "🎉 Installation verified successfully!"
```

---

## Next Steps

After successful installation:

1. **[Quick Start](quick-start.md)** - Get up and running in 5 minutes
2. **[Register Agents](first-tasks.md#registering-agents)** - Add AI agents to ORCHEX
3. **[Submit Tasks](first-tasks.md#submitting-tasks)** - Start using ORCHEX for development
4. **[Configuration](configuration.md)** - Advanced configuration options

---

## Support

- **Documentation**: [Full Documentation](../README.md)
- **Community**: [Discord](https://discord.gg/ORCHEX-platform)
- **Issues**: [GitHub Issues](https://github.com/ORCHEX-platform/ORCHEX/issues)
- **Enterprise**: [Contact Sales](mailto:sales@ORCHEX-platform.com)</instructions>
