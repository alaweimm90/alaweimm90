# Installation Guide

Complete installation instructions for ATLAS across all supported platforms and environments.

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

Install ATLAS CLI globally using npm:

```bash
npm install -g @atlas/cli
```

Verify installation:

```bash
atlas --version
# Output: ATLAS CLI v1.0.0
```

### Method 2: Yarn Global Install

If you prefer yarn:

```bash
yarn global add @atlas/cli
```

### Method 3: NPX (No Global Install)

Use ATLAS without global installation:

```bash
npx @atlas/cli --version
# Or create an alias
alias atlas="npx @atlas/cli"
```

### Method 4: Docker

Run ATLAS in a Docker container:

```bash
# Pull the official image
docker pull atlasplatform/atlas:latest

# Run ATLAS commands
docker run --rm atlasplatform/atlas:latest --version
```

### Method 5: From Source

For development or custom builds:

```bash
# Clone the repository
git clone https://github.com/atlas-platform/atlas.git
cd atlas

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

# Install ATLAS
sudo npm install -g @atlas/cli

# Verify
atlas --version
```

#### CentOS/RHEL/Fedora

```bash
# Install Node.js
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs

# Install ATLAS
sudo npm install -g @atlas/cli
```

#### Arch Linux

```bash
# Install Node.js
sudo pacman -S nodejs npm

# Install ATLAS
sudo npm install -g @atlas/cli
```

### macOS Installation

#### Using Homebrew (Recommended)

```bash
# Install Node.js
brew install node

# Install ATLAS
npm install -g @atlas/cli
```

#### Using MacPorts

```bash
# Install Node.js
sudo port install nodejs18

# Install ATLAS
npm install -g @atlas/cli
```

### Windows Installation

#### Using Chocolatey

```bash
# Install Node.js
choco install nodejs

# Install ATLAS
npm install -g @atlas/cli
```

#### Using Scoop

```bash
# Install Node.js
scoop install nodejs

# Install ATLAS
npm install -g @atlas/cli
```

#### Manual Installation

1. Download Node.js from [nodejs.org](https://nodejs.org/)
2. Install Node.js
3. Open Command Prompt or PowerShell as Administrator
4. Run: `npm install -g @atlas/cli`

---

## Post-Installation Setup

### 1. Verify Installation

```bash
# Check version
atlas --version

# Check help
atlas --help

# Check system compatibility
atlas doctor
```

### 2. Configure Shell Auto-Completion

#### Bash

Add to `~/.bashrc`:

```bash
# ATLAS CLI completion
if command -v atlas &> /dev/null; then
  eval "$(atlas completion bash)"
fi
```

#### Zsh

Add to `~/.zshrc`:

```bash
# ATLAS CLI completion
if command -v atlas &> /dev/null; then
  eval "$(atlas completion zsh)"
fi
```

#### Fish

```bash
# ATLAS CLI completion
atlas completion fish > ~/.config/fish/completions/atlas.fish
```

#### PowerShell

```powershell
# Add to PowerShell profile
atlas completion powershell >> $PROFILE
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

# ATLAS Configuration
ATLAS_LOG_LEVEL=info
ATLAS_CONFIG_DIR=./.atlas
EOF
```

### 4. Initialize ATLAS in Your Project

```bash
# Navigate to your project
cd your-project-directory

# Initialize ATLAS
atlas init

# This creates:
# - .atlas/ directory
# - .atlas/config.json
# - .atlas/agents/ directory
# - .atlas/tasks/ directory
```

---

## Enterprise Installation

### Docker Compose Setup

For enterprise deployments, use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  atlas:
    image: atlasplatform/atlas:latest
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - NODE_ENV=production
      - ATLAS_CONFIG_DIR=/app/config
    ports:
      - "3000:3000"
    restart: unless-stopped
```

```bash
# Start ATLAS
docker-compose up -d
```

### Kubernetes Deployment

For Kubernetes deployments:

```yaml
# atlas-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atlas
spec:
  replicas: 3
  selector:
    matchLabels:
      app: atlas
  template:
    metadata:
      labels:
        app: atlas
    spec:
      containers:
      - name: atlas
        image: atlasplatform/atlas:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: config-volume
        configMap:
          name: atlas-config
```

### CI/CD Integration

Integrate ATLAS into your CI/CD pipeline:

```yaml
# .github/workflows/atlas.yml
name: ATLAS Code Analysis
on: [push, pull_request]

jobs:
  atlas-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install -g @atlas/cli
      - run: atlas init
      - run: atlas agent register claude-sonnet-4 --api-key ${{ secrets.ANTHROPIC_API_KEY }}
      - run: atlas analyze repo . --format json
```

---

## Configuration

### Global Configuration

Configure ATLAS globally:

```bash
# Set default log level
atlas config set log.level info

# Set default timeout
atlas config set task.timeout 300

# Set cost limits
atlas config set cost.max_per_task 1.00
atlas config set cost.max_per_day 50.00
```

### Project Configuration

Configure ATLAS for a specific project:

```bash
# Initialize project config
atlas init

# Set project-specific settings
atlas config set project.name "My Project"
atlas config set project.language typescript
atlas config set project.framework nextjs
```

### Configuration File

The `.atlas/config.json` file contains:

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
    "file": "./.atlas/logs/atlas.log"
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

**"atlas: command not found"**
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
echo "🔍 ATLAS Installation Verification"
echo "=================================="

# Check Node.js
echo "Node.js version: $(node --version)"
echo "NPM version: $(npm --version)"

# Check ATLAS
if command -v atlas &> /dev/null; then
    echo "✅ ATLAS CLI installed: $(atlas --version)"
else
    echo "❌ ATLAS CLI not found"
    exit 1
fi

# Check permissions
if atlas --help &> /dev/null; then
    echo "✅ ATLAS CLI functional"
else
    echo "❌ ATLAS CLI not functional"
    exit 1
fi

echo "🎉 Installation verified successfully!"
```

---

## Next Steps

After successful installation:

1. **[Quick Start](quick-start.md)** - Get up and running in 5 minutes
2. **[Register Agents](first-tasks.md#registering-agents)** - Add AI agents to ATLAS
3. **[Submit Tasks](first-tasks.md#submitting-tasks)** - Start using ATLAS for development
4. **[Configuration](configuration.md)** - Advanced configuration options

---

## Support

- **Documentation**: [Full Documentation](../README.md)
- **Community**: [Discord](https://discord.gg/atlas-platform)
- **Issues**: [GitHub Issues](https://github.com/atlas-platform/atlas/issues)
- **Enterprise**: [Contact Sales](mailto:sales@atlas-platform.com)</instructions>