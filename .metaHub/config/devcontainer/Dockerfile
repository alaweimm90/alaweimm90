# Use the official Node.js 20 image as the base
FROM mcr.microsoft.com/devcontainers/javascript-node:1-20

# Install essential tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Install global Node packages
RUN npm install -g turbo typescript ts-node

# Install MCP SDK and core MCP servers
RUN npm install -g @modelcontextprotocol/sdk
RUN npm install -g \
  @modelcontextprotocol/server-filesystem \
  @modelcontextprotocol/server-git \
  @modelcontextprotocol/server-fetch

# Set environment variables for MCPs
ENV MCP_SERVERS="filesystem,git,fetch"
ENV WORKSPACE_ROOT="/workspace"

# Set up non-root user
ARG USERNAME=node
USER $USERNAME

# Set up workspace
WORKDIR /workspace
