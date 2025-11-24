# Development Container Setup

This directory contains configuration for the VS Code Dev Container.

## Features

- Node.js 20 with pnpm
- TypeScript and essential tools
- Git and Docker support
- Pre-configured VS Code extensions

## Getting Started

1. Install Docker and VS Code
2. Install the "Dev Containers" extension
3. Click the green button in the bottom-left corner
4. Select "Reopen in Container"

## Customization

- Add dependencies to `Dockerfile`
- Configure VS Code settings in `devcontainer.json`
- Add environment variables to `.env`

## Included Tools

- Node.js 20
- pnpm
- TypeScript
- Git
- Build essentials
- Python 3.11

## Ports

The following ports are forwarded by default:

- 3000: Common for Next.js/React apps
- 8000: Common for API servers
- 5173: Common for Vite dev server
