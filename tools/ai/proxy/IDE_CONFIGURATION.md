# IDE Configuration Guide for Universal AI Proxy

This guide explains how to configure each IDE to use the Universal AI Proxy for automatic model tiering.

## Quick Start

```bash
# Start the proxy server
npm run ai:proxy:start

# Proxy runs at http://localhost:4000
```

---

## Coverage Matrix

| IDE             | Extension    | Custom Endpoint | Coverage Level |
| --------------- | ------------ | --------------- | -------------- |
| **VS Code**     | Continue.dev | ✅ Yes          | 🟢 Full        |
| **VS Code**     | Copilot      | ❌ No           | 🔴 None        |
| **VS Code**     | Codeium      | ⚠️ Enterprise   | 🟡 Partial     |
| **Cursor**      | Built-in     | ✅ OpenAI only  | 🟢 Full        |
| **Windsurf**    | Codeium      | ❌ No           | 🔴 None        |
| **Blackbox AI** | Built-in     | ❌ No           | 🔴 None        |

---

## 🟢 Full Coverage (Interceptable)

### VS Code + Continue.dev

**Location:** `~/.continue/config.json`

```json
{
  "models": [
    {
      "title": "Tiered AI Proxy",
      "provider": "openai",
      "model": "gpt-4o",
      "apiBase": "http://localhost:4000/v1",
      "apiKey": "your-openai-api-key"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Tiered Autocomplete",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "apiBase": "http://localhost:4000/v1"
  }
}
```

### Cursor (OpenAI Models)

**Location:** Settings → Models → OpenAI Configuration

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Go to "Models" section
3. Enable "Use custom OpenAI base URL"
4. Set Base URL: `http://localhost:4000/v1`
5. Enter your OpenAI API key

> **Note:** Cursor's Claude integration bypasses custom endpoints. Only OpenAI models are routed through the proxy.

### CLI Tools (OpenAI SDK)

```bash
# Set environment variable
export OPENAI_BASE_URL=http://localhost:4000/v1

# Python
import openai
client = openai.OpenAI(base_url="http://localhost:4000/v1")

# Node.js
import OpenAI from 'openai';
const openai = new OpenAI({ baseURL: 'http://localhost:4000/v1' });
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_base="http://localhost:4000/v1",
    openai_api_key="your-key"
)
```

---

## 🔴 Non-Interceptable Tools

These tools **cannot** be routed through the proxy due to:

- Certificate pinning
- Proprietary protocols
- No custom endpoint support

### GitHub Copilot

- Uses GitHub's authenticated API with certificate pinning
- **Workaround:** Use Continue.dev alongside Copilot for tiered routing

### Windsurf (Codeium)

- Proprietary Codeium backend
- **Workaround:** Enterprise accounts may support custom endpoints

### Blackbox AI

- Closed proprietary system
- **Workaround:** None available

---

## Behavioral Guidance (For Non-Interceptable Tools)

For tools that can't be intercepted, use our **pre-task routing system**:

```bash
# Before starting a task, check recommended tier
npm run ai:tier "describe your task here"

# Example outputs:
npm run ai:tier "fix typo in README"
# → LIGHTWEIGHT: Use Copilot or simple autocomplete

npm run ai:tier "architect microservices for payment system"
# → HEAVYWEIGHT: Use Claude Code or full IDE AI chat
```

---

## Auto-Start Proxy on Boot

### Windows (Task Scheduler)

```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "npm" -Argument "run ai:proxy:start" -WorkingDirectory "C:\path\to\repo"
$trigger = New-ScheduledTaskTrigger -AtLogon
Register-ScheduledTask -TaskName "AI Proxy" -Action $action -Trigger $trigger
```

### macOS/Linux (launchd/systemd)

```bash
# ~/.config/systemd/user/ai-proxy.service
[Unit]
Description=Universal AI Proxy

[Service]
ExecStart=/usr/bin/npm run ai:proxy:start
WorkingDirectory=/path/to/repo

[Install]
WantedBy=default.target
```

---

## Verify Proxy is Working

```bash
# Health check
curl http://localhost:4000/health

# List models
curl http://localhost:4000/v1/models

# Test chat completion
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "fix typo test"}]}'

# Check response headers for tier info:
# X-AI-Tier: lightweight
# X-AI-Model: gpt-4o-mini
```

---

## Token Usage Dashboard

After using the proxy, view your token usage:

```bash
npm run ai:tokens stats
```
