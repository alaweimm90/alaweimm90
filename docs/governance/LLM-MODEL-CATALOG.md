# LLM Model Catalog

> Comprehensive reference for AI models available across coding assistants and IDEs.

**Last Updated:** 2025-12-03  
**Total Models Cataloged:** 500+  
**IDEs/Extensions Covered:** 8

---

## Table of Contents

1. [Overview](#overview)
2. [IDE Comparison Matrix](#ide-comparison-matrix)
3. [Models by Provider](#models-by-provider)
   - [Anthropic (Claude)](#anthropic-claude)
   - [OpenAI](#openai)
   - [Google (Gemini/Gemma)](#google-geminigemma)
   - [xAI (Grok)](#xai-grok)
   - [Mistral](#mistral)
   - [DeepSeek](#deepseek)
   - [Qwen (Alibaba)](#qwen-alibaba)
   - [Meta (Llama)](#meta-llama)
   - [Other Providers](#other-providers)
4. [Models by IDE/Extension](#models-by-ideextension)
   - [Windsurf](#windsurf)
   - [Cursor](#cursor)
   - [VS Code (GitHub Copilot)](#vs-code-github-copilot)
   - [Cline](#cline)
   - [Kilo Code](#kilo-code)
   - [Blackbox AI](#blackbox-ai)
   - [Augment Code](#augment-code)
   - [Amazon Q](#amazon-q)
5. [Model Selection Guide](#model-selection-guide)
6. [Pricing & Cost Optimization](#pricing--cost-optimization)
7. [Glossary](#glossary)

---

## Overview

This catalog documents all AI models available across major coding assistants and IDEs. Use it to:

- **Compare model availability** across different tools
- **Select the right model** for your task type
- **Optimize costs** by understanding pricing tiers
- **Stay current** with the latest model releases

### Key Concepts

| Term                   | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| **Thinking/Reasoning** | Extended chain-of-thought processing for complex problems  |
| **Extended Context**   | Models with 1M+ token context windows (e.g., `:1m` suffix) |
| **BYOK**               | Bring Your Own Key - use your own API credentials          |
| **MoE**                | Mixture of Experts - efficient sparse architecture         |
| **Distill**            | Knowledge distillation from larger models                  |

---

## IDE Comparison Matrix

| IDE/Extension       | Model Count | Pricing Model            | Free Tier | BYOK | Notes                         |
| ------------------- | ----------- | ------------------------ | --------- | ---- | ----------------------------- |
| **Kilo Code**       | 280+        | Per-model                | Yes       | Yes  | Largest selection, OpenRouter |
| **Cline**           | 150+        | OpenRouter               | Yes       | Yes  | Full provider/model naming    |
| **Windsurf**        | ~35         | Multipliers (0.125x-20x) | Limited   | Yes  | Cascade agent                 |
| **VS Code Copilot** | ~17         | Multipliers (0x-1x)      | Yes       | No   | Microsoft ecosystem           |
| **Blackbox**        | ~16         | Tiered                   | Yes       | No   | Pro/Pro Plus tiers            |
| **Cursor**          | ~8          | Subscription             | No        | No   | MAX Mode, Multi-model         |
| **Augment Code**    | ~6          | Curated                  | Limited   | No   | Enterprise focus              |
| **Amazon Q**        | ~4          | AWS Bedrock              | Limited   | No   | Claude-only                   |

---

## Models by Provider

### Anthropic (Claude)

The Claude family excels at reasoning, safety, and large context handling.

| Model                            | Context | Best For                        | Available In                           |
| -------------------------------- | ------- | ------------------------------- | -------------------------------------- |
| **Claude Opus 4.5**              | 200K    | Complex reasoning, architecture | Windsurf, Cline, Kilo, Augment, Cursor |
| **Claude Opus 4.1**              | 200K    | Premium tasks                   | Windsurf, Cline, Kilo                  |
| **Claude Opus 4**                | 200K    | Complex tasks                   | All IDEs                               |
| **Claude Sonnet 4.5**            | 200K    | Everyday coding                 | All IDEs                               |
| **Claude Sonnet 4.5 (1M)**       | 1M      | Large codebases                 | Windsurf, Cline                        |
| **Claude Sonnet 4**              | 200K    | Standard tasks                  | All IDEs                               |
| **Claude Sonnet 4 (1M)**         | 1M      | Extended context                | Cline                                  |
| **Claude Haiku 4.5**             | 200K    | Fast responses                  | All IDEs                               |
| **Claude 3.7 Sonnet**            | 200K    | Previous gen                    | Windsurf, Cline, Kilo, Blackbox        |
| **Claude 3.7 Sonnet (Thinking)** | 200K    | Extended reasoning              | Windsurf, Cline, Kilo                  |
| **Claude 3.5 Sonnet**            | 200K    | Legacy                          | Cline, Kilo, Blackbox                  |
| **Claude 3.5 Haiku**             | 200K    | Legacy fast                     | Cline, Kilo                            |
| **Claude 3 Opus**                | 200K    | Legacy premium                  | Cline, Kilo                            |
| **Claude 3 Haiku**               | 200K    | Legacy fast                     | Cline, Kilo                            |

**Thinking Variants:** Enable step-by-step reasoning with `:thinking` suffix. Best for debugging, architecture decisions, and complex refactoring.

---

### OpenAI

OpenAI offers the broadest model range from legacy GPT-3.5 to cutting-edge reasoning models.

#### GPT-5.x Series (Latest)

| Model                          | Best For          | Available In                            |
| ------------------------------ | ----------------- | --------------------------------------- |
| **GPT-5.1**                    | Latest flagship   | Windsurf, Cline, Kilo, VS Code, Augment |
| **GPT-5.1 (high reasoning)**   | Complex problems  | Windsurf                                |
| **GPT-5.1 (medium reasoning)** | Balanced          | Windsurf                                |
| **GPT-5.1 (low reasoning)**    | Fast responses    | Windsurf                                |
| **GPT-5.1-Codex**              | Code generation   | Windsurf, Cline, Kilo, VS Code          |
| **GPT-5.1-Codex-Mini**         | Lightweight code  | Windsurf, Cline, Kilo, VS Code          |
| **GPT-5**                      | Previous flagship | Cline, Kilo, VS Code                    |
| **GPT-5 Codex**                | Code              | Cline, Kilo, VS Code, Cursor            |
| **GPT-5 Mini/Nano/Pro**        | Size variants     | Cline, Kilo                             |
| **GPT-5 Fast**                 | Speed-optimized   | Cursor                                  |
| **GPT-5 Image**                | Vision            | Cline, Kilo                             |

#### GPT-4.x Series

| Model                 | Best For          | Available In                   |
| --------------------- | ----------------- | ------------------------------ |
| **GPT-4.1**           | Current stable    | Windsurf, Cline, Kilo, VS Code |
| **GPT-4.1 Mini/Nano** | Lightweight       | Cline, Kilo                    |
| **GPT-4o**            | Multimodal        | All IDEs                       |
| **GPT-4o (extended)** | Large context     | Cline                          |
| **GPT-4o-mini**       | Fast multimodal   | All IDEs                       |
| **GPT-4 Turbo**       | Previous flagship | Cline, Kilo, VS Code           |

#### Reasoning Models (o-series)

| Model                            | Best For              | Available In                    |
| -------------------------------- | --------------------- | ------------------------------- |
| **o4-mini**                      | Latest reasoning      | Cline, Kilo                     |
| **o4-mini (high/deep research)** | Complex analysis      | Cline, Kilo                     |
| **o3**                           | Advanced reasoning    | Windsurf, Cline, Kilo           |
| **o3-mini / o3-mini-high**       | Lightweight reasoning | Windsurf, Cline, Kilo           |
| **o3-pro**                       | Premium reasoning     | Cline, Kilo                     |
| **o3 Deep Research**             | Research tasks        | Cline, Kilo                     |
| **o1 / o1-pro**                  | Original reasoning    | Windsurf, Cline, Kilo, Blackbox |

#### Open-Source Variants

| Model                     | Notes           | Available In          |
| ------------------------- | --------------- | --------------------- |
| **gpt-oss-120b**          | Community 120B  | Windsurf, Cline, Kilo |
| **gpt-oss-120b (exacto)** | Precise variant | Cline, Kilo           |
| **gpt-oss-20b**           | Smaller variant | Cline, Kilo           |
| **gpt-oss-safeguard-20b** | Safety-focused  | Cline, Kilo           |

---

### Google (Gemini/Gemma)

Google's models excel at multimodal tasks and offer generous context windows.

#### Gemini (Proprietary)

| Model                         | Context | Best For           | Available In                   |
| ----------------------------- | ------- | ------------------ | ------------------------------ |
| **Gemini 3 Pro (Preview)**    | 1M+     | Latest flagship    | Windsurf, Cline, Kilo, VS Code |
| **Gemini 3 Pro (high/low)**   | 1M+     | Reasoning variants | Windsurf                       |
| **Gemini 2.5 Pro**            | 1M      | Current flagship   | Windsurf, Cline, Kilo, VS Code |
| **Gemini 2.5 Pro Preview**    | 1M      | Preview features   | Cline, Kilo, Blackbox          |
| **Gemini 2.5 Flash**          | 1M      | Fast responses     | Windsurf, Cline, Kilo          |
| **Gemini 2.5 Flash Lite**     | 1M      | Lightweight        | Windsurf, Cline, Kilo          |
| **Gemini 2.5 Flash Image**    | 1M      | Vision tasks       | Cline, Kilo                    |
| **Gemini 2.0 Flash**          | 1M      | Previous gen       | Cline, Kilo                    |
| **Gemini CLI 2.5 PRO (Free)** | 1M      | Free tier          | Blackbox                       |

#### Gemma (Open-Weight)

| Model              | Parameters | Available In |
| ------------------ | ---------- | ------------ |
| **Gemma 3 27B**    | 27B        | Cline, Kilo  |
| **Gemma 3 12B**    | 12B        | Cline, Kilo  |
| **Gemma 3 4B**     | 4B         | Cline, Kilo  |
| **Gemma 3n 4B**    | 4B (nano)  | Cline, Kilo  |
| **Gemma 2 27B/9B** | 27B/9B     | Cline, Kilo  |

---

### xAI (Grok)

Grok models are known for speed and code generation capabilities.

| Model                       | Best For         | Available In                    |
| --------------------------- | ---------------- | ------------------------------- |
| **Grok 4.1 Fast**           | Latest, speed    | Cline, Kilo                     |
| **Grok 4 / Grok 4 Fast**    | Current flagship | Cline, Kilo                     |
| **Grok 3**                  | Standard         | Windsurf, Cline, Kilo, Blackbox |
| **Grok 3 Beta**             | Beta features    | Cline, Kilo, Blackbox           |
| **Grok 3 Mini / Mini Beta** | Lightweight      | Windsurf, Cline, Kilo           |
| **Grok Code Fast 1**        | Code-specialized | Windsurf, Cline, Kilo, VS Code  |
| **Grok-3 mini (Thinking)**  | Reasoning        | Windsurf                        |

---

### Mistral

French AI lab known for efficient MoE architectures and code models.

#### Code Models

| Model                   | Best For        | Available In |
| ----------------------- | --------------- | ------------ |
| **Codestral 2508**      | Latest code     | Cline, Kilo  |
| **Codestral 2501**      | Code generation | Cline, Kilo  |
| **Devstral Medium**     | Development     | Kilo         |
| **Devstral Small 2505** | Lightweight dev | Kilo         |

#### General Models

| Model                     | Parameters | Available In |
| ------------------------- | ---------- | ------------ |
| **Mistral Large 3 2512**  | Large      | Kilo         |
| **Mistral Large 2411**    | Large      | Cline, Kilo  |
| **Mistral Medium 3.1**    | Medium     | Cline, Kilo  |
| **Mistral Small 3.2 24B** | 24B        | Cline, Kilo  |
| **Mistral Nemo**          | Efficient  | Cline, Kilo  |
| **Mixtral 8x22B**         | MoE        | Cline, Kilo  |
| **Mixtral 8x7B**          | MoE        | Cline, Kilo  |

#### Specialized

| Model                           | Best For           | Available In |
| ------------------------------- | ------------------ | ------------ |
| **Magistral Medium 2506**       | Reasoning          | Kilo         |
| **Magistral Medium (thinking)** | Extended reasoning | Kilo         |
| **Pixtral Large 2411**          | Vision             | Kilo         |
| **Voxtral Small 24B**           | Audio              | Kilo         |

---

### DeepSeek

Chinese lab known for efficient reasoning models and open weights.

| Model                              | Best For         | Available In                    |
| ---------------------------------- | ---------------- | ------------------------------- |
| **DeepSeek V3.2 / Exp / Speciale** | Latest           | Kilo                            |
| **DeepSeek V3.1 / Terminus**       | Current          | Kilo                            |
| **DeepSeek V3 / 0324**             | Base             | Windsurf, Cline, Kilo, Blackbox |
| **DeepSeek R1 0528**               | Latest reasoning | Windsurf, Kilo                  |
| **DeepSeek R1**                    | Reasoning        | Windsurf, Cline, Kilo, Blackbox |
| **R1 Distill Llama 70B**           | Distilled        | Kilo                            |
| **R1 Distill Qwen 32B/14B**        | Distilled        | Kilo                            |
| **DeepSeek Prover V2**             | Math/proofs      | Kilo                            |

---

### Qwen (Alibaba)

Alibaba's models with strong multilingual and code capabilities.

#### Qwen3 Series

| Model                      | Parameters | Available In   |
| -------------------------- | ---------- | -------------- |
| **Qwen3 Coder 480B A35B**  | 480B MoE   | Windsurf, Kilo |
| **Qwen3 235B A22B**        | 235B MoE   | Kilo           |
| **Qwen3 Max**              | Large      | Kilo           |
| **Qwen3 32B/14B/8B**       | Dense      | Kilo           |
| **Qwen3 Coder Flash/Plus** | Code       | Kilo           |
| **Qwen3 VL variants**      | Vision     | Kilo           |
| **QwQ 32B**                | Reasoning  | Windsurf, Kilo |

#### Qwen2.5 Series

| Model                    | Best For | Available In |
| ------------------------ | -------- | ------------ |
| **Qwen2.5 72B Instruct** | General  | Kilo         |
| **Qwen2.5 Coder 32B**    | Code     | Kilo         |
| **Qwen2.5-VL 72B**       | Vision   | Kilo         |

---

### Meta (Llama)

Open-weight models with strong community support.

| Model                    | Parameters     | Available In             |
| ------------------------ | -------------- | ------------------------ |
| **Llama 4 Maverick**     | Latest         | Windsurf, Kilo, Blackbox |
| **Llama 4 Scout**        | Latest         | Kilo                     |
| **Llama 3.3 70B**        | 70B            | Kilo                     |
| **Llama 3.2 90B Vision** | 90B multimodal | Kilo                     |
| **Llama 3.1 405B**       | 405B           | Kilo                     |
| **Llama 3.1 70B/8B**     | Dense          | Kilo                     |
| **LlamaGuard 4 12B**     | Safety         | Kilo                     |

---

### Other Providers

#### Amazon (Nova)

| Model            | Available In |
| ---------------- | ------------ |
| Nova Premier 1.0 | Kilo         |
| Nova Pro 1.0     | Kilo         |
| Nova Lite 1.0/2  | Kilo         |
| Nova Micro 1.0   | Kilo         |

#### Microsoft

| Model                               | Available In |
| ----------------------------------- | ------------ |
| Phi 4 / Multimodal / Reasoning Plus | Kilo         |
| Phi-3.5 Mini 128K                   | Kilo         |
| MAI DS R1                           | Kilo         |
| WizardLM-2 8x22B                    | Kilo         |

#### MiniMax

| Model             | Available In          |
| ----------------- | --------------------- |
| MiniMax M2 (free) | Windsurf, Cline, Kilo |
| MiniMax M1        | Cline, Kilo           |
| MiniMax-01        | Cline, Kilo           |

#### Cohere

| Model                | Available In |
| -------------------- | ------------ |
| Command A            | Kilo         |
| Command R+ (08-2024) | Kilo         |
| Command R7B          | Kilo         |

#### Perplexity

| Model                     | Best For           | Available In |
| ------------------------- | ------------------ | ------------ |
| Sonar Pro / Reasoning Pro | Search + reasoning | Kilo         |
| Sonar Deep Research       | Research           | Kilo         |

#### Z.AI (GLM/Zhipu)

| Model             | Available In |
| ----------------- | ------------ |
| GLM 4.6           | Cline, Kilo  |
| GLM 4.5 / Air / V | Cline, Kilo  |

#### AI21

| Model           | Available In |
| --------------- | ------------ |
| Jamba Large 1.7 | Kilo         |
| Jamba Mini 1.7  | Kilo         |

#### MoonshotAI (Kimi)

| Model            | Available In   |
| ---------------- | -------------- |
| Kimi K2 0905     | Windsurf, Kilo |
| Kimi K2 Thinking | Kilo           |
| Kimi Dev 72B     | Kilo           |

#### NVIDIA

| Model                 | Available In |
| --------------------- | ------------ |
| Nemotron Ultra 253B   | Kilo         |
| Nemotron 70B Instruct | Kilo         |
| Nemotron Nano 12B/9B  | Kilo         |

#### Nous Research

| Model             | Available In |
| ----------------- | ------------ |
| Hermes 4 405B/70B | Kilo         |
| Hermes 3 405B/70B | Kilo         |

#### Deep Cogito

| Model                       | Available In |
| --------------------------- | ------------ |
| Cogito v2.1 671B            | Kilo         |
| Cogito V2 Preview (various) | Cline, Kilo  |

#### Inception

| Model         | Available In |
| ------------- | ------------ |
| Mercury       | Kilo         |
| Mercury Coder | Kilo         |

#### Arcee AI

| Model             | Available In |
| ----------------- | ------------ |
| Coder Large       | Kilo         |
| Maestro Reasoning | Kilo         |
| Virtuoso Large    | Kilo         |

---

## Models by IDE/Extension

### Windsurf

**Total Models:** ~35  
**Pricing:** Multipliers (0.125x - 20x)  
**Special Features:** Cascade agent, BYOK support

| Model                                  | Cost      | Notes                |
| -------------------------------------- | --------- | -------------------- |
| Claude 3.7 Sonnet                      | 2x        | Standard             |
| Claude 3.7 Sonnet (Thinking)           | 3x        | Extended reasoning   |
| Claude Haiku 4.5                       | 1x        | Fast                 |
| Claude Opus 4 (BYOK)                   | BYOK      | Premium              |
| Claude Opus 4 (Thinking, BYOK)         | BYOK      | Premium + reasoning  |
| Claude Opus 4.1                        | 20x       | High-end             |
| Claude Opus 4.1 (Thinking)             | 20x       | High-end + reasoning |
| Claude Opus 4.5                        | 2x        | Latest premium       |
| Claude Opus 4.5 (Thinking)             | 3x        | Latest + reasoning   |
| GPT-5.1 (high/medium/low/no reasoning) | Free-2x   | Reasoning variants   |
| GPT-5.1-Codex                          | Free      | Code                 |
| GPT-5.1-Codex-Mini                     | Free      | Lightweight code     |
| DeepSeek R1 (0528)                     | Free      | Reasoning            |
| Gemini 3 Pro (high/low)                | 1x        | Google               |
| Grok Code Fast 1                       | —         | xAI                  |
| Kimi K2                                | 0.5x      | MoonshotAI           |
| MiniMax M2                             | 0.5x      | MiniMax              |
| o3 / o3 (high reasoning)               | 1x        | OpenAI reasoning     |
| Penguin Alpha                          | Free      | Experimental         |
| Qwen3-Coder                            | 0.5x      | Alibaba              |
| SWE-1 / SWE-1.5                        | Free-0.5x | SWE agent            |
| xAI Grok-3                             | 1x        | xAI                  |
| xAI Grok-3 mini (Thinking)             | 0.125x    | Lightweight          |

---

### Cursor

**Total Models:** ~8  
**Pricing:** Subscription-based  
**Special Features:** Auto mode, MAX Mode, Multi-model

| Model         | Notes           |
| ------------- | --------------- |
| Composer 1    | Default         |
| Opus 4.5      | Premium (slow)  |
| Sonnet 4.5    | Standard        |
| Sonnet 4.5    | With processing |
| GPT-5.1 Codex | Code (slow)     |
| GPT-5 Codex   | Code (slow)     |
| GPT-5.1       | Latest (slow)   |
| GPT-5 Fast    | Speed-optimized |

---

### VS Code (GitHub Copilot)

**Total Models:** ~17  
**Pricing:** Multipliers (0x - 1x)  
**Special Features:** 10% Auto discount, free tier

| Model                        | Cost         | Notes               |
| ---------------------------- | ------------ | ------------------- |
| Auto                         | 10% discount | Automatic selection |
| GPT-4.1                      | 0x           | Free                |
| GPT-4o                       | 0x           | Free                |
| GPT-5 mini                   | 0x           | Free                |
| Grok Code Fast 1             | 0x           | Free                |
| Raptor mini (Preview)        | 0x           | Free                |
| Claude Haiku 4.5             | 0.33x        | Discounted          |
| GPT-5.1-Codex-Mini (Preview) | 0.33x        | Discounted          |
| Claude Opus 4.5 (Preview)    | 1x           | Standard            |
| Claude Sonnet 4              | 1x           | Standard            |
| Claude Sonnet 4.5            | 1x           | Standard            |
| Gemini 2.5 Pro               | 1x           | Standard            |
| Gemini 3 Pro (Preview)       | 1x           | Standard            |
| GPT-5                        | 1x           | Standard            |
| GPT-5-Codex (Preview)        | 1x           | Standard            |
| GPT-5.1 (Preview)            | 1x           | Standard            |
| GPT-5.1-Codex (Preview)      | 1x           | Standard            |

---

### Cline

**Total Models:** 150+  
**Pricing:** OpenRouter  
**Special Features:** Full provider/model naming, browser control

Uses `provider/model` naming convention. See [Models by Provider](#models-by-provider) for full list.

**Unique to Cline:**

- `stealth/microwave` - Unknown/experimental
- Extended context variants (`:1m`)
- Exacto variants (`:exacto`)

---

### Kilo Code

**Total Models:** 280+  
**Pricing:** Per-model (OpenRouter)  
**Special Features:** Largest selection, community models

See [Models by Provider](#models-by-provider) for comprehensive list.

**Unique to Kilo Code:**

- Full community/fine-tuned model access
- Specialized models (Sao10K, TheDrummer, NeverSleep)
- Research models (EleutherAI, AllenAI)
- Regional models (Baidu ERNIE, Tencent Hunyuan)

---

### Blackbox AI

**Total Models:** ~16  
**Pricing:** Tiered (Base/Pro/Pro Plus)  
**Special Features:** YOLO mode, rapid prototyping

| Model                     | Provider  |
| ------------------------- | --------- |
| Blackbox Base             | Blackbox  |
| Blackbox Pro              | Blackbox  |
| Blackbox Pro Plus         | Blackbox  |
| Gemini CLI 2.5 PRO (Free) | Google    |
| Gemini 2.5 Pro Preview    | Google    |
| Claude Sonnet 3.5/3.7/4   | Anthropic |
| Claude Opus 4             | Anthropic |
| DeepSeek-R1 / V3          | DeepSeek  |
| Llama 4 Maverick          | Meta      |
| Grok 3 Beta               | xAI       |
| O3 Mini                   | OpenAI    |
| OpenAI o1                 | OpenAI    |

---

### Augment Code

**Total Models:** ~6  
**Pricing:** Curated/Enterprise  
**Special Features:** Use-case descriptions, enterprise focus

| Model           | Description                     |
| --------------- | ------------------------------- |
| Claude Opus 4.5 | "Best for complex tasks"        |
| Sonnet 4.5      | "Great for everyday tasks"      |
| GPT-5.1         | "Strong reasoning and planning" |
| Haiku 4.5       | "Fast and efficient responses"  |
| Sonnet 4        | Legacy model                    |
| GPT-5           | Legacy                          |

---

### Amazon Q

**Total Models:** ~4  
**Pricing:** AWS Bedrock  
**Special Features:** AWS integration, Claude-only

| Model                   | Notes    |
| ----------------------- | -------- |
| Claude Sonnet 4.5       | Primary  |
| Claude Sonnet 4         | Standard |
| Claude Haiku 4.5        | Fast     |
| Autocomplete (7 models) | Various  |

---

## Model Selection Guide

### By Task Type

| Task                     | Recommended Models                            | Why                         |
| ------------------------ | --------------------------------------------- | --------------------------- |
| **Complex Architecture** | Claude Opus 4.5, GPT-5.1 (high reasoning), o3 | Best reasoning capabilities |
| **Everyday Coding**      | Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Pro     | Balanced speed/quality      |
| **Quick Edits**          | Claude Haiku 4.5, GPT-5 mini, Gemini Flash    | Fast responses              |
| **Code Generation**      | GPT-5.1-Codex, Codestral, Qwen3 Coder         | Code-optimized              |
| **Large Codebases**      | Claude Sonnet 4.5 (1M), Gemini 2.5 Pro        | Extended context            |
| **Debugging**            | Claude (Thinking), o3, DeepSeek R1            | Step-by-step reasoning      |
| **Research**             | Perplexity Sonar, o3 Deep Research            | Search-augmented            |
| **Multimodal**           | GPT-4o, Gemini, Claude (vision)               | Image understanding         |

### By Budget

| Budget               | Recommended                                         |
| -------------------- | --------------------------------------------------- |
| **Free**             | GPT-5.1 (Windsurf), DeepSeek R1, Gemini CLI 2.5 PRO |
| **Low (0.33x)**      | Claude Haiku 4.5, GPT-5.1-Codex-Mini                |
| **Standard (1x)**    | Claude Sonnet 4.5, GPT-5, Gemini 2.5 Pro            |
| **Premium (2-3x)**   | Claude Opus 4.5, Thinking variants                  |
| **Enterprise (20x)** | Claude Opus 4.1                                     |

---

## Pricing & Cost Optimization

### Cost Multiplier Reference

| Multiplier  | Meaning          | Examples                                  |
| ----------- | ---------------- | ----------------------------------------- |
| **Free/0x** | No cost          | DeepSeek R1, GPT-5.1 (Windsurf free tier) |
| **0.125x**  | Very cheap       | Grok-3 mini (Thinking)                    |
| **0.33x**   | Discounted       | Claude Haiku 4.5, GPT-5.1-Codex-Mini      |
| **0.5x**    | Half price       | Kimi K2, MiniMax M2, Qwen3-Coder          |
| **1x**      | Standard         | Most Sonnet/Pro models                    |
| **2x**      | Premium          | Claude Opus 4.5, Sonnet (Thinking)        |
| **3x**      | High-end         | Opus (Thinking) variants                  |
| **10x**     | Extended context | Claude Sonnet 4.5 (1M)                    |
| **20x**     | Enterprise       | Claude Opus 4.1                           |
| **BYOK**    | Your API key     | Opus 4 BYOK variants                      |

### Optimization Strategies

1. **Use Auto mode** - Let the IDE select the optimal model (10% discount in VS Code)
2. **Match model to task** - Don't use Opus for simple edits
3. **Leverage free tiers** - DeepSeek R1, GPT-5.1 variants
4. **BYOK for heavy usage** - Bring your own key for premium models
5. **Batch similar tasks** - Use extended context models for multi-file work

---

## Glossary

| Term                   | Definition                                                             |
| ---------------------- | ---------------------------------------------------------------------- |
| **A3B/A22B/A35B**      | Active parameters in MoE models (e.g., 30B A3B = 30B total, 3B active) |
| **BYOK**               | Bring Your Own Key - use personal API credentials                      |
| **Distill**            | Knowledge distillation - smaller model trained on larger model outputs |
| **Exacto**             | Precise/deterministic variant (lower temperature)                      |
| **Extended Context**   | Models with 1M+ token context windows                                  |
| **Instruct**           | Fine-tuned for following instructions                                  |
| **MoE**                | Mixture of Experts - sparse architecture with specialized sub-networks |
| **Preview**            | Beta/experimental version                                              |
| **Reasoning/Thinking** | Extended chain-of-thought processing                                   |
| **VL**                 | Vision-Language - multimodal capabilities                              |

---

## Related Documentation

- [AI-TOOL-PROFILES.md](../AI-TOOL-PROFILES.md) - IDE/tool profiles and workflows
- [AI-TOOLS-ORCHESTRATION.md](../AI-TOOLS-ORCHESTRATION.md) - Multi-agent orchestration
- [ai-coding-tools/README.md](../ai-coding-tools/README.md) - Tool catalog

---

_This catalog is maintained as part of the governance documentation. Submit updates via PR._
