# Multi-AI DevOps Analysis Runner
# Uses existing IDE integrations (no API keys needed!)

param(
    [string]$PromptFile = ".metaHub\docs\analysis\gemini-devops-analysis-prompt.md",
    [string]$OutputDir = ".metaHub\docs\analysis\results",
    [switch]$OpenInIDE = $true
)

Write-Host "[*] Multi-AI DevOps Analysis Runner" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Read the prompt
if (-not (Test-Path $PromptFile)) {
    Write-Host "[X] Prompt file not found: $PromptFile" -ForegroundColor Red
    exit 1
}

$prompt = Get-Content $PromptFile -Raw
Write-Host "[OK] Loaded prompt from: $PromptFile" -ForegroundColor Green
Write-Host ""

# Create timestamp for this analysis run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$runDir = Join-Path $OutputDir $timestamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Write-Host "[>>] Results will be saved to: $runDir" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# Option 1: Use Claude Code (Current IDE)
# ============================================================================
Write-Host "[1] Option 1: Claude Code (Current Session)" -ForegroundColor Magenta
Write-Host "-------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Copy the prompt from: $PromptFile"
Write-Host "2. Paste it into this Claude Code chat"
Write-Host "3. I'll analyze and save the results automatically"
Write-Host ""

$claudeOutput = Join-Path $runDir "analysis-claude.md"
Write-Host "[>>] Save Claude's response to: $claudeOutput" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Option 2: Use Cursor AI
# ============================================================================
Write-Host "[2] Option 2: Cursor AI" -ForegroundColor Magenta
Write-Host "-------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Open Cursor AI"
Write-Host "2. Create new chat"
Write-Host "3. Paste prompt from: $PromptFile"
Write-Host "4. Copy response to: $(Join-Path $runDir 'analysis-cursor.md')"
Write-Host ""

# ============================================================================
# Option 3: Use Windsurf
# ============================================================================
Write-Host "[3] Option 3: Windsurf" -ForegroundColor Magenta
Write-Host "-------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Open Windsurf"
Write-Host "2. Create new AI session"
Write-Host "3. Paste prompt from: $PromptFile"
Write-Host "4. Copy response to: $(Join-Path $runDir 'analysis-windsurf.md')"
Write-Host ""

# ============================================================================
# Option 4: Use GitHub Copilot Chat
# ============================================================================
Write-Host "[4] Option 4: GitHub Copilot Chat" -ForegroundColor Magenta
Write-Host "-------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Open VSCode with Copilot"
Write-Host "2. Open Copilot Chat"
Write-Host "3. Paste prompt from: $PromptFile"
Write-Host "4. Copy response to: $(Join-Path $runDir 'analysis-copilot.md')"
Write-Host ""

# ============================================================================
# Option 5: Use Cline/Continue
# ============================================================================
Write-Host "[5] Option 5: Cline/Continue" -ForegroundColor Magenta
Write-Host "-------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Open VSCode with Cline/Continue extension"
Write-Host "2. Start new conversation"
Write-Host "3. Paste prompt from: $PromptFile"
Write-Host "4. Copy response to: $(Join-Path $runDir 'analysis-cline.md')"
Write-Host ""

# ============================================================================
# Create helper files
# ============================================================================
Write-Host "[>>] Creating Helper Files..." -ForegroundColor Cyan
Write-Host ""

# Create a simplified prompt file for easy copy-paste
$simplifiedPromptFile = Join-Path $runDir "prompt-to-copy.txt"
$promptContent = Get-Content $PromptFile -Raw
# Extract just the main prompt (between the markdown code blocks)
if ($promptContent -match '(?s)```\s*\n(.*?)\n```') {
    $extractedPrompt = $matches[1]
} else {
    $extractedPrompt = $promptContent
}
Set-Content -Path $simplifiedPromptFile -Value $extractedPrompt -Encoding UTF8
Write-Host "[OK] Created easy-to-copy prompt: $simplifiedPromptFile" -ForegroundColor Green

# Create analysis comparison template
$comparisonTemplate = @"
# Multi-AI Analysis Comparison
**Run Date**: $timestamp

## AI Models Used

- [ ] Claude Code (Sonnet 4.5)
- [ ] Cursor AI (GPT-4 / Claude)
- [ ] Windsurf (Cascade)
- [ ] GitHub Copilot (GPT-4)
- [ ] Cline/Continue (various models)

## Key Findings Comparison

### Immediate Actions Recommended

**Claude Code:**
- [Add findings here]

**Cursor AI:**
- [Add findings here]

**Windsurf:**
- [Add findings here]

### Areas of Agreement

[What all AIs agree on - highest confidence recommendations]

### Areas of Disagreement

[Where AIs differ - requires human judgment]

### Unique Insights

**Claude Code unique insights:**
- [Add here]

**Cursor AI unique insights:**
- [Add here]

**Windsurf unique insights:**
- [Add here]

## Synthesis & Action Plan

### Top 3 Priority Actions
1. [Synthesized from all analyses]
2. [Synthesized from all analyses]
3. [Synthesized from all analyses]

### Top 3 Things to Stop Immediately
1. [Synthesized from all analyses]
2. [Synthesized from all analyses]
3. [Synthesized from all analyses]

### What We're Over-Engineering
- [Synthesized findings]

### Critical Gaps
- [Synthesized findings]

## Implementation Plan

### Week 1
- [ ] Action 1
- [ ] Action 2
- [ ] Action 3

### Month 1-3
- [ ] Strategy 1
- [ ] Strategy 2
- [ ] Strategy 3

### Month 3-12
- [ ] Vision 1
- [ ] Vision 2
- [ ] Vision 3

---

**Next Review Date**: [Set date]
**Owner**: [Assign owner]
"@

$comparisonFile = Join-Path $runDir "00-comparison-synthesis.md"
Set-Content -Path $comparisonFile -Value $comparisonTemplate -Encoding UTF8
Write-Host "[OK] Created comparison template: $comparisonFile" -ForegroundColor Green
Write-Host ""

# Create README for the analysis run
$readmeContent = @"
# DevOps Analysis Run - $timestamp

## Overview

This directory contains multi-AI analysis of our DevOps practices and product success factors.

## Prompt Used

Source: $PromptFile
Extracted: prompt-to-copy.txt

## Expected AI Responses

1. **analysis-claude.md** - Claude Code (Sonnet 4.5) analysis
2. **analysis-cursor.md** - Cursor AI analysis
3. **analysis-windsurf.md** - Windsurf analysis
4. **analysis-copilot.md** - GitHub Copilot analysis
5. **analysis-cline.md** - Cline/Continue analysis

## Synthesis

See **00-comparison-synthesis.md** for:
- Comparison of findings across all AIs
- Areas of agreement (high confidence)
- Areas of disagreement (requires judgment)
- Synthesized action plan

## How to Use Results

1. **Read each AI analysis** independently
2. **Fill out comparison template** with key findings
3. **Identify patterns** across multiple AIs
4. **Prioritize actions** that multiple AIs recommend
5. **Investigate disagreements** - often reveal important nuances
6. **Create action plan** based on synthesis

## Quick Access

- Prompt to copy: [prompt-to-copy.txt](prompt-to-copy.txt)
- Comparison: [00-comparison-synthesis.md](00-comparison-synthesis.md)

---

**Status**: In Progress
**Created**: $timestamp
"@

$readmeFile = Join-Path $runDir "README.md"
Set-Content -Path $readmeFile -Value $readmeContent -Encoding UTF8
Write-Host "[OK] Created README: $readmeFile" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Open files in editor
# ============================================================================
if ($OpenInIDE) {
    Write-Host "[>>] Opening files..." -ForegroundColor Cyan

    # Open the prompt file for easy copying
    Write-Host "Opening prompt file for copying..."
    code $simplifiedPromptFile

    # Open comparison template for filling out
    Write-Host "Opening comparison template for results..."
    code $comparisonFile

    Write-Host ""
}

# ============================================================================
# Summary and Next Steps
# ============================================================================
Write-Host "[OK] Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. COPY THE PROMPT:" -ForegroundColor Cyan
Write-Host "   File opened: $simplifiedPromptFile"
Write-Host ""
Write-Host "2. RUN IN MULTIPLE AIs:" -ForegroundColor Cyan
Write-Host "   [ ] Claude Code (this chat) - Just paste the prompt!"
Write-Host "   [ ] Cursor AI - Open and paste"
Write-Host "   [ ] Windsurf - Open and paste"
Write-Host "   [ ] GitHub Copilot - Open and paste"
Write-Host "   [ ] Cline/Continue - Open and paste"
Write-Host ""
Write-Host "3. SAVE EACH RESPONSE:" -ForegroundColor Cyan
Write-Host "   Save to: $runDir\analysis-[ai-name].md"
Write-Host ""
Write-Host "4. FILL OUT COMPARISON:" -ForegroundColor Cyan
Write-Host "   Template opened: $comparisonFile"
Write-Host ""
Write-Host "5. SYNTHESIZE & ACT:" -ForegroundColor Cyan
Write-Host "   Create action plan from synthesis"
Write-Host ""
Write-Host "============================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "[TIP] Focus on areas where multiple AIs agree!" -ForegroundColor Green
Write-Host "[TIP] Disagreements often reveal important nuances!" -ForegroundColor Green
Write-Host ""
Write-Host "Results directory: $runDir" -ForegroundColor Magenta
