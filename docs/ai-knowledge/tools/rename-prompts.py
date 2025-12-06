#!/usr/bin/env python3
"""Rename prompts to be task-focused and project-agnostic."""

from pathlib import Path
import re

AI_KNOWLEDGE = Path(__file__).resolve().parents[1]
PROMPTS_DIR = AI_KNOWLEDGE / "prompts" / "superprompts"

# Mapping: old name pattern -> new task-focused name
RENAME_MAP = {
    # Project-specific -> Task-focused
    "REPZ_SUPERPROMPT": "fitness-platform-development",
    "TALAI_SUPERPROMPT": "ai-research-platform",
    "MAG_LOGIC_SUPERPROMPT": "magnetic-simulation-system",
    "QMAT_SIM_SUPERPROMPT": "quantum-materials-simulation",
    "QUBE_ML_SUPERPROMPT": "quantum-ml-framework",
    "SCI_COMP_SUPERPROMPT": "scientific-computing-library",
    "SIMCORE_CLAUDE_CODE_SUPERPROMPT": "physics-simulation-engine",
    "SPIN_CIRC_SUPERPROMPT": "quantum-circuit-simulator",
    "MEZAN_SUPERPROMPT": "optimization-framework",
    "OPTILIBRIA_SUPERPROMPT": "optimization-algorithms-library",
    
    # Make more generic
    "KILO_CONSOLIDATION_SUPERPROMPT": "repository-consolidation",
    "LOVABLE_TEMPLATE_SUPERPROMPT": "fullstack-saas-template",
    "LOVABLE_FULLSTACK_TEMPLATE_SYSTEM": "saas-template-system",
    
    # Task-focused naming
    "AI_ML_INTEGRATION_SUPERPROMPT": "ai-ml-integration",
    "CICD_PIPELINE_SUPERPROMPT": "cicd-pipeline-setup",
    "ENTERPRISE_AGENTIC_AI_SUPERPROMPT": "enterprise-ai-agents",
    "GATING_APPROVAL_SUPERPROMPT": "approval-gating-system",
    "GOVERNANCE_COMPLIANCE_SUPERPROMPT": "governance-compliance",
    "LOCAL_AI_ORCHESTRATION_SUPERPROMPT": "local-ai-orchestration",
    "MONOREPO_ARCHITECTURE_SUPERPROMPT": "monorepo-architecture",
    "PLATFORM_DEPLOYMENT_SUPERPROMPT": "platform-deployment",
    "PROMPT_OPTIMIZATION_SUPERPROMPT": "prompt-optimization",
    "SECURITY_CYBERSECURITY_SUPERPROMPT": "security-implementation",
    "TESTING_QA_SUPERPROMPT": "testing-qa-strategy",
    "UI_UX_DESIGN_SUPERPROMPT": "ui-ux-design",
    
    # Librex prompts -> Generic optimization tasks
    "PROMPT_Librex.Alloc": "resource-allocation-optimization",
    "PROMPT_Librex.Dual": "dual-problem-optimization",
    "PROMPT_Librex.Evo": "evolutionary-optimization",
    "PROMPT_Librex.Flow": "flow-optimization",
    "PROMPT_Librex.Graph": "graph-optimization",
    "PROMPT_Librex.Meta": "meta-optimization",
    "PROMPT_Librex.QAP": "quadratic-assignment-problem",
    
    # Task prompts
    "ATLAS_PROMPT_OPTIMIZER": "prompt-engineering-optimizer",
    "BRAINSTORMING_PROMPTS": "brainstorming-facilitation",
    "CRAZY_IDEAS_MASTER_PROMPT": "creative-ideation",
    "DESIGN_SYSTEM_PROMPTS": "design-system-creation",
    "MASTER_CLEANUP_PROMPT": "codebase-cleanup",
    "PROMPT_OPTIMIZER": "prompt-refinement",
    
    # Development tasks
    "api-development": "api-design-development",
    "automation-ts-implementation": "typescript-automation",
    "data-engineering-pipeline": "data-pipeline-engineering",
    "ml-pipeline-development": "ml-pipeline-design",
    "session-summary-2024-11-30": "session-summary-template",
    
    # System prompts - already good, but standardize
    "chain-of-thought-reasoning": "chain-of-thought-reasoning",
    "constitutional-self-alignment": "constitutional-ai-alignment",
    "context-engineering": "context-engineering",
    "crew_manager": "multi-agent-coordination",
    "debugger": "debugging-assistant",
    "evaluator": "code-evaluation",
    "orchestrator": "workflow-orchestration",
    "router": "request-routing",
    "state-of-the-art-ai-practices": "ai-best-practices",
    
    # Code review
    "agentic-code-review": "autonomous-code-review",
    
    # RAG
    "multi-hop-rag-processing": "multi-hop-rag",
    
    # Testing
    "test-generation": "automated-test-generation",
    
    # Config/Phase prompts
    "claude-opus-instructions": "claude-opus-configuration",
    "phase-1-infrastructure": "infrastructure-setup-phase",
    "phase-2-tooling": "tooling-setup-phase",
    "phase-3-ai-integration": "ai-integration-phase",
    "repository-consolidation-master": "repository-consolidation-master",
}

def rename_prompts():
    """Rename prompts to task-focused names."""
    renamed = []
    
    for old_name, new_name in RENAME_MAP.items():
        # Find file with old name
        old_file = PROMPTS_DIR / f"{old_name}.md"
        
        if not old_file.exists():
            continue
        
        # Create new filename
        new_file = PROMPTS_DIR / f"{new_name}.md"
        
        if new_file.exists():
            print(f"  [SKIP] {new_name}.md already exists")
            continue
        
        # Rename
        old_file.rename(new_file)
        renamed.append((old_name, new_name))
        print(f"  [OK] {old_name} -> {new_name}")
    
    return renamed

def update_readme(renamed):
    """Update README with new naming convention."""
    readme = PROMPTS_DIR.parent / "README.md"
    
    if not readme.exists():
        return
    
    content = readme.read_text(encoding='utf-8')
    
    for old_name, new_name in renamed:
        content = content.replace(old_name, new_name)
    
    readme.write_text(content, encoding='utf-8')
    print(f"\n[OK] Updated README with new names")

if __name__ == "__main__":
    print("Renaming prompts to task-focused names...\n")
    renamed = rename_prompts()
    
    if renamed:
        update_readme(renamed)
        print(f"\n[OK] Renamed {len(renamed)} prompts")
        print("\nNext: Run update-catalog.py to reindex")
    else:
        print("\n[OK] All prompts already have good names")
