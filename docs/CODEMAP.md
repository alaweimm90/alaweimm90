# Interactive Codemap

> Auto-generated system architecture visualization

## System Overview

```mermaid
flowchart TB
    subgraph User["👤 User"]
        CLI[CLI Commands]
        IDE[IDE/Editor]
    end

    subgraph Core["🔧 Core Tools"]
        devops[tools/cli/devops.ts]
        governance[tools/cli/governance.py]
        atlas[tools/atlas/*]
    end

    subgraph Templates["📦 Templates"]
        cicd[CI/CD]
        k8s[Kubernetes]
        db[Databases]
        iac[IaC]
        monitoring[Monitoring]
    end

    subgraph Automation["⚙️ GitHub Actions"]
        ci[ci.yml]
        catalog[catalog.yml]
        enforce[enforce.yml]
        checkpoint[checkpoint.yml]
    end

    subgraph Output["📊 Output"]
        metahub[.metaHub/]
        reports[enforcement-reports/]
        catalogjson[catalog.json]
    end

    CLI --> devops
    CLI --> governance
    IDE --> atlas

    devops --> Templates
    Templates --> metahub

    governance --> reports
    governance --> catalogjson

    Automation --> governance
    Automation --> devops
```

## CLI Command Flow

```mermaid
flowchart LR
    subgraph Commands["npm run"]
        list[devops:list]
        builder[devops:builder]
        coder[devops:coder]
        gov[governance]
    end

    subgraph DevOps["devops.ts"]
        cmdList[cmdTemplateList]
        cmdApply[cmdTemplateApply]
        cmdGen[cmdGenerate]
    end

    subgraph Lib["tools/lib"]
        config[config.ts]
        fs[fs.ts]
    end

    subgraph Output["Output"]
        console[Console]
        files[.metaHub/*]
    end

    list --> cmdList --> console
    builder --> cmdApply --> fs --> files
    coder --> cmdGen --> fs --> files

    cmdApply --> config
    cmdGen --> config
```

## Template Library Structure

```mermaid
flowchart TD
    templates[templates/devops/]

    templates --> cicd[cicd/]
    templates --> k8s[k8s/]
    templates --> db[db/]
    templates --> iac[iac/]
    templates --> logging[logging/]
    templates --> monitoring[monitoring/]
    templates --> ui[ui/]
    templates --> demos[demos/]

    cicd --> ga[github-actions]
    cicd --> circle[circleci]
    cicd --> jenkins[jenkins]

    k8s --> helm[helm]
    k8s --> manifests[manifests]

    db --> postgres[postgres]
    db --> mongo[mongo]
    db --> prisma[prisma]

    iac --> terraform[terraform]
    iac --> cloudform[cloudformation]

    monitoring --> prom[prometheus]
    monitoring --> grafana[grafana]

    style templates fill:#6366F1,color:#fff
    style cicd fill:#10B981,color:#fff
    style k8s fill:#F59E0B,color:#fff
    style db fill:#EC4899,color:#fff
```

## Governance Data Flow

```mermaid
flowchart TB
    subgraph Trigger["Triggers"]
        schedule[⏰ Schedule]
        push[📤 Push Event]
        manual[👆 Manual]
    end

    subgraph Workflows["GitHub Actions"]
        enforce[enforce.yml<br/>Every 6h]
        catalog[catalog.yml<br/>Daily 8AM]
        checkpoint[checkpoint.yml<br/>Weekly]
    end

    subgraph Scanner["Scanner"]
        scan[Scan organizations/**]
        validate[Validate .meta/repo.yaml]
    end

    subgraph Output["Output Artifacts"]
        enforcement[enforcement-reports/*.json]
        catalogjson[catalog.json]
        drift[drift-report.md]
        checkpoints[checkpoints/*.json]
    end

    schedule --> enforce
    schedule --> catalog
    schedule --> checkpoint
    push --> enforce
    push --> catalog
    manual --> enforce

    enforce --> scan --> validate --> enforcement
    catalog --> scan --> catalogjson
    checkpoint --> scan --> drift
    checkpoint --> checkpoints

    style enforce fill:#EF4444,color:#fff
    style catalog fill:#3B82F6,color:#fff
    style checkpoint fill:#8B5CF6,color:#fff
```

## ATLAS Code Analysis Engine

```mermaid
flowchart LR
    subgraph Input["Input"]
        code[Source Code]
        config[Config]
    end

    subgraph Analysis["tools/atlas/"]
        cli[cli/index.ts]
        analyzer[analysis/analyzer.ts]
        ast[analysis/ast-parser.ts]
        complexity[analysis/complexity.ts]
        chaos[analysis/chaos-calculator.ts]
    end

    subgraph Core["Core"]
        kilo[core/kilo-integration.ts]
        bridge[core/kilo-bridge.ts]
    end

    subgraph Output["Output"]
        metrics[Complexity Metrics]
        dashboard[Dashboard]
        report[Analysis Report]
    end

    code --> cli --> analyzer
    config --> cli

    analyzer --> ast --> complexity --> metrics
    analyzer --> chaos --> metrics
    analyzer --> kilo

    metrics --> dashboard
    metrics --> report

    style analyzer fill:#6366F1,color:#fff
    style kilo fill:#10B981,color:#fff
```

## Enforcement Stack

```mermaid
flowchart TB
    subgraph L4["Layer 4: GitHub"]
        branch[Branch Protection]
        checks[Required Checks]
        reviews[Required Reviews]
    end

    subgraph L3["Layer 3: CI/CD"]
        workflows[Reusable Workflows]
        security[Security Scanning]
        quality[Code Quality Gates]
    end

    subgraph L2["Layer 2: Local"]
        precommit[Pre-commit Hooks]
        lintstaged[Lint-staged]
        protected[Protected Files Check]
    end

    subgraph L1["Layer 1: AI"]
        claude[CLAUDE.md]
        cursor[.cursorrules]
        kilo[.kilorc]
    end

    L1 --> L2 --> L3 --> L4

    style L4 fill:#EF4444,color:#fff
    style L3 fill:#F59E0B,color:#fff
    style L2 fill:#3B82F6,color:#fff
    style L1 fill:#10B981,color:#fff
```

## File Dependencies

```mermaid
flowchart TD
    subgraph Entry["Entry Points"]
        pkg[package.json]
        devopsts[tools/cli/devops.ts]
        govpy[tools/cli/governance.py]
    end

    subgraph Libs["Shared Libraries"]
        configts[tools/lib/config.ts]
        fsts[tools/lib/fs.ts]
        checkpointpy[tools/lib/checkpoint.py]
    end

    subgraph Templates["Template Sources"]
        tmpl[templates/devops/**]
        manifest[template.json]
    end

    subgraph Hub["Central Hub"]
        metahub[.metaHub/]
        catalog[catalog/]
        checkpoints[checkpoints/]
    end

    pkg --> devopsts
    pkg --> govpy

    devopsts --> configts
    devopsts --> fsts
    devopsts --> tmpl

    govpy --> checkpointpy

    fsts --> manifest
    fsts --> metahub

    govpy --> catalog
    govpy --> checkpoints

    style pkg fill:#6366F1,color:#fff
    style metahub fill:#10B981,color:#fff
```

## Consumer Repository Integration

```mermaid
flowchart LR
    subgraph Meta["Meta-Governance Repo"]
        workflows[Reusable Workflows]
        templates[Templates]
        policies[Policies]
    end

    subgraph Consumer["Consumer Repos"]
        repo1[Project A]
        repo2[Project B]
        repo3[Project C]
    end

    subgraph Usage["Integration Methods"]
        uses[uses: owner/repo/.github/workflows/]
        cli[npx @owner/cli init]
        extends[extends: policies/]
    end

    Meta --> uses --> Consumer
    Meta --> cli --> Consumer
    Meta --> extends --> Consumer

    style Meta fill:#6366F1,color:#fff
    style Consumer fill:#10B981,color:#fff
```

---

## Quick Reference

| Component      | Path                                                  | Purpose               |
| -------------- | ----------------------------------------------------- | --------------------- |
| DevOps CLI     | [tools/cli/devops.ts](../tools/cli/devops.ts)         | Template management   |
| Governance CLI | [tools/cli/governance.py](../tools/cli/governance.py) | Policy enforcement    |
| ATLAS          | [tools/atlas/](../tools/atlas/)                       | Code analysis         |
| Templates      | [templates/devops/](../templates/devops/)             | Golden path templates |
| Workflows      | [.github/workflows/](../.github/workflows/)           | CI/CD automation      |
| Policies       | [.metaHub/policies/](../.metaHub/policies/)           | Governance rules      |
| Catalog        | [.metaHub/catalog/](../.metaHub/catalog/)             | Portfolio inventory   |

---

_Generated from codebase analysis. View on GitHub for interactive diagrams._


## Codebase Statistics

| Metric | Count |
|--------|-------|
| Templates | 17 |
| Workflows | 23 |
| Template Categories | 8 |

*Auto-generated on 2025-11-30*
