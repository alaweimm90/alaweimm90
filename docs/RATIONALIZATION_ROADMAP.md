# Repository Rationalization Roadmap

## Strategy: One Project at a Time

Work incrementally on self-contained projects to build momentum and refine the process.

## Prioritized Project List

### Tier 1: Quick Wins (Start Here)
*Small, self-contained projects - easiest to rationalize*

| Project | Files | Org | Priority | Complexity |
|---------|-------|-----|----------|------------|
| **AlaweinOS/Benchmarks** | 19 | AlaweinOS | 🟢 HIGH | Low |
| **alaweimm90-science/QMatSim** | 60 | Science | 🟢 HIGH | Low |
| **alaweimm90-science/MagLogic** | 71 | Science | 🟢 HIGH | Medium |
| **alaweimm90-business/BenchBarrier** | 81 | Business | 🟡 MEDIUM | Medium |

**Why start here:**
- Small file counts (manageable scope)
- Self-contained (limited dependencies)
- Good learning opportunity
- Fast wins build confidence

### Tier 2: Medium Projects
*Moderate size - do after mastering Tier 1*

| Project | Files | Org | Priority | Complexity |
|---------|-------|-----|----------|------------|
| **alaweimm90-science/SpinCirc** | 103 | Science | 🟡 MEDIUM | Medium |
| **AlaweinOS/CrazyIdeas** | 232 | AlaweinOS | 🟡 MEDIUM | High (experiments) |

**Approach:**
- Apply lessons learned from Tier 1
- May have more duplication/experiments
- Good candidates for CLI + Archive pattern

### Tier 3: Large Projects (Later)
*Complex projects - do last after process is refined*

| Project | Org | Notes |
|---------|-----|-------|
| **AlaweinOS/MEZAN** | AlaweinOS | Major system - multiple subsystems |
| **AlaweinOS/TalAI** | AlaweinOS | 28+ modules - significant effort |
| **AlaweinOS/Optilibria** | AlaweinOS | Well-structured already |

**Why later:**
- Already have established structure
- High impact of mistakes
- Need refined process first

## Recommended Starting Project: AlaweinOS/Benchmarks

### Why This One First?

✅ **Smallest** (only 19 files)
✅ **Clear purpose** (benchmarking utilities)
✅ **Self-contained** (few external dependencies)
✅ **Quick win** (can complete in 1-2 hours)
✅ **Learning opportunity** (establish the pattern)

### Expected Outcomes

**Before:**
```
AlaweinOS/Benchmarks/
├── Various benchmark scripts (19 files)
└── Possibly duplicated utilities
```

**After:**
```
AlaweinOS/
├── bin/
│   └── benchmark-suite (CLI tool)
└── .archive/
    └── benchmarks-src/
        ├── MANIFEST.json
        ├── COMPILE_INFO.md
        └── src/ (original 19 files)
```

**Impact:**
- Active files: 19 → 1 (CLI)
- Usability: ↑ (single command vs scattered scripts)
- Maintainability: ↑ (clear structure)

## Process for Each Project

### 1. Analysis (15 min)
```bash
# Navigate to project
cd organizations/AlaweinOS/Benchmarks

# Count files and structure
tree -L 3
find . -type f | wc -l

# Identify duplication
grep -r "def " . | sort | uniq -d

# Check for TODO/FIXME
grep -r "TODO\|FIXME" .
```

### 2. Planning (15 min)
- [ ] Identify core functionality
- [ ] List dependencies
- [ ] Determine CLI interface
- [ ] Plan archive structure
- [ ] Document decisions

### 3. Execution (30-60 min)
- [ ] Extract to consolidated module
- [ ] Create CLI entry point
- [ ] Write tests
- [ ] Build executable
- [ ] Create archive with manifest
- [ ] Update documentation

### 4. Validation (15 min)
- [ ] Test CLI works
- [ ] Verify archive completeness
- [ ] Update docs
- [ ] Commit changes

**Total time per project: 1.5-2 hours**

## Progress Tracking

### Completed Projects
- [ ] AlaweinOS/Benchmarks
- [ ] alaweimm90-science/QMatSim
- [ ] alaweimm90-science/MagLogic
- [ ] alaweimm90-business/BenchBarrier
- [ ] alaweimm90-science/SpinCirc
- [ ] AlaweinOS/CrazyIdeas

### Metrics

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Total Active Files | TBD | - | ↓ 80% |
| Projects Rationalized | 0 | - | 6 |
| CLIs Created | 0 | - | 6 |
| Archives Established | 0 | - | 6 |

## Next Steps

1. **Start with AlaweinOS/Benchmarks**
   - Smallest project
   - Fastest win
   - Establish the pattern

2. **Document learnings**
   - What worked well
   - What to improve
   - Template refinements

3. **Apply to QMatSim**
   - Slightly larger
   - Validate process scales

4. **Iterate and refine**
   - Improve automation
   - Refine templates
   - Build confidence

## Commands Cheatsheet

```bash
# Navigate to root
cd /mnt/c/Users/mesha/Desktop/GitHub

# Start with first project
cd organizations/AlaweinOS/Benchmarks

# Analyze
tree -L 3
find . -type f -name "*.py" | xargs wc -l | tail -1

# Create archive structure
mkdir -p .archive/benchmarks-src
git log --oneline -- . > .archive/benchmarks-src/git-history.txt

# Build CLI (example)
pyinstaller --onefile src/benchmark_cli.py

# Generate manifest
cat > .archive/benchmarks-src/MANIFEST.json << EOF
{
  "cli_name": "benchmark-suite",
  "version": "1.0.0",
  "compiled_date": "$(date -I)",
  "source_location": ".archive/benchmarks-src/",
  "git_commit": "$(git rev-parse HEAD)"
}
EOF
```

## Decision Log

### Project Selection Criteria
1. **Size:** Smaller projects first
2. **Complexity:** Self-contained before interdependent
3. **Impact:** Quick wins before major refactors
4. **Learning:** Use early projects to refine process

### Archive Strategy
- Keep ALL source code (never delete)
- Full git history preserved
- Manifest with rebuild instructions
- Index for easy navigation

### CLI Strategy
- One CLI per logical grouping
- Well-documented usage
- Versioned releases
- Tests included

---

**Start small. Build momentum. Refine the process. Scale gradually.**
