# Multi-Claude Coordination Guide
**Managing Parallel Development: ORCHEX (Claude 2) + Libria (Claude 1)**

---

## Quick Reference

| Component | Claude Instance | Directory | Priority | Deadline |
|-----------|----------------|-----------|----------|----------|
| **Libria Solvers** | Claude 1 (This) | `/Libria/` | 🔴 CRITICAL | Librex.Meta: Mar 31 |
| **ORCHEX Engine** | Claude 2 (Other) | `/ORCHEX/` | 🟡 HIGH | Week 6 |
| **Integration** | Both | `/Libria/libria-integration/` | 🟡 HIGH | Week 4 |

---

## For You (Managing Both Instances)

### Week 1 Kickoff Prompts

**To Claude Instance 1 (Libria - This Instance)**:
```
I need you to continue implementing the Itqān Libria Suite. Priority focus:

1. CRITICAL: Librex.Meta implementation (March 31 deadline)
   - Start with code from PROMPT_Librex.Meta.md
   - Create libria-meta/ package
   - Download ASlib benchmark
   - Begin baseline implementations

2. HIGH: Setup libria-integration/ package
   - Implement LibriaRouter class from ATLAS_LIBRIA_INTEGRATION_SPEC.md
   - Create feature extractors for ORCHEX ResearchAgent → features
   - Implement solve_assignment(), route_workflow_step(), etc.

3. Setup infrastructure:
   - Create libria-core/ base package
   - Test Redis/PostgreSQL connections
   - Verify docker-compose setup

Reference: IMPLEMENTATION_MASTER_PLAN.md for full roadmap
```

**To Claude Instance 2 (ORCHEX - New Instance)**:
```
I need you to implement the ORCHEX Engine that will integrate with the Libria Suite.

Background:
- ORCHEX = 40+ research agents for multi-agent AI orchestration
- Dialectical workflows (thesis-antithesis-synthesis)
- Quality gates and validation
- Integrates with Libria solvers for optimization

Your tasks:

1. Week 1: Implement ORCHEX Core
   - Create ORCHEX-core/atlas_core/engine.py with ATLASEngine class
   - Create ORCHEX-core/atlas_core/agent.py with ResearchAgent base class
   - Implement 5-10 initial agents (synthesis, literature_review, hypothesis_gen, critical_analysis, validation)
   - Connect to Redis blackboard (redis://localhost:6379/0)

2. Week 2: Implement Dialectical Workflows
   - Thesis-Antithesis-Synthesis workflow
   - Multi-perspective analysis
   - Quality gates

3. Week 2: Integrate with Libria
   - Import LibriaRouter from ../Libria/libria-integration/
   - Use for agent-task assignment (Librex.QAP)
   - Use for workflow routing (Librex.Flow)

Reference documents:
- /mnt/c/Users/mesha/Downloads/Important/ATLAS_LIBRIA_INTEGRATION_SPEC.md
- /mnt/c/Users/mesha/Downloads/Important/IMPLEMENTATION_MASTER_PLAN.md

Structure:
/ORCHEX/
├── ORCHEX-core/
│   ├── atlas_core/
│   │   ├── engine.py      # ATLASEngine main class
│   │   ├── agent.py       # ResearchAgent base class
│   │   ├── workflow.py    # Dialectical workflows
│   │   └── blackboard.py  # Redis connection
│   └── tests/

See IMPLEMENTATION_MASTER_PLAN.md Section "2.1 ORCHEX Core Architecture" for complete code templates.
```

### Daily Coordination Workflow

**Morning** (Check both instances):
```bash
# 1. Read integration changes
cat /mnt/c/Users/mesha/Downloads/Important/shared/docs/INTEGRATION_CHANGES.md

# 2. Check Redis for shared state
redis-cli
> KEYS ORCHEX:*
> KEYS libria:*

# 3. Review logs
docker-compose logs --tail=50 ORCHEX-core
docker-compose logs --tail=50 libria-api
```

**Evening** (Sync progress):
1. Ask Claude 1: "What did you implement today? Any API changes needed?"
2. Ask Claude 2: "What did you implement today? Any blockers with Libria integration?"
3. Update `shared/docs/INTEGRATION_CHANGES.md` with any changes
4. Run integration tests if both made progress on same integration point

### Week 1 Checklist

**Claude 1 (Libria)**:
- [ ] libria-core/ package created with LibriaSolver base class
- [ ] libria-meta/ Librex.Meta implementation started
- [ ] libria-integration/ LibriaRouter skeleton created
- [ ] Requirements files for all packages created (✅ Done)
- [ ] Docker setup tested (✅ docker-compose.yml ready)

**Claude 2 (ORCHEX)**:
- [ ] ORCHEX-core/ package created
- [ ] ATLASEngine class implemented
- [ ] ResearchAgent base class implemented
- [ ] 5-10 agents implemented
- [ ] Redis blackboard connection working

**Integration** (Both):
- [ ] LibriaRouter can be imported by ORCHEX
- [ ] Mock agent-task assignment test passes
- [ ] Redis connection shared between both systems

### Weekly Integration Tests

**Week 1 Test**: Agent Assignment
```python
# Run this from ORCHEX Engine (Claude 2)
from atlas_core.engine import ATLASEngine
from libria_integration import LibriaRouter

ORCHEX = ATLASEngine()
# Register 10 mock agents
# Create 5 mock tasks

router = LibriaRouter()
assignment = router.solve_assignment(
    agents=list(ORCHEX.agents.values()),
    tasks=tasks
)

assert assignment is not None
assert 'agent_id' in assignment
print(f"✅ Agent assignment working: {assignment['agent_id']}")
```

**Week 2 Test**: Workflow Routing
```python
# Run dialectical workflow
result = ORCHEX.execute_workflow(
    workflow_type="thesis_antithesis_synthesis",
    inputs={'topic': 'test'}
)

assert 'thesis' in result
assert 'antithesis' in result
assert 'synthesis' in result
print("✅ Dialectical workflow with Librex.Flow routing working")
```

### Conflict Resolution

**If API needs to change**:

1. Detecting Instance proposes change in `shared/docs/INTEGRATION_CHANGES.md`:
```markdown
## 2026-01-20 - Proposed Change

**Proposer**: Claude 1 (Libria)
**Change**: Add 'model_type' field to ResearchAgent schema
**Reason**: Need to differentiate claude-3-opus vs gpt-4-turbo for cost prediction
**Impact**: ORCHEX needs to add model_type to ResearchAgent dataclass

**Proposed Schema**:
```python
@dataclass
class ResearchAgent:
    # ... existing fields
    model_type: str  # NEW: "claude-3-opus", "gpt-4-turbo", etc.
```

**Approval Needed**: Claude 2 (ORCHEX)
```

2. You read this to Claude 2: "Claude 1 proposed adding model_type to ResearchAgent. Do you approve?"

3. Claude 2 responds: "Approved. I'll add model_type field to all agents."

4. Both instances update their code

5. You mark as resolved:
```markdown
**Status**: ✅ APPROVED - Implemented in both instances
```

### Debugging Multi-Instance Issues

**Problem**: "LibriaRouter import fails in ORCHEX"

**Solution**:
```python
# In ORCHEX code, add path
import sys
sys.path.append('../Libria/libria-integration')
from libria_integration import LibriaRouter
```

**Problem**: "Redis state not syncing between instances"

**Solution**:
```bash
# Check Redis connection
redis-cli ping

# Verify both can write/read
# From ORCHEX:
redis-cli SET ORCHEX:test "hello from ORCHEX"

# From Libria:
redis-cli GET ORCHEX:test
# Should return "hello from ORCHEX"
```

**Problem**: "Integration test fails - agent features mismatch"

**Solution**:
```python
# Check both instances using same schema
# In ORCHEX:
agent.to_features()  # Should return np.array([skill, workload, ...])

# In Libria:
self._extract_agent_features(agents)  # Should match ORCHEX format

# If mismatch, update schema in shared/schemas/agent_schema.json
```

### Progress Tracking Template

Create `WEEKLY_PROGRESS.md`:
```markdown
# Week 1 Progress (Jan 20-26)

## Claude 1 (Libria)
**Completed**:
- ✅ libria-core base package
- ✅ Librex.Meta 60% implemented
- ✅ LibriaRouter skeleton

**In Progress**:
- ⏳ Librex.Meta baselines (40% done)
- ⏳ Librex.QAP implementation (20% done)

**Blocked**:
- None

**Next Week**:
- Complete Librex.Meta baseline implementations
- Start Librex.QAP cost prediction
- Begin Librex.Flow LinUCB

## Claude 2 (ORCHEX)
**Completed**:
- ✅ ATLASEngine class
- ✅ ResearchAgent base
- ✅ 8 research agents implemented

**In Progress**:
- ⏳ Dialectical workflows (50% done)
- ⏳ LibriaRouter integration (30% done)

**Blocked**:
- Waiting for LibriaRouter.solve_assignment() implementation

**Next Week**:
- Complete thesis-antithesis-synthesis workflow
- Integrate Librex.QAP for assignment
- Test end-to-end with 10 agents

## Integration Status
✅ Docker stack running
✅ Redis connection working
⏳ LibriaRouter partially implemented (60%)
❌ End-to-end test not passing yet (expected - waiting for implementations)

## API Changes This Week
- Added model_type to ResearchAgent schema (approved)

## Issues
None
```

### Git Workflow (if using version control)

```bash
# Setup (one time)
cd /mnt/c/Users/mesha/Downloads/Important
git init
git add .
git commit -m "Initial commit: Itqān Suite with ORCHEX + Libria"

# Daily workflow for YOU (not Claude instances)
# After Claude 1 makes changes:
git add Libria/
git commit -m "Claude 1: Implemented Librex.Meta tournament selection"

# After Claude 2 makes changes:
git add ORCHEX/
git commit -m "Claude 2: Implemented ATLASEngine agent registration"

# After integration changes:
git add shared/
git commit -m "Integration: Updated ResearchAgent schema with model_type"

# View history
git log --oneline --graph --all
```

---

## Emergency Protocols

### If Claude 1 gets stuck:
1. Check PROMPT_[Solver].md for reference implementation
2. Review ACTIONABILITY_AUDIT.md for common issues
3. Test in isolation (without ORCHEX dependency)
4. Use mock data from superprompt examples

### If Claude 2 gets stuck:
1. Check IMPLEMENTATION_MASTER_PLAN.md Section 2.1 for ORCHEX architecture
2. Review ATLAS_LIBRIA_INTEGRATION_SPEC.md for integration patterns
3. Use mock LibriaRouter initially (return dummy data)
4. Test agents in isolation first

### If integration fails:
1. Test each component in isolation
2. Check Redis connectivity: `redis-cli ping`
3. Verify schemas match in `shared/schemas/`
4. Review `docker-compose logs`
5. Check `ATLAS_LIBRIA_INTEGRATION_SPEC.md` Section 3 (Data Schemas)

---

## Quick Commands Reference

```bash
# Start full stack
docker-compose up -d

# Stop everything
docker-compose down

# Rebuild after code changes
docker-compose up --build

# View logs
docker-compose logs -f [ORCHEX-core|libria-api|redis|postgres]

# Run tests
docker-compose exec ORCHEX-core pytest
docker-compose exec libria-api pytest

# Access Jupyter for experiments
open http://localhost:8888

# Monitor with Grafana
open http://localhost:3000

# Redis CLI
docker-compose exec redis redis-cli
```

---

## Success Indicators

**Week 1**:
- ✅ Both instances can import each other's code
- ✅ Docker stack runs without errors
- ✅ Redis shared state working
- ✅ First integration test passes (agent assignment mock)

**Week 2**:
- ✅ Real agent-task assignment working end-to-end
- ✅ Dialectical workflow routing via Librex.Flow
- ✅ < 5 integration conflicts total

**Week 4**:
- ✅ All 6 integration points working
- ✅ 40+ ORCHEX agents operational
- ✅ Librex.Meta benchmarking started

**Week 12**:
- ✅ Librex.Meta paper submitted (March 31)
- ✅ Full ORCHEX+Libria suite deployed
- ✅ End-to-end research workflows operational

---

## Final Tips

1. **Start Simple**: Get 1 agent + 1 solver working before scaling to 40 agents + 7 solvers
2. **Test Early**: Don't wait for full implementation - test integration points as you build
3. **Document Changes**: Always update `shared/docs/INTEGRATION_CHANGES.md`
4. **Use Mock Data**: Both instances can develop in parallel using mock data initially
5. **Weekly Sync**: Every Friday, run full integration test suite

**Most Important**: Both Claude instances have complete specifications. They can work independently and integrate later. Don't block Claude 2 waiting for Claude 1 or vice versa.

---

**You've got this!** 🚀

Both Claude instances are fully equipped to build their components. Just coordinate at the integration boundaries and you'll have a complete ORCHEX + Libria suite in 12 weeks.
