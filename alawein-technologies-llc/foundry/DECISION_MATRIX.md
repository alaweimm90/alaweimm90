# Decision Matrix & Tools

**Frameworks and matrices to help you make better decisions faster.**

Use these tools when you're facing a tough choice.

---

## 📋 Decision #1: Which Product Should You Build?

### Matrix: Product Scoring

**Instructions:**
1. Score each product 1-10 on each dimension
2. Multiply by weight
3. Total score = recommendation

| Product | Feasibility (×2) | Impact (×2) | Timeline (×1) | Personal Fit (×1) | **Total Score** |
|---------|------------------|------------|---------------|--------------------|---|
| NIGHTMARE MODE | 10×2=20 | 10×2=20 | 8×1=8 | ?×1=? | ? |
| CHAOS ENGINE | 9×2=18 | 9×2=18 | 9×1=9 | ?×1=? | ? |
| RESEARCH PRISON | 9×2=18 | 8×2=16 | 10×1=10 | ?×1=? | ? |
| SCIENTIFIC TINDER | 7×2=14 | 9×2=18 | 8×1=8 | ?×1=? | ? |
| Other | ?×2=? | ?×2=? | ?×1=? | ?×1=? | ? |

**Score interpretation:**
- 80-100: **Build this!** (Green light)
- 60-79: **Good option** (Yellow light)
- <60: **Probably not** (Red light)

**See:** PRIORITY_RANKING.md for detailed scoring

---

## 📊 Decision #2: Should You Validate or Build?

### Decision Tree

```
Have you validated the idea?

├─ NO
│  ├─ Have you talked to 20+ potential customers?
│  │  └─ NO → You need to validate! Go to VALIDATION_FIRST.md
│  │
│  ├─ Do 70%+ identify the problem?
│  │  └─ NO → Pivot or choose different idea
│  │
│  ├─ Do you have 10+ pre-sales?
│  │  └─ NO → Either continue validating or pivot
│  │
│  └─ YES → Ready to build!
│
└─ YES → Go to IMPLEMENTATION_GUIDE.md
```

### Validation Success Criteria

| Criteria | Target | Your Result |
|----------|--------|------------|
| Customer interviews | 20+ | _____ |
| % identifying problem | 70%+ | _____ |
| Pre-sales commitments | 10+ | _____ |
| Average price point | $99-299 | _____ |
| Customer urgency | High | _____ |

**If all criteria met:** ✅ **BUILD**
**If <3 criteria met:** ❌ **PIVOT or STOP**

---

## 🎯 Decision #3: Build vs. Outsource?

### Build vs. Outsource Matrix

| Task | Build In-House | Outsource | Hybrid |
|------|----------------|-----------|--------|
| **MVP Development** | If you have engineer | If bootstrapped | Not recommended |
| **Design** | If you have designer | Most common | Recommended |
| **Content Writing** | If founder good writer | Most common | Recommended |
| **Customer Support** | After MVP | Initially or contract | Recommended |
| **Infrastructure** | Lead engineer needed | Possible (costly) | Recommended |
| **Bookkeeping** | Not recommended | Recommended | Not needed |

**Decision factors:**
- Do you have internal talent? (Build)
- Is it core to business? (Build)
- Is it one-time task? (Outsource)
- Is it expensive to do poorly? (Build)
- Do you have budget? (Outsource)

---

## 💰 Decision #4: Bootstrap vs. Raise Capital?

### Bootstrap vs. Fundraising Matrix

| Factor | Bootstrap | Raise Capital |
|--------|-----------|------------------|
| **Speed** | Slower (6-12 months) | Faster (3-6 months) |
| **Equity** | 100% yours | Diluted (but ~$) |
| **Pressure** | Moderate | High (investor expectations) |
| **Network** | Your own | Investor connections |
| **Flexibility** | High (pivot easily) | Lower (investor constraints) |
| **Cash runway** | Short (3-6 months) | Long (18-24 months) |
| **Success rate** | Lower numbers, higher % | More funding, similar % |
| **Control** | You decide everything | Board/investor input |

**Decision tree:**

```
Do you have $50K+ runway?
├─ YES → Consider bootstrapping to MVP
│  ├─ Can you reach profitability in 12 months?
│  │  ├─ YES → Bootstrap! (saves dilution)
│  │  └─ NO → Raise capital
│
└─ NO → Raise capital or get very creative
   ├─ Can you reach customers cheaply?
   │  ├─ YES → Bootstrap on sweat equity
   │  └─ NO → Raise capital
```

**Recommendation:** Bootstrap to MVP, then decide

---

## 🎯 Decision #5: Hire Now or Later?

### Hiring Decision Matrix

**When to hire engineer:**
- [ ] You have working MVP
- [ ] You have 10+ paying customers
- [ ] You're bottlenecked on engineering (not sales/product)
- [ ] You have $5K+/month recurring revenue

**When to hire designer:**
- [ ] Your UI is clearly hurting retention
- [ ] You can't improve it yourself
- [ ] You have budget ($3-5K/month or $10-30K project)

**When to hire marketer:**
- [ ] Your product is working well
- [ ] You need help scaling growth
- [ ] You have budget ($3-5K/month)

**When to hire PM:**
- [ ] You have team of 4+ engineers
- [ ] You can't decide on roadmap yourself
- [ ] You need someone to talk to customers

**When to hire operations:**
- [ ] You have 5+ employees
- [ ] Founder spending 20%+ time on admin
- [ ] You need structured processes

**General rule:** Hire when you're 50% blocked by lack of that role

---

## 🔀 Decision #6: Pivot or Persevere?

### Pivot Decision Framework

**You should pivot if:**
- [ ] Validation failed (<10 pre-sales, <70% problem identification)
- [ ] Users aren't using core feature (low retention)
- [ ] Customer acquisition cost > lifetime value
- [ ] Team is misaligned on direction
- [ ] Market changed significantly
- [ ] Better opportunity emerged

**You should persevere if:**
- [ ] You have paying customers using product
- [ ] Retention is >80% monthly
- [ ] CAC < LTV/3
- [ ] Team is excited and aligned
- [ ] You're learning with each iteration
- [ ] Growth is trending positive

### Pivot Options

**Pivot type 1: Target Customer**
- Keep product same
- Sell to different customer type
- Example: "students" → "teachers"

**Pivot type 2: Problem**
- Keep solution same
- Solve different problem
- Example: "scheduling" → "time blocking"

**Pivot type 3: Distribution**
- Keep product same
- Use different sales channel
- Example: "direct sales" → "self-serve"

**Pivot type 4: Monetization**
- Keep product same
- Change revenue model
- Example: "$99/month" → "freemium"

**Pivot type 5: Core product**
- Complete product change
- Keep market understanding same
- Example: "App" → "Community"

**Which pivot?** The one with least wasted work

---

## 📈 Decision #7: Launch Now or Wait?

### Launch Readiness Checklist

**Must have (all required):**
- [ ] MVP works end-to-end
- [ ] 80%+ test coverage
- [ ] Security audit completed
- [ ] Legal/compliance reviewed
- [ ] Performance acceptable (<2s load time)

**Should have (most needed):**
- [ ] Customer support process ready
- [ ] Monitoring/alerting set up
- [ ] Marketing plan drafted
- [ ] 10 beta customers ready
- [ ] Rollback plan documented

**Nice to have (ideal):**
- [ ] Mobile optimized
- [ ] Advanced features complete
- [ ] Video demo prepared
- [ ] Case studies written
- [ ] Press release prepared

**Scoring:**
- 5/5 "Must have" → **LAUNCH NOW**
- 4/5 "Must have" + 4/7 "Should" → **LAUNCH WITH CAUTION**
- <4/5 "Must have" → **WAIT 1-2 MORE WEEKS**

---

## 🎯 Decision #8: Scale or Build Product #2?

### When to Scale vs. When to Launch Product #2

**Scale your existing product if:**
- [ ] You have 50+ paying users
- [ ] $2-5K MRR
- [ ] 80%+ retention
- [ ] Strong product-market fit signals
- [ ] Team wants to scale same product
- [ ] Market is growing

**Launch product #2 if:**
- [ ] You've reached $10K+ MRR
- [ ] First product has good retention (can coast on team)
- [ ] Different market opportunity is obvious
- [ ] You're excited about new product
- [ ] You have team capacity

**Both if:**
- [ ] You have 8+ person team
- [ ] Multiple VPs/department leads
- [ ] You can split resources

**Recommendation:** Scale to $10K MRR before product #2

---

## 🏆 Decision #9: When to Raise Capital?

### Fundraising Readiness Checklist

**Should raise if:**
- [ ] Proven product-market fit (10%+ MOD growth)
- [ ] Experienced founding team
- [ ] Large addressable market
- [ ] Competitive advantage
- [ ] Clear path to profitability
- [ ] Team already in place

**Should NOT raise if:**
- [ ] Still validating idea
- [ ] Market is unclear
- [ ] Solo founder (risky to investors)
- [ ] Can bootstrap to profitability
- [ ] Don't want to dilute equity

**Best timing:** After $5-10K MRR, when growth is obvious

---

## 🤖 Decision #10: AI/Automation Decisions

### Build with AI or Traditional Code?

**Use AI if:**
- [ ] Solving NLP problem (text understanding)
- [ ] Need content generation
- [ ] Building conversational interface
- [ ] Need recommendation engine
- [ ] Competitive advantage = AI

**Use traditional code if:**
- [ ] Simple CRUD operations
- [ ] Deterministic logic needed
- [ ] Edge cases are critical
- [ ] Cost is more important
- [ ] Team prefers traditional

**Hybrid approach (recommended):**
- Traditional code for core logic
- AI for enhancement features
- Start simple, add AI later

---

## 📝 Decision Log Template

**Keep a record of all major decisions:**

```markdown
# Decision: [Decision name]

**Date:** YYYY-MM-DD
**Context:** [Why this decision matters]
**Options considered:**
- Option A
- Option B
- Option C

**Decision:** Option X

**Reasoning:**
- Pro A: ...
- Pro B: ...
- Con C: ...

**Success metrics:**
- Metric 1: ___
- Metric 2: ___

**Review date:** YYYY-MM-DD (2 weeks later)
```

---

## 🎯 Quick Decision Templates

### "Should I do X?"

Ask yourself:

1. **Impact:** How much will this affect success? (1-10)
2. **Effort:** How much work is it? (1-10)
3. **Urgency:** How soon must we decide? (1-10)
4. **Confidence:** How confident am I in the answer? (1-10)

**Score:** Impact × Confidence ÷ (Effort × 0.5)

**If score >10:** DO IT
**If score 5-10:** PROBABLY
**If score <5:** SKIP IT

---

### "This or That?"

Comparison template:

| Factor | Option A | Option B | Weight |
|--------|----------|----------|--------|
| Cost | $X | $Y | ×2 |
| Speed | X weeks | Y weeks | ×2 |
| Quality | A | B | ×1 |
| Risk | High/Med/Low | High/Med/Low | ×2 |
| Alignment | Yes/No | Yes/No | ×1 |

**Weight highest factors most**

---

## 🚀 Decision-Making Tips

1. **Set decision deadline** - Don't overthink
2. **Gather data** - Make it evidence-based
3. **Trust your gut** - After analysis, go with instinct
4. **Get feedback** - Ask trusted advisors
5. **Remember:** Wrong decision made fast > right decision made slow
6. **Document it** - Write down your reasoning
7. **Review it** - Revisit after 2-4 weeks
8. **Learn from it** - What would you do differently?

---

**Use these frameworks to make confident decisions fast!** 🎯

*Still stuck? See WHEN_STUCK.md*
