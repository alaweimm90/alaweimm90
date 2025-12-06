# 🚨 CRISIS MANAGEMENT PLAYBOOK

Complete response protocols for launch week emergencies. What to do when X goes wrong.

---

## QUICK REFERENCE MATRIX

| Scenario | Severity | Detection Time | Response Time | Who Handles | Status Impact |
|----------|----------|---|---|---|---|
| Server crashes | Critical | 1m | 5m | Tech Lead | High |
| Negative HN comments | High | Immediate | 30m | Comms Lead | Medium |
| Zero applications (24h) | Medium | 24h | 2h | Marketing | Medium |
| Bug in algorithm | High | Test phase | 2h | CTO | High |
| Bad data in benchmark | High | Review | 1h | Research | Medium |
| Partner goes silent | Low | 3 days | N/A | Biz Dev | Low |
| Team burnout | Medium | Ongoing | Ongoing | Leadership | Medium |
| Press/media inquiry | Low | Varies | 1h | CEO | Low |
| GitHub issue flood | Low | Ongoing | 30m | Community | Low |
| API rate limiting | Medium | 6h | 1h | DevOps | Medium |

---

## SCENARIO 1: API SERVER CRASHES

**Severity:** 🔴 CRITICAL
**Detection time:** 1-2 minutes (automated alerts)
**Response window:** 5 minutes maximum

### What This Looks Like

- Health check endpoint returns 500 error
- Dashboard can't connect to API
- Users report "API unavailable"
- Monitoring alerts fire

### Immediate Response (First 5 minutes)

**Person responsible:** Tech Lead

**Step 1: Assess (1 minute)**
```bash
# SSH into server
ssh your-server

# Check server logs
tail -f /var/log/Librex.QAP/api.log

# Check system resources
top
ps aux | grep python
```

**Step 2: Identify cause (2 minutes)**
- Out of memory?
- Port conflict?
- Dependency missing?
- Code error?
- Database connection issue?

**Step 3: Emergency protocol (2 minutes)**

**Option A: Memory issue**
```bash
# Restart server with reduced resource usage
python server.py --workers=1 --pool-size=10
```

**Option B: Code error**
```bash
# Rollback to last known good version
git checkout HEAD~1
python server.py
```

**Option C: Dependency**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
python server.py
```

**Option D: Unknown (nuclear option)**
```bash
# Full restart
docker-compose down
docker-compose up --build
```

### Public Communication (Immediate)

**Who posts:** CEO/Comms Lead

**Twitter/Status page message:**
```
⚠️ We experienced a brief API service disruption (5:42pm PT).
Our team is investigating and restoring service immediately.
We'll post an update in 10 minutes. Thank you for patience.
```

**Status code:** Yellow (service degraded)

### Follow-up Communication (After fix)

**Once server is back up (within 15 minutes):**

**Twitter:**
```
✅ API service restored (5:52pm PT).
Root cause: [brief explanation, not technical jargon]
No data was lost. All service fully operational.
Incident report will be published tomorrow.
Thanks for bearing with us.
```

**Status code:** Green (operational)

**Incident review:** Post-mortem email to team within 24 hours
- What happened
- Why it happened
- How we're preventing it next time
- System improvements (redundancy, monitoring, etc.)

### Prevention Measures

```
For future:
☐ Add load balancer (handle traffic spikes)
☐ Set up auto-restart (systemd or supervisor)
☐ Add monitoring alerts (check every 30s)
☐ Add redundant API (2+ instances)
☐ Set memory limits (prevent OOM)
☐ Add rollback mechanism (automated)
```

---

## SCENARIO 2: NEGATIVE HACKER NEWS COMMENTS

**Severity:** 🟠 HIGH
**Detection time:** Immediate (monitoring)
**Response window:** 30 minutes (to avoid stoking further negativity)

### What This Looks Like

- HN post gets negative comments
- Common complaints:
  - "Just another quantum hype project"
  - "Benchmark claims are inflated"
  - "No real quantum advantage demonstrated"
  - "Research is incremental"
  - "Overselling the approach"

### Analysis Protocol (5 minutes)

**Read carefully. Is the criticism:**

- [ ] **Valid?** (Points out real limitation or false claim)
- [ ] **Unfair?** (Misunderstands the work or motives)
- [ ] **Bad faith?** (Purely negative with no substance)

### Response Strategy by Type

#### A) Valid Criticism

**Do:**
- Thank commenter
- Acknowledge the point
- Clarify if there's misunderstanding
- Admit limitations

**Example response:**
```
Thanks for pointing this out. You're right that:
[specific point]. We designed the system this way
because [explanation]. This is a limitation we
acknowledge and plan to address in [timeline].

Appreciate the feedback!
```

**Don't:**
- Get defensive
- Make excuses
- Argue technicalities
- Dismiss the concern

#### B) Unfair/Misunderstood Criticism

**Do:**
- Clarify with evidence
- Link to paper/code
- Provide specific numbers
- Be patient

**Example response:**
```
We appreciate the skepticism. To clarify:

Our claim is NOT quantum advantage (we're explicit
about this being a classical baseline). Rather, we've
developed a strong classical competitor, which the
quantum field actually needs for honest comparisons.

See Section 8.2 of the paper for detailed discussion
of limitations. Code is public for reproducibility.

Thanks for reading!
```

**Don't:**
- Act wounded
- Over-explain
- Link spam
- Respond multiple times to same person

#### C) Bad Faith Attacks

**Do:**
- One calm response
- Don't feed the troll
- Move to next commenter
- Stay professional

**Example response:**
```
We're confident in the work. Code, data, and paper
are all public for independent verification. We
welcome technical criticism grounded in the methods.
```

**Then stop responding to that commenter.**

### Communication Plan

**Who responds:** CTO or Research Lead (not CEO)
- Shows technical credibility
- Avoids defensive tone
- Demonstrates openness

**Timing:**
- First response: Within 30 minutes
- Follow-ups: Only if new substantive points
- Final cutoff: After 4 hours (let it die naturally)

### Escalation

**If negativity spreads (100+ negative comments):**

Contact HN moderators:
```
Subject: Moderation request - Librex.QAP-new thread

The Librex.QAP-new thread has attracted discussion.
Some comments are moving toward personal attacks
rather than technical critique. We'd appreciate
moderation if this continues.

We're committed to transparency and technical discourse.
```

**Document everything:**
- Screenshots of comments
- Timeline of responses
- Impact on metrics (if any)
- Learnings for next time

---

## SCENARIO 3: ZERO JOB APPLICATIONS (FIRST 24 HOURS)

**Severity:** 🟡 MEDIUM
**Detection time:** 24 hours
**Response window:** 2 hours (adjust before day 2)

### What This Looks Like

- Job postings went live Monday
- By Tuesday morning: 0 applications
- GitHub stars: Fine (100+)
- Other metrics: Good
- But: No one is applying

### Root Cause Analysis (30 minutes)

**Check:**

1. **Are postings visible?**
   - [ ] Can you find them on each platform?
   - [ ] URL search: "Librex.QAP" on site:github.com/jobs
   - [ ] LinkedIn job listing shows up in search

2. **Posting quality?**
   - [ ] Salaries visible?
   - [ ] Job descriptions clear?
   - [ ] Application process easy?
   - [ ] Email/link working?

3. **Timing issue?**
   - [ ] Posted too late Monday?
   - [ ] Right before holiday/weekend?
   - [ ] Bad time zone?

4. **Targeting issue?**
   - [ ] Right platforms?
   - [ ] Right audience?
   - [ ] Right seniority level?

### Immediate Adjustments (30 minutes)

**Option A: Repost on different platforms**
```
If not posted yet on:
☐ Reddit r/MachineLearning (high traffic)
☐ Reddit r/QuantumComputing (targeted)
☐ arXiv ML jobs section
☐ Physics jobs board
☐ Signal to early GitHub followers
```

**Option B: Refine job descriptions**

Before (too generic):
"Senior Optimization Engineer needed"

After (compelling):
"Build quantum-inspired algorithms: $150K + equity
We're advancing optimization beyond classical limits.
You'll publish, impact real problems, own a research direction."

**Option C: Pricing adjustment**

If no applicants, consider:
- Lower salary requirement temporarily ($100K → $80K minimum)
- Add more equity (% + 4-year vest clarified)
- Remote-first emphasis
- "Equity valued at $X" (show the number)

**Option D: Targeted outreach**

Don't just post. Reach out directly:
```
Email to 10-20 candidates you know:
Subject: Open role at Librex.QAP (custom for you)

[Name],

Given your work in [specific field], this might be interesting.
Senior [Role] at Librex.QAP-new.

[2-3 compelling sentences]
[Link to job]

Let's chat briefly if interested?

[Your name]
```

### Escalation (if still zero by Wednesday)

**Wednesday morning decision:**
- [ ] Are fundamentals right? (Role, salary, platform)
- [ ] Is the company positioning wrong?
- [ ] Is timing just slow?

**Actions:**
1. Post personal message to Twitter/Reddit:
   ```
   "We're hiring 5 roles. If you're interested in
   optimization + quantum + ML, send me a DM.
   Let's talk."
   ```

2. Reach out to:
   - Colleagues from prior work
   - Professors/lab connections
   - Friends in quantum/ML communities
   - Former classmates working in target roles

3. Consider temp contractor:
   - If waiting for hires is blocking
   - Use Upwork/Toptal for short-term work
   - Buy time while recruiting

### Prevention for Future Campaigns

```
☐ Post to at least 5 platforms simultaneously
☐ Test with 2-3 friends before posting (clarity check)
☐ Post on weekday morning (9am your timezone)
☐ Send personal outreach email on day 1 (don't rely on postings)
☐ Track applications by source (which platform worked)
☐ Have 2nd/3rd tier platforms ready (Reddit, niche sites)
```

---

## SCENARIO 4: ICML PAPER REJECTED

**Severity:** 🟡 MEDIUM
**Detection time:** May 2025 (4 months away)
**Response window:** 1 week

### What This Looks Like

- Decision email from ICML
- Subject: "ICML 2025 Submission Decision"
- Decision: Reject with reviewer comments

### Immediate Response (1 hour)

**Don't:**
- Email conference arguing decision
- Publicly complain about reviewers
- Get discouraged

**Do:**
- Read all reviewer comments carefully
- Identify common themes
- Note any valid criticisms

### Analysis Protocol (1 day)

**Questions to answer:**
1. Are reviewers pointing to missing comparisons?
2. Are there theoretical gaps in the proofs?
3. Is the writing unclear?
4. Are results questioned as reproducible?
5. Is the novelty insufficient (incremental)?

### Resubmission Strategy (1 week)

**Option A: NeurIPS 2025 (Sept deadline)**

Timeline:
- Week 1-2: Revise based on ICML feedback
- Week 3: Submit to NeurIPS
- Dec: NeurIPS decision

**Option B: ArXiv + Journal submission**

- Post to arXiv immediately (version 2)
- Submit to Journal of Machine Learning Research (JMLR)
- Lower bar, longer publication timeline (6-12 months)
- But: Fully published, peer-reviewed

**Option C: Maintain momentum anyway**

Even if rejected, continue:
- Publish on arXiv
- Submit to NeurIPS
- Use working paper in recruiting/partnerships
- Keep building community

### Communication to Team

**Internal message:**
```
ICML decision came back today—not accepted.

Reviewers raised [specific point], which is valid
and we'll address. This doesn't diminish the work.

Plan:
1. ArXiv posting this week (version 2)
2. NeurIPS submission in [timeline]
3. Continue partnerships/hiring (papers take time)

Strong papers often get rejected first. We'll keep iterating.
```

**Public message (optional):**
```
Our ICML submission didn't make the cut, but
the reviewer feedback is constructive.

We're publishing on arXiv this week +
heading to NeurIPS next.

Publication is one metric. Code, community,
and impact matter more.
```

### Prevention

```
For next paper:
☐ Submit to workshop first (lower bar, faster feedback)
☐ Get feedback from 3+ external reviewers before submission
☐ Test claims rigorously before publishing
☐ Have backup venue (NeurIPS if ICML, JMLR always)
```

---

## SCENARIO 5: BUG IN BENCHMARK RESULTS

**Severity:** 🔴 CRITICAL
**Detection time:** Varies (pre-launch or post-launch)
**Response window:** 2 hours if critical

### What This Looks Like

- Someone reports: "Your FFT-Laplace numbers seem high"
- You verify: Math error in benchmark script
- Effect: Results are wrong by 2-5%

### If Discovered BEFORE Launch

**Lucky. You're done in 1 hour:**

```bash
# Fix the bug
vim Librex.QAP/benchmark.py  # Fix the issue

# Re-run benchmarks
python Librex.QAP/benchmark.py

# Verify results
# (expected 0.54% average gap, not the wrong number)

# Update paper with correct numbers
vim docs/papers/DRAFT_FFT_LAPLACE_ICML2025.md

# Commit
git add .
git commit -m "Fix: benchmark calculation error"
git push
```

**No public communication needed.** Bug was caught internally.

### If Discovered AFTER Launch

**More serious. Follow this:**

**Step 1: Verify the bug (30 minutes)**
```bash
# Reproduce the issue
python test_integration.py  # Does it show wrong numbers?

# Check git history
git log --oneline Librex.QAP/benchmark.py

# Understand impact
# Was this in the paper? Social media? Proposals?
```

**Step 2: Assess severity (15 minutes)**
- [ ] Wrong by < 1%: Minor (rounding, numerical precision)
- [ ] Wrong by 1-5%: Medium (affects competitive comparison)
- [ ] Wrong by > 5%: Major (fundamental issue)

**Step 3: Public response (30 minutes)**

**What to say (honest and direct):**

```
🔴 UPDATE: Benchmark Results

We discovered a calculation error in our
benchmark suite today.

Impact: Results overstated by [X]%
Example: FFT-Laplace gap reported as 0.54%,
         actually 0.58% on QAPLIB instances

What we're doing:
✓ Fixed the bug (code updated)
✓ Re-ran all benchmarks
✓ Verified new numbers
✓ Updating paper & posts with correct figures

We're committed to accuracy above all.
Errors happen; transparency matters.

Full report: [link to corrected numbers]
```

**Post to:**
- Twitter (transparent)
- HackerNews (if thread still active)
- GitHub (issue + fix)
- Email to partners/investors

**Don't:**
- Hide it (it will come out)
- Blame tools/libraries
- Minimize ("just a small error")
- Claim results still show leadership (might not)

### Corrective Actions

```bash
# Update code
git checkout develop
git pull
# (bug fix already applied)

# Re-run full test suite
python test_integration.py  # Ensures fix works

# Update paper figures
# (with correct numbers)

# Update social media claims
# (remove overstated claims)

# Post-mortem
# (how to prevent this next time)
```

### Prevention

```
Before publishing ANY numbers:
☐ Have 2 different people run benchmarks
☐ Compare results (do they match?)
☐ Use published baseline to sanity-check
☐ Add unit tests for benchmark calculations
☐ Include uncertainty ranges (confidence intervals)
☐ Document calculation methodology in paper
```

---

## SCENARIO 6: PARTNERSHIP GOES SILENT

**Severity:** 🟢 LOW
**Detection time:** 3-5 days
**Response window:** N/A (no emergency)

### What This Looks Like

- Sent partnership proposal to IBM on Monday
- Got enthusiastic initial response
- Radio silence since Wednesday

### Normal expectations

**Days 0-2:** Quick response (CEO scans, forwards to team)
**Days 3-7:** First evaluation (team discusses)
**Days 8-14:** Request more info or schedule call
**Days 15-30:** Decision or negotiation
**Days 30+:** Signed or rejected

Silence at day 3-5 is **not unusual**. Corporate processes move slow.

### Actions (No Panic Needed)

**Day 5 (if silence continues):**

Send light follow-up:
```
Subject: Re: Librex.QAP Partnership Opportunity

Hi [Name],

Following up on our partnership proposal from Monday.
No pressure—just wanted to make sure it landed safely.

Would be great to chat briefly this week or next
if there's interest.

Looking forward to hearing from you.

[Your name]
```

**Day 10 (if still silence):**

Send one final note:
```
Hi [Name],

Respecting your time, so this is the last ping.

If there's interest, we're very motivated to work together.
If not, no worries—we'll keep building and stay in touch.

All the best,
[Your name]
```

**Then move on.** Don't keep emailing.

### What This Teaches

```
Partnership timeline:
- 0-30 days: Get 5-10% responses
- 30-60 days: Get another 5-10% responses
- 60+ days: Slow trickle

So:
☐ Send 10 partnership proposals (expect 1-2 responses)
☐ Not all of them will respond
☐ That's normal
☐ Keep sending more
☐ Celebrate the ones who do respond
```

---

## SCENARIO 7: TEAM MEMBER BURNING OUT

**Severity:** 🟡 MEDIUM
**Detection time:** Ongoing (watch for signs)
**Response window:** Immediate

### What This Looks Like

- Team member working 12-14 hour days all week
- Friday: "I can't do this pace"
- Sunday: "Not sure if I should come back Monday"
- Sickness: Physical symptoms from stress

### Prevention (Recommended)

**Week 1 is intense. Plan for sustainability:**

**During week 1:**
- Work sprints: 9am-1pm (4 hours focused), break
- Afternoon: meetings/lighter work, not execution
- 5:30pm: standup + plan for next day
- After 5:30pm: Off. No more work.

**Weekend (Sat-Sun):**
- Check email once (morning only)
- Respond to urgent items only
- Mostly off

**Week 2:**
- Dial down to 40 hour weeks
- Celebrate week 1 wins
- Maintain momentum without burning out

### Response (If someone burns out)

**Day 1 (they say they're struggling):**

**Immediate:**
1. Thank them for honesty
2. Acknowledge pace was unsustainable
3. Adjust timeline immediately

**What to say:**
```
I hear you. Week 1 was brutal.
You did incredible work.

Let's dial it back starting now:
- You're off for the next 2 days (fri/sat)
- We'll handle urgent items
- We'll regroup Monday with more sustainable plan
- This launch is a marathon, not a sprint

You matter more than any deadline.
```

4. Redistribute work to others

**Week 2:**
- Normal 40-hour weeks
- Celebrate week 1 success
- Build more gradually

### Prevention Checklist

```
Before launch, agree on:
☐ Max hours per day (9am-6pm = 8 hours, reasonable)
☐ No work after 6pm on weekdays
☐ Weekends mostly off (check email once/day only)
☐ Day off mid-week if possible (Wednesday)
☐ Budget: Celebrate with team lunch at end of week
☐ Clear: This is marathon, not sprint

During week 1:
☐ Check in daily: "How are you holding up?"
☐ If anyone says "struggling," immediately adjust
☐ Celebrate wins (each day, each milestone)
☐ Remember: Code > burnout (we can fix bugs, not people)
```

---

## SCENARIO 8: PRESS/MEDIA INQUIRY

**Severity:** 🟢 LOW (actually positive)
**Detection time:** Varies
**Response window:** 2 hours

### What This Looks Like

- Reporter emails: "Interested in interviewing you about quantum computing"
- Or: "Writing article about optimization—can I feature Librex.QAP?"
- Or: "Podcast wants you as guest"

### Your Response

**Decision matrix:**

| Type | Your timezone match? | Good fit for positioning? | Time available? | Your answer |
|------|---|---|---|---|
| TechCrunch | No | Yes | No | "Interested but timing tight, follow up in 2 weeks" |
| Quantum podcast | Yes | Perfect | Yes | "Yes! Send details" |
| Local news | Yes | Okay | Yes | "Sure, let me know" |
| Academic interview | Yes | Yes | Flexible | "Yes, when?" |

**General response:**

```
Subject: Re: Interview Request - Quantum Optimization

Hi [Name],

Thanks for reaching out! We're excited to talk
about our work.

Quick context:
[1-2 sentences about what you do]

I'm available [specific times/days] this week.
Otherwise, next week would work well.

What's your timeline?

Best,
[Your name]
```

### Messaging (What to say in interview)

**Key points to hit:**
1. "We're building bridges between quantum and classical"
2. "Our contribution: strong classical baseline (others lacked this)"
3. "Impact: enables honest quantum advantage research"
4. "It's open source—reproducible, transparent"
5. "We're hiring + building community"

**Don't say:**
- Technical details reporters won't understand
- Hype about quantum (you're the baseline, stay humble)
- Negative things about competitors
- Unverified claims

### After Interview

**Send thank you email:**
```
Hi [Name],

Thanks for the thoughtful interview.
Looking forward to the piece.

If you need any follow-up details, happy to help.

Best,
[Your name]
```

**Once published:**
- Share on Twitter/LinkedIn (with credit to reporter)
- Add to press coverage section of website
- Send to investors/partners (shows credibility)

---

## SCENARIO 9: GITHUB ISSUE FLOOD

**Severity:** 🟢 LOW (indicator of popularity)
**Detection time:** Continuous
**Response window:** 2-4 hours per day

### What This Looks Like

- 5 issues first day (expected)
- 15 issues by day 3 (good sign)
- 50 issues by day 7 (need process)
- Issues range: bugs, features, docs, questions

### Triage Process (30 minutes daily)

**Each morning, spend 30 min on issues:**

**Step 1: Read all new issues (10 min)**
```
Categorize mentally:
□ Bug (code doesn't work)
□ Feature request (nice to have)
□ Doc (missing documentation)
□ Question (user confusion, not our bug)
□ Duplicate (already reported)
```

**Step 2: Add labels (5 min)**
```
In GitHub:
- Add "bug", "feature", "documentation", "question" labels
- Add "good-first-issue" for easy ones
- Add priority: "critical", "high", "medium", "low"
```

**Step 3: Respond to critical items (15 min)**
```
For bugs:
"Thanks for reporting. Can you:
1. Include your Python version
2. Full error message (paste traceback)
3. Minimal code to reproduce

Help us help you!"

For features:
"Great idea. Adding to our backlog.
Here's where we see it fitting: [roadmap section]"

For docs:
"You're right, this should be clearer.
For now: [quick explanation]
Adding to docs improvements."

For questions:
"Check QUICKSTART.md section X for this.
Let me know if still unclear!"
```

### Escalation

**If issues are arriving faster than you can respond:**

1. Enable issue templates (GitHub)
```
Create .github/ISSUE_TEMPLATE/bug_report.md
Create .github/ISSUE_TEMPLATE/feature_request.md
```

2. Post pinned issue:
```
📌 WELCOME! Quick notes:

Before opening an issue, please:
☐ Check QUICKSTART.md
☐ Search existing issues
☐ Include Python version + full error message

We're reading all issues. Response time: 24-48 hours.
Thanks for being part of the community!
```

3. Ask for help (community moderation):
```
Issue comment:
"Anyone have experience with this?
[User] is trying to [task].

Community input welcome!"
```

---

## SCENARIO 10: API RATE LIMITING NEEDED

**Severity:** 🟡 MEDIUM
**Detection time:** 6-12 hours
**Response window:** 1-2 hours

### What This Looks Like

- Server is getting hammered
- Legitimate users getting timeouts
- One script/bot running too many requests
- CPU/memory spiking

### Immediate Actions (30 minutes)

**Step 1: Add rate limiting to API**

Add to `server.py`:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Add to endpoint
@app.post("/solve")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def solve_problem(request: SolveRequest):
    # ... rest of function
```

**Step 2: Restart server**
```bash
docker-compose restart api
```

**Step 3: Monitor**
```bash
# Check if abuse stopped
tail -f logs/api.log | grep "429"  # Rate limit errors
```

### Communication (if legitimate users affected)

**Post:**
```
⚠️ API Update (Tuesday 2pm)

We've implemented rate limiting to ensure
fair access for all users.

New limit: 10 requests/minute per IP

Bulk analysis? Contact us for higher tier.
```

### Prevention

```
Add from day 1:
☐ Rate limiting (10-100 req/min per IP)
☐ Monitoring (alert at 80% capacity)
☐ Autoscaling (add more servers if needed)
☐ Cache results (avoid duplicate computations)
☐ API tiers (free tier with limits, paid unlimited)
```

---

## CRISIS ESCALATION MATRIX

**Who to call when things go wrong:**

```
CRITICAL (5+ min downtime):
→ CTO immediately
→ Then: CEO, investors

HIGH (1-2 hour fix needed):
→ Tech lead
→ Then: CTO if not resolved in 30 min

MEDIUM (2-4 hour response):
→ Relevant team lead (comms, research, biz dev)
→ Then: CEO if impacts public image

LOW (next business day okay):
→ Handle in daily standup
→ Log for post-mortem
```

---

## POST-CRISIS PROTOCOL

**After ANY crisis (even small), do this:**

**Within 1 hour:**
- Document what happened
- Identify root cause
- Track time/impact

**Within 24 hours:**
- Send team debrief email
- Explain what happened (non-technical)
- What we're doing to prevent it

**Within 1 week:**
- Post-mortem meeting (30 min)
- Changes to code/process
- Update monitoring/alerts
- Document learnings

**Example post-mortem email:**

```
Subject: Post-Mortem: API Outage Jan 19, 5:42pm

What happened:
Server ran out of memory due to [cause]

Duration: 10 minutes (5:42-5:52pm)

Impact:
- API unavailable during that window
- Dashboard couldn't connect
- ~2 users affected (estimated)

Root cause:
[Technical explanation]

What we're doing:
✓ Immediate fix deployed
✓ Added memory monitoring
✓ Set up autoscaling
✓ Added backup server

Timeline for full fix: [date]

Thanks for your patience.
We're committed to reliability.

Questions? Ask in standup tomorrow.
```

---

## FINAL REMINDER

**Most "crises" in launch week are:**
- Visible (everyone sees them)
- Solvable (usually in 1-2 hours)
- Learnings (improve the system)
- Temporary (forgotten by day 3)

**The worst crisis is not responding/hiding it.**

**Best crisis is public + transparent + fixed + learned from.**

🚨 **Have this playbook open during launch week. You'll reference it.**
