# When You're Stuck

**Quick self-service guide for getting unstuck fast.**

You're probably not the first person to face this problem. Here's how to solve it.

---

## 🚀 How to Use This Guide

1. **Find your problem** in the list below
2. **Read the solution section**
3. **Try the suggested fixes**
4. **Still stuck?** Read "Still Stuck?" section at bottom

Most problems can be solved in <5 minutes.

---

## 📍 Index of Common Problems

- [Decision & Planning](#decision--planning)
- [Validation & Research](#validation--research)
- [Building & Development](#building--development)
- [Marketing & Growth](#marketing--growth)
- [Team & Collaboration](#team--collaboration)
- [General Frustration](#general-frustration)
- [Technical Issues](#technical-issues)
- [Still Stuck?](#still-stuck)

---

## Decision & Planning

### "I don't know where to start"

**Problem:** Overwhelmed by options. Don't know what to do first.

**Solution:**
1. Take a deep breath. You have a roadmap. Use it.
2. Read: START_HERE.md (10 min)
3. Choose ONE path:
   - Validate an idea? → VALIDATION_FIRST.md
   - Build a product? → IMPLEMENTATION_GUIDE.md
   - Understand business? → FINANCIAL_MODEL.md
   - Just learn? → QUICKSTART_FIRSTTIME.md
4. Follow that path step by step.

**Why this works:** You don't need to understand everything. Just follow one clear path.

---

### "I can't decide which product to build"

**Problem:** 10 ideas, can't pick one. Analysis paralysis.

**Solution:**
1. Read: PRIORITY_RANKING.md (15 min)
2. Score each on:
   - Feasibility (can I build it?)
   - Impact (how many people want it?)
   - Timeline (how long to MVP?)
3. Top 3:
   - NIGHTMARE MODE (95/100) ← Start here
   - CHAOS ENGINE (94/100)
   - RESEARCH PRISON (85/100)
4. Pick the highest score.
5. Move on. You can always build #2 later.

**Why this works:** Don't optimize the choice. Pick the best and move on.

---

### "I don't know if my idea is good"

**Problem:** Worried the idea won't work. Fear of wasting time.

**Solution:**
1. Good news: We already vetted 10 ideas. Pick one.
2. Bad news: Validation is the only real test.
3. Read: VALIDATION_FIRST.md (15 min)
4. Do 3 interviews this week.
5. If 70% of people identify the problem → it's probably good.

**Why this works:** Customer feedback beats your gut feeling. Always.

---

### "What if I pick wrong?"

**Problem:** Fear of making a bad decision.

**Solution:**
1. You can't pick "wrong" if you validate first.
2. If validation fails → pivot (not a loss, you learned something)
3. If validation succeeds → build
4. Either way → you make progress
5. Read: 100_STEP_ROADMAP.md (the pivot section)

**Why this works:** There's no wrong choice if you validate. You learn either way.

---

## Validation & Research

### "I can't find people to interview"

**Problem:** No one to talk to. Don't know where to find customers.

**Solution:**
1. Start with people you know:
   - LinkedIn (search for target job title)
   - Twitter/Reddit (communities interested in problem)
   - Facebook groups
   - Slack communities
   - Industry forums
2. Join 3-5 relevant communities this week
3. Post: "Looking for researchers who struggle with [problem]"
4. You'll get 10+ responses

**Pro tip:** Make a list of 50 people in 30 minutes. You'll find them.

**Related:** CUSTOMER_DEVELOPMENT.md (has templates)

---

### "People say 'maybe' but won't commit"

**Problem:** Lots of interest but no one will pre-pay.

**Solution:**
1. "Maybe" = "No"
2. You need clear YES (willing to pay) or NO (not interested)
3. Ask directly: "Would you pay $99/month if this shipped in 4 weeks?"
4. If they say "maybe" → ask why
5. Fix the top reasons people say "maybe"
6. Ask again in next interview

**Why this works:** Pre-sales require commitment. "Maybe" doesn't count.

**Related:** VALIDATION_FIRST.md (signals section)

---

### "I got <10 pre-sales. Should I pivot or stop?"

**Problem:** Validation failed. Don't know what to do.

**Solution:**
1. Don't panic. This is valuable learning.
2. Analyze why <10:
   - Wrong customer? (interview more types)
   - Wrong solution? (their idea is different)
   - Wrong timing? (too early/late)
   - Wrong price? (too expensive)
3. Pick ONE variable to change
4. Do 10 more interviews with change
5. Test again

**Read:** 100_STEP_ROADMAP.md (pivot section)

---

## Building & Development

### "I don't know how to code"

**Problem:** Want to build but can't code.

**Solution:**
1. **Option A:** Learn to code
   - Codecademy, Udemy, freeCodeCamp (start here)
   - 3-6 months to basic competency
   - But you don't have 3-6 months!

2. **Option B:** Hire/partner with developer
   - Find co-founder or contractor
   - Budget: $20-30K for MVP
   - Or: Learn + build with friend

3. **Option C:** Use no-code tools
   - Bubble, Airtable, Make.com
   - Build MVP in weeks, not months
   - Fastest path for non-technical founder

**Recommendation:** No-code MVP first, then hire dev if needed.

---

### "Setup is too complicated"

**Problem:** Dev environment, dependencies, database = overwhelming.

**Solution:**
1. Read: docs/guides/GETTING_STARTED.md (20 min)
2. Follow step-by-step setup
3. If stuck on one step:
   - Google the exact error message
   - Check StackOverflow
   - Post on GitHub discussions
4. You'll get through it. Millions have.

**Pro tip:** Use Docker. It solves 80% of setup problems.

---

### "My code has bugs"

**Problem:** Features broken. Can't figure out why.

**Solution:**
1. Narrow it down:
   - Does it work in specific case? (yes → narrow scope)
   - What changed? (revert last change)
   - Can you reproduce it? (yes → you can debug)
2. Read the error message carefully
3. Google exact error
4. Check StackOverflow
5. Write test case for bug (helps you understand it)
6. Fix bug
7. Write test to prevent regression

**Pro tip:** Most bugs are simple. You just need to narrow scope.

---

### "Tests are failing"

**Problem:** CI/CD pipeline failing. Can't deploy.

**Solution:**
1. Read the error output carefully
2. Run tests locally first: `pytest` or `npm test`
3. Fix local tests before pushing
4. Common causes:
   - Missing environment variables
   - Database not seeded
   - Different Python/Node version
   - Async timing issues
5. Check .env.example and create .env locally
6. Try again

**Related:** docs/guides/GETTING_STARTED.md

---

### "Deployment is broken"

**Problem:** Code works locally but fails in production.

**Solution:**
1. Check environment variables (different in prod vs local)
2. Check database (prod data different)
3. Check logs in production (tail -f logs)
4. Common causes:
   - Missing env var
   - Different database state
   - Memory/performance issues
   - Secrets not configured
5. SSH into prod and debug manually
6. Never give up. Millions have debugged this.

---

## Marketing & Growth

### "I don't know how to market"

**Problem:** Built product but don't know how to get customers.

**Solution:**
1. Read: docs/guides/MARKETING_PLAYBOOK.md (30 min)
2. Start simple:
   - Email your validation contacts ("We shipped!")
   - Tweet about your launch
   - Post in relevant communities
3. Pick ONE channel:
   - Email (highest conversion)
   - Twitter/Reddit (high reach)
   - Product Hunt (biggest launch day)
4. Focus there for 4 weeks
5. Measure what works, do more of it

**Simple formula:** Tell people → They use it → They tell others → Repeat

---

### "No one is using my product"

**Problem:** Launched but users aren't coming.

**Solution:**
1. Check metrics: docs/guides/METRICS_DASHBOARD.md
2. Where are you losing people?
   - Landing page: <10% converting? → Fix landing page
   - Signup: <50% converting? → Fix signup flow
   - Onboarding: <50% reaching core feature? → Fix onboarding
   - Retention: <80% coming back? → Fix product
3. Fix the biggest leak first
4. Measure impact
5. Move to next leak
6. Repeat until 10%+ are active

**Pro tip:** Focus on retention before growth. You're losing people faster than you're gaining them.

---

### "I can't afford marketing"

**Problem:** No budget for paid ads.

**Solution:**
1. Most successful startups started with $0 marketing budget
2. Do these instead (free):
   - Twitter: Tweet daily about learnings
   - Reddit: Answer questions in relevant subreddits
   - Communities: Join Slack/Discord groups, help people
   - Content: Write blog posts about problem
   - Word of mouth: Ask each customer to refer a friend
3. Build audience first → monetize later

**Why this works:** Free channels are slow but create loyal users.

---

## Team & Collaboration

### "I want to hire but don't have money"

**Problem:** Need help but can't afford salary.

**Solution:**
1. **Option A:** Find co-founder
   - Equity split (50/50, 40/60, etc.)
   - Aligned incentives
   - Someone takes salary risk with you

2. **Option B:** Contract/freelancer
   - Pay by the hour or project
   - Get help for specific tasks
   - Budget: $20-50/hour

3. **Option C:** Friends/family
   - Appeal to their interest in project
   - Some will work cheap/free early
   - Promise equity/revenue share

4. **Option D:** Do it yourself longer
   - You don't need perfect team yet
   - MVP with 1 person can work
   - Hire when you have revenue

**Most successful approach:** Find a co-founder with complementary skills.

---

### "My team isn't aligned"

**Problem:** Different opinions about direction.

**Solution:**
1. Have explicit conversation:
   - What are we trying to achieve? (goal)
   - How will we measure success? (metrics)
   - What's our priority order? (roadmap)
2. Document decisions in writing
3. Meet weekly to check alignment
4. If still misaligned → might need to part ways
5. Misalignment early << misalignment later

**Related:** CONTRIBUTING.md (team collaboration section)

---

## General Frustration

### "This is too hard, I want to quit"

**Problem:** Everything feels impossible. Thinking about giving up.

**Solution:**
1. **First:** It's normal. Most people feel this.
2. **Perspective:** You're probably closer than you think
3. **Reality check:**
   - Have you validated? (yes → push through to MVP)
   - Have you shipped? (yes → push through to first customers)
   - Do you have 10 customers? (yes → push through to profitability)
   - Each stage feels hard. You're not failing.
4. **Take a break:** Sleep on it. Often things look better tomorrow.
5. **Talk to someone:** Find a founder friend and vent.
6. **Remember why:** Why did you start? Focus on that.

**Pro tip:** The hard part is what keeps competition away. If it were easy, everyone would do it.

---

### "I feel like I'm moving slowly"

**Problem:** Progress feels glacial. Not moving fast enough.

**Solution:**
1. Track actual metrics:
   - Week 1 vs Week 4 → How much progress?
   - Month 1 vs Month 3 → How much progress?
2. Celebrate small wins:
   - First interview
   - First pre-sale
   - First user
   - First $100 revenue
3. Compare to reality, not to stories:
   - Startup stories exaggerate. Most take 12-24 months to early traction.
   - You're probably on track.
4. Focus on rate of learning, not speed of execution
5. Each day you learn something = progress

**Real timeline:**
- Weeks 1-4: Validation
- Weeks 5-24: MVP
- Weeks 25-52: Launch & traction
- Year 2: Scale

You're not slow. You're on a real timeline.

---

### "I don't have any money"

**Problem:** No budget to start.

**Solution:**
1. Validation costs ~$0
   - Your time
   - Coffee/meals when interviewing
   - That's it
2. MVP can cost $0-5K (using no-code tools)
3. Many successful startups started with <$1K

**Bootstrap path:**
1. Validate (free)
2. Build with no-code (free tier)
3. Get first 10 customers (free marketing)
4. Use early revenue to hire developers
5. Grow from there

**Famous examples:**
- Mailchimp: Started free, bootstrapped to $600M
- Buffer: Bootstrapped to $1M+ ARR
- Basecamp: Self-funded, now $100M+ revenue

You don't need money to start. You need customers.

---

## Technical Issues

### "My database is corrupted"

**Problem:** Database in bad state. Can't recover.

**Solution:**
1. **Immediately:** Don't touch it
2. **Backup:** Do you have a backup? (you should every day)
3. **Restore:** Restore from most recent backup
4. **Prevent future:** Setup automated backups
5. **Learn:** What caused corruption? (update queries? malformed data?)
6. **Mitigate:** Add validation to prevent in future

**Pro tip:** If you don't have automated backups, set them up today. Takes 30 min.

---

### "My server is down"

**Problem:** Website not responding. Traffic = 0.

**Solution:**
1. **Check status page** (AWS, Heroku, etc.)
2. **Is it their problem or yours?**
   - Provider status down → wait + notify users
   - Your code broken → rollback last deploy + fix
   - Out of resources → scale up
3. **Immediately notify users** (email, Twitter, status page)
4. **Fix root cause**
5. **Post-mortem:** What should prevent this next time?

**Prevention:**
- Setup monitoring (uptime alerts)
- Setup auto-scaling
- Setup 1-click rollback
- Periodic disaster recovery tests

---

### "Performance is slow"

**Problem:** App takes 10+ seconds to load.

**Solution:**
1. **Find the bottleneck:**
   - Frontend (rendering)? Use Chrome DevTools
   - Backend (database)? Check query performance
   - Network (large files)? Compress images
2. **Fix top issue** (usually 80% of problem)
3. **Measure improvement**
4. **Move to next issue**
5. **Setup monitoring** so you catch next issue early

**Common culprits:**
- N+1 database queries
- Unoptimized images
- Render-blocking JavaScript
- Missing database indexes

---

## Still Stuck?

### Try these (in order)

1. **Google the exact error message**
   - 95% of technical problems have solutions on Google
   - Copy-paste your error into Google

2. **Check FAQ.md**
   - 40+ common questions already answered

3. **Check TROUBLESHOOTING.md**
   - More detailed solutions for common issues

4. **Check DOCUMENTATION_INDEX.md**
   - Find any document related to your problem

5. **Ask in community**
   - Post on relevant subreddit (r/entrepreneur, r/webdev, etc.)
   - Post on Twitter
   - Post on relevant Slack/Discord

6. **Take a walk**
   - Sometimes breaks help
   - Fresh perspective solves problems
   - Come back in 30 min

7. **Call a friend**
   - Explain your problem to someone else
   - Often solving it mid-explanation
   - Rubber duck debugging: explain to a rubber duck, get unstuck

### If you're REALLY stuck

1. Post on GitHub Issues
2. Ask in Foundry Discord/Slack (if one exists)
3. Post on relevant subreddit with full context
4. Take a day off and come back
5. Hire someone to help ($100-500 can solve many things)

---

## Prevention is Better Than Cure

**Things that prevent getting stuck:**

1. ✅ **Read the documentation** - You have 50+ documents for this
2. ✅ **Ask for help early** - Don't suffer alone
3. ✅ **Break problems down** - Solve one small thing at a time
4. ✅ **Search before asking** - 80% have solutions online
5. ✅ **Test constantly** - Catch problems early, not late
6. ✅ **Keep learning** - Every problem teaches you something
7. ✅ **Take breaks** - Fresh perspective helps

---

## Remember

✨ **Every successful founder has been stuck many times.**

The difference between successful and unsuccessful:
- Unsuccessful: Get stuck → quit
- Successful: Get stuck → solve it → move on → get stuck again → solve it → etc.

**You're going to get stuck multiple times. That's normal. You'll solve it and move on.**

---

**Ready to get unstuck?** Pick your problem above and follow the solution. 🚀

*Good luck. You've got this.* 💪
