# 📊 METRICS DASHBOARD TEMPLATE

Google Sheets template for real-time Week 1 KPI tracking during launch.

---

## QUICK START

1. **Create new Google Sheet:** https://sheets.google.com/create
2. **Copy sheet structure below into tabs**
3. **Update daily** with new data
4. **Share with stakeholders** with view-only access
5. **Pin tab** in browser for easy access

**Recommended update cadence:** Daily at 5pm (during standup)

---

## SHEET 1: DAILY METRICS (Main Dashboard)

### Column Headers

```
A: Date
B: Day of Week
C: Twitter Followers
D: Twitter Followers (+Delta)
E: Twitter Impressions (cumulative)
F: Twitter Engagements
G: LinkedIn Followers
H: LinkedIn Followers (+Delta)
I: GitHub Stars
J: GitHub Stars (+Delta)
K: Job Applications
L: Job Applications (+Delta)
M: Email Signups
N: Email Signups (+Delta)
O: Partnership Inquiries
P: Partnership Inquiries (+Delta)
Q: Notes/Events
```

### Sample Data (Week 1)

```
Monday 1/19    Mon    150     150     2,400    450    200    200    100    100    5    5    25    25    0    0    "Launch day blitz"
Tuesday 1/20   Tue    320    +170    5,800   1,200   450   +250    200   +100   12   +7    85   +60    1   +1    "Morning podcast mentions"
Wednesday 1/21 Wed    680    +360   11,200   2,100   890   +440    400   +200   28  +16   155   +70    2   +1    "Reddit upvote surge"
Thursday 1/22  Thu   1,150   +470   18,500   3,200  1,450  +560    650   +250   38  +10   210   +55    2    0     "HN ranked #3"
Friday 1/23    Fri   1,680   +530   26,800   4,100  2,100  +650   1,050  +400   52  +14   285   +75    3   +1    "Week wrap + anticipation"
Saturday 1/24  Sat   2,200   +520   33,200   4,500  2,700  +600   1,450  +400   56   +4   320   +35    3    0     "Weekend organic reach"
Sunday 1/25    Sun   2,800   +600   38,100   4,800  3,200  +500   1,800  +350   60   +4   360   +40    3    0     "Community prep for week 2"
```

### Formulas to Include

**Cell D2 (Delta calculation):**
```
=C2-C1
```

**Cell E2 (Cumulative impressions):**
```
=E1+F2*0.5  (rough estimate: half of engagement count as new impressions)
```

**Cell P2 (Cumulative inquiries):**
```
=SUMIF(Q:Q, "partnership", A:A)  (count mentions of "partnership" in notes)
```

---

## SHEET 2: HOURLY LAUNCH DAY TRACKING

### Monday 1/19 Hourly Breakdown

```
Hour    | Task                              | Planned | Actual | Status  | Notes
--------|-----------------------------------|---------|--------|---------|------------------
9:00am  | Twitter + Reddit posts            | 9:30am  | 9:28am | ✓       | Posted 1+2
9:30am  | LinkedIn posts                    | 10:00am | 10:05am| ✓       | Slight delay
10:00am | Job postings deployed             | 10:30am | 10:18am| ✓ Done | 5/5 platforms live
10:30am | Partnership proposals sent        | 11:30am | 11:47am| ✓ Late | Two personalization rewrites
11:30am | Email recruitment blitz           | 12:30pm | 12:45pm| ✓ Late | 45 emails sent
12:30pm | Lunch + social engagement         | 1:30pm  | 1:30pm | ✓       | 12 replies, 8 new followers
1:30pm  | Press/HN/Dev.to                   | 2:30pm  | 2:15pm | ✓ Ahead| HN #3 rank!
2:30pm  | Interview scheduling              | 3:30pm  | 3:32pm | ✓       | 3 phone screens scheduled
3:30pm  | Metrics check                     | 4:00pm  | 3:55pm | ✓ Ahead| Stars: 100✓ Jobs: 7✓
4:00pm  | Content continuation              | 5:00pm  | 5:02pm | ✓       | Evening posts live
5:00pm  | Daily standup & celebration       | 5:30pm  | 5:45pm | ✓       | Day 1 success!
```

**Columns:**
- A: Hour
- B: Planned task
- C: Planned completion time
- D: Actual completion time
- E: Status (✓ On-time, ⚠ Late, ✗ Blocked)
- F: Notes/learnings

---

## SHEET 3: WEEK 1 FORECAST vs ACTUAL

### Target Tracking

```
Metric                  | Target | Mon | Tue | Wed | Thu | Fri | Sat | Sun | Cumulative | Status
------------------------|--------|-----|-----|-----|-----|-----|-----|-----|------------|--------
GitHub Stars            | 500+   | 100 | 200 | 300 | 400 | 500 | 700 |1000 |  1,000    | ✓ Exceeded
Job Applications        | 50+    |  5  | 12  | 28  | 38  | 52  | 56  | 60  |    60     | ✓ Exceeded
Email Signups           | 250+   | 25  | 85  |155  |210  |285  |320  |360  |    360    | ✓ Exceeded
Partnership Inquiries   | 2-3    |  0  |  1  |  2  |  2  |  3  |  3  |  3  |     3     | ✓ On-target
Twitter Reach (total)   | 500K+  |2.4K |5.8K |11.2K|18.5K|26.8K|33.2K|38.1K|  135.8K  | ⚠ Behind
LinkedIn Reach (total)  | 100K+  |1.2K |2.8K | 4.5K| 6.4K| 8.1K|10.2K|11.8K|  44.9K   | ⚠ Behind
Press Mentions          | 3+     |  1  |  1  |  1  |  1  |  1  |  1  |  1  |     7     | ✓ Exceeded
Community Issues/PRs    | 5+     |  0  |  1  |  2  |  2  |  3  |  4  |  5  |     5     | ✓ On-target
Phone Screens Completed | 3-5    |  0  |  1  |  2  |  4  |  5  |  5  |  5  |     5     | ✓ On-target
```

**Color coding:**
- Green ✓: Target met or exceeded
- Yellow ⚠: On track but slightly behind
- Red ✗: Significantly behind, needs adjustment

---

## SHEET 4: METHOD COMPARISON PERFORMANCE

### Daily Benchmark Results

```
Date       | Instance | Size | FFT-L | Reverse | GA   | SA   | Tabu | VNS  | ACO  | PSO  | Best | Time
-----------|----------|------|-------|---------|------|------|------|------|------|------|------|--------
1/19 (Mon) | tai12a   | 12   | 0.21  | 0.89    | 2.3  | 1.4  | 1.1  | 0.8  | 1.2  | 1.5  | VNS  | 0.045s
1/19 (Mon) | nug12    | 12   | 0.34  | 1.12    | 2.8  | 1.8  | 1.4  | 1.2  | 1.5  | 1.9  | VNS  | 0.042s
1/20 (Tue) | chr12a   | 12   | 0.18  | 0.76    | 2.1  | 1.2  | 0.9  | 0.7  | 1.0  | 1.3  | VNS  | 0.041s
1/20 (Tue) | tai14a   | 14   | 0.48  | 1.35    | 3.2  | 2.1  | 1.8  | 1.5  | 1.9  | 2.4  | VNS  | 0.053s
1/21 (Wed) | sko40    | 40   | 0.89  | 2.45    | 4.8  | 3.2  | 2.9  | 2.6  | 3.1  | 3.7  | VNS  | 0.124s
1/22 (Thu) | nug30    | 30   | 0.65  | 1.98    | 3.5  | 2.4  | 2.1  | 1.9  | 2.3  | 2.8  | VNS  | 0.089s
1/23 (Fri) | tai30a   | 30   | 0.72  | 2.15    | 3.8  | 2.6  | 2.2  | 2.0  | 2.4  | 3.0  | VNS  | 0.092s
```

**Tracking purpose:** Monitor if FFT-Laplace stays competitive
**Update frequency:** Daily (1 new instance tested)
**Key metric:** FFT-Laplace average gap (target: < 1.0%)

---

## SHEET 5: REVENUE & PARTNERSHIP FUNNEL

### Deals Pipeline

```
Company           | Stage        | Contact   | Date Contacted | Last Update | Value Est. | Status  | Notes
------------------|--------------|-----------|----------------|-------------|------------|---------|------------------
IBM Quantum       | Proposal sent| Dr. Smith | 1/19           | 1/20        | $100K      | Pending | Awaiting response
Google Quantum    | Proposal sent| Jane Doe  | 1/19           | -           | $80K       | Pending | No response yet
IonQ              | Proposal sent| Mike Lee  | 1/20           | 1/22        | $75K       | Positive| Call scheduled 1/25
PyTorch (Meta)    | Proposal sent| Sarah C.  | 1/20           | -           | $90K       | Pending | Will follow up 1/24
TensorFlow        | Proposal sent| Alex P.   | 1/20           | -           | $85K       | Pending | Slow response expected
MIT Professor     | Intro email  | Prof. X   | 1/19           | 1/21        | $50K/year  | Positive| Meeting 1/27
Stanford Professor| Intro email  | Prof. Y   | 1/20           | -           | $50K/year  | Pending | Will reach out 1/26
CMU Professor     | Intro email  | Prof. Z   | 1/21           | -           | $50K/year  | Pending | Follow-up needed
```

**Columns:**
- A: Company/Institution
- B: Deal stage (Intro, Proposal sent, In discussion, Positive, Negotiating, Closed)
- C: Primary contact name
- D: Date of initial contact
- E: Last update date
- F: Estimated value
- G: Status (Pending, Positive, Negotiating, Closed)
- H: Next actions/notes

**Calculation:** Total pipeline value shown in summary stats

---

## SHEET 6: DAILY STANDUP NOTES

### End-of-day Summaries

```
Date       | What Went Well                              | What Didn't               | Learnings/Adjustments
-----------|---------------------------------------------|-----------------------|------------------------
1/19 (Mon) | HN #3 ranking, job apps at 5 (ahead)     | Twitter reach lower   | Post earlier for TZ coverage
           | Partnership responses faster than expected  | Email open rate 32%   | Add personalization to emails
           |                                              |                       |
1/20 (Tue) | Reddit surge brought 50+ stars            | LinkedIn slower start | Boost LinkedIn with image posts
           | First genuine partner interest (IonQ)      | No calls yet          | Phone screen scheduling slower
           |                                              |                       |
1/21 (Wed) | 300 stars by noon, tracking well           | Algorithm posts less  | Thread format > single posts
           | 2nd partnership inquiry arrived            | CSS dashboard bug     | Fixed and redeployed quickly
           |                                              |                       |
1/22 (Thu) | #1 GitHub trending, organic momentum      | Email fatigue showing | Reduce frequency, up quality
           | First two interviews completed, solid      | Job descriptions      | Niche = better candidates
           |                                              | attracting wrong fit   |
1/23 (Fri) | Week summary: All targets exceeded         | Partnership slow pace | More aggressive follow-up
           | Great candidates, strong team interest      | needed               | Set 5-day callback reminder
           |                                              |                       |
1/24 (Sat) | Organic growth continuing                  | Burnout risk (worked  | Pace week 2, don't repeat
           | Weekend traffic surprisingly strong        | 60+ hours)            | intensity, it's unsustainable
           |                                              |                       |
1/25 (Sun) | Team assembly plan solid, hires close      | ICML paper deadline   | Submit Monday morning
           | Community momentum sustained                | still tight            | Quality > perfection
```

**Purpose:** Capture learnings for future campaigns
**Update:** Daily at 5pm standup (30 min write-up)

---

## SHEET 7: CONVERSION TRACKING

### Funnel Analysis

```
Channel         | Impressions | Clicks | Signups | Apps | Interviews | Offers | Conv %
----------------|-------------|--------|---------|------|------------|--------|--------
Twitter         | 26,800      | 1,340  | 85      | 12   | 2          | 0      | 0.31%
LinkedIn        | 14,200      | 710    | 120     | 18   | 3          | 1      | 0.84%
Reddit          | 8,400       | 630    | 65      | 8    | 1          | 0      | 0.77%
HackerNews      | 3,200       | 640    | 45      | 12   | 4          | 1      | 1.56%
Email           | 45 (sent)   | 15     | 35      | 10   | 3          | 1      | 2.22%
Job Boards      | N/A         | -      | -       | 15   | 4          | 1      | 6.67%
Direct/Other    | -           | -      | -       | 5    | 1          | 0      | N/A
TOTAL           | 52,600      | 3,915  | 350     | 60   | 18         | 4      | 0.76%
```

**Analysis:**
- Email = highest conversion (2.22%)
- Job boards = highest hire conversion (6.67%)
- Twitter = high volume, lower conversion (0.31%)
- LinkedIn = balanced approach (0.84%)

**Learnings:** Personalized channels outperform broadcast

---

## SHEET 8: COMMUNITY ENGAGEMENT METRICS

### Daily Activity Tracking

```
Date       | GitHub Issues | PRs Merged | Discussions | New Followers | Code Changes | Top Activity
-----------|---------------|-----------|-------------|---------------|--------------|----------------------
1/19 (Mon) | 0             | 0         | 2           | 50            | N/A          | Initial interest
1/20 (Tue) | 2             | 0         | 5           | 120           | 1 PR review  | Bug report + fix discussion
1/21 (Wed) | 5             | 1         | 12          | 200           | 2 method PRs | Feature requests
1/22 (Thu) | 8             | 2         | 18          | 150           | 1 optimization| First contributor!
1/23 (Fri) | 6             | 1         | 22          | 180           | Doc updates | Community momentum
1/24 (Sat) | 2             | 0         | 8           | 80            | 2 doc PRs   | Weekend casual browsing
1/25 (Sun) | 3             | 1         | 10          | 120           | 1 guide     | Prep for week 2
```

**Key metrics:**
- First contributor by Thursday ✓
- Sustained discussion engagement ✓
- Doc contributions starting ✓

---

## SUMMARY DASHBOARD (Visible on Main Sheet)

### Key Stats Box (Update Daily)

```
========================================
     Librex.QAP-NEW WEEK 1 SUMMARY
========================================

GitHub Stars:            1,800 ⬆️ 800
Job Applications:          60 ⬆️ 60
Email Subscribers:        360 ⬆️ 360
Partnership Inquiries:      3 ⬆️ 3

Social Media Reach:    135.8K people
Twitter Followers:       2,800 +1,150%
LinkedIn Followers:      3,200 +450%

Phone Screens:             18 completed
Interview Offers:           4 made
Conversion Rate (Jobs):    6.7%

ICML Paper:          Ready for submission
Status:              🟢 ALL TARGETS MET ✓
========================================
```

**Position:** Top of main dashboard sheet
**Refresh:** Daily at 5pm
**Share:** View-only link to stakeholders

---

## FORMULAS FOR AUTOMATION

### Formula 1: Dynamic Targets (Conditional formatting)

```
=IF(C2>=C$8, "✓ ON TARGET", IF(C2>=C$8*0.8, "⚠ BEHIND", "✗ CRITICAL"))
```

**Usage:** Highlight cells that meet, miss, or fall critically short of daily targets

### Formula 2: Daily Growth Rate

```
=(C2-C1)/C1*100
```

**Shows:** % day-over-day growth in followers, stars, etc.

### Formula 3: Cumulative Total

```
=SUM($C$2:C2)
```

**Shows:** Running total for the week

### Formula 4: 7-Day Moving Average

```
=AVERAGE(OFFSET(C2,-6,0,7,1))
```

**Shows:** Smoothed trend line (removes daily volatility)

---

## SHARING & COLLABORATION

### Who Gets Access (Recommended)

**View-only (public dashboard):**
- Investors/partners
- Stakeholders
- Team (read-only)

**Edit access (core team):**
- You (primary updater)
- Co-founder (for metrics discussions)
- Marketing lead (for social media tracking)

**Publish:**
1. Create public view-only link
2. Share in Slack: `#announcements`
3. Tweet: "Week 1 metrics live: github.com/.../metrics"
4. Newsletter: "Weekly numbers in our public dashboard"

---

## DAILY UPDATE CHECKLIST (5:30pm - 15 minutes)

```
☐ Update Twitter followers (from profile)
☐ Update GitHub stars (from repo)
☐ Update job applications (from email count)
☐ Update partnership inquiries (from email)
☐ Update email signups (from Mailchimp)
☐ Calculate all deltas (new vs. previous day)
☐ Add 1-2 sentence note about day's events
☐ Review targets: on track? (Green) | Behind? (Yellow) | Critical? (Red)
☐ Identify one main learning for standup
☐ Save sheet (Ctrl+S)
☐ Confirm all stakeholders can see latest
```

**Time commitment:** 15 minutes daily
**Best time:** 5:30pm (end of work, before dinner)

---

## WEEK 2+ TRACKING

### Adjusted Metrics

```
Week 2 Targets (Jan 26-Feb 1):
- GitHub Stars: 5,000+ (was 500+)
- Phone Screens: 20-30 (continue hiring)
- Offers Extended: 2-3 (was 1-2)
- Partnership Agreements: 1-2 signed (was just inquiries)
- ICML Paper: Submitted (was draft)
- Community Contributors: 5-10 (was 1)

Month 1 Targets (Feb 1):
- GitHub Stars: 10,000+
- Job Offers Signed: 3-5
- Partnerships: 2-3 active agreements
- Email List: 1,000+
- Paper: Under review
```

---

## TEMPLATE DOWNLOAD LINK

**Ready-to-use Google Sheet:**
`[Create new sheet and copy structure above]`

Or import CSV template:
```csv
Date,Day,TwitterFollowers,TwitterDelta,GitHubStars,JobApps,Emails,Partnerships,Notes
2025-01-19,Monday,150,150,100,5,25,0,"Launch day"
2025-01-20,Tuesday,320,170,200,12,85,1,"Good response"
...
```

---

📊 **Sheet updated daily. Targets visible at a glance. Shareable with team/investors.**
