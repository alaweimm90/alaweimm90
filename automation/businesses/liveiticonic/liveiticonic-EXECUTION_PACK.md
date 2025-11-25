# liveiticonic – Execution Pack (CRM, Challenges, Membership, Automations, Metrics)

This document operationalizes **liveiticonic** as a challenge + membership + transformation brand.

---

## 1. CRM Contact Schema

### 1.1 Core Identity

- `full_name` – Name (text)
- `email` – Email (email)
- `timezone` – Time Zone (dropdown/text)
- `country` – Country (dropdown/text)
- `instagram_handle` – Instagram Handle (text)
- `tiktok_handle` – TikTok Handle (text)
- `x_handle` – X / Twitter Handle (text)

### 1.2 Transformation Profile

- `current_state` – Where are you now? (long text)
- `desired_state` – What does “iconic” look like for you? (long text)
- `primary_obstacles` – What’s been in the way? (long text)
- `strengths_assets` – Strengths / assets they bring (long text)

### 1.3 Funnel & Programs

- `challenge_opt_in` – Joined a challenge? (bool)
- `challenge_cohort` – Challenge cohort ID (text; e.g. `2025-01-7day`)
- `challenge_engagement_level` – Challenge Engagement (dropdown)
  - low
  - medium
  - high
- `membership_level` – Membership / Program Level (dropdown)
  - none
  - core
  - elite
  - mastermind
- `joined_at` – Date they joined current level (date)
- `churned_at` – Date they left membership (nullable date)

### 1.4 Funnel & Attribution

- `source_channel` – Source Channel (dropdown)
  - instagram
  - tiktok
  - youtube
  - podcast
  - referral
  - email
  - web
  - other
- `campaign_tag` – Campaign / Challenge / Launch Tag (text)
- `engagement_score` – Engagement Score (number)
- `lifecycle_stage` – Lifecycle Stage (dropdown)
  - lead
  - challenge_participant
  - program_client
  - member
  - alumni

---

## 2. Pipelines

### 2.1 Pipeline: `liveiticonic_challenge_enrollment`

Stages:

1. `Registered`
2. `Day 1–2 Engaged`
3. `Highly Engaged`
4. `Offer Presented`
5. `Joined Program`
6. `Did Not Join`

Usage:

- Tracks challenge cohorts and who moves into programs.
- Deals can be tied to a specific challenge instance.


### 2.2 Pipeline: `liveiticonic_membership_lifecycle`

Stages:

1. `Trial`
2. `Active Member`
3. `At-Risk`
4. `Canceled`
5. `Rejoined`

Usage:

- Tracks membership lifecycle and churn/retention at a higher level.

---

## 3. Key Automations

### 3.1 Workflow A – Challenge Registration → Day 0 Prep

**Trigger:**

- Opt-in form for a challenge (5/7/14-day) with `brand_id = liveiticonic`.

**Actions:**

1. Create/update contact:
   - Set `challenge_opt_in = true`.
   - Set `challenge_cohort` to the specific cohort ID.
   - Set `lifecycle_stage = challenge_participant` (if lower before).
2. Create deal in `liveiticonic_challenge_enrollment`:
   - Stage: `Registered`.
   - Tag: cohort ID.
3. Enroll in `LI_CHALLENGE_DAY0_01` sequence:
   - Email: “Welcome to the [X-day] Iconic Challenge”
     - Explain schedule.
     - Link to community (Discord/Slack/Facebook, etc.).
     - Instructions for day 1.


### 3.2 Workflow B – Daily Challenge Emails & Engagement Tracking

**Trigger:**

- Enrollment in specific challenge cohort.

**Actions:**

1. Send daily emails `LI_CHALLENGE_D1` … `LI_CHALLENGE_DN`:
   - Each includes: tiny commitment, reflection, and share prompt.
2. Track engagement signals:
   - Email opens and clicks.
   - Form submissions or check-ins.
   - Community participation (optionally via tags or manual notes).
3. Update `challenge_engagement_level` and `engagement_score`:
   - High open/click + check-ins → move to `high`.
   - Low/no open/click → remain at `low`.
4. Pipeline updates:
   - If `challenge_engagement_level = high` → move deal to `Highly Engaged`.
   - If they click or attend an “offer” event → move deal to `Offer Presented`.


### 3.3 Workflow C – Challenge → Program Conversion

**Trigger:**

- Challenge ends OR contact performs a high-intent action (e.g. clicks “join program” link, books call).

**Actions:**

1. Enroll in program sales sequence `LI_PROGRAM_SALES_01`:
   - Email 1 (Day 1): “Your next 90 days if you keep going”.
   - Email 2 (Day 2–3): “What iconic looks like when you commit”.
   - Email 3 (Day 4–5): objection handling + deadline.
2. If they **purchase program** (via Stripe/PayPal):
   - Set `membership_level` to `core`/`elite`/`mastermind`.
   - Set `joined_at` to purchase date.
   - Set `lifecycle_stage = program_client` or `member`.
   - Create/transition deal:
     - In `liveiticonic_challenge_enrollment`: move to `Joined Program`.
     - In `liveiticonic_membership_lifecycle`: create at `Active Member`.
3. If they **do not join** by deadline:
   - Move challenge deal to `Did Not Join`.
   - Enroll in nurture sequence for next cohort or future offers.


### 3.4 Workflow D – Membership Retention & Churn Save

**Trigger:**

- Upcoming renewal date / billing event.
- Missed payments.
- Drop in engagement score.

**Actions (examples):

1. Pre-renewal (7 days before billing):
   - Email: “What you’ve gained so far + what’s next”.
   - Optional check-in form: asks if they’re staying, unsure, or leaving.
2. If they mark “unsure” or engagement drops:
   - Tag as `at_risk`.
   - Move membership deal to `At-Risk`.
   - Trigger personal reach-out task.
3. If they cancel:
   - Set `churned_at`.
   - Move membership deal to `Canceled`.
   - Enroll in alumni sequence `LI_ALUMNI_01` (light-touch, occasional invites).
4. If they rejoin later:
   - Move membership deal to `Rejoined`.
   - Reset `membership_level` and `joined_at`.

---

## 4. Email Sequence Outlines

### 4.1 `LI_CHALLENGE_DAY0_01`

- Subject: “You’re in – Welcome to the [X-Day] Iconic Challenge”
- Content:
  - Welcome message from Meshal & Rio (if desired).
  - Short story of what it means to “live iconic”.
  - Schedule overview (days, time windows, key tasks).
  - Links to community and any resources.


### 4.2 Daily Challenge Emails (`LI_CHALLENGE_D1` … `LI_CHALLENGE_DN`)

Structure for each day:

- 1–2 line reminder of the bigger promise.
- One **micro-commitment** (e.g. reflection prompt, 10-minute task).
- Optional share prompt (“Post this in the group/IG story and tag us”).
- Light call to stay engaged, no hard sell until late in the challenge.


### 4.3 `LI_PROGRAM_SALES_01`

**Email 1 – Vision & Path (Day 1)**

- Subject: “From this week to the next 90 days”.
- Content:
  - Reflect on what they did in the challenge.
  - Show what 90 days of structured work can do.
  - Soft introduction of the program.

**Email 2 – Offer & Structure (Day 2–3)**

- Subject: “What the Iconic [Program Name] includes”.
- Content:
  - Modules/paths, calls, community, accountability.
  - Who it’s for and who it’s not for.
  - Link to sales page or booking page.

**Email 3 – Objections & Deadline (Day 4–5)**

- Handle “not enough time”, “not ready”, “money” objections.
- Include case study / story.
- Clear deadline or start date.


### 4.4 `LI_ONBOARDING_01` (for new members)

**Email 1 – “Welcome to [Program Name] – Start Here”**

- How to log in or access content.
- What to do in first 48 hours.

**Email 2 – “Your First Wins”**

- Small, specific outcomes for first 1–2 weeks.
- Encourage posting first win in the community.

**Email 3 – “How we support you”**

- Call schedule.
- Support channels.
- How to ask for help.

---

## 5. Metrics & Dashboards for liveiticonic

Key monthly/launch metrics:

- `challenge_opt_ins` – number of people registered for each challenge.
- `challenge_show_up_rate` – people who open/click at least N emails or complete forms / `challenge_opt_ins`.
- `challenge_to_program_conversion` – # who buy program / `challenge_opt_ins`.
- `membership_count` – number of active members (by level).
- `monthly_churn_rate` – canceled members / prior month’s active members.
- `retention_rate` – 1 – churn.

Optional breakdowns:

- By source_channel.
- By challenge cohort.
- By membership level.

---

## 6. Implementation Notes

- Start as with repz and benchbarrier:
  1. Create the contact fields for transformation profile and funnel.
  2. Create both pipelines and stages.
  3. Implement workflows for challenge registration, daily delivery, sales, onboarding, and retention.
  4. Link payments (Stripe/PayPal) so program purchases correctly update `membership_level`, `joined_at`, and pipeline stages.
- Once stable, add more advanced mechanics like upsells, cross-sells between cohorts, and alumni reactivation.
