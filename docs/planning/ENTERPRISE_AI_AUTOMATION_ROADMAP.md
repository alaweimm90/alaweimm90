> ****
>
> # Deep Research: Enterprise AI Automation Asset Enhancement
> 
> ## Context
> I have an AI automation system with the following current inventory:
> - 6 system prompts (orchestration, debugging, code review, etc.)
> - 10 project prompts (superprompts for specific project types)
> - 13 task prompts (specific task execution guides)
> - 17 agents (research, development, content, security, operations)
> - 6 workflows (using Anthropic orchestration patterns)
> - 2 crews (development_crew, research_crew)
> 
> ## Research Request
> Conduct deep research on state-of-the-art AI automation systems, agentic frameworks, and enterprise prompt engineering to recommend:
> 
> ### 1. PROMPTS - What's Missing?
> 
> **System Prompts** (foundational behaviors):
> - What system prompts do leading AI platforms use?
> - Research: Anthropic's system prompts, OpenAI's instruction patterns, Google's Gemini guidelines
> - What governance/safety/alignment prompts are industry standard?
> - What meta-prompts improve AI reasoning (chain-of-thought, tree-of-thought, reflection)?
> 
> **Project Prompts** (domain-specific superprompts):
> - What project types need dedicated superprompts? (ML pipelines, API development, data engineering, DevOps, mobile apps, etc.)
> - Research: How do cursor rules, aider conventions, and windsurf configs structure project context?
> - What makes a "10x developer" superprompt vs a basic one?
> 
> **Task Prompts** (specific operations):
> - What atomic tasks should have dedicated prompts? (PR reviews, security audits, performance optimization, refactoring, testing, documentation)
> - Research: GitHub Copilot's task patterns, Cursor's command prompts, Cline's task templates
> - What task decomposition strategies work best?
> 
> ### 2. AGENTS - What Roles Are Missing?
> 
> Research leading multi-agent frameworks:
> - **CrewAI**: What agent roles do they recommend?
> - **AutoGen**: What specialist agents exist?
> - **LangGraph**: What agent architectures work?
> - **MetaGPT**: What software company simulation agents exist?
> 
> Consider gaps in:
> - Data engineering agents
> - DevOps/SRE agents  
> - Product management agents
> - QA/Testing agents
> - Documentation agents
> - Architecture/design agents
> - Cost optimization agents
> - Compliance/legal agents
> 
> ### 3. WORKFLOWS - What Patterns Are Missing?
> 
> Research Anthropic's orchestration patterns and beyond:
> - Prompt chaining (have)
> - Routing (have)
> - Parallelization (have)
> - Orchestrator-workers (have)
> - Evaluator-optimizer (have)
> 
> What else?
> - **Reflection patterns**: Self-critique and improvement loops
> - **Planning patterns**: Goal decomposition and task planning
> - **Memory patterns**: Long-term context management
> - **Tool-use patterns**: Dynamic tool selection and chaining
> - **Debate patterns**: Multi-perspective reasoning
> - **Verification patterns**: Output validation and fact-checking
> 
> ### 4. CREWS - What Team Compositions Are Missing?
> 
> Research effective multi-agent team structures:
> - What crew compositions solve specific problem domains?
> - How do hierarchical vs flat crew structures compare?
> - What communication patterns between agents work best?
> 
> Consider crews for:
> - Full-stack development
> - Data science pipelines
> - Security red team / blue team
> - Content creation pipeline
> - DevOps automation
> - Research and analysis
> - Customer support automation
> 
> ### 5. CUTTING-EDGE ADDITIONS
> 
> Research the latest in:
> - **Constitutional AI**: Self-alignment prompts
> - **RLHF patterns**: Human feedback integration
> - **Mixture of Experts**: Specialized routing
> - **RAG workflows**: Retrieval-augmented generation patterns
> - **Code generation**: Best practices from Codex, StarCoder, CodeLlama
> - **Agentic loops**: ReAct, MRKL, Toolformer patterns
> 
> ## Output Format
> 
> For each recommendation, provide:
> 
> 1. **Name**: Clear identifier
> 2. **Type**: prompt/agent/workflow/crew
> 3. **Category**: (for prompts: system/project/tasks)
> 4. **Purpose**: What problem it solves
> 5. **Priority**: High/Medium/Low based on impact
> 6. **Template**: Starter content or structure
> 7. **Source**: Where you found this pattern (paper, framework, best practice)
> 
> ## Constraints
> - Focus on practical, implementable additions
> - Prioritize patterns proven in production systems
> - Consider integration with existing assets
> - Avoid redundancy with current inventory
> 
> ## Expected Output
> A prioritized list of 20-30 recommended additions across all categories, with implementation templates for the top 10 highest-impact items.



-----------------------------

> ****
>
> I'm building a comprehensive TypeScript/React template library for a portfolio of SaaS products. I already have working templates for:
> 
> EXISTING (production-ready):
> - Supabase auth (signUp/signIn/signOut/OAuth)
> - Stripe payments (payment intents, checkout sessions)
> - Shopping cart (localStorage, tax, shipping)
> - Order management (CRUD, tracking, refunds)
> - Checkout flow (React hook, multi-step)
> - Analytics (GA4/GTM/Plausible)
> - ML pipeline (scikit-learn)
> 
> I need you to provide COMPREHENSIVE specifications for each of these categories. For EACH item, give me:
> 1. Full TypeScript interfaces/types
> 2. All API endpoints needed (REST)
> 3. Database schema (Prisma format)
> 4. React hooks if applicable
> 5. Environment variables required
> 6. Common edge cases to handle
> 
> CATEGORIES TO COVER:
> 
> 1. PRODUCT CATALOG
>    - Product types (physical, digital, subscription, bundle)
>    - Product variants (size, color, etc.)
>    - Product categories/collections
>    - Product images/gallery
>    - Inventory tracking
>    - Product reviews/ratings
>    - Related products/upsells
>    - Product search/filtering
> 
> 2. PRICING & DISCOUNTS
>    - Dynamic pricing
>    - Promo codes (percentage, fixed, BOGO, first-order)
>    - Discount rules (min purchase, specific products, user segments)
>    - Coupon limits (usage count, expiry, one-per-user)
>    - Bulk/tiered pricing
>    - Flash sales/time-limited offers
> 
> 3. SUBSCRIPTION & BILLING
>    - Stripe Subscriptions (monthly/annual)
>    - Usage-based billing (metered)
>    - Subscription tiers/plans
>    - Trial periods
>    - Upgrade/downgrade flows
>    - Proration handling
>    - Dunning management (failed payments)
>    - Invoices/receipts
>    - Tax handling (Stripe Tax)
> 
> 4. SCHEDULING & BOOKING
>    - Calendly-like booking widget
>    - Cal.com integration
>    - Availability management
>    - Time slots/duration options
>    - Buffer time between bookings
>    - Recurring appointments
>    - Booking confirmations
>    - Calendar sync (Google/Outlook)
>    - Timezone handling
>    - Cancellation/rescheduling policies
> 
> 5. EMAIL & NOTIFICATIONS
>    - Transactional emails (Resend/SendGrid)
>    - Email templates (order confirmation, shipping, password reset)
>    - Newsletter/marketing emails
>    - In-app notifications
>    - Push notifications (web)
>    - SMS notifications (Twilio)
>    - Notification preferences
>    - Email verification flows
> 
> 6. USER MANAGEMENT
>    - User profiles
>    - User roles (admin, member, guest)
>    - Teams/organizations
>    - Invitations
>    - API keys management
>    - Activity logs
>    - Account deletion
> 
> 7. CONTENT & CMS
>    - Blog/articles (MDX)
>    - Dynamic pages
>    - Media library
>    - Rich text editor
>    - SEO metadata
>    - Sitemap generation
> 
> 8. SEARCH & FILTERING
>    - Full-text search (Meilisearch/Algolia)
>    - Faceted filtering
>    - Search suggestions
>    - Recent searches
>    - Search analytics
> 
> 9. FORMS & VALIDATION
>    - Form builder patterns (react-hook-form + zod)
>    - Multi-step forms
>    - File uploads (with preview)
>    - Form persistence (draft saving)
>    - CAPTCHA integration
> 
> 10. INTERNATIONALIZATION
>     - i18n setup (next-intl)
>     - Locale detection
>     - Currency formatting
>     - Date/time formatting
>     - RTL support
>     - Translation management
> 
> 11. ERROR HANDLING & MONITORING
>     - Sentry integration
>     - Error boundaries (React)
>     - Retry logic with exponential backoff
>     - Circuit breaker pattern
>     - Health checks
> 
> 12. WEBHOOKS
>     - Webhook receiver (Stripe, etc.)
>     - Webhook signature verification
>     - Event queue/retry
>     - Webhook logs
> 
> 13. RATE LIMITING & SECURITY
>     - API rate limiting (Redis-based)
>     - Request throttling
>     - IP blocking
>     - CORS configuration
>     - CSP headers
> 
> 14. CACHING
>     - Redis caching patterns
>     - Cache invalidation
>     - SWR/React Query patterns
> 
> 15. FEATURE FLAGS
>     - Feature flag service
>     - A/B testing
>     - Gradual rollouts
>     - User segments
> 
> 16. AFFILIATE & REFERRALS
>     - Referral codes
>     - Affiliate tracking
>     - Commission calculation
>     - Payout management
> 
> 17. LOYALTY & REWARDS
>     - Points system
>     - Reward tiers
>     - Redemption flows
> 
> Please provide the most production-ready, type-safe, and battle-tested patterns for each.



-----------------------------

Continue

-----------------------------

Continue

-----------------------------

 

-----------------------------

Resume

-----------------------------

resume

-----------------------------

 