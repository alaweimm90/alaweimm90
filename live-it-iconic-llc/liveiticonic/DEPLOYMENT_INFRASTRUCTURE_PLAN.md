# Live It Iconic Wellness Platform - Infrastructure & Deployment Plan

## Overview
This document outlines the infrastructure needed to deploy the comprehensive wellness platform and utilize the $800 credit effectively.

---

## 🎯 Recommended Infrastructure Setup

### Option 1: AWS Full Stack (~$800/month capacity)

#### **Frontend Hosting**
- **Service:** AWS Amplify or S3 + CloudFront
- **Cost:** ~$20-50/month
- **Features:**
  - Global CDN
  - Automatic SSL
  - CI/CD from GitHub
  - PWA support

#### **Database**
- **Service:** Amazon RDS (PostgreSQL) or Amazon Aurora Serverless
- **Tier:** db.t3.medium (2 vCPU, 4GB RAM)
- **Cost:** ~$100-150/month
- **Storage:** 100GB SSD
- **Backup:** Automated daily backups

#### **Serverless Functions**
- **Service:** AWS Lambda + API Gateway
- **Cost:** ~$50-100/month (1M requests)
- **Use Cases:**
  - Wearable data sync
  - AI recommendations
  - Background jobs
  - Webhook handlers

#### **Caching & Sessions**
- **Service:** Amazon ElastiCache (Redis)
- **Tier:** cache.t3.micro
- **Cost:** ~$15-20/month
- **Use Cases:**
  - Rate limiting
  - Session management
  - API response caching

#### **File Storage**
- **Service:** Amazon S3
- **Cost:** ~$10-20/month
- **Storage:** 100GB for user uploads, progress photos, etc.

#### **AI/ML Services**
- **Service:** Amazon SageMaker (for AI recommendations)
- **Cost:** ~$50-100/month
- **OR:** Use OpenAI API (~$50/month)

#### **Monitoring & Analytics**
- **Service:** CloudWatch + X-Ray
- **Cost:** ~$30-50/month
- **Features:**
  - Real-time monitoring
  - Log aggregation
  - Performance tracing

#### **Push Notifications**
- **Service:** Amazon SNS
- **Cost:** ~$5-10/month
- **Capacity:** 1M push notifications

#### **Email Service**
- **Service:** Amazon SES
- **Cost:** ~$5-10/month
- **Capacity:** 50K emails/month

#### **Background Jobs**
- **Service:** AWS Step Functions + EventBridge
- **Cost:** ~$20-30/month
- **Use Cases:**
  - Daily sync jobs
  - Report generation
  - Data cleanup

**Total AWS Estimate:** ~$305-490/month

---

### Option 2: Google Cloud Platform (~$800/month capacity)

#### **Frontend**
- **Service:** Firebase Hosting or Cloud Storage + Cloud CDN
- **Cost:** ~$20-40/month

#### **Database**
- **Service:** Cloud SQL (PostgreSQL) or Firestore
- **Cost:** ~$120-180/month
- **Tier:** db-n1-standard-2 (2 vCPU, 7.5GB RAM)

#### **Functions**
- **Service:** Cloud Functions
- **Cost:** ~$50-100/month

#### **Redis Cache**
- **Service:** Memorystore for Redis
- **Cost:** ~$25-35/month

#### **Storage**
- **Service:** Cloud Storage
- **Cost:** ~$10-25/month

#### **AI/ML**
- **Service:** Vertex AI
- **Cost:** ~$100-150/month

#### **Monitoring**
- **Service:** Cloud Monitoring + Logging
- **Cost:** ~$40-60/month

**Total GCP Estimate:** ~$365-590/month

---

### Option 3: Vercel + Supabase (Recommended for Speed) 🚀

#### **Frontend & API**
- **Service:** Vercel Pro
- **Cost:** $20/month
- **Features:**
  - Unlimited deployments
  - Edge functions
  - Analytics
  - 100GB bandwidth

#### **Database & Backend**
- **Service:** Supabase Pro
- **Cost:** $25/month
- **Features:**
  - 8GB database
  - 50GB bandwidth
  - 250GB storage
  - Daily backups
  - Real-time subscriptions
  - Auth included

#### **Redis Cache**
- **Service:** Upstash Redis
- **Cost:** ~$20-40/month
- **Features:**
  - Serverless Redis
  - Global replication
  - REST API

#### **AI Services**
- **Service:** OpenAI API
- **Cost:** ~$50-100/month
- **Use Cases:**
  - Recommendations
  - Content generation
  - Image recognition for food logging

#### **Push Notifications**
- **Service:** OneSignal Free tier or Pusher (~$50/month)
- **Cost:** $0-50/month

#### **Email**
- **Service:** Resend or SendGrid
- **Cost:** ~$20/month (50K emails)

#### **Storage**
- **Service:** Cloudflare R2 or AWS S3
- **Cost:** ~$10-20/month

#### **Monitoring**
- **Service:** Sentry + LogRocket
- **Cost:** ~$50-80/month

**Total Vercel+Supabase Estimate:** ~$195-350/month

**Remaining budget:** ~$450-605 for:
- Additional compute resources
- More AI API usage
- Higher tier services
- Development environments
- Testing environments

---

### Option 4: Azure Full Stack (~$800/month capacity)

#### **Frontend**
- **Service:** Azure Static Web Apps
- **Cost:** ~$10-20/month

#### **Database**
- **Service:** Azure Database for PostgreSQL
- **Cost:** ~$150-200/month

#### **Functions**
- **Service:** Azure Functions
- **Cost:** ~$50-100/month

#### **Cache**
- **Service:** Azure Cache for Redis
- **Cost:** ~$20-40/month

#### **Storage**
- **Service:** Azure Blob Storage
- **Cost:** ~$10-25/month

#### **AI**
- **Service:** Azure OpenAI Service
- **Cost:** ~$100-150/month

**Total Azure Estimate:** ~$340-535/month

---

## 🚀 Quick Start Deployment (Vercel + Supabase)

### Step 1: Supabase Setup (15 minutes)

```bash
# Install Supabase CLI
npm install -g supabase

# Initialize Supabase
cd live-it-iconic
supabase init

# Link to cloud project
supabase link --project-ref YOUR_PROJECT_ID

# Run migrations
supabase db push
```

### Step 2: Deploy to Vercel (5 minutes)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd live-it-iconic
vercel --prod
```

### Step 3: Configure Environment Variables

```env
# Supabase
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key

# Wearable APIs
VITE_FITBIT_CLIENT_ID=
VITE_APPLE_CLIENT_ID=
VITE_GARMIN_CLIENT_ID=
VITE_WHOOP_CLIENT_ID=
VITE_OURA_CLIENT_ID=
VITE_SAMSUNG_CLIENT_ID=

# AI Services
VITE_OPENAI_API_KEY=

# Other Services
VITE_REDIS_URL=
VITE_SENDGRID_API_KEY=
```

---

## 📊 Database Schema Migration

```sql
-- Create wellness tables
CREATE TABLE health_metrics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  heart_rate INTEGER,
  steps INTEGER,
  calories_burned INTEGER,
  sleep_hours DECIMAL,
  sleep_quality TEXT,
  active_minutes INTEGER,
  distance DECIMAL,
  vo2_max DECIMAL,
  resting_heart_rate INTEGER,
  hrv INTEGER,
  source TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, timestamp, source)
);

CREATE TABLE mental_health_entries (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  mood TEXT NOT NULL,
  stress INTEGER NOT NULL CHECK (stress >= 1 AND stress <= 10),
  anxiety INTEGER CHECK (anxiety >= 1 AND anxiety <= 10),
  energy INTEGER CHECK (energy >= 1 AND energy <= 10),
  notes TEXT,
  activities TEXT[],
  triggers TEXT[],
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE habits (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  category TEXT NOT NULL,
  frequency TEXT NOT NULL,
  target_count INTEGER NOT NULL DEFAULT 1,
  icon TEXT,
  color TEXT,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE habit_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  habit_id UUID REFERENCES habits NOT NULL,
  user_id UUID REFERENCES auth.users NOT NULL,
  completed_at TIMESTAMPTZ NOT NULL,
  value DECIMAL,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE habit_streaks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  habit_id UUID REFERENCES habits UNIQUE NOT NULL,
  current_streak INTEGER DEFAULT 0,
  longest_streak INTEGER DEFAULT 0,
  last_completed_date TIMESTAMPTZ,
  total_completions INTEGER DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX idx_health_metrics_user_time ON health_metrics(user_id, timestamp DESC);
CREATE INDEX idx_mental_health_user_time ON mental_health_entries(user_id, timestamp DESC);
CREATE INDEX idx_habits_user ON habits(user_id) WHERE is_active = TRUE;
CREATE INDEX idx_habit_logs_habit ON habit_logs(habit_id, completed_at DESC);

-- Add RLS policies
ALTER TABLE health_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE mental_health_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE habits ENABLE ROW LEVEL SECURITY;
ALTER TABLE habit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE habit_streaks ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view own health metrics" ON health_metrics
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own health metrics" ON health_metrics
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Repeat for other tables...
```

---

## 🔧 Infrastructure as Code

### Terraform Configuration (AWS)

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# RDS Database
resource "aws_db_instance" "wellness_db" {
  identifier           = "liveiticonic-wellness"
  engine              = "postgres"
  engine_version      = "15.3"
  instance_class      = "db.t3.medium"
  allocated_storage   = 100
  storage_type        = "gp3"

  db_name             = "wellness"
  username            = var.db_username
  password            = var.db_password

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  skip_final_snapshot = false
  final_snapshot_identifier = "liveiticonic-wellness-final"

  tags = {
    Name        = "LiveItIconic Wellness DB"
    Environment = "production"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "liveiticonic-cache"
  engine              = "redis"
  node_type           = "cache.t3.micro"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379

  security_group_ids = [aws_security_group.redis.id]
  subnet_group_name  = aws_elasticache_subnet_group.main.name
}

# S3 Bucket for storage
resource "aws_s3_bucket" "wellness_storage" {
  bucket = "liveiticonic-wellness-storage"

  tags = {
    Name        = "LiveItIconic Wellness Storage"
    Environment = "production"
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket.wellness_storage.bucket_regional_domain_name
    origin_id   = "S3-liveiticonic"
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-liveiticonic"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
```

---

## 📈 Scaling Plan

### Immediate (Months 1-3)
- **Users:** 0-1,000
- **Infrastructure:** Vercel + Supabase Pro
- **Cost:** ~$200-350/month
- **Features:** Full platform, all wellness features

### Growth (Months 4-6)
- **Users:** 1,000-10,000
- **Infrastructure:** Upgrade Supabase, add Redis
- **Cost:** ~$400-600/month
- **Add:** Better caching, CDN optimization

### Scale (Months 7-12)
- **Users:** 10,000-100,000
- **Infrastructure:** Move to AWS/GCP, multi-region
- **Cost:** ~$800-1,500/month
- **Add:** Load balancers, auto-scaling, advanced monitoring

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Create Supabase project
- [ ] Set up Vercel account
- [ ] Configure domain DNS
- [ ] Generate API keys for wearables
- [ ] Set up monitoring (Sentry)
- [ ] Configure email service

### Deployment
- [ ] Run database migrations
- [ ] Deploy frontend to Vercel
- [ ] Configure environment variables
- [ ] Enable SSL/HTTPS
- [ ] Set up CI/CD pipeline
- [ ] Configure service worker

### Post-Deployment
- [ ] Test all features
- [ ] Run E2E tests
- [ ] Monitor performance
- [ ] Set up alerts
- [ ] Configure backups
- [ ] Load test

### Wearable Integration Setup
- [ ] Register Fitbit app
- [ ] Register Apple HealthKit
- [ ] Register Garmin Connect IQ
- [ ] Register WHOOP API access
- [ ] Register Oura Cloud
- [ ] Register Samsung Health

---

## 💰 Cost Optimization Tips

1. **Use Vercel + Supabase** for first 6 months (cheapest, fastest to deploy)
2. **Implement aggressive caching** to reduce database queries
3. **Use edge functions** for global performance
4. **Optimize images** with CDN
5. **Monitor usage** and scale based on actual needs
6. **Use serverless** for variable workloads
7. **Set up cost alerts** in cloud provider

---

## 🎯 Recommended Action Plan

**If you have $800/month credit:**

### Best Option: AWS or GCP Full Stack
1. Deploy complete infrastructure (~$500/month)
2. Use remaining $300 for:
   - Development environment
   - Testing environment
   - AI API usage (OpenAI)
   - Premium monitoring tools

### Quick Win: Vercel + Supabase + Extra Services
1. Core platform: ~$200/month
2. Remaining $600 for:
   - Premium AI services ($200/month)
   - Advanced analytics ($100/month)
   - Email service premium ($50/month)
   - SMS notifications ($50/month)
   - Premium support ($100/month)
   - Marketing tools ($100/month)

---

**Ready to deploy? Tell me which platform has the credit and I'll create the deployment scripts!** 🚀
