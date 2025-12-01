---
name: "Platform & Deployment Superprompt"
version: "1.0"
category: "project"
tags: ["deployment", "cloud", "infrastructure", "responsive", "seo", "performance"]
created: "2024-11-30"
---

# Platform & Deployment Superprompt

## Purpose

Comprehensive framework for web platform deployment, cloud infrastructure, responsive design implementation, SEO optimization, and performance engineering.

---

## System Prompt

```text
You are a Platform Engineer and Cloud Architect with expertise in:
- Cloud platforms (AWS, GCP, Azure, Vercel, Netlify)
- Infrastructure as Code (Terraform, Pulumi, CDK)
- Container orchestration (Kubernetes, Docker)
- CDN and edge computing
- SEO and web performance optimization
- Responsive and progressive web applications

Your mission is to build platforms that:
1. Scale automatically with demand
2. Maintain high availability and reliability
3. Optimize for performance and SEO
4. Support global distribution
5. Enable rapid deployment and rollback
```

---

## Cloud Infrastructure

### Multi-Cloud Architecture

```yaml
infrastructure:
  primary_cloud: aws
  regions:
    primary: us-east-1
    secondary: eu-west-1
    edge: cloudflare
    
  services:
    compute:
      - ECS Fargate (containers)
      - Lambda (serverless)
      - EC2 (when needed)
      
    storage:
      - S3 (objects)
      - EFS (shared files)
      - RDS (relational)
      - DynamoDB (NoSQL)
      
    networking:
      - VPC (isolation)
      - ALB (load balancing)
      - CloudFront (CDN)
      - Route53 (DNS)
      
    security:
      - WAF (firewall)
      - Shield (DDoS)
      - KMS (encryption)
      - Secrets Manager
```

### Terraform Configuration

```hcl
# infrastructure/main.tf
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "terraform-state-bucket"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# VPC Configuration
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway     = true
  single_nat_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
  
  tags = var.common_tags
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
      log_configuration {
        cloud_watch_log_group_name = aws_cloudwatch_log_group.ecs.name
      }
    }
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  access_logs {
    bucket  = aws_s3_bucket.logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"
  
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "alb"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "alb"
    
    forwarded_values {
      query_string = true
      headers      = ["Origin", "Authorization"]
      
      cookies {
        forward = "all"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.main.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
}
```

---

## Deployment Strategies

### Vercel Deployment

```json
// vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/$1"
    }
  ],
  "env": {
    "DATABASE_URL": "@database-url",
    "NEXT_PUBLIC_API_URL": "@api-url"
  },
  "regions": ["iad1", "sfo1", "cdg1"],
  "functions": {
    "api/**/*.ts": {
      "memory": 1024,
      "maxDuration": 30
    }
  },
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    },
    {
      "source": "/static/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

### Netlify Configuration

```toml
# netlify.toml
[build]
  command = "npm run build"
  publish = "dist"
  functions = "netlify/functions"

[build.environment]
  NODE_VERSION = "20"
  NPM_VERSION = "10"

[[plugins]]
  package = "@netlify/plugin-lighthouse"

[[plugins]]
  package = "netlify-plugin-cache"
  [plugins.inputs]
    paths = [".next/cache", "node_modules/.cache"]

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
    Permissions-Policy = "camera=(), microphone=(), geolocation=()"

[[headers]]
  for = "/static/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/*.js"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/*.css"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[redirects]]
  from = "/api/*"
  to = "/.netlify/functions/:splat"
  status = 200

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
  conditions = {Role = ["admin"]}

[context.production]
  environment = { NODE_ENV = "production" }

[context.deploy-preview]
  environment = { NODE_ENV = "preview" }

[context.branch-deploy]
  environment = { NODE_ENV = "staging" }
```

---

## SEO Optimization

### Technical SEO Checklist

```yaml
seo_checklist:
  technical:
    crawlability:
      - robots.txt properly configured
      - XML sitemap generated and submitted
      - No orphan pages
      - Proper canonical tags
      - Hreflang for multi-language
      
    indexability:
      - Meta robots tags correct
      - No accidental noindex
      - JavaScript content rendered
      - Dynamic rendering if needed
      
    site_architecture:
      - Flat URL structure
      - Breadcrumb navigation
      - Internal linking strategy
      - Hub and spoke content model
      
  performance:
    core_web_vitals:
      - LCP < 2.5s
      - FID < 100ms
      - CLS < 0.1
      - INP < 200ms
      
    optimization:
      - Image optimization (WebP, AVIF)
      - Code splitting
      - Tree shaking
      - Lazy loading
      - Preloading critical resources
      
  content:
    on_page:
      - Unique title tags (50-60 chars)
      - Meta descriptions (150-160 chars)
      - H1 tags (one per page)
      - Structured data (JSON-LD)
      - Alt text for images
      
    structured_data:
      - Organization schema
      - Website schema
      - Breadcrumb schema
      - Article/Product schema
      - FAQ schema
```

### Next.js SEO Implementation

```tsx
// app/layout.tsx
import { Metadata } from 'next';

export const metadata: Metadata = {
  metadataBase: new URL('https://example.com'),
  title: {
    default: 'Site Name',
    template: '%s | Site Name',
  },
  description: 'Default description for the site',
  keywords: ['keyword1', 'keyword2', 'keyword3'],
  authors: [{ name: 'Author Name' }],
  creator: 'Creator Name',
  publisher: 'Publisher Name',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://example.com',
    siteName: 'Site Name',
    title: 'Site Name',
    description: 'Default description',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'Site Name',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Site Name',
    description: 'Default description',
    creator: '@username',
    images: ['/twitter-image.jpg'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'google-verification-code',
    yandex: 'yandex-verification-code',
  },
  alternates: {
    canonical: 'https://example.com',
    languages: {
      'en-US': 'https://example.com/en-US',
      'de-DE': 'https://example.com/de-DE',
    },
  },
};

// Structured Data Component
export function JsonLd({ data }: { data: object }) {
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  );
}

// Organization Schema
const organizationSchema = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'Company Name',
  url: 'https://example.com',
  logo: 'https://example.com/logo.png',
  sameAs: [
    'https://twitter.com/company',
    'https://linkedin.com/company/company',
    'https://github.com/company',
  ],
  contactPoint: {
    '@type': 'ContactPoint',
    telephone: '+1-xxx-xxx-xxxx',
    contactType: 'customer service',
  },
};
```

### Sitemap Generation

```typescript
// app/sitemap.ts
import { MetadataRoute } from 'next';

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = 'https://example.com';
  
  // Static pages
  const staticPages = [
    '',
    '/about',
    '/contact',
    '/pricing',
    '/blog',
  ].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date(),
    changeFrequency: 'monthly' as const,
    priority: route === '' ? 1 : 0.8,
  }));
  
  // Dynamic pages (e.g., blog posts)
  const posts = await fetchBlogPosts();
  const blogPages = posts.map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: new Date(post.updatedAt),
    changeFrequency: 'weekly' as const,
    priority: 0.6,
  }));
  
  return [...staticPages, ...blogPages];
}
```

---

## Performance Optimization

### Web Vitals Monitoring

```typescript
// lib/web-vitals.ts
import { onCLS, onFID, onLCP, onINP, onTTFB } from 'web-vitals';

type MetricHandler = (metric: {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  id: string;
}) => void;

export function reportWebVitals(onPerfEntry?: MetricHandler) {
  if (onPerfEntry && typeof onPerfEntry === 'function') {
    onCLS(onPerfEntry);
    onFID(onPerfEntry);
    onLCP(onPerfEntry);
    onINP(onPerfEntry);
    onTTFB(onPerfEntry);
  }
}

// Send to analytics
export function sendToAnalytics(metric: any) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    rating: metric.rating,
    delta: metric.delta,
    id: metric.id,
    page: window.location.pathname,
  });

  // Use sendBeacon for reliability
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/api/analytics/vitals', body);
  } else {
    fetch('/api/analytics/vitals', {
      body,
      method: 'POST',
      keepalive: true,
    });
  }
}
```

### Image Optimization

```tsx
// components/OptimizedImage.tsx
import Image from 'next/image';

interface OptimizedImageProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  priority?: boolean;
  className?: string;
}

export function OptimizedImage({
  src,
  alt,
  width,
  height,
  priority = false,
  className,
}: OptimizedImageProps) {
  return (
    <Image
      src={src}
      alt={alt}
      width={width}
      height={height}
      priority={priority}
      loading={priority ? 'eager' : 'lazy'}
      placeholder="blur"
      blurDataURL={`data:image/svg+xml;base64,${toBase64(shimmer(width, height))}`}
      sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
      className={className}
    />
  );
}

// Shimmer placeholder
const shimmer = (w: number, h: number) => `
<svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g">
      <stop stop-color="#f6f7f8" offset="0%" />
      <stop stop-color="#edeef1" offset="50%" />
      <stop stop-color="#f6f7f8" offset="100%" />
    </linearGradient>
  </defs>
  <rect width="${w}" height="${h}" fill="#f6f7f8" />
  <rect width="${w}" height="${h}" fill="url(#g)">
    <animate attributeName="x" from="-${w}" to="${w}" dur="1s" repeatCount="indefinite" />
  </rect>
</svg>`;

const toBase64 = (str: string) =>
  typeof window === 'undefined'
    ? Buffer.from(str).toString('base64')
    : window.btoa(str);
```

---

## Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
        - name: web-app
          image: ghcr.io/org/web-app:latest
          ports:
            - containerPort: 3000
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /api/health
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /api/ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 5
          env:
            - name: NODE_ENV
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-url
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

## Execution Phases

### Phase 1: Infrastructure Setup

- [ ] Configure cloud provider
- [ ] Set up Terraform/IaC
- [ ] Create VPC and networking
- [ ] Configure DNS and SSL

### Phase 2: Deployment Pipeline

- [ ] Set up container registry
- [ ] Configure CI/CD for deployments
- [ ] Implement blue-green deployment
- [ ] Set up monitoring and alerting

### Phase 3: Performance Optimization

- [ ] Configure CDN
- [ ] Implement caching strategy
- [ ] Optimize images and assets
- [ ] Set up Web Vitals monitoring

### Phase 4: SEO Implementation

- [ ] Configure meta tags
- [ ] Generate sitemaps
- [ ] Implement structured data
- [ ] Set up search console

---

*Last updated: 2024-11-30*
