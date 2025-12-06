import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Book, Code, Terminal, FileText, Zap, Download } from 'lucide-react';
import CodeBlock from '@/components/dev/CodeBlock';
import { Breadcrumb, BreadcrumbList, BreadcrumbItem, BreadcrumbLink, BreadcrumbPage } from '@/components/ui/breadcrumb';
import { useSEO } from '@/hooks/useSEO';
import NeuralBackground from '@/components/dev/NeuralBackground';
import AnimatedGrid from '@/components/dev/AnimatedGrid';

export default function DocumentationLayout() {
  useSEO({
    title: 'Documentation — Local‑first usage & methods | Attributa.dev',
    description: 'How to use the in‑browser analyzer, privacy model, and detection methods. APIs/SDKs are planned.'
  });

  const pythonQuickStart = `# Install Attributa SDK
pip install attributa-sdk

# Initialize client
from attributa import AttributaClient

client = AttributaClient(
    api_key="your_api_key",
    endpoint="https://api.attributa.dev"
)

# Analyze text content
result = client.analyze(
    text="Your content to analyze...",
    methods=["gltr", "detectgpt", "watermark"],
    options={
        "confidence_threshold": 0.7,
        "detailed_results": True
    }
)

# Access results
print(f"AI Probability: {result.ai_probability:.3f}")
print(f"Confidence: {result.confidence:.3f}")

# Export findings
result.export("analysis_report.json")`;

  const nodeQuickStart = `// Install Attributa SDK
npm install @attributa/sdk

// Initialize client
import { AttributaClient } from '@attributa/sdk';

const client = new AttributaClient({
  apiKey: process.env.ATTRIBUTA_API_KEY,
  endpoint: 'https://api.attributa.dev'
});

// Analyze content
async function analyzeContent() {
  const result = await client.analyze({
    text: 'Your content to analyze...',
    methods: ['gltr', 'detectgpt', 'watermark'],
    options: {
      confidenceThreshold: 0.7,
      detailedResults: true
    }
  });

  console.log('AI Probability:', result.aiProbability);
  console.log('Confidence:', result.confidence);
  
  // Export results
  await result.export('analysis_report.json');
}`;

  const curlExample = `# Analyze text via REST API
curl -X POST "https://api.attributa.dev/v1/analyze" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Your content to analyze...",
    "methods": ["gltr", "detectgpt", "watermark"],
    "options": {
      "confidence_threshold": 0.7,
      "detailed_results": true
    }
  }'

# Response format
{
  "ai_probability": 0.892,
  "confidence": 0.834,
  "methods": {
    "gltr": {
      "score": 0.923,
      "p_value": 0.001,
      "distribution": [0.67, 0.23, 0.08, 0.02]
    },
    "detectgpt": {
      "score": 0.889,
      "z_score": -2.34,
      "criterion": 3.2
    },
    "watermark": {
      "detected": false,
      "confidence": 0.02,
      "z_score": 0.8
    }
  },
  "metadata": {
    "analysis_time": "2024-01-15T10:30:45Z",
    "word_count": 247,
    "language": "en"
  }
}`;

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary engineering-pattern">
      <NeuralBackground />
      <AnimatedGrid />
      
      <div className="relative z-10 container mx-auto px-4 py-8 space-y-8 animate-fade-in">
        {/* Breadcrumb Navigation */}
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href="/">Home</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbItem>
              <BreadcrumbPage>Documentation</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>

        {/* Header */}
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-primary/10 border border-primary/20">
              <Book className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h1 className="text-4xl font-bold display-tight font-mono">Documentation</h1>
              <p className="text-xl text-muted-foreground">
                Local‑first usage and methods; APIs/SDKs are planned.
              </p>
            </div>
          </div>
          
          <div className="flex gap-3">
            <Badge variant="outline" className="font-mono">Open‑source</Badge>
            <Badge variant="secondary" className="font-mono">Local‑first</Badge>
            <Badge variant="secondary" className="font-mono">APIs planned</Badge>
          </div>
        </div>

        {/* Quick Start Tabs */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-mono">
              <Zap className="h-5 w-5 text-accent" />
              Quick Start (local‑first)
            </CardTitle>
            <CardDescription>
              Use the in‑browser analyzer. SDK/API snippets below are planned.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="python" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="python" className="font-mono">Python (planned)</TabsTrigger>
                <TabsTrigger value="nodejs" className="font-mono">Node.js (planned)</TabsTrigger>
                <TabsTrigger value="curl" className="font-mono">cURL (planned)</TabsTrigger>
              </TabsList>
              
              <TabsContent value="python" className="mt-6">
                <CodeBlock
                  code={pythonQuickStart}
                  language="python"
                  title="quickstart.py"
                  showLineNumbers={true}
                  highlightLines={[8, 15, 23]}
                />
              </TabsContent>
              
              <TabsContent value="nodejs" className="mt-6">
                <CodeBlock
                  code={nodeQuickStart}
                  language="javascript"
                  title="quickstart.js"
                  showLineNumbers={true}
                  highlightLines={[7, 14, 26]}
                />
              </TabsContent>
              
              <TabsContent value="curl" className="mt-6">
                <CodeBlock
                  code={curlExample}
                  language="bash"
                  title="api-example.sh"
                  showLineNumbers={true}
                  collapsible={true}
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Documentation Sections */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card className="glass-card hover:shadow-lg transition-shadow duration-300">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 font-mono text-lg">
                <Code className="h-5 w-5 text-primary" />
                API Reference
              </CardTitle>
              <CardDescription>
                Planned REST API endpoints (not yet available)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">POST /v1/analyze</span>
                  <Badge variant="outline" className="text-xs">Core</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">GET /v1/methods</span>
                  <Badge variant="outline" className="text-xs">Info</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">POST /v1/batch</span>
                  <Badge variant="outline" className="text-xs">Batch</Badge>
                </div>
              </div>
              <Button variant="outline" size="sm" className="w-full font-mono">
                APIs planned
              </Button>
            </CardContent>
          </Card>

          <Card className="glass-card hover:shadow-lg transition-shadow duration-300">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 font-mono text-lg">
                <Terminal className="h-5 w-5 text-accent" />
                SDK Guides
              </CardTitle>
              <CardDescription>
                Planned SDKs
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">Python SDK</span>
                  <Badge variant="secondary" className="text-xs">v2.1.0</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">Node.js SDK</span>
                  <Badge variant="secondary" className="text-xs">v2.1.0</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">Go SDK</span>
                  <Badge variant="outline" className="text-xs">Beta</Badge>
                </div>
              </div>
              <Button variant="outline" size="sm" className="w-full font-mono">
                SDKs planned
              </Button>
            </CardContent>
          </Card>

          <Card className="glass-card hover:shadow-lg transition-shadow duration-300">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 font-mono text-lg">
                <FileText className="h-5 w-5 text-success" />
                Examples
              </CardTitle>
              <CardDescription>
                Real-world implementation patterns
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">Batch Processing</span>
                  <Badge variant="outline" className="text-xs">Guide</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">Real-time Analysis</span>
                  <Badge variant="outline" className="text-xs">Guide</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm">CI/CD Integration</span>
                  <Badge variant="outline" className="text-xs">Guide</Badge>
                </div>
              </div>
              <Button variant="outline" size="sm" className="w-full font-mono">
                View Examples
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Download Links */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-mono">
              <Download className="h-5 w-5 text-primary" />
              Resources & Downloads
            </CardTitle>
          </CardHeader>
          <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <Button variant="outline" className="justify-start gap-2 font-mono">
                  <FileText className="h-4 w-4" />
                  OpenAPI Spec (planned)
                </Button>
                <Button variant="outline" className="justify-start gap-2 font-mono">
                  <Code className="h-4 w-4" />
                  Postman Collection (planned)
                </Button>
                <Button variant="outline" className="justify-start gap-2 font-mono">
                  <Terminal className="h-4 w-4" />
                  CLI Tools (planned)
                </Button>
              </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}