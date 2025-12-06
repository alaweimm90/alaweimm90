import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Lightbulb, Play, RotateCcw, Palette, Target, Sparkles } from "lucide-react";

interface CreativeOutput {
  id: string;
  role: "creator" | "refiner" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  iteration?: number;
}

export const CreativeSandbox = () => {
  const [task, setTask] = useState("");
  const [creatorModel, setCreatorModel] = useState("");
  const [refinerModel, setRefinerModel] = useState("");
  const [brandVoice, setBrandVoice] = useState("");
  const [targetAudience, setTargetAudience] = useState("");
  const [outputs, setOutputs] = useState<CreativeOutput[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [finalScore, setFinalScore] = useState({ creativity: 0, brandAlignment: 0, audienceRelevance: 0 });

  const models = [
    "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.1 70B"
  ];

  const creativeTasks = [
    "Generate a marketing campaign for a sustainable coffee brand",
    "Create a product launch strategy for a new fitness app",
    "Design a brand identity for an eco-friendly fashion startup",
    "Develop a social media campaign for a tech conference"
  ];

  const brandVoices = [
    "Professional & Authoritative",
    "Friendly & Conversational", 
    "Bold & Innovative",
    "Warm & Trustworthy",
    "Playful & Creative"
  ];

  const audiences = [
    "Young professionals (25-35)",
    "Tech enthusiasts",
    "Eco-conscious consumers",
    "Small business owners",
    "College students"
  ];

  const startCreativeProcess = async () => {
    if (!task || !creatorModel || !refinerModel) return;
    
    setIsRunning(true);
    setCurrentIteration(1);
    setOutputs([{
      id: "system-start",
      role: "system",
      content: `Creative collaboration started\nTask: ${task}\nBrand Voice: ${brandVoice}\nTarget Audience: ${targetAudience}\n\nCreator (${creatorModel}) will generate initial concept...`,
      timestamp: new Date()
    }]);

    // Mock creative process simulation
    setTimeout(() => {
      setOutputs(prev => [...prev, {
        id: "creation-1",
        role: "creator",
        content: `# "EcoBlend Coffee: Brewing a Better Tomorrow" Campaign

## Core Concept
A storytelling-driven campaign that follows coffee beans from sustainable farms to consumer cups, highlighting environmental impact at each step.

## Key Elements:
- **Tagline**: "Every Sip Sustains"
- **Visual Identity**: Earth-tone palette with hand-drawn illustrations
- **Campaign Pillars**: 
  1. Farmer partnerships & fair trade
  2. Carbon-neutral shipping
  3. Biodegradable packaging

## Content Strategy:
- Farm-to-cup video series
- Interactive sustainability calculator
- Monthly "Impact Reports" showing environmental benefits
- User-generated content with #EverySipSustains

## Launch Plan:
Phase 1: Teaser campaign with mystery sustainability facts
Phase 2: Full reveal with founder story
Phase 3: Community challenge & engagement`,
        timestamp: new Date(),
        model: creatorModel,
        iteration: 1
      }]);
    }, 2000);

    setTimeout(() => {
      setOutputs(prev => [...prev, {
        id: "refinement-1",
        role: "refiner",
        content: `**Refinement Analysis & Suggestions:**

## Strengths:
✅ Strong environmental narrative
✅ Clear brand differentiation
✅ Authentic storytelling approach

## Areas for Enhancement:

### 1. Brand Voice Alignment
The current tone is slightly too corporate for our "Friendly & Conversational" brand voice. Suggest:
- More personal, intimate language
- Coffee community terminology
- Warmer, approachable messaging

### 2. Target Audience Specificity
Young professionals need:
- Mobile-first content strategy
- LinkedIn/Instagram focus over Facebook
- Career-growth sustainability angle ("fuel your ambition sustainably")

### 3. Competitive Differentiation
Add unique elements:
- Coffee subscription with impact tracking
- AR packaging that shows farm origins
- Partnership with local co-working spaces

## Refined Tagline Options:
- "Good Coffee, Good Conscience"
- "Sustainably Fueled" 
- "Coffee That Cares"

Recommend proceeding with iteration 2 focusing on tone adjustment and audience specificity.`,
        timestamp: new Date(),
        model: refinerModel,
        iteration: 1
      }]);
    }, 4000);

    setTimeout(() => {
      setCurrentIteration(2);
      setFinalScore({ creativity: 88, brandAlignment: 92, audienceRelevance: 85 });
    }, 5500);
  };

  const resetSandbox = () => {
    setOutputs([]);
    setIsRunning(false);
    setCurrentIteration(0);
    setFinalScore({ creativity: 0, brandAlignment: 0, audienceRelevance: 0 });
  };

  return (
    <div className="space-y-6">
      {/* Configuration Panel */}
      <Card className="p-6 gradient-surface">
        <div className="flex items-center gap-2 mb-4">
          <Lightbulb className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-bold text-foreground">Creative Brief Configuration</h3>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Creator Model</label>
            <Select value={creatorModel} onValueChange={setCreatorModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select creator model" />
              </SelectTrigger>
              <SelectContent>
                {models.map(model => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Refiner Model</label>
            <Select value={refinerModel} onValueChange={setRefinerModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select refiner model" />
              </SelectTrigger>
              <SelectContent>
                {models.map(model => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Brand Voice</label>
            <Select value={brandVoice} onValueChange={setBrandVoice}>
              <SelectTrigger>
                <SelectValue placeholder="Select brand voice" />
              </SelectTrigger>
              <SelectContent>
                {brandVoices.map(voice => (
                  <SelectItem key={voice} value={voice}>{voice}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Target Audience</label>
            <Select value={targetAudience} onValueChange={setTargetAudience}>
              <SelectTrigger>
                <SelectValue placeholder="Select target audience" />
              </SelectTrigger>
              <SelectContent>
                {audiences.map(audience => (
                  <SelectItem key={audience} value={audience}>{audience}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="mb-4">
          <label className="text-sm font-medium text-foreground mb-2 block">Creative Task</label>
          <Textarea
            placeholder="Describe the creative task you want the AI models to collaborate on..."
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="min-h-[80px]"
          />
          <div className="flex flex-wrap gap-2 mt-2">
            {creativeTasks.map((suggestion, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => setTask(suggestion)}
                className="text-xs"
              >
                {suggestion}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex gap-2">
          <Button 
            onClick={startCreativeProcess} 
            disabled={!task || !creatorModel || !refinerModel || isRunning}
            variant="gradient"
          >
            <Play className="h-4 w-4" />
            Start Collaboration
          </Button>
          <Button onClick={resetSandbox} variant="outline">
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
        </div>
      </Card>

      {/* Creative Workspace */}
      {outputs.length > 0 && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Collaboration Feed */}
          <div className="lg:col-span-2">
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-foreground">Creative Collaboration</h3>
                {isRunning && (
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-primary animate-pulse" />
                    <span className="text-sm text-primary">Iteration {currentIteration}</span>
                  </div>
                )}
              </div>

              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {outputs.map((output) => (
                  <div key={output.id} className="animate-fade-in">
                    {output.role === "system" ? (
                      <div className="text-center py-2">
                        <Badge variant="secondary">{output.content}</Badge>
                      </div>
                    ) : (
                      <Card className={`p-4 ${
                        output.role === "creator" 
                          ? "border-l-4 border-l-primary bg-primary/5" 
                          : "border-l-4 border-l-accent bg-accent/5"
                      }`}>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className={
                            output.role === "creator" 
                              ? "bg-primary/10 text-primary" 
                              : "bg-accent/10 text-accent"
                          }>
                            {output.role === "creator" ? "Creator" : "Refiner"} 
                          </Badge>
                          <span className="text-sm text-muted-foreground">{output.model}</span>
                          {output.iteration && (
                            <Badge variant="outline" className="text-xs">
                              v{output.iteration}
                            </Badge>
                          )}
                        </div>
                        <div className="text-foreground whitespace-pre-line prose prose-sm max-w-none">
                          {output.content}
                        </div>
                      </Card>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Scoring Panel */}
          <div className="space-y-4">
            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Target className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-bold text-foreground">Creative Scoring</h3>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Creativity</span>
                  <Badge className="bg-primary/10 text-primary">{finalScore.creativity}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-500"
                    style={{ width: `${finalScore.creativity}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Brand Alignment</span>
                  <Badge className="bg-accent/10 text-accent">{finalScore.brandAlignment}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-accent h-2 rounded-full transition-all duration-500"
                    style={{ width: `${finalScore.brandAlignment}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Audience Relevance</span>
                  <Badge className="bg-primary/10 text-primary">{finalScore.audienceRelevance}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-500"
                    style={{ width: `${finalScore.audienceRelevance}%` }}
                  ></div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-border">
                <h4 className="text-sm font-medium text-foreground mb-2">Evaluation Criteria</h4>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div>• Originality & innovation</div>
                  <div>• Brand voice adherence</div>
                  <div>• Target audience relevance</div>
                  <div>• Collaborative refinement</div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Palette className="h-5 w-5 text-accent" />
                <h3 className="text-lg font-bold text-foreground">Creative Insights</h3>
              </div>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>• Strong environmental storytelling approach</p>
                <p>• Effective brand voice alignment after refinement</p>
                <p>• Good audience-specific adaptation</p>
                <p>• Creative use of interactive elements</p>
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};