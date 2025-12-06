import { useEffect, useState, Suspense, lazy, memo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Navigation } from "@/components/Navigation";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import {
  Swords,
  Lightbulb,
  GraduationCap,
  Play,
  Settings,
  Users,
  MessageCircle,
  Trophy,
  ArrowRight
} from "lucide-react";
import { setSEO, injectJsonLd } from "@/lib/seo";
import { trackEvent } from "@/lib/analytics";

const DebateMode = lazy(() => import("@/components/arena/DebateMode").then(m => ({ default: m.DebateMode })));
const CreativeSandbox = lazy(() => import("@/components/arena/CreativeSandbox").then(m => ({ default: m.CreativeSandbox })));
const ExplanationChallenge = lazy(() => import("@/components/arena/ExplanationChallenge").then(m => ({ default: m.ExplanationChallenge })));
const EnhancedDebateArena = lazy(() => import("@/components/arena/EnhancedDebateArena").then(m => ({ default: m.EnhancedDebateArena })));
const DemoScenarios = lazy(() => import("@/components/arena/DemoScenarios").then(m => ({ default: m.DemoScenarios })));

const Arena = memo(() => {
  const [activeMode, setActiveMode] = useState("overview");
  const [selectedDemo, setSelectedDemo] = useState<any>(null);
  const [isDemoPlaying, setIsDemoPlaying] = useState(false);

  useEffect(() => {
    setSEO({
      title: "The Arena | LLM Works",
      description: "Open‑source interactive evaluation that runs in your browser. Debates, creative tasks, explanations.",
      path: "/arena",
    });
    injectJsonLd({
      "@context": "https://schema.org",
      "@type": "WebPage",
      name: "The Arena — Interactive AI Evaluation",
      url: `${window.location.origin}/arena`,
      description: "Local‑first multi‑agent testing: debates, creative, explanations.",
      isPartOf: {
        "@type": "SoftwareApplication",
        name: "LLM Works",
      },
    }, "ld-arena");
  }, []);

  const modes = [
    {
      id: "debate",
      icon: Swords,
      title: "Debate Mode",
      description: "Pit models against each other in structured debates",
      features: ["Proponent vs Skeptic", "Citation Requirements", "Multi-round scoring"],
      status: "Available",
      color: "bg-accent/10 text-accent"
    },
    {
      id: "enhanced-debate",
      icon: Trophy,
      title: "Neural Arena",
      description: "Epic AI battles with real-time combat visualization",
      features: ["AI Personality Avatars", "Epic Confrontation Effects", "Battle State Animations"],
      status: "Available",
      color: "bg-yellow-500/10 text-yellow-600"
    },
    {
      id: "creative",
      icon: Lightbulb,
      title: "Creative Sandbox",
      description: "Collaborative creative tasks with iterative refinement",
      features: ["Creator + Refiner roles", "Brand voice adherence", "Creative scoring"],
      status: "Available", 
      color: "bg-primary/10 text-primary"
    },
    {
      id: "explanation",
      icon: GraduationCap,
      title: "Explanation Challenge",
      description: "Test adaptive explanation abilities",
      features: ["Expert + Student roles", "Clarity optimization", "Empathy scoring"],
      status: "Available",
      color: "bg-accent/10 text-accent"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main id="main" className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 border border-accent/20 mb-6">
            <Swords className="h-4 w-4 text-accent" />
            <span className="text-sm font-medium text-accent neural-text">INTERACTIVE EVALUATION</span>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6">
            THE ARENA — INTERACTIVE TESTING
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Test models through debates, creative tasks, and explanations. Get comparative analysis through structured interactions.
          </p>
        </div>

        {/* Interactive Arena */}
        <Tabs value={activeMode} onValueChange={(val) => { setActiveMode(val); trackEvent('arena_tab_change', { tab: val }); }} className="w-full">
          <TabsList className="grid w-full grid-cols-6 mb-8">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="scenarios">Demo Scenarios</TabsTrigger>
            <TabsTrigger value="debate">Debate Mode</TabsTrigger>
            <TabsTrigger value="enhanced-debate">Neural Arena</TabsTrigger>
            <TabsTrigger value="creative">Creative Sandbox</TabsTrigger>
            <TabsTrigger value="explanation">Explanation Challenge</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-8">
            {/* Interaction Modes */}
            <div className="grid lg:grid-cols-2 xl:grid-cols-4 gap-8">
              {modes.map((mode) => (
                <Card key={mode.id} className="p-8 shadow-medium hover:shadow-strong transition-all duration-300 group">
                  <div className="flex flex-col h-full">
                    <div className="flex items-center justify-between mb-6">
                      <div className={`p-3 rounded-lg ${mode.color.split(' text-')[0]} group-hover:scale-110 transition-transform`}>
                        <mode.icon className={`h-8 w-8 ${mode.color.split(' ')[1]}`} />
                      </div>
                      <Badge variant={mode.status === "Available" ? "default" : "secondary"}>
                        {mode.status}
                      </Badge>
                    </div>

                    <h3 className="text-2xl font-bold text-foreground mb-4">{mode.title}</h3>
                    <p className="text-muted-foreground mb-6 flex-grow">{mode.description}</p>

                    <div className="space-y-3 mb-6">
                      {mode.features.map((feature, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-accent"></div>
                          <span className="text-sm text-muted-foreground">{feature}</span>
                        </div>
                      ))}
                    </div>

                    <Button 
                      variant={mode.status === "Available" ? "gradient" : "secondary"} 
                      className="w-full"
                      onClick={() => mode.status === "Available" && setActiveMode(mode.id)}
                      disabled={mode.status !== "Available"}
                    >
                      {mode.status === "Available" ? (
                        <>
                          <Play className="h-4 w-4" />
                          Launch Mode
                        </>
                      ) : (
                        "Coming Soon"
                      )}
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="scenarios">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Demo Scenarios..." />}>
              <DemoScenarios
                onScenarioSelect={(scenario) => {
                  setSelectedDemo(scenario);
                  trackEvent('demo_scenario_selected', { scenario: scenario.id });
                }}
                onScenarioStart={() => {
                  setIsDemoPlaying(true);
                  setActiveMode('enhanced-debate');
                  trackEvent('demo_scenario_started', { scenario: selectedDemo?.id });
                }}
                onScenarioStop={() => {
                  setIsDemoPlaying(false);
                  trackEvent('demo_scenario_stopped', { scenario: selectedDemo?.id });
                }}
                currentScenarioId={selectedDemo?.id}
                isPlaying={isDemoPlaying}
              />
            </Suspense>
          </TabsContent>

          <TabsContent value="debate">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Debate Mode..." />}>
              <DebateMode />
            </Suspense>
          </TabsContent>

          <TabsContent value="creative">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Creative Sandbox..." />}>
              <CreativeSandbox />
            </Suspense>
          </TabsContent>

          <TabsContent value="enhanced-debate">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Neural Arena..." />}>
              <EnhancedDebateArena
                topic="The future of artificial intelligence will be more beneficial than harmful to humanity"
                leftDebater={{
                  id: "analytical-ai",
                  name: "Logic Prime",
                  personality: "analytical",
                  energy: 85,
                  battleState: "idle",
                  arguments: []
                }}
                rightDebater={{
                  id: "creative-ai", 
                  name: "Spark Mind",
                  personality: "creative",
                  energy: 90,
                  battleState: "idle",
                  arguments: []
                }}
                onDebateComplete={(winner, finalScore) => {
                  trackEvent('enhanced_debate_complete', { winner, finalScore });
                  console.log('Debate completed:', { winner, finalScore });
                }}
              />
            </Suspense>
          </TabsContent>

          <TabsContent value="explanation">
            <Suspense fallback={<LoadingSpinner size="lg" text="Loading Explanation Challenge..." />}>
              <ExplanationChallenge />
            </Suspense>
          </TabsContent>
        </Tabs>

        {/* Quick Start Section */}
        <Card className="p-8 gradient-surface shadow-medium">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-foreground mb-4">Start Interactive Testing</h3>
              <p className="text-muted-foreground mb-4">
                Configure your models, set up interaction parameters, and run evaluations that reveal model capabilities and limitations.
              </p>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-accent" />
                  <span className="text-sm text-muted-foreground">Multi-agent orchestration</span>
                </div>
                <div className="flex items-center gap-2">
                  <MessageCircle className="h-4 w-4 text-accent" />
                  <span className="text-sm text-muted-foreground">Real-time interaction</span>
                </div>
                <div className="flex items-center gap-2">
                  <Trophy className="h-4 w-4 text-accent" />
                  <span className="text-sm text-muted-foreground">AI-powered judging</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-3">
              <Button variant="hero" size="lg">
                <Settings className="h-4 w-4" />
                Configure Models
                <ArrowRight className="h-4 w-4" />
              </Button>
              <Button variant="outline">
                View Documentation
              </Button>
            </div>
          </div>
        </Card>
      </main>
    </div>
  );
});

Arena.displayName = 'Arena';

export default Arena;