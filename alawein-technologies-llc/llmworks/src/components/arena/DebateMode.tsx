import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Swords, Play, RotateCcw, Crown, Timer, MessageSquare } from "lucide-react";

interface Message {
  id: string;
  role: "proponent" | "skeptic" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  citations?: string[];
}

export const DebateMode = () => {
  const [topic, setTopic] = useState("");
  const [proponentModel, setProponentModel] = useState("");
  const [skepticModel, setSkepticModel] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [scores, setScores] = useState({ proponent: 0, skeptic: 0 });

  const models = [
    "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.1 70B"
  ];

  const mockDebateTopics = [
    "Nuclear fusion is the best long-term solution to the energy crisis",
    "AI should be regulated at the international level",
    "Remote work is more beneficial than in-person work for most companies",
    "Universal Basic Income should be implemented globally"
  ];

  const startDebate = async () => {
    if (!topic || !proponentModel || !skepticModel) return;
    
    setIsRunning(true);
    setCurrentRound(1);
    setMessages([{
      id: "system-start",
      role: "system",
      content: `Debate started: "${topic}"\nProponent (${proponentModel}) vs Skeptic (${skepticModel})\nRound 1 of 3`,
      timestamp: new Date()
    }]);

    // Mock debate simulation
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: "msg-1",
        role: "proponent",
        content: `I strongly believe that ${topic.toLowerCase()}. Here's my opening argument:\n\nThe evidence clearly supports this position because of three key factors: sustainability, efficiency, and scalability. Recent studies from MIT and Stanford demonstrate that this approach offers unprecedented advantages over traditional alternatives.\n\nThe economic implications alone justify immediate investment and implementation.`,
        timestamp: new Date(),
        model: proponentModel,
        citations: ["MIT Energy Initiative 2024", "Stanford Research Paper"]
      }]);
    }, 1500);

    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: "msg-2", 
        role: "skeptic",
        content: `I respectfully disagree with this position. While the proponent raises interesting points, there are significant flaws in this reasoning:\n\nFirst, the studies cited don't account for real-world implementation challenges. Second, the cost-benefit analysis is incomplete. Third, there are viable alternatives that haven't been properly considered.\n\nThe evidence suggests a more nuanced approach is needed.`,
        timestamp: new Date(),
        model: skepticModel,
        citations: ["Economic Analysis Quarterly", "Implementation Studies Review"]
      }]);
    }, 3000);

    setTimeout(() => {
      setScores({ proponent: 85, skeptic: 78 });
      setCurrentRound(2);
    }, 4500);
  };

  const resetDebate = () => {
    setMessages([]);
    setIsRunning(false);
    setCurrentRound(0);
    setScores({ proponent: 0, skeptic: 0 });
  };

  return (
    <div className="space-y-6">
      {/* Configuration Panel */}
      <Card className="p-6 gradient-surface border-trust/20 hover:border-trust/40 transition-colors">
        <div className="flex items-center gap-2 mb-4">
          <Swords className="h-5 w-5 text-trust" />
          <h3 className="text-lg font-bold text-foreground">Combat Configuration</h3>
          <Badge className="bg-trust/10 text-trust">Arena Mode</Badge>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Proponent Model</label>
            <Select value={proponentModel} onValueChange={setProponentModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select proponent model" />
              </SelectTrigger>
              <SelectContent>
                {models.map(model => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Skeptic Model</label>
            <Select value={skepticModel} onValueChange={setSkepticModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select skeptic model" />
              </SelectTrigger>
              <SelectContent>
                {models.map(model => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="mb-4">
          <label className="text-sm font-medium text-foreground mb-2 block">Debate Topic</label>
          <Textarea
            placeholder="Enter a debatable topic or select from suggestions below..."
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            className="min-h-[80px]"
          />
          <div className="flex flex-wrap gap-2 mt-2">
            {mockDebateTopics.map((suggestion, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => setTopic(suggestion)}
                className="text-xs"
              >
                {suggestion}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex gap-2">
          <Button 
            onClick={startDebate} 
            disabled={!topic || !proponentModel || !skepticModel || isRunning}
            variant="trust"
            className="hover-scale"
          >
            <Play className="h-4 w-4" />
            Initiate Combat
          </Button>
          <Button onClick={resetDebate} variant="outline" className="hover:border-risk hover:text-risk">
            <RotateCcw className="h-4 w-4" />
            Reset Arena
          </Button>
        </div>
      </Card>

      {/* Debate Arena */}
      {messages.length > 0 && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Conversation */}
          <div className="lg:col-span-2">
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-foreground">Live Debate</h3>
                {isRunning && (
                  <div className="flex items-center gap-2">
                    <Timer className="h-4 w-4 text-accent animate-pulse" />
                    <span className="text-sm text-accent">Round {currentRound} of 3</span>
                  </div>
                )}
              </div>

              <div className="space-y-4 max-h-[600px] overflow-y-auto">
                {messages.map((message) => (
                  <div key={message.id} className="animate-fade-in">
                    {message.role === "system" ? (
                      <div className="text-center py-2">
                        <Badge variant="secondary">{message.content}</Badge>
                      </div>
                    ) : (
                      <Card className={`p-4 ${
                        message.role === "proponent" 
                          ? "border-l-4 border-l-accent bg-accent/5" 
                          : "border-l-4 border-l-primary bg-primary/5"
                      }`}>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className={
                            message.role === "proponent" 
                              ? "bg-accent/10 text-accent" 
                              : "bg-primary/10 text-primary"
                          }>
                            {message.role === "proponent" ? "Proponent" : "Skeptic"}
                          </Badge>
                          <span className="text-sm text-muted-foreground">{message.model}</span>
                        </div>
                        <p className="text-foreground whitespace-pre-line">{message.content}</p>
                        {message.citations && (
                          <div className="mt-2 pt-2 border-t border-border">
                            <div className="text-xs text-muted-foreground">
                              <strong>Citations:</strong> {message.citations.join(", ")}
                            </div>
                          </div>
                        )}
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
                <Crown className="h-5 w-5 text-accent" />
                <h3 className="text-lg font-bold text-foreground">Live Scoring</h3>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Proponent ({proponentModel})</span>
                  <Badge className="bg-accent/10 text-accent">{scores.proponent}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-accent h-2 rounded-full transition-all duration-500"
                    style={{ width: `${scores.proponent}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Skeptic ({skepticModel})</span>
                  <Badge className="bg-primary/10 text-primary">{scores.skeptic}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-500"
                    style={{ width: `${scores.skeptic}%` }}
                  ></div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-border">
                <h4 className="text-sm font-medium text-foreground mb-2">Scoring Criteria</h4>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div>• Logical consistency</div>
                  <div>• Factual accuracy</div>
                  <div>• Citation quality</div>
                  <div>• Persuasiveness</div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <MessageSquare className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-bold text-foreground">Arbiter Notes</h3>
              </div>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>• Proponent provides strong evidence but lacks counterargument consideration</p>
                <p>• Skeptic raises valid concerns about implementation</p>
                <p>• Both models cite relevant sources</p>
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};