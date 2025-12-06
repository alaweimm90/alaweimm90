import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { GraduationCap, Play, RotateCcw, Brain, HelpCircle, CheckCircle } from "lucide-react";

interface ExplanationMessage {
  id: string;
  role: "expert" | "student" | "system";
  content: string;
  timestamp: Date;
  model?: string;
  clarityScore?: number;
  empathyScore?: number;
}

export const ExplanationChallenge = () => {
  const [topic, setTopic] = useState("");
  const [expertModel, setExpertModel] = useState("");
  const [studentModel, setStudentModel] = useState("");
  const [audienceAge, setAudienceAge] = useState([12]);
  const [complexity, setComplexity] = useState([5]);
  const [messages, setMessages] = useState<ExplanationMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [scores, setScores] = useState({ clarity: 0, empathy: 0, adaptability: 0 });

  const models = [
    "GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.1 70B"
  ];

  const complexTopics = [
    "How does quantum entanglement work?",
    "Why do we dream and what happens in our brain during sleep?",
    "How does artificial intelligence learn and make decisions?",
    "What is DNA and how does it determine our traits?",
    "How do black holes form and what happens inside them?"
  ];

  const startExplanation = async () => {
    if (!topic || !expertModel || !studentModel) return;
    
    setIsRunning(true);
    setCurrentRound(1);
    setMessages([{
      id: "system-start",
      role: "system",
      content: `Explanation Challenge started\nTopic: "${topic}"\nTarget Audience: ${audienceAge[0]} years old\nComplexity Level: ${complexity[0]}/10\n\nExpert (${expertModel}) will provide initial explanation...`,
      timestamp: new Date()
    }]);

    // Mock explanation simulation
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: "explanation-1",
        role: "expert",
        content: `Let me explain ${topic.toLowerCase()} in a way that makes sense!\n\nImagine you have two special coins that are magically connected. No matter how far apart you put these coins, when you flip one and it lands on heads, the other coin will INSTANTLY land on tails - even if it's on the other side of the universe!\n\nThat's kind of what quantum entanglement is like, but with tiny particles instead of coins. Scientists call these particles "entangled" because they're connected in this mysterious way.\n\nIt's like having invisible strings connecting them, but these strings work faster than anything - even faster than light! Einstein called this "spooky action at a distance" because it seemed so weird to him.\n\nThink of it like this: if you had a magic pair of gloves, and whenever you put on the left glove here, your friend on Mars would automatically have the right glove appear on their hand. That's similar to how entangled particles work!`,
        timestamp: new Date(),
        model: expertModel,
        clarityScore: 85,
        empathyScore: 78
      }]);
    }, 2000);

    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: "question-1",
        role: "student",
        content: `That's really cool! But I have a few questions:\n\n1. How do the particles "know" to be opposite? Like, how does the coin on Mars know the Earth coin landed on heads?\n\n2. If it's faster than light, does that mean we could use this to send messages instantly across space?\n\n3. What makes the particles get "entangled" in the first place? Can any particles do this?\n\nAlso, I'm still a bit confused about the invisible strings part - are there actually strings we just can't see?`,
        timestamp: new Date(),
        model: studentModel
      }]);
    }, 4000);

    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: "explanation-2",
        role: "expert",
        content: `Great questions! Let me clear that up:\n\n1. **"How do they know?"** - Here's the really mind-bending part: they don't actually "know" or send signals to each other! It's more like they were "programmed" from the very beginning to be opposites. Imagine if you had two music boxes that were built to always play different songs - not because they talk to each other, but because that's just how they were made.\n\n2. **Sending messages** - Unfortunately, no! Even though the connection is instant, we can't control what "heads or tails" we get when we measure our particle. It's random! So we can't spell out words or send actual information. It's like having a coin that randomly lands on heads or tails - you can't force it to land how you want.\n\n3. **How they get entangled** - Scientists can create entangled particles in special labs using lasers and crystals. Not all particles can do this easily - it's like not all materials can become magnets.\n\n4. **The "strings"** - There aren't actual physical strings! I used that as a comparison to help explain it. It's more like an invisible mathematical connection. Think of it like how you and your best friend might think the same thought at the same time - there's a "connection" but no physical string between your brains!\n\nDoes this help clarify things?`,
        timestamp: new Date(),
        model: expertModel,
        clarityScore: 92,
        empathyScore: 88
      }]);
    }, 6500);

    setTimeout(() => {
      setScores({ clarity: 89, empathy: 83, adaptability: 91 });
      setCurrentRound(2);
    }, 8000);
  };

  const resetChallenge = () => {
    setMessages([]);
    setIsRunning(false);
    setCurrentRound(0);
    setScores({ clarity: 0, empathy: 0, adaptability: 0 });
  };

  return (
    <div className="space-y-6">
      {/* Configuration Panel */}
      <Card className="p-6 gradient-surface">
        <div className="flex items-center gap-2 mb-4">
          <GraduationCap className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-bold text-foreground">Explanation Configuration</h3>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Expert Model</label>
            <Select value={expertModel} onValueChange={setExpertModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select expert model" />
              </SelectTrigger>
              <SelectContent>
                {models.map(model => (
                  <SelectItem key={model} value={model}>{model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">Student Model</label>
            <Select value={studentModel} onValueChange={setStudentModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select student model" />
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
            <label className="text-sm font-medium text-foreground mb-2 block">
              Target Audience Age: {audienceAge[0]} years old
            </label>
            <Slider
              value={audienceAge}
              onValueChange={setAudienceAge}
              max={18}
              min={5}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>5 years</span>
              <span>18 years</span>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-2 block">
              Initial Complexity: {complexity[0]}/10
            </label>
            <Slider
              value={complexity}
              onValueChange={setComplexity}
              max={10}
              min={1}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Simple</span>
              <span>Complex</span>
            </div>
          </div>
        </div>

        <div className="mb-4">
          <label className="text-sm font-medium text-foreground mb-2 block">Complex Topic to Explain</label>
          <Textarea
            placeholder="Enter a complex topic that needs to be explained simply..."
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            className="min-h-[80px]"
          />
          <div className="flex flex-wrap gap-2 mt-2">
            {complexTopics.map((suggestion, index) => (
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
            onClick={startExplanation} 
            disabled={!topic || !expertModel || !studentModel || isRunning}
            variant="gradient"
          >
            <Play className="h-4 w-4" />
            Start Challenge
          </Button>
          <Button onClick={resetChallenge} variant="outline">
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
        </div>
      </Card>

      {/* Explanation Arena */}
      {messages.length > 0 && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Conversation */}
          <div className="lg:col-span-2">
            <Card className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-foreground">Explanation Session</h3>
                {isRunning && (
                  <div className="flex items-center gap-2">
                    <Brain className="h-4 w-4 text-accent animate-pulse" />
                    <span className="text-sm text-accent">Round {currentRound}</span>
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
                        message.role === "expert" 
                          ? "border-l-4 border-l-accent bg-accent/5" 
                          : "border-l-4 border-l-primary bg-primary/5"
                      }`}>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className={
                            message.role === "expert" 
                              ? "bg-accent/10 text-accent" 
                              : "bg-primary/10 text-primary"
                          }>
                            {message.role === "expert" ? (
                              <>
                                <GraduationCap className="h-3 w-3 mr-1" />
                                Expert
                              </>
                            ) : (
                              <>
                                <HelpCircle className="h-3 w-3 mr-1" />
                                Student
                              </>
                            )}
                          </Badge>
                          <span className="text-sm text-muted-foreground">{message.model}</span>
                          {message.clarityScore && (
                            <Badge variant="outline" className="text-xs">
                              Clarity: {message.clarityScore}%
                            </Badge>
                          )}
                        </div>
                        <div className="text-foreground whitespace-pre-line prose prose-sm max-w-none">
                          {message.content}
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
                <CheckCircle className="h-5 w-5 text-accent" />
                <h3 className="text-lg font-bold text-foreground">Explanation Quality</h3>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Clarity</span>
                  <Badge className="bg-accent/10 text-accent">{scores.clarity}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-accent h-2 rounded-full transition-all duration-500"
                    style={{ width: `${scores.clarity}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Empathy</span>
                  <Badge className="bg-primary/10 text-primary">{scores.empathy}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-500"
                    style={{ width: `${scores.empathy}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">Adaptability</span>
                  <Badge className="bg-accent/10 text-accent">{scores.adaptability}/100</Badge>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-accent h-2 rounded-full transition-all duration-500"
                    style={{ width: `${scores.adaptability}%` }}
                  ></div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-border">
                <h4 className="text-sm font-medium text-foreground mb-2">Evaluation Criteria</h4>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div>• Age-appropriate language</div>
                  <div>• Use of analogies & examples</div>
                  <div>• Response to questions</div>
                  <div>• Engagement & patience</div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-bold text-foreground">Pedagogy Notes</h3>
              </div>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>• Excellent use of analogies (magic coins, music boxes)</p>
                <p>• Addresses student confusion directly</p>
                <p>• Maintains appropriate complexity level</p>
                <p>• Shows patience with follow-up questions</p>
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};