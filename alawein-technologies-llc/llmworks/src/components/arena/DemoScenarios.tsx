import { memo, useState, useEffect } from 'react';
import { Play, SkipForward, Pause, RotateCcw, Sparkles, Trophy, Brain, Zap, MessageSquare, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { type AIPersonality } from './AIPersonalityAvatar';

interface DemoScenario {
  id: string;
  name: string;
  description: string;
  duration: number; // seconds
  topic: string;
  category: 'technology' | 'philosophy' | 'science' | 'ethics' | 'environment';
  participants: {
    left: {
      name: string;
      personality: AIPersonality;
      strengths: string[];
    };
    right: {
      name: string;
      personality: AIPersonality;
      strengths: string[];
    };
  };
  expectedHighlights: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'epic';
}

interface DemoScenariosProps {
  onScenarioSelect: (scenario: DemoScenario) => void;
  onScenarioStart: () => void;
  onScenarioStop: () => void;
  currentScenarioId?: string;
  isPlaying: boolean;
}

const DemoScenariosComponent = ({
  onScenarioSelect,
  onScenarioStart,
  onScenarioStop,
  currentScenarioId,
  isPlaying
}: DemoScenariosProps) => {
  const [selectedScenario, setSelectedScenario] = useState<DemoScenario | null>(null);
  const [scenarioProgress, setScenarioProgress] = useState(0);

  const demoScenarios: DemoScenario[] = [
    {
      id: 'epic-clash-ai-safety',
      name: 'The Epic AI Safety Showdown',
      description: 'Two titans of logic clash over the future of artificial intelligence safety measures',
      duration: 180,
      topic: 'Should we pause advanced AI development for comprehensive safety research?',
      category: 'technology',
      participants: {
        left: {
          name: 'Logic Prime',
          personality: 'analytical',
          strengths: ['Systematic reasoning', 'Data-driven arguments', 'Risk assessment']
        },
        right: {
          name: 'Velocity Mind',
          personality: 'speed',
          strengths: ['Rapid innovation', 'Market dynamics', 'Competitive advantage']
        }
      },
      expectedHighlights: [
        'Critical-strike combo on existential risk',
        'Deflection shield against innovation slowdown',
        'Epic clash over regulation timing',
        'Devastating blow with safety statistics'
      ],
      difficulty: 'epic'
    },
    {
      id: 'creative-philosophy',
      name: 'Art vs Logic: The Consciousness Debate',
      description: 'Creative genius meets analytical precision in a battle over machine consciousness',
      duration: 150,
      topic: 'Can artificial intelligence achieve true consciousness and creativity?',
      category: 'philosophy',
      participants: {
        left: {
          name: 'Spark Mind',
          personality: 'creative',
          strengths: ['Innovative perspectives', 'Metaphorical reasoning', 'Artistic insights']
        },
        right: {
          name: 'Data Core',
          personality: 'analytical',
          strengths: ['Logical frameworks', 'Scientific method', 'Empirical evidence']
        }
      },
      expectedHighlights: [
        'Creative combo chain on emergent properties',
        'Analytical counter-cascade on definitions',
        'Environmental shift to abstract visuals',
        'Energy signature resonance at peak argument'
      ],
      difficulty: 'advanced'
    },
    {
      id: 'strategic-environmental',
      name: 'Climate Tech Strategy War',
      description: 'Strategic masterminds debate optimal paths for climate technology deployment',
      duration: 120,
      topic: 'Should we prioritize adaptation or mitigation technologies for climate change?',
      category: 'environment',
      participants: {
        left: {
          name: 'Tactical Oracle',
          personality: 'strategic',
          strengths: ['Long-term planning', 'Resource optimization', 'Risk mitigation']
        },
        right: {
          name: 'Echo Mind',
          personality: 'conversational',
          strengths: ['Human factors', 'Social dynamics', 'Community engagement']
        }
      },
      expectedHighlights: [
        'Strategic overwhelming force maneuver',
        'Conversational precision strike on equity',
        'Environmental effects showing climate data',
        'Citation tracker activating on IPCC reports'
      ],
      difficulty: 'intermediate'
    },
    {
      id: 'speed-ethics',
      name: 'The Rapid Ethics Response',
      description: 'Lightning-fast arguments collide in an ethical dilemma about automated decision-making',
      duration: 90,
      topic: 'Should AI systems make life-critical decisions without human oversight?',
      category: 'ethics',
      participants: {
        left: {
          name: 'Flash Core',
          personality: 'speed',
          strengths: ['Quick analysis', 'Real-time processing', 'Emergency response']
        },
        right: {
          name: 'Guardian Mind',
          personality: 'strategic',
          strengths: ['Ethical frameworks', 'Precautionary principles', 'Accountability']
        }
      },
      expectedHighlights: [
        'Speed combo-chain in opening',
        'Strategic deflection of efficiency arguments',
        'Fact-check activation on accident statistics',
        'Epic moment on moral responsibility'
      ],
      difficulty: 'intermediate'
    },
    {
      id: 'science-showdown',
      name: 'Quantum Computing Revolution',
      description: 'Scientific minds clash over the timeline and impact of quantum supremacy',
      duration: 150,
      topic: 'Will quantum computing revolutionize AI within the next decade?',
      category: 'science',
      participants: {
        left: {
          name: 'Quantum Oracle',
          personality: 'analytical',
          strengths: ['Technical precision', 'Mathematical proofs', 'Research citations']
        },
        right: {
          name: 'Practical Mind',
          personality: 'conversational',
          strengths: ['Real-world applications', 'Industry perspectives', 'User needs']
        }
      },
      expectedHighlights: [
        'Analytical evidence-stack on qubit stability',
        'Conversational counter on implementation costs',
        'Science-themed environmental effects',
        'Model energy signatures showing frequency clash'
      ],
      difficulty: 'advanced'
    },
    {
      id: 'beginner-friendly',
      name: 'Introduction to AI Debates',
      description: 'A gentle introduction showcasing basic debate mechanics and visual effects',
      duration: 60,
      topic: 'Should AI assistants have personality traits?',
      category: 'technology',
      participants: {
        left: {
          name: 'Friendly Bot',
          personality: 'conversational',
          strengths: ['User experience', 'Engagement', 'Accessibility']
        },
        right: {
          name: 'Logic Bot',
          personality: 'analytical',
          strengths: ['Consistency', 'Predictability', 'Reliability']
        }
      },
      expectedHighlights: [
        'Basic argument exchanges',
        'Simple combo detection',
        'Gentle environmental effects',
        'Clear energy signature differentiation'
      ],
      difficulty: 'beginner'
    }
  ];

  useEffect(() => {
    if (isPlaying && selectedScenario) {
      const interval = setInterval(() => {
        setScenarioProgress(prev => {
          if (prev >= 100) {
            onScenarioStop();
            return 0;
          }
          return prev + (100 / selectedScenario.duration);
        });
      }, 1000);

      return () => clearInterval(interval);
    } else {
      setScenarioProgress(0);
    }
  }, [isPlaying, selectedScenario, onScenarioStop]);

  const getPersonalityIcon = (personality: AIPersonality) => {
    switch (personality) {
      case 'analytical': return Brain;
      case 'creative': return Sparkles;
      case 'speed': return Zap;
      case 'conversational': return MessageSquare;
      case 'strategic': return Target;
      default: return Brain;
    }
  };

  const getDifficultyColor = (difficulty: DemoScenario['difficulty']) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-500 bg-green-500/10';
      case 'intermediate': return 'text-yellow-500 bg-yellow-500/10';
      case 'advanced': return 'text-orange-500 bg-orange-500/10';
      case 'epic': return 'text-red-500 bg-red-500/10';
    }
  };

  const handleScenarioSelect = (scenario: DemoScenario) => {
    setSelectedScenario(scenario);
    onScenarioSelect(scenario);
    setScenarioProgress(0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="heading-refined text-2xl mb-2">Epic Battle Scenarios</h2>
        <p className="text-muted-foreground">
          Experience pre-configured debates showcasing the full power of the Neural Arena
        </p>
      </div>

      {/* Scenario Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {demoScenarios.map(scenario => {
          const LeftIcon = getPersonalityIcon(scenario.participants.left.personality);
          const RightIcon = getPersonalityIcon(scenario.participants.right.personality);
          const isSelected = selectedScenario?.id === scenario.id;

          return (
            <div
              key={scenario.id}
              className={`
                glass-panel p-4 rounded-xl cursor-pointer transition-all duration-300
                ${isSelected ? 'ring-2 ring-primary shadow-lg scale-105' : 'hover:shadow-md hover:scale-102'}
              `}
              onClick={() => handleScenarioSelect(scenario)}
            >
              {/* Scenario Header */}
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-bold text-sm">{scenario.name}</h3>
                  <Badge className={`mt-1 ${getDifficultyColor(scenario.difficulty)}`}>
                    {scenario.difficulty}
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  {Math.floor(scenario.duration / 60)}:{(scenario.duration % 60).toString().padStart(2, '0')}
                </div>
              </div>

              {/* Description */}
              <p className="text-xs text-muted-foreground mb-3">
                {scenario.description}
              </p>

              {/* Participants */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-full bg-primary/10">
                    <LeftIcon className="h-4 w-4 text-primary" />
                  </div>
                  <div className="text-xs">
                    <div className="font-medium">{scenario.participants.left.name}</div>
                    <div className="text-muted-foreground">{scenario.participants.left.personality}</div>
                  </div>
                </div>
                
                <div className="text-xs font-bold text-muted-foreground">VS</div>
                
                <div className="flex items-center gap-2">
                  <div className="text-xs text-right">
                    <div className="font-medium">{scenario.participants.right.name}</div>
                    <div className="text-muted-foreground">{scenario.participants.right.personality}</div>
                  </div>
                  <div className="p-2 rounded-full bg-accent/10">
                    <RightIcon className="h-4 w-4 text-accent" />
                  </div>
                </div>
              </div>

              {/* Topic */}
              <div className="glass-minimal p-2 rounded-lg mb-3">
                <div className="text-xs text-muted-foreground mb-1">Debate Topic:</div>
                <div className="text-xs italic">"{scenario.topic}"</div>
              </div>

              {/* Expected Highlights */}
              <div className="space-y-1">
                <div className="text-xs text-muted-foreground">Epic Moments:</div>
                {scenario.expectedHighlights.slice(0, 2).map((highlight, index) => (
                  <div key={index} className="flex items-center gap-1 text-xs">
                    <Trophy className="h-3 w-3 text-yellow-500" />
                    <span className="text-muted-foreground truncate">{highlight}</span>
                  </div>
                ))}
                {scenario.expectedHighlights.length > 2 && (
                  <div className="text-xs text-muted-foreground">
                    +{scenario.expectedHighlights.length - 2} more epic moments
                  </div>
                )}
              </div>

              {/* Selection Indicator */}
              {isSelected && (
                <div className="mt-3 pt-3 border-t border-primary/20">
                  <div className="flex items-center justify-between">
                    <Badge variant="default" className="text-xs">
                      Selected
                    </Badge>
                    {scenarioProgress > 0 && (
                      <div className="text-xs text-muted-foreground">
                        {Math.round(scenarioProgress)}% complete
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Control Panel */}
      {selectedScenario && (
        <div className="glass-panel p-6 rounded-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-bold text-lg">{selectedScenario.name}</h3>
              <p className="text-sm text-muted-foreground">
                {selectedScenario.participants.left.name} vs {selectedScenario.participants.right.name}
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={isPlaying ? onScenarioStop : onScenarioStart}
                size="lg"
                variant="hero"
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-5 w-5" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    Start Demo
                  </>
                )}
              </Button>
              <Button
                onClick={() => {
                  onScenarioStop();
                  setScenarioProgress(0);
                }}
                size="lg"
                variant="outline"
              >
                <RotateCcw className="h-5 w-5" />
                Reset
              </Button>
            </div>
          </div>

          {/* Progress Bar */}
          {scenarioProgress > 0 && (
            <div className="mb-4">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                <span>Progress</span>
                <span>{Math.round(scenarioProgress)}%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
                  style={{ width: `${scenarioProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Scenario Details */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div>
                <div className="text-sm font-medium mb-2">Left Combatant Strengths</div>
                {selectedScenario.participants.left.strengths.map((strength, index) => (
                  <Badge key={index} variant="outline" className="mr-2 mb-2">
                    {strength}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <div className="text-sm font-medium mb-2">Right Combatant Strengths</div>
                {selectedScenario.participants.right.strengths.map((strength, index) => (
                  <Badge key={index} variant="outline" className="mr-2 mb-2">
                    {strength}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {/* Expected Highlights */}
          <div className="mt-4 pt-4 border-t border-primary/20">
            <div className="text-sm font-medium mb-2">What to Watch For</div>
            <div className="grid md:grid-cols-2 gap-2">
              {selectedScenario.expectedHighlights.map((highlight, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <Trophy className="h-4 w-4 text-yellow-500" />
                  <span className="text-muted-foreground">{highlight}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const DemoScenarios = memo(DemoScenariosComponent);