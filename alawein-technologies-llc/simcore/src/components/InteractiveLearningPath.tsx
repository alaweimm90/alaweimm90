import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  PlayCircle, 
  PauseCircle, 
  RotateCcw, 
  CheckCircle, 
  Book, 
  Calculator, 
  Eye,
  ArrowRight,
  Star,
  Trophy
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface LearningStep {
  id: string;
  title: string;
  description: string;
  theory: string;
  simulation: string;
  exercise: string;
  completed: boolean;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface LearningPath {
  id: string;
  title: string;
  description: string;
  steps: LearningStep[];
  estimatedTime: string;
  prerequisites: string[];
}

const learningPaths: LearningPath[] = [
  {
    id: 'quantum-basics',
    title: 'Quantum Mechanics Fundamentals',
    description: 'Master the foundations of quantum mechanics through interactive simulations',
    estimatedTime: '2-3 hours',
    prerequisites: ['Basic calculus', 'Linear algebra'],
    steps: [
      {
        id: 'wave-particle',
        title: 'Wave-Particle Duality',
        description: 'Understand how matter exhibits both wave and particle properties',
        theory: 'Particles at quantum scales exhibit wave-like behavior. The de Broglie wavelength λ = h/p relates momentum to wavelength.',
        simulation: 'Observe electron diffraction patterns and see how particles create interference',
        exercise: 'Calculate the de Broglie wavelength for various particles',
        completed: false,
        difficulty: 'beginner'
      },
      {
        id: 'uncertainty',
        title: 'Heisenberg Uncertainty Principle',
        description: 'Explore the fundamental limits of simultaneous measurement',
        theory: 'ΔxΔp ≥ ℏ/2 - The more precisely we know position, the less precisely we know momentum',
        simulation: 'Interactive visualization of position-momentum uncertainty',
        exercise: 'Apply uncertainty relations to real quantum systems',
        completed: false,
        difficulty: 'intermediate'
      },
      {
        id: 'schrodinger',
        title: 'Schrödinger Equation',
        description: 'Learn the fundamental equation governing quantum systems',
        theory: 'iℏ∂ψ/∂t = Ĥψ - Time evolution of quantum states',
        simulation: 'Watch wavefunctions evolve in various potentials',
        exercise: 'Solve the equation for simple systems',
        completed: false,
        difficulty: 'advanced'
      }
    ]
  },
  {
    id: 'solid-state',
    title: 'Solid State Physics',
    description: 'Explore the physics of crystalline materials and electronic properties',
    estimatedTime: '3-4 hours',
    prerequisites: ['Quantum mechanics basics', 'Statistical mechanics'],
    steps: [
      {
        id: 'crystal-structure',
        title: 'Crystal Lattices',
        description: 'Understanding periodic arrangements of atoms',
        theory: 'Crystals have repeating unit cells that define their structure and properties',
        simulation: 'Build and visualize different crystal structures',
        exercise: 'Calculate lattice parameters and coordination numbers',
        completed: false,
        difficulty: 'beginner'
      },
      {
        id: 'band-theory',
        title: 'Electronic Band Structure',
        description: 'How electrons behave in periodic potentials',
        theory: 'Energy bands form from atomic orbital overlap in solids',
        simulation: 'Explore band diagrams and Fermi surfaces',
        exercise: 'Predict conductor, semiconductor, or insulator behavior',
        completed: false,
        difficulty: 'intermediate'
      },
      {
        id: 'graphene',
        title: 'Graphene Electronic Properties',
        description: 'Special case of 2D material with unique properties',
        theory: 'Linear dispersion relation creates massless Dirac fermions',
        simulation: 'Visualize graphene band structure and density of states',
        exercise: 'Calculate transport properties',
        completed: false,
        difficulty: 'advanced'
      }
    ]
  }
];

export function InteractiveLearningPath() {
  const [selectedPath, setSelectedPath] = useState<LearningPath>(learningPaths[0]);
  const [currentStep, setCurrentStep] = useState(0);
  const [showSimulation, setShowSimulation] = useState(false);
  const { toast } = useToast();

  const completedSteps = selectedPath.steps.filter(step => step.completed).length;
  const progress = (completedSteps / selectedPath.steps.length) * 100;

  const handleStepComplete = (stepId: string) => {
    const updatedSteps = selectedPath.steps.map(step =>
      step.id === stepId ? { ...step, completed: true } : step
    );
    setSelectedPath({ ...selectedPath, steps: updatedSteps });
    
    toast({
      title: "Step Completed! 🎉",
      description: "Great job! Move to the next step when ready.",
    });
  };

  const handleNextStep = () => {
    if (currentStep < selectedPath.steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-500';
      case 'intermediate': return 'bg-accent';
      case 'advanced': return 'bg-destructive';
      default: return 'bg-muted';
    }
  };

  const current = selectedPath.steps[currentStep];

  return (
    <div className="space-y-6">
      {/* Path Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Book className="h-5 w-5" />
            Choose Your Learning Path
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {learningPaths.map((path) => (
              <Card 
                key={path.id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedPath.id === path.id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => {
                  setSelectedPath(path);
                  setCurrentStep(0);
                }}
              >
                <CardContent className="p-4">
                  <h3 className="font-semibold mb-2">{path.title}</h3>
                  <p className="text-sm text-muted-foreground mb-3">{path.description}</p>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">⏱️ {path.estimatedTime}</span>
                    <Badge variant="outline">{path.steps.length} steps</Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Progress Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{selectedPath.title}</span>
            <div className="flex items-center gap-2">
              <Trophy className="h-4 w-4 text-yellow-500" />
              <span className="text-sm">{completedSteps}/{selectedPath.steps.length}</span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={progress} className="mb-4" />
          <div className="flex flex-wrap gap-2">
            {selectedPath.steps.map((step, index) => (
              <Button
                key={step.id}
                size="sm"
                variant={index === currentStep ? "default" : step.completed ? "secondary" : "outline"}
                onClick={() => setCurrentStep(index)}
                className="flex items-center gap-1"
              >
                {step.completed && <CheckCircle className="h-3 w-3" />}
                {index + 1}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Current Step Content */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span>Step {currentStep + 1}: {current.title}</span>
              <Badge 
                className={getDifficultyColor(current.difficulty)}
                variant="secondary"
              >
                {current.difficulty}
              </Badge>
            </div>
            {current.completed && <CheckCircle className="h-5 w-5 text-green-500" />}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground mb-6">{current.description}</p>

          <Tabs defaultValue="theory" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="theory" className="flex items-center gap-2">
                <Book className="h-4 w-4" />
                Theory
              </TabsTrigger>
              <TabsTrigger value="simulation" className="flex items-center gap-2">
                <Eye className="h-4 w-4" />
                Simulation
              </TabsTrigger>
              <TabsTrigger value="exercise" className="flex items-center gap-2">
                <Calculator className="h-4 w-4" />
                Exercise
              </TabsTrigger>
            </TabsList>

            <TabsContent value="theory" className="mt-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="prose prose-sm max-w-none">
                    <p>{current.theory}</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="simulation" className="mt-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">{current.simulation}</p>
                    
                    {/* Simulation Placeholder */}
                    <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-lg p-8 text-center">
                      <div className="animate-pulse space-y-4">
                        <div className="h-32 bg-gradient-to-r from-blue-200 to-purple-200 dark:from-blue-800 to-purple-800 rounded-lg opacity-50"></div>
                        <p className="text-sm text-muted-foreground">Interactive simulation loading...</p>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <Button size="sm" variant="outline">
                        <PlayCircle className="h-4 w-4 mr-2" />
                        Run Simulation
                      </Button>
                      <Button size="sm" variant="outline">
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Reset
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="exercise" className="mt-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <p className="text-sm">{current.exercise}</p>
                    
                    {/* Exercise Interface */}
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="space-y-3">
                        <div className="text-sm font-medium">Practice Problem:</div>
                        <div className="bg-background rounded border p-3 text-sm">
                          Calculate the de Broglie wavelength of an electron moving at 10% the speed of light.
                        </div>
                        <div className="flex gap-2">
                          <Button size="sm">Check Answer</Button>
                          <Button size="sm" variant="outline">Show Hint</Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Step Actions */}
          <div className="flex justify-between items-center mt-6 pt-6 border-t">
            <Button
              variant="outline"
              disabled={currentStep === 0}
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            >
              Previous Step
            </Button>

            <div className="flex gap-3">
              {!current.completed && (
                <Button
                  onClick={() => handleStepComplete(current.id)}
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Mark Complete
                </Button>
              )}
              
              <Button
                disabled={currentStep === selectedPath.steps.length - 1}
                onClick={handleNextStep}
              >
                Next Step
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}