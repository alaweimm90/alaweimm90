import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { logger } from '@/lib/logger';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  BookOpen, 
  CheckCircle, 
  Circle, 
  Trophy, 
  Target,
  Lightbulb,
  Atom,
  Zap,
  Brain
} from 'lucide-react';

interface LearningModule {
  id: string;
  title: string;
  description: string;
  iconType: 'atom' | 'target' | 'zap' | 'book' | 'brain';
  concepts: string[];
  completed: boolean;
  progress: number;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
}

const learningModules: LearningModule[] = [
  {
    id: 'quantum-basics',
    title: 'Quantum Fundamentals',
    description: 'Learn the core principles of quantum mechanics and quantum computing.',
    iconType: 'atom',
    concepts: ['Superposition', 'Qubits', 'Measurement', 'Quantum States'],
    completed: false,
    progress: 0,
    difficulty: 'Beginner'
  },
  {
    id: 'bloch-sphere',
    title: 'Bloch Sphere Mastery',
    description: 'Master quantum state visualization and manipulation on the Bloch sphere.',
    iconType: 'target',
    concepts: ['State Vectors', 'Rotations', 'Phases', 'Probabilities'],
    completed: false,
    progress: 0,
    difficulty: 'Beginner'
  },
  {
    id: 'quantum-gates',
    title: 'Quantum Gate Operations',
    description: 'Understand how quantum gates transform qubit states.',
    iconType: 'zap',
    concepts: ['Pauli Gates', 'Hadamard', 'CNOT', 'Gate Composition'],
    completed: false,
    progress: 0,
    difficulty: 'Intermediate'
  },
  {
    id: 'circuit-building',
    title: 'Circuit Construction',
    description: 'Build and analyze quantum circuits for various algorithms.',
    iconType: 'book',
    concepts: ['Circuit Design', 'Gate Sequences', 'Multi-qubit Gates', 'Optimization'],
    completed: false,
    progress: 0,
    difficulty: 'Intermediate'
  },
  {
    id: 'quantum-ml',
    title: 'Quantum Machine Learning',
    description: 'Explore variational algorithms and quantum ML applications.',
    iconType: 'brain',
    concepts: ['VQE', 'QAOA', 'Hybrid Algorithms', 'Parameter Training'],
    completed: false,
    progress: 0,
    difficulty: 'Advanced'
  }
];

const getIconComponent = (iconType: string) => {
  switch (iconType) {
    case 'atom': return <Atom className="w-5 h-5" />;
    case 'target': return <Target className="w-5 h-5" />;
    case 'zap': return <Zap className="w-5 h-5" />;
    case 'book': return <BookOpen className="w-5 h-5" />;
    case 'brain': return <Brain className="w-5 h-5" />;
    default: return <Circle className="w-5 h-5" />;
  }
};

export const QuantumLearningTracker: React.FC = () => {
  const [modules, setModules] = useState<LearningModule[]>(learningModules);
  const [overallProgress, setOverallProgress] = useState(0);
  const [achievementUnlocked, setAchievementUnlocked] = useState<string | null>(null);

  // Load progress from localStorage
  useEffect(() => {
    const savedProgress = localStorage.getItem('qmlab-learning-progress');
    if (savedProgress) {
      try {
        const parsedProgress = JSON.parse(savedProgress);
        setModules(parsedProgress);
      } catch (error) {
        logger.warn('Failed to load learning progress', { error });
      }
    }
  }, []);

  // Save progress to localStorage
  useEffect(() => {
    localStorage.setItem('qmlab-learning-progress', JSON.stringify(modules));
    
    // Calculate overall progress
    const totalProgress = modules.reduce((sum, module) => sum + module.progress, 0);
    const avgProgress = totalProgress / modules.length;
    setOverallProgress(avgProgress);

    // Check for achievements
    const completedModules = modules.filter(m => m.completed).length;
    if (completedModules === 1 && !achievementUnlocked) {
      setAchievementUnlocked('First Steps');
      setTimeout(() => setAchievementUnlocked(null), 3000);
    } else if (completedModules === 3 && achievementUnlocked !== 'Quantum Explorer') {
      setAchievementUnlocked('Quantum Explorer');
      setTimeout(() => setAchievementUnlocked(null), 3000);
    } else if (completedModules === 5 && achievementUnlocked !== 'Quantum Master') {
      setAchievementUnlocked('Quantum Master');
      setTimeout(() => setAchievementUnlocked(null), 3000);
    }
  }, [modules, achievementUnlocked]);

  const updateModuleProgress = (moduleId: string, progress: number) => {
    setModules(prev => prev.map(module => 
      module.id === moduleId 
        ? { 
            ...module, 
            progress,
            completed: progress >= 100
          }
        : module
    ));
  };

  const resetProgress = () => {
    const resetModules = modules.map(module => ({
      ...module,
      completed: false,
      progress: 0
    }));
    setModules(resetModules);
    localStorage.removeItem('qmlab-learning-progress');
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return 'bg-green-500/20 text-green-300 border-green-400/30';
      case 'Intermediate': return 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30';
      case 'Advanced': return 'bg-red-500/20 text-red-300 border-red-400/30';
      default: return 'bg-slate-500/20 text-slate-300 border-slate-400/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Achievement Notification */}
      {achievementUnlocked && (
        <div className="fixed top-4 right-4 z-50 animate-in slide-in-from-right duration-500">
          <Card className="bg-yellow-500/10 border border-yellow-400/50 shadow-xl">
            <CardContent className="p-4 flex items-center gap-3">
              <Trophy className="w-6 h-6 text-yellow-400" />
              <div>
                <div className="font-semibold text-yellow-300">Achievement Unlocked!</div>
                <div className="text-sm text-yellow-400">{achievementUnlocked}</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Overall Progress Header */}
      <Card className="bg-slate-900/80 border border-blue-400/40">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-400/20 border border-blue-400/30">
                <Lightbulb className="w-6 h-6 text-blue-400" />
              </div>
              <div>
                <CardTitle className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
                  Quantum Learning Journey
                </CardTitle>
                <p className="text-slate-400">Master quantum computing concepts step by step</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-mono font-bold text-blue-400">
                {Math.round(overallProgress)}%
              </div>
              <div className="text-sm text-slate-400">Complete</div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Progress value={overallProgress} className="h-3" />
            <div className="flex justify-between items-center">
              <div className="text-sm text-slate-400">
                {modules.filter(m => m.completed).length} of {modules.length} modules completed
              </div>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={resetProgress}
                className="text-slate-400 hover:text-slate-200"
              >
                Reset Progress
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Learning Modules */}
      <div className="grid gap-4">
        {modules.map((module, index) => (
          <Card 
            key={module.id}
            className={`transition-all duration-300 bg-slate-900/80 border ${
              module.completed 
                ? 'border-green-400/50 hover:border-green-400/60' 
                : 'border-slate-600/50 hover:border-slate-500/60'
            }`}
          >
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-lg ${
                    module.completed 
                      ? 'bg-green-400/20 border-green-400/30' 
                      : 'bg-slate-700/50 border-slate-600/30'
                  } border`}>
                    {module.completed ? (
                      <CheckCircle className="w-6 h-6 text-green-400" />
                    ) : (
                      getIconComponent(module.iconType)
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <CardTitle className={`text-lg ${
                        module.completed ? 'text-green-300' : 'text-slate-200'
                      }`}>
                        {module.title}
                      </CardTitle>
                      <Badge className={`text-xs ${getDifficultyColor(module.difficulty)}`}>
                        {module.difficulty}
                      </Badge>
                    </div>
                    <p className="text-sm text-slate-400">{module.description}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-xl font-mono font-bold ${
                    module.completed ? 'text-green-400' : 'text-slate-300'
                  }`}>
                    {module.progress}%
                  </div>
                  <div className="text-xs text-slate-500">Progress</div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Progress 
                  value={module.progress} 
                  className={`h-2 ${module.completed ? '[&>div]:bg-green-400' : ''}`}
                />
                
                {/* Concepts to Learn */}
                <div>
                  <div className="text-sm font-medium text-slate-300 mb-2">Key Concepts:</div>
                  <div className="flex flex-wrap gap-2">
                    {module.concepts.map((concept, idx) => (
                      <Badge 
                        key={idx}
                        variant="outline" 
                        className="text-xs bg-slate-800/50 text-slate-400 border-slate-600/50"
                      >
                        {concept}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Progress Controls */}
                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => updateModuleProgress(module.id, Math.min(100, module.progress + 25))}
                    disabled={module.completed}
                    className="flex-1 text-xs"
                  >
                    {module.progress === 0 ? 'Start Learning' : 'Continue'}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => updateModuleProgress(module.id, 100)}
                    disabled={module.completed}
                    className="text-xs"
                  >
                    Mark Complete
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Learning Tips */}
      <Card className="bg-slate-900/80 border border-purple-400/40">
        <CardHeader>
          <CardTitle className="text-lg text-purple-300">💡 Learning Tips</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-slate-400">
            <li className="flex items-start gap-2">
              <Circle className="w-3 h-3 mt-1 text-purple-400" />
              <span>Start with Quantum Fundamentals to build a solid foundation</span>
            </li>
            <li className="flex items-start gap-2">
              <Circle className="w-3 h-3 mt-1 text-purple-400" />
              <span>Use the interactive Bloch Sphere to visualize quantum states</span>
            </li>
            <li className="flex items-start gap-2">
              <Circle className="w-3 h-3 mt-1 text-purple-400" />
              <span>Experiment with the Circuit Builder to understand gate operations</span>
            </li>
            <li className="flex items-start gap-2">
              <Circle className="w-3 h-3 mt-1 text-purple-400" />
              <span>Click the help icons (?) throughout the platform for concept explanations</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};