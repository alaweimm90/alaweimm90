import React from 'react';
import { Atom, Brain, Zap, BookOpen, Calculator, Waves } from 'lucide-react';

export const QuantumFooter: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const quantumConcepts = [
    {
      title: "Superposition",
      description: "Quantum bits exist in multiple states simultaneously",
      icon: <Waves className="w-5 h-5 text-blue-400" />
    },
    {
      title: "Entanglement", 
      description: "Quantum particles become correlated across space",
      icon: <Atom className="w-5 h-5 text-purple-400" />
    },
    {
      title: "Interference",
      description: "Wave-like properties enable quantum computation",
      icon: <Zap className="w-5 h-5 text-indigo-400" />
    }
  ];

  const learningTopics = [
    "Quantum Gates & Circuits",
    "Bloch Sphere Representation", 
    "Variational Quantum Eigensolvers",
    "Quantum Support Vector Machines",
    "Quantum Neural Networks",
    "Quantum Approximate Optimization"
  ];

  return (
    <footer className="relative bg-slate-900/95 border-t border-slate-700/50 mt-20">
      {/* Quantum particle effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-4 left-1/5 w-1 h-1 bg-blue-400 rounded-full animate-pulse opacity-60"></div>
        <div className="absolute top-8 right-1/4 w-1 h-1 bg-purple-400 rounded-full animate-pulse opacity-40" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-4 left-1/2 w-1 h-1 bg-indigo-400 rounded-full animate-pulse opacity-50" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="container mx-auto px-6 py-12 relative z-10">
        <div className="grid lg:grid-cols-3 gap-12">
          
          {/* QMLab Branding & Mission */}
          <div className="space-y-6">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="w-10 h-10 rounded-full border-2 border-blue-400 flex items-center justify-center bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-indigo-600/20">
                  <div className="relative">
                    <div className="absolute inset-0 w-6 h-6 border border-blue-300/50 rounded-full animate-spin" style={{ animationDuration: '4s' }}></div>
                    <div className="w-1.5 h-1.5 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full mx-auto mt-2 animate-pulse"></div>
                  </div>
                </div>
              </div>
              <div>
                <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-indigo-600 bg-clip-text text-transparent">
                  QMLab
                </h3>
                <div className="text-sm text-blue-300/80 font-medium">
                  Quantum Machine Learning
                </div>
              </div>
            </div>
            
            <p className="text-muted-foreground text-sm leading-relaxed">
              An interactive educational platform for exploring quantum machine learning concepts. 
              Learn quantum computing through hands-on experimentation with circuits, visualizations, and algorithms.
            </p>
            
            <div className="text-xs text-muted-foreground-light">
              © {currentYear} QMLab. Educational quantum computing platform.
            </div>
          </div>

          {/* Quantum Concepts */}
          <div className="space-y-6">
            <h4 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Quantum Concepts
            </h4>
            <div className="space-y-4">
              {quantumConcepts.map((concept, index) => (
                <div key={index} className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30 hover:border-slate-600/50 transition-colors">
                  <div className="flex items-start gap-3">
                    {concept.icon}
                    <div>
                      <div className="font-medium text-slate-300 text-sm">{concept.title}</div>
                      <div className="text-xs text-muted-foreground mt-1">{concept.description}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Learning Resources */}
          <div className="space-y-6">
            <h4 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-400" />
              What You'll Learn
            </h4>
            <div className="grid gap-2">
              {learningTopics.map((topic, index) => (
                <div key={index} className="flex items-center gap-2 p-2 rounded-lg hover:bg-slate-800/30 transition-colors cursor-pointer">
                  <div className="w-1.5 h-1.5 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full"></div>
                  <span className="text-sm text-slate-300">{topic}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="mt-12 pt-8 border-t border-slate-700/30">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <span>Interactive Quantum Computing Education</span>
              <span className="hidden md:block">•</span>
              <span>Browser-Based Learning Platform</span>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="px-3 py-1 rounded-full bg-green-500/20 border border-green-400/30 text-green-300 text-xs font-medium">
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  System Online
                </div>
              </div>
              <div className="px-3 py-1 rounded-full bg-blue-500/20 border border-blue-400/30 text-blue-300 text-xs font-medium">
                Educational Use
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};