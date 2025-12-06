import React, { useState } from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, ChevronUp, Calculator, BookOpen, Brain } from 'lucide-react';

interface MathStep {
  equation: string;
  explanation: string;
  note?: string;
}

interface MathematicalDerivationProps {
  title: string;
  concept: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  introduction: string;
  steps: MathStep[];
  conclusion: string;
  applications?: string[];
  className?: string;
}

export const MathematicalDerivation: React.FC<MathematicalDerivationProps> = ({
  title,
  concept,
  difficulty,
  introduction,
  steps,
  conclusion,
  applications,
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const getDifficultyConfig = () => {
    switch (difficulty) {
      case 'beginner':
        return {
          color: 'bg-green-500/20 text-green-300 border-green-400/30',
          icon: <BookOpen className="w-3 h-3" />,
          emoji: '🟢'
        };
      case 'intermediate':
        return {
          color: 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30',
          icon: <Calculator className="w-3 h-3" />,
          emoji: '🟡'
        };
      case 'advanced':
        return {
          color: 'bg-red-500/20 text-red-300 border-red-400/30',
          icon: <Brain className="w-3 h-3" />,
          emoji: '🔴'
        };
    }
  };

  const difficultyConfig = getDifficultyConfig();

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className={`border border-slate-600/40 rounded-lg bg-slate-800/50 ${className}`}>
        <CollapsibleTrigger asChild>
          <Button
            variant="ghost"
            className="w-full p-4 justify-between text-left hover:bg-slate-700/50 rounded-lg"
          >
            <div className="flex items-center gap-3 flex-1">
              <div className="flex items-center gap-2">
                <Calculator className="w-4 h-4 text-blue-400" />
                <h4 className="font-semibold text-slate-200">{title}</h4>
              </div>
              <Badge className={`text-xs ${difficultyConfig.color}`}>
                {difficultyConfig.icon}
                <span className="ml-1 capitalize">{difficulty}</span>
              </Badge>
            </div>
            {isOpen ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </Button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-4 pb-4 space-y-4">
            {/* Introduction */}
            <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/30">
              <h5 className="font-medium text-slate-300 mb-2 flex items-center gap-2">
                <span className="text-lg">{difficultyConfig.emoji}</span>
                {concept}
              </h5>
              <p className="text-sm text-slate-400 leading-relaxed">{introduction}</p>
            </div>

            {/* Mathematical Steps */}
            <div className="space-y-3">
              <h5 className="font-medium text-slate-300 text-sm">Mathematical Derivation:</h5>
              {steps.map((step, index) => (
                <div key={index} className="p-3 rounded-lg bg-slate-900/30 border border-slate-700/20">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-blue-500/20 border border-blue-400/30 flex items-center justify-center flex-shrink-0 mt-1">
                      <span className="text-xs font-mono text-blue-300">{index + 1}</span>
                    </div>
                    <div className="flex-1">
                      {/* Equation */}
                      <div className="font-mono text-sm bg-slate-800 p-2 rounded border border-slate-600/30 mb-2 text-slate-200 overflow-x-auto">
                        {step.equation}
                      </div>
                      {/* Explanation */}
                      <p className="text-xs text-slate-400 leading-relaxed">{step.explanation}</p>
                      {/* Optional note */}
                      {step.note && (
                        <div className="mt-2 p-2 rounded bg-blue-500/10 border border-blue-400/20">
                          <p className="text-xs text-blue-300">💡 {step.note}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Conclusion */}
            <div className="p-3 rounded-lg bg-green-500/10 border border-green-400/30">
              <h5 className="font-medium text-green-300 mb-2">Conclusion</h5>
              <p className="text-sm text-slate-300 leading-relaxed">{conclusion}</p>
            </div>

            {/* Applications */}
            {applications && applications.length > 0 && (
              <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-400/30">
                <h5 className="font-medium text-purple-300 mb-2">Applications</h5>
                <ul className="space-y-1">
                  {applications.map((app, index) => (
                    <li key={index} className="text-sm text-slate-300 flex items-start gap-2">
                      <span className="text-purple-400 mt-1">•</span>
                      <span>{app}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
};