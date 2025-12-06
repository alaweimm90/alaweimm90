// Enhanced accessible ModuleCard component with WCAG 2.1 AA compliance
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BookOpen, Play, Clock } from "lucide-react";
import { forwardRef } from "react";
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { categoryToThemeDomain } from '@/lib/category-domain-map';

interface ModuleCardProps {
  title: string;
  description: string;
  category: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Research';
  tags: string[];
  equation?: string;
  isImplemented?: boolean;
  onExplore?: () => void;
  onTheory?: () => void;
}

// ... keep existing code (removed local category→domain map in favor of shared utility)

// Category color schemes
const categoryColors = {
  'Band Structure': 'bg-blue-500/10 text-blue-400 border-blue-400/30',
  'Quantum Dynamics': 'bg-purple-500/10 text-purple-400 border-purple-400/30', 
  'Statistical Physics': 'bg-green-500/10 text-green-400 border-green-400/30',
  'Materials & Crystals': 'bg-amber-500/10 text-amber-400 border-amber-400/30',
  'Machine Learning': 'bg-red-500/10 text-red-400 border-red-400/30',
  'Mathematical Methods': 'bg-cyan-500/10 text-cyan-400 border-cyan-400/30',
  'Field Theory': 'bg-pink-500/10 text-pink-400 border-pink-400/30',
  'Spin & Magnetism': 'bg-orange-500/10 text-orange-400 border-orange-400/30'
} as const;

const difficultyColors = {
  'Beginner': 'bg-accentStatistical/10 text-accentStatistical border-accentStatistical/20',
  'Intermediate': 'bg-accentQuantum/10 text-accentQuantum border-accentQuantum/20',
  'Advanced': 'bg-accentEnergy/10 text-accentEnergy border-accentEnergy/20',
  'Research': 'bg-accentField/10 text-accentField border-accentField/20',
};

const difficultyDescriptions = {
  'Beginner': 'Suitable for newcomers to scientific simulations',
  'Intermediate': 'Requires basic physics and mathematics background', 
  'Advanced': 'Designed for advanced students and researchers',
  'Research': 'Cutting-edge research tools for specialists',
};

export const ModuleCard = forwardRef<HTMLDivElement, ModuleCardProps>(({
  title,
  description,
  category,
  difficulty,
  tags,
  equation,
  isImplemented = false,
  onExplore,
  onTheory,
}, ref) => {
  const cardId = `module-${title.toLowerCase().replace(/\s+/g, '-')}`;
  const descriptionId = `${cardId}-description`;
  const statusId = `${cardId}-status`;
  
  // Get domain for theming
  const domain = categoryToThemeDomain(category);
  
  // Domain colors are applied via CSS utilities in index.css using [data-domain]
  return (
    <Card 
      ref={ref}
      data-domain={domain}
      className={`w-full rounded-2xl group bg-surface/90 backdrop-blur-xl border border-muted/30 shadow-elegant transition-all duration-700 min-h-[280px] sm:min-h-[320px] touch-manipulation ${
        isImplemented 
          ? 'cursor-pointer hover:shadow-glow hover:border-interactive/40 hover:scale-[1.02] hover:-translate-y-1' 
          : 'cursor-default opacity-60'
      }`}
      role="button"
      tabIndex={isImplemented ? 0 : -1}
      aria-labelledby={cardId}
      aria-describedby={`${descriptionId} ${statusId}`}
      onClick={isImplemented ? onExplore : undefined}
      onKeyDown={(e) => {
        if (isImplemented && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onExplore?.();
        }
      }}
      aria-label={isImplemented ? `Open ${title} interactive simulation` : `${title} - Coming soon`}
    >
      {/* Enhanced glow overlay */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-30 transition-opacity duration-700 bg-gradient-to-b from-primary/0 to-primary/10"
      />
      
      {/* Status indicator with domain theming */}
        {isImplemented && (
          <div className="absolute top-4 right-4 flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full animate-pulse domain-bg"
              aria-hidden="true"
            />
            <span className="text-xs font-medium domain-fg">Ready</span>
          </div>
        )}
      
      <div className="relative p-6 h-full flex flex-col">
        {/* Enhanced header */}
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <h3 
              id={cardId}
              className="text-lg font-bold text-foreground"
            >
              {title}
            </h3>
            <div className="flex items-center gap-2">
              <Badge 
                variant="secondary" 
                className="text-xs domain-border domain-bg-subtle"
                role="img"
                aria-label={`Category: ${category}`}
              >
                {category}
              </Badge>
              <Badge 
                variant="outline" 
                className="text-xs domain-border domain-bg-subtle"
                role="img"
                aria-label={`Difficulty level: ${difficulty}`}
                title={difficultyDescriptions[difficulty]}
              >
                {difficulty}
              </Badge>
            </div>
          </div>
        </div>

        {/* Enhanced description */}
        <p 
          id={descriptionId}
          className="text-slate-300 text-sm leading-relaxed mb-4 flex-1"
        >
          {description}
        </p>

        {/* Enhanced equation with domain theming */}
        {equation && (
          <div 
            className="rounded-lg p-4 border overflow-x-auto overflow-y-visible mb-4 domain-bg-subtle domain-border"
            role="img"
            aria-label={`Mathematical equation: ${equation}`}
          >
            <div className="text-center">
              <BlockMath math={equation} errorColor={'#cc0000'} throwOnError={false} />
            </div>
          </div>
        )}

        {/* Enhanced tags with domain colors */}
        <div className="flex flex-wrap gap-2 mb-4" role="list" aria-label="Module tags">
          {tags.map((tag, index) => (
            <Badge 
              key={index} 
              variant="outline" 
              className="text-xs domain-border domain-bg-subtle"
              role="listitem"
            >
              {tag}
            </Badge>
          ))}
        </div>

        {/* Status information for screen readers */}
        <div id={statusId} className="sr-only">
          {isImplemented 
            ? "Click anywhere on this card to open the interactive simulation."
            : "This module is coming soon and is not yet available for exploration."
          }
        </div>

        {/* Action buttons */}
        <div className="mt-auto flex items-center justify-between gap-2">
          <Button
            variant="default"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              if (isImplemented) onExplore?.();
            }}
            aria-label={`Launch ${title}`}
            className="transition-all duration-300"
            disabled={!isImplemented}
          >
            <Play className="w-4 h-4 mr-2" />
            Launch
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onTheory?.();
            }}
            aria-label={`View documentation for ${title}`}
            className="transition-all duration-300"
          >
            <BookOpen className="w-4 h-4 mr-2" />
            Documentation
          </Button>
        </div>
      </div>
    </Card>
  );
})

ModuleCard.displayName = "ModuleCard";