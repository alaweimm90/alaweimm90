/**
 * Enhanced Module Card with Domain Theme Integration
 * Demonstrates full token system usage
 */

import React from 'react';
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useDomainTheme } from "@/components/DomainThemeProvider";
import { Zap, BookOpen, Play, Clock, Atom } from "lucide-react";
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { categoryToThemeDomain } from '@/lib/category-domain-map';

interface EnhancedModuleCardProps {
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

export function EnhancedModuleCard({
  title,
  description,
  category,
  difficulty,
  tags,
  equation,
  isImplemented = false,
  onExplore,
  onTheory,
}: EnhancedModuleCardProps) {
  const { setDomain } = useDomainTheme();
  const domain = categoryToThemeDomain(category);

  const handleCardClick = () => {
    if (isImplemented) {
      // Switch to module's domain theme
      setDomain(domain);
      onExplore?.();
    }
  };

  return (
    <Card 
      className="group relative transition-all duration-500 hover:scale-105 cursor-pointer"
      variant="physics"
      domain={domain}
      data-domain={domain}
      onClick={handleCardClick}
    >
      {/* Domain-specific background pattern */}
      <div className="absolute inset-0 opacity-5">
        {domain === 'quantum' && (
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 20% 20%, hsl(var(--semantic-domain-quantum)) 1px, transparent 1px),
                             radial-gradient(circle at 80% 80%, hsl(var(--semantic-domain-quantum)) 1px, transparent 1px)`,
            backgroundSize: '30px 30px'
          }} />
        )}
        {domain === 'statistical' && (
          <div className="absolute inset-0" style={{
            backgroundImage: `linear-gradient(45deg, hsl(var(--semantic-domain-statistical))/0.1 25%, transparent 25%),
                             linear-gradient(-45deg, hsl(var(--semantic-domain-statistical))/0.1 25%, transparent 25%)`,
            backgroundSize: '20px 20px'
          }} />
        )}
      </div>

      {/* Enhanced status indicator */}
      {isImplemented && (
        <div className="absolute top-4 right-4 flex items-center gap-2">
          <div 
            className="w-3 h-3 rounded-full animate-pulse"
            style={{ backgroundColor: `hsl(var(--semantic-domain-${domain}))` }}
          />
          <span className="text-xs font-medium" style={{ color: `hsl(var(--semantic-domain-${domain}))` }}>
            Ready
          </span>
        </div>
      )}

      <div className="relative z-10 p-6 space-y-4">
        {/* Header with domain-aware styling */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Badge variant="secondary" className="text-xs">
              {category}
            </Badge>
            <Badge 
              variant="outline" 
              className="text-xs"
              style={{ 
                borderColor: `hsl(var(--semantic-domain-${domain}))`,
                color: `hsl(var(--semantic-domain-${domain}))`
              }}
            >
              {difficulty}
            </Badge>
          </div>
          
          <div className="flex items-center gap-3">
            <div 
              className="p-2 rounded-lg"
              style={{ backgroundColor: `hsl(var(--semantic-domain-${domain})/0.1)` }}
            >
              <Atom className="w-5 h-5" style={{ color: `hsl(var(--semantic-domain-${domain}))` }} />
            </div>
            <h3 className="text-xl font-semibold text-foreground group-hover:text-accent transition-colors">
              {title}
            </h3>
          </div>
        </div>

        {/* Description */}
        <p className="text-muted-foreground text-sm leading-relaxed">
          {description}
        </p>

        {/* Enhanced equation with domain theming */}
        {equation && (
          <div 
            className="rounded-lg p-4 border overflow-x-auto overflow-y-visible"
            style={{ 
              backgroundColor: `hsl(var(--semantic-domain-${domain})/0.05)`,
              borderColor: `hsl(var(--semantic-domain-${domain})/0.2)`
            }}
          >
            <div className="text-center">
              <BlockMath math={equation} errorColor={'#cc0000'} throwOnError={false} />
            </div>
          </div>
        )}

        {/* Tags with domain colors */}
        <div className="flex flex-wrap gap-2">
          {tags.map((tag, index) => (
            <Badge 
              key={index} 
              variant="outline" 
              className="text-xs"
              style={{ 
                borderColor: `hsl(var(--semantic-domain-${domain})/0.3)`,
                backgroundColor: `hsl(var(--semantic-domain-${domain})/0.05)`
              }}
            >
              {tag}
            </Badge>
          ))}
        </div>

        {/* Action buttons with domain theming */}
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center gap-2">
            {isImplemented ? (
              <>
                <Play 
                  className="w-4 h-4" 
                  style={{ color: `hsl(var(--semantic-domain-${domain}))` }}
                />
                <span 
                  className="text-sm font-medium"
                  style={{ color: `hsl(var(--semantic-domain-${domain}))` }}
                >
                  Click to Explore
                </span>
              </>
            ) : (
              <>
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Coming Soon</span>
              </>
            )}
          </div>
          
          <Button
            variant={domain as any}
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onTheory?.();
            }}
            className="opacity-70 hover:opacity-100"
          >
            <BookOpen className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Enhanced hover effect */}
      <div 
        className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500 pointer-events-none"
        style={{ 
          background: `radial-gradient(circle at center, hsl(var(--semantic-domain-${domain})) 0%, transparent 70%)`
        }}
      />
    </Card>
  );
}