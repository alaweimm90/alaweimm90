import React from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface CollapsibleSectionProps {
  id?: string;
  title: string;
  description?: string;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
  previewTags?: Array<{ label: string; icon?: React.ReactNode; color: string }>;
  titleClassName?: string;
  chevronColor?: string;
}

export const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  id,
  title,
  description,
  isOpen,
  onOpenChange,
  children,
  previewTags,
  titleClassName = "text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400",
  chevronColor = "text-blue-400"
}) => {
  return (
    <Collapsible open={isOpen} onOpenChange={onOpenChange}>
      <section id={id} className="mb-20">
        <div className="text-center mb-12 relative">
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className={`group ${titleClassName} mb-6 tracking-tight hover:bg-transparent p-0 h-auto flex items-center gap-4`}
            >
              <h2>{title}</h2>
              {isOpen ? (
                <ChevronUp className={`w-8 h-8 ${chevronColor} group-hover:opacity-80 transition-opacity`} />
              ) : (
                <ChevronDown className={`w-8 h-8 ${chevronColor} group-hover:opacity-80 transition-opacity`} />
              )}
            </Button>
          </CollapsibleTrigger>
          
          {description && (
            <p className="text-slate-400 max-w-3xl mx-auto text-lg leading-relaxed">
              {description}
            </p>
          )}
          
          {!isOpen && previewTags && (
            <div className="mt-4 flex justify-center gap-2 flex-wrap">
              {previewTags.map((tag, index) => (
                <div
                  key={index}
                  className={`px-3 py-1 text-sm rounded-full border ${tag.color}`}
                >
                  {tag.icon && <span className="mr-1">{tag.icon}</span>}
                  {tag.label}
                </div>
              ))}
            </div>
          )}
        </div>
        
        <CollapsibleContent>
          {children}
        </CollapsibleContent>
      </section>
    </Collapsible>
  );
};