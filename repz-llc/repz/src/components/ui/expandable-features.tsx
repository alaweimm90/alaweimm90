import { useState } from "react";
import { ChevronDown, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface Feature {
  text: string;
  included?: boolean;
}

interface ExpandableFeaturesProps {
  features: Feature[];
  defaultExpanded?: boolean;
  previewCount?: number;
}

const FeatureItem = ({ feature }: { feature: Feature }) => (
  <div className="flex items-center gap-2 text-sm">
    <Check className={cn(
      "w-4 h-4 flex-shrink-0",
      feature.included !== false ? "text-tier-adaptive" : "text-text-muted"
    )} />
    <span className={cn(
      feature.included !== false ? "text-text-primary" : "text-text-muted line-through"
    )}>
      {feature.text}
    </span>
  </div>
);

export const ExpandableFeatures = ({ 
  features, 
  defaultExpanded = false,
  previewCount = 3
}: ExpandableFeaturesProps) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  if (features.length <= previewCount) {
    return (
      <div className="space-y-2">
        {features.map((feature, i) => (
          <FeatureItem key={i} feature={feature} />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Preview features */}
      <div className="space-y-2">
        {features.slice(0, previewCount).map((feature, i) => (
          <FeatureItem key={i} feature={feature} />
        ))}
      </div>

      {/* Expand button */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full py-3 flex items-center justify-center gap-2 
                   text-sm text-text-secondary hover:text-text-primary transition-colors"
      >
        <span>
          {expanded ? 'Show less' : `View all ${features.length} features`}
        </span>
        <ChevronDown className={cn(
          'w-4 h-4 transition-transform duration-200',
          expanded && 'rotate-180'
        )} />
      </button>

      {/* Expanded content */}
      <div className={cn(
        "overflow-hidden transition-all duration-300",
        expanded ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
      )}>
        <div className="pt-3 border-t border-white/5 space-y-2">
          {features.slice(previewCount).map((feature, i) => (
            <FeatureItem key={i + previewCount} feature={feature} />
          ))}
        </div>
      </div>
    </div>
  );
};