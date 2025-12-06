import { Segment } from "@/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { FileText, Code, BookOpen } from "lucide-react";

interface RiskSectionsProps {
  sections: Segment[];
  scores?: Record<string, { score: number; confidence?: string }>;
}

const getTypeIcon = (type: Segment["type"]) => {
  switch (type) {
    case "code":
      return <Code className="h-4 w-4" />;
    case "latex":
      return <BookOpen className="h-4 w-4" />;
    default:
      return <FileText className="h-4 w-4" />;
  }
};

const riskClasses = (score?: number) => {
  if (score === undefined) return "border-border";
  if (score >= 0.7) return "border-ai-high/30 bg-ai-high/5";
  if (score >= 0.4) return "border-ai-medium/30 bg-ai-medium/5";
  return "border-ai-low/30 bg-ai-low/5";
};

const confidenceBadge = (conf?: string) => {
  if (!conf) return <Badge variant="outline">Analyzing…</Badge>;
  const cls =
    conf === "High"
      ? "bg-ai-high/10 text-ai-high border-ai-high/20"
      : conf === "Medium"
      ? "bg-ai-medium/10 text-ai-medium border-ai-medium/20"
      : "bg-ai-low/10 text-ai-low border-ai-low/20";
  return <Badge className={cls}>{conf} Confidence</Badge>;
};

export default function RiskSections({ sections, scores }: RiskSectionsProps) {
  return (
    <>
      {sections.map((seg) => {
        const score = scores?.[seg.segmentId]?.score;
        const conf = scores?.[seg.segmentId]?.confidence;
        const content = (seg as Segment & { content?: string }).content || seg.preview;
        return (
          <section
            key={seg.segmentId}
            id={`seg-${seg.segmentId}`}
            className={cn("rounded-lg border p-4 scroll-mt-20", riskClasses(score))}
          >
            <div className="flex items-center justify-between gap-2 mb-2">
              <div className="flex items-center gap-2">
                {getTypeIcon(seg.type)}
                <h3 className="font-medium">{seg.title || "Untitled"}</h3>
                <Badge variant="outline" className="capitalize">
                  {seg.type}
                </Badge>
              </div>
              {confidenceBadge(conf)}
            </div>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
              {content}
            </p>
          </section>
        );
      })}
    </>
  );
}
