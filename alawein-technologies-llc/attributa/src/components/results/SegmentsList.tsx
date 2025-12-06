import { Segment } from "@/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { FileText, Code, BookOpen } from "lucide-react";

interface SegmentsListProps {
  sections: Segment[];
  scores?: Record<string, { score: number }>;
  onJump: (id: string) => void;
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

export default function SegmentsList({ sections, scores, onJump }: SegmentsListProps) {
  return (
    <div className="space-y-2 max-h-96 overflow-y-auto">
      {sections.map((seg) => {
        const sc = scores?.[seg.segmentId];
        return (
          <Button
            key={seg.segmentId}
            variant="outline"
            className={cn(
              "w-full justify-start gap-2",
              sc?.score !== undefined && sc.score >= 0.7 && "border-ai-high/30 text-ai-high",
              sc?.score !== undefined && sc.score >= 0.4 && sc.score < 0.7 && "border-ai-medium/30 text-ai-medium",
              sc?.score !== undefined && sc.score < 0.4 && "border-ai-low/30 text-ai-low"
            )}
            onClick={() => onJump(seg.segmentId)}
            aria-label={`Jump to ${seg.title || 'Untitled'}`}
          >
            {getTypeIcon(seg.type)}
            <span className="truncate">{seg.title || "Untitled"}</span>
            <Badge variant="outline" className="ml-auto capitalize">{seg.type}</Badge>
          </Button>
        );
      })}
    </div>
  );
}
