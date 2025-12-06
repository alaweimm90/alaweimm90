import { useMemo, useState } from "react";
import { Report } from "@/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";
import SegmentsList from "./SegmentsList";
import PdfViewer from "./PdfViewer";
import RiskSections from "./RiskSections";
import PdfCompare from "./PdfCompare";

interface DocumentViewerProps {
  report: Report;
  pdfs?: { name: string; url: string }[];
}

export default function DocumentViewer({ report, pdfs }: DocumentViewerProps) {
  const sections = useMemo(() => report.segments, [report.segments]);
  const hasPdfs = !!pdfs && pdfs.length > 0;
  const canCompare = !!pdfs && pdfs.length >= 2;
  const [mode, setMode] = useState<"single" | "compare">("single");

  const onJump = (id: string) => {
    const el = document.getElementById(`seg-${id}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* Left: TOC / segments list */}
      <Card>
        <CardHeader>
          <CardTitle>Document View</CardTitle>
          <CardDescription>Click a segment to jump and highlight</CardDescription>
        </CardHeader>
        <CardContent>
          <SegmentsList sections={sections} scores={report.scores as Record<string, { score: number; confidence?: string }>} onJump={onJump} />
        </CardContent>
      </Card>

      {/* Right: content with risk highlighting or PDF */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {hasPdfs ? (
              <>{mode === "compare" ? "Compare PDFs" : "PDF Viewer"}</>
            ) : (
              <>
                <AlertTriangle className="h-4 w-4 text-warning" />
                Attribution highlights
              </>
            )}
          </CardTitle>
          <CardDescription>
            {hasPdfs ? (mode === "compare" ? "Side-by-side text diff with highlights." : "Use the viewer controls below.") : "Showing captured text previews."}
          </CardDescription>
          {canCompare && (
            <div className="mt-2">
              <div className="inline-flex rounded-md border p-1 bg-muted/30">
                <Button size="sm" variant={mode === "single" ? "default" : "ghost"} onClick={() => setMode("single")}>Single</Button>
                <Button size="sm" variant={mode === "compare" ? "default" : "ghost"} onClick={() => setMode("compare")}>Compare</Button>
              </div>
            </div>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          {hasPdfs ? (
            mode === "compare" && canCompare ? (
              <PdfCompare pdfs={pdfs!} />
            ) : (
              <PdfViewer pdfs={pdfs!} />
            )
          ) : (
            <RiskSections sections={sections} scores={report.scores as Record<string, { score: number; confidence?: string }>} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
