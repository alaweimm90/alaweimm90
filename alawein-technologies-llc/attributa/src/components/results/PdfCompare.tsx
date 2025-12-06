import { useEffect, useMemo, useState } from "react";
import { diffWords } from "diff";
import { extractTextFromPDF } from "@/lib/pdfExtractor";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface PdfEntry { name: string; url: string }
interface PdfCompareProps { pdfs: PdfEntry[] }

// Utility to fetch a URL into a File for our extractor
async function fileFromUrl(url: string, name: string): Promise<File> {
  const res = await fetch(url);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type || "application/pdf" });
}

export default function PdfCompare({ pdfs }: PdfCompareProps) {
  const [leftIndex, setLeftIndex] = useState(0);
  const [rightIndex, setRightIndex] = useState(Math.min(1, Math.max(0, pdfs.length - 1)));

  const [leftText, setLeftText] = useState<string>("");
  const [rightText, setRightText] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const left = pdfs[leftIndex];
  const right = pdfs[rightIndex];

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      try {
        setLoading(true);
        setError(null);
        const [lf, rf] = await Promise.all([
          fileFromUrl(left.url, left.name),
          fileFromUrl(right.url, right.name),
        ]);
        const [lt, rt] = await Promise.all([
          extractTextFromPDF(lf),
          extractTextFromPDF(rf),
        ]);
        if (!cancelled) {
          setLeftText(lt);
          setRightText(rt);
        }
      } catch (e) {
        console.error("Compare extraction failed", e);
        if (!cancelled) setError("Failed to extract text for comparison.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    if (left && right) run();
    return () => {
      cancelled = true;
    };
  }, [leftIndex, rightIndex, left?.url, right?.url]);

  // Build two synchronized render streams
  const diff = useMemo(() => diffWords(leftText, rightText), [leftText, rightText]);

  const renderLeft = () => (
    <div className="leading-7 text-sm whitespace-pre-wrap">
      {diff.map((part, i) => {
        if (part.added) return null; // don't show additions on left
        if (part.removed) {
          return (
            <mark
              key={i}
              style={{
                backgroundColor: `hsl(var(--destructive) / 0.15)`,
                color: `hsl(var(--destructive))`,
              }}
              className="rounded px-0.5"
            >
              {part.value}
            </mark>
          );
        }
        return <span key={i}>{part.value}</span>;
      })}
    </div>
  );

  const renderRight = () => (
    <div className="leading-7 text-sm whitespace-pre-wrap">
      {diff.map((part, i) => {
        if (part.removed) return null; // don't show removals on right
        if (part.added) {
          return (
            <mark
              key={i}
              style={{
                backgroundColor: `hsl(var(--warning) / 0.35)`,
                color: `hsl(var(--foreground))`,
              }}
              className="rounded px-0.5"
            >
              {part.value}
            </mark>
          );
        }
        return <span key={i}>{part.value}</span>;
      })}
    </div>
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Left:</span>
          <select
            value={leftIndex}
            onChange={(e) => setLeftIndex(parseInt(e.target.value))}
            className="border rounded px-2 py-1 bg-background"
            aria-label="Select left PDF"
          >
            {pdfs.map((p, i) => (
              <option key={`l-${i}`} value={i}>{p.name}</option>
            ))}
          </select>
          <span>Right:</span>
          <select
            value={rightIndex}
            onChange={(e) => setRightIndex(parseInt(e.target.value))}
            className="border rounded px-2 py-1 bg-background"
            aria-label="Select right PDF"
          >
            {pdfs.map((p, i) => (
              <option key={`r-${i}`} value={i}>{p.name}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span>
            <span
              className="inline-block align-middle rounded px-1 mr-1"
              style={{ backgroundColor: `hsl(var(--destructive) / 0.15)`, color: `hsl(var(--destructive))` }}
            >
              —
            </span>
            Deletions (left)
          </span>
          <span>
            <span
              className="inline-block align-middle rounded px-1 mr-1"
              style={{ backgroundColor: `hsl(var(--warning) / 0.35)`, color: `hsl(var(--foreground))` }}
            >
              +
            </span>
            Additions (right)
          </span>
        </div>
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <div className="grid md:grid-cols-2 gap-4">
        <div className="border rounded p-3 bg-muted/20 min-h-[240px] overflow-auto">
          {loading ? <div className="text-sm text-muted-foreground">Extracting left text…</div> : renderLeft()}
        </div>
        <div className="border rounded p-3 bg-muted/20 min-h-[240px] overflow-auto">
          {loading ? <div className="text-sm text-muted-foreground">Extracting right text…</div> : renderRight()}
        </div>
      </div>

      <div className="text-xs text-muted-foreground">Text-based comparison is shown. Layout may differ from the PDF rendering.</div>
    </div>
  );
}
