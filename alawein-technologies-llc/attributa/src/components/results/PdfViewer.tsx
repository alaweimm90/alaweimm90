import { useState, useEffect, useRef } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";

// Use Vite static asset import for worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

interface PdfViewerProps {
  pdfs: { name: string; url: string }[];
}

export default function PdfViewer({ pdfs }: PdfViewerProps) {
  const { toast } = useToast();
  // Maintain a local, mutable list so users can re-upload without leaving the page
  const [localPdfs, setLocalPdfs] = useState<{ name: string; url: string }[]>(pdfs);
  const [createdUrls, setCreatedUrls] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [pdfIndex, setPdfIndex] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [version, setVersion] = useState(0); // force rerender on retry

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState<number>(800);

  // Sync when parent pdf list changes
  useEffect(() => setLocalPdfs(pdfs), [pdfs]);

  // Resize observer to keep page width responsive
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        setContainerWidth(Math.max(320, Math.min(1000, Math.floor(w))));
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);
  const onSelectFile = (idx: number) => {
    setPdfIndex(idx);
    setPageNumber(1);
    setPdfError(null);
    const name = localPdfs[idx]?.name;
    toast({ title: 'Rendering PDF', description: name || 'PDF' });
  };

  const retry = () => {
    setPdfError(null);
    setVersion(v => v + 1);
    const name = localPdfs[pdfIndex]?.name;
    toast({ title: 'Retrying PDF load', description: name || 'PDF' });
  };

  const handleReuploadClick = () => fileInputRef.current?.click();
  const onFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setCreatedUrls(prev => [...prev, url]);
    const newEntry = { name: file.name, url };
    setLocalPdfs(prev => [newEntry, ...prev]);
    setPdfIndex(0);
    setPageNumber(1);
    setPdfError(null);
    setVersion(v => v + 1);
    toast({ title: 'PDF uploaded', description: file.name });
    e.currentTarget.value = '';
  };

  // Revoke object URLs we created when component unmounts
  useEffect(() => {
    return () => {
      createdUrls.forEach(u => URL.revokeObjectURL(u));
    };
  }, [createdUrls]);
  return (
    <div className="space-y-3">
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        onChange={onFileSelected}
        className="hidden"
      />
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>File:</span>
          <select
            value={pdfIndex}
            onChange={(e) => onSelectFile(parseInt(e.target.value))}
            className="border rounded px-2 py-1 bg-background"
          >
            {localPdfs.map((p, i) => (
              <option key={i} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <div className="text-sm text-muted-foreground">
            Page {pageNumber}{numPages ? ` / ${numPages}` : ''}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
            disabled={pageNumber <= 1 || !!pdfError}
            aria-label="Previous page"
          >
            Prev
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPageNumber((p) => (numPages ? Math.min(numPages, p + 1) : p + 1))}
            disabled={!!pdfError || (!!numPages && pageNumber >= (numPages || 1))}
            aria-label="Next page"
          >
            Next
          </Button>
        </div>
      </div>

      {pdfError && (
        <Alert variant="destructive">
          <AlertTitle>Failed to load PDF</AlertTitle>
          <AlertDescription className="flex items-center gap-3 flex-wrap">
            {pdfError}
            {localPdfs[pdfIndex]?.url ? (
              <a
                href={localPdfs[pdfIndex].url}
                target="_blank"
                rel="noreferrer"
                className="underline text-destructive-foreground/90"
                aria-label="Open original PDF in new tab"
              >
                Open original
              </a>
            ) : null}
            <Button variant="outline" size="sm" onClick={retry} aria-label="Retry loading PDF">
              Retry
            </Button>
            <Button size="sm" onClick={handleReuploadClick} aria-label="Re-upload PDF">
              Re-upload PDF
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div ref={containerRef} className="border rounded overflow-auto flex justify-center bg-muted/20 min-h-[200px]">
        {!pdfError ? (
          <Document
            key={`${localPdfs[pdfIndex].url}-${version}`}
            file={localPdfs[pdfIndex].url}
            onLoadSuccess={({ numPages }) => {
              setNumPages(numPages);
              setPdfError(null);
              const name = localPdfs[pdfIndex].name;
              toast({ title: 'PDF loaded', description: `${name} • ${numPages} page${numPages > 1 ? 's' : ''}` });
            }}
            onLoadError={(err) => {
              console.error('PDF load error:', err);
              setPdfError("We couldn't open the selected file. Try re-uploading or a different viewer.");
              toast({ variant: 'destructive', title: 'Failed to load PDF', description: "We couldn't open the selected file. Try re-uploading." });
            }}
          >
            <Page pageNumber={pageNumber} width={containerWidth} />
          </Document>
        ) : (
          <div className="w-full py-12 text-center text-sm text-muted-foreground">PDF viewer disabled due to load error.</div>
        )}
      </div>
    </div>
  );
}
