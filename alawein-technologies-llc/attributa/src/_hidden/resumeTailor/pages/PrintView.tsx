import { useParams } from "react-router-dom";

export default function PrintView() {
  const { docId } = useParams();
  return (
    <main className="container mx-auto px-4 py-8 print:p-0">
      <h1 className="text-2xl font-bold mb-4">Resume Tailor Export</h1>
      <p className="text-muted-foreground mb-8">Doc ID: {docId}</p>
      {/* TODO: Render print-friendly tailored resume */}
      <p className="text-sm">Use your browser's print dialog (Ctrl/Cmd+P) to export as PDF.</p>
    </main>
  );
}
