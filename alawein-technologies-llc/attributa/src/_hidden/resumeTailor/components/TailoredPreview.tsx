import { useRtStore } from "../store";
import { downloadJSON, downloadMarkdown, downloadDocx } from "../lib/export";

export function TailoredPreview() {
  const { results } = useRtStore();
  if (!results) return null;
  const md = `# Tailored Resume\n\nFinal Score: ${results.final}\n\n## Suggestions\n${results.suggestions.map((s) => `- ${s}`).join("\n")}`;
  return (
    <div className="flex gap-2">
      <button className="underline text-sm" onClick={() => downloadMarkdown(md)}>Export Markdown</button>
      <button className="underline text-sm" onClick={() => downloadDocx("Tailored Resume", [{ heading: "Suggestions", lines: results.suggestions }])}>Export DOCX</button>
      <button className="underline text-sm" onClick={() => downloadJSON(results)}>Export JSON</button>
    </div>
  );
}
