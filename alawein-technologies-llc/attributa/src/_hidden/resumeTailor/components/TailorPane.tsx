import { suggestBullets, draftSummary } from "../services/tailor";

export function TailorPane({ resumeText, jdText }: { resumeText: string; jdText: string }) {
  if (!resumeText || !jdText) return <p className="text-muted-foreground">Provide inputs to see suggestions.</p>;
  const bullets = suggestBullets(resumeText, jdText);
  const summary = draftSummary(resumeText, jdText);
  return (
    <div className="space-y-4">
      <div>
        <h3 className="font-medium mb-2">Rewritten bullets (STAR)</h3>
        <ul className="list-disc pl-5 space-y-1">
          {bullets.map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
      </div>
      <div>
        <h3 className="font-medium mb-2">Summary suggestion</h3>
        <p className="text-sm">{summary}</p>
      </div>
    </div>
  );
}
