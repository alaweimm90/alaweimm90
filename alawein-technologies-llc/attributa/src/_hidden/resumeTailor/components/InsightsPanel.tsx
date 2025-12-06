export function InsightsPanel({ results }: { results?: { suggestions: string[] } }) {
  if (!results) return <p className="text-muted-foreground">No insights yet.</p>;
  return (
    <ul className="list-disc pl-5 space-y-1">
      {results.suggestions.map((s, i) => (
        <li key={i}>{s}</li>
      ))}
    </ul>
  );
}
