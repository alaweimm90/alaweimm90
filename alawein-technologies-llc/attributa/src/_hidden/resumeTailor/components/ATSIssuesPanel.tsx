export function ATSIssuesPanel({ results }: { results?: { issues: { id: string; text: string; severity: string }[] } }) {
  if (!results) return <p className="text-muted-foreground">No issues detected.</p>;
  return (
    <ul className="space-y-2">
      {results.issues.map((i) => (
        <li key={i.id} className="text-sm">
          <span className={i.severity === 'high' ? 'text-destructive' : i.severity === 'medium' ? 'text-warning' : 'text-muted-foreground'}>
            [{i.severity.toUpperCase()}]
          </span>{' '}
          {i.text}
        </li>
      ))}
    </ul>
  );
}
