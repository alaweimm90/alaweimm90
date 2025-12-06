import { Badge } from "@/components/ui/badge";

export function ScoreDisplay({ results }: { results?: { semantic: number; coverage: number; ats: number; gap: number; final: number } }) {
  if (!results) return <p className="text-muted-foreground">Run analysis to see scores.</p>;
  const { semantic, coverage, ats, gap, final } = results;
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      <div><p className="text-xs text-muted-foreground">Semantic fit</p><p className="text-xl font-semibold">{semantic}</p></div>
      <div><p className="text-xs text-muted-foreground">Coverage</p><p className="text-xl font-semibold">{coverage}</p></div>
      <div><p className="text-xs text-muted-foreground">ATS</p><p className="text-xl font-semibold">{ats}</p></div>
      <div><p className="text-xs text-muted-foreground">Gap</p><p className="text-xl font-semibold">{gap}</p></div>
      <div className="col-span-2 md:col-span-1"><p className="text-xs text-muted-foreground">Final</p><p className="text-2xl font-bold">{final} <Badge variant="outline" className="ml-1">Weighted</Badge></p></div>
    </div>
  );
}
