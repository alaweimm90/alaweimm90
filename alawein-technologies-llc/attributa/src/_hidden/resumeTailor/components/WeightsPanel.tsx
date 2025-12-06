import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRtStore } from "../store";

export function WeightsPanel() {
  const { weights, setWeights } = useRtStore();
  const items: { key: keyof typeof weights; label: string; max?: number }[] = [
    { key: "semantic", label: "Semantic fit", max: 100 },
    { key: "coverage", label: "Coverage", max: 100 },
    { key: "ats", label: "ATS", max: 100 },
    { key: "gap", label: "Gap penalty", max: 100 },
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Weights</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {items.map(({ key, label }) => (
          <div key={key} className="space-y-2">
            <div className="flex justify-between"><Label>{label}</Label><span className="text-sm text-muted-foreground">{Math.round(weights[key] * 100)}%</span></div>
            <Slider value={[weights[key] * 100]} onValueChange={(v) => setWeights({ [key]: v[0] / 100 } as Partial<typeof weights>)} max={100} step={1} />
          </div>
        ))}
        <p className="text-xs text-muted-foreground">Weights are normalized during scoring.</p>
      </CardContent>
    </Card>
  );
}
