import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useRtStore } from "../store";

export default function SettingsTab() {
  const { options, setOptions, clearKeys, weights, setWeights } = useRtStore();
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Resume Tailor</CardTitle>
          <CardDescription>Local-first analysis. BYO keys are optional and kept in memory only.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label>Local only</Label>
              <p className="text-sm text-muted-foreground">Never send data to network</p>
            </div>
            <Switch checked={options.localOnly} onCheckedChange={(v) => setOptions({ localOnly: v })} />
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>OpenAI API Key</Label>
              <Input type="password" placeholder="sk-..." onChange={(e) => setOptions({ openaiApiKey: e.target.value })} />
            </div>
            <div className="space-y-2">
              <Label>Gemini API Key</Label>
              <Input type="password" placeholder="AI..." onChange={(e) => setOptions({ geminiApiKey: e.target.value })} />
            </div>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" onClick={clearKeys}>Clear credentials</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Weight presets</CardTitle>
          <CardDescription>Quickly adjust scoring focus</CardDescription>
        </CardHeader>
        <CardContent className="flex gap-2 flex-wrap">
          <Button size="sm" variant="outline" onClick={() => setWeights({ semantic: 0.4, coverage: 0.35, ats: 0.2, gap: 0.05 })}>Default</Button>
          <Button size="sm" variant="outline" onClick={() => setWeights({ ats: 0.35, semantic: 0.35, coverage: 0.25, gap: 0.05 })}>Conservative (ATS↑)</Button>
          <Button size="sm" variant="outline" onClick={() => setWeights({ semantic: 0.5, coverage: 0.3, ats: 0.15, gap: 0.05 })}>Aggressive (Semantic↑)</Button>
          <p className="text-xs text-muted-foreground">Weights are normalized during scoring.</p>
        </CardContent>
      </Card>
    </div>
  );
}
