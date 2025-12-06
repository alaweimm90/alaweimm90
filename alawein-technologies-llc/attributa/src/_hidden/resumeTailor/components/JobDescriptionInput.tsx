import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";

export function JobDescriptionInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="space-y-2">
      <Label>Job Description</Label>
      <Textarea rows={10} placeholder="Paste the job description..." value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
