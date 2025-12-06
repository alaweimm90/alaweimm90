import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function ResumeUploader({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const onFile = async (file: File) => {
    const text = await file.text();
    onChange(text);
  };
  return (
    <div className="space-y-2">
      <Label>Resume</Label>
      <Input type="file" accept=".txt,.md,.docx,.pdf" onChange={(e) => e.target.files && onFile(e.target.files[0])} />
      <Textarea rows={10} placeholder="Paste your resume text..." value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
