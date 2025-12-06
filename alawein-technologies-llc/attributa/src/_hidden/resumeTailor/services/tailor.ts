export function suggestBullets(resume: string, jd: string): string[] {
  const lines = resume.split(/\n+/).filter((l) => l.trim().length > 0).slice(0, 6);
  return lines.map((l) => `Led ${l.replace(/^[-*•]\s*/, "").trim()} using JD-aligned approach; achieved <X%> improvement (placeholder)`);
}

export function draftSummary(resume: string, jd: string, tone: "concise" | "impact" = "concise"): string {
  const base = `Engineer aligning experience with role requirements; focus on delivering outcomes.`;
  return tone === "impact" ? base + " Drove measurable results across key initiatives." : base;
}
