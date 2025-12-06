import { Link, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function Landing() {
  const navigate = useNavigate();
  return (
    <main className="container mx-auto px-4 py-10 max-w-4xl">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Resume Tailor</h1>
        <p className="text-muted-foreground">Local-first resume and job description tailoring with private, deterministic analysis.</p>
      </header>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Get Started</CardTitle>
            <CardDescription>Upload or paste your resume and a job description</CardDescription>
          </CardHeader>
          <CardContent className="flex gap-3">
            <Button asChild>
              <Link to="/resume/workspace">Open workspace</Link>
            </Button>
            <Button variant="outline" onClick={() => navigate("/resume/workspace?demo=1")}>Load demo</Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">What's new <Badge>NEW</Badge></CardTitle>
            <CardDescription>Private analysis, STAR rewrites, and exports</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>Runs locally by default. Optional BYO keys for higher quality embeddings/writing.</p>
            <p>Export tailored resume as Markdown or DOCX; print-friendly view provided.</p>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
