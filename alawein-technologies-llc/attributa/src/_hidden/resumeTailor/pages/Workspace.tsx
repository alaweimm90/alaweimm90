import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useRtStore } from "../store";
import { analyze } from "../lib/analysis";
import { ResumeUploader } from "../components/ResumeUploader";
import { JobDescriptionInput } from "../components/JobDescriptionInput";
import { ScoreDisplay } from "../components/ScoreDisplay";
import { InsightsPanel } from "../components/InsightsPanel";
import { ATSIssuesPanel } from "../components/ATSIssuesPanel";
import { WeightsPanel } from "../components/WeightsPanel";
import { TailorPane } from "../components/TailorPane";
import { TailoredPreview } from "../components/TailoredPreview";
import demoResume from "../demo/resume_software.md?raw";
import demoJD from "../demo/jd_frontend.md?raw";

export default function Workspace() {
  const [params] = useSearchParams();
  const { resumeText, jdText, setResumeText, setJdText, weights, results, setResults } = useRtStore();

  useEffect(() => {
    if (params.get("demo")) {
      if (!resumeText) setResumeText(demoResume);
      if (!jdText) setJdText(demoJD);
    }
  }, [params, resumeText, jdText, setResumeText, setJdText]);

  const run = () => {
    const r = analyze(resumeText, jdText, weights);
    setResults(r);
  };

  return (
    <main className="container mx-auto px-4 py-6">
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Inputs</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ResumeUploader value={resumeText} onChange={setResumeText} />
              <JobDescriptionInput value={jdText} onChange={setJdText} />
              <Button onClick={run} disabled={!resumeText || !jdText}>Run analysis</Button>
            </CardContent>
          </Card>

          <WeightsPanel />
        </div>

        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Scores</CardTitle>
            </CardHeader>
            <CardContent>
              <ScoreDisplay results={results} />
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <InsightsPanel results={results} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>ATS Issues</CardTitle>
              </CardHeader>
              <CardContent>
                <ATSIssuesPanel results={results} />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Tailor</CardTitle>
            </CardHeader>
            <CardContent>
              <TailorPane resumeText={resumeText} jdText={jdText} />
              <div className="mt-4">
                <TailoredPreview />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}
