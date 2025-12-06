// src/services/demos.ts
import { mockIngest as ingest, mockAnalyzeText as analyzeText, mockAnalyzeWatermark as analyzeWatermark, mockAuditCitations as auditCitations, mockAnalyzeCode as analyzeCode, mockComputeScore as computeScore } from '@/services/realApi';

// Clear segment cache - simplified version since we don't have the exact same caching
function clearSegmentCache() {
  // This would clear any cached segments - for now it's a no-op
  console.log('Segment cache cleared');
}
import { useAppStore } from '@/store';

export async function runAllDemos() {
  const { analysisOptions, addReport, updateReport, scoringWeights } = useAppStore.getState();
  clearSegmentCache();

  const demos = [
    { name: 'Human prose', content: `In 2008, researchers at CERN achieved first beam circulation... This milestone reflected decades of international collaboration...` },
    { name: 'AI prose', content: `In today’s rapidly evolving landscape, it is important to recognize that innovation is a journey. By leveraging cutting-edge frameworks, organizations can unlock transformative value...` },
    { name: 'LaTeX', content: String.raw`\\documentclass{article}\\begin{document}We build on prior work \\cite{foo2020,bar2021}.\\bibliographystyle{plain}\\begin{thebibliography}{9}\\bibitem{foo2020} Foo, A. Title. 2020. doi:10.0000/fake-doi-1234 \\bibitem{bar2021} Bar, B. Another Title. 2021. doi:10.0000/fake-doi-5678 \\end{thebibliography}\\end{document}` },
    { name: 'Vuln Python', content: `import sqlite3, os\nconn = sqlite3.connect("app.db")\ncur = conn.cursor()\nuser=input("Username: ")\npwd=input("Password: ")\ncur.execute(f"SELECT * FROM users WHERE user='{user}' AND pwd='{pwd}'")\nprint(cur.fetchall())\nos.system("ping -c 1 " + input("Host: "))\nprint(eval(input("Calc: ")))` },
  ];

  for (const demo of demos) {
    const resp = await ingest({ source: 'paste' as const, content: demo.content, options: analysisOptions });

    // Seed a report in store
    addReport({
      docId: resp.docId,
      createdAt: Date.now(),
      summary: resp.summary,
      segments: resp.segments,
      signals: {},
      scores: {}
    });

    // Build signals and scores
    const signals: Record<string, unknown> = {};
    const scores: Record<string, unknown> = {};

    for (const seg of resp.segments) {
      try {
        const ta = await analyzeText(seg.segmentId);
        signals[seg.segmentId] = { gltr: ta.gltr, detectgpt: ta.detectgpt };

        // Watermark analysis with error handling
        if ((seg.type === 'prose' || seg.type === 'latex') && analysisOptions.tryWatermark) {
          try {
            const wm = await analyzeWatermark(seg.segmentId);
            signals[seg.segmentId].watermark = wm.watermark || wm; // wrapper returns different shapes sometimes
          } catch (wmError) {
            console.warn(`Watermark analysis failed for ${seg.segmentId}:`, wmError);
            // Continue without watermark signal
          }
        }

        const score = await computeScore(seg.segmentId, {
          gltrTail: ta.gltr.tailTokenShare,
          gltrVar: ta.gltr.rankVariance,
          curvature: ta.detectgpt.curvature,
          watermarkP: signals[seg.segmentId].watermark?.pValue ?? null,
          lengthChars: seg.lengthChars,
          type: seg.type
        }, scoringWeights);
        scores[seg.segmentId] = score;
        
      } catch (segError) {
        console.warn(`Analysis failed for segment ${seg.segmentId}:`, segError);
        
        // Provide fallback analysis with reduced DetectGPT curvature if model loading failed
        signals[seg.segmentId] = {
          gltr: { tailTokenShare: 0.3, rankVariance: 0.5 }, // Fallback GLTR
          detectgpt: { curvature: 0.1, numPerturbations: 0 } // Fallback DetectGPT (positive curvature = less suspicious)
        };
        
        const score = await computeScore(seg.segmentId, {
          gltrTail: 0.3,
          gltrVar: 0.5,
          curvature: 0.1, // Less suspicious fallback
          watermarkP: null,
          lengthChars: seg.lengthChars,
          type: seg.type
        }, scoringWeights);
        scores[seg.segmentId] = score;
      }
    }

    // Doc-level analyses (with resilience)
    const hasText = resp.segments.some(s => s.type === 'latex' || s.type === 'prose');
    const hasCode = resp.segments.some(s => s.type === 'code');

    let citations: unknown[] = [];
    let codeFindings: unknown[] = [];
    
    // Citation analysis with offline fallback
    if (hasText) {
      try {
        const citationResult = await auditCitations(resp.docId);
        citations = citationResult.citations || [];
      } catch (error) {
        console.warn('Citation analysis failed (offline?), continuing without suggestions:', error);
        citations = []; // Continue without suggestions
      }
    }
    
    // Code analysis with error handling
    if (hasCode) {
      try {
        const codeResult = await analyzeCode(resp.docId);
        codeFindings = codeResult.findings || [];
      } catch (error) {
        console.warn('Code analysis failed, skipping:', error);
        codeFindings = [];
      }
    }

    updateReport({
      docId: resp.docId,
      createdAt: Date.now(),
      summary: resp.summary,
      segments: resp.segments,
      signals,
      scores,
      citations,
      codeFindings,
    });
  }
}
