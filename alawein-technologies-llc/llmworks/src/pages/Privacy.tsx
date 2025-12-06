import { memo } from "react";
import { Navigation } from "@/components/Navigation";
import { Footer } from "@/components/Footer";
import { Card } from "@/components/ui/card";

const PrivacyPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main id="main" className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-foreground mb-6">Privacy Policy</h1>
          <p className="text-muted-foreground mb-8">
            Last updated: January 12, 2025
          </p>

          <Card className="p-8 space-y-6">
            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Local Processing</h2>
              <p className="text-muted-foreground">
                LLM Works is designed with privacy-first principles. All model evaluations run locally in your browser. 
                We do not send your prompts, model outputs, or evaluation data to our servers.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Data We Collect</h2>
              <p className="text-muted-foreground mb-3">
                We collect minimal data to improve the platform:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-2">
                <li>Anonymous usage analytics (page views, feature usage)</li>
                <li>Error reports (when enabled in settings)</li>
                <li>Performance metrics to optimize the platform</li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Your Model Data</h2>
              <p className="text-muted-foreground">
                Your model configurations, API keys, evaluation results, and all sensitive data remain on your device. 
                We never have access to your model inputs, outputs, or evaluation data.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Third-Party Services</h2>
              <p className="text-muted-foreground">
                When you configure model providers (OpenAI, Anthropic, etc.), you connect directly to their APIs. 
                We do not intercept or store these communications.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Contact</h2>
              <p className="text-muted-foreground">
                Questions about this privacy policy? Open an issue on our{" "}
                <a 
                  href="https://github.com/alaweimm90/aegis-ai-evaluator" 
                  className="text-primary hover:underline"
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  GitHub repository
                </a>.
              </p>
            </section>
          </Card>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default memo(PrivacyPage);