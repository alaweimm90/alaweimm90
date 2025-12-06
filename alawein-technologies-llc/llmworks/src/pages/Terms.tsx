import { memo } from "react";
import { Navigation } from "@/components/Navigation";
import { Footer } from "@/components/Footer";
import { Card } from "@/components/ui/card";

const TermsPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main id="main" className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-foreground mb-6">Terms of Service</h1>
          <p className="text-muted-foreground mb-8">
            Last updated: January 12, 2025
          </p>

          <Card className="p-8 space-y-6">
            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Open Source License</h2>
              <p className="text-muted-foreground">
                LLM Works is licensed under the MIT License. You are free to use, modify, and distribute 
                this software for any purpose, including commercial use.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Usage Terms</h2>
              <p className="text-muted-foreground mb-3">
                By using LLM Works, you agree to:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-2">
                <li>Use the platform responsibly and in compliance with applicable laws</li>
                <li>Respect API rate limits and terms of service of model providers you connect</li>
                <li>Not attempt to reverse-engineer or compromise the platform security</li>
                <li>Not use the platform for illegal or harmful activities</li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Disclaimer</h2>
              <p className="text-muted-foreground">
                LLM Works is provided "as is" without warranty of any kind. We make no guarantees about 
                the accuracy of model evaluations or the availability of the service.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Limitation of Liability</h2>
              <p className="text-muted-foreground">
                In no event shall the LLM Works contributors be liable for any damages arising from 
                the use of this software.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Changes to Terms</h2>
              <p className="text-muted-foreground">
                We may update these terms from time to time. Changes will be posted in our{" "}
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

            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4">Contact</h2>
              <p className="text-muted-foreground">
                Questions about these terms? Open an issue on our{" "}
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

export default memo(TermsPage);