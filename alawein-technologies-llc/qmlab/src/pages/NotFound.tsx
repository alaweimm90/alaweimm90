import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { PageChrome } from "@/components/PageChrome";
import { Button } from "@/components/ui/button";
import { Home } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-[var(--surface-base)]">
      <PageChrome
        title="Page Not Found - QMLab"
        showTitle={false}
      >
        <div className="flex items-center justify-center min-h-[80vh] px-[var(--space-6)]">
          <div className="text-center glass-panel p-[var(--space-12)] max-w-md">
            <div className="mb-[var(--space-8)]">
              <h2 className="heading-refined-1 mb-[var(--space-4)]">404</h2>
              <p className="body-elegant mb-[var(--space-6)]">
                The quantum state you're looking for doesn't exist in this dimension.
              </p>
              <p className="body-elegant-sm text-[var(--text-muted)] mb-[var(--space-6)]">
                Path attempted: <code className="bg-[var(--surface-2)] px-2 py-1 rounded">{location.pathname}</code>
              </p>
            </div>
            <Button
              asChild
              variant="primary"
              size="lg"
              className="w-full"
            >
              <a href="/" aria-label="Return to QMLab homepage">
                <Home className="w-4 h-4 mr-2" aria-hidden="true" />
                Return to Laboratory
              </a>
            </Button>
          </div>
        </div>
      </PageChrome>
    </div>
  );
};

export default NotFound;
