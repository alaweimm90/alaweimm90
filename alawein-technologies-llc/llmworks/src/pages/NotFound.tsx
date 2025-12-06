import { useLocation, Link } from "react-router-dom";
import { useEffect, memo } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Home, ArrowLeft } from "lucide-react";
import { setSEO } from "@/lib/seo";
import { trackEvent } from "@/lib/analytics";

const NotFound = memo(() => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
    trackEvent("page_not_found", { 
      attempted_path: location.pathname,
      referrer: document.referrer 
    });
    setSEO({
      title: "404 Not Found | LLM Works",
      description: "The page you're looking for doesn't exist. Return to the LLM Works home page.",
      path: location.pathname,
    });
  }, [location.pathname]);

  const handleNavigateHome = () => {
    trackEvent("404_navigate_home", { from_path: location.pathname });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="max-w-md w-full p-8 text-center shadow-elegant">
        <div className="mb-6">
          <h1 className="text-6xl font-bold text-destructive mb-2">404</h1>
          <h2 className="text-2xl font-semibold text-foreground mb-4">Page Not Found</h2>
          <p className="text-muted-foreground mb-6">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>
        
        <div className="space-y-3">
          <Button asChild size="lg" className="w-full" onClick={handleNavigateHome}>
            <Link to="/">
              <Home className="h-4 w-4 mr-2" />
              Return Home
            </Link>
          </Button>
          
          <Button asChild variant="outline" size="lg" className="w-full">
            <Link to="#" onClick={() => window.history.back()}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Go Back
            </Link>
          </Button>
        </div>

        <p className="text-sm text-muted-foreground mt-6">
          Attempted path: <code className="bg-muted px-2 py-1 rounded text-xs">{location.pathname}</code>
        </p>
      </Card>
    </div>
  );
});

NotFound.displayName = 'NotFound';

export default NotFound;