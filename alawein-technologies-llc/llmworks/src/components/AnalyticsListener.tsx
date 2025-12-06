import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { trackPageView, trackPageLoad, trackError } from "@/lib/analytics";

export default function AnalyticsListener() {
  const location = useLocation();

  useEffect(() => {
    // Track page view
    trackPageView(location.pathname, document.title);
    
    // Track page load performance on initial load
    if (location.pathname === "/" || location.pathname === "") {
      const timer = setTimeout(() => {
        trackPageLoad();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [location.pathname]);

  useEffect(() => {
    // Global error handler
    const handleError = (event: ErrorEvent) => {
      trackError(new Error(event.message), `${event.filename}:${event.lineno}`);
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      trackError(new Error(event.reason), "Unhandled Promise Rejection");
    };

    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);

    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    };
  }, []);

  return null;
}
