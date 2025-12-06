import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { useSEO } from "@/hooks/useSEO";
import NeuralBackground from '@/components/dev/NeuralBackground';
import AnimatedGrid from '@/components/dev/AnimatedGrid';

const NotFound = () => {
  const location = useLocation();

  useSEO({
    title: "404 — Page Not Found | Attributa.dev",
    description: "The page you’re looking for doesn’t exist. Return to Attributa.dev home.",
  });

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route on Attributa.dev:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary engineering-pattern">
      <NeuralBackground />
      <AnimatedGrid />
      
      <div className="relative z-10 min-h-screen flex items-center justify-center">
        <div className="text-center animate-fade-in">
        <h1 className="text-4xl font-bold mb-4 display-tight font-mono">404</h1>
        <p className="text-xl text-muted-foreground mb-4">Oops! Page not found</p>
        <Link to="/" className="text-primary hover:text-primary/80 underline">
          Return to Home
        </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
