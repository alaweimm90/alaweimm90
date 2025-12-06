import { useEffect, useRef, useState } from "react";

interface UseCountUpOptions {
  durationMs?: number;
  startOnView?: boolean;
  formatter?: (n: number) => string;
}

export function useCountUp(target: number, {
  durationMs = 1200,
  startOnView = true,
  formatter,
}: UseCountUpOptions = {}) {
  const [value, setValue] = useState(0);
  const [display, setDisplay] = useState("0");
  const ref = useRef<HTMLDivElement | null>(null);
  const started = useRef(false);
  const startTs = useRef<number | null>(null);

  useEffect(() => {
    if (!startOnView) return;

    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !started.current) {
            started.current = true;
            startTs.current = null;
            const step = (ts: number) => {
              if (startTs.current === null) startTs.current = ts;
              const progress = Math.min(1, (ts - startTs.current) / durationMs);
              const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
              const next = target * eased;
              setValue(next);
              setDisplay(formatter ? formatter(next) : String(Math.floor(next)));
              if (progress < 1) requestAnimationFrame(step);
            };
            requestAnimationFrame(step);
          }
        });
      },
      { threshold: 0.3 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [target, durationMs, formatter, startOnView]);

  useEffect(() => {
    if (!startOnView) {
      // immediate animation without intersection
      let raf: number;
      startTs.current = null;
      const step = (ts: number) => {
        if (startTs.current === null) startTs.current = ts;
        const progress = Math.min(1, (ts - startTs.current) / durationMs);
        const eased = 1 - Math.pow(1 - progress, 3);
        const next = target * eased;
        setValue(next);
        setDisplay(formatter ? formatter(next) : String(Math.floor(next)));
        if (progress < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
      return () => cancelAnimationFrame(raf);
    }
  }, [startOnView, target, durationMs, formatter]);

  return { ref, value, display } as const;
}
