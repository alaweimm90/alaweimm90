import React, { useEffect, useRef, useState } from "react";

interface TypewriterProps {
  lines: string[];
  speedMs?: number;
  pauseMs?: number;
}

export default function Typewriter({ lines, speedMs = 28, pauseMs = 1200 }: TypewriterProps) {
  const [idx, setIdx] = useState(0);
  const [subIdx, setSubIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [blink, setBlink] = useState(true);
  const mounted = useRef(true);

  useEffect(() => {
    return () => { mounted.current = false };
  }, []);

  useEffect(() => {
    if (!mounted.current) return;
    const current = lines[idx % lines.length] || "";

    if (!deleting && subIdx === current.length) {
      const t = setTimeout(() => setDeleting(true), pauseMs);
      return () => clearTimeout(t);
    }

    if (deleting && subIdx === 0) {
      setDeleting(false);
      setIdx((v) => (v + 1) % lines.length);
      return;
    }

    const timeout = setTimeout(() => {
      setSubIdx((v) => v + (deleting ? -1 : 1));
      setBlink((b) => !b);
    }, deleting ? speedMs / 1.5 : speedMs);

    return () => clearTimeout(timeout);
  }, [subIdx, idx, deleting, lines, speedMs, pauseMs]);

  const text = (lines[idx % lines.length] || "").slice(0, subIdx);

  return (
    <div className="flex items-center">
      <span className="text-muted-foreground">$</span>
      <span className="ml-2 font-mono text-muted-foreground">{text}</span>
      <span className="type-caret" style={{ opacity: blink ? 1 : 0 }} />
    </div>
  );
}
