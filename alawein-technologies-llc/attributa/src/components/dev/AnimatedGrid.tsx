import React, { useState } from "react";

export default function AnimatedGrid() {
  const [isPaused, setIsPaused] = useState(false);

  return (
    <div 
      className={`grid-background ${isPaused ? 'paused' : ''}`}
      aria-hidden="true"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
      onFocus={() => setIsPaused(true)}
      onBlur={() => setIsPaused(false)}
    />
  );
}
