import React from "react";
import { Star, Server, Users } from "lucide-react";
import { useCountUp } from "@/hooks/useCountUp";

function formatShort(n: number) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

export default function DevMetricsBar() {
  const stars = useCountUp(2300, { formatter: formatShort });
  const requests = useCountUp(12800000, { formatter: formatShort });
  const devs = useCountUp(8420, { formatter: formatShort });

  return (
    <section aria-label="Developer metrics" className="glass-card rounded-lg p-4">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        <div ref={stars.ref} className="flex items-center gap-3">
          <div className="h-7 w-7 rounded-md bg-primary/15 flex items-center justify-center">
            <Star className="h-4 w-4 text-primary" />
          </div>
          <div className="leading-tight">
            <div className="font-medium">GitHub Stars</div>
            <div className="text-muted-foreground tabular-nums">{stars.display}+</div>
          </div>
        </div>
        <div ref={requests.ref} className="flex items-center gap-3">
          <div className="h-7 w-7 rounded-md bg-accent/15 flex items-center justify-center">
            <Server className="h-4 w-4 text-accent" />
          </div>
          <div className="leading-tight">
            <div className="font-medium">API Requests Served</div>
            <div className="text-muted-foreground tabular-nums">{requests.display}</div>
          </div>
        </div>
        <div ref={devs.ref} className="flex items-center gap-3">
          <div className="h-7 w-7 rounded-md bg-foreground/10 flex items-center justify-center">
            <Users className="h-4 w-4" />
          </div>
          <div className="leading-tight">
            <div className="font-medium">Active Developers</div>
            <div className="text-muted-foreground tabular-nums">{devs.display}+</div>
          </div>
        </div>
      </div>
    </section>
  );
}
