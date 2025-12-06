/**
 * PlatformsHub Integration Snippet
 *
 * This file demonstrates how to integrate the platforms registry
 * into the quantum-dev-profile Studios app.
 *
 * USAGE:
 * 1. Copy PROJECT-PLATFORMS-CONFIG.ts to quantum-dev-profile/src/data/platforms.ts
 * 2. Use this snippet as a reference for updating PlatformsHub.tsx
 */

import React from "react";
import { Link } from "react-router-dom";
import {
  PLATFORMS,
  getPlatformsGroupedByTier,
  getPlatformPrimaryUrl,
  TIER_ORDER,
  TIER_LABELS,
  type PlatformDefinition,
  type PlatformTier,
} from "@/data/platforms"; // Adjust path as needed

// Status badge component
function StatusBadge({ status }: { status: PlatformDefinition["status"] }) {
  const styles = {
    active: "bg-green-500/20 text-green-400 border-green-500/30",
    backend: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    planned: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  };

  const labels = {
    active: "Active",
    backend: "Backend",
    planned: "Planned",
  };

  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full border font-medium ${styles[status]}`}
    >
      {labels[status]}
    </span>
  );
}

// Platform card component
function PlatformCard({ platform }: { platform: PlatformDefinition }) {
  const [from, to] = platform.gradientColors || ["#38bdf8", "#0284c7"];
  const primaryUrl = getPlatformPrimaryUrl(platform);

  return (
    <div
      className="group relative rounded-xl border border-border/50 p-5 transition-all duration-300 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/5"
      style={{
        background: `linear-gradient(135deg, ${from}08, ${to}08)`,
      }}
    >
      {/* Gradient accent line */}
      <div
        className="absolute inset-x-0 top-0 h-0.5 rounded-t-xl opacity-0 transition-opacity group-hover:opacity-100"
        style={{
          background: `linear-gradient(90deg, ${from}, ${to})`,
        }}
      />

      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">
            {platform.name}
          </h3>
          {platform.domainUrl && (
            <span className="text-xs text-muted-foreground">
              {new URL(platform.domainUrl).hostname}
            </span>
          )}
        </div>
        <StatusBadge status={platform.status} />
      </div>

      {/* Tagline */}
      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
        {platform.tagline || platform.notes}
      </p>

      {/* Tags */}
      {platform.tags && platform.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {platform.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary/80 border border-primary/20"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 pt-2 border-t border-border/30">
        {platform.domainUrl && platform.status === "active" && (
          <a
            href={platform.domainUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 text-center text-xs font-medium py-2 px-3 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Open App
          </a>
        )}
        {platform.brandPageUrl && (
          <a
            href={platform.brandPageUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 text-center text-xs font-medium py-2 px-3 rounded-lg border border-border hover:border-primary hover:text-primary transition-colors"
          >
            Learn More
          </a>
        )}
        {platform.githubUrl && (
          <a
            href={platform.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs font-medium py-2 px-3 rounded-lg text-muted-foreground hover:text-foreground transition-colors"
          >
            GitHub
          </a>
        )}
      </div>
    </div>
  );
}

// Tier section component
function TierSection({
  tier,
  platforms,
}: {
  tier: PlatformTier;
  platforms: PlatformDefinition[];
}) {
  if (platforms.length === 0) return null;

  return (
    <section className="mb-10">
      <div className="flex items-center gap-3 mb-5">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          {TIER_LABELS[tier]}
        </h2>
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-muted-foreground">
          {platforms.length} {platforms.length === 1 ? "platform" : "platforms"}
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {platforms.map((platform) => (
          <PlatformCard key={platform.id} platform={platform} />
        ))}
      </div>
    </section>
  );
}

// Main PlatformsHub component
export function PlatformsHub() {
  const grouped = getPlatformsGroupedByTier();
  const totalPlatforms = PLATFORMS.length;
  const activePlatforms = PLATFORMS.filter((p) => p.status === "active").length;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-background/95 backdrop-blur sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <Link
                to="/studio"
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                ← Back to Studio
              </Link>
              <h1 className="text-2xl font-bold mt-1">Platforms Hub</h1>
            </div>
            <div className="flex gap-6 text-center">
              <div>
                <div className="text-2xl font-bold text-primary">
                  {totalPlatforms}
                </div>
                <div className="text-xs text-muted-foreground uppercase tracking-wider">
                  Total
                </div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-400">
                  {activePlatforms}
                </div>
                <div className="text-xs text-muted-foreground uppercase tracking-wider">
                  Active
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="container mx-auto px-4 py-8">
        {TIER_ORDER.map((tier) => (
          <TierSection key={tier} tier={tier} platforms={grouped[tier]} />
        ))}
      </main>
    </div>
  );
}

export default PlatformsHub;
