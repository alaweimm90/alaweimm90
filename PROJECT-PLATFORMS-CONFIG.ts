// Central platforms registry for Studios / PlatformsHub integration
// Note: repoPath values are from this meta-monorepo; adjust if your Studio app
// lives in a different repo. Portfolio and family projects are marked planned.

export type PlatformStatus = "active" | "backend" | "planned";

export type PlatformTier =
  | "scientific"
  | "ai-ml"
  | "cultural"
  | "business"
  | "lifestyle"
  | "family"
  | "portfolio";

export type PlatformAppType = "spa" | "backend";

export interface PlatformDefinition {
  /** Stable ID for routing/keys */
  id: string;
  name: string;
  tier: PlatformTier;
  status: PlatformStatus;

  /** Path to the project within this monorepo (or external repo hint) */
  repoPath?: string;

  /** High-level implementation style */
  appType?: PlatformAppType;

  /** Primary design token used by the platform (if any) */
  primaryColorToken?: string;

  /** Whether the UI expects an auth flow for core routes */
  hasAuth?: boolean;

  /** Whether there is a dedicated admin area */
  hasAdminArea?: boolean;

  /** Free-form tags for filtering/search */
  tags?: string[];

  /** Any important notes or integration caveats */
  notes?: string;

  /** Live domain URL if deployed (e.g. https://simcore.dev) */
  domainUrl?: string;

  /** Brand landing page URL (GitHub Pages or similar) */
  brandPageUrl?: string;

  /** GitHub repository URL */
  githubUrl?: string;

  /** Short tagline for UI display */
  tagline?: string;

  /** Gradient colors for card styling [from, to] */
  gradientColors?: [string, string];
}

export const PLATFORMS: PlatformDefinition[] = [
  // ---------------------------------------------------------------------------
  // Scientific / Technical
  // ---------------------------------------------------------------------------
  {
    id: "simcore",
    name: "SimCore",
    tier: "scientific",
    status: "active",
    repoPath: "alawein-technologies-llc/simcore",
    appType: "spa",
    primaryColorToken: "--primary",
    hasAuth: false,
    hasAdminArea: false,
    tags: ["scientific", "simulations", "quantum", "materials"],
    notes: "Full Vite + React + Tailwind + shadcn SPA for scientific simulations.",
    domainUrl: "https://simcore.dev",
    brandPageUrl: "/pages/brands/simcore/",
    githubUrl: "https://github.com/AlaweinOS/SimCore",
    tagline: "Interactive Scientific Computing Platform",
    gradientColors: ["#22c55e", "#15803d"],
  },
  {
    id: "qmlab",
    name: "QMLab",
    tier: "scientific",
    status: "active",
    repoPath: "alawein-technologies-llc/qmlab",
    appType: "spa",
    primaryColorToken: "--primary",
    hasAuth: false,
    hasAdminArea: false,
    tags: ["quantum", "lab", "accessibility", "playground"],
    notes: "Quantum mechanics playground SPA with strong accessibility focus.",
    domainUrl: "https://qmlab.online",
    brandPageUrl: "/pages/brands/qmlab/",
    githubUrl: "https://github.com/AlaweinOS/qmlab",
    tagline: "Quantum Mechanics in the Browser",
    gradientColors: ["#38bdf8", "#0284c7"],
  },
  {
    id: "optilibria",
    name: "OptiLibria",
    tier: "scientific",
    status: "planned",
    repoPath: "alawein-technologies-llc/librex",
    appType: "backend",
    tags: ["optimization", "portfolio", "librex", "libria"],
    notes: "Optimization logic exists inside Librex/Libria; front-end planned.",
    brandPageUrl: "/pages/brands/librex/",
    githubUrl: "https://github.com/AlaweinOS/Librex",
    tagline: "GPU-Accelerated Optimization Framework",
    gradientColors: ["#a855f7", "#7c3aed"],
  },

  // ---------------------------------------------------------------------------
  // AI / ML
  // ---------------------------------------------------------------------------
  {
    id: "talai",
    name: "TalAI",
    tier: "ai-ml",
    status: "backend",
    repoPath: "alawein-technologies-llc/talai",
    appType: "backend",
    tags: ["research", "automation", "workflows", "python"],
    notes: "Large Python research/orchestration system with 50+ modules.",
    domainUrl: "https://talai.dev",
    brandPageUrl: "/pages/brands/talai/",
    githubUrl: "https://github.com/AlaweinOS/TalAI",
    tagline: "Autonomous AI Research Assistant",
    gradientColors: ["#f97316", "#ea580c"],
  },
  {
    id: "llmworks",
    name: "LLMWorks",
    tier: "ai-ml",
    status: "active",
    repoPath: "alawein-technologies-llc/llmworks",
    appType: "spa",
    primaryColorToken: "--primary",
    hasAuth: false,
    hasAdminArea: true,
    tags: ["llm", "evaluation", "benchmarking", "react"],
    notes: "LLM evaluation and benchmarking SPA with Supabase backend.",
    domainUrl: "https://llmworks.dev",
    brandPageUrl: "/pages/brands/llmworks/",
    githubUrl: "https://github.com/alawein-tools/LLMWorks",
    tagline: "Interactive LLM Evaluation & Benchmarking",
    gradientColors: ["#a855f7", "#7c3aed"],
  },

  // ---------------------------------------------------------------------------
  // Cultural / Themed
  // ---------------------------------------------------------------------------
  {
    id: "mezan",
    name: "MEZAN",
    tier: "cultural",
    status: "backend",
    repoPath: "alawein-technologies-llc/mezan",
    appType: "backend",
    tags: ["orchestration", "libria", "optimization", "python"],
    notes: "Meta-Equilibrium Zero-regret Assignment Network — Python orchestration layer.",
    domainUrl: "https://mezan.dev",
    brandPageUrl: "/pages/brands/mezan/",
    githubUrl: "https://github.com/AlaweinOS/MEZAN",
    tagline: "Meta-Solver Orchestrator for Optimization",
    gradientColors: ["#ec4899", "#be185d"],
  },

  // ---------------------------------------------------------------------------
  // Business / Professional
  // ---------------------------------------------------------------------------
  {
    id: "attributa",
    name: "Attributa",
    tier: "business",
    status: "active",
    repoPath: "alawein-technologies-llc/attributa",
    appType: "spa",
    primaryColorToken: "--primary",
    hasAuth: true,
    hasAdminArea: false,
    tags: ["attribution", "analytics", "ai-detection", "react"],
    notes: "Attribution / AI detection analytics SPA with rich AI-specific color tokens.",
    domainUrl: "https://attributa.dev",
    brandPageUrl: "/pages/brands/attributa/",
    githubUrl: "https://github.com/alawein-tools/Attributa",
    tagline: "AI-Powered Attribution & Resume Analysis",
    gradientColors: ["#ec4899", "#be185d"],
  },
  {
    id: "repz",
    name: "REPZ Coach",
    tier: "business",
    status: "active",
    repoPath: "repz-llc/repz",
    appType: "spa",
    primaryColorToken: "--primary",
    hasAuth: true,
    hasAdminArea: true,
    tags: ["fitness", "coaching", "supabase", "stripe", "react"],
    notes: "Fitness coach platform SPA with auth, tiered access, Supabase, and Stripe.",
    domainUrl: "https://getrepz.app",
    brandPageUrl: "/pages/brands/repz/",
    githubUrl: "https://github.com/repz-llc/repz",
    tagline: "AI-Powered Fitness Coaching Platform",
    gradientColors: ["#22c55e", "#15803d"],
  },

  // ---------------------------------------------------------------------------
  // Lifestyle / E‑commerce
  // ---------------------------------------------------------------------------
  {
    id: "liveiticonic",
    name: "Live It Iconic",
    tier: "lifestyle",
    status: "active",
    repoPath: "live-it-iconic-llc/liveiticonic",
    appType: "spa",
    primaryColorToken: "--lii-gold",
    hasAuth: true,
    hasAdminArea: true,
    tags: ["ecommerce", "lifestyle", "luxury", "react"],
    notes: "Premium lifestyle merchandise SPA with custom design tokens and Stripe.",
    domainUrl: "https://liveiticonic.com",
    brandPageUrl: "/pages/brands/liveiticonic/",
    githubUrl: "https://github.com/live-it-iconic-llc/liveiticonic",
    tagline: "Luxury Automotive Lifestyle Brand",
    gradientColors: ["#facc15", "#ca8a04"],
  },
  {
    id: "peptidevault",
    name: "PeptideVault",
    tier: "lifestyle",
    status: "planned",
    repoPath: undefined,
    appType: "spa",
    tags: ["biotech", "ecommerce", "planned"],
    notes: "Planned biotech e-commerce platform.",
    tagline: "Biotech E-Commerce Platform",
    gradientColors: ["#22d3ee", "#0891b2"],
  },

  // ---------------------------------------------------------------------------
  // Portfolio
  // ---------------------------------------------------------------------------
  {
    id: "portfolio-main",
    name: "Portfolio",
    tier: "portfolio",
    status: "active",
    repoPath: undefined,
    appType: "spa",
    tags: ["portfolio", "personal"],
    notes: "Main portfolio lives in quantum-dev-profile repo.",
    domainUrl: "https://malawein.com",
    brandPageUrl: "/pages/brands/portfolio/",
    githubUrl: "https://github.com/alaweimm90/Portfolio",
    tagline: "Professional Portfolio & Resume Generator",
    gradientColors: ["#38bdf8", "#0284c7"],
  },
  {
    id: "drmalawein",
    name: "DrMalawein",
    tier: "family",
    status: "planned",
    repoPath: undefined,
    appType: "spa",
    tags: ["family", "medical", "planned"],
    notes: "Planned medical practice website.",
    brandPageUrl: "/pages/brands/drmalawein/",
    tagline: "Medical Practice Website",
    gradientColors: ["#22c55e", "#15803d"],
  },

  // ---------------------------------------------------------------------------
  // Family / Personal Projects
  // ---------------------------------------------------------------------------
  {
    id: "rounaq",
    name: "Rounaq",
    tier: "family",
    status: "planned",
    repoPath: undefined,
    appType: "spa",
    tags: ["family", "blog", "planned"],
    notes: "Planned personal blog for family writing.",
    brandPageUrl: "/pages/brands/rounaq/",
    tagline: "Personal Blog & Stories",
    gradientColors: ["#f97316", "#ea580c"],
  },

  // ---------------------------------------------------------------------------
  // Research Projects
  // ---------------------------------------------------------------------------
  {
    id: "maglogic",
    name: "MagLogic",
    tier: "scientific",
    status: "planned",
    repoPath: "research/maglogic",
    appType: "backend",
    tags: ["research", "magnetism", "spintronics"],
    notes: "Magnetic logic circuits research.",
    brandPageUrl: "/pages/brands/maglogic/",
    githubUrl: "https://github.com/alawein-science/mag-logic",
    tagline: "Magnetic Logic Circuits Research",
    gradientColors: ["#22c55e", "#15803d"],
  },
  {
    id: "qmatsim",
    name: "QMatSim",
    tier: "scientific",
    status: "planned",
    repoPath: "research/qmatsim",
    appType: "backend",
    tags: ["research", "quantum", "materials"],
    notes: "Quantum materials simulation research.",
    brandPageUrl: "/pages/brands/qmatsim/",
    githubUrl: "https://github.com/alawein-science/qmat-sim",
    tagline: "Quantum Materials Simulation",
    gradientColors: ["#38bdf8", "#0284c7"],
  },
  {
    id: "qubeml",
    name: "QubeML",
    tier: "ai-ml",
    status: "planned",
    repoPath: "research/qubeml",
    appType: "backend",
    tags: ["research", "quantum", "ml"],
    notes: "Quantum machine learning framework.",
    brandPageUrl: "/pages/brands/qubeml/",
    githubUrl: "https://github.com/alawein-science/QubeML",
    tagline: "Quantum Machine Learning Framework",
    gradientColors: ["#a855f7", "#7c3aed"],
  },
  {
    id: "scicomp",
    name: "SciComp",
    tier: "scientific",
    status: "planned",
    repoPath: "research/scicomp",
    appType: "backend",
    tags: ["research", "scientific", "computing"],
    notes: "Scientific computing utilities.",
    brandPageUrl: "/pages/brands/scicomp/",
    githubUrl: "https://github.com/alawein-science/SciComp",
    tagline: "Scientific Computing Utilities",
    gradientColors: ["#22c55e", "#15803d"],
  },
  {
    id: "spincirc",
    name: "SpinCirc",
    tier: "scientific",
    status: "planned",
    repoPath: "research/spincirc",
    appType: "backend",
    tags: ["research", "spintronics", "circuits"],
    notes: "Spintronics circuit research.",
    brandPageUrl: "/pages/brands/spincirc/",
    githubUrl: "https://github.com/alawein-science/SpinCirc",
    tagline: "Spintronics Circuit Research",
    gradientColors: ["#22c55e", "#15803d"],
  },
  {
    id: "meatheadphysicist",
    name: "MeatheadPhysicist",
    tier: "portfolio",
    status: "active",
    repoPath: undefined,
    appType: "spa",
    tags: ["education", "physics", "content"],
    notes: "Physics education persona and content hub.",
    brandPageUrl: "/pages/brands/meatheadphysicist/",
    tagline: "High-Intensity Physics Education",
    gradientColors: ["#f97316", "#ea580c"],
  },
  {
    id: "helios",
    name: "HELIOS",
    tier: "scientific",
    status: "planned",
    repoPath: "alawein-technologies-llc/helios",
    appType: "backend",
    tags: ["infrastructure", "scientific", "computing"],
    notes: "High-performance scientific computing platform.",
    brandPageUrl: "/pages/brands/helios/",
    githubUrl: "https://github.com/alawein-tools/HELIOS",
    tagline: "High-Performance Scientific Computing",
    gradientColors: ["#f97316", "#ea580c"],
  },
  {
    id: "foundry",
    name: "Foundry",
    tier: "business",
    status: "planned",
    repoPath: "alawein-technologies-llc/foundry",
    appType: "backend",
    tags: ["incubator", "templates", "internal"],
    notes: "Product incubator and template library.",
    brandPageUrl: "/pages/brands/foundry/",
    githubUrl: "https://github.com/alawein-tools/Foundry",
    tagline: "Product Incubator & Template Library",
    gradientColors: ["#22d3ee", "#0891b2"],
  },
];

// Convenience helpers for use in your Studios UI
export const getActivePlatformsByTier = (tier: PlatformTier): PlatformDefinition[] =>
  PLATFORMS.filter((p) => p.tier === tier && p.status === "active");

export const getAllActiveSpaPlatforms = (): PlatformDefinition[] =>
  PLATFORMS.filter((p) => p.status === "active" && p.appType === "spa");
