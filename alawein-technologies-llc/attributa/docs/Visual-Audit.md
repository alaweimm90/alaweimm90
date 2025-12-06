# Attributa.dev Visual Audit Report Template

Scope: Routes /, /scan, /results, /workspace, /settings, /documentation, 404
Persona: UX/UI Designer

1. Brand and Tokens
- Verify usage of semantic Tailwind tokens defined in index.css and tailwind.config.ts.
- Ensure shadcn component variants (Button, Card, Tabs, Tooltip, Dialog) align to tokens; avoid ad-hoc colors.
- Iconography: consistent stroke weights and sizes.

2. Typography & Readability
- Hierarchy: h1 one per page; clear h2/h3 nesting.
- Scales: responsive type ramp; ideal line length 60–80ch; line-height 1.4–1.6.
- Contrast: meet WCAG AA/AAA where feasible.

3. Layout & Information Hierarchy
- Primary actions (Scan, Analyze, Export) visually prominent.
- Density: cards and lists with adequate spacing; clear empty states.
- Progressive disclosure for advanced settings.

4. Theming & States
- Light/Dark parity: backgrounds, surfaces, borders, shadows.
- Focus, hover, active, disabled states visible and consistent.
- Motion: respect prefers-reduced-motion.

5. Accessibility
- Keyboard navigation order; visible focus rings.
- ARIA for interactive components (menus, dialogs, tabs).
- Color contrast ratios documented; text alternatives for images.

6. Performance Perception
- Skeletons and spinners: consistent; avoid layout shifts.
- Lazy load non-critical images; defer heavy scripts.

7. Findings (per page)
- Page: /
  - Observations:
  - Issues:
  - Recommendations:
- Page: /scan
- Page: /results (Overview, Segments, Citations, Code)
- Page: /workspace
- Page: /settings
- Page: /documentation
- Page: 404

8. Prioritized Recommendations
- Quick wins (1–2 days):
- Structural changes (1–2 weeks):
- Component variant updates (centralized in tokens and ui components):

9. Acceptance Criteria
- All text meets contrast ratios.
- Single H1 per page; SEO meta set via useSEO hook.
- No direct colors; only semantic tokens used.
- Keyboard-only navigation across critical flows.

[Placeholder: Annotated screenshots]
