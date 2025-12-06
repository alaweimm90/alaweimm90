# Resume Tailor (hidden)

This module has been temporarily disabled from the UI and routing.

- Location: This folder is intentionally placed under `src/_hidden/resumeTailor` to avoid accidental imports.
- Status: No authentication required; designed as a free, global tool when re-enabled.
- To reintegrate:
  1. Restore routes in `src/App.tsx` for `/resume`, `/resume/workspace`, and `/resume/print/:docId`.
  2. Add nav links back in `src/components/Layout.tsx` and `src/components/MobileNav.tsx`.
  3. Optionally expose module settings via `src/pages/Settings.tsx`.

Notes:
- The module is local-first (WASM/WebGPU) and does not depend on user accounts.
- Keep external API keys strictly opt-in (BYO keys), stored client-side unless you wire Supabase.
