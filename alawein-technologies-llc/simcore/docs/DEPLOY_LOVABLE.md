# Deploy to Lovable (with Custom Domain)

This project is Vite + React. Lovable builds from GitHub and serves the static `dist/` output.

## One-time setup

1. Push repo to GitHub (already at): `https://github.com/alaweimm90/SimCore`
2. In Lovable, import project from GitHub and select this repo.
3. Build settings:
   - Build command: `npm run build`
   - Output directory: `dist`
   - SPA routing: enabled (serve `index.html` for client routes)
4. Add your domain in Lovable → follow DNS instructions (usually a CNAME to the Lovable subdomain). HTTPS is automatic.

## Daily workflow (shortcut)

- Make changes → save → run:

```bash
npm run deploy:lovable
```

This will build and push `main`; Lovable will auto-redeploy.

## Notes

- Dev server: `npm run dev` → http://localhost:8080
- If deploying to a subpath, set `base` in `vite.config.ts` before building:

```ts
export default defineConfig(({ mode }) => ({
  base: '/app/',
  // ...existing config...
}));
```

- SPA fallback for static hosts is added via `public/_redirects`.

## Links

- Lovable project: https://lovable.dev/projects/f5c4348e-aff9-4eb4-bf03-0c2e06c7822a
- GitHub repo: https://github.com/alaweimm90/SimCore
