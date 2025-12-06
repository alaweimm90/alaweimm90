# Deploy and Run on Replit

This project is Vite + React. Replit will run the production build on its assigned `$PORT` using the provided `.replit` and `start` script.

## Your Replit
- https://replit.com/@alaweimm90/SimCore

## One-time setup
1. In Replit, create a Repl from GitHub and select `alaweimm90/SimCore`.
2. Replit will install dependencies automatically.
3. The `.replit` file uses `npm run start`, which builds and serves `dist/` on `$PORT`.

## Daily workflow (shortcut)
- Develop locally, then push to GitHub:
  ```bash
  npm run deploy:replit
  ```
- In Replit, click “Pull from GitHub” (or enable Git auto-pull) and click Run.

## Notes
- Local dev: `npm run dev` → http://localhost:8080
- Replit run: `npm run start` (builds then serves `dist/` on `$PORT`).
- Custom domain: Replit → Tools → Domains → add your domain and follow CNAME steps (HTTPS auto).
