# Migration
## Run Consolidation
```powershell
pwsh .\.tools\consolidate.ps1
```
- Creates `.tools`, `.config`, `.cache` if missing.
- Moves selected hidden directories and creates compatibility links.
- Writes `.cache/consolidation-manifest.json`.
- Copies backups to `.cache/backups-<timestamp>/` prior to moving.
## Validate Structure
```powershell
pwsh .\.tools\validate-structure.ps1
```
- Outputs JSON with `compliance`, `checks`, and `extraHidden`.
## Rollback
```powershell
pwsh .\.tools\consolidate.ps1 -Rollback
```
- Restores original locations using the manifest.
- Removes compatibility links.
## Notes
- Windows creates directory junctions if symlinks are unavailable.
- `.github` remains at root to avoid CI disruption.
## Dry-Run and Custom Map
- Preview changes without modifying the repository:
```powershell
pwsh .\.tools\consolidate.ps1 -DryRun | ConvertFrom-Json
```
- Customize consolidation using `/.config/consolidation-map.json`:
```json
[{ "old": ".automation", "new": ".tools/automation" }]
```
- Provide a custom map path:
```powershell
pwsh .\.tools\consolidate.ps1 -MapPath .\.config\my-map.json
```
