# Projects Management — Quick Reference

**TL;DR for day-to-day use of the project management system.**

---

## What is a Project?

A **project** is a directory in `organizations/<domain>/` with code/docs but **no dedicated GitHub repo yet**.

- Example: `organizations/alaweimm90-science/qmat-sim/`
- Must have: `.project.yaml` manifest
- Can become: A full GitHub repository (via promotion)

---

## Project Manifest (`.project.yaml`)

Every project needs a manifest. Minimal example:

```yaml
name: my-project
title: "My Project Title"
domain: science  # or tools, business, etc.
status: active   # or idea, planned, frozen, archived
type: library    # or service, research, demo, doc-only
language: python # or ts, mixed, na
priority: P1     # or P0, P2
owner: alaweimm90
contacts:
  - "github:alaweimm90"
description: |
  One paragraph about what this project does.

inputs:
  code_paths: []
  data_paths: []

links:
  github_project: ""
  tracking_docs: []

promotion:
  target_repo_name: ""
  template: ""
  visibility: public
  reason: ""
```

**Required fields:** name, title, domain, status, type, language, priority, owner, description

**Valid enums:**
- **domain:** science, tools, platforms, research, infra, business, misc
- **status:** idea, planned, active, frozen, archived
- **type:** doc-only, prototype, library, service, research-bundle, demo, monorepo
- **language:** python, ts, mixed, na
- **priority:** P0, P1, P2

---

## Common Commands

### Scan All Projects
```bash
python metaHub/cli/meta.py scan-projects
```
- Discovers all `.project.yaml` files
- Validates manifests
- Generates `organizations/.projects.json` (machine-readable)
- Generates `organizations/PROJECTS_INDEX.md` (human-readable table)
- **Exit code:** 0 if valid, 1 if errors

### Promote a Project to a Repo
```bash
python metaHub/cli/meta.py promote-project <domain>/<project-name>
```
- Scaffolds new repo from template (python-lib, ts-lib, research, monorepo)
- Creates `.meta/repo.yaml` with metadata
- Initializes git repo
- Prints next steps (no auto-push)

**Example:**
```bash
python metaHub/cli/meta.py promote-project alaweimm90-science/qmat-sim
```

Creates: `../qmat-sim/` ready to push to GitHub.

### Check Project Status
```bash
cat organizations/PROJECTS_INDEX.md
```
- Human-readable table of all projects
- Shows validation status (✓ or ❌)

Or:
```bash
cat organizations/.projects.json | jq .
```
Machine-readable project list.

---

## Creating a New Project

1. **Create directory:**
   ```bash
   mkdir organizations/science/my-new-project
   cd organizations/science/my-new-project
   ```

2. **Add code/docs** (optional at this stage)

3. **Create `.project.yaml`:**
   ```yaml
   name: my-new-project
   title: "My New Project"
   domain: science
   status: idea
   type: library
   language: python
   priority: P1
   owner: alaweimm90
   contacts: ["github:you"]
   description: |
     What I'm working on...

   inputs:
     code_paths: [src/]
     data_paths: [data/]

   links:
     github_project: ""
     tracking_docs: [README.md]

   promotion:
     target_repo_name: ""
     template: ""
     visibility: public
     reason: ""
   ```

4. **Validate locally:**
   ```bash
   python metaHub/cli/meta.py scan-projects
   ```
   Should show your project as ✓ (valid)

5. **Commit & push:**
   ```bash
   git add organizations/science/my-new-project/.project.yaml
   git commit -m "chore(projects): add my-new-project"
   git push
   ```

CI validates automatically.

---

## Promoting a Project to a Repo

When a project is **stable and ready for the world:**

### 1. Update Manifest
```yaml
promotion:
  target_repo_name: "my-new-repo"
  template: "python-lib"  # or ts-lib, research, monorepo
  visibility: public      # or private
  reason: "Library is stable with good API coverage"
```

**Valid templates:**
- `python-lib` — Python library with pyproject.toml, tests, type checking
- `ts-lib` — TypeScript library with package.json, vitest, pnpm
- `research` — Notebooks + code with Jupyter, uv, data LFS
- `monorepo` — Multi-package with turbo/pnpm or uv workspace

### 2. Run Promotion
```bash
python metaHub/cli/meta.py promote-project science/my-project
```

Output:
```
📦 Scaffolding my-new-repo from template 'python-lib'...
🏷️  Creating .meta/repo.yaml...
🔧 Initializing git repository...

✅ Promotion complete!

📍 Repo created: ../my-new-repo

Next steps:
  1. Review files in my-new-repo/
  2. Update README.md and SECURITY.md as needed
  3. Create the repo on GitHub: gh repo create --source=my-new-repo
  4. Push to GitHub: cd my-new-repo && git push -u origin main
```

### 3. Review & Push
```bash
cd ../my-new-repo
git log --oneline        # Verify clean history
cat .meta/repo.yaml      # Check metadata
cat README.md            # Review structure

# Create repo on GitHub and push
gh repo create --source=. --remote=origin --public
git push -u origin main
```

✅ Done! New repo is fully governance-compliant.

### 4. Archive Project (Optional)
```bash
# In organizations/science/my-project/.project.yaml
# Option A: Delete promotion config
promotion:
  target_repo_name: ""   # Clear this
  template: ""
  visibility: public
  reason: ""

# Option B: Mark as archived
status: archived
```

Commit and push. The original project record stays for history.

---

## Troubleshooting

### "No projects found"
- Check that you have `.project.yaml` files in `organizations/` subdirectories
- Manifests must be at leaf-level directories (not in intermediate folders)

### "Invalid project error"
- Run `meta scan-projects` to see detailed error messages
- Check `.project.yaml` YAML syntax
- Verify all required fields are non-empty
- Verify enum values match schema (domain, status, type, language, priority)

### "Duplicate project name"
- Project names must be unique across all domains
- Rename project or move to different domain folder

### "Template not found"
- Ensure template is one of: `python-lib`, `ts-lib`, `research`, `monorepo`
- Verify `metaHub/templates/<name>/` exists

### "Promotion target already exists"
- The destination directory already exists
- Either delete/rename existing directory or choose different target name

### "CI validation failed"
- GitHub Actions ran `organizations-policy.yml` and found errors
- Fix manifest and push again
- Details available in PR checks or commit status

---

## Files to Know

| File | Purpose |
|------|---------|
| `metaHub/cli/meta.py` | CLI tool (scan-projects, promote-project) |
| `organizations/README.md` | Complete project management guide |
| `organizations/.projects.json` | Auto-generated project inventory |
| `organizations/PROJECTS_INDEX.md` | Auto-generated project table |
| `organizations/.ignore` | Whitelist non-project directories |
| `metaHub/policies/organizations_policy.rego` | OPA validation rules |
| `.github/workflows/organizations-policy.yml` | CI validation workflow |
| `FINAL_ARCHITECTURE.md` | Full system design |

---

## Tips & Best Practices

✅ **Do:**
- Create `.project.yaml` early (even for prototypes)
- Use descriptive names and titles
- Keep descriptions to 1 paragraph
- Update status as project evolves
- Mark projects `archived` when done instead of deleting

❌ **Don't:**
- Leave promotion config empty if not promoting (leave fields blank, not null)
- Use spaces in project names (use kebab-case)
- Duplicate project names across domains
- Commit `.projects.json` manually (let CI auto-update)

---

## Integration with Inventory

Projects are automatically merged into root `inventory.json`:
- Run `python metaHub/cli/meta.py scan-projects` to generate `organizations/.projects.json`
- This file can be merged with root inventory for unified portfolio view
- Enables: single dashboard, cross-domain reporting, migration planning

---

## Cheat Sheet

```bash
# Quick commands
python metaHub/cli/meta.py scan-projects              # Validate all
python metaHub/cli/meta.py promote-project D/P        # Promote to repo
cat organizations/PROJECTS_INDEX.md                   # View all projects
grep -r "target_repo_name:" organizations/            # Find promotable projects
python metaHub/cli/meta.py scan-projects && echo OK   # Dry run
```

---

**For detailed guide:** Read [organizations/README.md](organizations/README.md)

**For technical details:** Read [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md#project-management-system)

