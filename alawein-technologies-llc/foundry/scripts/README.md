# Scripts - Automation & Scaffolding

Automation scripts for scaffolding new product repositories from templates.

---

## Overview

These scripts automate the process of creating new product repositories from templates, saving time and ensuring consistency.

---

## Scripts

### create-repo.sh

**Purpose:** Create a single new product repository

**Usage:**
```bash
./create-repo.sh <template-name> <new-repo-name>
```

**Examples:**

```bash
# Create a Nightmare Mode repository
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator

# Create a Research Prison repository
./create-repo.sh research-prison ORCHEX-interrogator

# Create a Chaos Engine repository
./create-repo.sh chaos-engine ORCHEX-chaos-engine

# Create an ORCHEX Core infrastructure repository
./create-repo.sh ORCHEX-core ORCHEX-core-infra
```

**What it does:**
1. ✅ Copies template directory
2. ✅ Renames project files
3. ✅ Updates configuration files
4. ✅ Initializes git repository
5. ✅ Sets up Python virtual environment
6. ✅ Installs dependencies
7. ✅ Creates initial migration
8. ✅ Displays next steps

**Output:**
```
Created: ../ORCHEX-nightmare-validator/
├── README.md (updated with your repo name)
├── requirements.txt (ready to use)
├── backend/ (Python FastAPI app)
├── alembic/ (database migrations)
├── k8s/ (Kubernetes configs)
├── .github/workflows/ (CI/CD)
└── [all other files]
```

**Next steps after running:**
```bash
cd ../ORCHEX-nightmare-validator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

---

### create-all-repos.sh

**Purpose:** Create all 4 product repositories at once

**Usage:**
```bash
./create-all-repos.sh
```

**What it creates:**
- `ORCHEX-nightmare-validator/` (Nightmare Mode template)
- `ORCHEX-chaos-engine/` (Chaos Engine template)
- `ORCHEX-interrogator/` (Research Prison template)
- `ORCHEX-core-infra/` (ORCHEX Core template)

**Customization:**
Edit the script to change repository names:
```bash
# Edit these lines in create-all-repos.sh
REPO_NAME_1="ORCHEX-nightmare-validator"
REPO_NAME_2="ORCHEX-chaos-engine"
REPO_NAME_3="ORCHEX-interrogator"
REPO_NAME_4="ORCHEX-core-infra"
```

**Time required:** ~5 minutes to create all 4 repos

**Disk space required:** ~500 MB

---

## Available Templates

| Template | Product | Files | Tech |
|----------|---------|-------|------|
| `nightmare-mode` | AI Paper Attacker | 6 files | Python, FastAPI, PostgreSQL |
| `chaos-engine` | Hypothesis Generator | 2 files | Python, FastAPI, PostgreSQL |
| `research-prison` | Research Interrogator | 2 files | Python, FastAPI, PostgreSQL |
| `ORCHEX-core` | Shared Infrastructure | 1 file | Python, FastAPI, PostgreSQL |

---

## Template Structure

Each template includes:

```
template-name/
├── README.md                       # Product overview
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── .env.example                    # Environment variables
├── Dockerfile                      # Docker image
├── docker-compose.yml              # Local dev stack
├── pytest.ini                      # Test config
├── backend/                        # Python FastAPI app
│   ├── app/
│   │   ├── main.py                # Entry point
│   │   ├── models.py              # DB models
│   │   ├── schemas.py             # Request schemas
│   │   └── routers/               # API routes
│   └── tests/
│       └── test_api.py            # Tests
├── alembic/                        # Database migrations
│   └── versions/
│       └── 001_initial.py         # Initial schema
├── k8s/                            # Kubernetes configs
│   └── production.yaml
└── .github/
    └── workflows/
        └── ci-cd.yml              # GitHub Actions
```

---

## Prerequisites

### For Running Scripts

- ✅ Bash shell (Linux/Mac) or Git Bash (Windows)
- ✅ Git installed
- ✅ Able to execute shell scripts

### For Resulting Repositories

- ✅ Python 3.10+
- ✅ PostgreSQL 15+
- ✅ Docker & Docker Compose (optional but recommended)

---

## Step-by-Step Guide

### Step 1: Navigate to Scripts Directory

```bash
cd /home/user/Foundry/scripts
```

### Step 2: Choose Template

Decide which product you want to create:
- `nightmare-mode` - AI paper attacker (recommended)
- `chaos-engine` - Hypothesis generator
- `research-prison` - Research interrogator (fastest)
- `ORCHEX-core` - Shared infrastructure

### Step 3: Run Script

```bash
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator
```

### Step 4: Wait for Completion

Script takes 2-5 minutes depending on internet speed.

### Step 5: Navigate to New Repo

```bash
cd ../ORCHEX-nightmare-validator
```

### Step 6: Follow Next Steps

Print messages from script will tell you next steps:

```bash
# Activate virtual environment
source venv/bin/activate

# Run development server
uvicorn app.main:app --reload

# In another terminal, run migrations
alembic upgrade head

# Or use Docker
docker-compose up
```

---

## Troubleshooting

### Script won't execute
```bash
# Make executable
chmod +x create-repo.sh
./create-repo.sh nightmare-mode ORCHEX-test
```

### Permission denied
```bash
# Give execute permissions
chmod +x scripts/*.sh
```

### Repository already exists
```bash
# Remove existing repo
rm -rf ../ORCHEX-nightmare-validator

# Run script again
./create-repo.sh nightmare-mode ORCHEX-nightmare-validator
```

### Python venv issues
```bash
# Use Python 3.10+
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database connection error
```bash
# Start PostgreSQL
# Mac: brew services start postgresql
# Linux: sudo systemctl start postgresql
# Or use Docker: docker-compose up -d postgres
```

---

## Customization

### Changing Repository Names

Edit the script before running:

```bash
# Open create-repo.sh
nano create-repo.sh

# Find this line:
./create-repo.sh <template-name> <your-custom-name>

# Run with custom name:
./create-repo.sh nightmare-mode my-custom-api
```

### Adding Custom Configuration

After creating repository:

```bash
cd ../my-custom-api

# Edit environment
cp .env.example .env
nano .env

# Update config values:
# DATABASE_URL=postgresql://user:password@localhost/dbname
# API_KEY=your_api_key_here
```

### Extending Templates

To create custom templates:

1. Copy an existing template
2. Modify it for your needs
3. Test with `create-repo.sh`
4. Share back (see CONTRIBUTING.md)

---

## Advanced Usage

### Creating with Specific Variables

Create `.env` file before running:

```bash
# Create config
cat > .env.template << EOF
REPO_NAME=ORCHEX-custom
PRODUCT_NAME=My Custom Product
GITHUB_OWNER=your-username
ENVIRONMENT=staging
EOF

# Reference in script
source .env.template
./create-repo.sh nightmare-mode $REPO_NAME
```

### Batch Creating Multiple Repos

```bash
# Create list of repos
REPOS=(
  "ORCHEX-validator"
  "ORCHEX-generator"
  "ORCHEX-interrogator"
)

# Loop through and create
for repo in "${REPOS[@]}"; do
  ./create-repo.sh nightmare-mode $repo
  echo "Created: $repo"
done
```

### Integration with CI/CD

Automated repository creation in deployment pipeline:

```yaml
# .github/workflows/scaffold-repo.yml
name: Scaffold Repository
on: [workflow_dispatch]

jobs:
  scaffold:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Create repository
        run: |
          cd scripts
          ./create-repo.sh nightmare-mode ORCHEX-prod-${{ github.run_number }}
```

---

## Best Practices

✅ **Do:**
- Run script outside the Foundry folder
- Test created repos before deploying
- Document any customizations
- Keep templates updated
- Back up before running on production

❌ **Don't:**
- Run script from inside template directories
- Commit .env files (use .env.example)
- Modify templates without testing
- Run multiple scripts simultaneously
- Mix local and cloud repositories

---

## Security Considerations

### Sensitive Information

- ❌ Don't commit API keys
- ❌ Don't commit database passwords
- ❌ Don't commit .env files
- ✅ Do use environment variables
- ✅ Do use secrets manager

### .gitignore

Created repos include `.gitignore` with:
```
.env
.env.local
.env.*.local
__pycache__/
*.egg-info/
.venv/
.pytest_cache/
*.pyc
```

---

## Next Steps

1. **Run a script:** `./create-repo.sh nightmare-mode ORCHEX-test`
2. **Navigate to repo:** `cd ../ORCHEX-test`
3. **Follow setup:** See created `README.md`
4. **Start development:** `docker-compose up` or `uvicorn...`
5. **See guide:** `../docs/guides/GETTING_STARTED.md`

---

## Related Documentation

- `../START_HERE.md` - Master index
- `../IMPLEMENTATION_GUIDE.md` - Build guide
- `../products/README.md` - Products overview
- `../docs/guides/GETTING_STARTED.md` - Dev setup

---

## Contributing

See `../CONTRIBUTING.md` for:
- How to improve scripts
- How to create new templates
- Submission process

---

**Ready to create a repository? Run `./create-repo.sh nightmare-mode your-repo-name`** 🚀

*Questions? See `../FAQ.md`* 📚
