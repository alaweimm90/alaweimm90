# HELIOS v0.1.0 Quick Start Guide

**Get HELIOS running in 5 minutes.**

---

## 🚀 Installation (1 minute)

### Option 1: pip (Recommended)
```bash
pip install helios
```

### Option 2: Docker
```bash
docker pull ghcr.io/alaweinoss/helios:v0.1.0
docker run -it ghcr.io/alaweinoss/helios:v0.1.0
```

### Option 3: From Source
```bash
git clone https://github.com/alaweinoss/helios.git
cd helios
pip install -e .
```

---

## ✅ Verify Installation (30 seconds)

```bash
# Python API
python -c "import helios; print(f'HELIOS {helios.__version__} ready!')"

# CLI
helios --help
helios domain list
```

---

## 📝 Generate Hypotheses (2 minutes)

### Via CLI
```bash
helios generate "quantum error correction methods"
```

### Via Python
```python
from helios import HypothesisGenerator

generator = HypothesisGenerator()
hypotheses = generator.generate(
    topic="Quantum error correction",
    domain="quantum",
    num_hypotheses=3
)

for h in hypotheses:
    print(h['text'])
```

---

## ✔️ Validate Hypotheses (2 minutes)

```python
from helios import TuringValidator

validator = TuringValidator()
results = validator.validate(hypotheses)

for r in results:
    print(f"{r['hypothesis']}: {r['quality']}%")
```

---

## 🔍 Learn from Results (1 minute)

```python
from helios import MetaLearner

learner = MetaLearner()
learner.learn_from_validation(
    hypotheses,
    results,
    domain="quantum"
)

# Get agent recommendation for next run
best_agent = learner.recommend_agent(
    topic="Quantum error correction",
    domain="quantum"
)
print(f"Recommended agent: {best_agent}")
```

---

## 📚 Find More Information

- **Getting Started**: `helios/docs/GETTING_STARTED.md`
- **API Reference**: `helios/docs/API.md`
- **Architecture**: `helios/docs/ARCHITECTURE.md`
- **Domains**: `helios/docs/DOMAINS.md`
- **Examples**: `helios/examples/`

---

## 💡 Common Tasks

### Run Full Research Workflow
```bash
helios research "drug binding affinity prediction" --domain synthesis
```

### List Available Domains
```bash
helios domain list
```

### Get Domain Information
```bash
helios domain info ml
```

### Run Tests
```bash
cd helios
bash scripts/test.sh
```

### Format Code
```bash
bash helios/scripts/format.sh
```

---

## 🆘 Troubleshooting

### Import Error
```bash
# Reinstall
pip install --upgrade helios

# Check installation
python -c "import helios; print(helios.__file__)"
```

### CLI Not Found
```bash
# Ensure pip bin is in PATH
which helios

# Or use python module
python -m helios.cli --help
```

### Missing Domain
```bash
# Install with specific domain
pip install helios[quantum]

# Or all domains
pip install helios[all]
```

---

## 🎯 Next Steps

1. ✅ **Install**: Done!
2. 📝 **Generate**: Try `helios generate "your topic"`
3. ✔️ **Validate**: Test your hypotheses
4. 🧠 **Learn**: Extract lessons from failures
5. 🚀 **Scale**: Deploy to production (see DEPLOYMENT.md)

---

## 📞 Get Help

- **Documentation**: See repository docs/
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: helios-team@example.com

---

**Ready to discover?** 🎉
