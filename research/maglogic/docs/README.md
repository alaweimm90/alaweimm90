# MagLogic Documentation

This directory contains the complete documentation for the MagLogic nanomagnetic logic simulation suite.

## Documentation Structure

```
docs/
├── README.md              # This file
├── index.rst              # Main documentation index
├── getting_started/       # Installation and quickstart guides
├── user_guide/           # Comprehensive user documentation  
├── api_reference/        # Complete API documentation
├── examples/             # Example usage and tutorials
├── tutorials/            # Step-by-step tutorials
├── _static/             # Static assets (images, CSS, etc.)
├── _templates/          # Custom Sphinx templates
└── conf.py              # Sphinx configuration
```

## Building Documentation

### Local Build

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e python/[docs]

# Build HTML documentation
cd docs
sphinx-build -b html . _build/html

# Build PDF documentation (requires LaTeX)
sphinx-build -b latex . _build/latex
cd _build/latex && make all-pdf
```

### Docker Build

Using the documentation Docker container:

```bash
# Build documentation image
docker build -f docker/Dockerfile.docs -t maglogic-docs .

# Build documentation
docker run --rm -v $(pwd)/docs-output:/workspace/build maglogic-docs build-docs

# Serve documentation locally
docker run --rm -p 8080:8080 maglogic-docs serve-docs
```

### Live Documentation Server

For development with auto-rebuild:

```bash
# Using Docker
docker run --rm -p 8080:8080 -v $(pwd):/workspace maglogic-docs live-docs

# Or locally
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html --host 0.0.0.0 --port 8080
```

## Documentation Content

### Getting Started
- **Installation**: Package installation via pip, Docker, or source
- **Quickstart**: Basic usage examples and first simulation
- **Configuration**: Setting up OOMMF/MuMax3 environments

### User Guide
- **File Formats**: Understanding OVF, ODT, and MuMax3 formats
- **Simulation Setup**: Creating and running micromagnetic simulations
- **Analysis Workflow**: Processing and analyzing simulation results
- **Visualization**: Creating plots and animations
- **Docker Usage**: Working with containerized environments

### API Reference
Complete documentation of all Python modules:
- `maglogic.parsers`: File parsing utilities
- `maglogic.analysis`: Magnetization analysis tools
- `maglogic.simulation`: Simulation execution and management
- `maglogic.demos`: Demonstration scripts and examples
- `maglogic.core`: Core constants and utilities

### Examples and Tutorials
- **Basic Parsing**: Reading OVF and ODT files
- **Magnetization Analysis**: Domain detection and energy calculations
- **Logic Gates**: NAND/NOR gate simulations
- **Parameter Sweeps**: Automated parameter exploration
- **Advanced Visualization**: Custom plots and interactive displays

## Documentation Standards

### Writing Guidelines
- Use clear, concise language
- Include code examples for all functions
- Provide both conceptual explanations and practical usage
- Cross-reference related sections
- Include figures and diagrams where helpful

### Code Documentation
- All public functions must have docstrings
- Use Google-style docstrings
- Include parameter types and return values
- Provide usage examples
- Document exceptions and edge cases

### Mathematical Notation
- Use LaTeX for mathematical expressions
- Define all variables and symbols
- Include units where applicable
- Reference equations in text

## Contributing to Documentation

### Content Guidelines
1. **Accuracy**: Ensure all information is technically correct
2. **Completeness**: Cover all features and edge cases  
3. **Clarity**: Write for users with varying expertise levels
4. **Examples**: Include working code examples
5. **Updates**: Keep documentation synchronized with code changes

### Review Process
1. Create documentation in appropriate section
2. Build locally to check formatting
3. Review for technical accuracy
4. Test all code examples
5. Submit pull request with documentation changes

### Style Guide
- Use present tense ("MagLogic provides..." not "MagLogic will provide...")
- Use active voice when possible
- Be specific rather than vague
- Include concrete examples
- Use consistent terminology throughout

## Automated Documentation

### API Documentation
API documentation is automatically generated from docstrings using Sphinx autodoc:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Example function demonstrating documentation standards.
    
    This function demonstrates the expected documentation format
    for all MagLogic functions.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter with default value
        
    Returns:
        Boolean indicating success or failure
        
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    if not isinstance(param2, int):
        raise TypeError("param2 must be an integer")
    return True
```

### Notebook Integration
Jupyter notebooks are automatically included in documentation using nbsphinx:

```rst
.. toctree::
   :maxdepth: 2
   
   notebooks/analysis_tutorial
   notebooks/simulation_example
```

## Publication and Deployment

### GitHub Pages
Documentation is automatically deployed to GitHub Pages via GitHub Actions:
- Triggered on pushes to main branch
- Builds with latest Sphinx version
- Deployed to: https://alaweimm90.github.io/MagLogic/

### Custom Domain
Documentation is available at: https://maglogic.berkeley.edu

### Versioning
Documentation versions correspond to software releases:
- `latest/`: Current development version
- `v1.0/`: Release version 1.0
- `stable/`: Latest stable release

## Troubleshooting

### Common Issues

**Build Failures**
```bash
# Clear build cache
rm -rf _build/

# Reinstall dependencies
pip install -e python/[docs] --upgrade

# Check for syntax errors
sphinx-build -W -b html . _build/html
```

**Missing References**
```bash
# Check for broken links
sphinx-build -b linkcheck . _build/linkcheck

# Update intersphinx mapping
# Edit conf.py intersphinx_mapping
```

**LaTeX/PDF Issues**
```bash
# Install LaTeX dependencies
sudo apt-get install texlive-latex-extra latexmk

# Check LaTeX log
cd _build/latex && cat MagLogic.log
```

### Getting Help
- Check the [troubleshooting guide](troubleshooting.rst)  
- Search existing [issues](https://github.com/alaweimm90/MagLogic/issues)
- Ask questions in [discussions](https://github.com/alaweimm90/MagLogic/discussions)

## License

This documentation is licensed under the same MIT License as the MagLogic software.

---

**MagLogic Documentation**  
UC Berkeley Nanomagnetic Logic Research  
Meshal Alawein