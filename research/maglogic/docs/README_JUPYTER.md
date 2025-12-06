---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
execution:
  execute_notebooks: force
---

# MagLogic: Nanomagnetic Logic Simulation Suite

```{admonition} Interactive Documentation
:class: tip
This Jupyter Book version includes executable code cells and interactive demonstrations.
```

## Live Demo

```{code-cell} python
:tags: [hide-input]

# Cell metadata for executable book
import matplotlib.pyplot as plt
import numpy as np

# Quick demonstration
print("MagLogic Framework - Live Demo")
print("=" * 40)

# Simulation parameters
angles = np.linspace(0, 360, 100)
magnetization = np.cos(np.radians(angles))

plt.figure(figsize=(8, 4))
plt.plot(angles, magnetization, 'b-', linewidth=2, label='Magnetization')
plt.xlabel('Field Angle (degrees)')
plt.ylabel('Normalized Magnetization')
plt.title('Sample Magnetization Dynamics')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

## Interactive Modules

```{code-cell} python
:tags: [remove-output]

# Example simulation setup (executable)
try:
    from maglogic.demos import demo_nand_nor
    print("✓ MagLogic modules loaded successfully")
    
    # Interactive parameter exploration
    def explore_logic_gate(clock_angle=60, input_a=1, input_b=1):
        """Interactive logic gate exploration"""
        result = demo_nand_nor.run_simulation(
            clock_angle=clock_angle, 
            input_A=input_a, 
            input_B=input_b
        )
        return f"Gate output: {result.get('logic_output', 0)}"
    
    print("Interactive demo ready!")
    
except ImportError:
    print("⚠ Install MagLogic to run interactive examples")
```

## Book Structure

```{tableofcontents}
```

## Execution Environment

```{code-cell} python
:tags: [hide-input, remove-output]

# Cell execution metadata
import sys
print(f"Python version: {sys.version}")
print(f"Execution environment configured")
```

---

**Author**: Meshal Alawein  
**Institution**: University of California, Berkeley  
**Interactive Book Version**: 2025

```{note}
This Jupyter Book version provides executable examples and interactive demonstrations of the MagLogic framework.
```