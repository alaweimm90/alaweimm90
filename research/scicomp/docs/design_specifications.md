# SciComp Visual Design Specifications
## Logo Design Concept
### Primary Logo
**Style**: Modern, academic, scientific
**Dimensions**: Scalable vector format (SVG preferred)
**Color Scheme**: Berkeley Blue (#003262) and California Gold (#FDB515)
**Logo Elements**:
1. **Main Text**: "SciComp" in clean, modern sans-serif font (like Helvetica or Open Sans)
2. **Atomic Symbol Integration**: Stylized atom symbol replacing the "o" in "SciComp"
   - Nucleus: Small circle in California Gold
   - Electron orbits: Two intersecting ellipses in Berkeley Blue
3. **Tagline**: "Cross-Platform Scientific Computing" in smaller text below
4. **UC Berkeley Integration**: Small "UC Berkeley" text or bear silhouette
**Layout Options**:
- **Horizontal**: Logo + tagline side by side
- **Vertical**: Logo above tagline
- **Icon Only**: Just the stylized atomic symbol for favicon/social media
### Color Variations
- **Primary**: Berkeley Blue text + California Gold accents
- **Monochrome**: Single color versions (white, black, Berkeley Blue)
- **Reversed**: Light versions for dark backgrounds
## Overview Picture Design Concept
### Main Framework Visualization
**Dimensions**: 800x600px (4:3 ratio) for documentation
**Style**: Clean, professional, scientific diagram
**Background**: Light gradient from white to pale Berkeley Blue
### Layout Structure (Left to Right):
#### 1. Input Layer (Left Side - 20%)
**Platform Icons**:
- Python logo with "Python 3.8+"
- MATLAB logo with "MATLAB R2021+"
- Mathematica logo with "Mathematica 12+"
- Terminal icon for "CLI Interface"
#### 2. Core SciComp Framework (Center - 60%)
**Central Hub Design**:
- Large hexagonal shape in Berkeley Blue outline
- "SciComp" logo at center
- Six connected modules radiating outward:
**Top Row**:
- **Quantum Physics**: Atom symbol + quantum state visualization
- **GPU Acceleration**: Graphics card icon + speedometer showing "55.7 GFLOPS"
**Middle Row**:
- **ML Physics**: Neural network diagram + physics equations
- **Thermal Transport**: Heat map visualization + temperature gradient
**Bottom Row**:
- **Signal Processing**: Waveform + FFT visualization
- **Cross-Platform**: Interconnected platform symbols
#### 3. Output Layer (Right Side - 20%)
**Applications**:
- Research papers icon
- GPU cluster visualization
- Interactive plots/graphs
- Publication-ready figures
### Visual Connections
- **Flow Arrows**: Berkeley Blue arrows showing data flow
- **Golden Ratio Grid**: Subtle California Gold grid lines
- **UC Berkeley Branding**: Small bear silhouette or "UC Berkeley" text
### Typography
- **Headers**: Bold, modern sans-serif in Berkeley Blue
- **Subtext**: Regular weight in dark gray
- **Metrics**: California Gold highlighting for performance numbers
## Technical Implementation Suggestions
### For Logo Creation:
1. **Adobe Illustrator**: Professional vector logo design
2. **Figma**: Free browser-based design tool
3. **Canva**: Template-based logo maker
4. **GIMP**: Free alternative with vector plugins
### For Overview Picture:
1. **Adobe Illustrator/Photoshop**: Professional diagrams
2. **Figma**: Collaborative design platform
3. **Lucidchart**: Scientific diagram specialization
4. **Draw.io**: Free flowchart and diagram tool
5. **Python + Matplotlib**: Programmatic generation with scientific styling
### Code-Generated Version (Python):
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
# Berkeley colors
berkeley_blue = '#003262'
california_gold = '#FDB515'
# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
# Central SciComp hexagon
hexagon = patches.RegularPolygon((5, 4), 6, radius=2,
                                facecolor='lightblue',
                                edgecolor=berkeley_blue, linewidth=3)
ax.add_patch(hexagon)
# Add SciComp text at center
ax.text(5, 4, 'SciComp', fontsize=20, fontweight='bold',
        ha='center', va='center', color=berkeley_blue)
# Add module boxes around hexagon
modules = [
    ('Quantum\nPhysics', 5, 6.5),
    ('GPU\nAcceleration', 7.5, 5.5),
    ('ML\nPhysics', 7.5, 2.5),
    ('Thermal\nTransport', 5, 1.5),
    ('Signal\nProcessing', 2.5, 2.5),
    ('Cross-\nPlatform', 2.5, 5.5)
]
for name, x, y in modules:
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.1",
                         facecolor=california_gold,
                         edgecolor=berkeley_blue)
    ax.add_patch(box)
    ax.text(x, y, name, fontsize=10, ha='center', va='center',
            color=berkeley_blue, fontweight='bold')
# Add platform inputs (left side)
platforms = ['Python', 'MATLAB', 'Mathematica', 'CLI']
for i, platform in enumerate(platforms):
    ax.text(1, 6-i*1.5, platform, fontsize=12,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
# Add output applications (right side)
outputs = ['Research', 'GPU Clusters', 'Publications', 'Interactive']
for i, output in enumerate(outputs):
    ax.text(9, 6-i*1.5, output, fontsize=12,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
plt.title('SciComp: Cross-Platform Scientific Computing Suite',
          fontsize=16, fontweight='bold', color=berkeley_blue, pad=20)
plt.tight_layout()
plt.savefig('scicomp_overview.png', dpi=300, bbox_inches='tight')
plt.show()
```
## Brand Guidelines
### Do's:
- Use Berkeley Blue and California Gold as primary colors
- Maintain clean, professional appearance
- Include UC Berkeley attribution
- Use modern, readable fonts
- Ensure scalability across sizes
### Don'ts:
- Don't use Comic Sans or overly decorative fonts
- Avoid cluttered designs
- Don't use colors that conflict with UC Berkeley branding
- Avoid low-resolution or pixelated elements
- Don't obscure readability with excessive effects
## File Naming Convention
- Logo files: `scicomp_logo_[variation].[format]`
- Overview: `scicomp_overview.[format]`
- Icons: `scicomp_icon_[size].[format]`
## Recommended Formats
- **Logo**: SVG (primary), PNG (web), EPS (print)
- **Overview**: PNG (documentation), PDF (presentations), SVG (web)
- **Social Media**: PNG at platform-specific dimensions