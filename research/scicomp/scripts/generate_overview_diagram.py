#!/usr/bin/env python3
"""
SciComp Overview Diagram Generator
Generates the framework overview diagram programmatically using matplotlib.
Creates a professional visualization showing the SciComp architecture.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
from pathlib import Path
# Berkeley color scheme
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
LIGHT_BLUE = '#E6F3FF'
LIGHT_GOLD = '#FFF4E6'
def create_scicomp_overview():
    """Create the main SciComp framework overview diagram."""
    # Create figure with high DPI for crisp output
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    # Background gradient effect
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, extent=[0, 14, 0, 10], aspect='auto',
              cmap='Blues', alpha=0.1, zorder=0)
    # Title
    ax.text(7, 9.2, 'SciComp: Cross-Platform Scientific Computing Suite',
            fontsize=22, fontweight='bold', ha='center', va='center',
            color=BERKELEY_BLUE)
    ax.text(7, 8.7, 'University of California, Berkeley',
            fontsize=14, ha='center', va='center',
            color=CALIFORNIA_GOLD, style='italic')
    # Central hexagonal framework
    hexagon = patches.RegularPolygon((7, 5), 6, radius=2.5,
                                   facecolor=LIGHT_BLUE,
                                   edgecolor=BERKELEY_BLUE,
                                   linewidth=4, zorder=2)
    ax.add_patch(hexagon)
    # SciComp logo at center
    ax.text(7, 5.3, 'SciComp', fontsize=24, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    ax.text(7, 4.7, 'Framework', fontsize=14, ha='center', va='center',
            color=BERKELEY_BLUE, zorder=3, style='italic')
    # Core modules around hexagon
    modules = [
        ('Quantum\nPhysics', 7, 7.8, '⚛️'),
        ('GPU\nAcceleration', 9.5, 6.5, '🚀'),
        ('ML Physics\nPINNs', 9.5, 3.5, '🧠'),
        ('Thermal\nTransport', 7, 2.2, '🌡️'),
        ('Signal\nProcessing', 4.5, 3.5, '📊'),
        ('Cross-Platform\nIntegration', 4.5, 6.5, '⚙️')
    ]
    for name, x, y, emoji in modules:
        # Module box
        box = FancyBboxPatch((x-1, y-0.6), 2, 1.2,
                           boxstyle="round,pad=0.15",
                           facecolor=CALIFORNIA_GOLD,
                           edgecolor=BERKELEY_BLUE,
                           linewidth=2, zorder=2)
        ax.add_patch(box)
        # Module text
        ax.text(x, y+0.1, name, fontsize=11, fontweight='bold',
                ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
        # Emoji icon
        ax.text(x, y-0.3, emoji, fontsize=16, ha='center', va='center', zorder=3)
        # Connection line to center
        ax.plot([x, 7], [y, 5], color=BERKELEY_BLUE, linewidth=2,
                alpha=0.6, zorder=1)
    # Input platforms (left side)
    platforms = [
        ('Python 3.8+', 1.5, 7.5, '🐍'),
        ('MATLAB R2021+', 1.5, 6, '📊'),
        ('Mathematica 12+', 1.5, 4.5, '🔣'),
        ('CLI Interface', 1.5, 3, '💻')
    ]
    for name, x, y, emoji in platforms:
        # Platform box
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor='lightgray',
                           edgecolor=BERKELEY_BLUE,
                           linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x-0.3, y, name, fontsize=10, ha='center', va='center',
                color=BERKELEY_BLUE, fontweight='bold', zorder=3)
        ax.text(x+0.5, y, emoji, fontsize=14, ha='center', va='center', zorder=3)
        # Arrow to framework
        arrow = FancyArrowPatch((x+0.8, y), (4, 5),
                              arrowstyle='->', mutation_scale=20,
                              color=BERKELEY_BLUE, alpha=0.7, zorder=1)
        ax.add_patch(arrow)
    # Output applications (right side)
    outputs = [
        ('Research\nPublications', 12.5, 7.5, '📝'),
        ('GPU Clusters\n55.7 GFLOPS', 12.5, 6, '🖥️'),
        ('Interactive\nNotebooks', 12.5, 4.5, '📓'),
        ('Production\nDeployment', 12.5, 3, '🚀')
    ]
    for name, x, y, emoji in outputs:
        # Output box
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor='lightgreen',
                           edgecolor=BERKELEY_BLUE,
                           linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x+0.3, y, name, fontsize=10, ha='center', va='center',
                color=BERKELEY_BLUE, fontweight='bold', zorder=3)
        ax.text(x-0.5, y, emoji, fontsize=14, ha='center', va='center', zorder=3)
        # Arrow from framework
        arrow = FancyArrowPatch((10, 5), (x-0.8, y),
                              arrowstyle='->', mutation_scale=20,
                              color=BERKELEY_BLUE, alpha=0.7, zorder=1)
        ax.add_patch(arrow)
    # Performance metrics box
    metrics_box = FancyBboxPatch((9.5, 0.5), 4, 1.5,
                                boxstyle="round,pad=0.2",
                                facecolor=LIGHT_GOLD,
                                edgecolor=CALIFORNIA_GOLD,
                                linewidth=2, zorder=2)
    ax.add_patch(metrics_box)
    ax.text(11.5, 1.6, 'Performance Metrics', fontsize=12, fontweight='bold',
            ha='center', va='top', color=BERKELEY_BLUE, zorder=3)
    ax.text(11.5, 1.2, '• 84.6% Validation Success', fontsize=10,
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    ax.text(11.5, 0.9, '• GPU Acceleration Ready', fontsize=10,
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    ax.text(11.5, 0.6, '• Cross-Platform Compatible', fontsize=10,
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    # UC Berkeley branding
    ax.text(0.5, 0.5, 'UC Berkeley', fontsize=14, fontweight='bold',
            ha='left', va='bottom', color=BERKELEY_BLUE, zorder=3)
    ax.text(0.5, 0.2, 'Scientific Computing Excellence', fontsize=10,
            ha='left', va='bottom', color=CALIFORNIA_GOLD,
            style='italic', zorder=3)
    # Bear emoji for Berkeley
    ax.text(2.8, 0.35, '🐻', fontsize=16, ha='center', va='center', zorder=3)
    return fig
def create_simple_logo():
    """Create a simple SciComp logo."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 3)
    ax.axis('off')
    # Main SciComp text
    ax.text(4, 1.8, 'SciComp', fontsize=48, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE)
    # Atomic symbol as "o" replacement - simplified version
    # Nucleus
    nucleus = Circle((3.85, 1.8), 0.08, facecolor=CALIFORNIA_GOLD,
                    edgecolor=BERKELEY_BLUE, linewidth=2)
    ax.add_patch(nucleus)
    # Electron orbits
    orbit1 = patches.Ellipse((3.85, 1.8), 0.3, 0.15, angle=30,
                           fill=False, edgecolor=BERKELEY_BLUE, linewidth=2)
    orbit2 = patches.Ellipse((3.85, 1.8), 0.3, 0.15, angle=-30,
                           fill=False, edgecolor=BERKELEY_BLUE, linewidth=2)
    ax.add_patch(orbit1)
    ax.add_patch(orbit2)
    # Tagline
    ax.text(4, 0.8, 'Cross-Platform Scientific Computing Suite',
            fontsize=16, ha='center', va='center',
            color=CALIFORNIA_GOLD, style='italic')
    # UC Berkeley attribution
    ax.text(4, 0.3, 'University of California, Berkeley',
            fontsize=12, ha='center', va='center', color=BERKELEY_BLUE)
    return fig
def main():
    """Generate both overview diagram and logo."""
    # Create output directory
    output_dir = Path("docs/images")
    output_dir.mkdir(exist_ok=True)
    # Generate overview diagram
    print("🎨 Generating SciComp overview diagram...")
    overview_fig = create_scicomp_overview()
    overview_fig.savefig(output_dir / "scicomp_overview.png",
                        dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
    overview_fig.savefig(output_dir / "scicomp_overview.pdf",
                        bbox_inches='tight', facecolor='white')
    print(f"✅ Overview diagram saved to {output_dir}/scicomp_overview.png")
    # Generate simple logo
    print("🎨 Generating SciComp logo...")
    logo_fig = create_simple_logo()
    logo_fig.savefig(output_dir / "scicomp_logo.png",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    logo_fig.savefig(output_dir / "scicomp_logo.pdf",
                    bbox_inches='tight', facecolor='white')
    print(f"✅ Logo saved to {output_dir}/scicomp_logo.png")
    # Show plots
    plt.show()
    print("\n🎉 Visual assets generated successfully!")
    print(f"📁 Files saved in: {output_dir.absolute()}")
    print("\n📋 Files created:")
    print("   • scicomp_overview.png (framework diagram)")
    print("   • scicomp_overview.pdf (vector version)")
    print("   • scicomp_logo.png (logo design)")
    print("   • scicomp_logo.pdf (vector version)")
if __name__ == "__main__":
    main()