#!/usr/bin/env python3
"""
SciComp Clean Visual Assets Generator
Generates clean, professional visual assets without emojis for maximum compatibility.
Creates overview diagram and logo suitable for all platforms and documentation.
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
def create_professional_overview():
    """Create a clean, professional overview diagram without emojis."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    # Subtle background gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, extent=[0, 14, 0, 10], aspect='auto',
              cmap='Blues', alpha=0.05, zorder=0)
    # Title and subtitle
    ax.text(7, 9.2, 'SciComp', fontsize=28, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE)
    ax.text(7, 8.7, 'Cross-Platform Scientific Computing Suite',
            fontsize=16, ha='center', va='center',
            color=CALIFORNIA_GOLD, style='italic')
    ax.text(7, 8.3, 'University of California, Berkeley',
            fontsize=12, ha='center', va='center', color=BERKELEY_BLUE)
    # Central framework hexagon
    hexagon = patches.RegularPolygon((7, 5), 6, radius=2.2,
                                   facecolor=LIGHT_BLUE,
                                   edgecolor=BERKELEY_BLUE,
                                   linewidth=3, zorder=2)
    ax.add_patch(hexagon)
    # Core framework label
    ax.text(7, 5.3, 'SciComp', fontsize=20, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    ax.text(7, 4.7, 'Core Framework', fontsize=12, ha='center', va='center',
            color=BERKELEY_BLUE, zorder=3, style='italic')
    # Core modules around hexagon
    modules = [
        ('Quantum Physics', 7, 7.6, CALIFORNIA_GOLD),
        ('GPU Acceleration', 9.3, 6.3, CALIFORNIA_GOLD),
        ('ML Physics', 9.3, 3.7, CALIFORNIA_GOLD),
        ('Thermal Transport', 7, 2.4, CALIFORNIA_GOLD),
        ('Signal Processing', 4.7, 3.7, CALIFORNIA_GOLD),
        ('Cross-Platform', 4.7, 6.3, CALIFORNIA_GOLD)
    ]
    for name, x, y, color in modules:
        # Module box with rounded corners
        box = FancyBboxPatch((x-0.9, y-0.35), 1.8, 0.7,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor=BERKELEY_BLUE,
                           linewidth=2, zorder=2)
        ax.add_patch(box)
        # Module text
        ax.text(x, y, name, fontsize=11, fontweight='bold',
                ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
        # Connection line to center
        ax.plot([x, 7], [y, 5], color=BERKELEY_BLUE, linewidth=1.5,
                alpha=0.5, linestyle='--', zorder=1)
    # Input platforms (left side)
    ax.text(1.5, 7.8, 'INPUT PLATFORMS', fontsize=12, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE)
    platforms = [
        ('Python 3.8+', 1.5, 7.2),
        ('MATLAB R2021+', 1.5, 6.6),
        ('Mathematica 12+', 1.5, 6.0),
        ('Command Line', 1.5, 5.4)
    ]
    for name, x, y in platforms:
        # Platform box
        box = FancyBboxPatch((x-0.7, y-0.2), 1.4, 0.4,
                           boxstyle="round,pad=0.05",
                           facecolor='lightgray',
                           edgecolor=BERKELEY_BLUE,
                           linewidth=1, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
        # Arrow to framework
        arrow = FancyArrowPatch((x+0.7, y), (4.5, 5),
                              arrowstyle='->', mutation_scale=15,
                              color=BERKELEY_BLUE, alpha=0.6, zorder=1)
        ax.add_patch(arrow)
    # Output applications (right side)
    ax.text(12.5, 7.8, 'OUTPUT APPLICATIONS', fontsize=12, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE)
    outputs = [
        ('Research Publications', 12.5, 7.2),
        ('HPC Clusters', 12.5, 6.6),
        ('Interactive Analysis', 12.5, 6.0),
        ('Production Systems', 12.5, 5.4)
    ]
    for name, x, y in outputs:
        # Output box
        box = FancyBboxPatch((x-0.8, y-0.2), 1.6, 0.4,
                           boxstyle="round,pad=0.05",
                           facecolor='lightgreen',
                           edgecolor=BERKELEY_BLUE,
                           linewidth=1, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold',
                ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
        # Arrow from framework
        arrow = FancyArrowPatch((9.5, 5), (x-0.8, y),
                              arrowstyle='->', mutation_scale=15,
                              color=BERKELEY_BLUE, alpha=0.6, zorder=1)
        ax.add_patch(arrow)
    # Performance metrics box
    metrics_box = FancyBboxPatch((9, 1.5), 4.5, 2,
                                boxstyle="round,pad=0.15",
                                facecolor=LIGHT_GOLD,
                                edgecolor=CALIFORNIA_GOLD,
                                linewidth=2, zorder=2)
    ax.add_patch(metrics_box)
    ax.text(11.25, 3.2, 'PERFORMANCE METRICS', fontsize=12, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    metrics = [
        '84.6% Validation Success Rate',
        '55.7 GFLOPS GPU Performance',
        'Cross-Platform Compatibility',
        'Production-Ready Deployment'
    ]
    for i, metric in enumerate(metrics):
        ax.text(11.25, 2.8 - i*0.25, f'• {metric}', fontsize=10,
                ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    # Footer with UC Berkeley branding
    footer_box = FancyBboxPatch((0.5, 0.3), 6, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=LIGHT_BLUE,
                               edgecolor=BERKELEY_BLUE,
                               linewidth=1, zorder=2)
    ax.add_patch(footer_box)
    ax.text(3.5, 0.8, 'UC Berkeley', fontsize=14, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE, zorder=3)
    ax.text(3.5, 0.5, 'Scientific Computing Excellence Since 1868', fontsize=10,
            ha='center', va='center', color=CALIFORNIA_GOLD,
            style='italic', zorder=3)
    return fig
def create_minimal_logo():
    """Create a minimal, professional logo design."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    # Main SciComp text
    ax.text(5, 2.4, 'SciComp', fontsize=54, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE)
    # Stylized atomic symbol integrated into design
    # Create a more sophisticated atom representation
    nucleus = Circle((4.72, 2.4), 0.06, facecolor=CALIFORNIA_GOLD,
                    edgecolor=BERKELEY_BLUE, linewidth=2, zorder=3)
    ax.add_patch(nucleus)
    # Three electron orbits at different angles
    orbit_angles = [0, 60, 120]
    for angle in orbit_angles:
        orbit = patches.Ellipse((4.72, 2.4), 0.24, 0.12, angle=angle,
                               fill=False, edgecolor=BERKELEY_BLUE,
                               linewidth=2, alpha=0.8, zorder=2)
        ax.add_patch(orbit)
    # Electrons as small dots
    electron_positions = [(4.84, 2.4), (4.66, 2.46), (4.66, 2.34)]
    for pos in electron_positions:
        electron = Circle(pos, 0.02, facecolor=BERKELEY_BLUE, zorder=4)
        ax.add_patch(electron)
    # Tagline
    ax.text(5, 1.6, 'Cross-Platform Scientific Computing Suite',
            fontsize=18, ha='center', va='center',
            color=CALIFORNIA_GOLD, style='italic')
    # UC Berkeley attribution
    ax.text(5, 0.8, 'University of California, Berkeley',
            fontsize=14, ha='center', va='center', color=BERKELEY_BLUE)
    # Decorative line elements
    ax.plot([1, 9], [0.4, 0.4], color=BERKELEY_BLUE, linewidth=2)
    ax.plot([3, 7], [0.2, 0.2], color=CALIFORNIA_GOLD, linewidth=1)
    return fig
def create_icon_versions():
    """Create square icon versions for favicons and social media."""
    # Square icon version
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')
    # Background circle
    bg_circle = Circle((2, 2), 1.8, facecolor=LIGHT_BLUE,
                      edgecolor=BERKELEY_BLUE, linewidth=4)
    ax.add_patch(bg_circle)
    # SciComp text (abbreviated)
    ax.text(2, 2.4, 'SC', fontsize=36, fontweight='bold',
            ha='center', va='center', color=BERKELEY_BLUE)
    # Atomic symbol
    nucleus = Circle((2, 1.5), 0.1, facecolor=CALIFORNIA_GOLD,
                    edgecolor=BERKELEY_BLUE, linewidth=2)
    ax.add_patch(nucleus)
    # Simplified orbits
    orbit1 = patches.Ellipse((2, 1.5), 0.4, 0.2, angle=45,
                           fill=False, edgecolor=BERKELEY_BLUE, linewidth=2)
    orbit2 = patches.Ellipse((2, 1.5), 0.4, 0.2, angle=-45,
                           fill=False, edgecolor=BERKELEY_BLUE, linewidth=2)
    ax.add_patch(orbit1)
    ax.add_patch(orbit2)
    return fig
def main():
    """Generate all visual assets."""
    # Create output directory
    output_dir = Path("docs/images")
    output_dir.mkdir(exist_ok=True)
    print("🎨 Generating professional SciComp visuals...")
    # Generate clean overview diagram
    print("📊 Creating overview diagram...")
    overview_fig = create_professional_overview()
    overview_fig.savefig(output_dir / "scicomp_overview_clean.png",
                        dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
    overview_fig.savefig(output_dir / "scicomp_overview_clean.pdf",
                        bbox_inches='tight', facecolor='white')
    plt.close(overview_fig)
    # Generate minimal logo
    print("🏷️  Creating logo design...")
    logo_fig = create_minimal_logo()
    logo_fig.savefig(output_dir / "scicomp_logo_clean.png",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    logo_fig.savefig(output_dir / "scicomp_logo_clean.pdf",
                    bbox_inches='tight', facecolor='white')
    plt.close(logo_fig)
    # Generate icon version
    print("🔲 Creating icon version...")
    icon_fig = create_icon_versions()
    icon_fig.savefig(output_dir / "scicomp_icon.png",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    icon_fig.savefig(output_dir / "scicomp_icon_256.png",
                    dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(icon_fig)
    print("\n✅ All visual assets generated successfully!")
    print(f"📁 Files saved in: {output_dir.absolute()}")
    print("\n📋 Clean versions created:")
    print("   • scicomp_overview_clean.png (professional overview)")
    print("   • scicomp_overview_clean.pdf (vector version)")
    print("   • scicomp_logo_clean.png (clean logo design)")
    print("   • scicomp_logo_clean.pdf (vector version)")
    print("   • scicomp_icon.png (square icon for favicons)")
    print("   • scicomp_icon_256.png (256px icon)")
    print("\n🎯 These versions are optimized for:")
    print("   • Professional documentation")
    print("   • Academic presentations")
    print("   • Website headers and branding")
    print("   • Social media profiles")
    print("   • Print materials and posters")
if __name__ == "__main__":
    main()