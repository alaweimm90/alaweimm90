import React, { useRef, useEffect } from 'react';

interface HoneycombLatticeVisualizationProps {
  className?: string;
}

export function HoneycombLatticeVisualization({ className }: HoneycombLatticeVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Lattice parameters (doubled periodicity)
    const a = 30; // lattice constant in pixels (reduced for more units)
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    // Lattice vectors for hexagonal lattice
    const a1 = [a * Math.sqrt(3), 0];
    const a2 = [a * Math.sqrt(3) / 2, a * 3 / 2];

    // Generate lattice sites (8x8 supercell for doubled periodicity)
    const sites: { pos: [number, number]; sublattice: 'A' | 'B' }[] = [];
    
    for (let i = -4; i <= 4; i++) {
      for (let j = -4; j <= 4; j++) {
        // A sublattice at (0, 0)
        const posA: [number, number] = [
          centerX + i * a1[0] + j * a2[0],
          centerY + i * a1[1] + j * a2[1]
        ];
        sites.push({ pos: posA, sublattice: 'A' });

        // B sublattice at nearest neighbor position
        const posB: [number, number] = [
          centerX + i * a1[0] + j * a2[0] + a * Math.sqrt(3) / 3,
          centerY + i * a1[1] + j * a2[1] + a / 3
        ];
        sites.push({ pos: posB, sublattice: 'B' });
      }
    }

    // Filter sites within canvas bounds
    const visibleSites = sites.filter(site => 
      site.pos[0] >= -20 && site.pos[0] <= rect.width + 20 &&
      site.pos[1] >= -20 && site.pos[1] <= rect.height + 20
    );

    // Draw bonds first (behind atoms)
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 2;
    const bondLength = a * Math.sqrt(3) / 3; // Correct NN distance

    for (const siteA of visibleSites) {
      if (siteA.sublattice !== 'A') continue;
      
      for (const siteB of visibleSites) {
        if (siteB.sublattice !== 'B') continue;
        
        const dx = siteB.pos[0] - siteA.pos[0];
        const dy = siteB.pos[1] - siteA.pos[1];
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Draw bond if distance is approximately bond length
        if (Math.abs(distance - bondLength) < bondLength * 0.1) {
          ctx.beginPath();
          ctx.moveTo(siteA.pos[0], siteA.pos[1]);
          ctx.lineTo(siteB.pos[0], siteB.pos[1]);
          ctx.stroke();
        }
      }
    }

    // Draw primitive unit cell
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    const cellCorners = [
      [centerX, centerY],
      [centerX + a1[0], centerY + a1[1]],
      [centerX + a1[0] + a2[0], centerY + a1[1] + a2[1]],
      [centerX + a2[0], centerY + a2[1]],
      [centerX, centerY]
    ];
    
    ctx.beginPath();
    ctx.moveTo(cellCorners[0][0], cellCorners[0][1]);
    for (let i = 1; i < cellCorners.length; i++) {
      ctx.lineTo(cellCorners[i][0], cellCorners[i][1]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Skip orbital drawing - just lattice structure

    // Draw carbon atoms
    for (const site of visibleSites) {
      ctx.fillStyle = site.sublattice === 'A' ? '#1e40af' : '#dc2626';
      ctx.beginPath();
      ctx.arc(site.pos[0], site.pos[1], 6, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add white border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Draw legend (positioned top right)
    const legendX = rect.width - 180;
    const legendY = 20;
    ctx.font = '12px Arial';
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Hexagonal Lattice (2×)', legendX, legendY);
    
    // A sublattice legend
    ctx.fillStyle = '#1e40af';
    ctx.beginPath();
    ctx.arc(legendX + 10, legendY + 15, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.fillText('A sublattice', legendX + 20, legendY + 20);
    
    // B sublattice legend
    ctx.fillStyle = '#dc2626';
    ctx.beginPath();
    ctx.arc(legendX + 10, legendY + 30, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.fillText('B sublattice', legendX + 20, legendY + 35);
    
    // Primitive unit cell legend
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.rect(legendX + 8, legendY + 42, 8, 6);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Unit cell', legendX + 20, legendY + 50);
    
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-full ${className}`}
      style={{ width: '100%', height: '100%' }}
    />
  );
}