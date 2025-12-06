import React, { useRef, useEffect } from 'react';

interface CorrectHoneycombLatticeProps {
  className?: string;
}

export function CorrectHoneycombLattice({ className }: CorrectHoneycombLatticeProps) {
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

    // CORRECT lattice parameters following the algorithm
    const bond_length = 30; // C-C bond length in pixels (scaled for visibility)
    const a = bond_length * Math.sqrt(3); // lattice constant: a = √3 × bond_length
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    // CORRECT primitive lattice vectors as specified
    const a1: [number, number] = [a * Math.sqrt(3) / 2, a / 2];
    const a2: [number, number] = [a * Math.sqrt(3) / 2, -a / 2];

    // CORRECT atom positions within unit cell
    const atom_A_pos: [number, number] = [0, 0];           // A atom at origin
    const atom_B_pos: [number, number] = [0, bond_length]; // B atom displaced by bond length

    // Generate lattice using the algorithm: R = n₁a₁ + n₂a₂
    const sites: { pos: [number, number]; sublattice: 'A' | 'B' }[] = [];
    
    const num_cells = 4; // Number of unit cells in each direction
    
    for (let n1 = -num_cells; n1 <= num_cells; n1++) {
      for (let n2 = -num_cells; n2 <= num_cells; n2++) {
        // Unit cell origin: R = n₁a₁ + n₂a₂
        const unit_cell_origin: [number, number] = [
          centerX + n1 * a1[0] + n2 * a2[0],
          centerY + n1 * a1[1] + n2 * a2[1]
        ];
        
        // Place A atom at unit cell origin + atom_A_pos
        sites.push({ 
          pos: [
            unit_cell_origin[0] + atom_A_pos[0],
            unit_cell_origin[1] + atom_A_pos[1]
          ], 
          sublattice: 'A' 
        });

        // Place B atom at unit cell origin + atom_B_pos
        sites.push({ 
          pos: [
            unit_cell_origin[0] + atom_B_pos[0],
            unit_cell_origin[1] + atom_B_pos[1]
          ], 
          sublattice: 'B' 
        });
      }
    }

    // Filter sites within canvas bounds
    const visibleSites = sites.filter(site => 
      site.pos[0] >= -20 && site.pos[0] <= rect.width + 20 &&
      site.pos[1] >= -20 && site.pos[1] <= rect.height + 20
    );

    // Draw bonds using CORRECT nearest-neighbor connectivity (bond_length distance)
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 2;
    const bondCutoff = bond_length * 1.1; // Slightly larger than bond_length

    for (const siteA of visibleSites) {
      if (siteA.sublattice !== 'A') continue;
      
      for (const siteB of visibleSites) {
        if (siteB.sublattice !== 'B') continue;
        
        const dx = siteB.pos[0] - siteA.pos[0];
        const dy = siteB.pos[1] - siteA.pos[1];
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Draw bond if distance equals bond_length (C-C bond)
        if (distance < bondCutoff && distance > bond_length * 0.9) {
          ctx.beginPath();
          ctx.moveTo(siteA.pos[0], siteA.pos[1]);
          ctx.lineTo(siteB.pos[0], siteB.pos[1]);
          ctx.stroke();
        }
      }
    }

    // Draw CORRECT primitive unit cell
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    // Unit cell corners using correct lattice vectors
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

    // Draw legend (top right)
    const legendX = rect.width - 200;
    const legendY = 30;
    ctx.font = '12px Arial';
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Honeycomb Lattice', legendX, legendY);
    
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
    
    // Bond length indicator
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`C-C bond = ${(bond_length / 30 * 1.42).toFixed(2)} Å`, legendX, legendY + 65);
    
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-full ${className}`}
      style={{ width: '100%', height: '100%' }}
    />
  );
}