import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BookOpen, Calculator, Atom, GitBranch } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';
import { useCrystalStore, crystalStructures } from '@/lib/crystal-store';

export function TheoryPanel() {
  const { selectedStructure } = useCrystalStore();
  const structure = crystalStructures[selectedStructure];

  return (
    <div className="space-y-6">
      <Tabs defaultValue="fundamentals" className="w-full">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="fundamentals">Fundamentals</TabsTrigger>
          <TabsTrigger value="symmetry">Symmetry</TabsTrigger>
          <TabsTrigger value="diffraction">Diffraction</TabsTrigger>
          <TabsTrigger value="examples">Examples</TabsTrigger>
        </TabsList>

        <TabsContent value="fundamentals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Crystal Fundamentals
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Bravais Lattices</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  There are 14 distinct Bravais lattices in three dimensions, grouped into 7 crystal systems.
                  Each lattice is characterized by its lattice parameters and symmetry.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Lattice Vectors</h4>
                  <p className="text-sm mb-2">
                    A crystal lattice is defined by three lattice vectors <InlineMath math="\mathbf{a}" />, 
                    <InlineMath math="\mathbf{b}" />, and <InlineMath math="\mathbf{c}" />:
                  </p>
                  <BlockMath math="\mathbf{R} = n_1\mathbf{a} + n_2\mathbf{b} + n_3\mathbf{c}" />
                  <p className="text-sm mt-2">
                    where <InlineMath math="n_1, n_2, n_3" /> are integers.
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Unit Cell</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  The unit cell is the smallest repeating unit that generates the entire crystal 
                  when translated by lattice vectors.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Volume Calculation</h4>
                  <BlockMath math="V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = abc\sqrt{1 + 2\cos\alpha\cos\beta\cos\gamma - \cos^2\alpha - \cos^2\beta - \cos^2\gamma}" />
                </div>
              </div>

              {structure && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Current Structure: {structure.name}</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Badge variant="secondary">{structure.system}</Badge>
                      <Badge variant="outline" className="ml-2">{structure.latticeType}</Badge>
                    </div>
                    <div className="text-sm">
                      <div>Space Group: {structure.spaceGroup}</div>
                      <div>Point Group: {structure.pointGroup}</div>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <h4 className="font-medium mb-2">Examples</h4>
                    <ul className="text-sm text-muted-foreground">
                      {structure.examples.map((example, i) => (
                        <li key={i}>• {example}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="symmetry" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5" />
                Crystallographic Symmetry
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Space Groups</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Space groups describe the symmetry of crystal structures, combining point group 
                  operations with translations.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Symmetry Operations</h4>
                  <ul className="text-sm space-y-1">
                    <li><strong>Identity (E):</strong> No change</li>
                    <li><strong>Rotation (C<sub>n</sub>):</strong> Rotation by 2π/n</li>
                    <li><strong>Reflection (σ):</strong> Mirror plane</li>
                    <li><strong>Inversion (i):</strong> Point inversion</li>
                    <li><strong>Improper rotation (S<sub>n</sub>):</strong> Rotation + reflection</li>
                  </ul>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Miller Indices</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Miller indices (hkl) describe crystallographic planes and directions.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Plane Equation</h4>
                  <BlockMath math="\frac{x}{a/h} + \frac{y}{b/k} + \frac{z}{c/l} = 1" />
                  <p className="text-sm mt-2">
                    where (h k l) are the Miller indices and a, b, c are lattice parameters.
                  </p>
                </div>

                <div className="mt-4">
                  <h4 className="font-medium mb-2">Common Planes</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>(100) - perpendicular to a-axis</div>
                    <div>(110) - diagonal face</div>
                    <div>(111) - body diagonal</div>
                    <div>(200) - parallel to (100), half spacing</div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Reciprocal Lattice</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  The reciprocal lattice is defined by reciprocal lattice vectors.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Reciprocal Vectors</h4>
                  <BlockMath math="\mathbf{a}^* = \frac{2\pi(\mathbf{b} \times \mathbf{c})}{V}" />
                  <BlockMath math="\mathbf{b}^* = \frac{2\pi(\mathbf{c} \times \mathbf{a})}{V}" />
                  <BlockMath math="\mathbf{c}^* = \frac{2\pi(\mathbf{a} \times \mathbf{b})}{V}" />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="diffraction" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calculator className="h-5 w-5" />
                X-ray Diffraction
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Bragg's Law</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Describes the condition for constructive interference in crystal diffraction.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <BlockMath math="n\lambda = 2d_{hkl}\sin\theta" />
                  <p className="text-sm mt-2">
                    where n is the order of diffraction, λ is wavelength, d<sub>hkl</sub> is 
                    interplanar spacing, and θ is the Bragg angle.
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Structure Factor</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Determines the intensity of diffracted beams.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <BlockMath math="F_{hkl} = \sum_j f_j e^{2\pi i(hx_j + ky_j + lz_j)}" />
                  <p className="text-sm mt-2">
                    where f<sub>j</sub> is the atomic form factor and (x<sub>j</sub>, y<sub>j</sub>, z<sub>j</sub>) 
                    are fractional coordinates.
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Systematic Absences</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Certain reflections are forbidden due to translational symmetry.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Common Rules</h4>
                  <ul className="text-sm space-y-1">
                    <li><strong>P lattice:</strong> No restrictions</li>
                    <li><strong>I lattice:</strong> h + k + l = even</li>
                    <li><strong>F lattice:</strong> h, k, l all even or all odd</li>
                    <li><strong>C lattice:</strong> h + k = even</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="examples" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Atom className="h-5 w-5" />
                Real Crystal Examples
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3">Diamond Structure</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Two interpenetrating FCC lattices displaced by (1/4, 1/4, 1/4).
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Properties</h4>
                  <ul className="text-sm space-y-1">
                    <li>Coordination number: 4</li>
                    <li>Packing efficiency: 34%</li>
                    <li>Each atom forms 4 covalent bonds</li>
                    <li>Tetrahedral coordination</li>
                  </ul>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">NaCl Structure</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Two FCC lattices (Na⁺ and Cl⁻) displaced by (1/2, 1/2, 1/2).
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Properties</h4>
                  <ul className="text-sm space-y-1">
                    <li>Coordination number: 6 (octahedral)</li>
                    <li>Ionic bonding</li>
                    <li>High melting point</li>
                    <li>Electrical insulator</li>
                  </ul>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Graphene Structure</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Two-dimensional honeycomb lattice of carbon atoms.
                </p>
                
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-medium mb-2">Properties</h4>
                  <ul className="text-sm space-y-1">
                    <li>Coordination number: 3 (sp² hybridization)</li>
                    <li>High electrical conductivity</li>
                    <li>Exceptional mechanical strength</li>
                    <li>Dirac cone band structure</li>
                  </ul>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">Applications</h3>
                <div className="grid grid-cols-1 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Semiconductors</h4>
                    <p className="text-sm">
                      Diamond and zinc blende structures are common in semiconductors 
                      like Si, Ge, GaAs, and InP.
                    </p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Ionic Crystals</h4>
                    <p className="text-sm">
                      NaCl-type structures appear in many ionic compounds and determine 
                      properties like solubility and conductivity.
                    </p>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <h4 className="font-medium mb-2">2D Materials</h4>
                    <p className="text-sm">
                      Graphene and similar 2D materials have applications in electronics, 
                      energy storage, and composite materials.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}