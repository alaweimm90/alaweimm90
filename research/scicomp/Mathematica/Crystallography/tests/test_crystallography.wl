(* Test Suite for Mathematica Crystallography Implementation *)
(*
 * Comprehensive tests for CrystalStructure, Diffraction, SpaceGroups, 
 * and StructureRefinement modules to ensure correctness and platform parity.
 *)

BeginPackage["CrystallographyTests`"]

(* Load modules *)
Get[FileNameJoin[{NotebookDirectory[], "..", "core", "CrystalStructure.wl"}]];
Get[FileNameJoin[{NotebookDirectory[], "..", "core", "Diffraction.wl"}]];
Get[FileNameJoin[{NotebookDirectory[], "..", "core", "SpaceGroups.wl"}]];
Get[FileNameJoin[{NotebookDirectory[], "..", "core", "StructureRefinement.wl"}]];

Begin["`Private`"]

(* Test framework utilities *)
AssertEqual[actual_, expected_, testName_] := Module[{tolerance = 10^-6},
  If[Abs[actual - expected] < tolerance,
    Print[Style["✓ PASS: " <> testName, Green]],
    Print[Style["✗ FAIL: " <> testName <> " (Expected: " <> ToString[expected] <> 
                ", Got: " <> ToString[actual] <> ")", Red]];
    $Failed
  ]
]

AssertTrue[condition_, testName_] := 
  If[condition,
    Print[Style["✓ PASS: " <> testName, Green]],
    Print[Style["✗ FAIL: " <> testName, Red]];
    $Failed
  ]

(* Test data structures *)
CreateTestCubicLattice[] := <|
  "a" -> 5.0, "b" -> 5.0, "c" -> 5.0,
  "alpha" -> 90, "beta" -> 90, "gamma" -> 90
|>

CreateTestSimpleAtoms[] := {
  <|"element" -> "Si", "x" -> 0.0, "y" -> 0.0, "z" -> 0.0|>,
  <|"element" -> "Si", "x" -> 0.5, "y" -> 0.5, "z" -> 0.5|>
}

CreateTestNaClAtoms[] := {
  <|"element" -> "Na", "x" -> 0.0, "y" -> 0.0, "z" -> 0.0|>,
  <|"element" -> "Cl", "x" -> 0.5, "y" -> 0.0, "z" -> 0.0|>
}

(* CrystalStructure Tests *)
TestCrystalStructureCreation[] := Module[{lattice, atoms, crystal},
  Print[Style["\n=== CrystalStructure Tests ===", Bold]];
  
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  AssertTrue[AssociationQ[crystal], "Crystal structure creation"];
  AssertTrue[KeyExistsQ[crystal, "lattice"], "Crystal has lattice parameters"];
  AssertTrue[KeyExistsQ[crystal, "atoms"], "Crystal has atomic positions"];
  AssertTrue[KeyExistsQ[crystal, "directMatrix"], "Crystal has direct matrix"];
]

TestUnitCellVolume[] := Module[{lattice, atoms, crystal, volume},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  volume = UnitCellVolume[crystal];
  AssertEqual[volume, 125.0, "Unit cell volume calculation (5³ = 125)"];
]

TestDensityCalculation[] := Module[{lattice, atoms, crystal, density},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  (* Silicon: MW = 28.085 g/mol, 2 atoms per unit cell *)
  density = Density[crystal, 28.085, 2];
  AssertTrue[density > 0, "Density calculation gives positive result"];
]

TestCoordinateTransformations[] := Module[{lattice, atoms, crystal, fractional, cartesian, roundTrip},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  fractional = {0.5, 0.5, 0.5};
  cartesian = FractionalToCartesian[crystal, fractional];
  AssertEqual[cartesian[[1]], 2.5, "Fractional to Cartesian X"];
  AssertEqual[cartesian[[2]], 2.5, "Fractional to Cartesian Y"];
  AssertEqual[cartesian[[3]], 2.5, "Fractional to Cartesian Z"];
  
  roundTrip = CartesianToFractional[crystal, cartesian];
  AssertEqual[roundTrip[[1]], 0.5, "Round trip transformation X"];
  AssertEqual[roundTrip[[2]], 0.5, "Round trip transformation Y"];
  AssertEqual[roundTrip[[3]], 0.5, "Round trip transformation Z"];
]

TestInteratomicDistances[] := Module[{lattice, atoms, crystal, distance},
  lattice = <|"a" -> 4.0, "b" -> 4.0, "c" -> 4.0, "alpha" -> 90, "beta" -> 90, "gamma" -> 90|>;
  atoms = CreateTestNaClAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  distance = InteratomicDistance[crystal, 1, 2, False];
  AssertEqual[distance, 2.0, "Interatomic distance calculation"];
]

TestDSpacing[] := Module[{lattice, atoms, crystal, d},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  d = DSpacing[crystal, 1, 0, 0];
  AssertEqual[d, 5.0, "d-spacing for (100) reflection"];
  
  d = DSpacing[crystal, 1, 1, 0];
  AssertEqual[d, 5.0/Sqrt[2], "d-spacing for (110) reflection"];
]

TestBraggAngle[] := Module[{lattice, atoms, crystal, angle, wavelength},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  wavelength = 1.54056;
  
  angle = BraggAngle[crystal, 1, 0, 0, wavelength];
  AssertTrue[angle > 0 && angle < 90, "Bragg angle in valid range"];
]

TestSupercell[] := Module[{lattice, atoms, crystal, supercell, originalVolume, supercellVolume},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  supercell = Supercell[crystal, 2, 2, 2];
  AssertTrue[AssociationQ[supercell], "Supercell creation"];
  
  originalVolume = UnitCellVolume[crystal];
  supercellVolume = UnitCellVolume[supercell];
  AssertEqual[supercellVolume/originalVolume, 8.0, "Supercell volume scaling"];
  
  AssertEqual[Length[supercell["atoms"]], 8 * Length[atoms], "Supercell atom count"];
]

(* Diffraction Tests *)
TestDiffractionModule[] := Module[{crystal, lattice, atoms},
  Print[Style["\n=== Diffraction Tests ===", Bold]];
  
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  TestAtomicScatteringFactor[];
  TestStructureFactor[crystal];
  TestDiffractionPattern[crystal];
]

TestAtomicScatteringFactor[] := Module[{f},
  f = AtomicScatteringFactor["Si", 0.1];
  AssertTrue[Re[f] > 0, "Atomic scattering factor has positive real part"];
  AssertTrue[Im[f] == 0, "Atomic scattering factor has zero imaginary part (no anomalous scattering)"];
]

TestStructureFactor[crystal_] := Module[{F},
  F = StructureFactor[crystal, 1, 0, 0];
  AssertTrue[NumericQ[Abs[F]], "Structure factor calculation returns numeric result"];
]

TestDiffractionPattern[crystal_] := Module[{pattern, reflections},
  pattern = DiffractionPattern[crystal];
  AssertTrue[AssociationQ[pattern], "Diffraction pattern object creation"];
  
  reflections = CalculateReflections[crystal, 1.54056, 60.0, 3];
  AssertTrue[Length[reflections] > 0, "Reflection calculation generates results"];
]

(* SpaceGroups Tests *)
TestSpaceGroups[] := Module[{},
  Print[Style["\n=== SpaceGroups Tests ===", Bold]];
  
  TestSpaceGroupCreation[];
  TestSymmetryOperations[];
]

TestSpaceGroupCreation[] := Module[{sg},
  sg = SpaceGroup["P 1"];
  AssertTrue[AssociationQ[sg], "Space group P1 creation"];
  AssertTrue[KeyExistsQ[sg, "operations"], "Space group has operations"];
  
  sg = SpaceGroup["P m -3 m"];
  AssertTrue[AssociationQ[sg], "Space group Pm-3m creation"];
]

TestSymmetryOperations[] := Module[{sg, equivalentPos},
  sg = SpaceGroup["P -1"];
  equivalentPos = GenerateEquivalentPositions[sg, {0.25, 0.25, 0.25}];
  AssertTrue[Length[equivalentPos] == 2, "P-1 generates 2 equivalent positions"];
]

(* Integration Tests *)
TestCrossPlatformConsistency[] := Module[{},
  Print[Style["\n=== Cross-Platform Consistency Tests ===", Bold]];
  
  TestNaClConsistency[];
  TestDiamondConsistency[];
]

TestNaClConsistency[] := Module[{lattice, atoms, crystal, volume, density},
  (* Test against known values for NaCl *)
  lattice = <|"a" -> 5.64, "b" -> 5.64, "c" -> 5.64, "alpha" -> 90, "beta" -> 90, "gamma" -> 90|>;
  atoms = {
    <|"element" -> "Na", "x" -> 0.0, "y" -> 0.0, "z" -> 0.0|>,
    <|"element" -> "Na", "x" -> 0.5, "y" -> 0.5, "z" -> 0.0|>,
    <|"element" -> "Na", "x" -> 0.5, "y" -> 0.0, "z" -> 0.5|>,
    <|"element" -> "Na", "x" -> 0.0, "y" -> 0.5, "z" -> 0.5|>,
    <|"element" -> "Cl", "x" -> 0.5, "y" -> 0.0, "z" -> 0.0|>,
    <|"element" -> "Cl", "x" -> 0.0, "y" -> 0.5, "z" -> 0.0|>,
    <|"element" -> "Cl", "x" -> 0.0, "y" -> 0.0, "z" -> 0.5|>,
    <|"element" -> "Cl", "x" -> 0.5, "y" -> 0.5, "z" -> 0.5|>
  };
  crystal = CrystalStructure[lattice, atoms];
  
  volume = UnitCellVolume[crystal];
  AssertEqual[volume, 179.406144, "NaCl unit cell volume"];
  
  density = Density[crystal, 58.44, 4];
  AssertTrue[Abs[density - 2.16] < 0.1, "NaCl density approximately correct"];
]

TestDiamondConsistency[] := Module[{lattice, atoms, crystal, volume, density},
  (* Test against known values for diamond *)
  lattice = <|"a" -> 3.567, "b" -> 3.567, "c" -> 3.567, "alpha" -> 90, "beta" -> 90, "gamma" -> 90|>;
  atoms = {
    <|"element" -> "C", "x" -> 0.0, "y" -> 0.0, "z" -> 0.0|>,
    <|"element" -> "C", "x" -> 0.25, "y" -> 0.25, "z" -> 0.25|>,
    <|"element" -> "C", "x" -> 0.5, "y" -> 0.5, "z" -> 0.0|>,
    <|"element" -> "C", "x" -> 0.75, "y" -> 0.75, "z" -> 0.25|>,
    <|"element" -> "C", "x" -> 0.5, "y" -> 0.0, "z" -> 0.5|>,
    <|"element" -> "C", "x" -> 0.75, "y" -> 0.25, "z" -> 0.75|>,
    <|"element" -> "C", "x" -> 0.0, "y" -> 0.5, "z" -> 0.5|>,
    <|"element" -> "C", "x" -> 0.25, "y" -> 0.75, "z" -> 0.75|>
  };
  crystal = CrystalStructure[lattice, atoms];
  
  volume = UnitCellVolume[crystal];
  AssertEqual[volume, 45.379263, "Diamond unit cell volume"];
  
  density = Density[crystal, 12.01, 8];
  AssertTrue[Abs[density - 3.52] < 0.2, "Diamond density approximately correct"];
]

(* Error handling tests *)
TestErrorHandling[] := Module[{},
  Print[Style["\n=== Error Handling Tests ===", Bold]];
  
  TestInvalidLatticeParameters[];
  TestInvalidAtomicPositions[];
]

TestInvalidLatticeParameters[] := Module[{result},
  (* Missing required field *)
  result = CrystalStructure[<|"a" -> 5.0|>, CreateTestSimpleAtoms[]];
  AssertTrue[result === $Failed, "Invalid lattice parameters rejected"];
  
  (* Negative lattice parameter *)
  result = CrystalStructure[<|"a" -> -5.0, "b" -> 5.0, "c" -> 5.0, "alpha" -> 90, "beta" -> 90, "gamma" -> 90|>, 
                           CreateTestSimpleAtoms[]];
  AssertTrue[result === $Failed, "Negative lattice parameter rejected"];
]

TestInvalidAtomicPositions[] := Module[{result, lattice},
  lattice = CreateTestCubicLattice[];
  
  (* Empty atom list *)
  result = CrystalStructure[lattice, {}];
  AssertTrue[result === $Failed, "Empty atom list rejected"];
  
  (* Missing required field *)
  result = CrystalStructure[lattice, {<|"x" -> 0.0, "y" -> 0.0, "z" -> 0.0|>}];
  AssertTrue[result === $Failed, "Incomplete atom definition rejected"];
]

(* Performance tests *)
TestPerformance[] := Module[{},
  Print[Style["\n=== Performance Tests ===", Bold]];
  
  TestLargeSupercellPerformance[];
  TestManyReflectionsPerformance[];
]

TestLargeSupercellPerformance[] := Module[{lattice, atoms, crystal, time},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  time = AbsoluteTiming[Supercell[crystal, 5, 5, 5]][[1]];
  AssertTrue[time < 10.0, "Large supercell creation completes in reasonable time"];
  Print["Large supercell (5×5×5) creation time: ", NumberForm[time, {4, 2}], " seconds"];
]

TestManyReflectionsPerformance[] := Module[{lattice, atoms, crystal, time, reflections},
  lattice = CreateTestCubicLattice[];
  atoms = CreateTestSimpleAtoms[];
  crystal = CrystalStructure[lattice, atoms];
  
  time = AbsoluteTiming[
    reflections = CalculateReflections[crystal, 1.54056, 120.0, 8]
  ][[1]];
  
  AssertTrue[time < 30.0, "Many reflections calculation completes in reasonable time"];
  AssertTrue[Length[reflections] > 50, "Many reflections generated"];
  Print["Reflection calculation time: ", NumberForm[time, {4, 2}], " seconds, ", Length[reflections], " reflections"];
]

(* Main test runner *)
RunAllTests[] := Module[{startTime, endTime, totalTime},
  Print[Style["Mathematica Crystallography Test Suite", Bold, Large]];
  Print[Style["=====================================", Bold]];
  Print["Running comprehensive tests for platform parity validation...\n"];
  
  startTime = AbsoluteTime[];
  
  (* Core functionality tests *)
  TestCrystalStructureCreation[];
  TestUnitCellVolume[];
  TestDensityCalculation[];
  TestCoordinateTransformations[];
  TestInteratomicDistances[];
  TestDSpacing[];
  TestBraggAngle[];
  TestSupercell[];
  
  (* Module tests *)
  TestDiffractionModule[];
  TestSpaceGroups[];
  
  (* Integration tests *)
  TestCrossPlatformConsistency[];
  
  (* Robustness tests *)
  TestErrorHandling[];
  TestPerformance[];
  
  endTime = AbsoluteTime[];
  totalTime = endTime - startTime;
  
  Print[Style["\n=====================================", Bold]];
  Print[Style["Test Suite Completed", Bold]];
  Print["Total execution time: ", NumberForm[totalTime, {5, 2}], " seconds"];
  Print[Style["All tests completed. Check results above for any failures.", Blue]];
]

End[]
EndPackage[]

(* Auto-run tests when this file is evaluated *)
CrystallographyTests`RunAllTests[]