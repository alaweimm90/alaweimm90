(* CrystalStructure - Crystal structure representation and analysis *)
(* 
 * Features:
 * - Lattice parameter calculations
 * - Unit cell volume and density
 * - Coordinate transformations (fractional ↔ Cartesian)
 * - Interatomic distance calculations
 * - Miller indices operations
 * - Structure factor calculations
 *
 * Example:
 *   lattice = <|"a" -> 5.0, "b" -> 5.0, "c" -> 5.0, "alpha" -> 90, "beta" -> 90, "gamma" -> 90|>;
 *   atoms = {<|"element" -> "Si", "x" -> 0.0, "y" -> 0.0, "z" -> 0.0|>, <|"element" -> "Si", "x" -> 0.5, "y" -> 0.5, "z" -> 0.5|>};
 *   crystal = CrystalStructure[lattice, atoms];
 *   volume = UnitCellVolume[crystal];
 *)

BeginPackage["CrystalStructure`"]

(* Public symbols *)
CrystalStructure::usage = "CrystalStructure[lattice, atoms] creates a crystal structure object"
UnitCellVolume::usage = "UnitCellVolume[crystal] calculates unit cell volume in Å³"
Density::usage = "Density[crystal, molecularWeight, z] calculates crystal density in g/cm³"
FractionalToCartesian::usage = "FractionalToCartesian[crystal, coords] converts fractional to Cartesian coordinates"
CartesianToFractional::usage = "CartesianToFractional[crystal, coords] converts Cartesian to fractional coordinates"
InteratomicDistance::usage = "InteratomicDistance[crystal, atom1, atom2, includeSymmetry] calculates distance between atoms"
DSpacing::usage = "DSpacing[crystal, h, k, l] calculates d-spacing for Miller indices"
BraggAngle::usage = "BraggAngle[crystal, h, k, l, wavelength] calculates Bragg diffraction angle"
CoordinationNumber::usage = "CoordinationNumber[crystal, atomIdx, cutoffRadius] calculates coordination number"
Supercell::usage = "Supercell[crystal, nx, ny, nz] creates supercell structure"

Begin["`Private`"]

(* Data structure validation *)
ValidateLatticeParameters[lattice_] := Module[{requiredFields},
  requiredFields = {"a", "b", "c", "alpha", "beta", "gamma"};
  
  (* Check all required fields exist *)
  If[!AllTrue[requiredFields, KeyExistsQ[lattice, #] &],
    Message[CrystalStructure::invlat, "Missing required lattice parameters"];
    $Failed,
    (* Validate parameter values *)
    If[!AllTrue[lattice[["a", "b", "c"]], NumericQ[#] && # > 0 &],
      Message[CrystalStructure::invlat, "Lattice parameters a, b, c must be positive numbers"];
      $Failed,
      If[!AllTrue[lattice[["alpha", "beta", "gamma"]], NumericQ[#] && 0 < # < 180 &],
        Message[CrystalStructure::invlat, "Lattice angles must be between 0 and 180 degrees"];
        $Failed,
        lattice
      ]
    ]
  ]
]

ValidateAtomicPositions[atoms_] := Module[{requiredFields},
  requiredFields = {"element", "x", "y", "z"};
  
  If[Length[atoms] == 0,
    Message[CrystalStructure::invatoms, "At least one atom must be provided"];
    $Failed,
    (* Check each atom *)
    If[AllTrue[atoms, 
      AllTrue[requiredFields, KeyExistsQ[#, #2] &] && 
      StringQ[#["element"]] && 
      AllTrue[{#["x"], #["y"], #["z"]}, NumericQ] &],
      (* Add default values for optional fields *)
      Map[Association[# /. {
        atom_ /; !KeyExistsQ[atom, "occupancy"] -> Append[atom, "occupancy" -> 1.0],
        atom_ /; !KeyExistsQ[atom, "thermalFactor"] -> Append[atom, "thermalFactor" -> 0.0]
      }] &, atoms],
      Message[CrystalStructure::invatoms, "Invalid atomic positions"];
      $Failed
    ]
  ]
]

(* Compute fundamental crystallographic matrices *)
ComputeMatrices[lattice_] := Module[{a, b, c, alpha, beta, gamma, cosAlpha, cosBeta, cosGamma, 
  sinAlpha, sinBeta, sinGamma, volume, directMatrix, metricTensor, reciprocalMatrix},
  
  {a, b, c} = lattice /@ {"a", "b", "c"};
  {alpha, beta, gamma} = Degree lattice /@ {"alpha", "beta", "gamma"};
  
  {cosAlpha, cosBeta, cosGamma} = Cos /@ {alpha, beta, gamma};
  {sinAlpha, sinBeta, sinGamma} = Sin /@ {alpha, beta, gamma};
  
  (* Volume calculation *)
  volume = a b c Sqrt[1 - cosAlpha^2 - cosBeta^2 - cosGamma^2 + 2 cosAlpha cosBeta cosGamma];
  
  (* Direct lattice matrix *)
  directMatrix = {
    {a, b cosGamma, c cosBeta},
    {0, b sinGamma, c (cosAlpha - cosBeta cosGamma)/sinGamma},
    {0, 0, volume/(a b sinGamma)}
  };
  
  (* Metric tensor *)
  metricTensor = {
    {a^2, a b cosGamma, a c cosBeta},
    {a b cosGamma, b^2, b c cosAlpha},
    {a c cosBeta, b c cosAlpha, c^2}
  };
  
  (* Reciprocal lattice matrix *)
  reciprocalMatrix = 2 Pi Transpose[Inverse[directMatrix]];
  
  <|"directMatrix" -> directMatrix, "metricTensor" -> metricTensor, 
    "reciprocalMatrix" -> reciprocalMatrix, "volume" -> volume|>
]

(* Main constructor *)
CrystalStructure[lattice_Association, atoms_List] := Module[{validLattice, validAtoms, matrices},
  (* Validate inputs *)
  validLattice = ValidateLatticeParameters[lattice];
  validAtoms = ValidateAtomicPositions[atoms];
  
  If[validLattice === $Failed || validAtoms === $Failed,
    $Failed,
    (* Compute matrices *)
    matrices = ComputeMatrices[validLattice];
    
    (* Return crystal structure object *)
    <|"lattice" -> validLattice, "atoms" -> validAtoms, 
      "directMatrix" -> matrices["directMatrix"],
      "metricTensor" -> matrices["metricTensor"],
      "reciprocalMatrix" -> matrices["reciprocalMatrix"]|>
  ]
]

(* Unit cell volume *)
UnitCellVolume[crystal_Association] := Abs[Det[crystal["directMatrix"]]]

(* Crystal density *)
Density[crystal_Association, molecularWeight_?NumericQ, z_Integer: 1] := Module[{avogadro, volumeCm3},
  avogadro = 6.02214076*10^23; (* mol^-1 *)
  volumeCm3 = UnitCellVolume[crystal] * 10^-24; (* Å³ to cm³ *)
  
  (z * molecularWeight)/(avogadro * volumeCm3)
]

(* Coordinate transformations *)
FractionalToCartesian[crystal_Association, coords_] := Module[{directMatrix},
  directMatrix = crystal["directMatrix"];
  
  If[MatrixQ[coords],
    (* Multiple coordinates *)
    Transpose[directMatrix . Transpose[coords]],
    (* Single coordinate *)
    directMatrix . coords
  ]
]

CartesianToFractional[crystal_Association, coords_] := Module[{invDirect},
  invDirect = Inverse[crystal["directMatrix"]];
  
  If[MatrixQ[coords],
    (* Multiple coordinates *)
    Transpose[invDirect . Transpose[coords]],
    (* Single coordinate *)
    invDirect . coords
  ]
]

(* Interatomic distance calculation *)
InteratomicDistance[crystal_Association, atom1Idx_Integer, atom2Idx_Integer, 
  includeSymmetry_: False] := Module[{atoms, atom1, atom2, df, metricTensor},
  
  atoms = crystal["atoms"];
  
  If[atom1Idx > Length[atoms] || atom2Idx > Length[atoms] || atom1Idx < 1 || atom2Idx < 1,
    Message[InteratomicDistance::invindex, "Atom index out of range"];
    $Failed,
    atom1 = atoms[[atom1Idx]];
    atom2 = atoms[[atom2Idx]];
    
    (* Fractional coordinate difference *)
    df = {atom2["x"] - atom1["x"], atom2["y"] - atom1["y"], atom2["z"] - atom1["z"]};
    
    (* Apply minimum image convention if requested *)
    If[includeSymmetry,
      df = df - Round[df]
    ];
    
    (* Distance using metric tensor *)
    metricTensor = crystal["metricTensor"];
    Sqrt[df . metricTensor . df]
  ]
]

(* d-spacing calculation *)
DSpacing[crystal_Association, h_Integer, k_Integer, l_Integer] := Module[{hkl, reciprocalMetric, gSquared},
  hkl = {h, k, l};
  
  (* Reciprocal metric tensor *)
  reciprocalMetric = Inverse[crystal["metricTensor"]];
  gSquared = hkl . reciprocalMetric . hkl;
  
  If[gSquared <= 0,
    Message[DSpacing::invhkl, "Invalid Miller indices result in zero d-spacing"];
    $Failed,
    1.0/Sqrt[gSquared]
  ]
]

(* Bragg angle calculation *)
BraggAngle[crystal_Association, h_Integer, k_Integer, l_Integer, wavelength_?NumericQ] := Module[{d, sinTheta},
  d = DSpacing[crystal, h, k, l];
  
  If[d === $Failed,
    $Failed,
    sinTheta = wavelength/(2 d);
    
    If[sinTheta > 1,
      Message[BraggAngle::nodiff, 
        StringForm["No diffraction possible for hkl=(``,``,``) at λ=`` Å", h, k, l, wavelength]];
      $Failed,
      ArcSin[sinTheta] / Degree
    ]
  ]
]

(* Coordination number calculation *)
CoordinationNumber[crystal_Association, atomIdx_Integer, cutoffRadius_: 3.0] := Module[{atoms, coordNum, distance},
  atoms = crystal["atoms"];
  
  If[atomIdx > Length[atoms] || atomIdx < 1,
    Message[CoordinationNumber::invindex, "Atom index out of range"];
    $Failed,
    coordNum = 0;
    
    (* Check all other atoms *)
    Do[
      If[i != atomIdx,
        distance = InteratomicDistance[crystal, atomIdx, i, True];
        If[distance <= cutoffRadius,
          coordNum++
        ]
      ],
      {i, Length[atoms]}
    ];
    
    coordNum
  ]
]

(* Supercell creation *)
Supercell[crystal_Association, nx_Integer, ny_Integer, nz_Integer] := Module[{
  lattice, atoms, newLattice, newAtoms, atom, newAtom},
  
  If[nx < 1 || ny < 1 || nz < 1,
    Message[Supercell::invdim, "Supercell dimensions must be positive"];
    $Failed,
    
    lattice = crystal["lattice"];
    atoms = crystal["atoms"];
    
    (* New lattice parameters *)
    newLattice = <|
      "a" -> lattice["a"] * nx,
      "b" -> lattice["b"] * ny,
      "c" -> lattice["c"] * nz,
      "alpha" -> lattice["alpha"],
      "beta" -> lattice["beta"],
      "gamma" -> lattice["gamma"]
    |>;
    
    (* Replicate atoms *)
    newAtoms = {};
    
    Do[
      Do[
        atom = atoms[[atomIdx]];
        newAtom = <|
          "element" -> atom["element"],
          "x" -> (atom["x"] + i)/nx,
          "y" -> (atom["y"] + j)/ny,
          "z" -> (atom["z"] + k)/nz,
          "occupancy" -> atom["occupancy"],
          "thermalFactor" -> atom["thermalFactor"]
        |>;
        AppendTo[newAtoms, newAtom],
        {atomIdx, Length[atoms]}
      ],
      {i, 0, nx-1}, {j, 0, ny-1}, {k, 0, nz-1}
    ];
    
    CrystalStructure[newLattice, newAtoms]
  ]
]

(* Error messages *)
CrystalStructure::invlat = "Invalid lattice parameters: ``"
CrystalStructure::invatoms = "Invalid atomic positions: ``"
InteratomicDistance::invindex = "``"
DSpacing::invhkl = "``"
BraggAngle::nodiff = "``"
CoordinationNumber::invindex = "``"
Supercell::invdim = "``"

End[]
EndPackage[]