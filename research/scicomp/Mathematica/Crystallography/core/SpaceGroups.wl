(* SpaceGroups - Crystallographic space group operations and symmetry *)
(*
 * Features:
 * - Space group definitions and operations
 * - Symmetry operation generation
 * - Equivalent position calculation
 * - Systematic absence prediction
 * - Wyckoff position analysis
 *
 * Example:
 *   sg = SpaceGroup["P m -3 m"];
 *   positions = GenerateEquivalentPositions[sg, {0.25, 0.25, 0.25}];
 *)

BeginPackage["SpaceGroups`"]

(* Public symbols *)
SpaceGroup::usage = "SpaceGroup[symbol] creates a space group object from Hermann-Mauguin symbol"
GenerateEquivalentPositions::usage = "GenerateEquivalentPositions[spaceGroup, position] generates all symmetry equivalent positions"
ApplySymmetryOperation::usage = "ApplySymmetryOperation[operation, position] applies symmetry operation to position"
IsSystematicallyAbsent::usage = "IsSystematicallyAbsent[spaceGroup, h, k, l] checks if reflection is systematically absent"
GetWyckoffPositions::usage = "GetWyckoffPositions[spaceGroup] returns Wyckoff positions for space group"
GetGeneralPosition::usage = "GetGeneralPosition[spaceGroup] returns general position for space group"

Begin["`Private`"]

(* Symmetry operation data structure *)
(* Each operation is represented as {rotationMatrix, translationVector} *)

(* Common rotation matrices *)
identity = IdentityMatrix[3];
inversion = -IdentityMatrix[3];

(* 2-fold rotations *)
rotation2x = {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
rotation2y = {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}};
rotation2z = {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}};

(* 3-fold rotations *)
rotation3xyz = {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}};
rotation3xyzInv = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};

(* 4-fold rotations *)
rotation4z = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
rotation4zInv = {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}};

(* Mirror planes *)
mirrorX = {{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
mirrorY = {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}};
mirrorZ = {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}};

(* Space group database (simplified) *)
spaceGroupDatabase = <|
  "P 1" -> { (* Triclinic P1 *)
    {identity, {0, 0, 0}}
  },
  
  "P -1" -> { (* Triclinic P-1 *)
    {identity, {0, 0, 0}},
    {inversion, {0, 0, 0}}
  },
  
  "P m m m" -> { (* Orthorhombic Pmmm *)
    {identity, {0, 0, 0}},
    {inversion, {0, 0, 0}},
    {mirrorX, {0, 0, 0}},
    {mirrorY, {0, 0, 0}},
    {mirrorZ, {0, 0, 0}},
    {rotation2x, {0, 0, 0}},
    {rotation2y, {0, 0, 0}},
    {rotation2z, {0, 0, 0}}
  },
  
  "P 4/m m m" -> { (* Tetragonal P4/mmm *)
    {identity, {0, 0, 0}},
    {rotation4z, {0, 0, 0}},
    {rotation4z.rotation4z, {0, 0, 0}}, (* 180° rotation *)
    {rotation4zInv, {0, 0, 0}},
    {rotation2x, {0, 0, 0}},
    {rotation2y, {0, 0, 0}},
    {{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}}, {0, 0, 0}}, (* diagonal mirrors *)
    {{{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}}, {0, 0, 0}},
    {inversion, {0, 0, 0}},
    {inversion.rotation4z, {0, 0, 0}},
    {inversion.rotation4z.rotation4z, {0, 0, 0}},
    {inversion.rotation4zInv, {0, 0, 0}},
    {mirrorX, {0, 0, 0}},
    {mirrorY, {0, 0, 0}},
    {mirrorZ, {0, 0, 0}},
    {inversion.{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}}, {0, 0, 0}}
  },
  
  "F m -3 m" -> { (* Cubic Fm-3m *)
    (* Face-centered cubic with full cubic symmetry *)
    (* 192 operations - simplified to key ones *)
    {identity, {0, 0, 0}},
    {inversion, {0, 0, 0}},
    {rotation2x, {0, 0, 0}},
    {rotation2y, {0, 0, 0}},
    {rotation2z, {0, 0, 0}},
    {rotation3xyz, {0, 0, 0}},
    {rotation3xyzInv, {0, 0, 0}},
    {rotation4z, {0, 0, 0}},
    {rotation4zInv, {0, 0, 0}},
    (* Face centering translations *)
    {identity, {1/2, 1/2, 0}},
    {identity, {1/2, 0, 1/2}},
    {identity, {0, 1/2, 1/2}}
  },
  
  "P m -3 m" -> { (* Cubic Pm-3m *)
    (* Primitive cubic with full cubic symmetry *)
    {identity, {0, 0, 0}},
    {inversion, {0, 0, 0}},
    {rotation2x, {0, 0, 0}},
    {rotation2y, {0, 0, 0}},
    {rotation2z, {0, 0, 0}},
    {rotation3xyz, {0, 0, 0}},
    {rotation3xyzInv, {0, 0, 0}},
    {rotation4z, {0, 0, 0}},
    {rotation4zInv, {0, 0, 0}},
    {mirrorX, {0, 0, 0}},
    {mirrorY, {0, 0, 0}},
    {mirrorZ, {0, 0, 0}},
    (* Additional cubic operations *)
    {{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {0, 0, 0}}, (* 3-fold *)
    {{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}, {0, 0, 0}},
    {{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}, {0, 0, 0}},
    {{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}}, {0, 0, 0}}
  }
|>;

(* Wyckoff positions database *)
wyckoffDatabase = <|
  "P 1" -> {
    <|"site" -> "a", "multiplicity" -> 1, "symmetry" -> "1", 
      "positions" -> {{x, y, z}}|>
  },
  
  "P -1" -> {
    <|"site" -> "a", "multiplicity" -> 1, "symmetry" -> "-1", 
      "positions" -> {{0, 0, 0}, {1/2, 1/2, 1/2}}|>,
    <|"site" -> "b", "multiplicity" -> 2, "symmetry" -> "1", 
      "positions" -> {{x, y, z}, {-x, -y, -z}}|>
  },
  
  "P m -3 m" -> {
    <|"site" -> "a", "multiplicity" -> 1, "symmetry" -> "m-3m", 
      "positions" -> {{0, 0, 0}}|>,
    <|"site" -> "b", "multiplicity" -> 3, "symmetry" -> "4/mm.m", 
      "positions" -> {{1/2, 0, 0}, {0, 1/2, 0}, {0, 0, 1/2}}|>,
    <|"site" -> "c", "multiplicity" -> 6, "symmetry" -> "4mm", 
      "positions" -> {{0, 1/2, 1/2}, {1/2, 0, 1/2}, {1/2, 1/2, 0}}|>,
    <|"site" -> "d", "multiplicity" -> 8, "symmetry" -> ".3m", 
      "positions" -> {{1/4, 1/4, 1/4}, {3/4, 3/4, 3/4}}|>
  }
|>;

(* Space group constructor *)
SpaceGroup[symbol_String] := Module[{operations},
  If[KeyExistsQ[spaceGroupDatabase, symbol],
    operations = spaceGroupDatabase[symbol];
    <|"symbol" -> symbol, "operations" -> operations, 
      "wyckoffPositions" -> Lookup[wyckoffDatabase, symbol, {}]|>,
    
    Message[SpaceGroup::unknown, symbol];
    $Failed
  ]
]

(* Apply symmetry operation to position *)
ApplySymmetryOperation[operation_, position_] := Module[{rotation, translation, newPos},
  {rotation, translation} = operation;
  newPos = rotation . position + translation;
  
  (* Bring back to unit cell *)
  Mod[newPos, 1]
]

(* Generate equivalent positions *)
GenerateEquivalentPositions[spaceGroup_Association, position_] := Module[{operations, equivalentPos, uniquePos},
  operations = spaceGroup["operations"];
  
  (* Apply all symmetry operations *)
  equivalentPos = ApplySymmetryOperation[#, position] & /@ operations;
  
  (* Remove duplicates (within tolerance) *)
  uniquePos = {};
  Do[
    If[!AnyTrue[uniquePos, Norm[# - pos] < 1*^-6 &],
      AppendTo[uniquePos, pos]
    ],
    {pos, equivalentPos}
  ];
  
  uniquePos
]

(* Check systematic absences *)
IsSystematicallyAbsent[spaceGroup_Association, h_Integer, k_Integer, l_Integer] := Module[{
  symbol, absenceRules},
  
  symbol = spaceGroup["symbol"];
  
  (* Common systematic absence rules *)
  absenceRules = <|
    "P 1" -> False, (* No systematic absences *)
    "P -1" -> False,
    "P m m m" -> False,
    "P 4/m m m" -> (Mod[h + k, 2] != 0), (* h+k must be even *)
    "F m -3 m" -> (Mod[h + k, 2] != 0 || Mod[h + l, 2] != 0 || Mod[k + l, 2] != 0), (* h+k, h+l, k+l all even *)
    "P m -3 m" -> False
  |>;
  
  Lookup[absenceRules, symbol, False]
]

(* Get Wyckoff positions *)
GetWyckoffPositions[spaceGroup_Association] := spaceGroup["wyckoffPositions"]

(* Get general position *)
GetGeneralPosition[spaceGroup_Association] := Module[{wyckoffPos},
  wyckoffPos = GetWyckoffPositions[spaceGroup];
  
  If[Length[wyckoffPos] > 0,
    Last[wyckoffPos], (* General position is usually last *)
    <|"site" -> "general", "multiplicity" -> Length[spaceGroup["operations"]], 
      "symmetry" -> "1", "positions" -> {{x, y, z}}|>
  ]
]

(* Space group utilities *)
GetLaueClass[spaceGroup_Association] := Module[{symbol},
  symbol = spaceGroup["symbol"];
  
  Which[
    StringContainsQ[symbol, "m -3 m"], "m-3m",
    StringContainsQ[symbol, "4/m"], "4/mmm", 
    StringContainsQ[symbol, "m m m"], "mmm",
    StringContainsQ[symbol, "-1"], "-1",
    True, "1"
  ]
]

GetCrystalSystem[spaceGroup_Association] := Module[{symbol},
  symbol = spaceGroup["symbol"];
  
  Which[
    StringContainsQ[symbol, "m -3"], "cubic",
    StringContainsQ[symbol, "4/m"] || StringContainsQ[symbol, "4 m"], "tetragonal",
    StringContainsQ[symbol, "m m m"], "orthorhombic",
    StringContainsQ[symbol, "-1"], "triclinic",
    True, "triclinic"
  ]
]

(* Calculate structure factor multiplicity *)
CalculateMultiplicity[spaceGroup_Association, h_Integer, k_Integer, l_Integer] := Module[{
  operations, equivalentReflections, uniqueReflections},
  
  operations = spaceGroup["operations"];
  
  (* Generate equivalent reflections *)
  equivalentReflections = Map[
    Function[op, 
      Module[{rotation = op[[1]]}, 
        Transpose[rotation] . {h, k, l}
      ]
    ],
    operations
  ];
  
  (* Remove duplicates *)
  uniqueReflections = DeleteDuplicates[equivalentReflections];
  
  Length[uniqueReflections]
]

(* Error messages *)
SpaceGroup::unknown = "Unknown space group symbol: ``"

End[]
EndPackage[]