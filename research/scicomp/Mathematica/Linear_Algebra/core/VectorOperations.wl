(* ::Package:: *)

(* Vector Operations for Scientific Computing *)
(* Comprehensive vector operations including norms, products, projections, *)
(* and orthogonalization algorithms for scientific applications. *)

BeginPackage["VectorOperations`"]

(* Public function declarations *)
ValidateVector::usage = "ValidateVector[vector] validates vector input and ensures proper format.";
VectorAddChecked::usage = "VectorAddChecked[u, v] performs vector addition with validation.";
VectorSubtractChecked::usage = "VectorSubtractChecked[u, v] performs vector subtraction with validation.";
ScalarMultiplyChecked::usage = "ScalarMultiplyChecked[scalar, u] performs scalar multiplication with validation.";
DotProductChecked::usage = "DotProductChecked[u, v] computes dot product with validation.";
CrossProductChecked::usage = "CrossProductChecked[u, v] computes cross product for 3D vectors.";
MagnitudeChecked::usage = "MagnitudeChecked[u] computes vector magnitude with validation.";
NormalizeChecked::usage = "NormalizeChecked[u] normalizes vector with validation.";
AngleBetweenChecked::usage = "AngleBetweenChecked[u, v, degrees] computes angle between vectors.";
ProjectChecked::usage = "ProjectChecked[u, v] projects vector u onto vector v.";
RejectChecked::usage = "RejectChecked[u, v] computes rejection (orthogonal component) of u from v.";
PNormChecked::usage = "PNormChecked[u, p] computes p-norm of vector.";
L1NormChecked::usage = "L1NormChecked[u] computes L1 (Manhattan) norm.";
L2NormChecked::usage = "L2NormChecked[u] computes L2 (Euclidean) norm.";
LInfNormChecked::usage = "LInfNormChecked[u] computes L-infinity (maximum) norm.";
VectorDistanceChecked::usage = "VectorDistanceChecked[u, v, p] computes distance between vectors.";
GramSchmidtVectorsChecked::usage = "GramSchmidtVectorsChecked[vectors, normalize] performs Gram-Schmidt orthogonalization.";
QRVectorsChecked::usage = "QRVectorsChecked[vectors] performs QR decomposition of vector list.";

CreateTestVectors::usage = "CreateTestVectors[] creates test vectors for validation.";

Begin["`Private`"]

(* Vector validation function *)
ValidateVector[vector_] := Module[{validated},
  If[!VectorQ[vector, NumericQ],
    Message[ValidateVector::notvector, "Input must be a numeric vector"];
    $Failed,
    If[Length[vector] == 0,
      Message[ValidateVector::empty, "Vector cannot be empty"];
      $Failed,
      vector
    ]
  ]
]

(* Vector addition with validation *)
VectorAddChecked[u_, v_] := Module[{validU, validV},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != Length[v],
    Message[VectorAddChecked::incompatible, 
      "Incompatible vector lengths: " <> ToString[Length[u]] <> " != " <> ToString[Length[v]]];
    $Failed,
    u + v
  ]
]

(* Vector subtraction with validation *)
VectorSubtractChecked[u_, v_] := Module[{validU, validV},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != Length[v],
    Message[VectorSubtractChecked::incompatible, 
      "Incompatible vector lengths: " <> ToString[Length[u]] <> " != " <> ToString[Length[v]]];
    $Failed,
    u - v
  ]
]

(* Scalar multiplication with validation *)
ScalarMultiplyChecked[scalar_, u_] := Module[{validU},
  If[!NumericQ[scalar],
    Message[ScalarMultiplyChecked::notscalar, "First argument must be a scalar"];
    Return[$Failed]
  ];
  
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  scalar * u
]

(* Dot product with validation *)
DotProductChecked[u_, v_] := Module[{validU, validV},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != Length[v],
    Message[DotProductChecked::incompatible, 
      "Incompatible vector lengths for dot product: " <> ToString[Length[u]] <> " != " <> ToString[Length[v]]];
    $Failed,
    u.v
  ]
]

(* Cross product with validation (3D vectors only) *)
CrossProductChecked[u_, v_] := Module[{validU, validV},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != 3 || Length[v] != 3,
    Message[CrossProductChecked::not3d, 
      "Cross product requires 3D vectors, got lengths: " <> ToString[Length[u]] <> ", " <> ToString[Length[v]]];
    $Failed,
    Cross[u, v]
  ]
]

(* Vector magnitude with validation *)
MagnitudeChecked[u_] := Module[{validU},
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  Norm[u]
]

(* Vector normalization with validation *)
NormalizeChecked[u_] := Module[{validU, mag},
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  mag = Norm[u];
  If[mag < 10^(-12),
    Message[NormalizeChecked::zerovector, "Cannot normalize zero vector"];
    $Failed,
    u / mag
  ]
]

(* Angle between vectors with validation *)
AngleBetweenChecked[u_, v_, degrees_:False] := Module[{validU, validV, dotProd, magU, magV, cosTheta},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != Length[v],
    Message[AngleBetweenChecked::incompatible, 
      "Incompatible vector lengths: " <> ToString[Length[u]] <> " != " <> ToString[Length[v]]];
    Return[$Failed]
  ];
  
  magU = Norm[u];
  magV = Norm[v];
  
  If[magU < 10^(-12) || magV < 10^(-12),
    Message[AngleBetweenChecked::zerovector, "Cannot compute angle with zero vector"];
    Return[$Failed]
  ];
  
  dotProd = u.v;
  cosTheta = dotProd / (magU * magV);
  
  (* Clamp to [-1, 1] to avoid numerical errors *)
  cosTheta = Max[-1, Min[1, cosTheta]];
  
  If[degrees,
    ArcCos[cosTheta] * 180 / Pi,
    ArcCos[cosTheta]
  ]
]

(* Vector projection with validation *)
ProjectChecked[u_, v_] := Module[{validU, validV, vDotV, uDotV},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != Length[v],
    Message[ProjectChecked::incompatible, 
      "Incompatible vector lengths: " <> ToString[Length[u]] <> " != " <> ToString[Length[v]]];
    Return[$Failed]
  ];
  
  vDotV = v.v;
  If[vDotV < 10^(-12),
    Message[ProjectChecked::zerovector, "Cannot project onto zero vector"];
    $Failed,
    uDotV = u.v;
    (uDotV / vDotV) * v
  ]
]

(* Vector rejection with validation *)
RejectChecked[u_, v_] := Module[{validU, validV, proj},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  proj = ProjectChecked[u, v];
  If[proj === $Failed, Return[$Failed]];
  
  u - proj
]

(* P-norm with validation *)
PNormChecked[u_, p_:2] := Module[{validU},
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  If[!NumericQ[p] || p <= 0,
    Message[PNormChecked::invalidp, "p must be a positive number"];
    Return[$Failed]
  ];
  
  Which[
    p == 1, Total[Abs[u]],
    p == 2, Norm[u],
    p == Infinity, Max[Abs[u]],
    True, (Total[Abs[u]^p])^(1/p)
  ]
]

(* L1 norm with validation *)
L1NormChecked[u_] := Module[{validU},
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  Total[Abs[u]]
]

(* L2 norm with validation *)
L2NormChecked[u_] := Module[{validU},
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  Norm[u]
]

(* L-infinity norm with validation *)
LInfNormChecked[u_] := Module[{validU},
  validU = ValidateVector[u];
  If[validU === $Failed, Return[$Failed]];
  
  Max[Abs[u]]
]

(* Vector distance with validation *)
VectorDistanceChecked[u_, v_, p_:2] := Module[{validU, validV, diff},
  validU = ValidateVector[u];
  validV = ValidateVector[v];
  If[validU === $Failed || validV === $Failed, Return[$Failed]];
  
  If[Length[u] != Length[v],
    Message[VectorDistanceChecked::incompatible, 
      "Incompatible vector lengths: " <> ToString[Length[u]] <> " != " <> ToString[Length[v]]];
    Return[$Failed]
  ];
  
  diff = u - v;
  PNormChecked[diff, p]
]

(* Gram-Schmidt orthogonalization for list of vectors *)
GramSchmidtVectorsChecked[vectors_, normalize_:True] := Module[{validVectors, result, v, proj, i, j},
  (* Validate all vectors *)
  validVectors = {};
  For[i = 1, i <= Length[vectors], i++,
    v = ValidateVector[vectors[[i]]];
    If[v === $Failed, Return[$Failed]];
    AppendTo[validVectors, v]
  ];
  
  (* Check consistent dimensions *)
  If[Length[Union[Length /@ validVectors]] > 1,
    Message[GramSchmidtVectorsChecked::inconsistent, "All vectors must have the same dimension"];
    Return[$Failed]
  ];
  
  result = {};
  
  For[i = 1, i <= Length[validVectors], i++,
    v = validVectors[[i]];
    
    (* Subtract projections onto previous orthogonal vectors *)
    For[j = 1, j <= Length[result], j++,
      proj = ProjectChecked[v, result[[j]]];
      If[proj === $Failed, Return[$Failed]];
      v = v - proj
    ];
    
    (* Check for linear dependence *)
    If[Norm[v] < 10^(-12),
      Message[GramSchmidtVectorsChecked::dependent, 
        "Vector " <> ToString[i] <> " is linearly dependent on previous vectors"];
      Continue[]
    ];
    
    (* Normalize if requested *)
    If[normalize,
      v = NormalizeChecked[v];
      If[v === $Failed, Return[$Failed]]
    ];
    
    AppendTo[result, v]
  ];
  
  result
]

(* QR decomposition of vector list *)
QRVectorsChecked[vectors_] := Module[{Q, R, n, m, i, j, proj, coeff},
  Q = GramSchmidtVectorsChecked[vectors, True];
  If[Q === $Failed, Return[$Failed]];
  
  n = Length[vectors];
  m = Length[Q];
  
  (* Construct R matrix *)
  R = Table[0, {i, 1, m}, {j, 1, n}];
  
  For[j = 1, j <= n, j++,
    For[i = 1, i <= Min[j, m], i++,
      coeff = DotProductChecked[Q[[i]], vectors[[j]]];
      If[coeff === $Failed, Return[$Failed]];
      R[[i, j]] = coeff
    ]
  ];
  
  {Q, R}
]

(* Create test vectors *)
CreateTestVectors[] := Module[{vectors},
  SeedRandom[42];
  vectors = <||>;
  
  (* 2D vectors *)
  vectors["2d_unit_x"] = {1., 0.};
  vectors["2d_unit_y"] = {0., 1.};
  vectors["2d_random"] = RandomReal[{-1, 1}, 2];
  vectors["2d_orthogonal"] = {{3., 4.}, {-4., 3.}};
  
  (* 3D vectors *)
  vectors["3d_unit_x"] = {1., 0., 0.};
  vectors["3d_unit_y"] = {0., 1., 0.};
  vectors["3d_unit_z"] = {0., 0., 1.};
  vectors["3d_random"] = RandomReal[{-1, 1}, 3];
  
  (* High-dimensional vector *)
  vectors["5d_random"] = RandomReal[{-1, 1}, 5];
  
  (* Zero vector *)
  vectors["zero_3d"] = {0., 0., 0.};
  
  (* Linearly dependent set *)
  v1 = {1., 2., 3.};
  v2 = {2., 1., 1.};
  v3 = v1 + 2*v2;  (* Linearly dependent *)
  vectors["dependent_set"] = {v1, v2, v3};
  
  (* Linearly independent set *)
  vectors["independent_set"] = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
  
  vectors
]

End[]
EndPackage[]