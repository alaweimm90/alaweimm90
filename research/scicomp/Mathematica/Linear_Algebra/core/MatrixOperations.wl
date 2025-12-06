(* ::Package:: *)

(* Matrix Operations for Scientific Computing *)
(* Comprehensive matrix operations including basic arithmetic, decompositions, *)
(* eigenvalue problems, and specialized algorithms for scientific applications. *)

BeginPackage["MatrixOperations`"]

(* Public function declarations *)
ValidateMatrix::usage = "ValidateMatrix[matrix] validates matrix input and ensures proper format.";
ValidateVector::usage = "ValidateVector[vector] validates vector input and ensures proper format.";
MatrixMultiplyChecked::usage = "MatrixMultiplyChecked[A, B] performs matrix multiplication with dimension checking.";
MatrixAddChecked::usage = "MatrixAddChecked[A, B] performs matrix addition with dimension checking.";
MatrixSubtractChecked::usage = "MatrixSubtractChecked[A, B] performs matrix subtraction with dimension checking.";
MatrixPowerChecked::usage = "MatrixPowerChecked[A, n] computes matrix power A^n with validation.";
TraceChecked::usage = "TraceChecked[A] computes matrix trace with validation.";
DeterminantChecked::usage = "DeterminantChecked[A] computes matrix determinant with validation.";
RankChecked::usage = "RankChecked[A] computes matrix rank with validation.";
ConditionNumberChecked::usage = "ConditionNumberChecked[A] computes condition number with validation.";
FrobeniusNormChecked::usage = "FrobeniusNormChecked[A] computes Frobenius norm with validation.";
SpectralNormChecked::usage = "SpectralNormChecked[A] computes spectral norm with validation.";
NuclearNormChecked::usage = "NuclearNormChecked[A] computes nuclear norm with validation.";

LUDecompositionChecked::usage = "LUDecompositionChecked[A] performs LU decomposition with partial pivoting.";
QRDecompositionChecked::usage = "QRDecompositionChecked[A] performs QR decomposition.";
CholeskyDecompositionChecked::usage = "CholeskyDecompositionChecked[A] performs Cholesky decomposition for SPD matrices.";
SVDChecked::usage = "SVDChecked[A] performs Singular Value Decomposition.";
EigendecompositionChecked::usage = "EigendecompositionChecked[A] performs eigenvalue decomposition.";
SymmetricEigendecompositionChecked::usage = "SymmetricEigendecompositionChecked[A] performs eigenvalue decomposition for symmetric matrices.";
SchurDecompositionChecked::usage = "SchurDecompositionChecked[A] performs Schur decomposition.";

IsSymmetricChecked::usage = "IsSymmetricChecked[A] checks if matrix is symmetric.";
IsHermitianChecked::usage = "IsHermitianChecked[A] checks if matrix is Hermitian.";
IsOrthogonalChecked::usage = "IsOrthogonalChecked[A] checks if matrix is orthogonal.";
IsUnitaryChecked::usage = "IsUnitaryChecked[A] checks if matrix is unitary.";
IsPositiveDefiniteChecked::usage = "IsPositiveDefiniteChecked[A] checks if matrix is positive definite.";
IsPositiveSemidefiniteChecked::usage = "IsPositiveSemidefiniteChecked[A] checks if matrix is positive semidefinite.";
MakeSymmetricChecked::usage = "MakeSymmetricChecked[A] makes matrix symmetric.";
MakeHermitianChecked::usage = "MakeHermitianChecked[A] makes matrix Hermitian.";
GramSchmidtChecked::usage = "GramSchmidtChecked[A] performs Gram-Schmidt orthogonalization.";
HouseholderReflectorChecked::usage = "HouseholderReflectorChecked[x] constructs Householder reflector.";
GivensRotationChecked::usage = "GivensRotationChecked[a, b] constructs Givens rotation.";

CreateTestMatrices::usage = "CreateTestMatrices[] creates test matrices for validation.";

Begin["`Private`"]

(* Matrix validation function *)
ValidateMatrix[matrix_] := Module[{validated},
  If[!MatrixQ[matrix, NumericQ],
    Message[ValidateMatrix::notmatrix, "Input must be a numeric matrix"];
    $Failed,
    If[Length[matrix] == 0 || Length[matrix[[1]]] == 0,
      Message[ValidateMatrix::empty, "Matrix cannot be empty"];
      $Failed,
      matrix
    ]
  ]
]

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

(* Matrix multiplication with dimension checking *)
MatrixMultiplyChecked[A_, B_] := Module[{validA, validB},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[VectorQ[B, NumericQ],
    validB = ValidateVector[B];
    If[validB === $Failed, Return[$Failed]];
    If[Length[A[[1]]] != Length[B],
      Message[MatrixMultiplyChecked::incompatible, 
        "Incompatible dimensions: A[" <> ToString[Dimensions[A]] <> "] \[Times] B[" <> ToString[Length[B]] <> "]"];
      $Failed,
      A.B
    ],
    validB = ValidateMatrix[B];
    If[validB === $Failed, Return[$Failed]];
    If[Length[A[[1]]] != Length[B],
      Message[MatrixMultiplyChecked::incompatible, 
        "Incompatible dimensions: A" <> ToString[Dimensions[A]] <> " \[Times] B" <> ToString[Dimensions[B]]];
      $Failed,
      A.B
    ]
  ]
]

(* Matrix addition with dimension checking *)
MatrixAddChecked[A_, B_] := Module[{validA, validB},
  validA = ValidateMatrix[A];
  validB = ValidateMatrix[B];
  If[validA === $Failed || validB === $Failed, Return[$Failed]];
  
  If[Dimensions[A] != Dimensions[B],
    Message[MatrixAddChecked::incompatible, 
      "Incompatible shapes: A" <> ToString[Dimensions[A]] <> " + B" <> ToString[Dimensions[B]]];
    $Failed,
    A + B
  ]
]

(* Matrix subtraction with dimension checking *)
MatrixSubtractChecked[A_, B_] := Module[{validA, validB},
  validA = ValidateMatrix[A];
  validB = ValidateMatrix[B];
  If[validA === $Failed || validB === $Failed, Return[$Failed]];
  
  If[Dimensions[A] != Dimensions[B],
    Message[MatrixSubtractChecked::incompatible, 
      "Incompatible shapes: A" <> ToString[Dimensions[A]] <> " - B" <> ToString[Dimensions[B]]];
    $Failed,
    A - B
  ]
]

(* Matrix power with validation *)
MatrixPowerChecked[A_, n_Integer] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[MatrixPowerChecked::nonsquare, "Matrix must be square for matrix power"];
    $Failed,
    If[n < 0,
      Message[MatrixPowerChecked::negative, "Power must be non-negative"];
      $Failed,
      If[n == 0,
        IdentityMatrix[Length[A]],
        MatrixPower[A, n]
      ]
    ]
  ]
]

(* Trace with validation *)
TraceChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[TraceChecked::nonsquare, "Matrix must be square to compute trace"];
    $Failed,
    Tr[A]
  ]
]

(* Determinant with validation *)
DeterminantChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[DeterminantChecked::nonsquare, "Matrix must be square to compute determinant"];
    $Failed,
    Det[A]
  ]
]

(* Matrix rank with validation *)
RankChecked[A_, tolerance_:Automatic] := Module[{validA, tol},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  tol = If[tolerance === Automatic,
    Max[Dimensions[A]] * $MachineEpsilon * Max[Abs[Flatten[A]]],
    tolerance
  ];
  
  MatrixRank[A, Tolerance -> tol]
]

(* Condition number with validation *)
ConditionNumberChecked[A_, p_:2] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  Which[
    p == 1, Norm[A, 1] * Norm[PseudoInverse[A], 1],
    p == 2, Norm[A, 2] * Norm[PseudoInverse[A], 2],
    p == Infinity, Norm[A, Infinity] * Norm[PseudoInverse[A], Infinity],
    p == "Frobenius", Norm[A, "Frobenius"] * Norm[PseudoInverse[A], "Frobenius"],
    True, Norm[A, 2] * Norm[PseudoInverse[A], 2]
  ]
]

(* Frobenius norm with validation *)
FrobeniusNormChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  Norm[A, "Frobenius"]
]

(* Spectral norm with validation *)
SpectralNormChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  Norm[A, 2]
]

(* Nuclear norm with validation *)
NuclearNormChecked[A_] := Module[{validA, svd},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  svd = SingularValueDecomposition[A];
  Total[svd[[2]]]
]

(* LU decomposition with validation *)
LUDecompositionChecked[A_] := Module[{validA, lu},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[LUDecompositionChecked::nonsquare, "Matrix must be square for LU decomposition"];
    $Failed,
    lu = LUDecomposition[A];
    {lu[[3]], lu[[1]], lu[[2]]}  (* Return {P, L, U} *)
  ]
]

(* QR decomposition with validation *)
QRDecompositionChecked[A_, mode_:"Full"] := Module[{validA, qr},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[!MemberQ[{"Full", "Economic"}, mode],
    Message[QRDecompositionChecked::invalidmode, "Mode must be 'Full' or 'Economic'"];
    $Failed,
    qr = QRDecomposition[A];
    If[mode == "Economic",
      {qr[[1]][[All, 1 ;; Min[Dimensions[A]]]], qr[[2]][[1 ;; Min[Dimensions[A]]]]},
      qr
    ]
  ]
]

(* Cholesky decomposition with validation *)
CholeskyDecompositionChecked[A_, lower_:True] := Module[{validA, chol},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[CholeskyDecompositionChecked::nonsquare, "Matrix must be square for Cholesky decomposition"];
    Return[$Failed]
  ];
  
  If[Norm[A - ConjugateTranspose[A], "Frobenius"] > 10^(-12),
    Message[CholeskyDecompositionChecked::notsymmetric, "Matrix is not symmetric, results may be unreliable"]
  ];
  
  If[!PositiveDefiniteMatrixQ[A],
    Message[CholeskyDecompositionChecked::notposdef, "Matrix is not positive definite"];
    $Failed,
    chol = CholeskyDecomposition[A];
    If[lower, chol, Transpose[chol]]
  ]
]

(* SVD with validation *)
SVDChecked[A_, fullMatrices_:True, computeUV_:True] := Module[{validA, svd},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[!computeUV,
    {SingularValueList[A]},
    svd = SingularValueDecomposition[A];
    If[fullMatrices,
      svd,
      (* Economic SVD *)
      {svd[[1]][[All, 1 ;; Min[Dimensions[A]]]], 
       Take[svd[[2]], Min[Dimensions[A]]], 
       svd[[3]][[1 ;; Min[Dimensions[A]]]]}
    ]
  ]
]

(* Eigenvalue decomposition with validation *)
EigendecompositionChecked[A_] := Module[{validA, eigs, vals, vecs, sortedIndices},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[EigendecompositionChecked::nonsquare, "Matrix must be square for eigendecomposition"];
    $Failed,
    eigs = Eigensystem[A];
    vals = eigs[[1]];
    vecs = Transpose[eigs[[2]]];
    
    (* Sort by eigenvalue magnitude (descending) *)
    sortedIndices = Reverse[Ordering[Abs[vals]]];
    {vals[[sortedIndices]], vecs[[All, sortedIndices]]}
  ]
]

(* Symmetric eigenvalue decomposition with validation *)
SymmetricEigendecompositionChecked[A_] := Module[{validA, eigs, vals, vecs, sortedIndices},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[SymmetricEigendecompositionChecked::nonsquare, "Matrix must be square"];
    Return[$Failed]
  ];
  
  If[Norm[A - ConjugateTranspose[A], "Frobenius"] > 10^(-12),
    Message[SymmetricEigendecompositionChecked::notsymmetric, "Matrix is not symmetric, results may be unreliable"]
  ];
  
  eigs = Eigensystem[A];
  vals = eigs[[1]];
  vecs = Transpose[eigs[[2]]];
  
  (* Sort in ascending order for symmetric matrices *)
  sortedIndices = Ordering[vals];
  {vals[[sortedIndices]], vecs[[All, sortedIndices]]}
]

(* Schur decomposition with validation *)
SchurDecompositionChecked[A_, outputType_:"Real"] := Module[{validA, schur},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[SchurDecompositionChecked::nonsquare, "Matrix must be square for Schur decomposition"];
    $Failed,
    schur = SchurDecomposition[A];
    {schur[[2]], schur[[1]]}  (* Return {T, Z} where A = Z.T.Z^H *)
  ]
]

(* Check if matrix is symmetric *)
IsSymmetricChecked[A_, tolerance_:10^(-12)] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    False,
    Norm[A - Transpose[A], "Frobenius"] < tolerance
  ]
]

(* Check if matrix is Hermitian *)
IsHermitianChecked[A_, tolerance_:10^(-12)] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    False,
    Norm[A - ConjugateTranspose[A], "Frobenius"] < tolerance
  ]
]

(* Check if matrix is orthogonal *)
IsOrthogonalChecked[A_, tolerance_:10^(-12)] := Module[{validA, ATA, I},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    False,
    ATA = Transpose[A].A;
    I = IdentityMatrix[Length[A]];
    Norm[ATA - I, "Frobenius"] < tolerance
  ]
]

(* Check if matrix is unitary *)
IsUnitaryChecked[A_, tolerance_:10^(-12)] := Module[{validA, AHA, I},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    False,
    AHA = ConjugateTranspose[A].A;
    I = IdentityMatrix[Length[A]];
    Norm[AHA - I, "Frobenius"] < tolerance
  ]
]

(* Check if matrix is positive definite *)
IsPositiveDefiniteChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    False,
    If[!IsSymmetricChecked[A],
      False,
      PositiveDefiniteMatrixQ[A]
    ]
  ]
]

(* Check if matrix is positive semidefinite *)
IsPositiveSemidefiniteChecked[A_] := Module[{validA, eigenvals},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    False,
    If[!IsSymmetricChecked[A],
      False,
      eigenvals = Eigenvalues[A];
      AllTrue[eigenvals, # >= -10^(-12) &]
    ]
  ]
]

(* Make matrix symmetric *)
MakeSymmetricChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[MakeSymmetricChecked::nonsquare, "Matrix must be square"];
    $Failed,
    (A + Transpose[A])/2
  ]
]

(* Make matrix Hermitian *)
MakeHermitianChecked[A_] := Module[{validA},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  If[Length[A] != Length[A[[1]]],
    Message[MakeHermitianChecked::nonsquare, "Matrix must be square"];
    $Failed,
    (A + ConjugateTranspose[A])/2
  ]
]

(* Gram-Schmidt orthogonalization *)
GramSchmidtChecked[A_, normalize_:True] := Module[{validA, result},
  validA = ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  result = Orthogonalize[Transpose[A]];
  If[normalize,
    Transpose[result],
    Transpose[result * Map[Norm, Transpose[result]]]
  ]
]

(* Householder reflector *)
HouseholderReflectorChecked[x_] := Module[{validX, n, e1, normX, v, H},
  validX = ValidateVector[x];
  If[validX === $Failed, Return[$Failed]];
  
  n = Length[x];
  e1 = UnitVector[n, 1];
  normX = Norm[x];
  
  If[normX < 10^(-12),
    IdentityMatrix[n],
    v = x + Sign[x[[1]]] * normX * e1;
    v = v / Norm[v];
    IdentityMatrix[n] - 2 * Outer[Times, v, v]
  ]
]

(* Givens rotation *)
GivensRotationChecked[a_, b_] := Module[{c, s, G},
  If[Abs[b] < 10^(-12),
    {1., 0., {{1., 0.}, {0., 1.}}},
    If[Abs[a] < 10^(-12),
      {0., 1., {{0., 1.}, {-1., 0.}}},
      If[Abs[b] > Abs[a],
        Module[{t = a/b},
          s = 1./Sqrt[1 + t^2];
          c = s * t;
          {c, s, {{c, s}, {-s, c}}}
        ],
        Module[{t = b/a},
          c = 1./Sqrt[1 + t^2];
          s = c * t;
          {c, s, {{c, s}, {-s, c}}}
        ]
      ]
    ]
  ]
]

(* Create test matrices *)
CreateTestMatrices[] := Module[{matrices, A, Q},
  SeedRandom[42];
  matrices = <||>;
  
  (* Random matrices *)
  matrices["random_3x3"] = RandomReal[{-1, 1}, {3, 3}];
  matrices["random_5x5"] = RandomReal[{-1, 1}, {5, 5}];
  
  (* Symmetric matrix *)
  A = RandomReal[{-1, 1}, {4, 4}];
  matrices["symmetric_4x4"] = A + Transpose[A];
  
  (* Positive definite matrix *)
  A = RandomReal[{-1, 1}, {3, 3}];
  matrices["positive_definite_3x3"] = Transpose[A].A + 0.1 * IdentityMatrix[3];
  
  (* Orthogonal matrix (from QR decomposition) *)
  A = RandomReal[{-1, 1}, {4, 4}];
  Q = QRDecomposition[A][[1]];
  matrices["orthogonal_4x4"] = Q;
  
  (* Singular matrix *)
  matrices["singular_3x3"] = {{1, 2, 3}, {2, 4, 6}, {1, 2, 3}};
  
  (* Hilbert matrix (ill-conditioned) *)
  matrices["hilbert_5x5"] = HilbertMatrix[5];
  
  matrices
]

End[]
EndPackage[]