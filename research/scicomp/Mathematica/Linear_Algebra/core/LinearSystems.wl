(* ::Package:: *)

(* Linear System Solvers for Scientific Computing *)
(* Comprehensive linear system solvers including direct methods, iterative methods, *)
(* system analysis, and automatic solver selection for scientific applications. *)

BeginPackage["LinearSystems`"]

Needs["MatrixOperations`"]

(* Public function declarations *)
LUSolveChecked::usage = "LUSolveChecked[A, b] solves Ax = b using LU decomposition.";
CholeskySolveChecked::usage = "CholeskySolveChecked[A, b] solves Ax = b using Cholesky decomposition for SPD matrices.";
QRSolveChecked::usage = "QRSolveChecked[A, b] solves Ax = b using QR decomposition (supports overdetermined systems).";
SVDSolveChecked::usage = "SVDSolveChecked[A, b] solves Ax = b using SVD (handles rank-deficient systems).";
JacobiSolveChecked::usage = "JacobiSolveChecked[A, b] solves Ax = b using Jacobi iteration.";
GaussSeidelSolveChecked::usage = "GaussSeidelSolveChecked[A, b] solves Ax = b using Gauss-Seidel iteration.";
ConjugateGradientChecked::usage = "ConjugateGradientChecked[A, b] solves Ax = b using Conjugate Gradient for SPD matrices.";
AnalyzeSystemChecked::usage = "AnalyzeSystemChecked[A, b] analyzes linear system properties.";
RecommendSolverChecked::usage = "RecommendSolverChecked[A, b] recommends appropriate solver.";
SolveAutoChecked::usage = "SolveAutoChecked[A, b] automatically selects and applies appropriate solver.";

CreateTestSystems::usage = "CreateTestSystems[] creates test linear systems for validation.";

Begin["`Private`"]

(* LU solve with validation *)
LUSolveChecked[A_, b_, opts___] := Module[{validA, validB, solution, residual, residualNorm},
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[LUSolveChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[A[[1]]],
    Message[LUSolveChecked::nonsquare, "Matrix must be square for LU solve"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[validB],
    Message[LUSolveChecked::incompatible, 
      "Matrix and vector dimensions incompatible: " <> ToString[Length[A]] <> " != " <> ToString[Length[validB]]];
    Return[$Failed]
  ];
  
  Try[
    solution = LinearSolve[A, validB];
    residual = A.solution - validB;
    residualNorm = Norm[residual];
    
    <|"solution" -> solution, 
      "success" -> True, 
      "iterations" -> 1, 
      "residualNorm" -> residualNorm,
      "info" -> <|"method" -> "LU", "conditionNumber" -> MatrixOperations`ConditionNumberChecked[A]|>|>,
    
    <|"solution" -> ConstantArray[Undefined, Length[A]], 
      "success" -> False, 
      "iterations" -> 0, 
      "residualNorm" -> Infinity,
      "info" -> <|"method" -> "LU", "error" -> "LU solve failed"|>|>
  ]
]

(* Cholesky solve with validation *)
CholeskySolveChecked[A_, b_, lower_:True] := Module[{validA, validB, L, solution, residual, residualNorm},
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[CholeskySolveChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[A[[1]]],
    Message[CholeskySolveChecked::nonsquare, "Matrix must be square for Cholesky solve"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[validB],
    Message[CholeskySolveChecked::incompatible, 
      "Matrix and vector dimensions incompatible: " <> ToString[Length[A]] <> " != " <> ToString[Length[validB]]];
    Return[$Failed]
  ];
  
  (* Check symmetry *)
  If[Norm[A - ConjugateTranspose[A], "Frobenius"] > 10^(-12),
    Message[CholeskySolveChecked::notsymmetric, "Matrix is not symmetric, Cholesky may fail"]
  ];
  
  Try[
    L = MatrixOperations`CholeskyDecompositionChecked[A, lower];
    If[L === $Failed, Return[$Failed]];
    
    If[lower,
      (* Solve L y = b, then L^T x = y *)
      solution = LinearSolve[ConjugateTranspose[L], LinearSolve[L, validB]],
      (* U is upper triangular *)
      solution = LinearSolve[L, LinearSolve[ConjugateTranspose[L], validB]]
    ];
    
    residual = A.solution - validB;
    residualNorm = Norm[residual];
    
    <|"solution" -> solution, 
      "success" -> True, 
      "iterations" -> 1, 
      "residualNorm" -> residualNorm,
      "info" -> <|"method" -> "Cholesky", "conditionNumber" -> MatrixOperations`ConditionNumberChecked[A]|>|>,
    
    <|"solution" -> ConstantArray[Undefined, Length[A]], 
      "success" -> False, 
      "iterations" -> 0, 
      "residualNorm" -> Infinity,
      "info" -> <|"method" -> "Cholesky", "error" -> "Cholesky solve failed"|>|>
  ]
]

(* QR solve with validation *)
QRSolveChecked[A_, b_, mode_:"Full"] := Module[{validA, validB, qr, Q, R, QtB, solution, residual, residualNorm},
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[QRSolveChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[validB],
    Message[QRSolveChecked::incompatible, 
      "Matrix and vector dimensions incompatible: " <> ToString[Length[A]] <> " != " <> ToString[Length[validB]]];
    Return[$Failed]
  ];
  
  Try[
    qr = MatrixOperations`QRDecompositionChecked[A, mode];
    If[qr === $Failed, Return[$Failed]];
    
    Q = qr[[1]];
    R = qr[[2]];
    
    (* Solve R x = Q^T b *)
    QtB = ConjugateTranspose[Q].validB;
    
    If[StringMatchQ[mode, "Economic"],
      solution = LinearSolve[R, QtB],
      solution = LinearSolve[R[[1 ;; Dimensions[A][[2]]]], QtB[[1 ;; Dimensions[A][[2]]]]]
    ];
    
    residual = A.solution - validB;
    residualNorm = Norm[residual];
    
    <|"solution" -> solution, 
      "success" -> True, 
      "iterations" -> 1, 
      "residualNorm" -> residualNorm,
      "info" -> <|"method" -> "QR", 
                   "conditionNumber" -> MatrixOperations`ConditionNumberChecked[A],
                   "overdetermined" -> (Length[A] > Length[A[[1]]])|>|>,
    
    <|"solution" -> ConstantArray[Undefined, Length[A[[1]]]], 
      "success" -> False, 
      "iterations" -> 0, 
      "residualNorm" -> Infinity,
      "info" -> <|"method" -> "QR", "error" -> "QR solve failed"|>|>
  ]
]

(* SVD solve with validation *)
SVDSolveChecked[A_, b_, tolerance_:Automatic] := Module[{validA, validB, svd, U, S, V, s, tol, r, solution, residual, residualNorm},
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[SVDSolveChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[validB],
    Message[SVDSolveChecked::incompatible, 
      "Matrix and vector dimensions incompatible: " <> ToString[Length[A]] <> " != " <> ToString[Length[validB]]];
    Return[$Failed]
  ];
  
  Try[
    svd = MatrixOperations`SVDChecked[A, True, True];
    If[svd === $Failed, Return[$Failed]];
    
    U = svd[[1]];
    s = svd[[2]];
    V = svd[[3]];
    
    tol = If[tolerance === Automatic,
      Max[Dimensions[A]] * $MachineEpsilon * Max[s],
      tolerance
    ];
    
    (* Compute pseudoinverse solution *)
    r = Count[s, x_ /; x > tol];
    solution = V[[All, 1 ;; r]].((ConjugateTranspose[U[[All, 1 ;; r]]].validB) / s[[1 ;; r]]);
    
    (* Compute residual *)
    residual = A.solution - validB;
    residualNorm = If[Length[A] >= Length[A[[1]]], Norm[residual], 0];
    
    <|"solution" -> solution, 
      "success" -> True, 
      "iterations" -> 1, 
      "residualNorm" -> residualNorm,
      "info" -> <|"method" -> "SVD", 
                   "rank" -> r,
                   "singularValues" -> s,
                   "conditionNumber" -> If[s[[-1]] > tol, s[[1]] / s[[-1]], Infinity]|>|>,
    
    <|"solution" -> ConstantArray[Undefined, Length[A[[1]]]], 
      "success" -> False, 
      "iterations" -> 0, 
      "residualNorm" -> Infinity,
      "info" -> <|"method" -> "SVD", "error" -> "SVD solve failed"|>|>
  ]
]

(* Jacobi iterative solve *)
JacobiSolveChecked[A_, b_, x0_:Automatic, maxIterations_:1000, tolerance_:10^(-6)] := Module[
  {validA, validB, x, n, diagA, R, residuals, iteration, xNew, residual, residualNorm},
  
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[JacobiSolveChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[A[[1]]],
    Message[JacobiSolveChecked::nonsquare, "Matrix must be square for Jacobi iteration"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[validB],
    Message[JacobiSolveChecked::incompatible, 
      "Matrix and vector dimensions incompatible"];
    Return[$Failed]
  ];
  
  n = Length[A];
  x = If[x0 === Automatic, ConstantArray[0., n], x0];
  
  (* Check diagonal elements *)
  diagA = Diagonal[A];
  If[AnyTrue[Abs[diagA], # < 10^(-12) &],
    Message[JacobiSolveChecked::nearzerodiagg, "Matrix has near-zero diagonal elements"]
  ];
  
  (* Extract off-diagonal part *)
  R = A - DiagonalMatrix[diagA];
  
  residuals = {};
  
  For[iteration = 1, iteration <= maxIterations, iteration++,
    (* Jacobi update: x^(k+1) = D^(-1) * (b - R * x^(k)) *)
    xNew = (validB - R.x) / diagA;
    
    (* Check convergence *)
    residual = A.xNew - validB;
    residualNorm = Norm[residual];
    AppendTo[residuals, residualNorm];
    
    If[residualNorm < tolerance,
      Return[<|"solution" -> xNew, 
               "success" -> True, 
               "iterations" -> iteration, 
               "residualNorm" -> residualNorm,
               "info" -> <|"method" -> "Jacobi", "residuals" -> residuals|>|>]
    ];
    
    x = xNew
  ];
  
  <|"solution" -> x, 
    "success" -> False, 
    "iterations" -> maxIterations, 
    "residualNorm" -> Last[residuals],
    "info" -> <|"method" -> "Jacobi", "residuals" -> residuals, "converged" -> False|>|>
]

(* Gauss-Seidel iterative solve *)
GaussSeidelSolveChecked[A_, b_, x0_:Automatic, maxIterations_:1000, tolerance_:10^(-6)] := Module[
  {validA, validB, x, n, residuals, iteration, i, sumAx, residual, residualNorm, xOld},
  
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[GaussSeidelSolveChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[A[[1]]],
    Message[GaussSeidelSolveChecked::nonsquare, "Matrix must be square for Gauss-Seidel iteration"];
    Return[$Failed]
  ];
  
  n = Length[A];
  x = If[x0 === Automatic, ConstantArray[0., n], x0];
  
  residuals = {};
  
  For[iteration = 1, iteration <= maxIterations, iteration++,
    xOld = x;
    
    (* Gauss-Seidel update *)
    For[i = 1, i <= n, i++,
      If[Abs[A[[i, i]]] < 10^(-12),
        Message[GaussSeidelSolveChecked::nearzerodiagg, 
          "Near-zero diagonal element at position " <> ToString[i]];
        Continue[]
      ];
      
      sumAx = Total[A[[i, 1 ;; i - 1]] * x[[1 ;; i - 1]]] + Total[A[[i, i + 1 ;; n]] * xOld[[i + 1 ;; n]]];
      x[[i]] = (validB[[i]] - sumAx) / A[[i, i]]
    ];
    
    (* Check convergence *)
    residual = A.x - validB;
    residualNorm = Norm[residual];
    AppendTo[residuals, residualNorm];
    
    If[residualNorm < tolerance,
      Return[<|"solution" -> x, 
               "success" -> True, 
               "iterations" -> iteration, 
               "residualNorm" -> residualNorm,
               "info" -> <|"method" -> "Gauss-Seidel", "residuals" -> residuals|>|>]
    ]
  ];
  
  <|"solution" -> x, 
    "success" -> False, 
    "iterations" -> maxIterations, 
    "residualNorm" -> Last[residuals],
    "info" -> <|"method" -> "Gauss-Seidel", "residuals" -> residuals, "converged" -> False|>|>
]

(* Conjugate Gradient solve *)
ConjugateGradientChecked[A_, b_, x0_:Automatic, maxIterations_:Automatic, tolerance_:10^(-6)] := Module[
  {validA, validB, x, maxIter, r, p, rsold, residuals, iteration, Ap, alpha, rsnew, beta, residualNorm},
  
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[ConjugateGradientChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  If[Length[A] != Length[A[[1]]],
    Message[ConjugateGradientChecked::nonsquare, "Matrix must be square for CG"];
    Return[$Failed]
  ];
  
  (* Check symmetry *)
  If[Norm[A - ConjugateTranspose[A], "Frobenius"] > 10^(-12),
    Message[ConjugateGradientChecked::notsymmetric, "Matrix is not symmetric, CG may not converge"]
  ];
  
  x = If[x0 === Automatic, ConstantArray[0., Length[A]], x0];
  maxIter = If[maxIterations === Automatic, Length[A], maxIterations];
  
  (* Initialize CG *)
  r = validB - A.x;
  p = r;
  rsold = r.r;
  
  residuals = {};
  
  For[iteration = 1, iteration <= maxIter, iteration++,
    Ap = A.p;
    alpha = rsold / (p.Ap);
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = r.r;
    
    residualNorm = Sqrt[rsnew];
    AppendTo[residuals, residualNorm];
    
    If[residualNorm < tolerance,
      Return[<|"solution" -> x, 
               "success" -> True, 
               "iterations" -> iteration, 
               "residualNorm" -> residualNorm,
               "info" -> <|"method" -> "CG", "residuals" -> residuals|>|>]
    ];
    
    beta = rsnew / rsold;
    p = r + beta * p;
    rsold = rsnew
  ];
  
  <|"solution" -> x, 
    "success" -> False, 
    "iterations" -> maxIter, 
    "residualNorm" -> Last[residuals],
    "info" -> <|"method" -> "CG", "residuals" -> residuals, "converged" -> False|>|>
]

(* System analysis *)
AnalyzeSystemChecked[A_, b_] := Module[{validA, validB, analysis, eigenvals, diagVals, offDiagSums},
  validA = MatrixOperations`ValidateMatrix[A];
  If[validA === $Failed, Return[$Failed]];
  
  validB = If[VectorQ[b], b, Flatten[b]];
  If[!VectorQ[validB, NumericQ],
    Message[AnalyzeSystemChecked::invalidb, "b must be a numeric vector"];
    Return[$Failed]
  ];
  
  analysis = <||>;
  analysis["matrixShape"] = Dimensions[A];
  analysis["vectorLength"] = Length[validB];
  analysis["isSquare"] = (Length[A] == Length[A[[1]]]);
  analysis["isOverdetermined"] = (Length[A] > Length[A[[1]]]);
  analysis["isUnderdetermined"] = (Length[A] < Length[A[[1]]]);
  
  If[analysis["isSquare"],
    (* Square matrix analysis *)
    analysis["determinant"] = Det[A];
    analysis["isSingular"] = (Abs[analysis["determinant"]] < 10^(-12));
    analysis["conditionNumber"] = MatrixOperations`ConditionNumberChecked[A];
    analysis["isWellConditioned"] = (analysis["conditionNumber"] < 10^12);
    
    (* Symmetry check *)
    analysis["isSymmetric"] = (Norm[A - ConjugateTranspose[A], "Frobenius"] < 10^(-12));
    
    If[analysis["isSymmetric"],
      eigenvals = Eigenvalues[A];
      analysis["isPositiveDefinite"] = AllTrue[eigenvals, # > 10^(-12) &];
      analysis["isPositiveSemidefinite"] = AllTrue[eigenvals, # >= -10^(-12) &]
    ];
    
    (* Diagonal dominance *)
    diagVals = Abs[Diagonal[A]];
    offDiagSums = Total[Abs[A], {2}] - diagVals;
    analysis["isDiagonallyDominant"] = AllTrue[MapThread[GreaterEqual, {diagVals, offDiagSums}], Identity]
  ];
  
  analysis
]

(* Solver recommendation *)
RecommendSolverChecked[A_, b_] := Module[{analysis},
  analysis = AnalyzeSystemChecked[A, b];
  If[analysis === $Failed, Return[$Failed]];
  
  Which[
    !analysis["isSquare"], "QRSolveChecked",
    analysis["isSingular"], "SVDSolveChecked",
    KeyExistsQ[analysis, "isPositiveDefinite"] && analysis["isPositiveDefinite"], "CholeskySolveChecked",
    !analysis["isWellConditioned"], "SVDSolveChecked",
    analysis["matrixShape"][[1]] > 1000 && KeyExistsQ[analysis, "isDiagonallyDominant"] && analysis["isDiagonallyDominant"], "GaussSeidelSolveChecked",
    True, "LUSolveChecked"
  ]
]

(* Automatic solver selection *)
SolveAutoChecked[A_, b_, opts___] := Module[{solverName},
  solverName = RecommendSolverChecked[A, b];
  If[solverName === $Failed, Return[$Failed]];
  
  Switch[solverName,
    "LUSolveChecked", LUSolveChecked[A, b, opts],
    "CholeskySolveChecked", CholeskySolveChecked[A, b, opts],
    "QRSolveChecked", QRSolveChecked[A, b, opts],
    "SVDSolveChecked", SVDSolveChecked[A, b, opts],
    "GaussSeidelSolveChecked", GaussSeidelSolveChecked[A, b, opts],
    _, LUSolveChecked[A, b, opts]
  ]
]

(* Create test systems *)
CreateTestSystems[] := Module[{systems, A, b},
  SeedRandom[42];
  systems = <||>;
  
  (* Well-conditioned system *)
  A = RandomReal[{-1, 1}, {5, 5}];
  A = A + 0.1 * IdentityMatrix[5];
  b = RandomReal[{-1, 1}, 5];
  systems["wellConditioned"] = <|"A" -> A, "b" -> b|>;
  
  (* Symmetric positive definite *)
  A = RandomReal[{-1, 1}, {4, 4}];
  A = Transpose[A].A + 0.1 * IdentityMatrix[4];
  b = RandomReal[{-1, 1}, 4];
  systems["symmetricPD"] = <|"A" -> A, "b" -> b|>;
  
  (* Overdetermined system *)
  A = RandomReal[{-1, 1}, {8, 5}];
  b = RandomReal[{-1, 1}, 8];
  systems["overdetermined"] = <|"A" -> A, "b" -> b|>;
  
  (* Ill-conditioned (Hilbert matrix) *)
  A = HilbertMatrix[5];
  b = ConstantArray[1., 5];
  systems["illConditioned"] = <|"A" -> A, "b" -> b|>;
  
  systems
]

End[]
EndPackage[]