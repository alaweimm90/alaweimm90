(* ::Package:: *)

(* Test Suite for Linear Algebra Package *)
(* Comprehensive tests for all linear algebra functionality *)

BeginPackage["TestLinearAlgebra`"]

Needs["MatrixOperations`"]
Needs["VectorOperations`"]
Needs["LinearSystems`"]

(* Public test functions *)
RunAllTests::usage = "RunAllTests[] runs all linear algebra tests and reports results.";
TestMatrixOperations::usage = "TestMatrixOperations[] tests matrix operations.";
TestVectorOperations::usage = "TestVectorOperations[] tests vector operations.";
TestLinearSystems::usage = "TestLinearSystems[] tests linear system solvers.";

Begin["`Private`"]

(* Test utility functions *)
AssertEqual[actual_, expected_, tolerance_:10^(-12)] := Module[{diff},
  If[NumericQ[actual] && NumericQ[expected],
    diff = Abs[actual - expected];
    If[diff > tolerance,
      Throw[{"FAIL", "Expected: " <> ToString[expected] <> ", Got: " <> ToString[actual] <> ", Difference: " <> ToString[diff]}],
      True
    ],
    If[actual === expected,
      True,
      Throw[{"FAIL", "Expected: " <> ToString[expected] <> ", Got: " <> ToString[actual]}]
    ]
  ]
]

AssertMatrixEqual[actual_, expected_, tolerance_:10^(-12)] := Module[{diff},
  If[Dimensions[actual] != Dimensions[expected],
    Throw[{"FAIL", "Matrix dimensions differ: " <> ToString[Dimensions[actual]] <> " vs " <> ToString[Dimensions[expected]]}]
  ];
  diff = MatrixOperations`FrobeniusNormChecked[actual - expected];
  If[diff > tolerance,
    Throw[{"FAIL", "Matrix difference too large: " <> ToString[diff]}],
    True
  ]
]

AssertVectorEqual[actual_, expected_, tolerance_:10^(-12)] := Module[{diff},
  If[Length[actual] != Length[expected],
    Throw[{"FAIL", "Vector lengths differ: " <> ToString[Length[actual]] <> " vs " <> ToString[Length[expected]]}]
  ];
  diff = Norm[actual - expected];
  If[diff > tolerance,
    Throw[{"FAIL", "Vector difference too large: " <> ToString[diff]}],
    True
  ]
]

AssertTrue[condition_, message_:"Assertion failed"] := 
  If[!condition, Throw[{"FAIL", message}], True]

RunTest[testName_, testFunction_] := Module[{result},
  Print[" Testing: " <> testName];
  result = Catch[testFunction[]; {"PASS", ""}];
  If[result[[1]] == "PASS",
    Print["   ✓ PASSED"],
    Print["   ✗ FAILED: " <> result[[2]]]
  ];
  result[[1]] == "PASS"
]

(* Matrix Operations Tests *)
TestMatrixOperations[] := Module[{passCount = 0, totalCount = 0, testResults},
  Print["Testing Matrix Operations"];
  Print["========================"];
  
  testResults = {
    RunTest["Matrix Validation", TestMatrixValidation],
    RunTest["Basic Arithmetic", TestMatrixArithmetic],
    RunTest["Matrix Properties", TestMatrixProperties],
    RunTest["Matrix Decompositions", TestMatrixDecompositions],
    RunTest["Matrix Norms", TestMatrixNorms],
    RunTest["Special Matrices", TestSpecialMatrices]
  };
  
  passCount = Count[testResults, True];
  totalCount = Length[testResults];
  
  Print["\nMatrix Operations: " <> ToString[passCount] <> "/" <> ToString[totalCount] <> " tests passed"];
  {passCount, totalCount}
]

TestMatrixValidation[] := Module[{},
  (* Test valid matrix *)
  AssertEqual[MatrixOperations`ValidateMatrix[{{1, 2}, {3, 4}}], {{1, 2}, {3, 4}}];
  
  (* Test invalid inputs should return $Failed *)
  AssertEqual[MatrixOperations`ValidateMatrix[{}], $Failed];
  AssertEqual[MatrixOperations`ValidateMatrix[{{}}], $Failed];
]

TestMatrixArithmetic[] := Module[{A, B, result},
  A = {{2, 1}, {1, 3}};
  B = {{1, 2}, {3, 1}};
  
  (* Test matrix addition *)
  result = MatrixOperations`MatrixAddChecked[A, B];
  AssertMatrixEqual[result, {{3, 3}, {4, 4}}];
  
  (* Test matrix subtraction *)
  result = MatrixOperations`MatrixSubtractChecked[A, B];
  AssertMatrixEqual[result, {{1, -1}, {-2, 2}}];
  
  (* Test matrix multiplication *)
  result = MatrixOperations`MatrixMultiplyChecked[A, B];
  AssertMatrixEqual[result, {{5, 5}, {10, 5}}];
  
  (* Test matrix power *)
  result = MatrixOperations`MatrixPowerChecked[A, 2];
  AssertMatrixEqual[result, A.A];
  
  (* Test matrix power A^0 = I *)
  result = MatrixOperations`MatrixPowerChecked[A, 0];
  AssertMatrixEqual[result, IdentityMatrix[2]];
]

TestMatrixProperties[] := Module[{A, result},
  A = {{2, 1}, {1, 3}};
  
  (* Test trace *)
  result = MatrixOperations`TraceChecked[A];
  AssertEqual[result, 5];
  
  (* Test determinant *)
  result = MatrixOperations`DeterminantChecked[A];
  AssertEqual[result, 5];
  
  (* Test rank *)
  result = MatrixOperations`RankChecked[A];
  AssertEqual[result, 2];
  
  (* Test singular matrix rank *)
  result = MatrixOperations`RankChecked[{{1, 2}, {2, 4}}];
  AssertEqual[result, 1];
]

TestMatrixDecompositions[] := Module[{A, result, P, L, U, Q, R},
  A = {{2, 1, 1}, {4, 3, 3}, {8, 7, 9}};
  
  (* Test LU decomposition *)
  result = MatrixOperations`LUDecompositionChecked[A];
  AssertTrue[result =!= $Failed, "LU decomposition should succeed"];
  P = result[[1]]; L = result[[2]]; U = result[[3]];
  AssertMatrixEqual[P.A, L.U, 10^(-10)];
  
  (* Test QR decomposition *)
  result = MatrixOperations`QRDecompositionChecked[A, "Full"];
  AssertTrue[result =!= $Failed, "QR decomposition should succeed"];
  Q = result[[1]]; R = result[[2]];
  AssertMatrixEqual[A, Q.R, 10^(-10)];
  
  (* Test SPD matrix Cholesky *)
  A = {{2, 1}, {1, 2}};
  result = MatrixOperations`CholeskyDecompositionChecked[A, True];
  AssertTrue[result =!= $Failed, "Cholesky decomposition should succeed"];
  L = result;
  AssertMatrixEqual[A, L.ConjugateTranspose[L], 10^(-10)];
]

TestMatrixNorms[] := Module[{A, result},
  A = {{1, 2}, {3, 4}};
  
  (* Test Frobenius norm *)
  result = MatrixOperations`FrobeniusNormChecked[A];
  AssertEqual[result, Sqrt[30], 10^(-12)];
  
  (* Test spectral norm *)
  result = MatrixOperations`SpectralNormChecked[A];
  AssertEqual[result, Norm[A, 2], 10^(-10)];
  
  (* Test condition number *)
  result = MatrixOperations`ConditionNumberChecked[A];
  AssertTrue[result > 1, "Condition number should be > 1"];
]

TestSpecialMatrices[] := Module[{A, ASymm, AOrth, result},
  (* Test symmetric matrix *)
  A = {{1, 2}, {3, 4}};
  ASymm = A + Transpose[A];
  AssertTrue[MatrixOperations`IsSymmetricChecked[ASymm], "Matrix should be symmetric"];
  AssertTrue[!MatrixOperations`IsSymmetricChecked[A], "Matrix should not be symmetric"];
  
  (* Test orthogonal matrix *)
  AOrth = {{1, 0}, {0, 1}};
  AssertTrue[MatrixOperations`IsOrthogonalChecked[AOrth], "Identity should be orthogonal"];
  
  (* Test positive definite *)
  A = {{2, 1}, {1, 2}};
  AssertTrue[MatrixOperations`IsPositiveDefiniteChecked[A], "Matrix should be positive definite"];
]

(* Vector Operations Tests *)
TestVectorOperations[] := Module[{passCount = 0, totalCount = 0, testResults},
  Print["\nTesting Vector Operations"];
  Print["========================="];
  
  testResults = {
    RunTest["Vector Validation", TestVectorValidation],
    RunTest["Vector Arithmetic", TestVectorArithmetic],
    RunTest["Vector Products", TestVectorProducts],
    RunTest["Vector Norms", TestVectorNorms],
    RunTest["Vector Projections", TestVectorProjections],
    RunTest["Gram-Schmidt", TestGramSchmidt]
  };
  
  passCount = Count[testResults, True];
  totalCount = Length[testResults];
  
  Print["\nVector Operations: " <> ToString[passCount] <> "/" <> ToString[totalCount] <> " tests passed"];
  {passCount, totalCount}
]

TestVectorValidation[] := Module[{},
  (* Test valid vector *)
  AssertEqual[VectorOperations`ValidateVector[{1, 2, 3}], {1, 2, 3}];
  
  (* Test invalid inputs *)
  AssertEqual[VectorOperations`ValidateVector[{}], $Failed];
  AssertEqual[VectorOperations`ValidateVector[{{1, 2}}], $Failed];
]

TestVectorArithmetic[] := Module[{u, v, result},
  u = {3, 4};
  v = {1, 2};
  
  (* Test vector addition *)
  result = VectorOperations`VectorAddChecked[u, v];
  AssertVectorEqual[result, {4, 6}];
  
  (* Test vector subtraction *)
  result = VectorOperations`VectorSubtractChecked[u, v];
  AssertVectorEqual[result, {2, 2}];
  
  (* Test scalar multiplication *)
  result = VectorOperations`ScalarMultiplyChecked[2, u];
  AssertVectorEqual[result, {6, 8}];
]

TestVectorProducts[] := Module[{u, v, w, x, result},
  u = {3, 4};
  v = {1, 2};
  w = {1, 0, 0};
  x = {0, 1, 0};
  
  (* Test dot product *)
  result = VectorOperations`DotProductChecked[u, v];
  AssertEqual[result, 11];
  
  (* Test cross product *)
  result = VectorOperations`CrossProductChecked[w, x];
  AssertVectorEqual[result, {0, 0, 1}];
  
  (* Test angle between vectors *)
  result = VectorOperations`AngleBetweenChecked[{1, 0}, {0, 1}, False];
  AssertEqual[result, Pi/2, 10^(-12)];
]

TestVectorNorms[] := Module[{v, result},
  v = {3, 4, 0};
  
  (* Test L1 norm *)
  result = VectorOperations`L1NormChecked[v];
  AssertEqual[result, 7];
  
  (* Test L2 norm *)
  result = VectorOperations`L2NormChecked[v];
  AssertEqual[result, 5];
  
  (* Test L-infinity norm *)
  result = VectorOperations`LInfNormChecked[v];
  AssertEqual[result, 4];
  
  (* Test magnitude *)
  result = VectorOperations`MagnitudeChecked[v];
  AssertEqual[result, 5];
]

TestVectorProjections[] := Module[{u, v, proj, rej, result},
  u = {4, 2};
  v = {3, 0};
  
  (* Test projection *)
  proj = VectorOperations`ProjectChecked[u, v];
  AssertVectorEqual[proj, {4, 0}];
  
  (* Test rejection *)
  rej = VectorOperations`RejectChecked[u, v];
  AssertVectorEqual[rej, {0, 2}];
  
  (* Test orthogonality *)
  result = VectorOperations`DotProductChecked[proj, rej];
  AssertEqual[result, 0, 10^(-12)];
]

TestGramSchmidt[] := Module[{vectors, result, Q, i, j},
  vectors = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
  
  (* Test Gram-Schmidt orthogonalization *)
  result = VectorOperations`GramSchmidtVectorsChecked[vectors, True];
  AssertTrue[result =!= $Failed, "Gram-Schmidt should succeed"];
  
  Q = result;
  
  (* Test orthogonality *)
  For[i = 1, i <= Length[Q], i++,
    For[j = i + 1, j <= Length[Q], j++,
      AssertEqual[VectorOperations`DotProductChecked[Q[[i]], Q[[j]]], 0, 10^(-10)]
    ]
  ];
  
  (* Test normalization *)
  For[i = 1, i <= Length[Q], i++,
    AssertEqual[VectorOperations`MagnitudeChecked[Q[[i]]], 1, 10^(-12)]
  ];
]

(* Linear Systems Tests *)
TestLinearSystems[] := Module[{passCount = 0, totalCount = 0, testResults},
  Print["\nTesting Linear Systems"];
  Print["======================"];
  
  testResults = {
    RunTest["LU Solver", TestLUSolver],
    RunTest["Cholesky Solver", TestCholeskySolver],
    RunTest["QR Solver", TestQRSolver],
    RunTest["SVD Solver", TestSVDSolver],
    RunTest["Iterative Solvers", TestIterativeSolvers],
    RunTest["System Analysis", TestSystemAnalysis]
  };
  
  passCount = Count[testResults, True];
  totalCount = Length[testResults];
  
  Print["\nLinear Systems: " <> ToString[passCount] <> "/" <> ToString[totalCount] <> " tests passed"];
  {passCount, totalCount}
]

TestLUSolver[] := Module[{A, b, result, x},
  A = {{2, 1}, {1, 3}};
  b = {5, 6};
  
  result = LinearSystems`LUSolveChecked[A, b];
  AssertTrue[result["success"] === True, "LU solve should succeed"];
  
  x = result["solution"];
  AssertVectorEqual[A.x, b, 10^(-12)];
]

TestCholeskySolver[] := Module[{A, b, result, x},
  A = {{2, 1}, {1, 2}};
  b = {3, 3};
  
  result = LinearSystems`CholeskySolveChecked[A, b];
  AssertTrue[result["success"] === True, "Cholesky solve should succeed"];
  
  x = result["solution"];
  AssertVectorEqual[A.x, b, 10^(-12)];
]

TestQRSolver[] := Module[{A, b, result, x},
  A = {{1, 1}, {1, 2}, {1, 3}};
  b = {2, 3, 4};
  
  result = LinearSystems`QRSolveChecked[A, b];
  AssertTrue[result["success"] === True, "QR solve should succeed"];
  
  x = result["solution"];
  (* For overdetermined system, check that residual is small *)
  AssertTrue[result["residualNorm"] < 1, "Residual should be reasonable"];
]

TestSVDSolver[] := Module[{A, b, result, x},
  A = {{1, 2}, {2, 4}};  (* Singular matrix *)
  b = {3, 6};
  
  result = LinearSystems`SVDSolveChecked[A, b];
  AssertTrue[result["success"] === True, "SVD solve should succeed"];
  
  x = result["solution"];
  (* For rank-deficient system, check consistency *)
  AssertTrue[result["residualNorm"] < 10^(-10), "Residual should be very small for consistent system"];
]

TestIterativeSolvers[] := Module[{A, b, result},
  (* Create diagonally dominant matrix *)
  A = {{3, 1}, {1, 3}};
  b = {4, 4};
  
  (* Test Jacobi *)
  result = LinearSystems`JacobiSolveChecked[A, b, Automatic, 100, 10^(-8)];
  AssertTrue[result["success"] === True, "Jacobi should converge"];
  
  (* Test Gauss-Seidel *)
  result = LinearSystems`GaussSeidelSolveChecked[A, b, Automatic, 100, 10^(-8)];
  AssertTrue[result["success"] === True, "Gauss-Seidel should converge"];
  
  (* Test Conjugate Gradient *)
  result = LinearSystems`ConjugateGradientChecked[A, b, Automatic, 10, 10^(-8)];
  AssertTrue[result["success"] === True, "CG should converge"];
]

TestSystemAnalysis[] := Module[{A, b, analysis, solver},
  A = {{2, 1}, {1, 2}};
  b = {3, 3};
  
  (* Test system analysis *)
  analysis = LinearSystems`AnalyzeSystemChecked[A, b];
  AssertTrue[analysis =!= $Failed, "System analysis should succeed"];
  AssertTrue[analysis["isSquare"], "Matrix should be detected as square"];
  AssertTrue[analysis["isSymmetric"], "Matrix should be detected as symmetric"];
  
  (* Test solver recommendation *)
  solver = LinearSystems`RecommendSolverChecked[A, b];
  AssertTrue[solver =!= $Failed, "Solver recommendation should succeed"];
  
  (* Test automatic solver *)
  result = LinearSystems`SolveAutoChecked[A, b];
  AssertTrue[result["success"] === True, "Auto solver should succeed"];
]

(* Main test runner *)
RunAllTests[] := Module[{matrixResults, vectorResults, systemResults, totalPass, totalCount},
  Print["Running Linear Algebra Test Suite"];
  Print["==================================="];
  
  SeedRandom[42];  (* For reproducible tests *)
  
  matrixResults = TestMatrixOperations[];
  vectorResults = TestVectorOperations[];
  systemResults = TestLinearSystems[];
  
  totalPass = matrixResults[[1]] + vectorResults[[1]] + systemResults[[1]];
  totalCount = matrixResults[[2]] + vectorResults[[2]] + systemResults[[2]];
  
  Print["\n==================================="];
  Print["Overall Results: " <> ToString[totalPass] <> "/" <> ToString[totalCount] <> " tests passed"];
  
  If[totalPass == totalCount,
    Print["🎉 ALL TESTS PASSED! 🎉"],
    Print["❌ Some tests failed. Please review the output above."]
  ];
  
  {totalPass, totalCount}
]

End[]
EndPackage[]