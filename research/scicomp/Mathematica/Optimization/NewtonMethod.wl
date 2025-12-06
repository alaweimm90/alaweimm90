(* ::Package:: *)

(* Berkeley SciComp - Optimization Package *)
(* Newton's Method Implementation *)
(* Author: Berkeley SciComp Team *)
(* Date: 2024 *)

BeginPackage["BerkeleySciComp`Optimization`NewtonMethod`"]

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

(* Public functions *)
NewtonMinimize::usage = "NewtonMinimize[f, x0, options] minimizes function f starting from initial point x0 using Newton's method.";
PlotNewtonConvergence::usage = "PlotNewtonConvergence[result] plots the convergence history of Newton optimization.";

(* Options *)
Options[NewtonMinimize] = {
    "MaxIterations" -> 1000,
    "Tolerance" -> 10^-6,
    "LineSearch" -> "Backtrack",
    "HessianModification" -> "Regularization",
    "RegularizationParam" -> 10^-6,
    "Verbose" -> False,
    "Gradient" -> Automatic,
    "Hessian" -> Automatic
};

Begin["`Private`"]

(* Main Newton method function *)
NewtonMinimize[f_, x0_List, opts:OptionsPattern[]] := Module[{
    maxIter, tol, lineSearch, hessModification, regParam, verbose, gradFunc, hessFunc,
    x, grad, hess, history, iter, direction, alpha, xNew, fNew, gradNew, hessNew, hessModified
    },
    
    (* Extract options *)
    maxIter = OptionValue["MaxIterations"];
    tol = OptionValue["Tolerance"];
    lineSearch = OptionValue["LineSearch"];
    hessModification = OptionValue["HessianModification"];
    regParam = OptionValue["RegularizationParam"];
    verbose = OptionValue["Verbose"];
    gradFunc = OptionValue["Gradient"];
    hessFunc = OptionValue["Hessian"];
    
    (* Set up gradient and Hessian functions *)
    If[gradFunc === Automatic,
        gradFunc = Grad[f[Sequence @@ #], #] &
    ];
    
    If[hessFunc === Automatic,
        hessFunc = D[f[Sequence @@ #], {#, 2}] &
    ];
    
    (* Initialize *)
    x = N[x0];
    grad = gradFunc[x];
    hess = hessFunc[x];
    history = {
        "x" -> {x},
        "f" -> {f[Sequence @@ x]},
        "grad" -> {grad},
        "gradNorm" -> {Norm[grad]},
        "conditionNumber" -> {If[MatrixQ[hess], N[LinearAlgebra`MatrixConditionNumber[hess]], Undefined]}
    };
    
    If[verbose, Print["Starting Newton method optimization..."]];
    
    (* Main optimization loop *)
    For[iter = 1, iter <= maxIter && Norm[grad] > tol, iter++,
        (* Modify Hessian if needed *)
        hessModified = ModifyHessian[hess, hessModification, regParam];
        
        (* Compute Newton direction *)
        direction = Quiet[
            Check[
                -LinearSolve[hessModified, grad],
                -grad (* Fallback to steepest descent *)
            ]
        ];
        
        (* Line search *)
        alpha = Switch[lineSearch,
            "Fixed", 1.0,
            "Backtrack", BacktrackLineSearch[f, x, direction, grad],
            "Wolfe", WolfeLineSearch[f, gradFunc, x, direction, grad],
            _, 1.0
        ];
        
        (* Update variables *)
        xNew = x + alpha * direction;
        fNew = f[Sequence @@ xNew];
        gradNew = gradFunc[xNew];
        hessNew = hessFunc[xNew];
        
        (* Store history *)
        history["x"] = Append[history["x"], xNew];
        history["f"] = Append[history["f"], fNew];
        history["grad"] = Append[history["grad"], gradNew];
        history["gradNorm"] = Append[history["gradNorm"], Norm[gradNew]];
        history["conditionNumber"] = Append[history["conditionNumber"], 
            If[MatrixQ[hessNew], N[LinearAlgebra`MatrixConditionNumber[hessNew]], Undefined]];
        
        (* Display progress *)
        If[verbose && Mod[iter, 10] == 0,
            Print["Iteration ", iter, ": f = ", fNew, ", ||grad|| = ", Norm[gradNew], 
                  ", cond(H) = ", If[MatrixQ[hessNew], N[LinearAlgebra`MatrixConditionNumber[hessNew]], "N/A"]]
        ];
        
        (* Update for next iteration *)
        x = xNew;
        grad = gradNew;
        hess = hessNew;
    ];
    
    (* Return result *)
    <|
        "x" -> x,
        "f" -> f[Sequence @@ x],
        "gradient" -> grad,
        "hessian" -> hess,
        "success" -> Norm[grad] <= tol,
        "iterations" -> iter - 1,
        "message" -> If[Norm[grad] <= tol, "Converged", "Maximum iterations reached"],
        "history" -> history
    |>
]

(* Hessian modification methods *)
ModifyHessian[hess_, method_, regParam_] := Switch[method,
    "None", hess,
    "Regularization", hess + regParam * IdentityMatrix[Length[hess]],
    "Eigenvalue", ModifyEigenvalues[hess, regParam],
    "Cholesky", ModifiedCholesky[hess, regParam],
    _, hess + regParam * IdentityMatrix[Length[hess]]
]

(* Eigenvalue modification *)
ModifyEigenvalues[hess_, regParam_] := Module[{
    eigenSys, eigenVecs, eigenVals, modifiedEigenVals
    },
    eigenSys = Eigensystem[hess];
    eigenVecs = eigenSys[[2]];
    eigenVals = eigenSys[[1]];
    
    modifiedEigenVals = Max[#, regParam] & /@ eigenVals;
    
    Transpose[eigenVecs].DiagonalMatrix[modifiedEigenVals].eigenVecs
]

(* Modified Cholesky factorization *)
ModifiedCholesky[hess_, regParam_] := Module[{
    modified, beta
    },
    (* Simple implementation - check if positive definite *)
    modified = hess;
    beta = regParam;
    
    While[!PositiveDefiniteMatrixQ[modified] && beta < 10^6,
        modified = hess + beta * IdentityMatrix[Length[hess]];
        beta *= 10;
    ];
    
    modified
]

(* Backtracking line search *)
BacktrackLineSearch[f_, x_, direction_, grad_] := Module[{
    alpha, c1, rho, f0, gradDotDir
    },
    alpha = 1.0;
    c1 = 10^-4; (* Armijo parameter *)
    rho = 0.5; (* Backtracking parameter *)
    
    f0 = f[Sequence @@ x];
    gradDotDir = grad.direction;
    
    While[f[Sequence @@ (x + alpha * direction)] > f0 + c1 * alpha * gradDotDir && alpha > 10^-10,
        alpha = rho * alpha;
    ];
    
    alpha
]

(* Wolfe line search *)
WolfeLineSearch[f_, gradFunc_, x_, direction_, grad_] := Module[{
    alpha, c1, c2, f0, gradDotDir, xNew, fNew, gradNew, iter
    },
    alpha = 1.0;
    c1 = 10^-4; (* Armijo parameter *)
    c2 = 0.9; (* Curvature parameter *)
    
    f0 = f[Sequence @@ x];
    gradDotDir = grad.direction;
    
    For[iter = 1, iter <= 10, iter++,
        xNew = x + alpha * direction;
        fNew = f[Sequence @@ xNew];
        gradNew = gradFunc[xNew];
        
        (* Check Armijo condition *)
        If[fNew <= f0 + c1 * alpha * gradDotDir,
            (* Check curvature condition *)
            If[gradNew.direction >= c2 * gradDotDir,
                Break[];
            ];
        ];
        
        alpha = 0.5 * alpha;
    ];
    
    alpha
]

(* Convergence plotting function *)
PlotNewtonConvergence[result_Association] := Module[{
    history, iterations, fValues, gradNorms, condNumbers
    },
    
    If[!KeyExistsQ[result, "history"],
        Print["No history data available in result."];
        Return[$Failed];
    ];
    
    history = result["history"];
    iterations = Range[0, Length[history["f"]] - 1];
    fValues = history["f"];
    gradNorms = history["gradNorm"];
    condNumbers = history["conditionNumber"];
    
    GraphicsGrid[{{
        ListLogPlot[
            Transpose[{iterations, fValues}],
            PlotStyle -> {Thick, BerkeleyBlue},
            Frame -> True,
            FrameLabel -> {"Iteration", "Objective Function"},
            PlotLabel -> Style["Newton Method Convergence", BerkeleyBlue, Bold],
            GridLines -> Automatic,
            ImageSize -> 300
        ],
        ListLogPlot[
            Transpose[{iterations, gradNorms}],
            PlotStyle -> {Thick, CaliforniaGold},
            Frame -> True,
            FrameLabel -> {"Iteration", "||Gradient||"},
            PlotLabel -> Style["Gradient Norm", BerkeleyBlue, Bold],
            GridLines -> Automatic,
            ImageSize -> 300
        ]
    }, {
        ListLogPlot[
            Select[Transpose[{iterations, condNumbers}], NumericQ[#[[2]]] &],
            PlotStyle -> {Thick, BerkeleyLightBlue},
            Frame -> True,
            FrameLabel -> {"Iteration", "Condition Number"},
            PlotLabel -> Style["Hessian Condition Number", BerkeleyBlue, Bold],
            GridLines -> Automatic,
            ImageSize -> 300
        ],
        If[Length[result["history"]["x"][[1]]] == 2,
            ListLinePlot[
                result["history"]["x"],
                PlotStyle -> {Thick, Red},
                PlotMarkers -> {Automatic, Small},
                Frame -> True,
                FrameLabel -> {"x\[Subscript]1", "x\[Subscript]2"},
                PlotLabel -> Style["Optimization Path", BerkeleyBlue, Bold],
                GridLines -> Automatic,
                ImageSize -> 300
            ],
            Graphics[{
                Text[Style["Path plot only available\nfor 2D problems", BerkeleyBlue, 12], {0, 0}]
            }, ImageSize -> 300]
        ]
    }}]
]

(* Numerical Hessian calculation *)
NumericalHessian[f_, x_, h_:10^-6] := Module[{
    n, hess, i, j, xPP, xPM, xMP, xMM, xPlus, xMinus, xCenter
    },
    n = Length[x];
    hess = ConstantArray[0., {n, n}];
    xCenter = f[Sequence @@ x];
    
    (* Diagonal elements *)
    For[i = 1, i <= n, i++,
        xPlus = x; xPlus[[i]] += h;
        xMinus = x; xMinus[[i]] -= h;
        
        hess[[i, i]] = (f[Sequence @@ xPlus] - 2*xCenter + f[Sequence @@ xMinus])/h^2;
    ];
    
    (* Off-diagonal elements *)
    For[i = 1, i <= n, i++,
        For[j = i + 1, j <= n, j++,
            xPP = x; xPP[[i]] += h; xPP[[j]] += h;
            xPM = x; xPM[[i]] += h; xPM[[j]] -= h;
            xMP = x; xMP[[i]] -= h; xMP[[j]] += h;
            xMM = x; xMM[[i]] -= h; xMM[[j]] -= h;
            
            hess[[i, j]] = (f[Sequence @@ xPP] - f[Sequence @@ xPM] - f[Sequence @@ xMP] + f[Sequence @@ xMM])/(4*h^2);
            hess[[j, i]] = hess[[i, j]];
        ];
    ];
    
    hess
]

(* Demo function *)
NewtonMethodDemo[] := Module[{
    rosenbrockFunc, rosenbrockGrad, rosenbrockHess, x0, result, result2
    },
    
    Print[Style["Berkeley SciComp - Newton Method Demo", BerkeleyBlue, Bold, 16]];
    Print[StringRepeat["=", 50]];
    
    (* Define Rosenbrock function *)
    rosenbrockFunc[x_, y_] := 100*(y - x^2)^2 + (1 - x)^2;
    rosenbrockGrad[{x_, y_}] := {-400*x*(y - x^2) - 2*(1 - x), 200*(y - x^2)};
    rosenbrockHess[{x_, y_}] := {{-400*(y - 3*x^2) + 2, -400*x}, {-400*x, 200}};
    
    x0 = {-1.2, 1.0};
    
    Print["Test Problem: Rosenbrock Function"];
    Print["f(x,y) = 100*(y - x^2)^2 + (1 - x)^2"];
    Print["Global minimum: f(1,1) = 0"];
    Print["Starting point: ", x0];
    Print[];
    
    (* Test with analytical derivatives *)
    Print["Testing with analytical gradient and Hessian..."];
    result = NewtonMinimize[
        rosenbrockFunc[#1, #2] &,
        x0,
        "MaxIterations" -> 100,
        "Tolerance" -> 10^-6,
        "LineSearch" -> "Backtrack",
        "HessianModification" -> "Regularization",
        "Gradient" -> rosenbrockGrad,
        "Hessian" -> rosenbrockHess,
        "Verbose" -> True
    ];
    
    Print["Result with analytical derivatives:"];
    Print["  Solution: ", result["x"]];
    Print["  Function value: ", result["f"]];
    Print["  Iterations: ", result["iterations"]];
    Print["  Success: ", result["success"]];
    Print[];
    
    (* Test with numerical derivatives *)
    Print["Testing with numerical derivatives..."];
    result2 = NewtonMinimize[
        rosenbrockFunc[#1, #2] &,
        x0,
        "MaxIterations" -> 100,
        "Tolerance" -> 10^-6,
        "LineSearch" -> "Backtrack",
        "HessianModification" -> "Eigenvalue",
        "Verbose" -> False
    ];
    
    Print["Result with numerical derivatives:"];
    Print["  Solution: ", result2["x"]];
    Print["  Function value: ", result2["f"]];
    Print["  Iterations: ", result2["iterations"]];
    Print["  Success: ", result2["success"]];
    Print[];
    
    (* Test different Hessian modifications *)
    Print["Testing different Hessian modifications..."];
    
    modifications = {"Regularization", "Eigenvalue", "Cholesky"};
    Do[
        Print["Testing ", mod, " modification:"];
        result = NewtonMinimize[
            rosenbrockFunc[#1, #2] &,
            x0,
            "MaxIterations" -> 50,
            "Tolerance" -> 10^-6,
            "HessianModification" -> mod,
            "Gradient" -> rosenbrockGrad,
            "Hessian" -> rosenbrockHess,
            "Verbose" -> False
        ];
        Print["  Iterations: ", result["iterations"], ", Success: ", result["success"]];
        ,
    {mod, modifications}];
    
    Print[];
    
    (* Display convergence plots *)
    Print["Generating convergence plots..."];
    PlotNewtonConvergence[result]
]

End[]

EndPackage[]