(* ::Package:: *)

(* Berkeley SciComp - Optimization Package *)
(* Gradient Descent Implementation *)
(* Author: Berkeley SciComp Team *)
(* Date: 2024 *)

BeginPackage["BerkeleySciComp`Optimization`GradientDescent`"]

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

(* Public functions *)
GradientDescentMinimize::usage = "GradientDescentMinimize[f, x0, options] minimizes function f starting from initial point x0 using gradient descent.";
PlotGradientDescentConvergence::usage = "PlotGradientDescentConvergence[result] plots the convergence history of gradient descent optimization.";

(* Options *)
Options[GradientDescentMinimize] = {
    "MaxIterations" -> 1000,
    "Tolerance" -> 10^-6,
    "StepSize" -> 0.01,
    "LineSearch" -> "Backtrack",
    "Momentum" -> 0.0,
    "Verbose" -> False,
    "Gradient" -> Automatic
};

Begin["`Private`"]

(* Main gradient descent function *)
GradientDescentMinimize[f_, x0_List, opts:OptionsPattern[]] := Module[{
    maxIter, tol, stepSize, lineSearch, momentum, verbose, gradFunc,
    x, grad, velocity, history, iter, direction, alpha, xNew, fNew, gradNew
    },
    
    (* Extract options *)
    maxIter = OptionValue["MaxIterations"];
    tol = OptionValue["Tolerance"];
    stepSize = OptionValue["StepSize"];
    lineSearch = OptionValue["LineSearch"];
    momentum = OptionValue["Momentum"];
    verbose = OptionValue["Verbose"];
    gradFunc = OptionValue["Gradient"];
    
    (* Set up gradient function *)
    If[gradFunc === Automatic,
        gradFunc = Grad[f[Sequence @@ #], #] &
    ];
    
    (* Initialize *)
    x = N[x0];
    grad = gradFunc[x];
    velocity = ConstantArray[0., Length[x]];
    history = {
        "x" -> {x},
        "f" -> {f[Sequence @@ x]},
        "grad" -> {grad},
        "gradNorm" -> {Norm[grad]}
    };
    
    If[verbose, Print["Starting gradient descent optimization..."]];
    
    (* Main optimization loop *)
    For[iter = 1, iter <= maxIter && Norm[grad] > tol, iter++,
        (* Compute search direction *)
        direction = -grad;
        
        (* Apply momentum *)
        If[momentum > 0,
            velocity = momentum * velocity + (1 - momentum) * direction;
            direction = velocity;
        ];
        
        (* Line search *)
        alpha = Switch[lineSearch,
            "Fixed", stepSize,
            "Backtrack", BacktrackLineSearch[f, x, direction, grad, stepSize],
            "Wolfe", WolfeLineSearch[f, gradFunc, x, direction, grad],
            _, stepSize
        ];
        
        (* Update variables *)
        xNew = x + alpha * direction;
        fNew = f[Sequence @@ xNew];
        gradNew = gradFunc[xNew];
        
        (* Store history *)
        history["x"] = Append[history["x"], xNew];
        history["f"] = Append[history["f"], fNew];
        history["grad"] = Append[history["grad"], gradNew];
        history["gradNorm"] = Append[history["gradNorm"], Norm[gradNew]];
        
        (* Display progress *)
        If[verbose && Mod[iter, 100] == 0,
            Print["Iteration ", iter, ": f = ", fNew, ", ||grad|| = ", Norm[gradNew]]
        ];
        
        (* Update for next iteration *)
        x = xNew;
        grad = gradNew;
    ];
    
    (* Return result *)
    <|
        "x" -> x,
        "f" -> f[Sequence @@ x],
        "gradient" -> grad,
        "success" -> Norm[grad] <= tol,
        "iterations" -> iter - 1,
        "message" -> If[Norm[grad] <= tol, "Converged", "Maximum iterations reached"],
        "history" -> history
    |>
]

(* Backtracking line search *)
BacktrackLineSearch[f_, x_, direction_, grad_, initialStep_:1.0] := Module[{
    alpha, c1, rho, f0, gradDotDir
    },
    alpha = initialStep;
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
PlotGradientDescentConvergence[result_Association] := Module[{
    history, iterations, fValues, gradNorms
    },
    
    If[!KeyExistsQ[result, "history"],
        Print["No history data available in result."];
        Return[$Failed];
    ];
    
    history = result["history"];
    iterations = Range[0, Length[history["f"]] - 1];
    fValues = history["f"];
    gradNorms = history["gradNorm"];
    
    GraphicsGrid[{{
        ListLogPlot[
            Transpose[{iterations, fValues}],
            PlotStyle -> {Thick, BerkeleyBlue},
            Frame -> True,
            FrameLabel -> {"Iteration", "Objective Function"},
            PlotLabel -> Style["Gradient Descent Convergence", BerkeleyBlue, Bold],
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
    }}]
]

(* Path plotting for 2D problems *)
PlotGradientDescentPath[result_Association, f_, xRange_:{-2, 2}, yRange_:{-2, 2}] := Module[{
    history, path, contourPlot, pathPlot
    },
    
    If[!KeyExistsQ[result, "history"],
        Print["No history data available in result."];
        Return[$Failed];
    ];
    
    history = result["history"];
    path = history["x"];
    
    If[Length[path[[1]]] != 2,
        Print["Path plotting only supported for 2D problems."];
        Return[$Failed];
    ];
    
    (* Create contour plot *)
    contourPlot = ContourPlot[
        f[x, y], {x, xRange[[1]], xRange[[2]]}, {y, yRange[[1]], yRange[[2]]},
        Contours -> 20,
        ContourStyle -> Gray,
        ColorFunction -> (Blend[{BerkeleyLightBlue, CaliforniaGold}, #] &)
    ];
    
    (* Create path plot *)
    pathPlot = ListLinePlot[
        path,
        PlotStyle -> {Thick, Red},
        PlotMarkers -> {Automatic, Medium}
    ];
    
    Show[contourPlot, pathPlot,
        PlotLabel -> Style["Gradient Descent Optimization Path", BerkeleyBlue, Bold],
        Frame -> True,
        FrameLabel -> {"x", "y"},
        ImageSize -> 400
    ]
]

(* Numerical gradient calculation *)
NumericalGradient[f_, x_, h_:10^-8] := Module[{
    n, grad, i, xPlus, xMinus
    },
    n = Length[x];
    grad = ConstantArray[0., n];
    
    For[i = 1, i <= n, i++,
        xPlus = x;
        xMinus = x;
        xPlus[[i]] += h;
        xMinus[[i]] -= h;
        
        grad[[i]] = (f[Sequence @@ xPlus] - f[Sequence @@ xMinus])/(2 * h);
    ];
    
    grad
]

(* Demo function *)
GradientDescentDemo[] := Module[{
    rosenbrockFunc, rosenbrockGrad, x0, result, result2
    },
    
    Print[Style["Berkeley SciComp - Gradient Descent Demo", BerkeleyBlue, Bold, 16]];
    Print[StringRepeat["=", 50]];
    
    (* Define Rosenbrock function *)
    rosenbrockFunc[x_, y_] := 100*(y - x^2)^2 + (1 - x)^2;
    rosenbrockGrad[{x_, y_}] := {-400*x*(y - x^2) - 2*(1 - x), 200*(y - x^2)};
    
    x0 = {-1.2, 1.0};
    
    Print["Test Problem: Rosenbrock Function"];
    Print["f(x,y) = 100*(y - x^2)^2 + (1 - x)^2"];
    Print["Global minimum: f(1,1) = 0"];
    Print["Starting point: ", x0];
    Print[];
    
    (* Test with analytical gradient *)
    Print["Testing with analytical gradient..."];
    result = GradientDescentMinimize[
        rosenbrockFunc[#1, #2] &,
        x0,
        "MaxIterations" -> 2000,
        "Tolerance" -> 10^-6,
        "LineSearch" -> "Backtrack",
        "Gradient" -> rosenbrockGrad,
        "Verbose" -> True
    ];
    
    Print["Result with analytical gradient:"];
    Print["  Solution: ", result["x"]];
    Print["  Function value: ", result["f"]];
    Print["  Iterations: ", result["iterations"]];
    Print["  Success: ", result["success"]];
    Print[];
    
    (* Test with numerical gradient *)
    Print["Testing with numerical gradient..."];
    result2 = GradientDescentMinimize[
        rosenbrockFunc[#1, #2] &,
        x0,
        "MaxIterations" -> 2000,
        "Tolerance" -> 10^-6,
        "LineSearch" -> "Backtrack",
        "Verbose" -> False
    ];
    
    Print["Result with numerical gradient:"];
    Print["  Solution: ", result2["x"]];
    Print["  Function value: ", result2["f"]];
    Print["  Iterations: ", result2["iterations"]];
    Print["  Success: ", result2["success"]];
    Print[];
    
    (* Display convergence plots *)
    Print["Generating convergence plots..."];
    PlotGradientDescentConvergence[result]
]

End[]

EndPackage[]