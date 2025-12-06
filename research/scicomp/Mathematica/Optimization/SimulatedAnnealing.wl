(* ::Package:: *)

(* Berkeley SciComp - Optimization Package *)
(* Simulated Annealing Implementation *)
(* Author: Berkeley SciComp Team *)
(* Date: 2024 *)

BeginPackage["BerkeleySciComp`Optimization`SimulatedAnnealing`"]

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

(* Public functions *)
SimulatedAnnealingMinimize::usage = "SimulatedAnnealingMinimize[f, bounds, options] minimizes function f using simulated annealing within given bounds.";
PlotSimulatedAnnealingConvergence::usage = "PlotSimulatedAnnealingConvergence[result] plots the convergence history of simulated annealing optimization.";

(* Options *)
Options[SimulatedAnnealingMinimize] = {
    "MaxIterations" -> 10000,
    "InitialTemperature" -> 100,
    "FinalTemperature" -> 10^-8,
    "CoolingSchedule" -> "Exponential",
    "StepSize" -> 1.0,
    "InitialPoint" -> Automatic,
    "Verbose" -> False
};

Begin["`Private`"]

(* Main simulated annealing function *)
SimulatedAnnealingMinimize[f_, bounds_List, opts:OptionsPattern[]] := Module[{
    maxIter, T0, Tf, coolingSchedule, stepSize, x0, verbose,
    x, fVal, bestX, bestF, history, iter, xNew, fNew, temperature,
    accepted, acceptanceCount
    },
    
    (* Extract options *)
    maxIter = OptionValue["MaxIterations"];
    T0 = OptionValue["InitialTemperature"];
    Tf = OptionValue["FinalTemperature"];
    coolingSchedule = OptionValue["CoolingSchedule"];
    stepSize = OptionValue["StepSize"];
    x0 = OptionValue["InitialPoint"];
    verbose = OptionValue["Verbose"];
    
    (* Initialize starting point *)
    If[x0 === Automatic,
        x = Table[bounds[[i, 1]] + RandomReal[]*(bounds[[i, 2]] - bounds[[i, 1]]), {i, Length[bounds]}];
        ,
        x = N[x0];
    ];
    
    (* Initial evaluation *)
    fVal = f[Sequence @@ x];
    bestX = x;
    bestF = fVal;
    acceptanceCount = 0;
    
    (* Initialize history *)
    history = {
        "x" -> {x},
        "f" -> {fVal},
        "bestX" -> {bestX},
        "bestF" -> {bestF},
        "temperature" -> {T0},
        "accepted" -> {}
    };
    
    If[verbose, Print["Starting simulated annealing optimization..."]];
    
    temperature = T0;
    
    (* Main optimization loop *)
    For[iter = 1, iter <= maxIter && temperature > Tf, iter++,
        (* Generate new candidate solution *)
        xNew = GenerateCandidate[x, bounds, stepSize];
        fNew = f[Sequence @@ xNew];
        
        (* Acceptance criterion *)
        accepted = AcceptCandidate[fVal, fNew, temperature];
        
        If[accepted,
            x = xNew;
            fVal = fNew;
            acceptanceCount++;
            
            (* Update best solution *)
            If[fNew < bestF,
                bestX = xNew;
                bestF = fNew;
            ];
        ];
        
        (* Update temperature *)
        temperature = UpdateTemperature[T0, Tf, iter, maxIter, coolingSchedule, history["accepted"]];
        
        (* Store history *)
        history["x"] = Append[history["x"], x];
        history["f"] = Append[history["f"], fVal];
        history["bestX"] = Append[history["bestX"], bestX];
        history["bestF"] = Append[history["bestF"], bestF];
        history["temperature"] = Append[history["temperature"], temperature];
        history["accepted"] = Append[history["accepted"], accepted];
        
        (* Display progress *)
        If[verbose && Mod[iter, 1000] == 0,
            acceptanceRate = N[acceptanceCount/iter];
            Print["Iteration ", iter, ": f = ", fVal, ", best = ", bestF, 
                  ", T = ", temperature, ", acc_rate = ", acceptanceRate];
        ];
    ];
    
    (* Return result *)
    <|
        "x" -> bestX,
        "f" -> bestF,
        "success" -> True,
        "iterations" -> iter - 1,
        "finalTemperature" -> temperature,
        "acceptanceRate" -> N[acceptanceCount/(iter - 1)],
        "message" -> "Simulated annealing completed",
        "history" -> history
    |>
]

(* Generate new candidate solution *)
GenerateCandidate[x_, bounds_, stepSize_] := Module[{
    xNew, i, n
    },
    n = Length[x];
    xNew = x + stepSize * RandomVariate[NormalDistribution[0, 1], n];
    
    (* Apply bounds *)
    For[i = 1, i <= n, i++,
        If[xNew[[i]] < bounds[[i, 1]],
            xNew[[i]] = bounds[[i, 1]] + RandomReal[]*(bounds[[i, 2]] - bounds[[i, 1]]);
        ];
        If[xNew[[i]] > bounds[[i, 2]],
            xNew[[i]] = bounds[[i, 1]] + RandomReal[]*(bounds[[i, 2]] - bounds[[i, 1]]);
        ];
    ];
    
    xNew
]

(* Acceptance criterion *)
AcceptCandidate[fCurrent_, fNew_, temperature_] := Module[{
    deltaF, probability
    },
    If[fNew < fCurrent,
        (* Always accept better solutions *)
        True,
        (* Accept worse solutions with probability exp(-ΔF/T) *)
        deltaF = fNew - fCurrent;
        probability = Exp[-deltaF/temperature];
        RandomReal[] < probability
    ]
]

(* Temperature update functions *)
UpdateTemperature[T0_, Tf_, iteration_, maxIter_, schedule_, acceptedHistory_] := Switch[schedule,
    "Linear",
    (* Linear cooling *)
    T0*(1 - iteration/maxIter),
    
    "Exponential",
    (* Exponential cooling *)
    T0*Exp[Log[Tf/T0]*iteration/maxIter],
    
    "Logarithmic",
    (* Logarithmic cooling *)
    T0/Log[1 + iteration],
    
    "Power",
    (* Power law cooling *)
    T0*(0.95^iteration),
    
    "Adaptive",
    (* Adaptive cooling based on acceptance rate *)
    If[iteration > 100,
        recentAcceptance = Count[Take[acceptedHistory, -100], True]/100.;
        alpha = Which[
            recentAcceptance > 0.8, 0.99, (* Cool faster *)
            recentAcceptance < 0.2, 0.999, (* Cool slower *)
            True, 0.995 (* Standard cooling *)
        ];
        Max[T0*(alpha^iteration), Tf],
        T0
    ],
    
    _,
    (* Default to exponential *)
    T0*Exp[Log[Tf/T0]*iteration/maxIter]
]

(* Convergence plotting function *)
PlotSimulatedAnnealingConvergence[result_Association] := Module[{
    history, iterations, fValues, bestValues, temperatures, accepted, acceptanceRate
    },
    
    If[!KeyExistsQ[result, "history"],
        Print["No history data available in result."];
        Return[$Failed];
    ];
    
    history = result["history"];
    iterations = Range[0, Length[history["f"]] - 1];
    fValues = history["f"];
    bestValues = history["bestF"];
    temperatures = history["temperature"];
    accepted = history["accepted"];
    
    (* Calculate rolling acceptance rate *)
    acceptanceRate = MovingAverage[Boole[accepted], Min[1000, Length[accepted]]];
    
    GraphicsGrid[{{
        ListLogPlot[{
            Transpose[{iterations, bestValues}],
            Transpose[{iterations, fValues}]
        },
            PlotStyle -> {{Thick, BerkeleyBlue}, {Thin, Opacity[0.3], CaliforniaGold}},
            Frame -> True,
            FrameLabel -> {"Iteration", "Objective Function"},
            PlotLabel -> Style["Simulated Annealing Convergence", BerkeleyBlue, Bold],
            PlotLegends -> {"Best", "Current"},
            GridLines -> Automatic,
            ImageSize -> 300
        ],
        LogLogPlot[
            Interpolation[Transpose[{iterations, temperatures}]][t], 
            {t, 0, Max[iterations]},
            PlotStyle -> {Thick, CaliforniaGold},
            Frame -> True,
            FrameLabel -> {"Iteration", "Temperature"},
            PlotLabel -> Style["Cooling Schedule", BerkeleyBlue, Bold],
            GridLines -> Automatic,
            ImageSize -> 300
        ]
    }, {
        ListLinePlot[
            Transpose[{Range[Length[acceptanceRate]], acceptanceRate}],
            PlotStyle -> {Thick, BerkeleyLightBlue},
            Frame -> True,
            FrameLabel -> {"Iteration", "Acceptance Rate"},
            PlotLabel -> Style["Acceptance Rate (Rolling Average)", BerkeleyBlue, Bold],
            GridLines -> Automatic,
            PlotRange -> {All, {0, 1}},
            ImageSize -> 300
        ],
        If[Length[result["history"]["x"][[1]]] == 2,
            ListLinePlot[{
                result["history"]["x"],
                result["history"]["bestX"]
            },
                PlotStyle -> {{Thin, Opacity[0.3], Gray}, {Thick, BerkeleyBlue}},
                PlotMarkers -> {None, {Automatic, Small}},
                Frame -> True,
                FrameLabel -> {"x\[Subscript]1", "x\[Subscript]2"},
                PlotLabel -> Style["Optimization Path", BerkeleyBlue, Bold],
                PlotLegends -> {"Current", "Best"},
                GridLines -> Automatic,
                ImageSize -> 300
            ],
            Graphics[{
                Text[Style["Path plot only available\nfor 2D problems", BerkeleyBlue, 12], {0, 0}]
            }, ImageSize -> 300]
        ]
    }}]
]

(* Test function definitions *)
AckleyFunction[x_List] := -20*Exp[-0.2*Sqrt[0.5*Total[x^2]]] - Exp[0.5*Total[Cos[2*Pi*x]]] + E + 20;
RastriginFunction[x_List] := 10*Length[x] + Total[x^2 - 10*Cos[2*Pi*x]];
RosenbrockFunction[x_List] := Total[100*(x[[2;;]] - x[[;;-2]]^2)^2 + (1 - x[[;;-2]])^2];

(* Demo function *)
SimulatedAnnealingDemo[] := Module[{
    ackley, rastrigin, rosenbrock, bounds2D, bounds10D, result1, result2, result3
    },
    
    Print[Style["Berkeley SciComp - Simulated Annealing Demo", BerkeleyBlue, Bold, 16]];
    Print[StringRepeat["=", 60]];
    
    bounds2D = {{-5, 5}, {-5, 5}};
    bounds10D = Table[{-5, 5}, {10}];
    
    (* Test Problem 1: Ackley Function (2D) *)
    Print["Test Problem 1: Ackley Function (2D)"];
    Print["Global minimum: f(0,0) = 0"];
    Print[];
    
    result1 = SimulatedAnnealingMinimize[
        AckleyFunction,
        bounds2D,
        "MaxIterations" -> 10000,
        "InitialTemperature" -> 100,
        "CoolingSchedule" -> "Exponential",
        "Verbose" -> True
    ];
    
    Print["Result:"];
    Print["  Solution: ", result1["x"]];
    Print["  Function value: ", result1["f"]];
    Print["  Iterations: ", result1["iterations"]];
    Print["  Final temperature: ", result1["finalTemperature"]];
    Print["  Acceptance rate: ", result1["acceptanceRate"]];
    Print[];
    
    (* Test Problem 2: Rastrigin Function (2D) *)
    Print["Test Problem 2: Rastrigin Function (2D)"];
    Print["Global minimum: f(0,0) = 0"];
    Print[];
    
    result2 = SimulatedAnnealingMinimize[
        RastriginFunction,
        bounds2D,
        "MaxIterations" -> 15000,
        "InitialTemperature" -> 50,
        "CoolingSchedule" -> "Adaptive",
        "Verbose" -> False
    ];
    
    Print["Result:"];
    Print["  Solution: ", result2["x"]];
    Print["  Function value: ", result2["f"]];
    Print["  Iterations: ", result2["iterations"]];
    Print["  Acceptance rate: ", result2["acceptanceRate"]];
    Print[];
    
    (* Test Problem 3: Rosenbrock Function (10D) *)
    Print["Test Problem 3: Rosenbrock Function (10D)"];
    Print["Global minimum: f([1,1,...,1]) = 0"];
    Print[];
    
    result3 = SimulatedAnnealingMinimize[
        RosenbrockFunction,
        bounds10D,
        "MaxIterations" -> 20000,
        "InitialTemperature" -> 200,
        "CoolingSchedule" -> "Power",
        "StepSize" -> 0.5,
        "Verbose" -> False
    ];
    
    Print["Result:"];
    Print["  Function value: ", result3["f"]];
    Print["  Solution error: ", Norm[result3["x"] - ConstantArray[1, 10]]];
    Print["  Iterations: ", result3["iterations"]];
    Print["  Acceptance rate: ", result3["acceptanceRate"]];
    Print[];
    
    (* Compare different cooling schedules *)
    Print["Comparing cooling schedules on Ackley function..."];
    Print[];
    
    schedules = {"Linear", "Exponential", "Logarithmic", "Power", "Adaptive"};
    Do[
        Print["Testing ", schedule, " cooling:"];
        result = SimulatedAnnealingMinimize[
            AckleyFunction,
            bounds2D,
            "MaxIterations" -> 5000,
            "InitialTemperature" -> 50,
            "CoolingSchedule" -> schedule,
            "Verbose" -> False
        ];
        Print["  Function value: ", result["f"], ", Acceptance rate: ", result["acceptanceRate"]];
        ,
    {schedule, schedules}];
    
    Print[];
    Print["Generating convergence plots..."];
    PlotSimulatedAnnealingConvergence[result1]
]

End[]

EndPackage[]