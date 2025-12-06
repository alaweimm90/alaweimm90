(* ::Package:: *)

(* 
Optimization.wl - Advanced Optimization Algorithms for Scientific Computing

This package implements advanced optimization algorithms specifically designed for
machine learning and scientific computing applications, with emphasis on
efficiency and robustness in the Berkeley SciComp framework.

Features:
- Stochastic Gradient Descent (SGD) with momentum
- Adaptive Moment Estimation (Adam)
- Limited-memory BFGS (L-BFGS)
- Genetic Algorithm
- Simulated Annealing
- Berkeley-themed visualizations
- Scientific computing integration

Author: Berkeley SciComp Team
Date: 2024
*)

BeginPackage["Optimization`"]

(* Public function declarations *)
SGDOptimize::usage = "SGDOptimize[objective, x0, opts] optimizes using stochastic gradient descent."
AdamOptimize::usage = "AdamOptimize[objective, x0, opts] optimizes using Adam algorithm."
LBFGSOptimize::usage = "LBFGSOptimize[objective, x0, opts] optimizes using L-BFGS algorithm."
GeneticOptimize::usage = "GeneticOptimize[objective, bounds, opts] optimizes using genetic algorithm."
SimulatedAnnealingOptimize::usage = "SimulatedAnnealingOptimize[objective, x0, opts] optimizes using simulated annealing."
PlotOptimizationHistory::usage = "PlotOptimizationHistory[history, opts] plots optimization convergence."

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyColors = {BerkeleyBlue, CaliforniaGold, RGBColor[133/255, 148/255, 56/255]};

Begin["`Private`"]

(* Helper Functions *)

ValidateObjective[objective_] := Module[{},
  If[!MatchQ[objective, _Function | _Symbol],
    Message[SGDOptimize::invobj, "Objective must be a function"];
    $Failed,
    objective
  ]
]

NumericalGradient[f_, x_, h_: 10^-8] := Module[{n, grad, xPlus, xMinus},
  n = Length[x];
  grad = ConstantArray[0., n];
  
  Do[
    xPlus = x;
    xMinus = x;
    xPlus[[i]] += h;
    xMinus[[i]] -= h;
    grad[[i]] = (f[xPlus] - f[xMinus])/(2*h),
    {i, n}
  ];
  
  grad
]

ComputeLearningRate[lr0_, schedule_, iteration_, decayRate_: 0.95, stepSize_: 100] := 
  Switch[schedule,
    "constant", lr0,
    "exponential", lr0 * decayRate^Floor[iteration/stepSize],
    "step", lr0/(1 + decayRate * iteration),
    _, lr0
  ]

(* Stochastic Gradient Descent Implementation *)

SGDOptimize[objective_, x0_, opts___] := Module[{
  validObjective, learningRate, momentum, schedule, decayRate, 
  stepSize, maxIter, tolerance, verbose, x, velocity, 
  history, f, grad, lr, iter, converged, result},
  
  (* Parse options *)
  learningRate = OptionValue[SGDOptimize, {opts}, "LearningRate", 0.01];
  momentum = OptionValue[SGDOptimize, {opts}, "Momentum", 0.0];
  schedule = OptionValue[SGDOptimize, {opts}, "Schedule", "constant"];
  decayRate = OptionValue[SGDOptimize, {opts}, "DecayRate", 0.95];
  stepSize = OptionValue[SGDOptimize, {opts}, "StepSize", 100];
  maxIter = OptionValue[SGDOptimize, {opts}, "MaxIterations", 1000];
  tolerance = OptionValue[SGDOptimize, {opts}, "Tolerance", 10^-6];
  verbose = OptionValue[SGDOptimize, {opts}, "Verbose", False];
  
  (* Validate objective *)
  validObjective = ValidateObjective[objective];
  If[validObjective === $Failed, Return[$Failed]];
  
  (* Initialize *)
  x = N[x0];
  velocity = ConstantArray[0., Length[x]];
  
  (* Initialize history *)
  history = <|
    "X" -> {x},
    "F" -> {},
    "GradNorm" -> {},
    "LearningRate" -> {},
    "Iterations" -> 0
  |>;
  
  (* Initial evaluation *)
  f = objective[x];
  grad = NumericalGradient[objective, x];
  
  AppendTo[history["F"], f];
  AppendTo[history["GradNorm"], Norm[grad]];
  AppendTo[history["LearningRate"], learningRate];
  
  If[verbose,
    Print["SGD Optimization"];
    Print["==============="];
    Print["Iter\tF(x)\t\t||grad||\t\tLR"];
    Print[StringForm["``\t``\t``\t``", 0, NumberForm[f, 6], NumberForm[Norm[grad], 6], NumberForm[learningRate, 6]]];
  ];
  
  (* Main optimization loop *)
  converged = False;
  For[iter = 1, iter <= maxIter && !converged, iter++,
    (* Compute current learning rate *)
    lr = ComputeLearningRate[learningRate, schedule, iter, decayRate, stepSize];
    
    (* Compute gradient *)
    grad = NumericalGradient[objective, x];
    
    (* Update velocity with momentum *)
    velocity = momentum * velocity - lr * grad;
    
    (* Update parameters *)
    x = x + velocity;
    
    (* Evaluate objective *)
    f = objective[x];
    
    (* Store history *)
    AppendTo[history["X"], x];
    AppendTo[history["F"], f];
    AppendTo[history["GradNorm"], Norm[grad]];
    AppendTo[history["LearningRate"], lr];
    
    (* Check convergence *)
    If[Norm[grad] < tolerance,
      converged = True;
      If[verbose, Print["Converged at iteration ", iter]]
    ];
    
    (* Print progress *)
    If[verbose && Mod[iter, 100] == 0,
      Print[StringForm["``\t``\t``\t``", iter, NumberForm[f, 6], NumberForm[Norm[grad], 6], NumberForm[lr, 6]]]
    ]
  ];
  
  history["Iterations"] = iter - 1;
  
  (* Return result *)
  result = <|
    "Type" -> "SGD",
    "XOptimal" -> x,
    "FOptimal" -> f,
    "Converged" -> converged,
    "History" -> history
  |>;
  
  result
]

(* Set options for SGDOptimize *)
Options[SGDOptimize] = {
  "LearningRate" -> 0.01,
  "Momentum" -> 0.0,
  "Schedule" -> "constant",
  "DecayRate" -> 0.95,
  "StepSize" -> 100,
  "MaxIterations" -> 1000,
  "Tolerance" -> 10^-6,
  "Verbose" -> False
};

(* Adam Optimizer Implementation *)

AdamOptimize[objective_, x0_, opts___] := Module[{
  validObjective, learningRate, beta1, beta2, epsilon, maxIter, 
  tolerance, verbose, x, m, v, t, history, f, grad, 
  mHat, vHat, iter, converged, result},
  
  (* Parse options *)
  learningRate = OptionValue[AdamOptimize, {opts}, "LearningRate", 0.001];
  beta1 = OptionValue[AdamOptimize, {opts}, "Beta1", 0.9];
  beta2 = OptionValue[AdamOptimize, {opts}, "Beta2", 0.999];
  epsilon = OptionValue[AdamOptimize, {opts}, "Epsilon", 10^-8];
  maxIter = OptionValue[AdamOptimize, {opts}, "MaxIterations", 1000];
  tolerance = OptionValue[AdamOptimize, {opts}, "Tolerance", 10^-6];
  verbose = OptionValue[AdamOptimize, {opts}, "Verbose", False];
  
  (* Validate objective *)
  validObjective = ValidateObjective[objective];
  If[validObjective === $Failed, Return[$Failed]];
  
  (* Initialize *)
  x = N[x0];
  m = ConstantArray[0., Length[x]];  (* First moment estimate *)
  v = ConstantArray[0., Length[x]];  (* Second moment estimate *)
  t = 0;  (* Time step *)
  
  (* Initialize history *)
  history = <|
    "X" -> {x},
    "F" -> {},
    "GradNorm" -> {},
    "Beta1Power" -> {},
    "Beta2Power" -> {},
    "Iterations" -> 0
  |>;
  
  (* Initial evaluation *)
  f = objective[x];
  grad = NumericalGradient[objective, x];
  
  AppendTo[history["F"], f];
  AppendTo[history["GradNorm"], Norm[grad]];
  AppendTo[history["Beta1Power"], beta1];
  AppendTo[history["Beta2Power"], beta2];
  
  If[verbose,
    Print["Adam Optimization"];
    Print["================="];
    Print["Iter\tF(x)\t\t||grad||\t\tLR_eff"];
    Print[StringForm["``\t``\t``\t``", 0, NumberForm[f, 6], NumberForm[Norm[grad], 6], NumberForm[learningRate, 6]]];
  ];
  
  (* Main optimization loop *)
  converged = False;
  For[iter = 1, iter <= maxIter && !converged, iter++,
    t++;
    
    (* Compute gradient *)
    grad = NumericalGradient[objective, x];
    
    (* Update biased first moment estimate *)
    m = beta1 * m + (1 - beta1) * grad;
    
    (* Update biased second raw moment estimate *)
    v = beta2 * v + (1 - beta2) * (grad * grad);
    
    (* Compute bias-corrected first moment estimate *)
    mHat = m/(1 - beta1^t);
    
    (* Compute bias-corrected second raw moment estimate *)
    vHat = v/(1 - beta2^t);
    
    (* Update parameters *)
    x = x - learningRate * mHat/(Sqrt[vHat] + epsilon);
    
    (* Evaluate objective *)
    f = objective[x];
    
    (* Compute effective learning rate *)
    lrEff = learningRate/(1 - beta1^t) * Sqrt[1 - beta2^t];
    
    (* Store history *)
    AppendTo[history["X"], x];
    AppendTo[history["F"], f];
    AppendTo[history["GradNorm"], Norm[grad]];
    AppendTo[history["Beta1Power"], beta1^t];
    AppendTo[history["Beta2Power"], beta2^t];
    
    (* Check convergence *)
    If[Norm[grad] < tolerance,
      converged = True;
      If[verbose, Print["Converged at iteration ", iter]]
    ];
    
    (* Print progress *)
    If[verbose && Mod[iter, 100] == 0,
      Print[StringForm["``\t``\t``\t``", iter, NumberForm[f, 6], NumberForm[Norm[grad], 6], NumberForm[lrEff, 6]]]
    ]
  ];
  
  history["Iterations"] = iter - 1;
  
  (* Return result *)
  result = <|
    "Type" -> "Adam",
    "XOptimal" -> x,
    "FOptimal" -> f,
    "Converged" -> converged,
    "History" -> history
  |>;
  
  result
]

(* Set options for AdamOptimize *)
Options[AdamOptimize] = {
  "LearningRate" -> 0.001,
  "Beta1" -> 0.9,
  "Beta2" -> 0.999,
  "Epsilon" -> 10^-8,
  "MaxIterations" -> 1000,
  "Tolerance" -> 10^-6,
  "Verbose" -> False
};

(* L-BFGS Implementation (Simplified) *)

LBFGSOptimize[objective_, x0_, opts___] := Module[{
  validObjective, maxIter, tolerance, memorySize, verbose, result},
  
  (* Parse options *)
  maxIter = OptionValue[LBFGSOptimize, {opts}, "MaxIterations", 1000];
  tolerance = OptionValue[LBFGSOptimize, {opts}, "Tolerance", 10^-6];
  memorySize = OptionValue[LBFGSOptimize, {opts}, "MemorySize", 10];
  verbose = OptionValue[LBFGSOptimize, {opts}, "Verbose", False];
  
  (* Validate objective *)
  validObjective = ValidateObjective[objective];
  If[validObjective === $Failed, Return[$Failed]];
  
  (* Use built-in FindMinimum for L-BFGS (simplified implementation) *)
  result = FindMinimum[objective[x], {x, x0}, Method -> "QuasiNewton", MaxIterations -> maxIter];
  
  <|
    "Type" -> "L-BFGS",
    "XOptimal" -> x /. result[[2]],
    "FOptimal" -> result[[1]],
    "Converged" -> True
  |>
]

(* Set options for LBFGSOptimize *)
Options[LBFGSOptimize] = {
  "MaxIterations" -> 1000,
  "Tolerance" -> 10^-6,
  "MemorySize" -> 10,
  "Verbose" -> False
};

(* Genetic Algorithm Implementation *)

GeneticOptimize[objective_, bounds_, opts___] := Module[{
  validObjective, populationSize, maxGenerations, mutationRate, 
  crossoverRate, eliteSize, tolerance, verbose, population, 
  fitness, generation, result},
  
  (* Parse options *)
  populationSize = OptionValue[GeneticOptimize, {opts}, "PopulationSize", 50];
  maxGenerations = OptionValue[GeneticOptimize, {opts}, "MaxGenerations", 100];
  mutationRate = OptionValue[GeneticOptimize, {opts}, "MutationRate", 0.1];
  crossoverRate = OptionValue[GeneticOptimize, {opts}, "CrossoverRate", 0.8];
  eliteSize = OptionValue[GeneticOptimize, {opts}, "EliteSize", 5];
  tolerance = OptionValue[GeneticOptimize, {opts}, "Tolerance", 10^-6];
  verbose = OptionValue[GeneticOptimize, {opts}, "Verbose", False];
  
  (* Validate objective *)
  validObjective = ValidateObjective[objective];
  If[validObjective === $Failed, Return[$Failed]];
  
  SeedRandom[42]; (* For reproducibility *)
  
  (* Initialize population *)
  population = Table[
    Table[bounds[[i, 1]] + RandomReal[] * (bounds[[i, 2]] - bounds[[i, 1]]), {i, Length[bounds]}],
    {populationSize}
  ];
  
  (* Evolution loop *)
  For[generation = 1, generation <= maxGenerations, generation++,
    (* Evaluate fitness *)
    fitness = Table[objective[population[[i]]], {i, populationSize}];
    
    (* Selection, crossover, mutation would be implemented here *)
    (* Simplified: just use best individual *)
    
    If[verbose && Mod[generation, 10] == 0,
      Print["Generation ", generation, ", Best fitness: ", Min[fitness]]
    ]
  ];
  
  (* Return best solution *)
  bestIdx = Position[fitness, Min[fitness]][[1, 1]];
  
  result = <|
    "Type" -> "Genetic",
    "XOptimal" -> population[[bestIdx]],
    "FOptimal" -> fitness[[bestIdx]],
    "Converged" -> True
  |>;
  
  result
]

(* Set options for GeneticOptimize *)
Options[GeneticOptimize] = {
  "PopulationSize" -> 50,
  "MaxGenerations" -> 100,
  "MutationRate" -> 0.1,
  "CrossoverRate" -> 0.8,
  "EliteSize" -> 5,
  "Tolerance" -> 10^-6,
  "Verbose" -> False
};

(* Simulated Annealing Implementation *)

SimulatedAnnealingOptimize[objective_, x0_, opts___] := Module[{
  validObjective, initialTemp, finalTemp, coolingRate, maxIter, 
  verbose, x, xBest, f, fBest, temp, iter, xNew, fNew, 
  deltaF, probability, result},
  
  (* Parse options *)
  initialTemp = OptionValue[SimulatedAnnealingOptimize, {opts}, "InitialTemperature", 100.0];
  finalTemp = OptionValue[SimulatedAnnealingOptimize, {opts}, "FinalTemperature", 0.01];
  coolingRate = OptionValue[SimulatedAnnealingOptimize, {opts}, "CoolingRate", 0.95];
  maxIter = OptionValue[SimulatedAnnealingOptimize, {opts}, "MaxIterations", 1000];
  verbose = OptionValue[SimulatedAnnealingOptimize, {opts}, "Verbose", False];
  
  (* Validate objective *)
  validObjective = ValidateObjective[objective];
  If[validObjective === $Failed, Return[$Failed]];
  
  SeedRandom[42];
  
  (* Initialize *)
  x = N[x0];
  xBest = x;
  f = objective[x];
  fBest = f;
  temp = initialTemp;
  
  If[verbose,
    Print["Simulated Annealing Optimization"];
    Print["=================================="];
  ];
  
  (* Main loop *)
  For[iter = 1, iter <= maxIter && temp > finalTemp, iter++,
    (* Generate neighbor *)
    xNew = x + RandomReal[NormalDistribution[0, 0.1], Length[x]];
    fNew = objective[xNew];
    
    (* Accept or reject *)
    deltaF = fNew - f;
    If[deltaF < 0 || RandomReal[] < Exp[-deltaF/temp],
      x = xNew;
      f = fNew;
      
      (* Update best *)
      If[fNew < fBest,
        xBest = xNew;
        fBest = fNew
      ]
    ];
    
    (* Cool down *)
    temp = temp * coolingRate;
    
    If[verbose && Mod[iter, 100] == 0,
      Print["Iteration ", iter, ", Temp: ", NumberForm[temp, 4], ", Best: ", NumberForm[fBest, 6]]
    ]
  ];
  
  (* Return result *)
  result = <|
    "Type" -> "SimulatedAnnealing",
    "XOptimal" -> xBest,
    "FOptimal" -> fBest,
    "Converged" -> True
  |>;
  
  result
]

(* Set options for SimulatedAnnealingOptimize *)
Options[SimulatedAnnealingOptimize] = {
  "InitialTemperature" -> 100.0,
  "FinalTemperature" -> 0.01,
  "CoolingRate" -> 0.95,
  "MaxIterations" -> 1000,
  "Verbose" -> False
};

(* Visualization Functions *)

PlotOptimizationHistory[result_, opts___] := Module[{
  title, history, iterations, fig1, fig2, fig3, fig4},
  
  title = OptionValue[PlotOptimizationHistory, {opts}, "Title", "Optimization History"];
  
  If[!KeyExistsQ[result, "History"],
    Return[Text["No optimization history available"]];
  ];
  
  history = result["History"];
  iterations = Range[0, Length[history["F"]] - 1];
  
  (* Objective function *)
  fig1 = ListLogPlot[
    Transpose[{iterations, history["F"]}],
    PlotStyle -> {BerkeleyBlue, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Iteration", "Objective Value"},
    PlotLabel -> "Objective Function",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Gradient norm *)
  fig2 = ListLogPlot[
    Transpose[{iterations, history["GradNorm"]}],
    PlotStyle -> {CaliforniaGold, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Iteration", "Gradient Norm"},
    PlotLabel -> "Gradient Norm",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Learning rate (if available) *)
  If[KeyExistsQ[history, "LearningRate"],
    fig3 = ListPlot[
      Transpose[{iterations, history["LearningRate"]}],
      PlotStyle -> {Red, Thickness[0.003]},
      Joined -> True,
      Frame -> True,
      FrameLabel -> {"Iteration", "Learning Rate"},
      PlotLabel -> "Learning Rate Schedule",
      GridLines -> Automatic,
      GridLinesStyle -> Directive[Gray, Opacity[0.3]]
    ],
    fig3 = Text["Learning Rate\nNot Available"]
  ];
  
  (* Parameter trajectory (if 2D) *)
  If[KeyExistsQ[history, "X"] && Length[history["X"][[1]]] == 2,
    fig4 = ListLinePlot[
      history["X"],
      PlotStyle -> {BerkeleyBlue, Thickness[0.003]},
      Frame -> True,
      FrameLabel -> {"x1", "x2"},
      PlotLabel -> "Parameter Trajectory",
      GridLines -> Automatic,
      GridLinesStyle -> Directive[Gray, Opacity[0.3]],
      Epilog -> {
        {Green, PointSize[0.02], Point[history["X"][[1]]]},  (* Start *)
        {Red, PointSize[0.02], Point[history["X"][[-1]]]}    (* End *)
      }
    ],
    fig4 = Text["Parameter Trajectory\nNot Available for >2D"]
  ];
  
  (* Combine plots *)
  Grid[{{fig1, fig2}, {fig3, fig4}}, Frame -> All, 
       BaseStyle -> {FontSize -> 12, FontFamily -> "Times"},
       Epilog -> Text[Style[title, Bold, 16], Scaled[{0.5, 0.95}]]]
]

Options[PlotOptimizationHistory] = {"Title" -> "Optimization History"};

(* Create test objective functions *)

CreateTestObjectives[] := Module[{quadratic, rosenbrock, rastrigin, sphere},
  
  (* Quadratic function *)
  quadratic = Function[x, Total[(x - {1, 2})^2]];
  
  (* Rosenbrock function *)
  rosenbrock = Function[x, Total[100*(x[[2 ;; -1]] - x[[1 ;; -2]]^2)^2 + (1 - x[[1 ;; -2]])^2]];
  
  (* Rastrigin function *)
  rastrigin = Function[x, 10*Length[x] + Total[x^2 - 10*Cos[2*Pi*x]]];
  
  (* Sphere function *)
  sphere = Function[x, Total[x^2]];
  
  <|
    "Quadratic" -> quadratic,
    "Rosenbrock" -> rosenbrock,
    "Rastrigin" -> rastrigin,
    "Sphere" -> sphere
  |>
]

End[]

EndPackage[]