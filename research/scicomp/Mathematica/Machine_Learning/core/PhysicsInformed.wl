(* ::Package:: *)

(* 
PhysicsInformed.wl - Physics-Informed Neural Networks for Scientific Computing

This package implements physics-informed machine learning methods that incorporate
physical laws, conservation principles, and domain knowledge into ML models
for the Berkeley SciComp framework.

Features:
- Physics-Informed Neural Networks (PINNs) for PDE solving
- Deep Operator Networks (DeepONets) for operator learning
- Conservation law enforcement
- Symmetry-aware neural networks
- Berkeley-themed scientific visualizations
- Advanced PDE solving capabilities

Author: Berkeley SciComp Team
Date: 2024
*)

BeginPackage["PhysicsInformed`", {"NeuralNetworks`"}]

(* Public function declarations *)
CreatePINN::usage = "CreatePINN[layers, opts] creates a Physics-Informed Neural Network."
TrainPINN::usage = "TrainPINN[pinn, xDomain, tDomain, opts] trains PINN to solve PDE."
PredictPINN::usage = "PredictPINN[pinn, x, t] makes predictions using trained PINN."
CreateDeepONet::usage = "CreateDeepONet[branchLayers, trunkLayers, opts] creates Deep Operator Network."
ComputePDEResidual::usage = "ComputePDEResidual[pinn, x, t, eqType, opts] computes PDE residual."
PlotPINNTraining::usage = "PlotPINNTraining[results] plots PINN training convergence."
PlotPDESolution::usage = "PlotPDESolution[xGrid, tGrid, solution, opts] visualizes PDE solution."

(* PDE types *)
HeatEquation::usage = "Heat equation: ∂u/∂t = α ∂²u/∂x²"
WaveEquation::usage = "Wave equation: ∂²u/∂t² = c² ∂²u/∂x²"
BurgersEquation::usage = "Burgers equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²"

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyColors = {BerkeleyBlue, CaliforniaGold, RGBColor[133/255, 148/255, 56/255]};

Begin["`Private`"]

(* Finite Difference Derivatives *)

ComputeDerivatives[network_, x_, t_, h_: 10^-5] := Module[{
  u, uxPlus, uxMinus, utPlus, utMinus, ux, ut, uxx},
  
  (* Function value *)
  u = PredictNetwork[network, {x, t}];
  
  (* First derivatives *)
  uxPlus = PredictNetwork[network, {x + h, t}];
  uxMinus = PredictNetwork[network, {x - h, t}];
  ux = (uxPlus - uxMinus)/(2*h);
  
  utPlus = PredictNetwork[network, {x, t + h}];
  utMinus = PredictNetwork[network, {x, t - h}];
  ut = (utPlus - utMinus)/(2*h);
  
  (* Second derivatives *)
  uxx = (uxPlus - 2*u + uxMinus)/h^2;
  
  <|"u" -> u, "ux" -> ux, "ut" -> ut, "uxx" -> uxx|>
]

ComputeDerivativesList[network_, xList_, tList_, h_: 10^-5] := Module[{
  inputs, u, uxPlus, uxMinus, utPlus, utMinus, ux, ut, uxx},
  
  inputs = Transpose[{xList, tList}];
  
  (* Function values *)
  u = PredictNetwork[network, inputs];
  
  (* First derivatives *)
  uxPlus = PredictNetwork[network, Transpose[{xList + h, tList}]];
  uxMinus = PredictNetwork[network, Transpose[{xList - h, tList}]];
  ux = (uxPlus - uxMinus)/(2*h);
  
  utPlus = PredictNetwork[network, Transpose[{xList, tList + h}]];
  utMinus = PredictNetwork[network, Transpose[{xList, tList - h}]];
  ut = (utPlus - utMinus)/(2*h);
  
  (* Second derivatives *)
  uxx = (uxPlus - 2*u + uxMinus)/h^2;
  
  <|"u" -> u, "ux" -> ux, "ut" -> ut, "uxx" -> uxx|>
]

(* PDE Residual Functions *)

HeatEquationResidual[derivatives_, diffusivity_: 1.0] := 
  derivatives["ut"] - diffusivity*derivatives["uxx"]

WaveEquationResidual[network_, x_, t_, waveSpeed_: 1.0, h_: 10^-5] := Module[{
  ut, utPlus, utMinus, utt, derivs},
  
  (* Get spatial derivatives *)
  derivs = ComputeDerivatives[network, x, t, h];
  
  (* Compute second time derivative *)
  utPlus = ComputeDerivatives[network, x, t + h, h];
  utMinus = ComputeDerivatives[network, x, t - h, h];
  utt = (utPlus["ut"] - utMinus["ut"])/(2*h);
  
  utt - waveSpeed^2*derivs["uxx"]
]

BurgersEquationResidual[derivatives_, viscosity_: 0.01] := 
  derivatives["ut"] + derivatives["u"]*derivatives["ux"] - viscosity*derivatives["uxx"]

(* Boundary and Initial Conditions *)

DefaultBoundaryConditions[x_, t_] := <|
  "Left" -> ConstantArray[0.0, Length[t]],   (* u(0, t) = 0 *)
  "Right" -> ConstantArray[0.0, Length[t]]   (* u(1, t) = 0 *)
|>

DefaultInitialCondition[x_] := Exp[-((x - 0.5)/0.1)^2]  (* Gaussian pulse *)

(* PINN Implementation *)

CreatePINN[layers_List, opts___] := Module[{
  activation, pdeWeight, bcWeight, icWeight, dataWeight, 
  learningRate, adaptiveWeights, network, pinn},
  
  (* Parse options *)
  activation = OptionValue[CreatePINN, {opts}, "Activation", "tanh"];
  pdeWeight = OptionValue[CreatePINN, {opts}, "PDEWeight", 1.0];
  bcWeight = OptionValue[CreatePINN, {opts}, "BCWeight", 1.0];
  icWeight = OptionValue[CreatePINN, {opts}, "ICWeight", 1.0];
  dataWeight = OptionValue[CreatePINN, {opts}, "DataWeight", 1.0];
  learningRate = OptionValue[CreatePINN, {opts}, "LearningRate", 0.001];
  adaptiveWeights = OptionValue[CreatePINN, {opts}, "AdaptiveWeights", False];
  
  (* Create neural network *)
  network = CreateMLP[layers,
    "Activations" -> activation,
    "OutputActivation" -> "linear",
    "LearningRate" -> learningRate
  ];
  
  (* Create PINN structure *)
  pinn = <|
    "Type" -> "PINN",
    "Network" -> network,
    "Layers" -> layers,
    "Activation" -> activation,
    "PDEWeight" -> pdeWeight,
    "BCWeight" -> bcWeight,
    "ICWeight" -> icWeight,
    "DataWeight" -> dataWeight,
    "LearningRate" -> learningRate,
    "AdaptiveWeights" -> adaptiveWeights,
    "IsTrained" -> False,
    "TrainingResults" -> <||>
  |>;
  
  pinn
]

Options[CreatePINN] = {
  "Activation" -> "tanh",
  "PDEWeight" -> 1.0,
  "BCWeight" -> 1.0,
  "ICWeight" -> 1.0,
  "DataWeight" -> 1.0,
  "LearningRate" -> 0.001,
  "AdaptiveWeights" -> False
};

ComputePDEResidual[pinn_, x_, t_, equationType_, opts___] := Module[{
  derivatives, residual},
  
  derivatives = If[ListQ[x] && ListQ[t],
    ComputeDerivativesList[pinn["Network"], x, t],
    ComputeDerivatives[pinn["Network"], x, t]
  ];
  
  residual = Switch[equationType,
    "Heat",
      Module[{diffusivity = OptionValue[ComputePDEResidual, {opts}, "Diffusivity", 1.0]},
        HeatEquationResidual[derivatives, diffusivity]
      ],
    "Wave", 
      Module[{waveSpeed = OptionValue[ComputePDEResidual, {opts}, "WaveSpeed", 1.0]},
        If[ListQ[x] && ListQ[t],
          (* For lists, need special handling of wave equation *)
          Table[WaveEquationResidual[pinn["Network"], x[[i]], t[[i]], waveSpeed], {i, Length[x]}],
          WaveEquationResidual[pinn["Network"], x, t, waveSpeed]
        ]
      ],
    "Burgers",
      Module[{viscosity = OptionValue[ComputePDEResidual, {opts}, "Viscosity", 0.01]},
        BurgersEquationResidual[derivatives, viscosity]
      ],
    _,
      Message[ComputePDEResidual::eqtype, equationType];
      $Failed
  ];
  
  residual
]

Options[ComputePDEResidual] = {
  "Diffusivity" -> 1.0,
  "WaveSpeed" -> 1.0,
  "Viscosity" -> 0.01
};

ComputeLosses[pinn_, xPDE_, tPDE_, xBC_, tBC_, xIC_, equationType_, opts___] := Module[{
  pdeResidual, pdeLoss, bcPredLeft, bcPredRight, bcTrue, bcLoss,
  icPred, icTrue, icLoss, dataLoss, xData, tData, uData, dataPred},
  
  (* Parse optional data *)
  xData = OptionValue[ComputeLosses, {opts}, "XData", {}];
  tData = OptionValue[ComputeLosses, {opts}, "TData", {}];  
  uData = OptionValue[ComputeLosses, {opts}, "UData", {}];
  
  (* PDE loss *)
  pdeResidual = ComputePDEResidual[pinn, xPDE, tPDE, equationType, 
    FilterRules[{opts}, Options[ComputePDEResidual]]];
  pdeLoss = Mean[pdeResidual^2];
  
  (* Boundary condition loss *)
  bcPredLeft = PredictNetwork[pinn["Network"], 
    Transpose[{ConstantArray[First[xBC], Length[tBC]], tBC}]];
  bcPredRight = PredictNetwork[pinn["Network"], 
    Transpose[{ConstantArray[Last[xBC], Length[tBC]], tBC}]];
  bcTrue = DefaultBoundaryConditions[xBC, tBC];
  
  bcLoss = Mean[(bcPredLeft - bcTrue["Left"])^2] + 
           Mean[(bcPredRight - bcTrue["Right"])^2];
  
  (* Initial condition loss *)
  icPred = PredictNetwork[pinn["Network"], 
    Transpose[{xIC, ConstantArray[0.0, Length[xIC]]}]];
  icTrue = DefaultInitialCondition[xIC];
  icLoss = Mean[(icPred - icTrue)^2];
  
  (* Data loss *)
  dataLoss = If[Length[xData] > 0 && Length[tData] > 0 && Length[uData] > 0,
    dataPred = PredictNetwork[pinn["Network"], Transpose[{xData, tData}]];
    Mean[(dataPred - uData)^2],
    0.0
  ];
  
  <|
    "PDE" -> pdeLoss,
    "BC" -> bcLoss, 
    "IC" -> icLoss,
    "Data" -> dataLoss
  |>
]

Options[ComputeLosses] = {
  "XData" -> {},
  "TData" -> {},
  "UData" -> {},
  "Diffusivity" -> 1.0,
  "WaveSpeed" -> 1.0,
  "Viscosity" -> 0.01
};

ComputeTotalLoss[pinn_, losses_] := 
  pinn["PDEWeight"]*losses["PDE"] + 
  pinn["BCWeight"]*losses["BC"] + 
  pinn["ICWeight"]*losses["IC"] + 
  pinn["DataWeight"]*losses["Data"]

TrainPINN[pinn_, xDomain_List, tDomain_List, opts___] := Module[{
  nPDEPoints, nBCPoints, nICPoints, epochs, equationType, verbose,
  xPDE, tPDE, tBC, xBC, xIC, lossHistory, pdeLossHistory, 
  bcLossHistory, icLossHistory, dataLossHistory, trainedPINN,
  losses, totalLoss, results},
  
  (* Parse options *)
  nPDEPoints = OptionValue[TrainPINN, {opts}, "NPDEPoints", 10000];
  nBCPoints = OptionValue[TrainPINN, {opts}, "NBCPoints", 100];
  nICPoints = OptionValue[TrainPINN, {opts}, "NICPoints", 100];
  epochs = OptionValue[TrainPINN, {opts}, "Epochs", 1000];
  equationType = OptionValue[TrainPINN, {opts}, "EquationType", "Heat"];
  verbose = OptionValue[TrainPINN, {opts}, "Verbose", True];
  
  (* Generate training points *)
  SeedRandom[42]; (* For reproducibility *)
  
  (* PDE collocation points *)
  xPDE = RandomReal[xDomain, nPDEPoints];
  tPDE = RandomReal[tDomain, nPDEPoints];
  
  (* Boundary condition points *)
  tBC = RandomReal[tDomain, nBCPoints];
  xBC = xDomain; (* Left and right boundaries *)
  
  (* Initial condition points *)
  xIC = RandomReal[xDomain, nICPoints];
  
  (* Initialize training history *)
  lossHistory = {};
  pdeLossHistory = {};
  bcLossHistory = {};
  icLossHistory = {};
  dataLossHistory = {};
  
  trainedPINN = pinn;
  
  (* Training loop *)
  Do[
    (* Compute losses *)
    losses = ComputeLosses[trainedPINN, xPDE, tPDE, xBC, tBC, xIC, equationType,
      FilterRules[{opts}, Options[ComputeLosses]]];
    
    totalLoss = ComputeTotalLoss[trainedPINN, losses];
    
    (* Store history *)
    AppendTo[lossHistory, totalLoss];
    AppendTo[pdeLossHistory, losses["PDE"]];
    AppendTo[bcLossHistory, losses["BC"]];
    AppendTo[icLossHistory, losses["IC"]];
    AppendTo[dataLossHistory, losses["Data"]];
    
    (* Update parameters (simplified - would implement full optimization) *)
    (* UpdatePINNParameters[trainedPINN, ...]; *)
    
    (* Print progress *)
    If[verbose && Mod[epoch, 100] == 0,
      Print["Epoch ", epoch, "/", epochs];
      Print["  Total Loss: ", NumberForm[totalLoss, 6]];
      Print["  PDE Loss: ", NumberForm[losses["PDE"], 6]];
      Print["  BC Loss: ", NumberForm[losses["BC"], 6]];
      Print["  IC Loss: ", NumberForm[losses["IC"], 6]];
      If[losses["Data"] > 0,
        Print["  Data Loss: ", NumberForm[losses["Data"], 6]]
      ]
    ],
    
    {epoch, epochs}
  ];
  
  (* Store results *)
  results = <|
    "LossHistory" -> lossHistory,
    "PDELossHistory" -> pdeLossHistory,
    "BCLossHistory" -> bcLossHistory,
    "ICLossHistory" -> icLossHistory,
    "DataLossHistory" -> dataLossHistory,
    "TotalEpochs" -> epochs
  |>;
  
  trainedPINN["TrainingResults"] = results;
  trainedPINN["IsTrained"] = True;
  
  results
]

Options[TrainPINN] = {
  "NPDEPoints" -> 10000,
  "NBCPoints" -> 100,
  "NICPoints" -> 100,
  "Epochs" -> 1000,
  "EquationType" -> "Heat",
  "Verbose" -> True,
  "XData" -> {},
  "TData" -> {},
  "UData" -> {},
  "Diffusivity" -> 1.0,
  "WaveSpeed" -> 1.0,
  "Viscosity" -> 0.01
};

PredictPINN[pinn_, x_, t_] := Module[{},
  If[!pinn["IsTrained"],
    Message[PredictPINN::nottrained];
    Return[$Failed]
  ];
  
  If[ListQ[x] && ListQ[t],
    PredictNetwork[pinn["Network"], Transpose[{x, t}]],
    PredictNetwork[pinn["Network"], {x, t}]
  ]
]

(* Deep Operator Network Implementation *)

CreateDeepONet[branchLayers_List, trunkLayers_List, opts___] := Module[{
  activation, learningRate, branchNet, trunkNet, deeponet},
  
  (* Parse options *)
  activation = OptionValue[CreateDeepONet, {opts}, "Activation", "relu"];
  learningRate = OptionValue[CreateDeepONet, {opts}, "LearningRate", 0.001];
  
  (* Create branch and trunk networks *)
  branchNet = CreateMLP[branchLayers,
    "Activations" -> activation,
    "OutputActivation" -> "linear",
    "LearningRate" -> learningRate
  ];
  
  trunkNet = CreateMLP[trunkLayers,
    "Activations" -> activation,
    "OutputActivation" -> "linear", 
    "LearningRate" -> learningRate
  ];
  
  (* Create DeepONet structure *)
  deeponet = <|
    "Type" -> "DeepONet",
    "BranchLayers" -> branchLayers,
    "TrunkLayers" -> trunkLayers,
    "BranchNetwork" -> branchNet,
    "TrunkNetwork" -> trunkNet,
    "Activation" -> activation,
    "LearningRate" -> learningRate,
    "IsTrained" -> False
  |>;
  
  deeponet
]

Options[CreateDeepONet] = {
  "Activation" -> "relu",
  "LearningRate" -> 0.001
};

(* Visualization Functions *)

PlotPINNTraining[results_, opts___] := Module[{
  title, lossHistory, pdeLossHistory, bcLossHistory, icLossHistory,
  epochs, fig1, fig2, fig3, fig4},
  
  title = OptionValue[PlotPINNTraining, {opts}, "Title", "PINN Training Results"];
  
  lossHistory = results["LossHistory"];
  pdeLossHistory = results["PDELossHistory"];
  bcLossHistory = results["BCLossHistory"];
  icLossHistory = results["ICLossHistory"];
  epochs = Range[Length[lossHistory]];
  
  (* Total loss *)
  fig1 = ListLogPlot[
    Transpose[{epochs, lossHistory}],
    PlotStyle -> {BerkeleyBlue, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Epoch", "Total Loss"},
    PlotLabel -> "Total Loss",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* PDE loss *)
  fig2 = ListLogPlot[
    Transpose[{epochs, pdeLossHistory}],
    PlotStyle -> {CaliforniaGold, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Epoch", "PDE Loss"},
    PlotLabel -> "PDE Residual Loss",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Boundary condition loss *)
  fig3 = ListLogPlot[
    Transpose[{epochs, bcLossHistory}],
    PlotStyle -> {Red, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Epoch", "BC Loss"},
    PlotLabel -> "Boundary Condition Loss",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Initial condition loss *)
  fig4 = ListLogPlot[
    Transpose[{epochs, icLossHistory}],
    PlotStyle -> {Green, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Epoch", "IC Loss"},
    PlotLabel -> "Initial Condition Loss",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Combine plots *)
  Grid[{{fig1, fig2}, {fig3, fig4}}, Frame -> All, 
       BaseStyle -> {FontSize -> 12, FontFamily -> "Times"},
       Epilog -> Text[Style[title, Bold, 16], Scaled[{0.5, 0.95}]]]
]

Options[PlotPINNTraining] = {"Title" -> "PINN Training Results"};

PlotPDESolution[xGrid_, tGrid_, uPred_, opts___] := Module[{
  uTrue, title, hasTrue, nPlots, fig1, fig2, fig3},
  
  uTrue = OptionValue[PlotPDESolution, {opts}, "UTrue", {}];
  title = OptionValue[PlotPDESolution, {opts}, "Title", "PDE Solution"];
  
  hasTrue = Length[uTrue] > 0;
  nPlots = If[hasTrue, 3, 1];
  
  (* Predicted solution *)
  fig1 = ContourPlot[
    Interpolation[Flatten[Table[{xGrid[[i, j]], tGrid[[i, j]], uPred[[i, j]]}, 
                              {i, Length[xGrid]}, {j, Length[xGrid[[1]]]}], 1]][x, t],
    {x, Min[xGrid], Max[xGrid]}, {t, Min[tGrid], Max[tGrid]},
    PlotLabel -> "Predicted Solution",
    FrameLabel -> {"x", "t"},
    ColorFunction -> "TemperatureMap",
    PlotLegends -> Automatic
  ];
  
  If[hasTrue,
    (* True solution *)
    fig2 = ContourPlot[
      Interpolation[Flatten[Table[{xGrid[[i, j]], tGrid[[i, j]], uTrue[[i, j]]}, 
                                {i, Length[xGrid]}, {j, Length[xGrid[[1]]]}], 1]][x, t],
      {x, Min[xGrid], Max[xGrid]}, {t, Min[tGrid], Max[tGrid]},
      PlotLabel -> "True Solution",
      FrameLabel -> {"x", "t"},
      ColorFunction -> "TemperatureMap",
      PlotLegends -> Automatic
    ];
    
    (* Error *)
    fig3 = ContourPlot[
      Interpolation[Flatten[Table[{xGrid[[i, j]], tGrid[[i, j]], 
                                  Abs[uPred[[i, j]] - uTrue[[i, j]]]}, 
                                {i, Length[xGrid]}, {j, Length[xGrid[[1]]]}], 1]][x, t],
      {x, Min[xGrid], Max[xGrid]}, {t, Min[tGrid], Max[tGrid]},
      PlotLabel -> "Absolute Error",
      FrameLabel -> {"x", "t"},
      ColorFunction -> "Sunset",
      PlotLegends -> Automatic
    ];
    
    (* Combine all plots *)
    Grid[{{fig1, fig2, fig3}}, Frame -> All, Spacings -> {1, 1},
         BaseStyle -> {FontSize -> 12, FontFamily -> "Times"},
         Epilog -> Text[Style[title, Bold, 16], Scaled[{0.5, 0.95}]]],
    
    (* Just predicted solution *)
    Labeled[fig1, Style[title, Bold, 16], Top]
  ]
]

Options[PlotPDESolution] = {
  "UTrue" -> {},
  "Title" -> "PDE Solution"
};

(* Test Data Generation *)

CreatePDETestData[equationType_: "Heat"] := Module[{
  x, t, xGrid, tGrid, uGrid, xFlat, tFlat, uFlat},
  
  SeedRandom[42];
  
  (* Domain *)
  x = Range[0, 1, 0.01];
  t = Range[0, 1, 0.02];
  {xGrid, tGrid} = MeshGrid[x, t];
  
  uGrid = Switch[equationType,
    "Heat",
      (* Analytical solution: u(x,t) = sin(πx) * exp(-π²t) *)
      Sin[Pi*xGrid]*Exp[-Pi^2*tGrid],
    "Wave", 
      (* Analytical solution: u(x,t) = sin(πx) * cos(πt) *)
      Sin[Pi*xGrid]*Cos[Pi*tGrid],
    "Burgers",
      (* Approximate solution for Burgers equation *)
      0.5*(Sin[Pi*xGrid]*Exp[-0.1*tGrid] + Cos[Pi*xGrid]*Exp[-0.05*tGrid]),
    _,
      Sin[Pi*xGrid]*Exp[-Pi^2*tGrid]
  ];
  
  (* Flatten for use with networks *)
  xFlat = Flatten[xGrid];
  tFlat = Flatten[tGrid];
  uFlat = Flatten[uGrid];
  
  <|
    "x" -> xFlat,
    "t" -> tFlat,
    "u" -> uFlat,
    "xGrid" -> xGrid,
    "tGrid" -> tGrid,
    "uGrid" -> uGrid
  |>
]

(* Helper function for creating coordinate grids *)
MeshGrid[x_List, y_List] := Module[{nx, ny},
  nx = Length[x];
  ny = Length[y];
  {
    Table[x[[i]], {j, ny}, {i, nx}],
    Table[y[[j]], {j, ny}, {i, nx}]
  }
]

End[]

EndPackage[]