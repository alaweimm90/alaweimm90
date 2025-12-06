(* ::Package:: *)

(* 
SupervisedLearning.wl - Advanced Supervised Learning for Scientific Computing

This package implements supervised learning algorithms specifically designed for
scientific computing applications, with emphasis on interpretability and
physics-aware modeling in the Berkeley SciComp framework.

Features:
- LinearRegression with multiple solvers and uncertainty quantification
- PolynomialRegression for nonlinear relationships  
- RidgeRegression with cross-validation
- LogisticRegression for classification
- Berkeley-themed visualizations
- Scientific computing integration

Author: Berkeley SciComp Team
Date: 2024
*)

BeginPackage["SupervisedLearning`"]

(* Public function declarations *)
LinearRegressionFit::usage = "LinearRegressionFit[X, y, opts] fits linear regression model with advanced options."
LinearRegressionPredict::usage = "LinearRegressionPredict[model, X] makes predictions using fitted linear regression."
PolynomialRegressionFit::usage = "PolynomialRegressionFit[X, y, degree, opts] fits polynomial regression model."
LogisticRegressionFit::usage = "LogisticRegressionFit[X, y, opts] fits logistic regression for classification."
ModelScore::usage = "ModelScore[model, X, y] computes R² score for regression models."
PlotRegressionResults::usage = "PlotRegressionResults[model, X, y] creates Berkeley-styled regression plots."

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyColors = {BerkeleyBlue, CaliforniaGold, RGBColor[133/255, 148/255, 56/255]};

Begin["`Private`"]

(* Helper Functions *)

ValidateInputData[X_, y_] := Module[{validX, validY},
  validX = If[MatrixQ[X], X, {X}\[Transpose]];
  validY = If[VectorQ[y], y, Flatten[y]];
  
  If[Length[validX] != Length[validY],
    Message[LinearRegressionFit::invdim, "X and y must have same number of samples"];
    $Failed,
    {validX, validY}
  ]
]

SVDSolver[X_, y_, regularization_: 0] := Module[{U, w, Vt, d, coeffs},
  If[regularization > 0,
    (* Ridge regression via SVD *)
    {U, w, Vt} = SingularValueDecomposition[X];
    d = w/(w^2 + regularization);
    coeffs = Vt\[Transpose].DiagonalMatrix[d].(U\[Transpose].y),
    (* Standard pseudoinverse *)
    coeffs = PseudoInverse[X].y
  ];
  coeffs
]

NormalEquationSolver[X_, y_, regularization_: 0] := Module[{XTX, XTy},
  XTX = X\[Transpose].X;
  XTy = X\[Transpose].y;
  
  If[regularization > 0,
    XTX = XTX + regularization*IdentityMatrix[Length[XTX]]
  ];
  
  LinearSolve[XTX, XTy]
]

ComputeUncertainty[X_, y_, coeffs_, regularization_: 0] := Module[{residuals, sigmaSq, XTXInv, covMatrix},
  residuals = y - X.coeffs;
  sigmaSq = Total[residuals^2]/(Length[y] - Length[coeffs]);
  
  Try[
    XTXInv = Inverse[X\[Transpose].X + regularization*IdentityMatrix[Dimensions[X][[2]]]];
    covMatrix = sigmaSq*XTXInv;
    covMatrix,
    $Failed
  ]
]

(* Linear Regression Implementation *)

LinearRegressionFit[X_, y_, opts___] := Module[{
  validData, validX, validY, fitIntercept, solver, regularization, 
  uncertaintyEstimation, XAug, coeffs, intercept, coefficients, 
  covMatrix, model},
  
  (* Parse options *)
  fitIntercept = OptionValue[LinearRegressionFit, {opts}, "FitIntercept", True];
  solver = OptionValue[LinearRegressionFit, {opts}, "Solver", "SVD"];
  regularization = OptionValue[LinearRegressionFit, {opts}, "Regularization", 0.0];
  uncertaintyEstimation = OptionValue[LinearRegressionFit, {opts}, "UncertaintyEstimation", True];
  
  (* Validate input *)
  validData = ValidateInputData[X, y];
  If[validData === $Failed, Return[$Failed]];
  {validX, validY} = validData;
  
  (* Add intercept column if needed *)
  XAug = If[fitIntercept, 
    Join[ConstantArray[1, {Length[validX], 1}], validX, 2],
    validX
  ];
  
  (* Solve using specified method *)
  coeffs = Switch[solver,
    "SVD", SVDSolver[XAug, validY, regularization],
    "Normal", NormalEquationSolver[XAug, validY, regularization],
    _, Message[LinearRegressionFit::solver, solver]; $Failed
  ];
  
  If[coeffs === $Failed, Return[$Failed]];
  
  (* Extract intercept and coefficients *)
  If[fitIntercept,
    intercept = First[coeffs];
    coefficients = Rest[coeffs],
    intercept = 0.0;
    coefficients = coeffs
  ];
  
  (* Compute uncertainty estimates *)
  covMatrix = If[uncertaintyEstimation,
    ComputeUncertainty[XAug, validY, coeffs, regularization],
    $Failed
  ];
  
  (* Return model association *)
  model = <|
    "Type" -> "LinearRegression",
    "Intercept" -> intercept,
    "Coefficients" -> coefficients,
    "CovarianceMatrix" -> covMatrix,
    "FitIntercept" -> fitIntercept,
    "Solver" -> solver,
    "Regularization" -> regularization,
    "NFeatures" -> Length[coefficients],
    "IsFitted" -> True
  |>;
  
  model
]

(* Set options for LinearRegressionFit *)
Options[LinearRegressionFit] = {
  "FitIntercept" -> True,
  "Solver" -> "SVD", 
  "Regularization" -> 0.0,
  "UncertaintyEstimation" -> True
};

LinearRegressionPredict[model_, X_, opts___] := Module[{
  validX, predictions, returnUncertainty, XAug, predVar, uncertainties},
  
  (* Validate model *)
  If[!KeyExistsQ[model, "IsFitted"] || !model["IsFitted"],
    Message[LinearRegressionPredict::notfitted];
    Return[$Failed]
  ];
  
  returnUncertainty = OptionValue[LinearRegressionPredict, {opts}, "ReturnUncertainty", False];
  
  (* Validate input *)
  validX = If[MatrixQ[X], X, {X}\[Transpose]];
  
  (* Make predictions *)
  predictions = validX.model["Coefficients"] + model["Intercept"];
  
  (* Compute uncertainties if requested *)
  If[returnUncertainty && model["CovarianceMatrix"] =!= $Failed,
    (* Add intercept column for uncertainty calculation *)
    XAug = If[model["FitIntercept"],
      Join[ConstantArray[1, {Length[validX], 1}], validX, 2],
      validX
    ];
    
    (* Prediction variance *)
    predVar = Table[
      XAug[[i]].model["CovarianceMatrix"].XAug[[i]],
      {i, Length[XAug]}
    ];
    uncertainties = Sqrt[predVar];
    
    {predictions, uncertainties},
    predictions
  ]
]

Options[LinearRegressionPredict] = {"ReturnUncertainty" -> False};

(* Polynomial Regression Implementation *)

GeneratePolynomialFeatures[X_, degree_, opts___] := Module[{
  interactionOnly, features, n, d, powers, featureMatrix},
  
  interactionOnly = OptionValue[GeneratePolynomialFeatures, {opts}, "InteractionOnly", False];
  
  If[VectorQ[X], X = {X}\[Transpose]];
  
  {n, d} = Dimensions[X];
  
  (* Generate all polynomial combinations up to degree *)
  powers = Flatten[
    Table[
      IntegerPartitions[deg, {d}, Range[0, deg]],
      {deg, 0, degree}
    ], 1
  ];
  
  (* Filter for interaction-only if specified *)
  If[interactionOnly,
    powers = Select[powers, Max[#] <= 1 &]
  ];
  
  (* Create feature matrix *)
  featureMatrix = Table[
    Product[X[[i, j]]^powers[[k, j]], {j, d}],
    {i, n}, {k, Length[powers]}
  ];
  
  featureMatrix
]

Options[GeneratePolynomialFeatures] = {"InteractionOnly" -> False};

PolynomialRegressionFit[X_, y_, degree_, opts___] := Module[{
  polyFeatures, model},
  
  (* Generate polynomial features *)
  polyFeatures = GeneratePolynomialFeatures[X, degree, 
    FilterRules[{opts}, Options[GeneratePolynomialFeatures]]];
  
  (* Fit linear model on polynomial features *)
  model = LinearRegressionFit[polyFeatures, y, 
    FilterRules[{opts}, Options[LinearRegressionFit]]];
  
  (* Add polynomial information to model *)
  If[model =!= $Failed,
    model["Type"] = "PolynomialRegression";
    model["Degree"] = degree;
    model["OriginalFeatures"] = X;
  ];
  
  model
]

(* Logistic Regression Implementation *)

SigmoidFunction[z_] := 1/(1 + Exp[-Clip[z, {-250, 250}]])

LogisticCostFunction[params_, X_, y_, C_: 1.0] := Module[{
  intercept, coeffs, z, h, cost, regTerm},
  
  If[Length[params] > Length[X[[1]]],
    intercept = First[params];
    coeffs = Rest[params],
    intercept = 0;
    coeffs = params
  ];
  
  z = X.coeffs + intercept;
  h = SigmoidFunction[z];
  
  (* Clip to prevent log(0) *)
  h = Clip[h, {10^-15, 1 - 10^-15}];
  
  (* Cost function *)
  cost = -Mean[y*Log[h] + (1 - y)*Log[1 - h]];
  
  (* Add regularization *)
  regTerm = Total[coeffs^2]/(2*C);
  cost + regTerm
]

LogisticRegressionFit[X_, y_, opts___] := Module[{
  validData, validX, validY, fitIntercept, C, maxIter, 
  nParams, initialParams, result, params, intercept, coefficients, model},
  
  (* Parse options *)
  fitIntercept = OptionValue[LogisticRegressionFit, {opts}, "FitIntercept", True];
  C = OptionValue[LogisticRegressionFit, {opts}, "C", 1.0];
  maxIter = OptionValue[LogisticRegressionFit, {opts}, "MaxIterations", 1000];
  
  (* Validate input *)
  validData = ValidateInputData[X, y];
  If[validData === $Failed, Return[$Failed]];
  {validX, validY} = validData;
  
  (* Initialize parameters *)
  nParams = If[fitIntercept, Length[validX[[1]]] + 1, Length[validX[[1]]]];
  initialParams = ConstantArray[0.0, nParams];
  
  (* Optimize using FindMinimum *)
  result = FindMinimum[
    LogisticCostFunction[vars, validX, validY, C],
    Table[{vars[[i]], initialParams[[i]]}, {i, nParams}],
    MaxIterations -> maxIter
  ];
  
  If[result === $Failed, Return[$Failed]];
  
  params = vars /. result[[2]];
  
  (* Extract intercept and coefficients *)
  If[fitIntercept,
    intercept = First[params];
    coefficients = Rest[params],
    intercept = 0.0;
    coefficients = params
  ];
  
  (* Return model *)
  model = <|
    "Type" -> "LogisticRegression",
    "Intercept" -> intercept,
    "Coefficients" -> coefficients,
    "FitIntercept" -> fitIntercept,
    "C" -> C,
    "Classes" -> Union[validY],
    "IsFitted" -> True
  |>;
  
  model
]

Options[LogisticRegressionFit] = {
  "FitIntercept" -> True,
  "C" -> 1.0,
  "MaxIterations" -> 1000
};

(* Model Evaluation *)

ModelScore[model_, X_, y_, opts___] := Module[{
  predictions, metric, ssRes, ssTot, r2},
  
  metric = OptionValue[ModelScore, {opts}, "Metric", "R2"];
  
  predictions = Switch[model["Type"],
    "LinearRegression" | "PolynomialRegression", 
      LinearRegressionPredict[model, X],
    "LogisticRegression",
      LogisticRegressionPredict[model, X],
    _, 
      Message[ModelScore::unsupported, model["Type"]]; 
      Return[$Failed]
  ];
  
  If[predictions === $Failed, Return[$Failed]];
  
  Switch[metric,
    "R2",
      ssRes = Total[(y - predictions)^2];
      ssTot = Total[(y - Mean[y])^2];
      r2 = 1 - (ssRes/ssTot),
    "MSE",
      Mean[(y - predictions)^2],
    "MAE", 
      Mean[Abs[y - predictions]],
    _,
      Message[ModelScore::metric, metric];
      $Failed
  ]
]

Options[ModelScore] = {"Metric" -> "R2"};

LogisticRegressionPredict[model_, X_, opts___] := Module[{
  validX, z, probabilities, predictions, returnProba},
  
  returnProba = OptionValue[LogisticRegressionPredict, {opts}, "ReturnProbabilities", False];
  
  validX = If[MatrixQ[X], X, {X}\[Transpose]];
  
  z = validX.model["Coefficients"] + model["Intercept"];
  probabilities = SigmoidFunction[z];
  
  If[returnProba,
    Transpose[{1 - probabilities, probabilities}],
    Round[probabilities]
  ]
]

Options[LogisticRegressionPredict] = {"ReturnProbabilities" -> False};

(* Visualization Functions *)

PlotRegressionResults[model_, X_, y_, opts___] := Module[{
  predictions, residuals, title, plotTitle, fig1, fig2},
  
  title = OptionValue[PlotRegressionResults, {opts}, "Title", "Regression Results"];
  
  predictions = Switch[model["Type"],
    "LinearRegression" | "PolynomialRegression",
      LinearRegressionPredict[model, X],
    _,
      Message[PlotRegressionResults::unsupported, model["Type"]];
      Return[$Failed]
  ];
  
  If[predictions === $Failed, Return[$Failed]];
  
  residuals = y - predictions;
  
  (* Plot 1: Predictions vs Actual *)
  fig1 = ListPlot[
    Transpose[{y, predictions}],
    PlotStyle -> {PointSize[0.01], BerkeleyBlue},
    Epilog -> {
      CaliforniaGold, Dashed, Thickness[0.005],
      Line[{{Min[Join[y, predictions]], Min[Join[y, predictions]]}, 
            {Max[Join[y, predictions]], Max[Join[y, predictions]]}}]
    },
    Frame -> True,
    FrameLabel -> {"Actual Values", "Predicted Values"},
    PlotLabel -> "Predictions vs Actual",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Plot 2: Residuals *)
  fig2 = ListPlot[
    Transpose[{predictions, residuals}],
    PlotStyle -> {PointSize[0.01], BerkeleyBlue},
    Epilog -> {
      CaliforniaGold, Dashed, Thickness[0.005],
      Line[{{Min[predictions], 0}, {Max[predictions], 0}}]
    },
    Frame -> True,
    FrameLabel -> {"Predicted Values", "Residuals"},
    PlotLabel -> "Residual Plot",
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]]
  ];
  
  (* Combine plots *)
  Grid[{{fig1, fig2}}, Frame -> All, FrameStyle -> Directive[Thick, Black],
       BaseStyle -> {FontSize -> 12, FontFamily -> "Times"},
       Epilog -> Text[Style[title, Bold, 16], Scaled[{0.5, 0.95}]]]
]

Options[PlotRegressionResults] = {"Title" -> "Regression Results"};

(* Create test datasets *)

CreateTestDatasets[] := Module[{nSamples, XLinear, trueCoeffs, yLinear,
  XPoly, yPoly, XClass, yClass},
  
  nSamples = 100;
  SeedRandom[42];
  
  (* Linear regression dataset *)
  XLinear = RandomReal[NormalDistribution[], {nSamples, 3}];
  trueCoeffs = {2.5, -1.3, 0.8};
  yLinear = XLinear.trueCoeffs + 0.1*RandomReal[NormalDistribution[], nSamples];
  
  (* Polynomial regression dataset *)
  XPoly = RandomReal[{-2, 2}, {nSamples, 1}];
  yPoly = 2*Flatten[XPoly]^2 - 3*Flatten[XPoly] + 1 + 
          0.2*RandomReal[NormalDistribution[], nSamples];
  
  (* Classification dataset *)
  XClass = RandomReal[NormalDistribution[], {nSamples, 2}];
  yClass = UnitStep[Total[XClass, {2}]];
  
  <|
    "Linear" -> {XLinear, yLinear},
    "Polynomial" -> {XPoly, yPoly},
    "Classification" -> {XClass, yClass}
  |>
]

End[]

EndPackage[]