(* ::Package:: *)

(* 
Utilities.wl - Machine Learning Utilities for Scientific Computing

This package provides comprehensive data processing and utility functions
for machine learning workflows in scientific computing applications
in the Berkeley SciComp framework.

Features:
- Data preprocessing and cleaning
- Feature scaling and normalization
- Train/validation/test splitting
- Cross-validation utilities
- Model evaluation metrics
- Berkeley-themed visualizations
- Scientific computing integration

Author: Berkeley SciComp Team
Date: 2024
*)

BeginPackage["Utilities`"]

(* Public function declarations *)
CleanData::usage = "CleanData[data, opts] cleans data by handling missing values and outliers."
ScaleFeatures::usage = "ScaleFeatures[data, method, opts] scales features using specified method."
TrainTestSplit::usage = "TrainTestSplit[X, y, trainRatio, opts] splits data into training and testing sets."
CrossValidationSplit::usage = "CrossValidationSplit[X, y, nFolds, opts] creates cross-validation folds."
EvaluateModel::usage = "EvaluateModel[yTrue, yPred, task, opts] computes evaluation metrics."
PlotDataQuality::usage = "PlotDataQuality[data, opts] visualizes data quality and distributions."
PlotModelEvaluation::usage = "PlotModelEvaluation[yTrue, yPred, task, opts] plots model evaluation results."

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyColors = {BerkeleyBlue, CaliforniaGold, RGBColor[133/255, 148/255, 56/255]};

Begin["`Private`"]

(* Helper Functions *)

ValidateData[data_] := Module[{},
  If[!MatrixQ[data] && !VectorQ[data],
    Message[CleanData::invdata, "Data must be a matrix or vector"];
    $Failed,
    data
  ]
]

DetectMissingValues[data_] := Module[{missingPattern},
  missingPattern = Map[MissingQ[#] || !NumericQ[#] &, data, {-1}];
  missingPattern
]

DetectOutliers[data_, method_: "iqr"] := Module[{outliers, q1, q3, iqr, lowerBound, upperBound, mean, std},
  Switch[method,
    "iqr",
      q1 = Quantile[data, 0.25];
      q3 = Quantile[data, 0.75];
      iqr = q3 - q1;
      lowerBound = q1 - 1.5*iqr;
      upperBound = q3 + 1.5*iqr;
      outliers = Map[# < lowerBound || # > upperBound &, data],
    
    "zscore",
      mean = Mean[data];
      std = StandardDeviation[data];
      outliers = Map[Abs[(# - mean)/std] > 3 &, data],
    
    _,
      outliers = ConstantArray[False, Length[data]]
  ];
  
  outliers
]

(* Data Cleaning Functions *)

CleanData[data_, opts___] := Module[{
  validData, handleMissing, removeOutliers, outlierMethod, 
  missingPattern, cleanedData, outliers, finalData, statistics},
  
  (* Parse options *)
  handleMissing = OptionValue[CleanData, {opts}, "HandleMissing", "mean"];
  removeOutliers = OptionValue[CleanData, {opts}, "RemoveOutliers", True];
  outlierMethod = OptionValue[CleanData, {opts}, "OutlierMethod", "iqr"];
  
  (* Validate input *)
  validData = ValidateData[data];
  If[validData === $Failed, Return[$Failed]];
  
  (* Handle missing values *)
  missingPattern = DetectMissingValues[validData];
  
  cleanedData = Switch[handleMissing,
    "mean",
      If[MatrixQ[validData],
        Table[
          Table[
            If[missingPattern[[i, j]], Mean[DeleteMissing[validData[[All, j]]]], validData[[i, j]]],
            {j, Dimensions[validData][[2]]}
          ],
          {i, Dimensions[validData][[1]]}
        ],
        Table[
          If[missingPattern[[i]], Mean[DeleteMissing[validData]], validData[[i]]],
          {i, Length[validData]}
        ]
      ],
    
    "median",
      If[MatrixQ[validData],
        Table[
          Table[
            If[missingPattern[[i, j]], Median[DeleteMissing[validData[[All, j]]]], validData[[i, j]]],
            {j, Dimensions[validData][[2]]}
          ],
          {i, Dimensions[validData][[1]]}
        ],
        Table[
          If[missingPattern[[i]], Median[DeleteMissing[validData]], validData[[i]]],
          {i, Length[validData]}
        ]
      ],
    
    "drop",
      If[MatrixQ[validData],
        Select[validData, !AnyTrue[DetectMissingValues[{#}][[1]], TrueQ] &],
        Select[validData, NumericQ]
      ],
    
    _,
      validData
  ];
  
  (* Remove outliers *)
  If[removeOutliers && VectorQ[cleanedData],
    outliers = DetectOutliers[cleanedData, outlierMethod];
    finalData = Pick[cleanedData, outliers, False],
    finalData = cleanedData
  ];
  
  (* Compute statistics *)
  statistics = <|
    "OriginalSamples" -> If[MatrixQ[validData], Dimensions[validData][[1]], Length[validData]],
    "CleanedSamples" -> If[MatrixQ[finalData], Dimensions[finalData][[1]], Length[finalData]],
    "MissingValues" -> Count[Flatten[missingPattern], True],
    "OutliersRemoved" -> If[removeOutliers && VectorQ[cleanedData], Count[outliers, True], 0]
  |>;
  
  <|"Data" -> finalData, "Statistics" -> statistics|>
]

(* Set options for CleanData *)
Options[CleanData] = {
  "HandleMissing" -> "mean",
  "RemoveOutliers" -> True,
  "OutlierMethod" -> "iqr"
};

(* Feature Scaling Functions *)

ScaleFeatures[data_, method_: "standard", opts___] := Module[{
  validData, center, scale, scaledData, mean, std, min, max, median, iqr, result},
  
  (* Parse options *)
  center = OptionValue[ScaleFeatures, {opts}, "Center", True];
  scale = OptionValue[ScaleFeatures, {opts}, "Scale", True];
  
  (* Validate input *)
  validData = ValidateData[data];
  If[validData === $Failed, Return[$Failed]];
  
  (* Ensure matrix format *)
  If[VectorQ[validData], validData = {validData}\[Transpose]];
  
  scaledData = Switch[method,
    "standard",
      mean = Mean[validData];
      std = StandardDeviation[validData];
      std = Map[If[# == 0, 1, #] &, std]; (* Avoid division by zero *)
      
      If[center && scale,
        Map[(# - mean)/std &, validData],
        If[center,
          Map[# - mean &, validData],
          If[scale,
            Map[#/std &, validData],
            validData
          ]
        ]
      ],
    
    "minmax",
      min = Min[validData];
      max = Max[validData];
      range = max - min;
      range = Map[If[# == 0, 1, #] &, range]; (* Avoid division by zero *)
      
      Map[(# - min)/range &, validData],
    
    "robust",
      median = Median[validData];
      q1 = Quantile[validData, 0.25];
      q3 = Quantile[validData, 0.75];
      iqr = q3 - q1;
      iqr = Map[If[# == 0, 1, #] &, iqr]; (* Avoid division by zero *)
      
      Map[(# - median)/iqr &, validData],
    
    _,
      validData
  ];
  
  (* Store scaling parameters *)
  result = <|
    "Data" -> scaledData,
    "Method" -> method,
    "Parameters" -> Switch[method,
      "standard", <|"Mean" -> mean, "Std" -> std|>,
      "minmax", <|"Min" -> min, "Max" -> max|>,
      "robust", <|"Median" -> median, "IQR" -> iqr|>,
      _, <||>
    ]
  |>;
  
  result
]

(* Set options for ScaleFeatures *)
Options[ScaleFeatures] = {
  "Center" -> True,
  "Scale" -> True
};

(* Data Splitting Functions *)

TrainTestSplit[X_, y_, trainRatio_: 0.8, opts___] := Module[{
  validX, validY, stratify, shuffle, randomSeed, nSamples, 
  nTrain, indices, trainIndices, testIndices, result},
  
  (* Parse options *)
  stratify = OptionValue[TrainTestSplit, {opts}, "Stratify", False];
  shuffle = OptionValue[TrainTestSplit, {opts}, "Shuffle", True];
  randomSeed = OptionValue[TrainTestSplit, {opts}, "RandomSeed", 42];
  
  (* Validate input *)
  validX = ValidateData[X];
  validY = ValidateData[y];
  If[validX === $Failed || validY === $Failed, Return[$Failed]];
  
  (* Ensure matrix format for X *)
  If[VectorQ[validX], validX = {validX}\[Transpose]];
  
  nSamples = Dimensions[validX][[1]];
  If[Length[validY] != nSamples,
    Message[TrainTestSplit::invdim, "X and y must have same number of samples"];
    Return[$Failed]
  ];
  
  SeedRandom[randomSeed];
  
  If[stratify,
    (* Stratified sampling *)
    result = StratifiedSplit[validX, validY, trainRatio, shuffle],
    
    (* Random sampling *)
    nTrain = Floor[trainRatio * nSamples];
    
    indices = If[shuffle, RandomSample[Range[nSamples]], Range[nSamples]];
    trainIndices = Take[indices, nTrain];
    testIndices = Drop[indices, nTrain];
    
    result = <|
      "XTrain" -> validX[[trainIndices]],
      "XTest" -> validX[[testIndices]],
      "YTrain" -> validY[[trainIndices]],
      "YTest" -> validY[[testIndices]],
      "TrainIndices" -> trainIndices,
      "TestIndices" -> testIndices
    |>
  ];
  
  result
]

(* Set options for TrainTestSplit *)
Options[TrainTestSplit] = {
  "Stratify" -> False,
  "Shuffle" -> True,
  "RandomSeed" -> 42
};

StratifiedSplit[X_, y_, trainRatio_, shuffle_] := Module[{
  uniqueClasses, trainIndices, testIndices, classIndices, nClassSamples, nTrain, shuffledIndices},
  
  uniqueClasses = Union[y];
  trainIndices = {};
  testIndices = {};
  
  Do[
    classIndices = Position[y, class][[All, 1]];
    nClassSamples = Length[classIndices];
    nTrain = Floor[trainRatio * nClassSamples];
    
    shuffledIndices = If[shuffle, RandomSample[classIndices], classIndices];
    trainIndices = Join[trainIndices, Take[shuffledIndices, nTrain]];
    testIndices = Join[testIndices, Drop[shuffledIndices, nTrain]],
    
    {class, uniqueClasses}
  ];
  
  <|
    "XTrain" -> X[[trainIndices]],
    "XTest" -> X[[testIndices]],
    "YTrain" -> y[[trainIndices]],
    "YTest" -> y[[testIndices]],
    "TrainIndices" -> trainIndices,
    "TestIndices" -> testIndices
  |>
]

(* Cross-Validation Functions *)

CrossValidationSplit[X_, y_, nFolds_: 5, opts___] := Module[{
  validX, validY, stratify, shuffle, randomSeed, nSamples, 
  foldSize, indices, folds, result},
  
  (* Parse options *)
  stratify = OptionValue[CrossValidationSplit, {opts}, "Stratify", False];
  shuffle = OptionValue[CrossValidationSplit, {opts}, "Shuffle", True];
  randomSeed = OptionValue[CrossValidationSplit, {opts}, "RandomSeed", 42];
  
  (* Validate input *)
  validX = ValidateData[X];
  validY = ValidateData[y];
  If[validX === $Failed || validY === $Failed, Return[$Failed]];
  
  (* Ensure matrix format for X *)
  If[VectorQ[validX], validX = {validX}\[Transpose]];
  
  nSamples = Dimensions[validX][[1]];
  If[Length[validY] != nSamples,
    Message[CrossValidationSplit::invdim, "X and y must have same number of samples"];
    Return[$Failed]
  ];
  
  SeedRandom[randomSeed];
  
  (* Create folds *)
  foldSize = Floor[nSamples/nFolds];
  indices = If[shuffle, RandomSample[Range[nSamples]], Range[nSamples]];
  
  folds = Table[
    Module[{testIndices, trainIndices},
      testIndices = Take[indices, {(fold - 1)*foldSize + 1, Min[fold*foldSize, nSamples]}];
      trainIndices = Complement[indices, testIndices];
      
      <|
        "XTrain" -> validX[[trainIndices]],
        "XTest" -> validX[[testIndices]],
        "YTrain" -> validY[[trainIndices]],
        "YTest" -> validY[[testIndices]],
        "TrainIndices" -> trainIndices,
        "TestIndices" -> testIndices
      |>
    ],
    {fold, nFolds}
  ];
  
  folds
]

(* Set options for CrossValidationSplit *)
Options[CrossValidationSplit] = {
  "Stratify" -> False,
  "Shuffle" -> True,
  "RandomSeed" -> 42
};

(* Model Evaluation Functions *)

EvaluateModel[yTrue_, yPred_, task_: "regression", opts___] := Module[{
  metrics, mse, rmse, mae, r2, accuracy, precision, recall, f1, confusion, result},
  
  metrics = OptionValue[EvaluateModel, {opts}, "Metrics", "all"];
  
  result = Switch[task,
    "regression",
      mse = Mean[(yTrue - yPred)^2];
      rmse = Sqrt[mse];
      mae = Mean[Abs[yTrue - yPred]];
      r2 = 1 - Total[(yTrue - yPred)^2]/Total[(yTrue - Mean[yTrue])^2];
      
      <|
        "MSE" -> mse,
        "RMSE" -> rmse,
        "MAE" -> mae,
        "R2" -> r2
      |>,
    
    "classification",
      accuracy = Mean[UnitStep[-Abs[yTrue - yPred]]];
      
      (* Simplified metrics for binary classification *)
      If[Length[Union[yTrue]] == 2,
        Module[{tp, fp, tn, fn},
          tp = Count[Thread[{yTrue, yPred}], {1, 1}];
          fp = Count[Thread[{yTrue, yPred}], {0, 1}];
          tn = Count[Thread[{yTrue, yPred}], {0, 0}];
          fn = Count[Thread[{yTrue, yPred}], {1, 0}];
          
          precision = If[tp + fp > 0, tp/(tp + fp), 0];
          recall = If[tp + fn > 0, tp/(tp + fn), 0];
          f1 = If[precision + recall > 0, 2*precision*recall/(precision + recall), 0];
          
          <|
            "Accuracy" -> accuracy,
            "Precision" -> precision,
            "Recall" -> recall,
            "F1" -> f1,
            "TP" -> tp,
            "FP" -> fp,
            "TN" -> tn,
            "FN" -> fn
          |>
        ],
        
        <|"Accuracy" -> accuracy|>
      ],
    
    _,
      <|"Error" -> "Unknown task type"|>
  ];
  
  result
]

(* Set options for EvaluateModel *)
Options[EvaluateModel] = {"Metrics" -> "all"};

(* Visualization Functions *)

PlotDataQuality[data_, opts___] := Module[{
  title, validData, missingPattern, fig1, fig2, fig3, fig4},
  
  title = OptionValue[PlotDataQuality, {opts}, "Title", "Data Quality Analysis"];
  
  validData = ValidateData[data];
  If[validData === $Failed, Return[$Failed]];
  
  (* Ensure matrix format *)
  If[VectorQ[validData], validData = {validData}\[Transpose]];
  
  (* Missing values heatmap *)
  missingPattern = Map[If[MissingQ[#] || !NumericQ[#], 1, 0] &, validData, {-1}];
  
  fig1 = ArrayPlot[missingPattern,
    ColorRules -> {0 -> White, 1 -> BerkeleyBlue},
    Frame -> True,
    FrameLabel -> {"Features", "Samples"},
    PlotLabel -> "Missing Values (blue = missing)"
  ];
  
  (* Feature distributions *)
  If[Dimensions[validData][[2]] <= 10,
    fig2 = BoxWhiskerChart[Transpose[validData],
      ChartStyle -> BerkeleyBlue,
      Frame -> True,
      FrameLabel -> {"Features", "Values"},
      PlotLabel -> "Feature Distributions"
    ],
    fig2 = Histogram[Flatten[validData],
      ChartStyle -> BerkeleyBlue,
      Frame -> True,
      FrameLabel -> {"Feature Values", "Frequency"},
      PlotLabel -> "Overall Feature Distribution"
    ]
  ];
  
  (* Correlation matrix *)
  If[Dimensions[validData][[2]] <= 20,
    Module[{corrMatrix},
      corrMatrix = Correlation[validData];
      fig3 = ArrayPlot[corrMatrix,
        ColorFunction -> "TemperatureMap",
        Frame -> True,
        FrameLabel -> {"Features", "Features"},
        PlotLabel -> "Feature Correlation Matrix",
        PlotLegends -> Automatic
      ]
    ],
    fig3 = Text["Correlation Matrix\n(Too many features)"]
  ];
  
  (* Summary statistics *)
  fig4 = Text[Grid[{
    {"Samples:", Dimensions[validData][[1]]},
    {"Features:", Dimensions[validData][[2]]},
    {"Missing:", Count[Flatten[missingPattern], 1]},
    {"Complete:", Count[Flatten[missingPattern], 0]}
  }, Frame -> All, Alignment -> Left]];
  
  (* Combine plots *)
  Grid[{{fig1, fig2}, {fig3, fig4}}, Frame -> All, 
       BaseStyle -> {FontSize -> 12, FontFamily -> "Times"},
       Epilog -> Text[Style[title, Bold, 16], Scaled[{0.5, 0.95}]]]
]

Options[PlotDataQuality] = {"Title" -> "Data Quality Analysis"};

PlotModelEvaluation[yTrue_, yPred_, task_: "regression", opts___] := Module[{
  title, fig1, fig2, fig3, fig4, residuals, minVal, maxVal},
  
  title = OptionValue[PlotModelEvaluation, {opts}, "Title", "Model Evaluation"];
  
  Switch[task,
    "regression",
      (* Predictions vs Actual *)
      minVal = Min[Join[yTrue, yPred]];
      maxVal = Max[Join[yTrue, yPred]];
      
      fig1 = ListPlot[Transpose[{yTrue, yPred}],
        PlotStyle -> {PointSize[0.015], BerkeleyBlue},
        Epilog -> {
          CaliforniaGold, Dashed, Thickness[0.005],
          Line[{{minVal, minVal}, {maxVal, maxVal}}]
        },
        Frame -> True,
        FrameLabel -> {"Actual Values", "Predicted Values"},
        PlotLabel -> "Predictions vs Actual",
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]]
      ];
      
      (* Residuals *)
      residuals = yTrue - yPred;
      fig2 = ListPlot[Transpose[{yPred, residuals}],
        PlotStyle -> {PointSize[0.015], BerkeleyBlue},
        Epilog -> {
          CaliforniaGold, Dashed, Thickness[0.005],
          Line[{{Min[yPred], 0}, {Max[yPred], 0}}]
        },
        Frame -> True,
        FrameLabel -> {"Predicted Values", "Residuals"},
        PlotLabel -> "Residual Plot",
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]]
      ];
      
      (* Residual distribution *)
      fig3 = Histogram[residuals,
        ChartStyle -> BerkeleyBlue,
        Frame -> True,
        FrameLabel -> {"Residuals", "Frequency"},
        PlotLabel -> "Residual Distribution"
      ];
      
      (* Metrics *)
      Module[{metrics},
        metrics = EvaluateModel[yTrue, yPred, "regression"];
        fig4 = Text[Grid[{
          {"R²:", NumberForm[metrics["R2"], 4]},
          {"RMSE:", NumberForm[metrics["RMSE"], 4]},
          {"MAE:", NumberForm[metrics["MAE"], 4]},
          {"MSE:", NumberForm[metrics["MSE"], 4]}
        }, Frame -> All, Alignment -> Left]]
      ],
    
    "classification",
      (* Accuracy by class *)
      Module[{uniqueClasses, classAccuracy},
        uniqueClasses = Union[yTrue];
        classAccuracy = Table[
          Mean[UnitStep[-Abs[Select[Thread[{yTrue, yPred}], #[[1]] == class &][[All, 1]] - 
                              Select[Thread[{yTrue, yPred}], #[[1]] == class &][[All, 2]]]]],
          {class, uniqueClasses}
        ];
        
        fig1 = BarChart[classAccuracy,
          ChartLabels -> uniqueClasses,
          ChartStyle -> BerkeleyBlue,
          Frame -> True,
          FrameLabel -> {"Class", "Accuracy"},
          PlotLabel -> "Accuracy by Class"
        ]
      ];
      
      (* Confusion matrix would go here *)
      fig2 = Text["Confusion Matrix\n(Implementation needed)"];
      fig3 = Text["Classification\nResults"];
      fig4 = Text["Performance\nMetrics"],
    
    _,
      fig1 = fig2 = fig3 = fig4 = Text["Unknown task type"]
  ];
  
  (* Combine plots *)
  Grid[{{fig1, fig2}, {fig3, fig4}}, Frame -> All, 
       BaseStyle -> {FontSize -> 12, FontFamily -> "Times"},
       Epilog -> Text[Style[title, Bold, 16], Scaled[{0.5, 0.95}]]]
]

Options[PlotModelEvaluation] = {"Title" -> "Model Evaluation"};

(* Create test datasets *)

CreateTestDatasets[] := Module[{nSamples, X, y, XClassification, yClassification},
  nSamples = 200;
  SeedRandom[42];
  
  (* Regression dataset *)
  X = RandomReal[NormalDistribution[], {nSamples, 3}];
  y = X.{2, -1, 3} + 0.1*RandomReal[NormalDistribution[], nSamples];
  
  (* Classification dataset *)
  XClassification = RandomReal[NormalDistribution[], {nSamples, 2}];
  yClassification = UnitStep[Total[XClassification, {2}]];
  
  <|
    "Regression" -> {X, y},
    "Classification" -> {XClassification, yClassification}
  |>
]

End[]

EndPackage[]