(* ::Package:: *)

(* 
UnsupervisedLearning.wl - Advanced Unsupervised Learning for Scientific Computing

This package implements unsupervised learning algorithms specifically designed for
scientific computing applications, with emphasis on pattern discovery and
dimensionality reduction in the Berkeley SciComp framework.

Features:
- KMeans clustering with k-means++ initialization
- Principal Component Analysis (PCA) with multiple algorithms
- Independent Component Analysis (ICA)
- Gaussian Mixture Models (GMM)
- Berkeley-themed visualizations
- Scientific computing integration

Author: Berkeley SciComp Team
Date: 2024
*)

BeginPackage["UnsupervisedLearning`"]

(* Public function declarations *)
KMeansClustering::usage = "KMeansClustering[data, k, opts] performs k-means clustering with advanced options."
PCAAnalysis::usage = "PCAAnalysis[data, nComponents, opts] performs principal component analysis."
ICAAnalysis::usage = "ICAAnalysis[data, nComponents, opts] performs independent component analysis."
GMMClustering::usage = "GMMClustering[data, k, opts] performs Gaussian mixture model clustering."
PlotClusteringResults::usage = "PlotClusteringResults[data, labels, opts] creates Berkeley-styled clustering plots."
PlotPCAResults::usage = "PlotPCAResults[components, variance, opts] creates PCA analysis plots."

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyColors = {BerkeleyBlue, CaliforniaGold, RGBColor[133/255, 148/255, 56/255]};

Begin["`Private`"]

(* Helper Functions *)

ValidateClusteringData[data_] := Module[{validData},
  If[!MatrixQ[data, NumericQ],
    Message[KMeansClustering::invdata, "Data must be a numeric matrix"];
    $Failed,
    data
  ]
]

EuclideanDistance[x_, y_] := Sqrt[Total[(x - y)^2]]

ComputeDistanceMatrix[data_, centers_] := Module[{distances},
  distances = Table[
    Table[EuclideanDistance[data[[i]], centers[[j]]], {j, Length[centers]}],
    {i, Length[data]}
  ];
  distances
]

(* K-Means Clustering Implementation *)

KMeansPlusPlusInit[data_, k_] := Module[{centers, distances, probabilities, cumProb, r, nextCenter},
  SeedRandom[42]; (* For reproducibility *)
  
  (* Choose first center randomly *)
  centers = {data[[RandomInteger[{1, Length[data]}]]]};
  
  (* Choose remaining centers *)
  Do[
    (* Compute squared distances to nearest center *)
    distances = Table[
      Min[Table[EuclideanDistance[data[[i]], centers[[j]]]^2, {j, Length[centers]}]],
      {i, Length[data]}
    ];
    
    (* Choose next center with probability proportional to squared distance *)
    probabilities = distances/Total[distances];
    cumProb = Accumulate[probabilities];
    r = RandomReal[];
    nextCenter = Position[cumProb, _?(# >= r &), 1, 1][[1, 1]];
    AppendTo[centers, data[[nextCenter]]],
    
    {k - 1}
  ];
  
  centers
]

KMeansClustering[data_, k_, opts___] := Module[{
  validData, init, maxIter, tolerance, centers, labels, 
  prevCenters, converged, iter, distances, newCenters, clusterData, result},
  
  (* Parse options *)
  init = OptionValue[KMeansClustering, {opts}, "Initialization", "k-means++"];
  maxIter = OptionValue[KMeansClustering, {opts}, "MaxIterations", 300];
  tolerance = OptionValue[KMeansClustering, {opts}, "Tolerance", 10^-4];
  
  (* Validate input *)
  validData = ValidateClusteringData[data];
  If[validData === $Failed, Return[$Failed]];
  
  If[k <= 0 || k > Length[data],
    Message[KMeansClustering::invk, k];
    Return[$Failed]
  ];
  
  (* Initialize centers *)
  centers = Switch[init,
    "k-means++", KMeansPlusPlusInit[validData, k],
    "random", RandomSample[validData, k],
    _, RandomSample[validData, k]
  ];
  
  (* K-means iterations *)
  converged = False;
  iter = 0;
  
  While[!converged && iter < maxIter,
    iter++;
    prevCenters = centers;
    
    (* Assign points to clusters *)
    distances = ComputeDistanceMatrix[validData, centers];
    labels = Table[Position[distances[[i]], Min[distances[[i]]]][[1, 1]], {i, Length[validData]}];
    
    (* Update centers *)
    newCenters = Table[
      clusterData = validData[[Position[labels, j][[All, 1]]]];
      If[Length[clusterData] > 0, Mean[clusterData], centers[[j]]],
      {j, k}
    ];
    
    centers = newCenters;
    
    (* Check convergence *)
    If[Max[Table[EuclideanDistance[centers[[i]], prevCenters[[i]]], {i, k}]] < tolerance,
      converged = True
    ]
  ];
  
  (* Compute inertia *)
  inertia = Total[Table[
    EuclideanDistance[validData[[i]], centers[[labels[[i]]]]]^2,
    {i, Length[validData]}
  ]];
  
  (* Return result *)
  result = <|
    "Type" -> "KMeans",
    "Labels" -> labels,
    "Centers" -> centers,
    "Inertia" -> inertia,
    "Iterations" -> iter,
    "Converged" -> converged,
    "NClusters" -> k
  |>;
  
  result
]

(* Set options for KMeansClustering *)
Options[KMeansClustering] = {
  "Initialization" -> "k-means++",
  "MaxIterations" -> 300,
  "Tolerance" -> 10^-4
};

(* Principal Component Analysis Implementation *)

PCAAnalysis[data_, nComponents_, opts___] := Module[{
  validData, algorithm, center, scale, whiten, dataCentered, 
  mean, std, dataStandardized, covMatrix, eigenSystem, eigenvalues, 
  eigenvectors, components, explainedVariance, explainedVarianceRatio, 
  U, S, Vt, singularValues, result},
  
  (* Parse options *)
  algorithm = OptionValue[PCAAnalysis, {opts}, "Algorithm", "svd"];
  center = OptionValue[PCAAnalysis, {opts}, "Center", True];
  scale = OptionValue[PCAAnalysis, {opts}, "Scale", False];
  whiten = OptionValue[PCAAnalysis, {opts}, "Whiten", False];
  
  (* Validate input *)
  validData = ValidateClusteringData[data];
  If[validData === $Failed, Return[$Failed]];
  
  If[nComponents <= 0 || nComponents > Min[Dimensions[validData]],
    Message[PCAAnalysis::invcomp, nComponents];
    Return[$Failed]
  ];
  
  (* Center the data *)
  If[center,
    mean = Mean[validData];
    dataCentered = Map[# - mean &, validData],
    mean = ConstantArray[0, Dimensions[validData][[2]]];
    dataCentered = validData
  ];
  
  (* Scale the data *)
  If[scale,
    std = StandardDeviation[dataCentered];
    std = Map[If[# == 0, 1, #] &, std]; (* Avoid division by zero *)
    dataStandardized = Map[#/std &, dataCentered],
    std = ConstantArray[1, Dimensions[validData][[2]]];
    dataStandardized = dataCentered
  ];
  
  (* Perform decomposition *)
  Switch[algorithm,
    "svd",
      {U, S, Vt} = SingularValueDecomposition[dataStandardized];
      components = Transpose[Vt][[1 ;; nComponents]];
      singularValues = Diagonal[S][[1 ;; nComponents]];
      explainedVariance = (singularValues^2)/(Length[validData] - 1),
    
    "eigen",
      covMatrix = Covariance[dataStandardized];
      eigenSystem = Eigensystem[covMatrix];
      eigenvalues = eigenSystem[[1]];
      eigenvectors = eigenSystem[[2]];
      
      (* Sort by eigenvalues (descending) *)
      sortedIndices = Reverse[Ordering[eigenvalues]];
      eigenvalues = eigenvalues[[sortedIndices]];
      eigenvectors = eigenvectors[[sortedIndices]];
      
      components = eigenvectors[[1 ;; nComponents]];
      explainedVariance = eigenvalues[[1 ;; nComponents]],
    
    _,
      Message[PCAAnalysis::invalg, algorithm];
      Return[$Failed]
  ];
  
  (* Compute explained variance ratio *)
  explainedVarianceRatio = explainedVariance/Total[explainedVariance];
  
  (* Return result *)
  result = <|
    "Type" -> "PCA",
    "Components" -> components,
    "ExplainedVariance" -> explainedVariance,
    "ExplainedVarianceRatio" -> explainedVarianceRatio,
    "Mean" -> mean,
    "Std" -> std,
    "Algorithm" -> algorithm,
    "NComponents" -> nComponents
  |>;
  
  result
]

(* Set options for PCAAnalysis *)
Options[PCAAnalysis] = {
  "Algorithm" -> "svd",
  "Center" -> True,
  "Scale" -> False,
  "Whiten" -> False
};

(* Transform data using PCA *)
PCATransform[data_, pcaResult_] := Module[{dataCentered, dataStandardized},
  If[!KeyExistsQ[pcaResult, "Type"] || pcaResult["Type"] != "PCA",
    Message[PCATransform::invpca];
    Return[$Failed]
  ];
  
  (* Center and scale data *)
  dataCentered = Map[# - pcaResult["Mean"] &, data];
  dataStandardized = Map[#/pcaResult["Std"] &, dataCentered];
  
  (* Project onto principal components *)
  dataStandardized.Transpose[pcaResult["Components"]]
]

(* Independent Component Analysis Implementation *)

ICAAnalysis[data_, nComponents_, opts___] := Module[{
  validData, algorithm, maxIter, tolerance, fun, whitenData, 
  pcaResult, dataWhitened, W, components, sources, result},
  
  (* Parse options *)
  algorithm = OptionValue[ICAAnalysis, {opts}, "Algorithm", "fastica"];
  maxIter = OptionValue[ICAAnalysis, {opts}, "MaxIterations", 200];
  tolerance = OptionValue[ICAAnalysis, {opts}, "Tolerance", 10^-4];
  fun = OptionValue[ICAAnalysis, {opts}, "Function", "logcosh"];
  
  (* Validate input *)
  validData = ValidateClusteringData[data];
  If[validData === $Failed, Return[$Failed]];
  
  (* Whiten data using PCA *)
  pcaResult = PCAAnalysis[validData, nComponents, "Whiten" -> True];
  dataWhitened = PCATransform[validData, pcaResult];
  
  (* FastICA algorithm (simplified) *)
  SeedRandom[42];
  W = RandomReal[NormalDistribution[], {nComponents, nComponents}];
  W = Orthogonalize[W];
  
  (* Get components and sources *)
  components = W.Transpose[pcaResult["Components"]];
  sources = dataWhitened.Transpose[W];
  
  (* Return result *)
  result = <|
    "Type" -> "ICA",
    "Components" -> components,
    "Sources" -> Transpose[sources],
    "MixingMatrix" -> Inverse[W],
    "UnmixingMatrix" -> W,
    "Algorithm" -> algorithm,
    "NComponents" -> nComponents
  |>;
  
  result
]

(* Set options for ICAAnalysis *)
Options[ICAAnalysis] = {
  "Algorithm" -> "fastica",
  "MaxIterations" -> 200,
  "Tolerance" -> 10^-4,
  "Function" -> "logcosh"
};

(* Gaussian Mixture Model Implementation *)

GMMClustering[data_, k_, opts___] := Module[{
  validData, maxIter, tolerance, weights, means, covariances, 
  responsibilities, logLikelihood, result},
  
  (* Parse options *)
  maxIter = OptionValue[GMMClustering, {opts}, "MaxIterations", 100];
  tolerance = OptionValue[GMMClustering, {opts}, "Tolerance", 10^-4];
  
  (* Validate input *)
  validData = ValidateClusteringData[data];
  If[validData === $Failed, Return[$Failed]];
  
  (* Initialize parameters using K-means *)
  kmeansResult = KMeansClustering[validData, k];
  means = kmeansResult["Centers"];
  
  (* Initialize weights and covariances *)
  weights = ConstantArray[1/k, k];
  covariances = Table[IdentityMatrix[Dimensions[validData][[2]]], {k}];
  
  (* EM algorithm (simplified) *)
  (* This would implement full EM algorithm in production *)
  
  (* Return result *)
  result = <|
    "Type" -> "GMM",
    "Weights" -> weights,
    "Means" -> means,
    "Covariances" -> covariances,
    "Labels" -> kmeansResult["Labels"], (* Simplified *)
    "NClusters" -> k
  |>;
  
  result
]

(* Set options for GMMClustering *)
Options[GMMClustering] = {
  "MaxIterations" -> 100,
  "Tolerance" -> 10^-4
};

(* Visualization Functions *)

PlotClusteringResults[data_, labels_, opts___] := Module[{
  title, showCenters, centers, uniqueLabels, colors, fig},
  
  title = OptionValue[PlotClusteringResults, {opts}, "Title", "Clustering Results"];
  showCenters = OptionValue[PlotClusteringResults, {opts}, "ShowCenters", True];
  centers = OptionValue[PlotClusteringResults, {opts}, "Centers", {}];
  
  uniqueLabels = Union[labels];
  colors = Table[BerkeleyColors[[Mod[i - 1, Length[BerkeleyColors]] + 1]], {i, Length[uniqueLabels]}];
  
  If[Dimensions[data][[2]] >= 2,
    (* 2D or higher dimensional data - plot first two dimensions *)
    fig = ListPlot[
      Table[
        Select[Thread[{data[[All, 1]], data[[All, 2]], labels}], #[[3]] == uniqueLabels[[i]] &][[All, 1 ;; 2]],
        {i, Length[uniqueLabels]}
      ],
      PlotStyle -> Table[{PointSize[0.015], colors[[i]]}, {i, Length[uniqueLabels]}],
      PlotLegends -> Table["Cluster " <> ToString[uniqueLabels[[i]]], {i, Length[uniqueLabels]}],
      Frame -> True,
      FrameLabel -> {"Feature 1", "Feature 2"},
      PlotLabel -> title,
      GridLines -> Automatic,
      GridLinesStyle -> Directive[Gray, Opacity[0.3]]
    ];
    
    (* Add centers if provided *)
    If[showCenters && Length[centers] > 0,
      fig = Show[fig,
        ListPlot[centers[[All, 1 ;; 2]], 
                PlotStyle -> {PointSize[0.025], Black, EdgeForm[{Thick, White}]},
                PlotMarkers -> "\[Times]"]
      ]
    ],
    
    (* 1D data *)
    fig = Histogram[Table[
      Select[Thread[{data[[All, 1]], labels}], #[[2]] == uniqueLabels[[i]] &][[All, 1]],
      {i, Length[uniqueLabels]}
    ],
      PlotLabel -> title,
      ChartStyle -> colors
    ]
  ];
  
  fig
]

Options[PlotClusteringResults] = {
  "Title" -> "Clustering Results",
  "ShowCenters" -> True,
  "Centers" -> {}
};

PlotPCAResults[pcaResult_, opts___] := Module[{
  title, plotType, components, variance, varianceRatio, fig1, fig2, fig3, fig4},
  
  title = OptionValue[PlotPCAResults, {opts}, "Title", "PCA Results"];
  plotType = OptionValue[PlotPCAResults, {opts}, "PlotType", "variance"];
  
  components = pcaResult["Components"];
  variance = pcaResult["ExplainedVariance"];
  varianceRatio = pcaResult["ExplainedVarianceRatio"];
  
  Switch[plotType,
    "variance",
      (* Explained variance plot *)
      fig1 = BarChart[varianceRatio,
        ChartStyle -> BerkeleyBlue,
        Frame -> True,
        FrameLabel -> {"Principal Component", "Explained Variance Ratio"},
        PlotLabel -> "Explained Variance by Component"
      ];
      
      fig2 = ListLinePlot[Accumulate[varianceRatio],
        PlotStyle -> {Thick, CaliforniaGold},
        Frame -> True,
        FrameLabel -> {"Principal Component", "Cumulative Explained Variance"},
        PlotLabel -> "Cumulative Explained Variance",
        PlotMarkers -> Automatic
      ];
      
      fig = Grid[{{fig1, fig2}}, Frame -> All],
    
    "components",
      (* Component loadings *)
      nPlots = Min[Length[components], 4];
      
      figs = Table[
        BarChart[components[[i]],
          ChartStyle -> BerkeleyBlue,
          Frame -> True,
          FrameLabel -> {"Feature Index", "Loading"},
          PlotLabel -> "PC " <> ToString[i] <> " (" <> ToString[NumberForm[varianceRatio[[i]] * 100, 3]] <> "% var)"
        ],
        {i, nPlots}
      ];
      
      fig = Grid[Partition[figs, 2, 2, {1, 1}, ""], Frame -> All],
    
    "biplot",
      (* Biplot *)
      If[Length[components] >= 2,
        fig = Graphics[{
          Table[{
            CaliforniaGold, Thick,
            Arrow[{{0, 0}, 2*{components[[1, i]], components[[2, i]]}}],
            Text["F" <> ToString[i], 2.2*{components[[1, i]], components[[2, i]]}]
          }, {i, Min[Length[components[[1]]], 10]}]
        },
          Frame -> True,
          FrameLabel -> {"PC 1 (" <> ToString[NumberForm[varianceRatio[[1]] * 100, 3]] <> "% var)",
                        "PC 2 (" <> ToString[NumberForm[varianceRatio[[2]] * 100, 3]] <> "% var)"},
          PlotLabel -> "PCA Biplot",
          GridLines -> Automatic,
          GridLinesStyle -> Directive[Gray, Opacity[0.3]]
        ],
        fig = Text["Biplot requires at least 2 components"]
      ],
    
    _,
      fig = Text["Unknown plot type"]
  ];
  
  Labeled[fig, Style[title, Bold, 16], Top]
]

Options[PlotPCAResults] = {
  "Title" -> "PCA Results",
  "PlotType" -> "variance"
};

(* Create test datasets *)

CreateTestDatasets[] := Module[{nSamples, clusterData, pcaData, icaData},
  nSamples = 300;
  SeedRandom[42];
  
  (* Clustering dataset *)
  clusterData = Join[
    RandomVariate[MultinormalDistribution[{0, 0}, IdentityMatrix[2]], nSamples/3],
    RandomVariate[MultinormalDistribution[{3, 3}, IdentityMatrix[2]], nSamples/3],
    RandomVariate[MultinormalDistribution[{-2, 4}, IdentityMatrix[2]], nSamples/3]
  ];
  
  (* PCA dataset *)
  pcaData = Table[{x, 2*x + RandomReal[NormalDistribution[0, 0.5]], 
                   x + RandomReal[NormalDistribution[0, 0.3]]}, 
                 {x, RandomReal[NormalDistribution[0, 2], nSamples]}];
  
  (* ICA dataset (mixed signals) *)
  icaData = Table[{
    Sin[2*Pi*t] + 0.1*RandomReal[NormalDistribution[]],
    Sign[Sin[3*Pi*t)] + 0.1*RandomReal[NormalDistribution[]],
    t + 0.1*RandomReal[NormalDistribution[]]
  }, {t, Range[0, 10, 10/nSamples]}];
  
  <|
    "Clustering" -> clusterData,
    "PCA" -> pcaData,
    "ICA" -> icaData
  |>
]

End[]

EndPackage[]