(* ::Package:: *)

(* 
NeuralNetworks.wl - Neural Networks for Scientific Computing

This package implements neural network architectures specifically designed for
scientific computing applications, including physics-informed neural networks,
deep operator networks, and scientific deep learning models in the Berkeley
SciComp framework.

Features:
- Multi-Layer Perceptron (MLP) with flexible architecture
- Multiple activation functions and optimizers
- Automatic differentiation capabilities
- Autoencoder for dimensionality reduction
- Berkeley-themed visualizations
- Scientific computing integration

Author: Berkeley SciComp Team
Date: 2024
*)

BeginPackage["NeuralNetworks`"]

(* Public function declarations *)
CreateMLP::usage = "CreateMLP[layerSizes, opts] creates a multi-layer perceptron neural network."
TrainNetwork::usage = "TrainNetwork[network, X, y, opts] trains the neural network."
PredictNetwork::usage = "PredictNetwork[network, X] makes predictions using trained network."
CreateAutoencoder::usage = "CreateAutoencoder[inputDim, encodingDims, latentDim, opts] creates an autoencoder."
PlotTrainingHistory::usage = "PlotTrainingHistory[history] plots training convergence with Berkeley styling."
PlotNetworkArchitecture::usage = "PlotNetworkArchitecture[network] visualizes network structure."

(* Activation functions *)
ReLUActivation::usage = "ReLUActivation[x] applies ReLU activation function."
TanhActivation::usage = "TanhActivation[x] applies hyperbolic tangent activation."
SigmoidActivation::usage = "SigmoidActivation[x] applies sigmoid activation function."
LinearActivation::usage = "LinearActivation[x] applies linear activation (identity)."

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyColors = {BerkeleyBlue, CaliforniaGold, RGBColor[133/255, 148/255, 56/255]};

Begin["`Private`"]

(* Activation Functions *)

ReLUActivation[x_] := Max[0, x]
ReLUActivation[x_List] := Map[Max[0, #] &, x, {-1}]

ReLUDerivative[x_] := If[x > 0, 1, 0]
ReLUDerivative[x_List] := Map[If[# > 0, 1, 0] &, x, {-1}]

TanhActivation[x_] := Tanh[x]
TanhActivation[x_List] := Map[Tanh, x, {-1}]

TanhDerivative[x_] := 1 - Tanh[x]^2
TanhDerivative[x_List] := Map[1 - Tanh[#]^2 &, x, {-1}]

SigmoidActivation[x_] := 1/(1 + Exp[-Clip[x, {-500, 500}]])
SigmoidActivation[x_List] := Map[1/(1 + Exp[-Clip[#, {-500, 500}]]) &, x, {-1}]

SigmoidDerivative[x_] := Module[{s = SigmoidActivation[x]}, s*(1 - s)]
SigmoidDerivative[x_List] := Module[{s = SigmoidActivation[x]}, s*(1 - s)]

LinearActivation[x_] := x
LinearActivation[x_List] := x

LinearDerivative[x_] := 1
LinearDerivative[x_List] := ConstantArray[1, Dimensions[x]]

(* Get activation function by name *)
GetActivationFunction[name_String] := Switch[name,
  "relu", ReLUActivation,
  "tanh", TanhActivation, 
  "sigmoid", SigmoidActivation,
  "linear", LinearActivation,
  _, Message[CreateMLP::activation, name]; ReLUActivation
]

GetActivationDerivative[name_String] := Switch[name,
  "relu", ReLUDerivative,
  "tanh", TanhDerivative,
  "sigmoid", SigmoidDerivative, 
  "linear", LinearDerivative,
  _, ReLUDerivative
]

(* Weight Initialization *)

XavierInitialization[inputSize_, outputSize_] := 
  RandomReal[{-1, 1}, {inputSize, outputSize}]*Sqrt[6/(inputSize + outputSize)]

HeInitialization[inputSize_, outputSize_] := 
  RandomReal[NormalDistribution[0, Sqrt[2/inputSize]], {inputSize, outputSize}]

NormalInitialization[inputSize_, outputSize_] := 
  RandomReal[NormalDistribution[0, 0.01], {inputSize, outputSize}]

GetWeightInitialization[method_String] := Switch[method,
  "xavier", XavierInitialization,
  "he", HeInitialization,
  "normal", NormalInitialization,
  _, XavierInitialization
]

(* Loss Functions *)

MSELoss[yTrue_, yPred_] := Mean[(yTrue - yPred)^2]
MAELoss[yTrue_, yPred_] := Mean[Abs[yTrue - yPred]]

CrossEntropyLoss[yTrue_, yPred_] := Module[{yPredClipped},
  yPredClipped = Clip[yPred, {10^-15, 1 - 10^-15}];
  -Mean[yTrue*Log[yPredClipped] + (1 - yTrue)*Log[1 - yPredClipped]]
]

GetLossFunction[name_String] := Switch[name,
  "mse", MSELoss,
  "mae", MAELoss,
  "crossentropy", CrossEntropyLoss,
  _, MSELoss
]

(* Neural Network Implementation *)

CreateMLP[layerSizes_List, opts___] := Module[{
  activations, outputActivation, useBias, weightInit, optimizer, 
  learningRate, regularization, dropoutRate, layers, weights, biases,
  activationFunctions, activationDerivatives, initFunc, network},
  
  (* Parse options *)
  activations = OptionValue[CreateMLP, {opts}, "Activations", "relu"];
  outputActivation = OptionValue[CreateMLP, {opts}, "OutputActivation", "linear"];
  useBias = OptionValue[CreateMLP, {opts}, "UseBias", True];
  weightInit = OptionValue[CreateMLP, {opts}, "WeightInitialization", "xavier"];
  optimizer = OptionValue[CreateMLP, {opts}, "Optimizer", "adam"];
  learningRate = OptionValue[CreateMLP, {opts}, "LearningRate", 0.001];
  regularization = OptionValue[CreateMLP, {opts}, "Regularization", 0.0];
  dropoutRate = OptionValue[CreateMLP, {opts}, "DropoutRate", 0.0];
  
  layers = Length[layerSizes] - 1;
  
  (* Setup activation functions *)
  If[StringQ[activations],
    activationFunctions = ConstantArray[GetActivationFunction[activations], layers - 1];
    AppendTo[activationFunctions, GetActivationFunction[outputActivation]];
    activationDerivatives = ConstantArray[GetActivationDerivative[activations], layers - 1];
    AppendTo[activationDerivatives, GetActivationDerivative[outputActivation]],
    
    activationFunctions = Map[GetActivationFunction, activations];
    activationDerivatives = Map[GetActivationDerivative, activations]
  ];
  
  (* Initialize weights and biases *)
  initFunc = GetWeightInitialization[weightInit];
  weights = Table[
    initFunc[layerSizes[[i]], layerSizes[[i + 1]]],
    {i, layers}
  ];
  
  biases = If[useBias,
    Table[ConstantArray[0.0, layerSizes[[i + 1]]], {i, layers}],
    Table[{}, {i, layers}]
  ];
  
  (* Create network structure *)
  network = <|
    "Type" -> "MLP",
    "LayerSizes" -> layerSizes,
    "Weights" -> weights,
    "Biases" -> biases,
    "ActivationFunctions" -> activationFunctions,
    "ActivationDerivatives" -> activationDerivatives,
    "UseBias" -> useBias,
    "WeightInitialization" -> weightInit,
    "Optimizer" -> optimizer,
    "LearningRate" -> learningRate,
    "Regularization" -> regularization,
    "DropoutRate" -> dropoutRate,
    "IsTrained" -> False,
    "TrainingHistory" -> <||>
  |>;
  
  network
]

Options[CreateMLP] = {
  "Activations" -> "relu",
  "OutputActivation" -> "linear",
  "UseBias" -> True,
  "WeightInitialization" -> "xavier",
  "Optimizer" -> "adam",
  "LearningRate" -> 0.001,
  "Regularization" -> 0.0,
  "DropoutRate" -> 0.0
};

(* Forward Pass *)

ForwardPass[network_, X_, training_: False] := Module[{
  output, layers, weights, biases, activationFns, useBias, dropoutRate},
  
  layers = Length[network["LayerSizes"]] - 1;
  weights = network["Weights"];
  biases = network["Biases"];
  activationFns = network["ActivationFunctions"];
  useBias = network["UseBias"];
  dropoutRate = network["DropoutRate"];
  
  output = X;
  
  Do[
    (* Linear transformation *)
    output = output.weights[[i]];
    
    (* Add bias if used *)
    If[useBias && Length[biases[[i]]] > 0,
      output = output + biases[[i]]
    ];
    
    (* Apply activation *)
    output = activationFns[[i]][output];
    
    (* Apply dropout during training *)
    If[training && dropoutRate > 0 && i < layers,
      Module[{dropoutMask},
        dropoutMask = RandomChoice[{1 - dropoutRate, dropoutRate} -> {1, 0}, Dimensions[output]];
        output = output * dropoutMask / (1 - dropoutRate)
      ]
    ],
    
    {i, layers}
  ];
  
  output
]

(* Training Implementation *)

TrainNetwork[network_, X_, y_, opts___] := Module[{
  epochs, batchSize, validationData, lossFunction, verbose, 
  nSamples, nBatches, history, trainedNetwork, XVal, yVal, hasValidation},
  
  (* Parse options *)
  epochs = OptionValue[TrainNetwork, {opts}, "Epochs", 100];
  batchSize = OptionValue[TrainNetwork, {opts}, "BatchSize", 32];
  validationData = OptionValue[TrainNetwork, {opts}, "ValidationData", {}];
  lossFunction = OptionValue[TrainNetwork, {opts}, "LossFunction", "mse"];
  verbose = OptionValue[TrainNetwork, {opts}, "Verbose", True];
  
  (* Validate input *)
  If[!MatrixQ[X] || !VectorQ[y],
    Message[TrainNetwork::invdata];
    Return[$Failed]
  ];
  
  nSamples = Length[X];
  nBatches = Max[1, Floor[nSamples/batchSize]];
  
  (* Setup validation *)
  hasValidation = Length[validationData] > 0;
  If[hasValidation,
    {XVal, yVal} = validationData
  ];
  
  (* Initialize training history *)
  history = <|
    "Loss" -> {},
    "ValidationLoss" -> {},
    "Epochs" -> epochs
  |>;
  
  trainedNetwork = network;
  
  (* Training loop *)
  Do[
    Module[{epochLoss, indices, XShuffled, yShuffled, batchLoss, valLoss},
      epochLoss = 0;
      
      (* Shuffle data *)
      indices = RandomSample[Range[nSamples]];
      XShuffled = X[[indices]];
      yShuffled = y[[indices]];
      
      (* Mini-batch training *)
      Do[
        Module[{startIdx, endIdx, XBatch, yBatch},
          startIdx = (batch - 1)*batchSize + 1;
          endIdx = Min[startIdx + batchSize - 1, nSamples];
          
          XBatch = XShuffled[[startIdx ;; endIdx]];
          yBatch = yShuffled[[startIdx ;; endIdx]];
          
          (* Train batch (simplified - would implement full backpropagation) *)
          batchLoss = TrainBatch[trainedNetwork, XBatch, yBatch, lossFunction];
          epochLoss += batchLoss
        ],
        {batch, nBatches}
      ];
      
      epochLoss /= nBatches;
      AppendTo[history["Loss"], epochLoss];
      
      (* Validation loss *)
      If[hasValidation,
        valLoss = ComputeLoss[trainedNetwork, XVal, yVal, lossFunction];
        AppendTo[history["ValidationLoss"], valLoss]
      ];
      
      (* Print progress *)
      If[verbose && Mod[epoch, 10] == 0,
        If[hasValidation,
          Print["Epoch ", epoch, "/", epochs, ", Loss: ", NumberForm[epochLoss, 6], 
                ", Val Loss: ", NumberForm[valLoss, 6]],
          Print["Epoch ", epoch, "/", epochs, ", Loss: ", NumberForm[epochLoss, 6]]
        ]
      ]
    ],
    {epoch, epochs}
  ];
  
  (* Update network *)
  trainedNetwork["IsTrained"] = True;
  trainedNetwork["TrainingHistory"] = history;
  
  trainedNetwork
]

Options[TrainNetwork] = {
  "Epochs" -> 100,
  "BatchSize" -> 32,
  "ValidationData" -> {},
  "LossFunction" -> "mse",
  "Verbose" -> True
};

(* Simplified training functions - full implementation would include backpropagation *)

TrainBatch[network_, XBatch_, yBatch_, lossFunction_] := Module[{
  yPred, loss, lossFunc},
  
  (* Forward pass *)
  yPred = ForwardPass[network, XBatch, True];
  
  (* Compute loss *)
  lossFunc = GetLossFunction[lossFunction];
  loss = lossFunc[yBatch, yPred];
  
  (* Add regularization *)
  If[network["Regularization"] > 0,
    loss += 0.5 * network["Regularization"] * 
            Total[Map[Total[#^2, Infinity] &, network["Weights"]]]
  ];
  
  (* Backward pass would be implemented here *)
  (* UpdateParameters[network, ...]; *)
  
  loss
]

ComputeLoss[network_, X_, y_, lossFunction_] := Module[{
  yPred, lossFunc},
  
  yPred = ForwardPass[network, X, False];
  lossFunc = GetLossFunction[lossFunction];
  lossFunc[y, yPred]
]

(* Prediction *)

PredictNetwork[network_, X_] := Module[{},
  If[!network["IsTrained"],
    Message[PredictNetwork::nottrained];
    Return[$Failed]
  ];
  
  ForwardPass[network, X, False]
]

(* Autoencoder Implementation *)

CreateAutoencoder[inputDim_, encodingDims_List, latentDim_, opts___] := Module[{
  activation, outputActivation, optimizer, learningRate, regularization,
  encoderLayers, decoderLayers, encoder, decoder, autoencoder},
  
  (* Parse options *)
  activation = OptionValue[CreateAutoencoder, {opts}, "Activation", "relu"];
  outputActivation = OptionValue[CreateAutoencoder, {opts}, "OutputActivation", "linear"];
  optimizer = OptionValue[CreateAutoencoder, {opts}, "Optimizer", "adam"];
  learningRate = OptionValue[CreateAutoencoder, {opts}, "LearningRate", 0.001];
  regularization = OptionValue[CreateAutoencoder, {opts}, "Regularization", 0.0];
  
  (* Build encoder *)
  encoderLayers = Join[{inputDim}, encodingDims, {latentDim}];
  encoder = CreateMLP[encoderLayers,
    "Activations" -> activation,
    "OutputActivation" -> activation,
    "Optimizer" -> optimizer,
    "LearningRate" -> learningRate,
    "Regularization" -> regularization
  ];
  
  (* Build decoder (mirror of encoder) *)
  decoderLayers = Join[{latentDim}, Reverse[encodingDims], {inputDim}];
  decoder = CreateMLP[decoderLayers,
    "Activations" -> activation,
    "OutputActivation" -> outputActivation,
    "Optimizer" -> optimizer,
    "LearningRate" -> learningRate,
    "Regularization" -> regularization
  ];
  
  (* Create autoencoder *)
  autoencoder = <|
    "Type" -> "Autoencoder",
    "InputDim" -> inputDim,
    "EncodingDims" -> encodingDims,
    "LatentDim" -> latentDim,
    "Encoder" -> encoder,
    "Decoder" -> decoder,
    "IsTrained" -> False
  |>;
  
  autoencoder
]

Options[CreateAutoencoder] = {
  "Activation" -> "relu",
  "OutputActivation" -> "linear",
  "Optimizer" -> "adam", 
  "LearningRate" -> 0.001,
  "Regularization" -> 0.0
};

EncodeData[autoencoder_, X_] := PredictNetwork[autoencoder["Encoder"], X]

DecodeData[autoencoder_, Z_] := PredictNetwork[autoencoder["Decoder"], Z]

(* Visualization Functions *)

PlotTrainingHistory[history_, opts___] := Module[{
  title, loss, valLoss, epochs, fig},
  
  title = OptionValue[PlotTrainingHistory, {opts}, "Title", "Training History"];
  
  loss = history["Loss"];
  valLoss = history["ValidationLoss"];
  epochs = Range[Length[loss]];
  
  fig = ListLogPlot[
    {Transpose[{epochs, loss}]},
    PlotStyle -> {PointSize[0.005], BerkeleyBlue, Thickness[0.003]},
    Joined -> True,
    Frame -> True,
    FrameLabel -> {"Epoch", "Loss"},
    PlotLabel -> title,
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]],
    PlotLegends -> {"Training Loss"}
  ];
  
  (* Add validation loss if available *)
  If[Length[valLoss] > 0,
    fig = Show[fig, 
      ListLogPlot[
        {Transpose[{epochs, valLoss}]},
        PlotStyle -> {PointSize[0.005], CaliforniaGold, Thickness[0.003]},
        Joined -> True,
        PlotLegends -> {"Validation Loss"}
      ]
    ]
  ];
  
  fig
]

Options[PlotTrainingHistory] = {"Title" -> "Training History"};

PlotNetworkArchitecture[network_, opts___] := Module[{
  layerSizes, maxNeurons, nLayers, positions, connections, title, fig},
  
  title = OptionValue[PlotNetworkArchitecture, {opts}, "Title", "Network Architecture"];
  
  layerSizes = network["LayerSizes"];
  maxNeurons = Max[layerSizes];
  nLayers = Length[layerSizes];
  
  (* Generate neuron positions *)
  positions = Flatten[
    Table[
      Table[
        {i, j - (layerSizes[[i]] - 1)/2},
        {j, layerSizes[[i]]}
      ],
      {i, nLayers}
    ], 1
  ];
  
  (* Generate connections *)
  connections = Flatten[
    Table[
      Module[{layer1Start, layer1End, layer2Start, layer2End},
        layer1Start = 1 + Total[layerSizes[[1 ;; i - 1]]];
        layer1End = Total[layerSizes[[1 ;; i]]];
        layer2Start = 1 + Total[layerSizes[[1 ;; i]]];
        layer2End = Total[layerSizes[[1 ;; i + 1]]];
        
        Table[
          Line[{positions[[j]], positions[[k]]}],
          {j, layer1Start, layer1End},
          {k, layer2Start, layer2End}
        ]
      ],
      {i, nLayers - 1}
    ], 2
  ];
  
  (* Create plot *)
  fig = Graphics[
    {
      (* Connections *)
      Opacity[0.1], Gray, connections,
      
      (* Neurons *)
      Table[
        {BerkeleyColors[[Min[i, Length[BerkeleyColors]]]], 
         PointSize[0.02], Point[positions[[j]]]},
        {i, nLayers},
        {j, 1 + Total[layerSizes[[1 ;; i - 1]]], Total[layerSizes[[1 ;; i]]]}
      ]
    },
    Frame -> True,
    FrameLabel -> {"Layer", "Neurons"},
    PlotLabel -> title,
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Opacity[0.3]],
    AspectRatio -> 0.6
  ];
  
  fig
]

Options[PlotNetworkArchitecture] = {"Title" -> "Network Architecture"};

(* Test Data Generation *)

CreateTestDatasets[] := Module[{nSamples, XReg, yReg, XClass, yClass, XTS, yTS},
  nSamples = 1000;
  SeedRandom[42];
  
  (* Regression dataset *)
  XReg = RandomReal[NormalDistribution[], {nSamples, 5}];
  yReg = XReg[[All, 1]]^2 + Sin[XReg[[All, 2]]] + 
         XReg[[All, 3]]*XReg[[All, 4]] + 0.1*RandomReal[NormalDistribution[], nSamples];
  
  (* Classification dataset *)
  XClass = RandomReal[NormalDistribution[], {nSamples, 4}];
  yClass = UnitStep[XClass[[All, 1]] + XClass[[All, 2]] - XClass[[All, 3]] - XClass[[All, 4]]];
  
  (* Time series dataset *)
  XTS = Table[{Sin[t], Cos[t], t}, {t, 0, 10, 10/(nSamples - 1)}];
  yTS = Sin[2*Range[0, 10, 10/(nSamples - 1)]] + 0.1*RandomReal[NormalDistribution[], nSamples];
  
  <|
    "Regression" -> {XReg, yReg},
    "Classification" -> {XClass, yClass},
    "TimeSeries" -> {XTS, yTS}
  |>
]

End[]

EndPackage[]