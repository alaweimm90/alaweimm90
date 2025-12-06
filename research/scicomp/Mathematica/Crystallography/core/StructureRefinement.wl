(* StructureRefinement - Advanced crystallographic refinement methods *)
(*
 * Features:
 * - Least squares refinement
 * - Rietveld refinement for powder data
 * - Parameter constraint handling
 * - Uncertainty estimation
 * - Goodness-of-fit statistics
 *
 * Example:
 *   crystal = CrystalStructure[lattice, atoms];
 *   refinement = LeastSquaresRefinement[crystal, observedIntensities, reflectionData];
 *   results = RefineStructure[refinement];
 *)

BeginPackage["StructureRefinement`", {"CrystalStructure`", "Diffraction`"}]

(* Public symbols *)
LeastSquaresRefinement::usage = "LeastSquaresRefinement[crystal, observedData, reflectionData] creates least squares refinement object"
RietveldRefinement::usage = "RietveldRefinement[crystal, twoThetaObs, intensityObs, wavelength] creates Rietveld refinement object"
RefineStructure::usage = "RefineStructure[refinement, options] performs structure refinement"
RefinementParameter::usage = "RefinementParameter[name, value, refined, bounds] defines refinement parameter"
CalculateRFactors::usage = "CalculateRFactors[observed, calculated, weights] calculates R-factors"
EstimateUncertainties::usage = "EstimateUncertainties[jacobian, residuals] estimates parameter uncertainties"

Begin["`Private`"]

(* Refinement parameter structure *)
CreateRefinementParameter[name_String, value_?NumericQ, refined_: True, 
  lowerBound_: -Infinity, upperBound_: Infinity] := <|
  "name" -> name,
  "value" -> value,
  "refined" -> refined,
  "lowerBound" -> lowerBound,
  "upperBound" -> upperBound,
  "uncertainty" -> Undefined
|>

(* Least squares refinement object *)
LeastSquaresRefinement[crystal_Association, observedIntensities_List, reflectionData_List] := Module[{parameters, parameterOrder},
  (* Setup initial parameters from crystal structure *)
  parameters = SetupLeastSquaresParameters[crystal];
  parameterOrder = Select[Keys[parameters], parameters[#]["refined"] &];
  
  <|
    "type" -> "LeastSquares",
    "crystal" -> crystal,
    "observedIntensities" -> observedIntensities,
    "reflectionData" -> reflectionData,
    "parameters" -> parameters,
    "parameterOrder" -> parameterOrder,
    "maxIterations" -> 100,
    "convergenceTolerance" -> 10^-6
  |>
]

(* Rietveld refinement object *)
RietveldRefinement[crystal_Association, twoThetaObs_List, intensityObs_List, wavelength_: 1.54056] := Module[{parameters, parameterOrder},
  (* Setup Rietveld parameters *)
  parameters = SetupRietveldParameters[crystal];
  parameterOrder = Select[Keys[parameters], parameters[#]["refined"] &];
  
  <|
    "type" -> "Rietveld",
    "crystal" -> crystal,
    "twoThetaObs" -> twoThetaObs,
    "intensityObs" -> intensityObs,
    "wavelength" -> wavelength,
    "parameters" -> parameters,
    "parameterOrder" -> parameterOrder,
    "peakShape" -> "PseudoVoigt",
    "backgroundType" -> "Polynomial",
    "backgroundOrder" -> 5,
    "maxIterations" -> 1000,
    "convergenceTolerance" -> 10^-8
  |>
]

(* Setup parameters for least squares refinement *)
SetupLeastSquaresParameters[crystal_Association] := Module[{parameters, lattice, atoms},
  parameters = <||>;
  lattice = crystal["lattice"];
  atoms = crystal["atoms"];
  
  (* Lattice parameters *)
  parameters["a"] = CreateRefinementParameter["a", lattice["a"], True, 0.1, 50.0];
  parameters["b"] = CreateRefinementParameter["b", lattice["b"], True, 0.1, 50.0];
  parameters["c"] = CreateRefinementParameter["c", lattice["c"], True, 0.1, 50.0];
  parameters["alpha"] = CreateRefinementParameter["alpha", lattice["alpha"], True, 30.0, 150.0];
  parameters["beta"] = CreateRefinementParameter["beta", lattice["beta"], True, 30.0, 150.0];
  parameters["gamma"] = CreateRefinementParameter["gamma", lattice["gamma"], True, 30.0, 150.0];
  
  (* Atomic parameters *)
  Do[
    parameters["x_" <> ToString[i]] = CreateRefinementParameter["x_" <> ToString[i], atoms[[i]]["x"], True, 0.0, 1.0];
    parameters["y_" <> ToString[i]] = CreateRefinementParameter["y_" <> ToString[i], atoms[[i]]["y"], True, 0.0, 1.0];
    parameters["z_" <> ToString[i]] = CreateRefinementParameter["z_" <> ToString[i], atoms[[i]]["z"], True, 0.0, 1.0];
    parameters["occ_" <> ToString[i]] = CreateRefinementParameter["occ_" <> ToString[i], 
      Lookup[atoms[[i]], "occupancy", 1.0], True, 0.0, 1.0];
    parameters["B_" <> ToString[i]] = CreateRefinementParameter["B_" <> ToString[i], 
      Lookup[atoms[[i]], "thermalFactor", 0.0], True, 0.0, 50.0];,
    {i, Length[atoms]}
  ];
  
  parameters
]

(* Setup parameters for Rietveld refinement *)
SetupRietveldParameters[crystal_Association] := Module[{parameters, lattice, atoms},
  parameters = SetupLeastSquaresParameters[crystal];
  
  (* Profile parameters *)
  parameters["U"] = CreateRefinementParameter["U", 0.01, True, 0.0, 1.0];
  parameters["V"] = CreateRefinementParameter["V", 0.01, True, -1.0, 1.0];
  parameters["W"] = CreateRefinementParameter["W", 0.01, True, 0.0, 1.0];
  parameters["eta"] = CreateRefinementParameter["eta", 0.5, True, 0.0, 1.0];
  
  (* Scale factor *)
  parameters["scale"] = CreateRefinementParameter["scale", 1.0, True, 0.001, 1000.0];
  
  (* Background parameters *)
  Do[
    initialValue = If[i == 0, 10.0, 0.0];
    parameters["bg_" <> ToString[i]] = CreateRefinementParameter["bg_" <> ToString[i], initialValue, True];,
    {i, 0, 5}
  ];
  
  (* Zero point correction *)
  parameters["zeroPoint"] = CreateRefinementParameter["zeroPoint", 0.0, True, -0.1, 0.1];
  
  parameters
]

(* Get parameter vector for optimization *)
GetParameterVector[refinement_Association] := Module[{parameters, parameterOrder},
  parameters = refinement["parameters"];
  parameterOrder = refinement["parameterOrder"];
  
  parameters[#]["value"] & /@ parameterOrder
]

(* Set parameter vector from optimization *)
SetParameterVector[refinement_Association, values_List] := Module[{parameters, parameterOrder},
  parameters = refinement["parameters"];
  parameterOrder = refinement["parameterOrder"];
  
  Do[
    parameters[parameterOrder[[i]]]["value"] = values[[i]],
    {i, Length[parameterOrder]}
  ];
  
  (* Update crystal structure *)
  UpdateCrystalStructure[refinement]
]

(* Update crystal structure from current parameters *)
UpdateCrystalStructure[refinement_Association] := Module[{parameters, lattice, atoms, newLattice, newAtoms},
  parameters = refinement["parameters"];
  
  (* Update lattice parameters *)
  newLattice = <|
    "a" -> parameters["a"]["value"],
    "b" -> parameters["b"]["value"],
    "c" -> parameters["c"]["value"],
    "alpha" -> parameters["alpha"]["value"],
    "beta" -> parameters["beta"]["value"],
    "gamma" -> parameters["gamma"]["value"]
  |>;
  
  (* Update atomic positions *)
  atoms = refinement["crystal"]["atoms"];
  newAtoms = Table[
    <|
      "element" -> atoms[[i]]["element"],
      "x" -> parameters["x_" <> ToString[i]]["value"],
      "y" -> parameters["y_" <> ToString[i]]["value"],
      "z" -> parameters["z_" <> ToString[i]]["value"],
      "occupancy" -> parameters["occ_" <> ToString[i]]["value"],
      "thermalFactor" -> parameters["B_" <> ToString[i]]["value"]
    |>,
    {i, Length[atoms]}
  ];
  
  refinement["crystal"] = CrystalStructure[newLattice, newAtoms];
]

(* Calculate structure factors for all reflections *)
CalculateStructureFactors[refinement_Association] := Module[{crystal, reflectionData, calculated},
  crystal = refinement["crystal"];
  reflectionData = refinement["reflectionData"];
  
  calculated = Table[
    Module[{refl = reflectionData[[i]]},
      Abs[StructureFactor[crystal, refl["h"], refl["k"], refl["l"]]]
    ],
    {i, Length[reflectionData]}
  ];
  
  calculated
]

(* Residual function for least squares *)
LeastSquaresResidual[refinement_Association, parameterValues_List] := Module[{
  observed, calculated, weights, residuals},
  
  (* Update parameters *)
  SetParameterVector[refinement, parameterValues];
  
  (* Calculate model intensities *)
  calculated = CalculateStructureFactors[refinement];
  observed = refinement["observedIntensities"];
  
  (* Calculate residuals with unit weights for now *)
  weights = ConstantArray[1.0, Length[observed]];
  residuals = weights (observed - calculated);
  
  residuals
]

(* Calculate powder pattern for Rietveld *)
CalculatePowderPattern[refinement_Association, parameterValues_: Null] := Module[{
  crystal, twoThetaObs, wavelength, parameters, calculatedIntensity, background, reflections, scale, zeroPoint},
  
  If[parameterValues =!= Null, SetParameterVector[refinement, parameterValues]];
  
  crystal = refinement["crystal"];
  twoThetaObs = refinement["twoThetaObs"];
  wavelength = refinement["wavelength"];
  parameters = refinement["parameters"];
  
  (* Initialize pattern with background *)
  background = CalculateBackground[refinement];
  calculatedIntensity = background;
  
  (* Generate reflections *)
  reflections = CalculateReflections[crystal, wavelength, Max[twoThetaObs], 5];
  
  (* Add Bragg peaks *)
  scale = parameters["scale"]["value"];
  zeroPoint = parameters["zeroPoint"]["value"];
  
  Do[
    Module[{refl = reflections[[i]], peakPosition, fwhm, profile},
      peakPosition = refl["twoTheta"] + zeroPoint;
      
      (* Calculate peak width using Caglioti function *)
      fwhm = CalculatePeakWidth[refinement, peakPosition];
      
      (* Calculate peak profile *)
      profile = CalculatePeakProfile[refinement, twoThetaObs, peakPosition, fwhm];
      
      (* Add scaled peak to pattern *)
      calculatedIntensity += scale refl["intensity"] profile;
    ],
    {i, Length[reflections]}
  ];
  
  calculatedIntensity
]

(* Calculate background *)
CalculateBackground[refinement_Association] := Module[{parameters, twoThetaObs, background, i},
  parameters = refinement["parameters"];
  twoThetaObs = refinement["twoThetaObs"];
  
  background = ConstantArray[0.0, Length[twoThetaObs]];
  
  (* Polynomial background *)
  Do[
    Module[{bgCoeff = parameters["bg_" <> ToString[i]]["value"]},
      background += bgCoeff (twoThetaObs/100.0)^i;
    ],
    {i, 0, refinement["backgroundOrder"]}
  ];
  
  background
]

(* Calculate peak width using Caglioti function *)
CalculatePeakWidth[refinement_Association, twoTheta_] := Module[{parameters, tanTheta, U, V, W, fwhmSquared},
  parameters = refinement["parameters"];
  tanTheta = Tan[Degree twoTheta/2];
  
  U = parameters["U"]["value"];
  V = parameters["V"]["value"];  
  W = parameters["W"]["value"];
  
  fwhmSquared = U tanTheta^2 + V tanTheta + W;
  Sqrt[Max[fwhmSquared, 0.001]] (* Prevent negative FWHM *)
]

(* Calculate peak profile *)
CalculatePeakProfile[refinement_Association, twoTheta_List, center_, fwhm_] := Module[{peakShape, eta, gaussian, lorentzian},
  peakShape = refinement["peakShape"];
  
  Switch[peakShape,
    "Gaussian", GaussianProfile[twoTheta, center, fwhm],
    "Lorentzian", LorentzianProfile[twoTheta, center, fwhm],
    "PseudoVoigt",
      eta = refinement["parameters"]["eta"]["value"];
      gaussian = GaussianProfile[twoTheta, center, fwhm];
      lorentzian = LorentzianProfile[twoTheta, center, fwhm];
      eta lorentzian + (1 - eta) gaussian,
    _, GaussianProfile[twoTheta, center, fwhm]
  ]
]

(* Rietveld residual function *)
RietveldResidual[refinement_Association, parameterValues_List] := Module[{
  calculated, observed, weights, residuals},
  
  calculated = CalculatePowderPattern[refinement, parameterValues];
  observed = refinement["intensityObs"];
  
  (* Statistical weights *)
  weights = 1.0/Sqrt[Max[#, 1.0] & /@ observed];
  
  residuals = weights (observed - calculated);
  
  residuals
]

(* Main refinement function *)
RefineStructure[refinement_Association, opts___] := Module[{
  initialParams, residualFunction, result, finalParams, uncertainties, rFactors, chiSquared, gof},
  
  initialParams = GetParameterVector[refinement];
  
  (* Choose residual function based on refinement type *)
  residualFunction = Switch[refinement["type"],
    "LeastSquares", LeastSquaresResidual[refinement, #] &,
    "Rietveld", RietveldResidual[refinement, #] &,
    _, LeastSquaresResidual[refinement, #] &
  ];
  
  (* Perform nonlinear least squares *)
  result = FindMinimum[
    Norm[residualFunction[vars]]^2,
    Table[{vars[[i]], initialParams[[i]]}, {i, Length[initialParams]}],
    Method -> "LevenbergMarquardt",
    MaxIterations -> refinement["maxIterations"]
  ];
  
  (* Extract final parameters *)
  finalParams = vars /. result[[2]];
  SetParameterVector[refinement, finalParams];
  
  (* Calculate uncertainties and statistics *)
  uncertainties = EstimateUncertainties[refinement, finalParams];
  rFactors = CalculateRFactors[refinement];
  chiSquared = result[[1]];
  
  (* Goodness of fit *)
  Module[{nData, nParams},
    nData = Switch[refinement["type"],
      "LeastSquares", Length[refinement["observedIntensities"]],
      "Rietveld", Length[refinement["intensityObs"]],
      _, 0
    ];
    nParams = Length[refinement["parameterOrder"]];
    gof = If[nData > nParams, Sqrt[chiSquared/(nData - nParams)], 0.0];
  ];
  
  <|
    "converged" -> True,
    "finalParameters" -> AssociationThread[refinement["parameterOrder"], finalParams],
    "parameterUncertainties" -> uncertainties,
    "rFactors" -> rFactors,
    "chiSquared" -> chiSquared,
    "goodnessOfFit" -> gof,
    "finalResidual" -> Sqrt[chiSquared]
  |>
]

(* Estimate parameter uncertainties *)
EstimateUncertainties[refinement_Association, finalParams_List] := Module[{jacobian, covMatrix, uncertainties},
  (* Calculate numerical Jacobian *)
  jacobian = CalculateNumericalJacobian[refinement, finalParams];
  
  (* Calculate covariance matrix *)
  If[MatrixQ[jacobian] && Det[Transpose[jacobian].jacobian] != 0,
    covMatrix = Inverse[Transpose[jacobian].jacobian];
    uncertainties = Sqrt[Diagonal[covMatrix]];
    AssociationThread[refinement["parameterOrder"], uncertainties],
    
    (* Return empty association if calculation fails *)
    <||>
  ]
]

(* Calculate numerical Jacobian *)
CalculateNumericalJacobian[refinement_Association, params_List] := Module[{
  jacobian, stepSize, residualFunction, nData, nParams, i, paramsPlus, paramsMinus, residualsPlus, residualsMinus},
  
  residualFunction = Switch[refinement["type"],
    "LeastSquares", LeastSquaresResidual[refinement, #] &,
    "Rietveld", RietveldResidual[refinement, #] &,
    _, LeastSquaresResidual[refinement, #] &
  ];
  
  nData = Length[residualFunction[params]];
  nParams = Length[params];
  jacobian = ConstantArray[0.0, {nData, nParams}];
  stepSize = 10^-6;
  
  Do[
    (* Forward difference *)
    paramsPlus = params;
    paramsPlus[[i]] += stepSize;
    residualsPlus = residualFunction[paramsPlus];
    
    (* Backward difference *)
    paramsMinus = params;
    paramsMinus[[i]] -= stepSize;
    residualsMinus = residualFunction[paramsMinus];
    
    (* Central difference *)
    jacobian[[All, i]] = (residualsPlus - residualsMinus)/(2 stepSize);,
    {i, nParams}
  ];
  
  jacobian
]

(* Calculate R-factors *)
CalculateRFactors[refinement_Association] := Module[{observed, calculated, weights, rFactor, weightedRFactor},
  Switch[refinement["type"],
    "LeastSquares",
      observed = refinement["observedIntensities"];
      calculated = CalculateStructureFactors[refinement];
      weights = ConstantArray[1.0, Length[observed]];,
    "Rietveld",
      observed = refinement["intensityObs"];
      calculated = CalculatePowderPattern[refinement];
      weights = 1.0/Sqrt[Max[#, 1.0] & /@ observed];
  ];
  
  (* R-factor *)
  rFactor = Total[Abs[observed - calculated]]/Total[observed];
  
  (* Weighted R-factor *)
  weightedRFactor = Sqrt[Total[weights^2 (observed - calculated)^2]]/Sqrt[Total[weights^2 observed^2]];
  
  <|"R" -> rFactor, "Rw" -> weightedRFactor|>
]

End[]
EndPackage[]