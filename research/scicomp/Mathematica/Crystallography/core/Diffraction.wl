(* Diffraction - X-ray diffraction pattern simulation and analysis *)
(*
 * Features:
 * - Structure factor calculations
 * - Powder diffraction pattern simulation
 * - Peak profile modeling (Gaussian, Lorentzian, Voigt)
 * - Atomic scattering factor database
 * - Systematic absence prediction
 *
 * Example:
 *   crystal = CrystalStructure[lattice, atoms];
 *   diffraction = DiffractionPattern[crystal];
 *   {twoTheta, intensity} = SimulatePowderPattern[diffraction, 1.54056];
 *)

BeginPackage["Diffraction`", {"CrystalStructure`", "SpaceGroups`"}]

(* Public symbols *)
AtomicScatteringFactor::usage = "AtomicScatteringFactor[element, sinThetaOverLambda] calculates atomic scattering factor"
StructureFactor::usage = "StructureFactor[crystal, h, k, l, wavelength] calculates structure factor F_hkl"
StructureFactorIntensity::usage = "StructureFactorIntensity[crystal, h, k, l, wavelength] calculates |F_hkl|²"
DiffractionPattern::usage = "DiffractionPattern[crystal, spaceGroup] creates diffraction pattern object"
SimulatePowderPattern::usage = "SimulatePowderPattern[pattern, wavelength, options] simulates powder diffraction pattern"
CalculateReflections::usage = "CalculateReflections[crystal, wavelength, maxTwoTheta, maxHKL] calculates all allowed reflections"
GaussianProfile::usage = "GaussianProfile[twoTheta, center, fwhm] generates Gaussian peak profile"
LorentzianProfile::usage = "LorentzianProfile[twoTheta, center, fwhm] generates Lorentzian peak profile"
VoigtProfile::usage = "VoigtProfile[twoTheta, center, fwhmG, fwhmL] generates Voigt peak profile"

Begin["`Private`"]

(* Atomic scattering factor database *)
(* International Tables for Crystallography, Vol. C *)
scatteringFactorDatabase = <|
  "H" -> <|"a" -> {0.493002, 0.322912, 0.140191, 0.040810},
           "b" -> {10.5109, 26.1257, 3.14236, 57.7997},
           "c" -> 0.003038|>,
  "C" -> <|"a" -> {2.31000, 1.02000, 1.58860, 0.865000},
           "b" -> {20.8439, 10.2075, 0.568700, 51.6512},
           "c" -> 0.215600|>,
  "N" -> <|"a" -> {12.2126, 3.13220, 2.01250, 1.16630},
           "b" -> {0.005700, 9.89330, 28.9975, 0.582600},
           "c" -> -11.529|>,
  "O" -> <|"a" -> {3.04850, 2.28680, 1.54630, 0.867000},
           "b" -> {13.2771, 5.70110, 0.323900, 32.9089},
           "c" -> 0.250800|>,
  "Na" -> <|"a" -> {4.76260, 3.17360, 1.26740, 1.11280},
            "b" -> {3.28500, 8.84220, 0.313600, 129.424},
            "c" -> 0.676000|>,
  "Cl" -> <|"a" -> {11.4604, 7.19640, 6.25560, 1.64550},
            "b" -> {0.010400, 1.16620, 18.5194, 47.7784},
            "c" -> -9.55740|>,
  "Si" -> <|"a" -> {6.29150, 3.03530, 1.98910, 1.54100},
            "b" -> {2.43860, 32.3337, 0.678500, 81.6937},
            "c" -> 1.14070|>,
  "Fe" -> <|"a" -> {11.7695, 7.35730, 3.52220, 2.30450},
            "b" -> {4.76110, 0.307200, 15.3535, 76.8805},
            "c" -> 1.03690|>,
  "Cu" -> <|"a" -> {13.3380, 7.16760, 5.61580, 1.67350},
            "b" -> {3.58280, 0.247000, 11.3966, 64.8126},
            "c" -> 1.19100|>
|>;

(* Calculate atomic scattering factor *)
AtomicScatteringFactor[element_String, sinThetaOverLambda_?NumericQ] := Module[{params, a, b, c, sSquared, fReal},
  If[KeyExistsQ[scatteringFactorDatabase, element],
    params = scatteringFactorDatabase[element];
    a = params["a"];
    b = params["b"];
    c = params["c"];
    
    sSquared = sinThetaOverLambda^2;
    
    (* Calculate real part of scattering factor *)
    fReal = Sum[a[[i]] Exp[-b[[i]] sSquared], {i, 4}] + c;
    
    (* Return as complex number (ignoring anomalous scattering for now) *)
    Complex[fReal, 0],
    
    (* Default to carbon if element not found *)
    Message[AtomicScatteringFactor::unknown, element];
    AtomicScatteringFactor["C", sinThetaOverLambda]
  ]
]

(* Calculate structure factor *)
StructureFactor[crystal_Association, h_Integer, k_Integer, l_Integer, wavelength_: 1.54056] := Module[{
  dSpacing, sinTheta, sinThetaOverLambda, structureFactor, atoms, atom, fj, thermalFactor, phase, phaseFactor, contribution},
  
  (* Calculate sin(θ)/λ *)
  dSpacing = DSpacing[crystal, h, k, l];
  If[dSpacing === $Failed, Return[$Failed]];
  
  sinTheta = wavelength/(2 dSpacing);
  sinThetaOverLambda = sinTheta/wavelength;
  
  structureFactor = 0;
  atoms = crystal["atoms"];
  
  (* Sum over all atoms in unit cell *)
  Do[
    atom = atoms[[i]];
    
    (* Get atomic scattering factor *)
    fj = AtomicScatteringFactor[atom["element"], sinThetaOverLambda];
    
    (* Apply thermal factor (Debye-Waller factor) *)
    thermalFactor = Exp[-atom["thermalFactor"] sinThetaOverLambda^2];
    
    (* Phase factor *)
    phase = 2 Pi (h atom["x"] + k atom["y"] + l atom["z"]);
    phaseFactor = Exp[I phase];
    
    (* Add contribution *)
    contribution = atom["occupancy"] fj thermalFactor phaseFactor;
    structureFactor += contribution,
    
    {i, Length[atoms]}
  ];
  
  structureFactor
]

(* Calculate structure factor intensity *)
StructureFactorIntensity[crystal_Association, h_Integer, k_Integer, l_Integer, wavelength_: 1.54056] := Module[{fHkl},
  fHkl = StructureFactor[crystal, h, k, l, wavelength];
  If[fHkl === $Failed, $Failed, Abs[fHkl]^2]
]

(* Create diffraction pattern object *)
DiffractionPattern[crystal_Association, spaceGroup_: Null] := <|
  "crystal" -> crystal,
  "spaceGroup" -> spaceGroup
|>

(* Calculate all reflections *)
CalculateReflections[crystal_Association, wavelength_: 1.54056, maxTwoTheta_: 80.0, maxHKL_: 5] := Module[{
  reflections, h, k, l, dSpacing, sinTheta, theta, twoTheta, fHkl, intensity, multiplicity, correctedIntensity},
  
  reflections = {};
  
  Do[
    If[h == 0 && k == 0 && l == 0, Continue[]];
    
    (* Try to calculate d-spacing and angle *)
    dSpacing = DSpacing[crystal, h, k, l];
    If[dSpacing === $Failed, Continue[]];
    
    sinTheta = wavelength/(2 dSpacing);
    If[sinTheta > 1.0, Continue[]]; (* No diffraction possible *)
    
    theta = ArcSin[sinTheta];
    twoTheta = 2 theta / Degree;
    
    If[twoTheta > maxTwoTheta, Continue[]];
    
    (* Calculate structure factor and intensity *)
    fHkl = StructureFactor[crystal, h, k, l, wavelength];
    If[fHkl === $Failed, Continue[]];
    
    intensity = Abs[fHkl]^2;
    If[intensity < 10^-6, Continue[]]; (* Skip systematically absent reflections *)
    
    (* Calculate multiplicity (simplified) *)
    multiplicity = CalculateMultiplicity[h, k, l];
    
    (* Apply corrections *)
    correctedIntensity = ApplyIntensityCorrections[intensity, theta, multiplicity];
    
    AppendTo[reflections, <|
      "h" -> h, "k" -> k, "l" -> l,
      "dSpacing" -> dSpacing,
      "twoTheta" -> twoTheta,
      "intensity" -> correctedIntensity,
      "structureFactor" -> fHkl,
      "multiplicity" -> multiplicity
    |>],
    
    {h, -maxHKL, maxHKL}, {k, -maxHKL, maxHKL}, {l, -maxHKL, maxHKL}
  ];
  
  (* Sort by 2θ and normalize intensities *)
  reflections = SortBy[reflections, #["twoTheta"] &];
  
  If[Length[reflections] > 0,
    Module[{maxIntensity},
      maxIntensity = Max[#["intensity"] & /@ reflections];
      reflections = Map[ReplacePart[#, "intensity" -> 100 #["intensity"]/maxIntensity] &, reflections];
    ]
  ];
  
  reflections
]

(* Calculate reflection multiplicity (simplified) *)
CalculateMultiplicity[h_Integer, k_Integer, l_Integer] := Module[{uniqueIndices, zeroIndices},
  uniqueIndices = Length[DeleteDuplicates[{Abs[h], Abs[k], Abs[l]}]];
  zeroIndices = Count[{h, k, l}, 0];
  
  Which[
    zeroIndices == 2, 6,   (* e.g., (1,0,0) *)
    zeroIndices == 1, 12,  (* e.g., (1,1,0) *)
    uniqueIndices == 1, 8, (* e.g., (1,1,1) *)
    uniqueIndices == 2, 24,(* e.g., (1,1,2) *)
    True, 48               (* e.g., (1,2,3) *)
  ]
]

(* Apply intensity corrections *)
ApplyIntensityCorrections[intensity_, theta_, multiplicity_] := Module[{lorentzFactor, polarizationFactor},
  (* Lorentz factor *)
  lorentzFactor = 1.0/(Sin[theta] Sin[2 theta]);
  
  (* Polarization factor (unpolarized radiation) *)
  polarizationFactor = (1 + Cos[2 theta]^2)/2;
  
  (* Apply corrections *)
  intensity * multiplicity * lorentzFactor * polarizationFactor
]

(* Simulate powder diffraction pattern *)
SimulatePowderPattern[pattern_Association, wavelength_: 1.54056, opts___] := Module[{
  crystal, twoThetaRange, stepSize, peakWidth, background, maxHKL, twoThetaMin, twoThetaMax, 
  twoTheta, intensity, reflections, refl},
  
  (* Parse options *)
  {twoThetaRange, stepSize, peakWidth, background, maxHKL} = {
    "TwoThetaRange", "StepSize", "PeakWidth", "Background", "MaxHKL"
  } /. {opts} /. {
    "TwoThetaRange" -> {10.0, 80.0},
    "StepSize" -> 0.02,
    "PeakWidth" -> 0.1,
    "Background" -> 10.0,
    "MaxHKL" -> 5
  };
  
  crystal = pattern["crystal"];
  {twoThetaMin, twoThetaMax} = twoThetaRange;
  
  (* Generate 2θ array *)
  twoTheta = Range[twoThetaMin, twoThetaMax, stepSize];
  intensity = ConstantArray[background, Length[twoTheta]];
  
  (* Calculate reflections *)
  reflections = CalculateReflections[crystal, wavelength, twoThetaMax, maxHKL];
  
  (* Add peaks to pattern *)
  Do[
    refl = reflections[[i]];
    If[twoThetaMin <= refl["twoTheta"] <= twoThetaMax,
      (* Add Gaussian peak *)
      intensity += refl["intensity"] GaussianProfile[twoTheta, refl["twoTheta"], peakWidth];
    ],
    {i, Length[reflections]}
  ];
  
  {twoTheta, intensity}
]

(* Peak profile functions *)
GaussianProfile[twoTheta_, center_, fwhm_] := Module[{sigma},
  sigma = fwhm/(2 Sqrt[2 Log[2]]);
  Exp[-0.5 ((twoTheta - center)/sigma)^2]
]

LorentzianProfile[twoTheta_, center_, fwhm_] := Module[{gamma},
  gamma = fwhm/2;
  gamma^2/((twoTheta - center)^2 + gamma^2)
]

VoigtProfile[twoTheta_, center_, fwhmGaussian_, fwhmLorentzian_] := Module[{eta, gaussian, lorentzian},
  (* Pseudo-Voigt approximation *)
  eta = 1.36603 (fwhmLorentzian/(fwhmGaussian + fwhmLorentzian)) - 
        0.47719 (fwhmLorentzian/(fwhmGaussian + fwhmLorentzian))^2 + 
        0.11116 (fwhmLorentzian/(fwhmGaussian + fwhmLorentzian))^3;
  
  gaussian = GaussianProfile[twoTheta, center, fwhmGaussian];
  lorentzian = LorentzianProfile[twoTheta, center, fwhmLorentzian];
  
  eta lorentzian + (1 - eta) gaussian
]

(* Peak finding and analysis *)
FindPeaks[twoTheta_, intensity_, prominence_: 10.0, minDistance_: 0.1] := Module[{peaks, i, tooClose, y1, y2, y3, x1, x2, x3, denom, a, b, peakPos, peakIntensity},
  peaks = {};
  
  Do[
    (* Check if it's a local maximum *)
    If[intensity[[i]] > intensity[[i-1]] && intensity[[i]] > intensity[[i+1]] && 
       intensity[[i]] > prominence,
      
      (* Check minimum distance to existing peaks *)
      tooClose = AnyTrue[peaks, Abs[twoTheta[[i]] - #["twoTheta"]] < minDistance &];
      
      If[!tooClose,
        (* Fit peak position more precisely (parabolic interpolation) *)
        If[i > 1 && i < Length[intensity],
          {y1, y2, y3} = intensity[[{i-1, i, i+1}]];
          {x1, x2, x3} = twoTheta[[{i-1, i, i+1}]];
          
          denom = (x1-x2)(x1-x3)(x2-x3);
          If[Abs[denom] > 10^-10,
            a = (x3(y2-y1) + x2(y1-y3) + x1(y3-y2))/denom;
            b = (x3^2(y1-y2) + x2^2(y3-y1) + x1^2(y2-y3))/denom;
            
            If[a < 0, (* Peak (not valley) *)
              peakPos = -b/(2a);
              peakIntensity = a peakPos^2 + b peakPos + (x1^2(x2 y3-x3 y2) + x1(x3^2 y2-x2^2 y3) + x2^2 x3 y1-x2 x3^2 y1)/denom,
              
              peakPos = twoTheta[[i]];
              peakIntensity = intensity[[i]];
            ],
            
            peakPos = twoTheta[[i]];
            peakIntensity = intensity[[i]];
          ],
          
          peakPos = twoTheta[[i]];
          peakIntensity = intensity[[i]];
        ];
        
        AppendTo[peaks, <|
          "twoTheta" -> peakPos,
          "intensity" -> peakIntensity,
          "dSpacing" -> wavelength/(2 Sin[Degree peakPos/2])
        |>];
      ]
    ],
    {i, 2, Length[intensity] - 1}
  ];
  
  peaks
]

(* Error messages *)
AtomicScatteringFactor::unknown = "Unknown element: ``. Using carbon instead."

End[]
EndPackage[]