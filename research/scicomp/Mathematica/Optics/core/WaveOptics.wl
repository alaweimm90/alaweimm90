(* ::Package:: *)

(* ::Title:: *)
(*Wave Optics Package - Mathematica Implementation*)

(* ::Subtitle:: *)
(*Berkeley SciComp Framework*)

(* ::Author:: *)
(*Berkeley SciComp Team*)

(* ::Date:: *)
(*2024*)

BeginPackage["BerkeleySciComp`Optics`WaveOptics`"]

(* ::Section:: *)
(*Usage Messages*)

WaveOptics::usage = "WaveOptics represents a wave optics system with wavelength and medium properties.";
PlaneWave::usage = "PlaneWave[wavelength, amplitude, phase, direction] creates a plane wave.";
SphericalWave::usage = "SphericalWave[wavelength, power, position] creates a spherical wave.";
GaussianBeam::usage = "GaussianBeam[wavelength, waistRadius, opts] creates a Gaussian beam.";
CalculateDiffraction::usage = "CalculateDiffraction[type, size, wavelength, distance, screenSize] calculates diffraction patterns.";
AnalyzeInterference::usage = "AnalyzeInterference[wavelength, separation, distance, opts] analyzes interference patterns.";
FresnelNumber::usage = "FresnelNumber[radius, wavelength, distance] calculates the Fresnel number.";
RayleighRange::usage = "RayleighRange[wavelength, waistRadius, n] calculates the Rayleigh range.";

(* ::Section:: *)
(*Physical Constants*)

$SpeedOfLight = 2.99792458*^8; (* m/s *)
$PlanckConstant = 6.62607015*^-34; (* J⋅s *)

(* Berkeley Color Scheme *)
$BerkeleyBlue = RGBColor[0, 50/255, 98/255];
$CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
$BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

(* Common Wavelengths (meters) *)
$CommonWavelengths = <|
    "UV_A" -> 365*^-9,
    "Violet" -> 400*^-9,
    "Blue" -> 450*^-9,
    "Green" -> 532*^-9,
    "Yellow" -> 589*^-9,
    "Red" -> 633*^-9,
    "NIR" -> 800*^-9,
    "Telecom_C" -> 1550*^-9
|>;

(* ::Section:: *)
(*Begin Private Context*)

Begin["`Private`"]

(* ::Subsection:: *)
(*WaveOptics Class*)

WaveOptics[wavelengthVacuum_, mediumIndex_: 1.0] := Module[{obj},
    obj = <|
        "WavelengthVacuum" -> wavelengthVacuum,
        "MediumIndex" -> mediumIndex,
        "Wavelength" -> wavelengthVacuum/mediumIndex,
        "WaveNumber" -> 2*Pi/(wavelengthVacuum/mediumIndex),
        "Frequency" -> $SpeedOfLight/wavelengthVacuum
    |>;
    obj
];

(* ::Subsection:: *)
(*Plane Wave*)

PlaneWave[wavelength_, amplitude_, phase_: 0, direction_: {0, 0, 1}] := Module[{obj},
    obj = <|
        "Type" -> "PlaneWave",
        "WavelengthVacuum" -> wavelength,
        "Amplitude" -> amplitude,
        "Phase" -> phase,
        "Direction" -> Normalize[direction],
        "WaveNumber" -> 2*Pi/wavelength
    |>;
    
    obj["FieldAtPoint"] = Function[{r, t},
        With[{k = obj["WaveNumber"], d = obj["Direction"]},
            obj["Amplitude"]*Exp[I*(k*d.r - 2*Pi*$SpeedOfLight/wavelength*t + obj["Phase"])]
        ]
    ];
    
    obj["IntensityAtPoint"] = Function[{r},
        obj["Amplitude"]^2
    ];
    
    obj["PropagateDistance"] = Function[{distance},
        PlaneWave[wavelength, amplitude, phase + obj["WaveNumber"]*distance, direction]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Spherical Wave*)

SphericalWave[wavelength_, sourcePower_, sourcePosition_: {0, 0, 0}] := Module[{obj},
    obj = <|
        "Type" -> "SphericalWave",
        "WavelengthVacuum" -> wavelength,
        "SourcePower" -> sourcePower,
        "SourcePosition" -> sourcePosition,
        "WaveNumber" -> 2*Pi/wavelength
    |>;
    
    obj["FieldAtPoint"] = Function[{r, t},
        With[{
            distance = Norm[r - obj["SourcePosition"]],
            k = obj["WaveNumber"]
        },
            If[distance > 0,
                Sqrt[obj["SourcePower"]/(4*Pi)]/distance*
                Exp[I*(k*distance - 2*Pi*$SpeedOfLight/wavelength*t)],
                0
            ]
        ]
    ];
    
    obj["IntensityAtPoint"] = Function[{r},
        With[{distance = Norm[r - obj["SourcePosition"]]},
            If[distance > 0,
                obj["SourcePower"]/(4*Pi*distance^2),
                0
            ]
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Gaussian Beam*)

GaussianBeam[wavelength_, waistRadius_, opts___] := Module[{
    options, waistPosition, mediumIndex, power, obj
},
    options = Association[opts];
    waistPosition = Lookup[options, "WaistPosition", 0];
    mediumIndex = Lookup[options, "MediumIndex", 1.0];
    power = Lookup[options, "Power", 1*^-3];
    
    obj = <|
        "Type" -> "GaussianBeam",
        "WavelengthVacuum" -> wavelength,
        "WaistRadius" -> waistRadius,
        "WaistPosition" -> waistPosition,
        "MediumIndex" -> mediumIndex,
        "Power" -> power,
        "Wavelength" -> wavelength/mediumIndex,
        "WaveNumber" -> 2*Pi/(wavelength/mediumIndex),
        "RayleighRange" -> Pi*waistRadius^2/(wavelength/mediumIndex),
        "DivergenceAngle" -> (wavelength/mediumIndex)/(Pi*waistRadius)
    |>;
    
    obj["BeamRadius"] = Function[{z},
        obj["WaistRadius"]*Sqrt[1 + ((z - obj["WaistPosition"])/obj["RayleighRange"])^2]
    ];
    
    obj["RadiusOfCurvature"] = Function[{z},
        With[{zRel = z - obj["WaistPosition"]},
            If[Abs[zRel] < 10^-12,
                Infinity,
                zRel*(1 + (obj["RayleighRange"]/zRel)^2)
            ]
        ]
    ];
    
    obj["GouyPhase"] = Function[{z},
        ArcTan[(z - obj["WaistPosition"])/obj["RayleighRange"]]
    ];
    
    obj["FieldProfile"] = Function[{x, y, z},
        With[{
            rSquared = x^2 + y^2,
            wz = obj["BeamRadius"][z],
            Rz = obj["RadiusOfCurvature"][z],
            phiz = obj["GouyPhase"][z],
            k = obj["WaveNumber"]
        },
            With[{
                amplitude = obj["WaistRadius"]/wz*Sqrt[2*obj["Power"]/(Pi*obj["WaistRadius"]^2)],
                phaseCurvature = If[Rz === Infinity, 0, k*rSquared/(2*Rz)]
            },
                amplitude*Exp[-rSquared/wz^2]*
                Exp[-I*(k*(z - obj["WaistPosition"]) - phiz + phaseCurvature)]
            ]
        ]
    ];
    
    obj["IntensityProfile"] = Function[{x, y, z},
        Abs[obj["FieldProfile"][x, y, z]]^2
    ];
    
    obj["PropagateDistance"] = Function[{distance},
        GaussianBeam[wavelength, waistRadius, 
            "WaistPosition" -> obj["WaistPosition"] + distance,
            "MediumIndex" -> mediumIndex,
            "Power" -> power
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Diffraction Calculations*)

CalculateDiffraction[apertureType_String, apertureSize_, wavelength_, screenDistance_, screenSize_, opts___] := Module[{
    options, numPoints, x, intensity, result
},
    options = Association[opts];
    numPoints = Lookup[options, "NumPoints", 1000];
    
    x = Table[i, {i, -screenSize/2, screenSize/2, screenSize/(numPoints-1)}];
    
    intensity = Switch[apertureType,
        "single_slit",
        SingleSlitDiffraction[x, apertureSize, wavelength, screenDistance],
        
        "double_slit",
        DoubleSlitDiffraction[x, apertureSize, wavelength, screenDistance],
        
        "circular",
        CircularAperture[x, apertureSize, wavelength, screenDistance],
        
        _,
        Message[CalculateDiffraction::unknowntype, apertureType];
        $Failed
    ];
    
    If[intensity === $Failed, Return[$Failed]];
    
    result = <|
        "Position" -> x,
        "Intensity" -> intensity,
        "IntensityNormalized" -> intensity/Max[intensity],
        "ApertureType" -> apertureType,
        "ApertureSize" -> apertureSize,
        "Wavelength" -> wavelength,
        "ScreenDistance" -> screenDistance
    |>;
    
    result
];

CalculateDiffraction::unknowntype = "Unknown aperture type: `1`";

(* Single slit diffraction *)
SingleSlitDiffraction[x_List, slitWidth_, wavelength_, distance_] := Module[{theta, beta},
    theta = x/distance; (* Small angle approximation *)
    beta = Pi*slitWidth*Sin[theta]/wavelength;
    beta = beta /. {b_ /; Abs[b] < 10^-10 :> 10^-10}; (* Avoid division by zero *)
    (Sin[beta]/beta)^2
];

(* Double slit diffraction *)
DoubleSlitDiffraction[x_List, slitSeparation_, wavelength_, distance_] := Module[{
    slitWidth, theta, beta, alpha, envelope, interference
},
    slitWidth = slitSeparation/10; (* Assume thin slits *)
    theta = x/distance;
    beta = Pi*slitWidth*Sin[theta]/wavelength;
    alpha = Pi*slitSeparation*Sin[theta]/wavelength;
    
    beta = beta /. {b_ /; Abs[b] < 10^-10 :> 10^-10};
    envelope = (Sin[beta]/beta)^2;
    interference = Cos[alpha]^2;
    envelope*interference
];

(* Circular aperture (Airy disk) *)
CircularAperture[x_List, apertureRadius_, wavelength_, distance_] := Module[{theta, u},
    theta = x/distance;
    u = Pi*apertureRadius*Sin[theta]/wavelength;
    u = u /. {val_ /; Abs[val] < 10^-10 :> 10^-10};
    (2*BesselJ[1, u]/u)^2
];

(* ::Subsection:: *)
(*Interference Analysis*)

AnalyzeInterference[wavelength_, sourceSeparation_, screenDistance_, opts___] := Module[{
    options, patternType, coherenceLength, numPoints, screenSize, x, intensity, 
    pathDifference, phaseDifference, fringeSpacing, visibility, result
},
    options = Association[opts];
    patternType = Lookup[options, "PatternType", "double_slit"];
    coherenceLength = Lookup[options, "CoherenceLength", None];
    numPoints = Lookup[options, "NumPoints", 1000];
    
    (* Screen size *)
    screenSize = 10*wavelength*screenDistance/sourceSeparation;
    x = Table[i, {i, -screenSize/2, screenSize/2, screenSize/(numPoints-1)}];
    
    intensity = Switch[patternType,
        "double_slit" | "young",
        With[{
            theta = x/screenDistance,
            pathDiff = sourceSeparation*Sin[x/screenDistance],
            phaseDiff = 2*Pi*pathDiff/wavelength
        },
            If[coherenceLength === None,
                4*Cos[phaseDiff/2]^2,
                With[{
                    coherenceFactor = Exp[-Abs[pathDiff]/coherenceLength],
                    visibility = coherenceFactor
                },
                    2*(1 + visibility*Cos[phaseDiff])
                ]
            ]
        ],
        
        "michelson",
        With[{phaseDiff = 4*Pi*sourceSeparation/wavelength},
            2*(1 + Cos[phaseDiff])
        ],
        
        _,
        Message[AnalyzeInterference::unknownpattern, patternType];
        $Failed
    ];
    
    If[intensity === $Failed, Return[$Failed]];
    
    (* Calculate fringe parameters *)
    fringeSpacing = wavelength*screenDistance/sourceSeparation;
    
    (* Visibility calculation *)
    With[{iMax = Max[intensity], iMin = Min[intensity]},
        visibility = (iMax - iMin)/(iMax + iMin)
    ];
    
    result = <|
        "Position" -> x,
        "Intensity" -> intensity,
        "IntensityNormalized" -> intensity/Max[intensity],
        "FringeSpacing" -> fringeSpacing,
        "Visibility" -> visibility,
        "PatternType" -> patternType,
        "Wavelength" -> wavelength,
        "SourceSeparation" -> sourceSeparation,
        "ScreenDistance" -> screenDistance
    |>;
    
    result
];

AnalyzeInterference::unknownpattern = "Unknown interference pattern type: `1`";

(* ::Subsection:: *)
(*Utility Functions*)

FresnelNumber[apertureRadius_, wavelength_, distance_] := 
    apertureRadius^2/(wavelength*distance);

RayleighRange[wavelength_, waistRadius_, mediumIndex_: 1.0] := 
    Pi*waistRadius^2/(wavelength/mediumIndex);

(* ::Subsection:: *)
(*Visualization Functions*)

PlotBeamPropagation[beam_Association, zRange_List, opts___] := Module[{
    options, numPoints, zPositions, beamRadii, plot
},
    options = Association[opts];
    numPoints = Lookup[options, "NumPoints", 200];
    
    zPositions = Table[z, {z, zRange[[1]], zRange[[2]], (zRange[[2]] - zRange[[1]])/(numPoints-1)}];
    beamRadii = beam["BeamRadius"] /@ zPositions;
    
    plot = Plot[{
        Interpolation[Transpose[{zPositions, beamRadii}]][z],
        -Interpolation[Transpose[{zPositions, beamRadii}]][z]
    }, {z, zRange[[1]], zRange[[2]]},
        PlotStyle -> {Directive[Thick, $BerkeleyBlue], Directive[Thick, $BerkeleyBlue]},
        PlotRange -> All,
        Frame -> True,
        FrameLabel -> {"Position (m)", "Beam Radius (m)"},
        PlotLabel -> Style["Gaussian Beam Propagation", 16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]],
        opts
    ];
    
    (* Add waist and Rayleigh range markers *)
    Show[plot,
        Graphics[{
            $CaliforniaGold, PointSize[0.01],
            Point[{beam["WaistPosition"], beam["WaistRadius"]}],
            Point[{beam["WaistPosition"], -beam["WaistRadius"]}],
            Dashed, $BerkeleyLightBlue,
            Line[{{beam["WaistPosition"] - beam["RayleighRange"], -1}, 
                  {beam["WaistPosition"] - beam["RayleighRange"], 1}}],
            Line[{{beam["WaistPosition"] + beam["RayleighRange"], -1}, 
                  {beam["WaistPosition"] + beam["RayleighRange"], 1}}]
        }]
    ]
];

PlotInterferencePattern[data_Association, opts___] := Module[{plot},
    plot = ListLinePlot[
        Transpose[{data["Position"]*1000, data["IntensityNormalized"]}],
        PlotStyle -> Directive[Thick, $BerkeleyBlue],
        Filling -> Axis,
        FillingStyle -> Directive[$BerkeleyBlue, Opacity[0.3]],
        Frame -> True,
        FrameLabel -> {"Position (mm)", "Normalized Intensity"},
        PlotLabel -> Style["Interference Pattern", 16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]],
        opts
    ];
    
    plot
];

PlotDiffractionPattern[data_Association, opts___] := Module[{plot1, plot2},
    plot1 = ListLinePlot[
        Transpose[{data["Position"]*1000, data["IntensityNormalized"]}],
        PlotStyle -> Directive[Thick, $BerkeleyBlue],
        Filling -> Axis,
        FillingStyle -> Directive[$BerkeleyBlue, Opacity[0.3]],
        Frame -> True,
        FrameLabel -> {"Position (mm)", "Intensity"},
        PlotLabel -> Style["Diffraction Pattern", 16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]]
    ];
    
    plot2 = ListLogLinearPlot[
        Transpose[{data["Position"]*1000, data["IntensityNormalized"]}],
        PlotStyle -> Directive[Thick, $CaliforniaGold],
        Frame -> True,
        FrameLabel -> {"Position (mm)", "Intensity (log scale)"},
        PlotLabel -> Style["Diffraction Pattern (Log Scale)", 16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]]
    ];
    
    GraphicsGrid[{{plot1}, {plot2}}, ImageSize -> Large, opts]
];

(* ::Subsection:: *)
(*Demo Function*)

WaveOpticsDemo[] := Module[{wavelength, beam, zRange, interference, diffraction},
    Print["Wave Optics Demo"];
    Print["================"];
    Print[];
    
    (* Parameters *)
    wavelength = 633*^-9; (* HeNe laser *)
    
    (* 1. Gaussian beam propagation *)
    Print["1. Gaussian Beam Propagation"];
    beam = GaussianBeam[wavelength, 1*^-3, "Power" -> 1*^-3];
    
    Print["Rayleigh range: ", beam["RayleighRange"]*1000, " mm"];
    Print["Divergence angle: ", beam["DivergenceAngle"]*1000, " mrad"];
    
    zRange = {-5*^-3, 5*^-3};
    PlotBeamPropagation[beam, zRange]
    
    (* 2. Double slit interference *)
    Print[];
    Print["2. Double Slit Interference"];
    interference = AnalyzeInterference[wavelength, 100*^-6, 1.0];
    
    Print["Fringe spacing: ", interference["FringeSpacing"]*1000, " mm"];
    Print["Visibility: ", interference["Visibility"]];
    
    PlotInterferencePattern[interference]
    
    (* 3. Single slit diffraction *)
    Print[];
    Print["3. Single Slit Diffraction"];
    diffraction = CalculateDiffraction["single_slit", 50*^-6, wavelength, 2.0, 0.02];
    
    PlotDiffractionPattern[diffraction]
    
    Print[];
    Print["Demo completed!"];
];

End[] (* End Private Context *)

EndPackage[]