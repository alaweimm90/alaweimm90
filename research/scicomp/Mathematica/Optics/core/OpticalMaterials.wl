(* ::Package:: *)

(* ::Title:: *)
(*Optical Materials Package - Mathematica Implementation*)

(* ::Subtitle:: *)
(*Berkeley SciComp Framework*)

(* ::Author:: *)
(*Berkeley SciComp Team*)

(* ::Date:: *)
(*2024*)

BeginPackage["BerkeleySciComp`Optics`OpticalMaterials`"]

(* ::Section:: *)
(*Usage Messages*)

OpticalMaterials::usage = "OpticalMaterials[] creates an optical materials database.";
RefractiveIndex::usage = "RefractiveIndex[material, wavelength, temperature] calculates refractive index.";
GroupIndex::usage = "GroupIndex[material, wavelength] calculates group index.";
GroupVelocityDispersion::usage = "GroupVelocityDispersion[material, wavelength] calculates GVD.";
ChromaticDispersion::usage = "ChromaticDispersion[material, wavelength] calculates chromatic dispersion.";
AbbeNumber::usage = "AbbeNumber[material] calculates Abbe number.";
SellmeierIndex::usage = "SellmeierIndex[wavelength, B, C] calculates refractive index using Sellmeier equation.";
CauchyIndex::usage = "CauchyIndex[wavelength, A, B, C] calculates refractive index using Cauchy equation.";
DispersionAnalysis::usage = "DispersionAnalysis[material, wavelengthRange, opts] performs dispersion analysis.";
PlotDispersion::usage = "PlotDispersion[material, wavelengthRange, opts] plots material dispersion.";
CompareMaterials::usage = "CompareMaterials[materials, wavelengthRange, opts] compares multiple materials.";

(* ::Section:: *)
(*Physical Constants*)

$SpeedOfLight = 2.99792458*^8; (* m/s *)
$PlanckConstant = 6.62607015*^-34; (* J⋅s *)

(* Berkeley Color Scheme *)
$BerkeleyBlue = RGBColor[0, 50/255, 98/255];
$CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
$BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

(* ::Section:: *)
(*Begin Private Context*)

Begin["`Private`"]

(* ::Subsection:: *)
(*Materials Database*)

$MaterialsDatabase = <|
    "BK7" -> <|
        "Name" -> "BK7",
        "DispersionType" -> "Sellmeier",
        "B" -> {1.03961212, 0.231792344, 1.01046945},
        "C" -> {6.00069867*^-3, 2.00179144*^-2, 103.560653},
        "WavelengthRange" -> {310*^-9, 2500*^-9},
        "AbbeNumber" -> 64.17,
        "Density" -> 2.51 (* g/cm³ *)
    |>,
    
    "SiO2" -> <|
        "Name" -> "Fused Silica",
        "DispersionType" -> "Sellmeier",
        "B" -> {0.6961663, 0.4079426, 0.8974794},
        "C" -> {4.67914826*^-3, 1.35120631*^-2, 97.9340025},
        "WavelengthRange" -> {210*^-9, 3700*^-9},
        "AbbeNumber" -> 67.8,
        "Density" -> 2.20
    |>,
    
    "Si" -> <|
        "Name" -> "Silicon",
        "DispersionType" -> "Sellmeier",
        "B" -> {10.6684293, 0.0030434748, 1.54133408},
        "C" -> {0.301516485, 1.13475115, 1104.0},
        "WavelengthRange" -> {1200*^-9, 14000*^-9},
        "Bandgap" -> 1.12, (* eV *)
        "Density" -> 2.33
    |>,
    
    "Al2O3" -> <|
        "Name" -> "Sapphire",
        "DispersionType" -> "Sellmeier",
        "B" -> {1.4313493, 0.65054713, 5.3414021},
        "C" -> {5.2799261*^-3, 1.42382647*^-2, 325.017834},
        "WavelengthRange" -> {150*^-9, 5500*^-9},
        "AbbeNumber" -> 72.2,
        "Density" -> 3.98
    |>,
    
    "H2O" -> <|
        "Name" -> "Water",
        "DispersionType" -> "Sellmeier",
        "B" -> {5.684027565*^-1, 1.726177391*^-1, 2.086189578*^-2},
        "C" -> {5.101829712*^-3, 1.821153936*^-2, 2.620722293*^-2},
        "WavelengthRange" -> {200*^-9, 200000*^-9},
        "Density" -> 1.0
    |>,
    
    "air" -> <|
        "Name" -> "Air",
        "DispersionType" -> "Cauchy",
        "A" -> 1.000293,
        "B" -> 0,
        "C" -> 0,
        "WavelengthRange" -> {200*^-9, 2000000*^-9},
        "Density" -> 1.225*^-3
    |>
|>;

(* ::Subsection:: *)
(*Optical Materials Class*)

OpticalMaterials[] := Module[{obj},
    obj = <|
        "MaterialsDatabase" -> $MaterialsDatabase
    |>;
    
    obj["ListMaterials"] = Function[{},
        Print["Available Materials:"];
        Print["==================="];
        Do[
            With[{name = Keys[$MaterialsDatabase][[i]], params = Values[$MaterialsDatabase][[i]]},
                Print[i, ". ", params["Name"], " (", name, ")"];
                Print["   Dispersion: ", params["DispersionType"]];
                If[KeyExistsQ[params, "WavelengthRange"],
                    Print["   Range: ", params["WavelengthRange"][[1]]*10^9, " - ", 
                          params["WavelengthRange"][[2]]*10^9, " nm"]
                ];
                If[KeyExistsQ[params, "AbbeNumber"],
                    Print["   Abbe number: ", params["AbbeNumber"]]
                ];
                Print[];
            ],
            {i, Length[$MaterialsDatabase]}
        ];
    ];
    
    obj
];

(* ::Subsection:: *)
(*Dispersion Models*)

SellmeierIndex[wavelength_, B_List, C_List] := Module[{lambdaUm, lambdaSq, nSquared},
    lambdaUm = wavelength*10^6; (* Convert to micrometers *)
    lambdaSq = lambdaUm^2;
    
    nSquared = 1 + Total[MapThread[#1*lambdaSq/(lambdaSq - #2)&, {B, C}]];
    Sqrt[nSquared]
];

CauchyIndex[wavelength_, A_, B_: 0, C_: 0] := Module[{lambdaUm},
    lambdaUm = wavelength*10^6; (* Convert to micrometers *)
    A + B/lambdaUm^2 + C/lambdaUm^4
];

(* ::Subsection:: *)
(*Refractive Index Calculation*)

RefractiveIndex[material_, wavelength_, temperature_: 293.15] := Module[{params, n},
    If[StringQ[material],
        If[KeyExistsQ[$MaterialsDatabase, material],
            params = $MaterialsDatabase[material];
            
            (* Check wavelength range *)
            If[KeyExistsQ[params, "WavelengthRange"],
                If[wavelength < params["WavelengthRange"][[1]] || 
                   wavelength > params["WavelengthRange"][[2]],
                    Message[RefractiveIndex::wavelengthrange, wavelength*10^9, params["Name"]]
                ]
            ];
            
            (* Calculate refractive index based on dispersion model *)
            Switch[params["DispersionType"],
                "Sellmeier",
                SellmeierIndex[wavelength, params["B"], params["C"]],
                
                "Cauchy",
                CauchyIndex[wavelength, params["A"], 
                    Lookup[params, "B", 0], Lookup[params, "C", 0]],
                
                "Constant",
                params["Value"],
                
                _,
                Message[RefractiveIndex::unknowndispersion, params["DispersionType"]];
                $Failed
            ],
            
            (* Default values for common materials *)
            Switch[ToLowerCase[material],
                "air", 1.000293,
                "vacuum", 1.0,
                "water", 1.333,
                "glass", 1.5,
                "diamond", 2.4,
                _, 
                Message[RefractiveIndex::unknownmaterial, material];
                $Failed
            ]
        ],
        
        (* Material is an Association *)
        Switch[material["DispersionType"],
            "Sellmeier",
            SellmeierIndex[wavelength, material["B"], material["C"]],
            
            "Cauchy",
            CauchyIndex[wavelength, material["A"], 
                Lookup[material, "B", 0], Lookup[material, "C", 0]],
            
            "Constant",
            material["Value"],
            
            _,
            Message[RefractiveIndex::unknowndispersion, material["DispersionType"]];
            $Failed
        ]
    ]
];

RefractiveIndex::wavelengthrange = "Wavelength `1` nm outside valid range for `2`";
RefractiveIndex::unknowndispersion = "Unknown dispersion model: `1`";
RefractiveIndex::unknownmaterial = "Unknown material: `1`";

(* ::Subsection:: *)
(*Group Index*)

GroupIndex[material_, wavelength_] := Module[{dlambda, nPlus, nMinus, dnDlambda, n},
    (* Numerical derivative *)
    dlambda = wavelength*10^-6; (* Small wavelength increment *)
    nPlus = RefractiveIndex[material, wavelength + dlambda];
    nMinus = RefractiveIndex[material, wavelength - dlambda];
    dnDlambda = (nPlus - nMinus)/(2*dlambda);
    
    n = RefractiveIndex[material, wavelength];
    n - wavelength*dnDlambda
];

(* ::Subsection:: *)
(*Group Velocity Dispersion*)

GroupVelocityDispersion[material_, wavelength_] := Module[{
    dlambda, ngPlus, ngCenter, ngMinus, d2ngDlambda2, lambdaUm, gvd
},
    (* Numerical second derivative *)
    dlambda = wavelength*10^-6;
    ngPlus = GroupIndex[material, wavelength + dlambda];
    ngCenter = GroupIndex[material, wavelength];
    ngMinus = GroupIndex[material, wavelength - dlambda];
    
    d2ngDlambda2 = (ngPlus - 2*ngCenter + ngMinus)/dlambda^2;
    
    (* Convert to standard units *)
    lambdaUm = wavelength*10^6;
    gvd = -(lambdaUm^3/(2*Pi*$SpeedOfLight))*d2ngDlambda2*10^21; (* ps²/km *)
    
    gvd
];

(* ::Subsection:: *)
(*Chromatic Dispersion*)

ChromaticDispersion[material_, wavelength_] := Module[{gvd, lambdaNm},
    gvd = GroupVelocityDispersion[material, wavelength];
    lambdaNm = wavelength*10^9;
    -2*Pi*$SpeedOfLight*gvd/lambdaNm^2*10^-6 (* ps/(nm⋅km) *)
];

(* ::Subsection:: *)
(*Abbe Number*)

AbbeNumber[material_] := Module[{lambdaD, lambdaF, lambdaC, nD, nF, nC},
    (* Standard wavelengths for Abbe number *)
    lambdaD = 589.3*^-9; (* Sodium D-line *)
    lambdaF = 486.1*^-9; (* Hydrogen F-line *)
    lambdaC = 656.3*^-9; (* Hydrogen C-line *)
    
    nD = RefractiveIndex[material, lambdaD];
    nF = RefractiveIndex[material, lambdaF];
    nC = RefractiveIndex[material, lambdaC];
    
    (nD - 1)/(nF - nC)
];

(* ::Subsection:: *)
(*Dispersion Analysis*)

DispersionAnalysis[material_, wavelengthRange_List, opts___] := Module[{
    options, numPoints, wavelengths, n, ng, gvd, zeroDispIdx, zeroDispWavelengths, 
    abbeNumber, analysis
},
    options = Association[opts];
    numPoints = Lookup[options, "NumPoints", 1000];
    
    wavelengths = Table[w, {w, wavelengthRange[[1]], wavelengthRange[[2]], 
        (wavelengthRange[[2]] - wavelengthRange[[1]])/(numPoints-1)}];
    
    (* Calculate dispersion properties *)
    n = RefractiveIndex[material, #]& /@ wavelengths;
    ng = GroupIndex[material, #]& /@ wavelengths;
    gvd = GroupVelocityDispersion[material, #]& /@ wavelengths;
    
    (* Find zero dispersion wavelengths *)
    zeroDispIdx = Position[Differences[Sign[gvd]], Except[0]][[All, 1]];
    zeroDispWavelengths = wavelengths[[zeroDispIdx]];
    
    (* Calculate Abbe number if in visible range *)
    abbeNumber = If[wavelengthRange[[1]] <= 656.3*^-9 && wavelengthRange[[2]] >= 486.1*^-9,
        AbbeNumber[material],
        None
    ];
    
    analysis = <|
        "Wavelengths" -> wavelengths,
        "RefractiveIndex" -> n,
        "GroupIndex" -> ng,
        "GVD" -> gvd,
        "ZeroDispersionWavelengths" -> zeroDispWavelengths,
        "AbbeNumber" -> abbeNumber,
        "WavelengthRange" -> wavelengthRange
    |>;
    
    analysis
];

(* ::Subsection:: *)
(*Visualization Functions*)

PlotDispersion[material_, wavelengthRange_List, opts___] := Module[{
    options, numPoints, showGVD, analysis, wavelengthsNm, plot1, plot2, plots
},
    options = Association[opts];
    numPoints = Lookup[options, "NumPoints", 1000];
    showGVD = Lookup[options, "ShowGVD", True];
    
    (* Get dispersion analysis *)
    analysis = DispersionAnalysis[material, wavelengthRange, "NumPoints" -> numPoints];
    wavelengthsNm = analysis["Wavelengths"]*10^9;
    
    (* Refractive index plot *)
    plot1 = ListLinePlot[{
        Transpose[{wavelengthsNm, analysis["RefractiveIndex"]}],
        Transpose[{wavelengthsNm, analysis["GroupIndex"]}]
    },
        PlotStyle -> {Directive[Thick, $BerkeleyBlue], Directive[Thick, $CaliforniaGold]},
        PlotLegends -> {"n", "ng"},
        Frame -> True,
        FrameLabel -> {"Wavelength (nm)", "Refractive Index"},
        PlotLabel -> Style[If[StringQ[material], material <> " - Refractive Index", "Refractive Index"], 
            16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]]
    ];
    
    If[showGVD,
        (* GVD plot *)
        plot2 = ListLinePlot[Transpose[{wavelengthsNm, analysis["GVD"]}],
            PlotStyle -> Directive[Thick, $BerkeleyBlue],
            Frame -> True,
            FrameLabel -> {"Wavelength (nm)", "GVD (ps²/km)"},
            PlotLabel -> Style[If[StringQ[material], material <> " - Group Velocity Dispersion", 
                "Group Velocity Dispersion"], 16, Bold, $BerkeleyBlue],
            GridLines -> Automatic,
            GridLinesStyle -> Directive[Gray, Opacity[0.3]]
        ];
        
        (* Mark zero dispersion wavelengths *)
        If[Length[analysis["ZeroDispersionWavelengths"]] > 0,
            plot2 = Show[plot2,
                Graphics[Table[
                    {Dashed, $CaliforniaGold, Thick,
                        Line[{{zdw*10^9, Min[analysis["GVD"]]}, {zdw*10^9, Max[analysis["GVD"]]}}]},
                    {zdw, analysis["ZeroDispersionWavelengths"]}
                ]]
            ]
        ];
        
        plots = GraphicsGrid[{{plot1}, {plot2}}, ImageSize -> Large],
        plots = plot1
    ];
    
    (* Add Abbe number if available *)
    If[analysis["AbbeNumber"] =!= None,
        plots = Labeled[plots, 
            Style["Abbe number: " <> ToString[NumberForm[analysis["AbbeNumber"], {4, 1}]], 
                12, Background -> White], Bottom]
    ];
    
    plots
];

CompareMaterials[materials_List, wavelengthRange_List, opts___] := Module[{
    options, property, wavelengths, values, plot, colors
},
    options = Association[opts];
    property = Lookup[options, "Property", "n"];
    
    wavelengths = Table[w, {w, wavelengthRange[[1]], wavelengthRange[[2]], 
        (wavelengthRange[[2]] - wavelengthRange[[1]])/999}];
    
    colors = {$BerkeleyBlue, $CaliforniaGold, $BerkeleyLightBlue, 
             RGBColor[0.8, 0.2, 0.2], RGBColor[0.2, 0.8, 0.2], RGBColor[0.8, 0.2, 0.8]};
    
    values = Table[
        Switch[property,
            "n", RefractiveIndex[materials[[i]], #]& /@ wavelengths,
            "ng", GroupIndex[materials[[i]], #]& /@ wavelengths,
            "gvd", GroupVelocityDispersion[materials[[i]], #]& /@ wavelengths
        ],
        {i, Length[materials]}
    ];
    
    plot = ListLinePlot[
        Table[Transpose[{wavelengths*10^9, values[[i]]}], {i, Length[materials]}],
        PlotStyle -> Table[Directive[Thick, colors[[Mod[i-1, Length[colors]]+1]]], {i, Length[materials]}],
        PlotLegends -> materials,
        Frame -> True,
        FrameLabel -> {"Wavelength (nm)", 
            Switch[property,
                "n", "Refractive Index",
                "ng", "Group Index", 
                "gvd", "GVD (ps²/km)"
            ]},
        PlotLabel -> Style[
            Switch[property,
                "n", "Refractive Index Comparison",
                "ng", "Group Index Comparison",
                "gvd", "Group Velocity Dispersion Comparison"
            ], 16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]],
        opts
    ];
    
    plot
];

(* ::Subsection:: *)
(*Demo Function*)

OpticalMaterialsDemo[] := Module[{materials, wavelength, materialNames, testWavelengths, lineNames},
    Print["Optical Materials Demo"];
    Print["====================="];
    Print[];
    
    (* Create materials database *)
    materials = OpticalMaterials[];
    
    (* List available materials *)
    materials["ListMaterials"][];
    
    (* Calculate refractive indices at common wavelengths *)
    Print["Refractive indices at 589.3 nm (sodium D-line):"];
    Print["================================================"];
    
    wavelength = 589.3*^-9;
    materialNames = {"BK7", "SiO2", "air", "H2O"};
    
    Do[
        With[{n = RefractiveIndex[mat, wavelength]},
            Print[mat, ": n = ", NumberForm[n, {4, 6}]]
        ],
        {mat, materialNames}
    ];
    
    (* Analyze BK7 dispersion *)
    Print[];
    Print["BK7 Glass Analysis:"];
    Print["=================="];
    
    With[{
        wlRange = {400*^-9, 800*^-9}, (* Visible range *)
        analysis = DispersionAnalysis["BK7", {400*^-9, 800*^-9}]
    },
        Print["Wavelength range: ", wlRange[[1]]*10^9, " - ", wlRange[[2]]*10^9, " nm"];
        If[analysis["AbbeNumber"] =!= None,
            Print["Abbe number: ", NumberForm[analysis["AbbeNumber"], {4, 1}]]
        ];
        
        (* Find refractive index at specific wavelengths *)
        testWavelengths = {486.1*^-9, 589.3*^-9, 656.3*^-9}; (* F, d, C lines *)
        lineNames = {"F-line", "d-line", "C-line"};
        
        Do[
            With[{
                wl = testWavelengths[[i]],
                n = RefractiveIndex["BK7", testWavelengths[[i]]],
                ng = GroupIndex["BK7", testWavelengths[[i]]],
                gvd = GroupVelocityDispersion["BK7", testWavelengths[[i]]]
            },
                Print[lineNames[[i]], " (", NumberForm[wl*10^9, {4, 1}], " nm): n = ", 
                      NumberForm[n, {4, 6}], ", ng = ", NumberForm[ng, {4, 6}], 
                      ", GVD = ", NumberForm[gvd, {4, 2}], " ps²/km"]
            ],
            {i, Length[testWavelengths]}
        ];
        
        (* Plot dispersion curves *)
        Print[];
        Print["Plotting BK7 dispersion..."];
        PlotDispersion["BK7", {400*^-9, 1600*^-9}]
    ];
    
    (* Compare materials *)
    Print["Comparing materials..."];
    With[{compareMaterialsList = {"BK7", "SiO2", "H2O"}},
        CompareMaterials[compareMaterialsList, {400*^-9, 800*^-9}]
    ];
    
    Print[];
    Print["Demo completed!"];
];

End[] (* End Private Context *)

EndPackage[]