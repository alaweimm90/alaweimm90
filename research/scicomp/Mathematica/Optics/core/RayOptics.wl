(* ::Package:: *)

(* ::Title:: *)
(*Ray Optics Package - Mathematica Implementation*)

(* ::Subtitle:: *)
(*Berkeley SciComp Framework*)

(* ::Author:: *)
(*Berkeley SciComp Team*)

(* ::Date:: *)
(*2024*)

BeginPackage["BerkeleySciComp`Optics`RayOptics`"]

(* ::Section:: *)
(*Usage Messages*)

RayOptics::usage = "RayOptics[opts] creates a ray tracing system.";
Ray::usage = "Ray[position, direction, wavelength, opts] creates an optical ray.";
ThinLens::usage = "ThinLens[position, focalLength, diameter, opts] creates a thin lens.";
SphericalSurface::usage = "SphericalSurface[position, radius, diameter, opts] creates a spherical surface.";
Mirror::usage = "Mirror[position, radius, diameter, opts] creates a mirror.";
TraceRay::usage = "TraceRay[system, ray] traces a ray through an optical system.";
TraceRayBundle::usage = "TraceRayBundle[system, rays] traces multiple rays through a system.";
CalculateSpotDiagram::usage = "CalculateSpotDiagram[system, objectHeight, screenDistance, opts] calculates spot diagram.";
PlotRayDiagram::usage = "PlotRayDiagram[system, rays, opts] plots ray diagram.";

(* ::Section:: *)
(*Physical Constants*)

$SpeedOfLight = 2.99792458*^8; (* m/s *)

(* Berkeley Color Scheme *)
$BerkeleyBlue = RGBColor[0, 50/255, 98/255];
$CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
$BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

(* ::Section:: *)
(*Begin Private Context*)

Begin["`Private`"]

(* ::Subsection:: *)
(*Ray Class*)

Ray[position_List, direction_List, wavelength_, opts___] := Module[{
    options, intensity, mediumIndex, opticalPath, obj
},
    options = Association[opts];
    intensity = Lookup[options, "Intensity", 1.0];
    mediumIndex = Lookup[options, "MediumIndex", 1.0];
    opticalPath = Lookup[options, "OpticalPath", 0];
    
    obj = <|
        "Type" -> "Ray",
        "Position" -> position,
        "Direction" -> Normalize[direction],
        "Wavelength" -> wavelength,
        "Intensity" -> intensity,
        "OpticalPath" -> opticalPath,
        "MediumIndex" -> mediumIndex
    |>;
    
    obj["PropagateDistance"] = Function[{distance},
        Ray[
            obj["Position"] + distance*obj["Direction"],
            obj["Direction"],
            obj["Wavelength"],
            "Intensity" -> obj["Intensity"],
            "MediumIndex" -> obj["MediumIndex"],
            "OpticalPath" -> obj["OpticalPath"] + distance*obj["MediumIndex"]
        ]
    ];
    
    obj["PropagateToZ"] = Function[{zPosition},
        With[{dz = obj["Direction"][[3]]},
            If[Abs[dz] < 10^-12,
                $Failed, (* Ray parallel to z-plane *)
                With[{distance = (zPosition - obj["Position"][[3]])/dz},
                    If[distance < 10^-12,
                        $Failed, (* Target behind ray *)
                        obj["PropagateDistance"][distance]
                    ]
                ]
            ]
        ]
    ];
    
    obj["GetAngleWithAxis"] = Function[{axis},
        With[{axisVector = UnitVector[3, axis]},
            ArcCos[Dot[obj["Direction"], axisVector]]
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Optical Surface Base Class*)

OpticalSurface[position_, diameter_, opts___] := Module[{
    options, materialBefore, materialAfter, obj
},
    options = Association[opts];
    materialBefore = Lookup[options, "MaterialBefore", "air"];
    materialAfter = Lookup[options, "MaterialAfter", "air"];
    
    obj = <|
        "Position" -> position,
        "Diameter" -> diameter,
        "MaterialBefore" -> materialBefore,
        "MaterialAfter" -> materialAfter
    |>;
    
    obj["CheckAperture"] = Function[{point},
        With[{r = Sqrt[point[[1]]^2 + point[[2]]^2]},
            r <= obj["Diameter"]/2
        ]
    ];
    
    obj["GetRefractiveIndex"] = Function[{material, wavelength},
        Switch[material,
            "air", 1.000293,
            "vacuum", 1.0,
            "water", 1.333,
            "bk7", (* Simplified BK7 dispersion *)
            With[{lambdaUm = wavelength*10^6},
                Sqrt[1 + 1.03961212*lambdaUm^2/(lambdaUm^2 - 0.00600069867) + 
                      0.231792344*lambdaUm^2/(lambdaUm^2 - 0.0200179144) + 
                      1.01046945*lambdaUm^2/(lambdaUm^2 - 103.560653)]
            ],
            "silica", (* Simplified fused silica *)
            With[{lambdaUm = wavelength*10^6},
                Sqrt[1 + 0.6961663*lambdaUm^2/(lambdaUm^2 - 0.00467914826) + 
                      0.4079426*lambdaUm^2/(lambdaUm^2 - 0.0135120631) + 
                      0.8974794*lambdaUm^2/(lambdaUm^2 - 97.9340025)]
            ],
            _, 1.5 (* Default glass *)
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Thin Lens*)

ThinLens[position_, focalLength_, diameter_, opts___] := Module[{surface, obj},
    surface = OpticalSurface[position, diameter, opts];
    
    obj = Join[surface, <|
        "Type" -> "ThinLens",
        "FocalLength" -> focalLength
    |>];
    
    obj["FindIntersection"] = Function[{ray},
        With[{dz = ray["Direction"][[3]]},
            If[Abs[dz] < 10^-12,
                <|"Hit" -> False, "IntersectionPoint" -> None|>,
                With[{
                    t = (obj["Position"] - ray["Position"][[3]])/dz,
                    intersectionPoint = ray["Position"] + t*ray["Direction"]
                },
                    If[t < 10^-12,
                        <|"Hit" -> False, "IntersectionPoint" -> None|>,
                        <|"Hit" -> obj["CheckAperture"][intersectionPoint], 
                          "IntersectionPoint" -> intersectionPoint|>
                    ]
                ]
            ]
        ]
    ];
    
    obj["RefractRay"] = Function[{ray, intersectionPoint, wavelength},
        With[{
            h = Sqrt[intersectionPoint[[1]]^2 + intersectionPoint[[2]]^2],
            incidentDir = ray["Direction"],
            lensPower = 1/obj["FocalLength"]
        },
            With[{
                deflectionAngle = If[h > 10^-12, -h*lensPower, 0],
                radialUnit = If[h > 10^-12, 
                    {intersectionPoint[[1]], intersectionPoint[[2]], 0}/h, 
                    {0, 0, 0}
                ]
            },
                With[{newDirection = Normalize[incidentDir + deflectionAngle*radialUnit]},
                    Ray[intersectionPoint, newDirection, wavelength, 
                        "Intensity" -> ray["Intensity"], 
                        "MediumIndex" -> ray["MediumIndex"]]
                ]
            ]
        ]
    ];
    
    obj["GetABCDMatrix"] = Function[{},
        {{1, 0}, {-1/obj["FocalLength"], 1}}
    ];
    
    obj["CalculateImagePosition"] = Function[{objectPos},
        With[{objectDistance = obj["Position"] - objectPos},
            If[Abs[objectDistance] < 10^-12,
                Infinity,
                With[{imageDistance = 1/(1/obj["FocalLength"] - 1/objectDistance)},
                    obj["Position"] + imageDistance
                ]
            ]
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Spherical Surface*)

SphericalSurface[position_, radius_, diameter_, opts___] := Module[{surface, obj},
    surface = OpticalSurface[position, diameter, opts];
    
    obj = Join[surface, <|
        "Type" -> "SphericalSurface",
        "Radius" -> radius
    |>];
    
    obj["FindIntersection"] = Function[{ray},
        (* Simplified intersection for spherical surface at z=position *)
        With[{dz = ray["Direction"][[3]]},
            If[Abs[dz] < 10^-12,
                <|"Hit" -> False, "IntersectionPoint" -> None|>,
                With[{
                    t = (obj["Position"] - ray["Position"][[3]])/dz,
                    intersectionPoint = ray["Position"] + t*ray["Direction"]
                },
                    If[t < 10^-12,
                        <|"Hit" -> False, "IntersectionPoint" -> None|>,
                        <|"Hit" -> obj["CheckAperture"][intersectionPoint], 
                          "IntersectionPoint" -> intersectionPoint|>
                    ]
                ]
            ]
        ]
    ];
    
    obj["SurfaceNormal"] = Function[{point},
        {0, 0, 1} (* Simplified - always pointing in +z direction *)
    ];
    
    obj["RefractRay"] = Function[{ray, intersectionPoint, wavelength},
        (* Simplified refraction using Snell's law *)
        With[{
            n1 = obj["GetRefractiveIndex"][obj["MaterialBefore"], wavelength],
            n2 = obj["GetRefractiveIndex"][obj["MaterialAfter"], wavelength],
            normal = obj["SurfaceNormal"][intersectionPoint],
            incidentDir = ray["Direction"]
        },
            With[{
                cosTheta1 = -Dot[incidentDir, normal],
                discriminant = 1 - (n1/n2)^2*(1 - cosTheta1^2)
            },
                If[discriminant < 0,
                    (* Total internal reflection *)
                    With[{newDirection = incidentDir - 2*Dot[incidentDir, normal]*normal},
                        Ray[intersectionPoint, newDirection, wavelength, 
                            "Intensity" -> ray["Intensity"], 
                            "MediumIndex" -> ray["MediumIndex"]]
                    ],
                    (* Refraction *)
                    With[{
                        cosTheta2 = Sqrt[discriminant],
                        newDirection = (n1/n2)*incidentDir + ((n1/n2)*cosTheta1 - cosTheta2)*normal
                    },
                        Ray[intersectionPoint, Normalize[newDirection], wavelength, 
                            "Intensity" -> ray["Intensity"], 
                            "MediumIndex" -> n2]
                    ]
                ]
            ]
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Mirror*)

Mirror[position_, radius_, diameter_, opts___] := Module[{surface, obj},
    surface = OpticalSurface[position, diameter, opts];
    
    obj = Join[surface, <|
        "Type" -> "Mirror",
        "Radius" -> radius
    |>];
    
    obj["FindIntersection"] = Function[{ray},
        With[{dz = ray["Direction"][[3]]},
            If[Abs[dz] < 10^-12,
                <|"Hit" -> False, "IntersectionPoint" -> None|>,
                With[{
                    t = (obj["Position"] - ray["Position"][[3]])/dz,
                    intersectionPoint = ray["Position"] + t*ray["Direction"]
                },
                    If[t < 10^-12,
                        <|"Hit" -> False, "IntersectionPoint" -> None|>,
                        <|"Hit" -> obj["CheckAperture"][intersectionPoint], 
                          "IntersectionPoint" -> intersectionPoint|>
                    ]
                ]
            ]
        ]
    ];
    
    obj["ReflectRay"] = Function[{ray, intersectionPoint},
        With[{
            normal = {0, 0, 1}, (* Simplified normal *)
            incidentDir = ray["Direction"]
        },
            With[{reflectedDir = incidentDir - 2*Dot[incidentDir, normal]*normal},
                Ray[intersectionPoint, Normalize[reflectedDir], ray["Wavelength"], 
                    "Intensity" -> ray["Intensity"], 
                    "MediumIndex" -> ray["MediumIndex"]]
            ]
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Ray Optics System*)

RayOptics[opts___] := Module[{options, mediumIndex, wavelength, obj},
    options = Association[opts];
    mediumIndex = Lookup[options, "MediumIndex", 1.0];
    wavelength = Lookup[options, "Wavelength", 589*^-9];
    
    obj = <|
        "Type" -> "RayOptics",
        "Surfaces" -> {},
        "MediumIndex" -> mediumIndex,
        "Wavelength" -> wavelength
    |>;
    
    obj["AddSurface"] = Function[{surface},
        obj["Surfaces"] = Append[obj["Surfaces"], surface];
        (* Sort surfaces by position *)
        obj["Surfaces"] = SortBy[obj["Surfaces"], #["Position"]&];
    ];
    
    obj["TraceRay"] = Function[{ray},
        Module[{currentRay, rayPath, success, errorMessage},
            currentRay = ray;
            rayPath = {currentRay};
            success = True;
            errorMessage = "";
            
            Do[
                With[{surface = obj["Surfaces"][[i]]},
                    With[{intersection = surface["FindIntersection"][currentRay]},
                        If[intersection["Hit"],
                            With[{intersectionPoint = intersection["IntersectionPoint"]},
                                (* Add intersection ray *)
                                With[{intersectionRay = Ray[intersectionPoint, currentRay["Direction"], 
                                    currentRay["Wavelength"], 
                                    "Intensity" -> currentRay["Intensity"],
                                    "MediumIndex" -> currentRai["MediumIndex"]]},
                                    AppendTo[rayPath, intersectionRay];
                                    
                                    (* Refract/reflect ray *)
                                    currentRay = Switch[surface["Type"],
                                        "ThinLens",
                                        surface["RefractRay"][currentRay, intersectionPoint, currentRay["Wavelength"]],
                                        
                                        "SphericalSurface",
                                        surface["RefractRay"][currentRay, intersectionPoint, currentRay["Wavelength"]],
                                        
                                        "Mirror",
                                        surface["ReflectRay"][currentRay, intersectionPoint],
                                        
                                        _,
                                        (success = False; errorMessage = "Unknown surface type"; Break[])
                                    ];
                                    
                                    AppendTo[rayPath, currentRay];
                                ]
                            ]
                        ]
                    ]
                ],
                {i, Length[obj["Surfaces"]]}
            ];
            
            <|
                "Rays" -> rayPath,
                "Success" -> success,
                "ErrorMessage" -> errorMessage,
                "FinalRay" -> currentRay
            |>
        ]
    ];
    
    obj
];

(* ::Subsection:: *)
(*Ray Tracing Functions*)

TraceRay[system_Association, ray_Association] := system["TraceRay"][ray];

TraceRayBundle[system_Association, rays_List] := 
    TraceRay[system, #]& /@ rays;

CalculateSpotDiagram[system_Association, objectHeight_, screenDistance_, opts___] := Module[{
    options, numRays, maxAperture, rays, results, hitPositions, spotDiagram
},
    options = Association[opts];
    numRays = Lookup[options, "NumRays", 100];
    maxAperture = Lookup[options, "MaxAperture", 0.01]; (* 1 cm default *)
    
    (* Generate ray bundle *)
    rays = Table[
        With[{
            angle = 2*Pi*i/numRays,
            radius = Sqrt[RandomReal[]]*maxAperture
        },
            Ray[
                {radius*Cos[angle], radius*Sin[angle], -Abs[screenDistance]/2},
                {0, 0, 1}, (* Parallel rays *)
                system["Wavelength"]
            ]
        ],
        {i, numRays}
    ];
    
    (* Trace rays *)
    results = TraceRayBundle[system, rays];
    
    (* Extract final ray positions *)
    hitPositions = {};
    Do[
        If[results[[i]]["Success"] && Length[results[[i]]["Rays"]] > 0,
            With[{finalRay = results[[i]]["FinalRay"]},
                With[{tToScreen = (screenDistance - finalRay["Position"][[3]])/finalRay["Direction"][[3]]},
                    If[tToScreen > 0,
                        With[{screenPos = finalRay["Position"] + tToScreen*finalRay["Direction"]},
                            AppendTo[hitPositions, screenPos[[{1, 2}]]]
                        ]
                    ]
                ]
            ]
        ],
        {i, Length[results]}
    ];
    
    spotDiagram = <|
        "Positions" -> hitPositions,
        "ScreenDistance" -> screenDistance,
        "ObjectHeight" -> objectHeight,
        "NumRays" -> Length[hitPositions]
    |>;
    
    If[Length[hitPositions] > 0,
        spotDiagram["RMSRadius"] = Sqrt[Mean[#.#& /@ hitPositions]];
        spotDiagram["Centroid"] = Mean[hitPositions],
        spotDiagram["RMSRadius"] = 0;
        spotDiagram["Centroid"] = {0, 0}
    ];
    
    spotDiagram
];

(* ::Subsection:: *)
(*Visualization Functions*)

PlotRayDiagram[system_Association, rays_List, opts___] := Module[{
    options, zRange, yRange, showSurfaces, results, rayPaths, surfacePlots, plot
},
    options = Association[opts];
    zRange = Lookup[options, "ZRange", Automatic];
    yRange = Lookup[options, "YRange", Automatic];
    showSurfaces = Lookup[options, "ShowSurfaces", True];
    
    (* Trace rays *)
    results = TraceRayBundle[system, rays];
    
    (* Extract ray paths *)
    rayPaths = {};
    Do[
        If[results[[i]]["Success"],
            With[{rayPath = results[[i]]["Rays"]},
                With[{
                    zPositions = #["Position"][[3]]& /@ rayPath,
                    xPositions = #["Position"][[1]]& /@ rayPath
                },
                    AppendTo[rayPaths, Transpose[{zPositions, xPositions}]]
                ]
            ]
        ],
        {i, Length[results]}
    ];
    
    (* Create ray plot *)
    plot = ListLinePlot[rayPaths,
        PlotStyle -> Table[Directive[Thick, Hue[i/Length[rayPaths]]], {i, Length[rayPaths]}],
        PlotRange -> All,
        Frame -> True,
        FrameLabel -> {"Optical Axis (m)", "Height (m)"},
        PlotLabel -> Style["Ray Diagram", 16, Bold, $BerkeleyBlue],
        GridLines -> Automatic,
        GridLinesStyle -> Directive[Gray, Opacity[0.3]]
    ];
    
    (* Add optical surfaces if requested *)
    If[showSurfaces && Length[system["Surfaces"]] > 0,
        surfacePlots = Table[
            With[{surface = system["Surfaces"][[i]]},
                With[{
                    zPos = surface["Position"],
                    height = surface["Diameter"]/2
                },
                    Graphics[{
                        Switch[surface["Type"],
                            "ThinLens", $CaliforniaGold,
                            "Mirror", Gray,
                            _, $BerkeleyBlue
                        ],
                        Thick,
                        Line[{{zPos, -height}, {zPos, height}}]
                    }]
                ]
            ],
            {i, Length[system["Surfaces"]]}
        ];
        
        Show[plot, surfacePlots],
        plot
    ]
];

PlotSpotDiagram[spotData_Association, opts___] := Module[{plot},
    If[Length[spotData["Positions"]] > 0,
        plot = ListPlot[spotData["Positions"]*1000,
            PlotStyle -> Directive[PointSize[0.01], $BerkeleyBlue, Opacity[0.6]],
            AspectRatio -> 1,
            Frame -> True,
            FrameLabel -> {"X Position (mm)", "Y Position (mm)"},
            PlotLabel -> Style["Spot Diagram", 16, Bold, $BerkeleyBlue],
            GridLines -> Automatic,
            GridLinesStyle -> Directive[Gray, Opacity[0.3]],
            opts
        ];
        
        (* Add RMS circle *)
        With[{
            rmsRadius = spotData["RMSRadius"]*1000,
            centroid = spotData["Centroid"]*1000
        },
            Show[plot,
                Graphics[{
                    Dashed, $CaliforniaGold, Thick,
                    Circle[centroid, rmsRadius]
                }]
            ]
        ],
        
        Graphics[{
            Text[Style["No rays hit screen", 16, Bold, Red], {0, 0}]
        }]
    ]
];

(* ::Subsection:: *)
(*Demo Function*)

RayOpticsDemo[] := Module[{system, lens, rays, results, spotData},
    Print["Ray Optics Demo"];
    Print["==============="];
    Print[];
    
    (* Create optical system *)
    system = RayOptics["Wavelength" -> 589*^-9];
    
    (* Add thin lens *)
    lens = ThinLens[0, 0.1, 0.025]; (* f=100mm, D=25mm at z=0 *)
    system["AddSurface"][lens];
    
    Print["Optical system created:"];
    Print["Lens: f = ", lens["FocalLength"]*1000, " mm, D = ", lens["Diameter"]*1000, " mm"];
    Print[];
    
    (* Create ray bundle *)
    With[{
        objectDistance = 0.15, (* 150 mm *)
        objectHeight = 0.005,  (* 5 mm *)
        rayHeights = Table[h, {h, -objectHeight, objectHeight, 2*objectHeight/10}]
    },
        rays = Table[
            Ray[{h, 0, -objectDistance}, {0, 0, 1}, 589*^-9],
            {h, rayHeights}
        ];
        
        Print["Ray bundle created: ", Length[rays], " rays"];
        Print["Object distance: ", objectDistance*1000, " mm"];
        Print["Object height: ±", objectHeight*1000, " mm"];
        Print[];
        
        (* Calculate image distance *)
        With[{imageDistance = lens["CalculateImagePosition"][-objectDistance]},
            Print["Paraxial calculation:"];
            Print["Image distance: ", imageDistance*1000, " mm"];
            Print["Magnification: ", -imageDistance/objectDistance, "x"];
            Print[];
            
            (* Plot ray diagram *)
            Print["Plotting ray diagram..."];
            PlotRayDiagram[system, rays]
            
            (* Calculate spot diagram *)
            Print["Calculating spot diagram..."];
            spotData = CalculateSpotDiagram[system, 0, imageDistance, "NumRays" -> 100];
            
            Print["Spot diagram results:"];
            Print["Number of rays hitting screen: ", spotData["NumRays"]];
            Print["RMS spot radius: ", spotData["RMSRadius"]*1000, " mm"];
            Print["Centroid position: (", spotData["Centroid"][[1]]*1000, ", ", 
                  spotData["Centroid"][[2]]*1000, ") mm"];
            
            (* Plot spot diagram *)
            PlotSpotDiagram[spotData]
        ]
    ];
    
    Print[];
    Print["Demo completed!"];
];

End[] (* End Private Context *)

EndPackage[]