(* ::Package:: *)

(* PDE Solvers Package for Mathematica
   Comprehensive PDE solving functionality with Berkeley SciComp standards
   
   Author: Berkeley SciComp Team
   Date: 2024
*)

BeginPackage["PDESolvers`"];

(* Public function declarations *)
HeatEquationSolve::usage = "HeatEquationSolve[domain, bc, alpha, source, opts] solves heat equation";
WaveEquationSolve::usage = "WaveEquationSolve[domain, bc, speed, initial, opts] solves wave equation";
PoissonSolve::usage = "PoissonSolve[domain, bc, source, opts] solves Poisson equation";
FiniteDifferenceMatrix::usage = "FiniteDifferenceMatrix[domain, bc, order] builds finite difference matrices";
PDEConvergenceStudy::usage = "PDEConvergenceStudy[solver, domain, bc, exact, gridSizes] performs grid convergence";
BerkeleyPDEPlot::usage = "BerkeleyPDEPlot[solution, opts] creates Berkeley-styled plots of PDE solutions";
PDEAnimation::usage = "PDEAnimation[solution, opts] creates animations of time-dependent PDE solutions";

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];

Begin["`Private`"];

(* Heat Equation Solver *)
HeatEquationSolve[domain_, bc_, alpha_, source_, opts___] := Module[
  {x, u, t, dx, dt, n, nt, A, b, initial, scheme, tspan, solType},
  
  (* Parse options *)
  scheme = "Scheme" /. {opts} /. "Scheme" -> "Implicit";
  tspan = "TimeSpan" /. {opts} /. "TimeSpan" -> {0, 1};
  dt = "TimeStep" /. {opts} /. "TimeStep" -> 0.01;
  initial = "InitialCondition" /. {opts} /. "InitialCondition" -> Automatic;
  solType = "SolutionType" /. {opts} /. "SolutionType" -> "Steady";
  
  (* Setup spatial domain *)
  x = domain;
  n = Length[x];
  dx = x[[2]] - x[[1]]; (* Assume uniform grid *)
  
  If[solType == "Steady",
    (* Steady-state heat equation: -alpha * d²u/dx² = f(x) *)
    SolveHeatSteady[x, bc, alpha, source]
  ,
    (* Transient heat equation: du/dt = alpha * d²u/dx² + f(x,t) *)
    SolveHeatTransient[x, bc, alpha, source, initial, tspan, dt, scheme]
  ]
];

(* Steady Heat Equation *)
SolveHeatSteady[x_, bc_, alpha_, source_] := Module[
  {n, dx, A, b, u, f},
  
  n = Length[x];
  dx = x[[2]] - x[[1]];
  
  (* Build finite difference matrix for -d²u/dx² *)
  A = SparseArray[{
    {i_, i_} /; 2 <= i <= n-1 -> 2,
    {i_, j_} /; 2 <= i <= n-1 && j == i-1 -> -1,
    {i_, j_} /; 2 <= i <= n-1 && j == i+1 -> -1
  }, {n, n}]/dx^2;
  
  (* Source term *)
  If[NumericQ[source],
    f = ConstantArray[source, n];
  ,
    f = source /@ x;
  ];
  b = f/alpha;
  
  (* Apply boundary conditions *)
  {A, b} = ApplyBoundaryConditions[A, b, bc, n];
  
  (* Solve linear system *)
  u = LinearSolve[A, b];
  
  <|
    "Type" -> "Steady Heat",
    "Domain" -> x,
    "Solution" -> u,
    "BoundaryConditions" -> bc,
    "ThermalDiffusivity" -> alpha,
    "Success" -> True
  |>
];

(* Transient Heat Equation *)
SolveHeatTransient[x_, bc_, alpha_, source_, initial_, {t0_, tf_}, dt_, scheme_] := Module[
  {n, dx, L, u0, u, t, nt, A, b, f, i, uNew},
  
  n = Length[x];
  dx = x[[2]] - x[[1]];
  nt = Ceiling[(tf - t0)/dt] + 1;
  t = Table[t0 + (i-1)*dt, {i, 1, nt}];
  
  (* Spatial operator: alpha * d²/dx² *)
  L = alpha * SparseArray[{
    {i_, i_} /; 2 <= i <= n-1 -> -2,
    {i_, j_} /; 2 <= i <= n-1 && j == i-1 -> 1,
    {i_, j_} /; 2 <= i <= n-1 && j == i+1 -> 1
  }, {n, n}]/dx^2;
  
  (* Initial condition *)
  If[initial === Automatic,
    u0 = ConstantArray[0, n];
  , NumericQ[initial],
    u0 = ConstantArray[initial, n];
  ,
    u0 = initial /@ x;
  ];
  
  (* Storage for solution *)
  u = {u0};
  
  (* Time stepping *)
  For[i = 2, i <= nt, i++,
    If[scheme == "Explicit",
      uNew = ExplicitTimeStep[Last[u], L, source, x, t[[i-1]], dt];
    ,
      uNew = ImplicitTimeStep[Last[u], L, source, x, t[[i]], dt, bc, n];
    ];
    AppendTo[u, uNew];
  ];
  
  <|
    "Type" -> "Transient Heat",
    "Domain" -> x,
    "Time" -> t,
    "Solution" -> u,
    "Scheme" -> scheme,
    "TimeStep" -> dt,
    "Success" -> True
  |>
];

(* Explicit time step *)
ExplicitTimeStep[u_, L_, source_, x_, t_, dt_] := Module[{f},
  If[source === 0,
    f = ConstantArray[0, Length[x]];
  , NumericQ[source],
    f = ConstantArray[source, Length[x]];
  ,
    f = source[x, t];
  ];
  u + dt*(L.u + f)
];

(* Implicit time step *)
ImplicitTimeStep[u_, L_, source_, x_, t_, dt_, bc_, n_] := Module[{A, b, f, I},
  I = SparseArray[{i_, i_} -> 1, {n, n}];
  
  If[source === 0,
    f = ConstantArray[0, n];
  , NumericQ[source],
    f = ConstantArray[source, n];
  ,
    f = source[x, t];
  ];
  
  A = I - dt*L;
  b = u + dt*f;
  
  (* Apply boundary conditions *)
  {A, b} = ApplyBoundaryConditions[A, b, bc, n];
  
  LinearSolve[A, b]
];

(* Poisson Equation Solver *)
PoissonSolve[domain_, bc_, source_, opts___] := Module[
  {x, n, dx, A, b, u, f},
  
  x = domain;
  n = Length[x];
  dx = x[[2]] - x[[1]];
  
  (* Build finite difference matrix for -d²u/dx² *)
  A = SparseArray[{
    {i_, i_} /; 2 <= i <= n-1 -> 2,
    {i_, j_} /; 2 <= i <= n-1 && j == i-1 -> -1,
    {i_, j_} /; 2 <= i <= n-1 && j == i+1 -> -1
  }, {n, n}]/dx^2;
  
  (* Source term *)
  If[NumericQ[source],
    f = ConstantArray[source, n];
  ,
    f = source /@ x;
  ];
  b = f;
  
  (* Apply boundary conditions *)
  {A, b} = ApplyBoundaryConditions[A, b, bc, n];
  
  (* Solve *)
  u = LinearSolve[A, b];
  
  <|
    "Type" -> "Poisson",
    "Domain" -> x,
    "Solution" -> u,
    "Source" -> source,
    "Success" -> True
  |>
];

(* Wave Equation Solver *)
WaveEquationSolve[domain_, bc_, speed_, initial_, opts___] := Module[
  {x, u, v, t, dx, dt, n, nt, tspan, c, CFL, u0, v0, uOld, uCur, uNew, i},
  
  (* Parse options *)
  tspan = "TimeSpan" /. {opts} /. "TimeSpan" -> {0, 1};
  dt = "TimeStep" /. {opts} /. "TimeStep" -> Automatic;
  
  x = domain;
  n = Length[x];
  dx = x[[2]] - x[[1]];
  c = speed;
  
  (* Automatic time step based on CFL condition *)
  If[dt === Automatic,
    dt = 0.8*dx/c; (* CFL condition: c*dt/dx <= 1 *)
  ];
  
  CFL = c*dt/dx;
  If[CFL > 1, Message[WaveEquationSolve::cfl, CFL]];
  
  {t0, tf} = tspan;
  nt = Ceiling[(tf - t0)/dt] + 1;
  t = Table[t0 + (i-1)*dt, {i, 1, nt}];
  
  (* Initial conditions *)
  {u0, v0} = initial;
  If[NumericQ[u0], u0 = ConstantArray[u0, n]];
  If[NumericQ[v0], v0 = ConstantArray[v0, n]];
  If[Head[u0] =!= List, u0 = u0 /@ x];
  If[Head[v0] =!= List, v0 = v0 /@ x];
  
  (* First time step using initial velocity *)
  uOld = u0;
  uCur = u0 + dt*v0 + (dt^2/2)*c^2*(SecondDerivative[u0, dx]);
  
  (* Apply boundary conditions *)
  uCur = ApplyWaveBoundaryConditions[uCur, bc, n];
  
  u = {uOld, uCur};
  
  (* Time stepping (leapfrog scheme) *)
  For[i = 3, i <= nt, i++,
    uNew = 2*uCur - uOld + (c*dt)^2*SecondDerivative[uCur, dx];
    uNew = ApplyWaveBoundaryConditions[uNew, bc, n];
    
    AppendTo[u, uNew];
    uOld = uCur;
    uCur = uNew;
  ];
  
  <|
    "Type" -> "Wave",
    "Domain" -> x,
    "Time" -> t,
    "Solution" -> u,
    "WaveSpeed" -> c,
    "CFL" -> CFL,
    "Success" -> True
  |>
];

(* Helper function for second derivative *)
SecondDerivative[u_, dx_] := Module[{n, d2u},
  n = Length[u];
  d2u = ConstantArray[0, n];
  
  Do[
    d2u[[i]] = (u[[i+1]] - 2*u[[i]] + u[[i-1]])/dx^2,
    {i, 2, n-1}
  ];
  
  d2u
];

(* Apply boundary conditions to matrix system *)
ApplyBoundaryConditions[A_, b_, bc_, n_] := Module[{Anew, bnew},
  Anew = A;
  bnew = b;
  
  (* Dirichlet boundary conditions *)
  If[KeyExistsQ[bc, "Dirichlet"],
    Do[
      {i, val} = entry;
      i = i + 1; (* Convert to 1-based indexing *)
      If[1 <= i <= n,
        Anew[[i, All]] = 0;
        Anew[[i, i]] = 1;
        bnew[[i]] = val;
      ],
      {entry, Normal[bc["Dirichlet"]]}
    ];
  ];
  
  (* Neumann boundary conditions *)
  If[KeyExistsQ[bc, "Neumann"],
    Do[
      {i, val} = entry;
      i = i + 1; (* Convert to 1-based indexing *)
      If[i == 1, (* Left boundary *)
        Anew[[1, 1]] = -1;
        Anew[[1, 2]] = 1;
        bnew[[1]] = val*(x[[2]] - x[[1]]);
      ];
      If[i == n, (* Right boundary *)
        Anew[[n, n-1]] = -1;
        Anew[[n, n]] = 1;
        bnew[[n]] = val*(x[[n]] - x[[n-1]]);
      ];
      ,
      {entry, Normal[bc["Neumann"]]}
    ];
  ];
  
  {Anew, bnew}
];

(* Apply boundary conditions for wave equation *)
ApplyWaveBoundaryConditions[u_, bc_, n_] := Module[{unew},
  unew = u;
  
  If[KeyExistsQ[bc, "Dirichlet"],
    Do[
      {i, val} = entry;
      i = i + 1; (* Convert to 1-based indexing *)
      If[1 <= i <= n, unew[[i]] = val];
      ,
      {entry, Normal[bc["Dirichlet"]]}
    ];
  ];
  
  unew
];

(* Grid convergence study *)
PDEConvergenceStudy[solver_, domain_, bc_, exact_, gridSizes_] := Module[
  {results, errors, rates, i, x, sol, uExact},
  
  results = Table[
    x = Subdivide[First[domain], Last[domain], n-1];
    sol = solver[x, bc];
    If[sol["Success"],
      uExact = exact /@ x;
      Max[Abs[sol["Solution"] - uExact]]
    ,
      Undefined
    ],
    {n, gridSizes}
  ];
  
  errors = results;
  
  (* Calculate convergence rates *)
  rates = Table[
    If[i > 1 && NumericQ[errors[[i]]] && NumericQ[errors[[i-1]]],
      h1 = (Last[domain] - First[domain])/(gridSizes[[i-1]] - 1);
      h2 = (Last[domain] - First[domain])/(gridSizes[[i]] - 1);
      Log[errors[[i-1]]/errors[[i]]]/Log[h1/h2]
    ,
      Undefined
    ],
    {i, 2, Length[gridSizes]}
  ];
  
  <|
    "GridSizes" -> gridSizes,
    "Errors" -> errors,
    "ConvergenceRates" -> rates
  |>
];

(* Berkeley-styled PDE plotting *)
BerkeleyPDEPlot[solution_, opts___] := Module[
  {x, u, t, plotType, colors},
  
  x = solution["Domain"];
  u = solution["Solution"];
  colors = {BerkeleyBlue, CaliforniaGold, BerkeleyLightBlue};
  
  If[KeyExistsQ[solution, "Time"],
    (* Time-dependent solution *)
    t = solution["Time"];
    plotType = "TimeDependent";
  ,
    (* Steady-state solution *)
    plotType = "Steady";
  ];
  
  Switch[plotType,
    "Steady",
    ListLinePlot[Transpose[{x, u}],
      PlotStyle -> BerkeleyBlue,
      Frame -> True,
      FrameStyle -> BerkeleyBlue,
      GridLines -> Automatic,
      GridLinesStyle -> Directive[Gray, Dashed, Opacity[0.3]],
      PlotLabel -> Style[solution["Type"] <> " Solution", 16, BerkeleyBlue, Bold],
      FrameLabel -> {Style["x", 14, BerkeleyBlue], Style["u(x)", 14, BerkeleyBlue]},
      ImageSize -> 600,
      opts
    ],
    
    "TimeDependent",
    (* Create surface plot *)
    ListPlot3D[
      Flatten[Table[{x[[i]], t[[j]], u[[j, i]]}, {j, Length[t]}, {i, Length[x]}], 1],
      ColorFunction -> (ColorData[{"TemperatureMap", {Min[u], Max[u]}}][#3] &),
      PlotLabel -> Style[solution["Type"] <> " Evolution", 16, BerkeleyBlue, Bold],
      AxesLabel -> {Style["x", 14], Style["t", 14], Style["u", 14]},
      ImageSize -> 700,
      opts
    ]
  ]
];

(* Animation for time-dependent solutions *)
PDEAnimation[solution_, opts___] := Module[
  {x, u, t, frames, colors},
  
  If[!KeyExistsQ[solution, "Time"],
    Message[PDEAnimation::notime];
    Return[$Failed];
  ];
  
  x = solution["Domain"];
  u = solution["Solution"];
  t = solution["Time"];
  
  frames = Table[
    ListLinePlot[Transpose[{x, u[[i]]}],
      PlotStyle -> BerkeleyBlue,
      PlotRange -> {All, {Min[u], Max[u]}},
      Frame -> True,
      FrameStyle -> BerkeleyBlue,
      GridLines -> Automatic,
      GridLinesStyle -> Directive[Gray, Dashed, Opacity[0.3]],
      PlotLabel -> Style["t = " <> ToString[NumberForm[t[[i]], 3]], 14, BerkeleyBlue],
      FrameLabel -> {Style["x", 12], Style["u(x,t)", 12]},
      ImageSize -> 500
    ],
    {i, 1, Length[t], Max[1, Floor[Length[t]/50]]} (* Sample frames *)
  ];
  
  ListAnimate[frames, opts]
];

(* Error messages *)
WaveEquationSolve::cfl = "CFL number `1` > 1. Solution may be unstable.";
PDEAnimation::notime = "Solution does not contain time-dependent data.";

End[];
EndPackage[];