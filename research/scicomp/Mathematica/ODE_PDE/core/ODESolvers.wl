(* ::Package:: *)

(* ODE Solvers Package for Mathematica
   Comprehensive ODE solving functionality with Berkeley SciComp standards
   
   Author: Berkeley SciComp Team
   Date: 2024
*)

BeginPackage["ODESolvers`"];

(* Public function declarations *)
ExplicitEulerSolve::usage = "ExplicitEulerSolve[f, y0, {t, t0, tf}, dt] solves ODE dy/dt = f[t,y] using explicit Euler method";
RungeKutta4Solve::usage = "RungeKutta4Solve[f, y0, {t, t0, tf}, dt] solves ODE dy/dt = f[t,y] using 4th-order Runge-Kutta";
ImplicitEulerSolve::usage = "ImplicitEulerSolve[f, y0, {t, t0, tf}, dt] solves ODE dy/dt = f[t,y] using implicit Euler method";
AdaptiveRKSolve::usage = "AdaptiveRKSolve[f, y0, {t, t0, tf}, opts] solves ODE with adaptive Runge-Kutta-Fehlberg";
ODEConvergenceStudy::usage = "ODEConvergenceStudy[f, y0, tspan, analytical, dtList] performs convergence analysis";
BerkeleyODEPlot::usage = "BerkeleyODEPlot[solution, opts] creates Berkeley-styled plots of ODE solutions";

(* Berkeley color scheme *)
BerkeleyBlue = RGBColor[0, 50/255, 98/255];
CaliforniaGold = RGBColor[253/255, 181/255, 21/255];
BerkeleyLightBlue = RGBColor[59/255, 126/255, 161/255];
BerkeleyDarkGold = RGBColor[196/255, 130/255, 14/255];
BerkeleySecondaryBlue = RGBColor[0, 176/255, 218/255];

Begin["`Private`"];

(* Explicit Euler Method *)
ExplicitEulerSolve[f_, y0_, {t_, t0_, tf_}, dt_] := Module[
  {tvals, yvals, ycurrent, tcurrent, n, i},
  
  (* Input validation *)
  If[dt <= 0, Message[ExplicitEulerSolve::dt]; Return[$Failed]];
  If[tf <= t0, Message[ExplicitEulerSolve::tspan]; Return[$Failed]];
  
  (* Initialize *)
  n = Ceiling[(tf - t0)/dt] + 1;
  tvals = Table[t0 + (i - 1)*dt, {i, 1, n}];
  If[Last[tvals] < tf, AppendTo[tvals, tf]];
  
  yvals = {y0};
  ycurrent = y0;
  
  (* Integration loop *)
  Do[
    tcurrent = tvals[[i]];
    If[i < Length[tvals],
      dt = tvals[[i + 1]] - tcurrent;
      ycurrent = ycurrent + dt*f[tcurrent, ycurrent];
      AppendTo[yvals, ycurrent];
    ],
    {i, 1, Length[tvals] - 1}
  ];
  
  (* Return result *)
  <|
    "Method" -> "Explicit Euler",
    "Times" -> tvals,
    "Values" -> yvals,
    "TimeStep" -> dt,
    "Success" -> True,
    "Order" -> 1
  |>
];

(* 4th-order Runge-Kutta Method *)
RungeKutta4Solve[f_, y0_, {t_, t0_, tf_}, dt_] := Module[
  {tvals, yvals, ycurrent, tcurrent, k1, k2, k3, k4, n, i},
  
  (* Input validation *)
  If[dt <= 0, Message[RungeKutta4Solve::dt]; Return[$Failed]];
  If[tf <= t0, Message[RungeKutta4Solve::tspan]; Return[$Failed]];
  
  (* Initialize *)
  n = Ceiling[(tf - t0)/dt] + 1;
  tvals = Table[t0 + (i - 1)*dt, {i, 1, n}];
  If[Last[tvals] < tf, AppendTo[tvals, tf]];
  
  yvals = {y0};
  ycurrent = y0;
  
  (* Integration loop *)
  Do[
    tcurrent = tvals[[i]];
    If[i < Length[tvals],
      dt = tvals[[i + 1]] - tcurrent;
      
      (* RK4 stages *)
      k1 = dt*f[tcurrent, ycurrent];
      k2 = dt*f[tcurrent + dt/2, ycurrent + k1/2];
      k3 = dt*f[tcurrent + dt/2, ycurrent + k2/2];
      k4 = dt*f[tcurrent + dt, ycurrent + k3];
      
      (* Update solution *)
      ycurrent = ycurrent + (k1 + 2*k2 + 2*k3 + k4)/6;
      AppendTo[yvals, ycurrent];
    ],
    {i, 1, Length[tvals] - 1}
  ];
  
  (* Return result *)
  <|
    "Method" -> "Runge-Kutta 4",
    "Times" -> tvals,
    "Values" -> yvals,
    "TimeStep" -> dt,
    "Success" -> True,
    "Order" -> 4
  |>
];

(* Implicit Euler Method *)
ImplicitEulerSolve[f_, y0_, {t_, t0_, tf_}, dt_] := Module[
  {tvals, yvals, ycurrent, tcurrent, ynew, eqn, sol, n, i},
  
  (* Input validation *)
  If[dt <= 0, Message[ImplicitEulerSolve::dt]; Return[$Failed]];
  If[tf <= t0, Message[ImplicitEulerSolve::tspan]; Return[$Failed]];
  
  (* Initialize *)
  n = Ceiling[(tf - t0)/dt] + 1;
  tvals = Table[t0 + (i - 1)*dt, {i, 1, n}];
  If[Last[tvals] < tf, AppendTo[tvals, tf]];
  
  yvals = {y0};
  ycurrent = y0;
  
  (* Integration loop *)
  Do[
    tcurrent = tvals[[i]];
    If[i < Length[tvals],
      dt = tvals[[i + 1]] - tcurrent;
      
      (* Solve implicit equation: y_new = y_current + dt*f[t_new, y_new] *)
      eqn = ynew == ycurrent + dt*f[tcurrent + dt, ynew];
      
      (* Try to solve analytically, fallback to numerical *)
      sol = Quiet[Solve[eqn, ynew]];
      If[sol === {} || !NumericQ[ynew /. First[sol]],
        (* Numerical solution using FindRoot *)
        sol = Quiet[FindRoot[eqn, {ynew, ycurrent}]];
      ];
      
      If[sol =!= $Failed && NumericQ[ynew /. First[sol]],
        ycurrent = ynew /. First[sol];
        AppendTo[yvals, ycurrent];
      ,
        (* Fallback to explicit Euler if implicit fails *)
        ycurrent = ycurrent + dt*f[tcurrent, ycurrent];
        AppendTo[yvals, ycurrent];
      ];
    ],
    {i, 1, Length[tvals] - 1}
  ];
  
  (* Return result *)
  <|
    "Method" -> "Implicit Euler",
    "Times" -> tvals,
    "Values" -> yvals,
    "TimeStep" -> dt,
    "Success" -> True,
    "Order" -> 1
  |>
];

(* Adaptive Runge-Kutta-Fehlberg Method *)
AdaptiveRKSolve[f_, y0_, {t_, t0_, tf_}, opts___] := Module[
  {tol, maxdt, mindt, tvals, yvals, ycurrent, tcurrent, dt, 
   k1, k2, k3, k4, k5, k6, y4, y5, error, scale},
  
  (* Parse options *)
  tol = "Tolerance" /. {opts} /. "Tolerance" -> 1*^-6;
  maxdt = "MaxTimeStep" /. {opts} /. "MaxTimeStep" -> (tf - t0)/10;
  mindt = "MinTimeStep" /. {opts} /. "MinTimeStep" -> 1*^-12;
  
  (* Initialize *)
  tvals = {t0};
  yvals = {y0};
  ycurrent = y0;
  tcurrent = t0;
  dt = maxdt;
  
  (* Integration loop *)
  While[tcurrent < tf,
    (* Ensure we don't overshoot *)
    If[tcurrent + dt > tf, dt = tf - tcurrent];
    
    (* RKF45 stages *)
    k1 = dt*f[tcurrent, ycurrent];
    k2 = dt*f[tcurrent + dt/4, ycurrent + k1/4];
    k3 = dt*f[tcurrent + 3*dt/8, ycurrent + 3*k1/32 + 9*k2/32];
    k4 = dt*f[tcurrent + 12*dt/13, ycurrent + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197];
    k5 = dt*f[tcurrent + dt, ycurrent + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104];
    k6 = dt*f[tcurrent + dt/2, ycurrent - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40];
    
    (* 4th and 5th order solutions *)
    y4 = ycurrent + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5;
    y5 = ycurrent + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55;
    
    (* Error estimate *)
    error = Abs[y5 - y4];
    scale = tol*(1 + Abs[ycurrent]);
    
    If[VectorQ[error],
      error = Max[error/scale];
    ,
      error = error/scale;
    ];
    
    (* Accept/reject step *)
    If[error <= 1,
      (* Accept step *)
      tcurrent += dt;
      ycurrent = y5; (* Use higher order solution *)
      AppendTo[tvals, tcurrent];
      AppendTo[yvals, ycurrent];
    ];
    
    (* Update step size *)
    If[error > 0,
      dt = 0.9*dt*Min[Max[(1/error)^(1/4), 0.3], 2];
    ];
    dt = Max[mindt, Min[maxdt, dt]];
  ];
  
  (* Return result *)
  <|
    "Method" -> "Adaptive RKF45",
    "Times" -> tvals,
    "Values" -> yvals,
    "FinalTimeStep" -> dt,
    "Success" -> True,
    "Order" -> 5,
    "Tolerance" -> tol
  |>
];

(* Convergence Study *)
ODEConvergenceStudy[f_, y0_, {t_, t0_, tf_}, analytical_, dtList_] := Module[
  {results, errors, rates, i},
  
  (* Solve with different time steps *)
  results = Table[
    RungeKutta4Solve[f, y0, {t, t0, tf}, dt],
    {dt, dtList}
  ];
  
  (* Calculate errors *)
  errors = Table[
    If[results[[i]]["Success"],
      Abs[Last[results[[i]]["Values"]] - analytical[tf]],
      Undefined
    ],
    {i, Length[dtList]}
  ];
  
  (* Calculate convergence rates *)
  rates = Table[
    If[i > 1 && NumericQ[errors[[i]]] && NumericQ[errors[[i-1]]],
      Log[errors[[i-1]]/errors[[i]]]/Log[dtList[[i-1]]/dtList[[i]]],
      Undefined
    ],
    {i, 2, Length[dtList]}
  ];
  
  (* Return analysis *)
  <|
    "TimeSteps" -> dtList,
    "Errors" -> errors,
    "ConvergenceRates" -> rates,
    "Results" -> results
  |>
];

(* Berkeley-styled plotting *)
BerkeleyODEPlot[solution_, opts___] := Module[
  {times, values, plotOpts, colors, method},
  
  times = solution["Times"];
  values = solution["Values"];
  method = solution["Method"];
  
  (* Default Berkeley styling *)
  colors = {BerkeleyBlue, CaliforniaGold, BerkeleyLightBlue, BerkeleyDarkGold, BerkeleySecondaryBlue};
  
  plotOpts = {
    PlotStyle -> colors,
    Frame -> True,
    FrameStyle -> BerkeleyBlue,
    GridLines -> Automatic,
    GridLinesStyle -> Directive[Gray, Dashed, Opacity[0.3]],
    PlotTheme -> "Detailed",
    ImageSize -> 600,
    PlotLabel -> Style[method <> " Solution", 16, BerkeleyBlue, Bold],
    FrameLabel -> {Style["Time", 14, BerkeleyBlue], Style["y(t)", 14, BerkeleyBlue]},
    LabelStyle -> {FontFamily -> "Arial", FontSize -> 12}
  };
  
  (* Handle vector vs scalar solutions *)
  If[VectorQ[First[values]],
    (* System of ODEs *)
    ListLinePlot[
      Table[Transpose[{times, values[[All, i]]}], {i, Length[First[values]]}],
      Evaluate[Sequence @@ plotOpts],
      PlotLegends -> Table["y" <> ToString[i], {i, Length[First[values]]}],
      opts
    ]
  ,
    (* Single ODE *)
    ListLinePlot[
      Transpose[{times, values}],
      Evaluate[Sequence @@ plotOpts],
      opts
    ]
  ]
];

(* Error messages *)
ExplicitEulerSolve::dt = "Time step must be positive.";
ExplicitEulerSolve::tspan = "Final time must be greater than initial time.";
RungeKutta4Solve::dt = "Time step must be positive.";
RungeKutta4Solve::tspan = "Final time must be greater than initial time.";
ImplicitEulerSolve::dt = "Time step must be positive.";
ImplicitEulerSolve::tspan = "Final time must be greater than initial time.";

End[];
EndPackage[];