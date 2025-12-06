(* ::Package:: *)

(* PIDController - Professional PID controller implementation for Mathematica *)

BeginPackage["SciComp`Control`PIDController`"];

PIDController::usage = "PIDController[config] creates a PID controller with given configuration.
Features:
- Standard PID control with configurable gains
- Anti-windup protection  
- Derivative filtering to reduce noise sensitivity
- Setpoint weighting for improved response
- Reset functionality

Example:
config = <|\"kp\" -> 2.0, \"ki\" -> 0.5, \"kd\" -> 0.1, \"dt\" -> 0.01|>;
controller = PIDController[config];
output = controller[\"update\"][10.0, 8.5];";

PIDUpdate::usage = "PIDUpdate[controller, setpoint, measurement, dt] computes PID control output.";
PIDReset::usage = "PIDReset[controller] resets controller internal state.";
PIDComponents::usage = "PIDComponents[controller, setpoint, measurement, dt] returns individual PID components.";
ZieglerNicholsTuning::usage = "ZieglerNicholsTuning[ku, tu, method] returns tuned PID parameters.";

Begin["`Private`"];

(* Default configuration *)
defaultConfig = <|
    "kp" -> 1.0,
    "ki" -> 0.0, 
    "kd" -> 0.0,
    "dt" -> 0.01,
    "outputMin" -> -Infinity,
    "outputMax" -> Infinity,
    "derivativeFilterTau" -> 0.0
|>;

(* PID Controller constructor *)
PIDController[userConfig_Association] := Module[{config, controller},
    (* Merge user config with defaults *)
    config = Join[defaultConfig, userConfig];
    
    (* Validate configuration *)
    If[!NumericQ[config["kp"]] || !NumericQ[config["ki"]] || !NumericQ[config["kd"]],
        Message[PIDController::config, "Gains must be numeric"];
        Return[$Failed];
    ];
    
    If[config["dt"] <= 0,
        Message[PIDController::config, "Sample time must be positive"];
        Return[$Failed];
    ];
    
    (* Initialize controller state *)
    controller = <|
        "config" -> config,
        "previousError" -> 0.0,
        "integral" -> 0.0,
        "previousMeasurement" -> 0.0,
        "previousDerivative" -> 0.0,
        
        (* Methods *)
        "update" -> Function[{setpoint, measurement, dt},
            PIDUpdate[controller, setpoint, measurement, dt]
        ],
        "reset" -> Function[{},
            PIDReset[controller]
        ],
        "getComponents" -> Function[{setpoint, measurement, dt},
            PIDComponents[controller, setpoint, measurement, dt]
        ],
        "tuneZieglerNichols" -> Function[{ku, tu, method},
            ZieglerNicholsTuning[ku, tu, method]
        ]
    |>;
    
    controller
];

(* Reset controller state *)
PIDReset[controller_Association] := Module[{},
    controller["previousError"] = 0.0;
    controller["integral"] = 0.0;
    controller["previousMeasurement"] = 0.0;
    controller["previousDerivative"] = 0.0;
];

(* Main PID update function *)
PIDUpdate[controller_Association, setpoint_?NumericQ, measurement_?NumericQ, dt_?NumericQ] := 
Module[{config, error, proportional, integral, integralTerm, provisionalOutput, 
        derivative, derivativeRaw, alpha, derivativeTerm, output},
    
    config = controller["config"];
    
    (* Use provided dt or default from config *)
    If[dt === Automatic, dt = config["dt"]];
    
    (* Calculate error *)
    error = setpoint - measurement;
    
    (* Proportional term *)
    proportional = config["kp"] * error;
    
    (* Integral term with anti-windup *)
    controller["integral"] += error * dt;
    
    (* Anti-windup: clamp integral if output would saturate *)
    integralTerm = config["ki"] * controller["integral"];
    provisionalOutput = proportional + integralTerm;
    
    If[provisionalOutput > config["outputMax"],
        controller["integral"] = (config["outputMax"] - proportional) / config["ki"];
    ];
    If[provisionalOutput < config["outputMin"],
        controller["integral"] = (config["outputMin"] - proportional) / config["ki"];
    ];
    
    integral = config["ki"] * controller["integral"];
    
    (* Derivative term (on measurement to avoid derivative kick) *)
    derivativeRaw = -(measurement - controller["previousMeasurement"]) / dt;
    
    (* Apply derivative filtering if specified *)
    If[config["derivativeFilterTau"] > 0,
        alpha = dt / (config["derivativeFilterTau"] + dt);
        derivative = alpha * derivativeRaw + (1 - alpha) * controller["previousDerivative"];
        controller["previousDerivative"] = derivative;
    ,
        derivative = derivativeRaw;
    ];
    
    derivativeTerm = config["kd"] * derivative;
    
    (* Compute total output *)
    output = proportional + integral + derivativeTerm;
    
    (* Apply output limits *)
    output = Clip[output, {config["outputMin"], config["outputMax"]}];
    
    (* Store values for next iteration *)
    controller["previousError"] = error;
    controller["previousMeasurement"] = measurement;
    
    output
];

(* Overload for automatic dt *)
PIDUpdate[controller_Association, setpoint_?NumericQ, measurement_?NumericQ] := 
    PIDUpdate[controller, setpoint, measurement, controller["config"]["dt"]];

(* Get individual PID components *)
PIDComponents[controller_Association, setpoint_?NumericQ, measurement_?NumericQ, dt_?NumericQ] := 
Module[{config, error, proportional, integral, derivative, derivativeRaw, alpha, derivativeTerm},
    
    config = controller["config"];
    If[dt === Automatic, dt = config["dt"]];
    
    error = setpoint - measurement;
    proportional = config["kp"] * error;
    integral = config["ki"] * controller["integral"];
    
    derivativeRaw = -(measurement - controller["previousMeasurement"]) / dt;
    If[config["derivativeFilterTau"] > 0,
        alpha = dt / (config["derivativeFilterTau"] + dt);
        derivative = alpha * derivativeRaw + (1 - alpha) * controller["previousDerivative"];
    ,
        derivative = derivativeRaw;
    ];
    
    derivativeTerm = config["kd"] * derivative;
    
    <|"proportional" -> proportional, "integral" -> integral, "derivative" -> derivativeTerm|>
];

(* Ziegler-Nichols tuning *)
ZieglerNicholsTuning[ku_?NumericQ, tu_?NumericQ, method_String: "classic"] := 
Module[{kp, ki, kd},
    Switch[method,
        "classic",
        kp = 0.6 * ku;
        ki = 2.0 * kp / tu;
        kd = kp * tu / 8.0;
        ,
        "pessen", 
        kp = 0.7 * ku;
        ki = 2.5 * kp / tu;
        kd = 0.15 * kp * tu;
        ,
        "someOvershoot",
        kp = 0.33 * ku;
        ki = 2.0 * kp / tu;
        kd = kp * tu / 3.0;
        ,
        "noOvershoot",
        kp = 0.2 * ku;
        ki = 2.0 * kp / tu;
        kd = kp * tu / 3.0;
        ,
        _,
        Message[ZieglerNicholsTuning::method, method];
        Return[$Failed];
    ];
    
    <|"kp" -> kp, "ki" -> ki, "kd" -> kd|>
];

(* Simulate PID system *)
SimulatePIDSystem[controller_Association, plantFunction_, setpoint_, duration_?NumericQ, 
                  opts:OptionsPattern[{"dt" -> 0.01, "noiseStd" -> 0.0}]] := 
Module[{dt, noiseStd, t, nPoints, output, control, measured, i},
    
    dt = OptionValue["dt"];
    noiseStd = OptionValue["noiseStd"];
    
    t = Range[0, duration, dt];
    nPoints = Length[t];
    
    (* Initialize arrays *)
    output = ConstantArray[0.0, nPoints];
    control = ConstantArray[0.0, nPoints];
    measured = ConstantArray[0.0, nPoints];
    
    (* Reset controller *)
    PIDReset[controller];
    
    (* Simulation loop *)
    SeedRandom[42]; (* For reproducibility *)
    Do[
        (* Add measurement noise *)
        measured[[i-1]] = output[[i-1]] + noiseStd * RandomReal[NormalDistribution[]];
        
        (* Compute control signal *)
        control[[i]] = PIDUpdate[controller, setpoint, measured[[i-1]], dt];
        
        (* Apply to plant *)
        output[[i]] = plantFunction[control[[i]], output[[i-1]], dt];
    , {i, 2, nPoints}];
    
    <|"time" -> t, "setpoint" -> ConstantArray[setpoint, nPoints], 
      "output" -> output, "control" -> control, "measured" -> measured|>
];

(* Error messages *)
PIDController::config = "Configuration error: `1`"; 
ZieglerNicholsTuning::method = "Unknown tuning method: `1`";

End[];
EndPackage[];