function pid_temperature_control()
    % PID Temperature Control Example (MATLAB)
    %
    % A simple temperature control system using PID controller.
    % This example demonstrates basic PID tuning and performance analysis.
    %
    % Learning Objectives:
    % - Understand PID controller components
    % - Learn basic tuning methods
    % - Analyze step response and stability
    fprintf('PID Temperature Control Example\n');
    fprintf('=====================================\n');
    % Add path to core Control classes
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'core'));
    % Set Berkeley visual defaults
    setBerkeleyDefaults();
    % Simulation parameters
    dt = 1.0;        % Sample time (seconds)
    duration = 600;  % Simulation time (seconds)
    setpoint = 75.0; % Target temperature (°C)
    % Initial PID tuning (conservative)
    config = struct(...
        'kp', 2.0, ...      % Proportional gain
        'ki', 0.05, ...     % Integral gain
        'kd', 20.0, ...     % Derivative gain
        'dt', dt, ...
        'output_min', 0.0, ...    % Minimum heater power (%)
        'output_max', 100.0, ...  % Maximum heater power (%)
        'derivative_filter_tau', 10.0 ... % Derivative filter
    );
    % Create controller
    controller = PIDController(config);
    fprintf('Initial PID gains: Kp=%.2f, Ki=%.4f, Kd=%.2f\n', ...
            config.kp, config.ki, config.kd);
    fprintf('Target temperature: %.1f°C\n', setpoint);
    fprintf('Simulation time: %.0f seconds\n', duration);
    % Run simulation
    results = simulatePIDSystem(controller, setpoint, duration, dt);
    % Performance analysis
    final_error = abs(setpoint - results.output(end));
    settling_time = calculateSettlingTime(results.time, results.output, setpoint);
    overshoot = calculateOvershoot(results.output, setpoint);
    fprintf('\nPerformance Analysis:\n');
    fprintf('Final error: %.2f°C\n', final_error);
    fprintf('Settling time (2%%): %.1f seconds\n', settling_time);
    fprintf('Overshoot: %.1f%%\n', overshoot);
    % Plot results
    createPlots(results, setpoint);
    % Try different tuning
    fprintf('\nTrying Ziegler-Nichols tuning...\n');
    ku = 8.0;   % Ultimate gain (estimated)
    tu = 120.0; % Ultimate period (estimated)
    zn_config = controller.tuneZieglerNichols(ku, tu, 'classic');
    fprintf('Z-N gains: Kp=%.2f, Ki=%.4f, Kd=%.2f\n', ...
            zn_config.kp, zn_config.ki, zn_config.kd);
    % Test new tuning
    controller_zn = PIDController(zn_config);
    results_zn = simulatePIDSystem(controller_zn, setpoint, duration, dt);
    % Compare controllers
    compareControllers(results, results_zn, setpoint);
end
function results = simulatePIDSystem(controller, setpoint, duration, dt)
    % Simulate closed-loop PID control system
    t = 0:dt:duration;
    n_points = length(t);
    % Initialize arrays
    output = zeros(1, n_points);
    control = zeros(1, n_points);
    measured = zeros(1, n_points);
    % Reset controller
    controller.reset();
    % Add measurement noise
    noise_std = 0.5;
    rng(42); % For reproducibility
    % Simulation loop
    for i = 2:n_points
        % Add measurement noise
        measured(i-1) = output(i-1) + noise_std * randn();
        % Compute control signal
        control(i) = controller.update(setpoint, measured(i-1), dt);
        % Apply to plant (simple first-order system)
        output(i) = temperaturePlant(control(i), output(i-1), dt);
    end
    % Package results
    results = struct();
    results.time = t;
    results.setpoint = setpoint * ones(size(t));
    results.output = output;
    results.control = control;
    results.measured = measured;
end
function y_new = temperaturePlant(u, y_prev, dt)
    % Simple first-order temperature plant model
    %
    % Represents: τ dy/dt + y = K*u
    % where τ = time constant, K = gain
    % Plant parameters
    tau = 60.0;  % Time constant (seconds)
    K = 2.0;     % Steady-state gain (°C per % power)
    % Discretize: y[k+1] = a*y[k] + b*u[k]
    a = exp(-dt / tau);
    b = K * (1 - a);
    y_new = a * y_prev + b * u;
end
function settling_time = calculateSettlingTime(time, output, setpoint)
    % Calculate 2% settling time
    tolerance = 0.02;
    error_band = setpoint * tolerance;
    settling_time = time(end);
    for i = length(output):-1:1
        if abs(output(i) - setpoint) > error_band
            if i < length(time)
                settling_time = time(i + 1);
            end
            break;
        end
    end
    if settling_time == time(end)
        settling_time = 0.0;
    end
end
function overshoot = calculateOvershoot(output, setpoint)
    % Calculate percentage overshoot
    max_value = max(output);
    if max_value > setpoint
        overshoot = 100 * (max_value - setpoint) / setpoint;
    else
        overshoot = 0.0;
    end
end
function createPlots(results, setpoint)
    % Create comprehensive plots of PID performance
    figure('Position', [100, 100, 1200, 800]);
    % Temperature response
    subplot(2, 2, 1);
    plot(results.time, results.output, 'Color', [0, 0.2, 0.38], 'LineWidth', 2);
    hold on;
    plot(results.time, results.setpoint, 'Color', [0.99, 0.71, 0.08], ...
         'LineWidth', 2, 'LineStyle', '--');
    xlabel('Time (s)');
    ylabel('Temperature (°C)');
    title('Temperature Response');
    legend('Temperature', 'Setpoint', 'Location', 'best');
    grid on;
    % Control signal
    subplot(2, 2, 2);
    plot(results.time, results.control, 'Color', [0, 0.2, 0.38], 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Heater Power (%)');
    title('Control Signal');
    grid on;
    % Error signal
    subplot(2, 2, 3);
    error = results.setpoint - results.output;
    plot(results.time, error, 'Color', [0, 0.2, 0.38], 'LineWidth', 2);
    hold on;
    plot(results.time, zeros(size(results.time)), 'k-', 'LineWidth', 0.5);
    xlabel('Time (s)');
    ylabel('Error (°C)');
    title('Tracking Error');
    grid on;
    % Control effort histogram
    subplot(2, 2, 4);
    histogram(results.control, 30, 'FaceColor', [0.99, 0.71, 0.08], 'FaceAlpha', 0.7);
    xlabel('Heater Power (%)');
    ylabel('Frequency');
    title('Control Effort Distribution');
    grid on;
    sgtitle('PID Temperature Control Performance', 'FontSize', 16, 'FontWeight', 'bold');
end
function compareControllers(results1, results2, setpoint)
    % Compare two controller performances
    figure('Position', [200, 200, 1200, 500]);
    % Temperature comparison
    subplot(1, 2, 1);
    plot(results1.time, results1.output, 'Color', [0, 0.2, 0.38], ...
         'LineWidth', 2, 'DisplayName', 'Conservative Tuning');
    hold on;
    plot(results2.time, results2.output, 'Color', [0.99, 0.71, 0.08], ...
         'LineWidth', 2, 'DisplayName', 'Ziegler-Nichols');
    plot(results1.time, setpoint * ones(size(results1.time)), 'k--', ...
         'LineWidth', 1, 'DisplayName', 'Setpoint');
    xlabel('Time (s)');
    ylabel('Temperature (°C)');
    title('Controller Comparison');
    legend('Location', 'best');
    grid on;
    % Control effort comparison
    subplot(1, 2, 2);
    plot(results1.time, results1.control, 'Color', [0, 0.2, 0.38], ...
         'LineWidth', 2, 'DisplayName', 'Conservative');
    hold on;
    plot(results2.time, results2.control, 'Color', [0.99, 0.71, 0.08], ...
         'LineWidth', 2, 'DisplayName', 'Ziegler-Nichols');
    xlabel('Time (s)');
    ylabel('Heater Power (%)');
    title('Control Effort Comparison');
    legend('Location', 'best');
    grid on;
    sgtitle('PID Controller Comparison', 'FontSize', 16, 'FontWeight', 'bold');
end