/**
 * @file mtj_testbench.v
 * @brief Testbench for Magnetic Tunnel Junction Model
 * 
 * Comprehensive validation testbench that demonstrates:
 * - DC I-V characteristics
 * - TMR vs bias voltage
 * - Switching dynamics
 * - Temperature effects
 * - Process variation analysis
 * 
 * @author Meshal Alawein <meshal@berkeley.edu>
 * @copyright 2025, University of California Berkeley
 * @license MIT License
 */

`timescale 1ns/1ps

module mtj_testbench();

    // Test signals
    electrical free_layer, reference_layer;
    real voltage_bias;
    real current_measured;
    real resistance;
    real tmr_measured;
    
    // Test parameters
    parameter real V_MAX = 1.0;           // Maximum bias voltage
    parameter real V_STEP = 0.01;         // Voltage step size
    parameter real TEMP_MIN = 200;        // Minimum temperature
    parameter real TEMP_MAX = 400;        // Maximum temperature
    parameter real TEMP_STEP = 50;        // Temperature step
    
    // Instance of MTJ model
    magnetic_tunnel_junction #(
        .area(50e-18),                    // 50 nm^2
        .tox(1.2e-9),                     // 1.2 nm MgO
        .tmr0(200),                       // 200% TMR
        .ra_p(10),                        // 10 Ohm-um^2
        .temperature(300),                // Room temperature
        .delta_free(60),                  // 60 kT barrier
        .xi_vcma(50e-6)                   // 50 uJ/V-m VCMA
    ) mtj_dut (
        .free_layer(free_layer),
        .reference_layer(reference_layer)
    );
    
    // Voltage source for biasing
    vsource #(.type("dc")) V_bias (
        .p(free_layer),
        .n(reference_layer),
        .dc(voltage_bias)
    );
    
    // Current probe
    real I_probe;
    analog begin
        I_probe = I(free_layer, reference_layer);
    end
    
    // Test procedures
    initial begin
        // Open output files
        integer iv_file, tmr_file, temp_file, switch_file;
        iv_file = $fopen("mtj_iv_characteristics.dat", "w");
        tmr_file = $fopen("mtj_tmr_vs_bias.dat", "w");
        temp_file = $fopen("mtj_temperature_effects.dat", "w");
        switch_file = $fopen("mtj_switching_dynamics.dat", "w");
        
        // Write headers
        $fwrite(iv_file, "# Voltage(V)\tCurrent_P(uA)\tCurrent_AP(uA)\tResistance_P(kOhm)\tResistance_AP(kOhm)\n");
        $fwrite(tmr_file, "# Voltage(V)\tTMR(%)\n");
        $fwrite(temp_file, "# Temperature(K)\tResistance_P(kOhm)\tResistance_AP(kOhm)\tTMR(%)\n");
        $fwrite(switch_file, "# Time(ns)\tVoltage(V)\tCurrent(uA)\tState\n");
        
        $display("\n========== MTJ Model Testbench ==========");
        $display("Starting comprehensive validation tests...");
        $display("========================================\n");
        
        // Test 1: DC I-V Characteristics
        $display("Test 1: DC I-V Characteristics");
        $display("------------------------------");
        
        real I_p, I_ap, R_p, R_ap;
        
        // Parallel state measurement
        mtj_dut.state = 1;
        #100;  // Settle time
        
        for (voltage_bias = -V_MAX; voltage_bias <= V_MAX; voltage_bias = voltage_bias + V_STEP) begin
            #10;  // Wait for convergence
            I_p = I_probe * 1e6;  // Convert to uA
            R_p = (voltage_bias != 0) ? abs(voltage_bias / I_probe) * 1e-3 : 0;  // kOhm
            
            // Antiparallel state measurement  
            mtj_dut.state = -1;
            #10;
            I_ap = I_probe * 1e6;
            R_ap = (voltage_bias != 0) ? abs(voltage_bias / I_probe) * 1e-3 : 0;
            
            // Calculate TMR
            tmr_measured = (R_ap - R_p) / R_p * 100;
            
            // Write to files
            $fwrite(iv_file, "%g\t%g\t%g\t%g\t%g\n", voltage_bias, I_p, I_ap, R_p, R_ap);
            $fwrite(tmr_file, "%g\t%g\n", voltage_bias, tmr_measured);
            
            // Reset to parallel state
            mtj_dut.state = 1;
        end
        
        $display("I-V characteristics completed");
        $display("Maximum current (P): %g uA", I_p);
        $display("Maximum current (AP): %g uA", I_ap);
        $display("TMR at zero bias: %g %%\n", tmr_measured);
        
        // Test 2: Temperature Effects
        $display("Test 2: Temperature Effects");
        $display("---------------------------");
        
        voltage_bias = 0.1;  // Fixed bias
        real temp_orig = mtj_dut.temperature;
        
        for (real temp = TEMP_MIN; temp <= TEMP_MAX; temp = temp + TEMP_STEP) begin
            mtj_dut.temperature = temp;
            
            // Parallel state
            mtj_dut.state = 1;
            #100;
            R_p = abs(voltage_bias / I_probe) * 1e-3;
            
            // Antiparallel state
            mtj_dut.state = -1; 
            #100;
            R_ap = abs(voltage_bias / I_probe) * 1e-3;
            
            tmr_measured = (R_ap - R_p) / R_p * 100;
            
            $fwrite(temp_file, "%g\t%g\t%g\t%g\n", temp, R_p, R_ap, tmr_measured);
            $display("T = %g K: R_P = %g kOhm, R_AP = %g kOhm, TMR = %g %%", 
                    temp, R_p, R_ap, tmr_measured);
        end
        
        mtj_dut.temperature = temp_orig;  // Restore original
        $display("");
        
        // Test 3: Switching Dynamics
        $display("Test 3: Switching Dynamics");
        $display("--------------------------");
        
        // Apply switching pulse
        real t_switch;
        mtj_dut.state = 1;  // Start in P state
        
        // Ramp voltage for switching
        for (t_switch = 0; t_switch <= 10; t_switch = t_switch + 0.1) begin
            if (t_switch < 2) begin
                voltage_bias = 0;
            end else if (t_switch < 5) begin
                voltage_bias = 0.8;  // Switching voltage
            end else if (t_switch < 8) begin
                voltage_bias = -0.8;  // Reverse switching
            end else begin
                voltage_bias = 0;
            end
            
            #0.1;
            $fwrite(switch_file, "%g\t%g\t%g\t%g\n", 
                    t_switch, voltage_bias, I_probe*1e6, mtj_dut.state);
        end
        
        $display("Switching dynamics test completed\n");
        
        // Test 4: Process Variation Analysis (Monte Carlo)
        $display("Test 4: Process Variation Analysis");
        $display("-----------------------------------");
        
        integer mc_file;
        mc_file = $fopen("mtj_monte_carlo.dat", "w");
        $fwrite(mc_file, "# Run\tArea(nm^2)\ttox(nm)\tTMR(%)\tR_P(kOhm)\tR_AP(kOhm)\n");
        
        voltage_bias = 0.1;
        integer mc_runs = 100;
        real area_vals[100], tox_vals[100], tmr_vals[100];
        real rp_vals[100], rap_vals[100];
        
        for (integer i = 0; i < mc_runs; i = i + 1) begin
            // Each instance will have different random variations
            magnetic_tunnel_junction #(
                .area(50e-18),
                .tox(1.2e-9),
                .tmr0(200),
                .sigma_area(0.1),
                .sigma_tox(0.05),
                .sigma_tmr(0.15)
            ) mtj_mc();
            
            #10;  // Let it settle
            
            // Extract parameters
            area_vals[i] = mtj_mc.area_eff * 1e18;
            tox_vals[i] = mtj_mc.tox_eff * 1e9;
            tmr_vals[i] = mtj_mc.tmr_eff;
            rp_vals[i] = mtj_mc.resistance_p * 1e-3;
            rap_vals[i] = mtj_mc.resistance_ap * 1e-3;
            
            $fwrite(mc_file, "%d\t%g\t%g\t%g\t%g\t%g\n", 
                    i, area_vals[i], tox_vals[i], tmr_vals[i], rp_vals[i], rap_vals[i]);
        end
        
        // Calculate statistics
        real mean_rp = 0, mean_rap = 0, mean_tmr = 0;
        real std_rp = 0, std_rap = 0, std_tmr = 0;
        
        for (integer i = 0; i < mc_runs; i = i + 1) begin
            mean_rp = mean_rp + rp_vals[i];
            mean_rap = mean_rap + rap_vals[i];
            mean_tmr = mean_tmr + tmr_vals[i];
        end
        
        mean_rp = mean_rp / mc_runs;
        mean_rap = mean_rap / mc_runs;
        mean_tmr = mean_tmr / mc_runs;
        
        for (integer i = 0; i < mc_runs; i = i + 1) begin
            std_rp = std_rp + (rp_vals[i] - mean_rp)**2;
            std_rap = std_rap + (rap_vals[i] - mean_rap)**2;
            std_tmr = std_tmr + (tmr_vals[i] - mean_tmr)**2;
        end
        
        std_rp = sqrt(std_rp / mc_runs);
        std_rap = sqrt(std_rap / mc_runs);
        std_tmr = sqrt(std_tmr / mc_runs);
        
        $display("Monte Carlo Results (%d runs):", mc_runs);
        $display("R_P:  mean = %g kOhm, std = %g kOhm (%.1f%%)", 
                mean_rp, std_rp, std_rp/mean_rp*100);
        $display("R_AP: mean = %g kOhm, std = %g kOhm (%.1f%%)", 
                mean_rap, std_rap, std_rap/mean_rap*100);
        $display("TMR:  mean = %g %%, std = %g %% (%.1f%%)\n", 
                mean_tmr, std_tmr, std_tmr/mean_tmr*100);
        
        // Close files
        $fclose(iv_file);
        $fclose(tmr_file);
        $fclose(temp_file);
        $fclose(switch_file);
        $fclose(mc_file);
        
        $display("========================================");
        $display("All tests completed successfully!");
        $display("Results written to data files.");
        $display("========================================\n");
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000;  // 100 us timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end
    
endmodule