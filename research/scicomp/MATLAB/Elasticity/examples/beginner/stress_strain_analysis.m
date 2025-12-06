function stress_strain_analysis()
    % Basic Stress-Strain Analysis (MATLAB)
    %
    % This example demonstrates fundamental stress and strain tensor operations,
    % elastic material behavior, and simple stress analysis for engineering applications.
    %
    % Learning Objectives:
    % - Understand stress and strain tensor concepts
    % - Apply Hooke's law for elastic materials
    % - Calculate principal stresses and strains
    % - Perform basic failure analysis using von Mises criterion
    fprintf('Basic Stress-Strain Analysis - MATLAB\n');
    fprintf('=====================================\n');
    fprintf('This example demonstrates fundamental concepts in elasticity theory\n');
    fprintf('including stress/strain tensors, Hooke''s law, and failure analysis.\n\n');
    % Add path to core Elasticity classes
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'core'));
    % Set Berkeley visual defaults
    setBerkeleyDefaults();
    % Create material
    steel = createSteelMaterial();
    % Demonstrate stress tensor operations
    [stressUniaxial, stressShear, stressComplex] = demonstrateStressTensorOperations();
    % Demonstrate strain tensor operations
    strain = demonstrateStrainTensorOperations();
    % Apply Hooke's law
    stressFromStrain = demonstrateHookesLaw(steel, strain);
    % Analyze wave velocities
    [vP, vS] = analyzeElasticWaveVelocities(steel);
    % Coordinate transformation
    [stressOrig, stressRot] = demonstrateCoordinateTransformation();
    % Energy analysis
    energyDensity = energyAnalysisExample(steel, stressComplex, strain);
    % Failure analysis
    stressStates = failureAnalysisExample();
    % Generate plots
    plotStressAnalysisResults(stressStates);
    fprintf('\n%s\n', repmat('=', 1, 50));
    fprintf('Analysis Complete!\n');
    fprintf('\nKey Results Summary:\n');
    fprintf('• Steel Young''s modulus: %.0f GPa\n', steel.getYoungsModulus()/1e9);
    fprintf('• P-wave velocity: %.0f m/s\n', vP);
    fprintf('• S-wave velocity: %.0f m/s\n', vS);
    fprintf('• Elastic energy density: %.1f kJ/m³\n', energyDensity/1e3);
end
function steel = createSteelMaterial()
    % Create steel material with typical properties
    fprintf('Creating Steel Material Properties\n');
    fprintf('%s\n', repmat('-', 1, 40));
    % Typical mild steel properties
    youngsModulus = 200e9;  % Pa (200 GPa)
    poissonsRatio = 0.30;
    density = 7850;  % kg/m³
    steel = IsotropicElasticity(youngsModulus, poissonsRatio, density);
    fprintf('Young''s modulus: %.1f GPa\n', youngsModulus/1e9);
    fprintf('Poisson''s ratio: %.2f\n', poissonsRatio);
    fprintf('Shear modulus: %.1f GPa\n', steel.getShearModulus()/1e9);
    fprintf('Bulk modulus: %.1f GPa\n', steel.getBulkModulus()/1e9);
    fprintf('Density: %d kg/m³\n', density);
end
function [stressUniaxial, stressShear, stressComplex] = demonstrateStressTensorOperations()
    % Demonstrate stress tensor creation and operations
    fprintf('\n\nStress Tensor Operations\n');
    fprintf('%s\n', repmat('=', 1, 30));
    % Create stress tensor for uniaxial tension
    fprintf('\n1. Uniaxial Tension (σx = 100 MPa)\n');
    fprintf('%s\n', repmat('-', 1, 35));
    stressUniaxial = StressTensor(100e6, 0, 0, 0, 0, 0);  % 100 MPa in x-direction
    fprintf('Stress tensor (MPa):\n');
    disp(stressUniaxial.getTensor()/1e6);
    % Calculate principal stresses
    [principalStresses, ~] = stressUniaxial.principalStresses();
    fprintf('Principal stresses: [%.1f, %.1f, %.1f] MPa\n', principalStresses/1e6);
    fprintf('von Mises stress: %.1f MPa\n', stressUniaxial.vonMisesStress()/1e6);
    % Create stress tensor for pure shear
    fprintf('\n2. Pure Shear (τxy = 50 MPa)\n');
    fprintf('%s\n', repmat('-', 1, 30));
    stressShear = StressTensor(0, 0, 0, 50e6, 0, 0);  % 50 MPa shear
    fprintf('Stress tensor (MPa):\n');
    disp(stressShear.getTensor()/1e6);
    [principalStresses, ~] = stressShear.principalStresses();
    fprintf('Principal stresses: [%.1f, %.1f, %.1f] MPa\n', principalStresses/1e6);
    fprintf('von Mises stress: %.1f MPa\n', stressShear.vonMisesStress()/1e6);
    fprintf('Maximum shear stress: %.1f MPa\n', stressShear.maximumShearStress()/1e6);
    % Create complex stress state
    fprintf('\n3. Complex Stress State\n');
    fprintf('%s\n', repmat('-', 1, 25));
    stressComplex = StressTensor(80e6, 40e6, 20e6, 30e6, 0, 0);
    fprintf('Stress tensor (MPa):\n');
    disp(stressComplex.getTensor()/1e6);
    [principalStresses, ~] = stressComplex.principalStresses();
    fprintf('Principal stresses: [%.1f, %.1f, %.1f] MPa\n', principalStresses/1e6);
    fprintf('von Mises stress: %.1f MPa\n', stressComplex.vonMisesStress()/1e6);
    % Stress invariants
    [I1, I2, I3] = stressComplex.invariants();
    fprintf('Stress invariants:\n');
    fprintf('  I₁ = %.1f MPa\n', I1/1e6);
    fprintf('  I₂ = %.1f (MPa)²\n', I2/1e12);
    fprintf('  I₃ = %.1f (MPa)³\n', I3/1e18);
end
function strain = demonstrateStrainTensorOperations()
    % Demonstrate strain tensor creation and operations
    fprintf('\n\nStrain Tensor Operations\n');
    fprintf('%s\n', repmat('=', 1, 30));
    % Create strain tensor
    fprintf('\n1. Simple Strain State\n');
    fprintf('%s\n', repmat('-', 1, 22));
    strain = StrainTensor(0.001, 0.0005, 0, 0.0008, 0, 0);  % Engineering strains
    fprintf('Strain tensor (microstrain):\n');
    disp(strain.getTensor()*1e6);
    % Calculate principal strains
    [principalStrains, ~] = strain.principalStrains();
    fprintf('Principal strains: [%.0f, %.0f, %.0f] microstrain\n', principalStrains*1e6);
    % Volumetric and deviatoric strains
    volumetricStrain = strain.volumetricStrain();
    equivalentStrain = strain.equivalentStrain();
    fprintf('Volumetric strain: %.0f microstrain\n', volumetricStrain*1e6);
    fprintf('Equivalent strain: %.0f microstrain\n', equivalentStrain*1e6);
end
function stressFromStrain = demonstrateHookesLaw(steel, strain)
    % Demonstrate Hooke's law relationships
    fprintf('\n\nHooke''s Law Application\n');
    fprintf('%s\n', repmat('=', 1, 25));
    % Calculate stress from strain
    stressFromStrain = steel.stressFromStrain(strain);
    fprintf('Stress from strain (Hooke''s law):\n');
    fprintf('Stress tensor (MPa):\n');
    disp(stressFromStrain.getTensor()/1e6);
    % Calculate strain from stress
    strainFromStress = steel.strainFromStress(stressFromStrain);
    fprintf('\nStrain from stress (compliance):\n');
    fprintf('Strain tensor (microstrain):\n');
    disp(strainFromStress.getTensor()*1e6);
    % Verify round-trip accuracy
    error = max(max(abs(strain.getTensor() - strainFromStress.getTensor())));
    fprintf('\nRound-trip error: %.2e microstrain\n', error*1e6);
    % Display material matrices
    fprintf('\nStiffness matrix (GPa):\n');
    stiffness = steel.getStiffnessMatrix() / 1e9;
    disp(stiffness);
    fprintf('\nCompliance matrix (1/TPa):\n');
    compliance = steel.getComplianceMatrix() * 1e12;
    disp(compliance);
end
function [vP, vS] = analyzeElasticWaveVelocities(steel)
    % Analyze elastic wave velocities in material
    fprintf('\n\nElastic Wave Velocities\n');
    fprintf('%s\n', repmat('=', 1, 25));
    [vP, vS] = steel.elasticWaveVelocities();
    fprintf('Longitudinal wave velocity (P-wave): %.0f m/s\n', vP);
    fprintf('Transverse wave velocity (S-wave): %.0f m/s\n', vS);
    fprintf('Velocity ratio (vₚ/vₛ): %.2f\n', vP/vS);
    % Calculate Poisson's ratio from wave velocities
    nuFromWaves = (vP^2 - 2*vS^2) / (2 * (vP^2 - vS^2));
    fprintf('Poisson''s ratio from wave velocities: %.3f\n', nuFromWaves);
    fprintf('Original Poisson''s ratio: %.3f\n', steel.getPoissonsRatio());
end
function stressStates = failureAnalysisExample()
    % Demonstrate failure analysis using von Mises criterion
    fprintf('\n\nFailure Analysis Example\n');
    fprintf('%s\n', repmat('=', 1, 30));
    fprintf('Material: Steel\n');
    yieldStrength = 250e6;  % Typical mild steel yield strength
    fprintf('Yield strength: %.0f MPa\n', yieldStrength/1e6);
    % Create various stress states
    stressStates = {
        {'Uniaxial tension', StressTensor(200e6, 0, 0, 0, 0, 0)}, ...
        {'Pure shear', StressTensor(0, 0, 0, 120e6, 0, 0)}, ...
        {'Biaxial tension', StressTensor(150e6, 100e6, 0, 0, 0, 0)}, ...
        {'Triaxial stress', StressTensor(180e6, 120e6, 80e6, 0, 0, 0)}
    };
    fprintf('\nFailure Analysis Results:\n');
    fprintf('%-20s %-12s %-15s %s\n', 'Load Case', 'von Mises', 'Safety Factor', 'Status');
    fprintf('%s\n', repmat('-', 1, 65));
    for i = 1:length(stressStates)
        name = stressStates{i}{1};
        stress = stressStates{i}{2};
        vonMises = stress.vonMisesStress();
        safetyFactor = yieldStrength / vonMises;
        status = 'SAFE';
        if safetyFactor <= 1.0
            status = 'FAILURE';
        end
        fprintf('%-20s %8.1f MPa %10.2f     %s\n', name, vonMises/1e6, safetyFactor, status);
    end
end
function [stressOrig, stressRot] = demonstrateCoordinateTransformation()
    % Demonstrate stress tensor coordinate transformation
    fprintf('\n\nCoordinate Transformation Example\n');
    fprintf('%s\n', repmat('=', 1, 40));
    % Original stress state in xy coordinate system
    stressOrig = StressTensor(100e6, 50e6, 0, 30e6, 0, 0);
    fprintf('Original stress tensor (MPa):\n');
    disp(stressOrig.getTensor()/1e6);
    % Rotation by 45 degrees about z-axis
    angle = pi/4;  % 45 degrees
    rotationMatrix = [cos(angle), -sin(angle), 0; ...
                     sin(angle),  cos(angle), 0; ...
                     0,           0,          1];
    % Transform stress tensor
    stressRot = stressOrig.transform(rotationMatrix);
    fprintf('\nRotated stress tensor (45° rotation, MPa):\n');
    disp(stressRot.getTensor()/1e6);
    % Principal stresses should remain invariant
    [principalOrig, ~] = stressOrig.principalStresses();
    [principalRot, ~] = stressRot.principalStresses();
    fprintf('\nPrincipal stress invariance check:\n');
    fprintf('Original: [%.1f, %.1f, %.1f] MPa\n', principalOrig/1e6);
    fprintf('Rotated:  [%.1f, %.1f, %.1f] MPa\n', principalRot/1e6);
    fprintf('Difference: %.6f MPa\n', max(abs(principalOrig - principalRot))/1e6);
end
function energyDensity = energyAnalysisExample(steel, stress, strain)
    % Demonstrate elastic energy calculations
    fprintf('\n\nElastic Energy Analysis\n');
    fprintf('%s\n', repmat('=', 1, 25));
    % Calculate elastic energy density
    energyDensity = steel.elasticEnergyDensity(stress, strain);
    fprintf('Elastic energy density: %.2f kJ/m³\n', energyDensity/1e3);
    % Compare with theoretical calculation
    [principalStresses, ~] = stress.principalStresses();
    s1 = principalStresses(1);
    s2 = principalStresses(2);
    s3 = principalStresses(3);
    E = steel.getYoungsModulus();
    nu = steel.getPoissonsRatio();
    theoreticalEnergy = (1/(2*E)) * (s1^2 + s2^2 + s3^2 - ...
                                    2*nu*(s1*s2 + s2*s3 + s3*s1));
    fprintf('Theoretical energy density: %.2f kJ/m³\n', theoreticalEnergy/1e3);
    fprintf('Relative error: %.3f%%\n', abs(energyDensity - theoreticalEnergy)/theoreticalEnergy*100);
end
function plotStressAnalysisResults(stressStates)
    % Plot stress analysis results
    fprintf('\n\nGenerating Stress Analysis Plots\n');
    fprintf('%s\n', repmat('-', 1, 35));
    % Extract data for plotting
    loadCases = cell(length(stressStates), 1);
    vonMisesStresses = zeros(length(stressStates), 1);
    for i = 1:length(stressStates)
        loadCases{i} = stressStates{i}{1};
        vonMisesStresses(i) = stressStates{i}{2}.vonMisesStress()/1e6;
    end
    yieldStrength = 250;  % MPa
    % Create figure with Berkeley styling
    figure('Position', [100, 100, 1200, 500]);
    % Plot 1: von Mises stress comparison
    subplot(1, 2, 1);
    bars = bar(1:length(loadCases), vonMisesStresses, ...
               'FaceColor', [0, 0.2, 0.38], 'FaceAlpha', 0.7, 'EdgeColor', 'black');
    hold on;
    yline(yieldStrength, '--', 'Color', [0.99, 0.71, 0.08], 'LineWidth', 2, ...
          'Label', sprintf('Yield Strength (%d MPa)', yieldStrength));
    xlabel('Load Case');
    ylabel('von Mises Stress (MPa)');
    title('von Mises Stress Analysis');
    set(gca, 'XTick', 1:length(loadCases), 'XTickLabel', loadCases);
    xtickangle(45);
    grid on;
    grid('alpha', 0.3);
    % Add value labels on bars
    for i = 1:length(bars.YData)
        text(i, bars.YData(i) + 5, sprintf('%.1f', bars.YData(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    % Plot 2: Safety factors
    subplot(1, 2, 2);
    safetyFactors = yieldStrength ./ vonMisesStresses;
    colors = repmat([0, 0.2, 0.38], length(safetyFactors), 1);
    colors(safetyFactors <= 1.0, :) = repmat([1, 0, 0], sum(safetyFactors <= 1.0), 1);
    bars2 = bar(1:length(loadCases), safetyFactors, 'FaceAlpha', 0.7, 'EdgeColor', 'black');
    for i = 1:length(safetyFactors)
        bars2.CData(i,:) = colors(i,:);
    end
    hold on;
    yline(1.0, '--', 'Color', [0.99, 0.71, 0.08], 'LineWidth', 2, ...
          'Label', 'Safety Limit (SF = 1.0)');
    xlabel('Load Case');
    ylabel('Safety Factor');
    title('Safety Factor Analysis');
    set(gca, 'XTick', 1:length(loadCases), 'XTickLabel', loadCases);
    xtickangle(45);
    grid on;
    grid('alpha', 0.3);
    % Add value labels on bars
    for i = 1:length(safetyFactors)
        text(i, safetyFactors(i) + 0.05, sprintf('%.2f', safetyFactors(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    sgtitle('Stress Analysis Results', 'FontSize', 16, 'FontWeight', 'bold');
end
function setBerkeleyDefaults()
    % Set Berkeley visual defaults
    set(0, 'DefaultFigureColor', 'white');
    set(0, 'DefaultAxesColor', 'white');
    set(0, 'DefaultAxesFontSize', 10);
    set(0, 'DefaultAxesGridAlpha', 0.3);
    set(0, 'DefaultAxesBox', 'on');
end