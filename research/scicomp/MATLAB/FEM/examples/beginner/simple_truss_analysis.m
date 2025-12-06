function simple_truss_analysis()
    % Simple Truss Analysis - Beginner FEM Example (MATLAB)
    %
    % This example demonstrates basic finite element analysis of a simple truss structure.
    % It covers fundamental concepts including mesh creation, boundary conditions,
    % loading, and result visualization.
    %
    % Learning Objectives:
    % - Understand basic FEM workflow
    % - Create simple 1D truss elements
    % - Apply boundary conditions and loads
    % - Interpret displacement and stress results
    fprintf('Simple Truss Analysis - Beginner FEM Example\n');
    fprintf('%s\n', repmat('=', 1, 50));
    fprintf('This example analyzes a 3-member truss under point loading\n');
    fprintf('Learning: Basic FEM workflow, boundary conditions, result interpretation\n\n');
    % Add path to core FEM classes
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'core'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'Elasticity', 'core'));
    % Set Berkeley visual defaults
    setBerkeleyDefaults();
    % Create material
    steel = createSteelMaterial();
    % Create simple truss mesh
    mesh = createSimpleTruss();
    fprintf('Mesh created: %d nodes, %d elements\n', mesh.getNumNodes(), mesh.getNumElements());
    % Visualize mesh geometry
    plotTrussGeometry(mesh);
    % Setup assembly system
    assembly = setupTrussAssembly(mesh, steel);
    % Apply boundary conditions and loads
    applyBoundaryConditions(assembly);
    applyLoads(assembly);
    % Solve the system
    displacement = solveTrussSystem(assembly);
    % Analyze results
    analyzeResults(assembly, displacement, steel);
    % Visualize deformed shape
    plotDeformedShape(mesh, assembly, displacement);
    fprintf('\n%s\n', repmat('=', 1, 50));
    fprintf('Truss Analysis Complete!\n');
    fprintf('Key Learning Points:\n');
    fprintf('• FEM converts continuous problems to discrete matrix equations\n');
    fprintf('• Boundary conditions constrain the structure\n');
    fprintf('• Loads create internal forces and deformations\n');
    fprintf('• Results must be interpreted in engineering context\n');
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
    fprintf('Density: %d kg/m³\n', density);
end
function mesh = createSimpleTruss()
    % Create a simple 3-member truss structure
    mesh = Mesh();
    mesh.setDimension(2);  % 2D truss
    % Define nodes (x, y coordinates in meters)
    nodes = [
        0.0, 0.0;    % Node 1: Left support
        1.0, 1.0;    % Node 2: Top left
        2.0, 1.0;    % Node 3: Top right
        3.0, 0.0     % Node 4: Right support
    ];
    % Add nodes to mesh
    for i = 1:size(nodes, 1)
        node = Node(i, nodes(i, :));
        mesh.addNode(node);
    end
    % Define elements (truss members)
    % Each element connects two nodes
    crossSectionArea = 0.01;  % 100 cm² = 0.01 m²
    elements = [
        1, 2;  % Element 1: Left diagonal member
        2, 3;  % Element 2: Top horizontal member
        3, 4   % Element 3: Right diagonal member
    ];
    % Add elements to mesh (material will be set in assembly)
    for i = 1:size(elements, 1)
        mesh.addElement('bar1d', elements(i, :), [], crossSectionArea);
    end
end
function plotTrussGeometry(mesh)
    % Plot the truss geometry
    figure('Position', [100, 100, 800, 600]);
    % Plot mesh
    mesh.plotMesh('showNodeIds', true, 'showElementIds', false, ...
                  'nodeColor', [0.99, 0.71, 0.08], 'elementColor', [0, 0.2, 0.38]);
    % Add support symbols
    addSupportSymbols();
    % Add load arrows (for visualization only)
    addLoadArrows();
    title('Simple Truss Geometry', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    grid on;
    axis equal;
    % Add legend
    hold on;
    plot(NaN, NaN, 'o', 'Color', [0.99, 0.71, 0.08], 'MarkerSize', 8, 'MarkerFaceColor', [0.99, 0.71, 0.08], 'DisplayName', 'Joints');
    plot(NaN, NaN, '-', 'Color', [0, 0.2, 0.38], 'LineWidth', 2, 'DisplayName', 'Members');
    legend('Location', 'best');
end
function addSupportSymbols()
    % Add support symbols to the plot
    supportSize = 0.1;
    % Left support (pin)
    plot([0, 0], [-supportSize, 0], 'k-', 'LineWidth', 4);
    plot([-supportSize, supportSize], [-supportSize, -supportSize], 'k-', 'LineWidth', 2);
    % Right support (roller)
    plot([3, 3], [-supportSize, 0], 'k-', 'LineWidth', 4);
    circle = viscircles([3, -supportSize/2], supportSize/3, 'Color', 'k', 'LineWidth', 2);
end
function addLoadArrows()
    % Add load arrows for visualization
    arrowLength = 0.3;
    % Load at node 2 (10 kN downward)
    quiver(1.0, 1.0, 0, -arrowLength, 'r', 'LineWidth', 3, 'MaxHeadSize', 0.2);
    text(1.0, 1.0 - arrowLength - 0.1, '10 kN', 'HorizontalAlignment', 'center', 'Color', 'red', 'FontWeight', 'bold');
    % Load at node 3 (5 kN downward)
    quiver(2.0, 1.0, 0, -arrowLength/2, 'r', 'LineWidth', 3, 'MaxHeadSize', 0.3);
    text(2.0, 1.0 - arrowLength/2 - 0.1, '5 kN', 'HorizontalAlignment', 'center', 'Color', 'red', 'FontWeight', 'bold');
end
function assembly = setupTrussAssembly(mesh, material)
    % Setup assembly system for truss analysis
    fprintf('\nSetting up assembly system...\n');
    % Create global assembly (simplified version)
    assembly = struct();
    assembly.mesh = mesh;
    assembly.material = material;
    % Setup DOF mapping (2 DOF per node in 2D)
    numNodes = mesh.getNumNodes();
    assembly.numDOF = 2 * numNodes;
    % DOF mapping: node i has DOFs [2*i-1, 2*i] for [x, y]
    assembly.dofMap = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    for i = 1:numNodes
        assembly.dofMap(i) = [2*i-1, 2*i];
    end
    % Assemble global stiffness matrix
    assembly.K = assembleGlobalStiffness(mesh, material);
    fprintf('Global stiffness matrix: %dx%d\n', size(assembly.K));
end
function K_global = assembleGlobalStiffness(mesh, material)
    % Assemble global stiffness matrix
    numNodes = mesh.getNumNodes();
    numDOF = 2 * numNodes;
    K_global = zeros(numDOF, numDOF);
    % Get element IDs
    elementIds = cell2mat(mesh.elements.keys);
    for i = 1:length(elementIds)
        element = mesh.getElement(elementIds(i));
        % Create element with material
        nodes = element.getNodes();
        area = element.getArea();
        femElement = LinearBar1D(elementIds(i), nodes, material, area);
        % Get element stiffness matrix
        K_element = femElement.stiffnessMatrix();
        % Get node IDs
        nodeIds = [nodes{1}.getId(), nodes{2}.getId()];
        % Transform to global coordinates and assemble
        K_global_element = transformBarStiffness(K_element, nodes);
        % Assembly into global matrix
        dofs = [2*nodeIds(1)-1, 2*nodeIds(1), 2*nodeIds(2)-1, 2*nodeIds(2)];
        K_global(dofs, dofs) = K_global(dofs, dofs) + K_global_element;
    end
end
function K_global = transformBarStiffness(K_local, nodes)
    % Transform bar element stiffness to global coordinates
    % Calculate element direction
    coords1 = nodes{1}.getCoordinates();
    coords2 = nodes{2}.getCoordinates();
    dx = coords2(1) - coords1(1);
    dy = coords2(2) - coords1(2);
    length = sqrt(dx^2 + dy^2);
    % Direction cosines
    c = dx / length;  % cos(theta)
    s = dy / length;  % sin(theta)
    % Transformation matrix
    T = [c, s, 0, 0;
         0, 0, c, s];
    % Transform: K_global = T' * K_local * T
    K_global = T' * K_local * T;
end
function applyBoundaryConditions(assembly)
    % Apply boundary conditions (supports)
    fprintf('\nApplying boundary conditions...\n');
    % Node 1: Fixed in both directions (pin support)
    % Node 4: Fixed in y-direction only (roller support)
    % Store boundary conditions
    assembly.constrainedDOF = [1, 2, 8];  % DOFs 1,2 (node 1), DOF 8 (node 4-y)
    assembly.prescribedValues = [0, 0, 0];  % All prescribed to zero
    fprintf('Applied constraints to %d DOFs\n', length(assembly.constrainedDOF));
end
function applyLoads(assembly)
    % Apply loads to the truss
    fprintf('Applying loads...\n');
    % Create load vector
    assembly.F = zeros(assembly.numDOF, 1);
    % Node 2: 10 kN downward (DOF 4)
    assembly.F(4) = -10000;  % Negative for downward
    % Node 3: 5 kN downward (DOF 6)
    assembly.F(6) = -5000;   % Negative for downward
    fprintf('Applied loads:\n');
    fprintf('  Node 2: 10 kN downward\n');
    fprintf('  Node 3: 5 kN downward\n');
end
function displacement = solveTrussSystem(assembly)
    % Solve the truss system
    fprintf('\nSolving system of equations...\n');
    % Apply boundary conditions using penalty method
    K_modified = assembly.K;
    F_modified = assembly.F;
    penalty = 1e12 * max(max(abs(assembly.K)));
    for i = 1:length(assembly.constrainedDOF)
        dof = assembly.constrainedDOF(i);
        value = assembly.prescribedValues(i);
        K_modified(dof, dof) = K_modified(dof, dof) + penalty;
        F_modified(dof) = F_modified(dof) + penalty * value;
    end
    % Solve system
    displacement = K_modified \ F_modified;
    fprintf('System solved successfully\n');
    fprintf('Maximum displacement: %.6f mm\n', max(abs(displacement)) * 1000);
end
function analyzeResults(assembly, displacement, material)
    % Analyze and display results
    fprintf('\nDisplacement Results:\n');
    fprintf('%s\n', repmat('-', 1, 30));
    numNodes = assembly.mesh.getNumNodes();
    for nodeId = 1:numNodes
        dofs = assembly.dofMap(nodeId);
        ux = displacement(dofs(1));
        uy = displacement(dofs(2));
        magnitude = sqrt(ux^2 + uy^2);
        fprintf('Node %d: dx = %7.3f mm, dy = %7.3f mm, |d| = %7.3f mm\n', ...
                nodeId, ux*1000, uy*1000, magnitude*1000);
    end
    % Calculate member forces and stresses
    fprintf('\nMember Forces and Stresses:\n');
    fprintf('%s\n', repmat('-', 1, 40));
    elementIds = cell2mat(assembly.mesh.elements.keys);
    yieldStrength = 250e6;  % Typical mild steel yield strength
    for i = 1:length(elementIds)
        element = assembly.mesh.getElement(elementIds(i));
        nodes = element.getNodes();
        nodeIds = [nodes{1}.getId(), nodes{2}.getId()];
        % Get nodal displacements
        dofs = [assembly.dofMap(nodeIds(1)), assembly.dofMap(nodeIds(2))];
        u_element = displacement(dofs);
        % Create FE element and calculate stress
        area = element.getArea();
        femElement = LinearBar1D(elementIds(i), nodes, material, area);
        % Transform displacements to local coordinates
        coords1 = nodes{1}.getCoordinates();
        coords2 = nodes{2}.getCoordinates();
        dx = coords2(1) - coords1(1);
        dy = coords2(2) - coords1(2);
        length = sqrt(dx^2 + dy^2);
        c = dx / length;
        s = dy / length;
        % Local displacements
        u_local = [c*u_element(1) + s*u_element(2); c*u_element(3) + s*u_element(4)];
        % Calculate stress and force
        stress = femElement.computeStress(u_local);
        force = femElement.internalForce(u_local);
        safetyFactor = yieldStrength / abs(stress);
        stressType = 'Tension';
        if stress < 0
            stressType = 'Compression';
        end
        fprintf('Element %d: Force = %8.1f N, Stress = %6.1f MPa (%s), SF = %.1f\n', ...
                elementIds(i), force, stress/1e6, stressType, safetyFactor);
    end
    % Check equilibrium
    reactions = calculateReactions(assembly, displacement);
    fprintf('\nReaction Forces:\n');
    fprintf('%s\n', repmat('-', 1, 20));
    fprintf('Node 1: Rx = %8.1f N, Ry = %8.1f N\n', reactions(1), reactions(2));
    fprintf('Node 4: Rx = %8.1f N, Ry = %8.1f N\n', reactions(7), reactions(8));
    totalReactionY = reactions(2) + reactions(8);
    totalAppliedY = -15000;  % Total applied load
    equilibriumError = abs(totalReactionY - totalAppliedY);
    fprintf('Equilibrium check (should be ~0): %.2e N\n', equilibriumError);
end
function reactions = calculateReactions(assembly, displacement)
    % Calculate reaction forces
    internalForces = assembly.K * displacement;
    reactions = internalForces - assembly.F;
end
function plotDeformedShape(mesh, assembly, displacement)
    % Plot deformed shape
    figure('Position', [200, 200, 900, 600]);
    % Plot undeformed shape
    mesh.plotMesh('showNodeIds', false, 'showElementIds', false, ...
                  'nodeColor', [0.5, 0.5, 0.5], 'elementColor', [0.5, 0.5, 0.5]);
    hold on;
    % Plot deformed shape with scaling
    scaleFactor = 1000;  % Amplify displacements for visibility
    numNodes = mesh.getNumNodes();
    deformedCoords = zeros(numNodes, 2);
    for nodeId = 1:numNodes
        node = mesh.getNode(nodeId);
        originalCoords = node.getCoordinates();
        dofs = assembly.dofMap(nodeId);
        deformedCoords(nodeId, :) = originalCoords + scaleFactor * [displacement(dofs(1)), displacement(dofs(2))];
    end
    % Plot deformed elements
    elementIds = cell2mat(mesh.elements.keys);
    for i = 1:length(elementIds)
        element = mesh.getElement(elementIds(i));
        nodes = element.getNodes();
        nodeIds = [nodes{1}.getId(), nodes{2}.getId()];
        x_def = [deformedCoords(nodeIds(1), 1), deformedCoords(nodeIds(2), 1)];
        y_def = [deformedCoords(nodeIds(1), 2), deformedCoords(nodeIds(2), 2)];
        plot(x_def, y_def, '-', 'Color', [0, 0.2, 0.38], 'LineWidth', 3);
    end
    % Plot deformed nodes
    scatter(deformedCoords(:, 1), deformedCoords(:, 2), 100, [0.99, 0.71, 0.08], 'filled', 'o', 'MarkerEdgeColor', 'black');
    title(sprintf('Deformed Shape (Scale Factor: %d)', scaleFactor), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    % Add legend
    plot(NaN, NaN, '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2, 'DisplayName', 'Undeformed');
    plot(NaN, NaN, '-', 'Color', [0, 0.2, 0.38], 'LineWidth', 3, 'DisplayName', 'Deformed');
    legend('Location', 'best');
    grid on;
    axis equal;
end
function setBerkeleyDefaults()
    % Set Berkeley visual defaults
    set(0, 'DefaultFigureColor', 'white');
    set(0, 'DefaultAxesColor', 'white');
    set(0, 'DefaultAxesFontSize', 10);
    set(0, 'DefaultAxesGridAlpha', 0.3);
    set(0, 'DefaultAxesBox', 'on');
end