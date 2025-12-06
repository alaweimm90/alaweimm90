function basic_operations()
    % BASIC_OPERATIONS Basic Linear Algebra Operations - Beginner Example
    %
    % This example demonstrates fundamental linear algebra operations including
    % vector operations, matrix arithmetic, and basic problem solving.
    %
    % Learning Objectives:
    % - Understand vector and matrix representations
    % - Perform basic arithmetic operations
    % - Compute norms, dot products, and cross products
    % - Solve simple linear systems
    % - Visualize geometric interpretations
    fprintf('Basic Linear Algebra Operations - Beginner Example\n');
    fprintf('=======================================================\n');
    fprintf('This example covers fundamental vector and matrix operations\n');
    fprintf('Learning: Basic operations, geometric interpretation, simple systems\n\n');
    % Add path to core functions
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'core'));
    % Set up Berkeley color scheme
    berkeleyBlue = [0, 50, 98] / 255;
    californiaGold = [253, 181, 21] / 255;
    % Vector operations demonstration
    demonstrateVectorOperations();
    % Matrix operations demonstration
    demonstrateMatrixOperations();
    % Geometric interpretation
    demonstrateGeometricConcepts();
    % Simple linear systems
    demonstrateLinearSystems();
    % Real-world application
    physicsApplicationExample();
    fprintf('\n=======================================================\n');
    fprintf('Basic Linear Algebra Complete!\n');
    fprintf('Key Learning Points:\n');
    fprintf('• Vectors represent quantities with magnitude and direction\n');
    fprintf('• Matrices can represent transformations and systems of equations\n');
    fprintf('• Dot products measure alignment, cross products measure perpendicularity\n');
    fprintf('• Linear systems appear everywhere in science and engineering\n');
end
function demonstrateVectorOperations()
    % Demonstrate basic vector operations
    fprintf('Vector Operations\n');
    fprintf('====================\n');
    % Create example vectors
    fprintf('1. Creating Vectors\n');
    fprintf('------------------\n');
    u = [3; 4];
    v = [1; 2];
    w = [2; -1; 3];  % 3D vector
    fprintf('2D Vector u: [%g, %g]\n', u(1), u(2));
    fprintf('2D Vector v: [%g, %g]\n', v(1), v(2));
    fprintf('3D Vector w: [%g, %g, %g]\n', w(1), w(2), w(3));
    % Basic arithmetic
    fprintf('\n2. Vector Arithmetic\n');
    fprintf('------------------\n');
    % Addition and subtraction
    sumUV = VectorOperations.add(u, v);
    diffUV = VectorOperations.subtract(u, v);
    fprintf('u + v = [%g, %g]\n', sumUV(1), sumUV(2));
    fprintf('u - v = [%g, %g]\n', diffUV(1), diffUV(2));
    % Scalar multiplication
    scaledU = VectorOperations.scalarMultiply(2.5, u);
    fprintf('2.5 * u = [%g, %g]\n', scaledU(1), scaledU(2));
    % Vector magnitude and normalization
    fprintf('\n3. Vector Magnitude and Direction\n');
    fprintf('--------------------------------\n');
    magU = VectorOperations.magnitude(u);
    magV = VectorOperations.magnitude(v);
    fprintf('Magnitude of u: %.3f\n', magU);
    fprintf('Magnitude of v: %.3f\n', magV);
    % Unit vectors
    uUnit = VectorOperations.normalize(u);
    vUnit = VectorOperations.normalize(v);
    fprintf('Unit vector in direction of u: [%.3f, %.3f]\n', uUnit(1), uUnit(2));
    fprintf('Unit vector in direction of v: [%.3f, %.3f]\n', vUnit(1), vUnit(2));
    fprintf('Magnitude of normalized u: %.3f\n', VectorOperations.magnitude(uUnit));
    % Dot product
    fprintf('\n4. Dot Product (measures alignment)\n');
    fprintf('-----------------------------------\n');
    dotUV = VectorOperations.dotProduct(u, v);
    angleUV = VectorOperations.angleBetween(u, v, true);
    fprintf('u · v = %g\n', dotUV);
    fprintf('Angle between u and v: %.1f degrees\n', angleUV);
    % Demonstrate orthogonal vectors
    orthogonalToU = [-u(2); u(1)];  % Rotate 90 degrees
    dotOrthogonal = VectorOperations.dotProduct(u, orthogonalToU);
    fprintf('Vector orthogonal to u: [%g, %g]\n', orthogonalToU(1), orthogonalToU(2));
    fprintf('u · orthogonal_vector = %g (should be 0)\n', dotOrthogonal);
    % Cross product (3D)
    fprintf('\n5. Cross Product (3D vectors)\n');
    fprintf('----------------------------\n');
    a3d = [1; 0; 0];  % x-axis
    b3d = [0; 1; 0];  % y-axis
    crossAB = VectorOperations.crossProduct(a3d, b3d);
    fprintf('x-axis × y-axis = [%g, %g, %g] (should be z-axis)\n', ...
            crossAB(1), crossAB(2), crossAB(3));
    % Verify orthogonality
    fprintf('Cross product is orthogonal to both inputs:\n');
    fprintf('  (x × y) · x = %g\n', VectorOperations.dotProduct(crossAB, a3d));
    fprintf('  (x × y) · y = %g\n', VectorOperations.dotProduct(crossAB, b3d));
    % Different norms
    fprintf('\n6. Different Vector Norms\n');
    fprintf('------------------------\n');
    testVector = [3; 4; 0];
    l1Norm = norm(testVector, 1);
    l2Norm = norm(testVector, 2);
    infNorm = norm(testVector, inf);
    fprintf('Vector: [%g, %g, %g]\n', testVector(1), testVector(2), testVector(3));
    fprintf('L1 norm (Manhattan): %g\n', l1Norm);
    fprintf('L2 norm (Euclidean): %g\n', l2Norm);
    fprintf('L∞ norm (Maximum): %g\n', infNorm);
end
function demonstrateMatrixOperations()
    % Demonstrate basic matrix operations
    fprintf('\n\nMatrix Operations\n');
    fprintf('====================\n');
    % Create example matrices
    fprintf('1. Creating Matrices\n');
    fprintf('------------------\n');
    A = [2, 1; 1, 3];
    B = [1, 2; 3, 1];
    C = [1, 2, 1; 0, 1, 2];  % Non-square
    fprintf('Matrix A:\n');
    disp(A);
    fprintf('Matrix B:\n');
    disp(B);
    fprintf('Matrix C (2×3):\n');
    disp(C);
    % Basic arithmetic
    fprintf('2. Matrix Arithmetic\n');
    fprintf('------------------\n');
    % Addition and subtraction
    sumAB = MatrixOperations.matrixAdd(A, B);
    diffAB = MatrixOperations.matrixSubtract(A, B);
    fprintf('A + B:\n');
    disp(sumAB);
    fprintf('A - B:\n');
    disp(diffAB);
    % Matrix multiplication
    productAB = MatrixOperations.matrixMultiply(A, B);
    productAC = MatrixOperations.matrixMultiply(A, C);
    fprintf('A × B:\n');
    disp(productAB);
    fprintf('A × C:\n');
    disp(productAC);
    % Matrix properties
    fprintf('3. Matrix Properties\n');
    fprintf('------------------\n');
    traceA = MatrixOperations.trace(A);
    detA = MatrixOperations.determinant(A);
    rankA = MatrixOperations.rank(A);
    fprintf('Trace of A: %g\n', traceA);
    fprintf('Determinant of A: %g\n', detA);
    fprintf('Rank of A: %g\n', rankA);
    % Transpose
    AT = MatrixOperations.transpose(A);
    fprintf('\nA transpose:\n');
    disp(AT);
    % Matrix norms
    frobeniusNorm = MatrixOperations.frobeniusNorm(A);
    spectralNorm = MatrixOperations.spectralNorm(A);
    conditionNumber = MatrixOperations.conditionNumber(A);
    fprintf('Frobenius norm of A: %.3f\n', frobeniusNorm);
    fprintf('Spectral norm of A: %.3f\n', spectralNorm);
    fprintf('Condition number of A: %.3f\n', conditionNumber);
    % Matrix powers
    fprintf('\n4. Matrix Powers\n');
    fprintf('---------------\n');
    ASquared = MatrixOperations.matrixPower(A, 2);
    ACubed = MatrixOperations.matrixPower(A, 3);
    fprintf('A²:\n');
    disp(ASquared);
    fprintf('A³:\n');
    disp(ACubed);
end
function demonstrateGeometricConcepts()
    % Demonstrate geometric interpretation of linear algebra
    fprintf('\n\nGeometric Concepts\n');
    fprintf('====================\n');
    % Vector projections
    fprintf('1. Vector Projections\n');
    fprintf('-------------------\n');
    u = [4; 2];
    v = [3; 0];  % Along x-axis
    proj = VectorOperations.project(u, v);
    rejection = VectorOperations.reject(u, v);
    fprintf('Vector u: [%g, %g]\n', u(1), u(2));
    fprintf('Vector v: [%g, %g]\n', v(1), v(2));
    fprintf('Projection of u onto v: [%g, %g]\n', proj(1), proj(2));
    fprintf('Rejection (orthogonal component): [%g, %g]\n', rejection(1), rejection(2));
    % Verify orthogonality
    dotCheck = VectorOperations.dotProduct(proj, rejection);
    fprintf('Projection · Rejection = %.10f (should be 0)\n', dotCheck);
    % Create visualization
    figure('Name', 'Vector Operations', 'Position', [100, 100, 1000, 400]);
    subplot(1, 2, 1);
    % Plot vectors
    quiver(0, 0, u(1), u(2), 0, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.2);
    hold on;
    quiver(0, 0, v(1), v(2), 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.2);
    quiver(0, 0, proj(1), proj(2), 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.2);
    % Draw projection lines
    plot([u(1), proj(1)], [u(2), proj(2)], 'k--', 'LineWidth', 1);
    xlim([-1, 5]);
    ylim([-1, 3]);
    grid on;
    xlabel('X');
    ylabel('Y');
    title('Vector Projection');
    legend({'u', 'v', 'proj_v(u)', 'rejection'}, 'Location', 'best');
    axis equal;
    % 3D cross product visualization
    subplot(1, 2, 2);
    % 3D vectors
    a = [1; 0; 0];
    b = [0; 1; 0];
    c = VectorOperations.crossProduct(a, b);
    % Plot vectors
    quiver3(0, 0, 0, a(1), a(2), a(3), 0, 'r', 'LineWidth', 2);
    hold on;
    quiver3(0, 0, 0, b(1), b(2), b(3), 0, 'b', 'LineWidth', 2);
    quiver3(0, 0, 0, c(1), c(2), c(3), 0, 'g', 'LineWidth', 2);
    xlim([0, 1.2]);
    ylim([0, 1.2]);
    zlim([0, 1.2]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Cross Product in 3D');
    legend({'a', 'b', 'a × b'}, 'Location', 'best');
    view(45, 30);
    % Gram-Schmidt orthogonalization
    fprintf('\n2. Gram-Schmidt Orthogonalization\n');
    fprintf('----------------------------------\n');
    % Start with linearly independent but not orthogonal vectors
    v1 = [1; 1; 0];
    v2 = [1; 0; 1];
    v3 = [0; 1; 1];
    vectors = {v1, v2, v3};
    orthogonalVectors = VectorOperations.gramSchmidtOrthogonalization(vectors, true);
    fprintf('Original vectors:\n');
    for i = 1:length(vectors)
        fprintf('  v%d: [%g, %g, %g]\n', i, vectors{i}(1), vectors{i}(2), vectors{i}(3));
    end
    fprintf('\nOrthogonal vectors:\n');
    for i = 1:length(orthogonalVectors)
        v = orthogonalVectors{i};
        fprintf('  u%d: [%.3f, %.3f, %.3f]\n', i, v(1), v(2), v(3));
        fprintf('  |u%d|: %.3f\n', i, VectorOperations.magnitude(v));
    end
    % Verify orthogonality
    fprintf('\nOrthogonality check:\n');
    for i = 1:length(orthogonalVectors)
        for j = i+1:length(orthogonalVectors)
            dotProd = VectorOperations.dotProduct(orthogonalVectors{i}, orthogonalVectors{j});
            fprintf('  u%d · u%d = %.10f\n', i, j, dotProd);
        end
    end
end
function demonstrateLinearSystems()
    % Demonstrate solving linear systems
    fprintf('\n\nLinear Systems\n');
    fprintf('===============\n');
    fprintf('1. Simple 2×2 System\n');
    fprintf('-------------------\n');
    % System: 2x + y = 5
    %         x + 3y = 6
    A = [2, 1; 1, 3];
    b = [5; 6];
    fprintf('System of equations:\n');
    fprintf('  2x + y = 5\n');
    fprintf('  x + 3y = 6\n');
    fprintf('\nMatrix form: A x = b\n');
    fprintf('A =\n');
    disp(A);
    fprintf('b = [%g; %g]\n', b(1), b(2));
    % Solve using LU decomposition
    result = LinearSystems.luSolve(A, b);
    if result.success
        x = result.solution(1);
        y = result.solution(2);
        fprintf('\nSolution: x = %.3f, y = %.3f\n', x, y);
        % Verify solution
        verification = A * result.solution;
        fprintf('Verification: A × solution = [%.3f; %.3f]\n', verification(1), verification(2));
        fprintf('Original b = [%g; %g]\n', b(1), b(2));
        fprintf('Residual = %.2e\n', result.residualNorm);
    else
        fprintf('Failed to solve system\n');
    end
    fprintf('\n2. Overdetermined System (Least Squares)\n');
    fprintf('----------------------------------------\n');
    % More equations than unknowns - find best fit
    AOver = [1, 1; 1, 2; 1, 3; 1, 4];
    bOver = [2.1; 2.9; 4.2; 4.8];  % Noisy line data
    fprintf('Overdetermined system (4 equations, 2 unknowns):\n');
    fprintf('Finding line y = mx + c that best fits points\n');
    fprintf('Points: (1, 2.1), (2, 2.9), (3, 4.2), (4, 4.8)\n');
    % Solve using QR (least squares)
    result = LinearSystems.qrSolve(AOver, bOver);
    if result.success
        c = result.solution(1);
        m = result.solution(2);
        fprintf('\nBest fit line: y = %.3fx + %.3f\n', m, c);
        fprintf('Residual norm: %.3f\n', result.residualNorm);
        % Plot the fit
        figure('Name', 'Least Squares Fitting', 'Position', [200, 200, 600, 400]);
        xData = [1; 2; 3; 4];
        yData = bOver;
        scatter(xData, yData, 50, [0, 50, 98]/255, 'filled');
        hold on;
        xLine = linspace(0.5, 4.5, 100);
        yLine = m * xLine + c;
        plot(xLine, yLine, 'Color', [253, 181, 21]/255, 'LineWidth', 2);
        % Show residuals
        yPredicted = AOver * result.solution;
        for i = 1:length(xData)
            plot([xData(i), xData(i)], [yData(i), yPredicted(i)], 'r--', 'LineWidth', 1);
        end
        xlabel('x');
        ylabel('y');
        title('Least Squares Line Fitting');
        legend({'Data points', sprintf('Best fit: y = %.3fx + %.3f', m, c), 'Residuals'}, ...
               'Location', 'best');
        grid on;
    end
end
function physicsApplicationExample()
    % Demonstrate linear algebra in physics applications
    fprintf('\n\nPhysics Application: Force Balance\n');
    fprintf('===================================\n');
    fprintf('Problem: Find tension forces in cables supporting a mass\n');
    fprintf('Three cables support a 100 kg mass at equilibrium\n');
    fprintf('Cable 1: 30° above horizontal\n');
    fprintf('Cable 2: 45° above horizontal\n');
    fprintf('Cable 3: Vertical\n');
    % Force balance equations:
    % Sum of forces in x-direction: T1*cos(30°) - T2*cos(45°) = 0
    % Sum of forces in y-direction: T1*sin(30°) + T2*sin(45°) + T3 = mg
    % Cable 3 is vertical, so T3 provides no horizontal force
    mass = 100;  % kg
    g = 9.81;    % m/s²
    weight = mass * g;
    % Angles in radians
    angle1 = deg2rad(30);
    angle2 = deg2rad(45);
    % Set up system of equations
    % [cos(30°)  -cos(45°)   0] [T1]   [0]
    % [sin(30°)   sin(45°)   1] [T2] = [mg]
    %                              [T3]
    A = [cos(angle1), -cos(angle2), 0;
         sin(angle1),  sin(angle2), 1];
    b = [0; weight];
    fprintf('\nWeight = %.1f N\n', weight);
    fprintf('\nForce balance equations:\n');
    fprintf('Horizontal: T₁cos(30°) - T₂cos(45°) = 0\n');
    fprintf('Vertical:   T₁sin(30°) + T₂sin(45°) + T₃ = mg\n');
    % This is an underdetermined system (2 equations, 3 unknowns)
    % Add constraint that T3 = 0 (no vertical cable) to make it solvable
    fprintf('\nAssuming no vertical cable (T₃ = 0):\n');
    AReduced = A(:, 1:2);  % Remove T3 column
    result = LinearSystems.luSolve(AReduced, b);
    if result.success
        T1 = result.solution(1);
        T2 = result.solution(2);
        fprintf('\nTension forces:\n');
        fprintf('Cable 1 (30°): T₁ = %.1f N\n', T1);
        fprintf('Cable 2 (45°): T₂ = %.1f N\n', T2);
        % Verify force balance
        Fx = T1 * cos(angle1) - T2 * cos(angle2);
        Fy = T1 * sin(angle1) + T2 * sin(angle2);
        fprintf('\nVerification:\n');
        fprintf('Net horizontal force: %.2e N (should be 0)\n', Fx);
        fprintf('Net vertical force: %.1f N (should equal weight)\n', Fy);
        fprintf('Force balance error: %.2e N\n', abs(Fy - weight));
        % Visualization
        figure('Name', 'Force Analysis', 'Position', [300, 300, 800, 400]);
        % Plot force diagram
        subplot(1, 2, 1);
        % Mass at origin
        scatter(0, 0, 200, 'k', 'filled', 's');
        hold on;
        % Cable forces
        scale = 0.002;  % Scale for visualization
        % Cable 1
        F1x = T1 * cos(angle1) * scale;
        F1y = T1 * sin(angle1) * scale;
        quiver(0, 0, F1x, F1y, 0, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.3);
        % Cable 2
        F2x = -T2 * cos(angle2) * scale;
        F2y = T2 * sin(angle2) * scale;
        quiver(0, 0, F2x, F2y, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.3);
        % Weight
        quiver(0, 0, 0, -weight * scale, 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.3);
        xlim([-0.4, 0.4]);
        ylim([-0.3, 0.3]);
        xlabel('Horizontal Position');
        ylabel('Vertical Position');
        title('Force Diagram');
        legend({sprintf('Mass (%.0f kg)', mass), sprintf('T₁ = %.0f N', T1), ...
                sprintf('T₂ = %.0f N', T2), sprintf('Weight = %.0f N', weight)}, ...
               'Location', 'best');
        grid on;
        axis equal;
        % Plot angle diagram
        subplot(1, 2, 2);
        % Draw cables
        cableLength = 1.0;
        x1 = cableLength * cos(angle1);
        y1 = cableLength * sin(angle1);
        x2 = -cableLength * cos(angle2);
        y2 = cableLength * sin(angle2);
        plot([0, x1], [0, y1], 'b-', 'LineWidth', 3);
        hold on;
        plot([0, x2], [0, y2], 'r-', 'LineWidth', 3);
        % Mass
        scatter(0, 0, 200, 'k', 'filled', 's');
        % Angle annotations
        text(0.3, 0.1, sprintf('%.0f°', rad2deg(angle1)), 'FontSize', 12);
        text(-0.3, 0.1, sprintf('%.0f°', rad2deg(angle2)), 'FontSize', 12);
        xlim([-1.2, 1.2]);
        ylim([-0.2, 1.0]);
        xlabel('Horizontal Position');
        ylabel('Vertical Position');
        title('Cable Configuration');
        legend({'Cable 1 (30°)', 'Cable 2 (45°)', 'Mass'}, 'Location', 'best');
        grid on;
        axis equal;
    end
end