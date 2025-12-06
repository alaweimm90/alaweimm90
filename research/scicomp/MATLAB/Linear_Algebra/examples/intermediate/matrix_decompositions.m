function matrix_decompositions()
    % MATRIX_DECOMPOSITIONS Matrix Decompositions - Intermediate Example
    %
    % This example demonstrates matrix decomposition techniques including LU, QR,
    % Cholesky, and SVD decompositions with practical applications.
    %
    % Learning Objectives:
    % - Understand different matrix decomposition methods
    % - Apply decompositions to solve linear systems efficiently
    % - Use SVD for data analysis and dimensionality reduction
    % - Analyze matrix properties through decompositions
    % - Implement numerical algorithms using decompositions
    fprintf('Matrix Decompositions - Intermediate Example\n');
    fprintf('=============================================\n');
    fprintf('This example covers matrix decomposition techniques and applications\n');
    fprintf('Learning: LU, QR, Cholesky, SVD decompositions and their uses\n\n');
    % Add path to core functions
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'core'));
    % Set up Berkeley color scheme
    berkeleyBlue = [0, 50, 98] / 255;
    californiaGold = [253, 181, 21] / 255;
    % LU decomposition demonstration
    demonstrateLUDecomposition();
    % QR decomposition and applications
    demonstrateQRDecomposition();
    % Cholesky decomposition for positive definite matrices
    demonstrateCholeskyDecomposition();
    % SVD and its applications
    demonstrateSVDApplications();
    % Eigenvalue decomposition
    demonstrateEigenvalueDecomposition();
    % Practical application: Image compression using SVD
    imageCompressionExample();
    fprintf('\n=============================================\n');
    fprintf('Matrix Decompositions Complete!\n');
    fprintf('Key Learning Points:\n');
    fprintf('• LU decomposition efficiently solves multiple systems with same matrix\n');
    fprintf('• QR decomposition is robust for least squares problems\n');
    fprintf('• Cholesky decomposition is fastest for positive definite systems\n');
    fprintf('• SVD provides optimal low-rank approximations\n');
    fprintf('• Eigendecomposition reveals matrix spectral properties\n');
end
function demonstrateLUDecomposition()
    % Demonstrate LU decomposition and its applications
    fprintf('LU Decomposition\n');
    fprintf('================\n');
    % Create test matrix
    A = [2, 1, 1; 4, 3, 3; 8, 7, 9];
    fprintf('Matrix A:\n');
    disp(A);
    % Perform LU decomposition
    [P, L, U] = MatrixDecompositions.luDecomposition(A);
    fprintf('LU Decomposition: PA = LU\n');
    fprintf('Permutation matrix P:\n');
    disp(P);
    fprintf('Lower triangular L:\n');
    disp(L);
    fprintf('Upper triangular U:\n');
    disp(U);
    % Verify decomposition
    PA = P * A;
    LU = L * U;
    fprintf('Verification: ||PA - LU|| = %.2e\n', norm(PA - LU, 'fro'));
    % Application: Solve multiple systems with same A
    fprintf('\n1. Solving Multiple Systems Efficiently\n');
    fprintf('--------------------------------------\n');
    % Multiple right-hand sides
    b1 = [4; 10; 24];
    b2 = [1; 2; 3];
    b3 = [0; 1; 0];
    fprintf('Solving Ax = b for multiple b vectors:\n');
    fprintf('b1 = [%g; %g; %g]\n', b1(1), b1(2), b1(3));
    fprintf('b2 = [%g; %g; %g]\n', b2(1), b2(2), b2(3));
    fprintf('b3 = [%g; %g; %g]\n', b3(1), b3(2), b3(3));
    % Solve using LU decomposition
    bs = {b1, b2, b3};
    for i = 1:length(bs)
        result = LinearSystems.luSolve(A, bs{i});
        if result.success
            fprintf('Solution x%d = [%.3f; %.3f; %.3f]\n', i, ...
                    result.solution(1), result.solution(2), result.solution(3));
            fprintf('  Residual norm: %.2e\n', result.residualNorm);
        end
    end
    % Application: Matrix inversion using LU
    fprintf('\n2. Matrix Inversion via LU\n');
    fprintf('-------------------------\n');
    % Solve A * X = I to find A^(-1)
    n = size(A, 1);
    I = eye(n);
    AInv = zeros(n);
    for i = 1:n
        result = LinearSystems.luSolve(A, I(:, i));
        if result.success
            AInv(:, i) = result.solution;
        end
    end
    fprintf('Matrix inverse A^(-1):\n');
    disp(AInv);
    % Verify inversion
    product = A * AInv;
    fprintf('Verification: ||A * A^(-1) - I|| = %.2e\n', norm(product - I, 'fro'));
    % Determinant from LU
    fprintf('\n3. Determinant from LU\n');
    fprintf('--------------------\n');
    detFromLU = prod(diag(U)) * det(P);
    detDirect = det(A);
    fprintf('Determinant from LU: %.6f\n', detFromLU);
    fprintf('Direct determinant:  %.6f\n', detDirect);
    fprintf('Difference: %.2e\n', abs(detFromLU - detDirect));
end
function demonstrateQRDecomposition()
    % Demonstrate QR decomposition and least squares
    fprintf('\n\nQR Decomposition\n');
    fprintf('================\n');
    % Create overdetermined system
    rng(42);
    m = 6; n = 4;
    A = randn(m, n);
    fprintf('Matrix A (%dx%d):\n', m, n);
    disp(A);
    % QR decomposition
    [Q, R] = MatrixDecompositions.qrDecomposition(A, 'economic');
    fprintf('Economic QR: A = QR\n');
    fprintf('Q matrix (%dx%d):\n', size(Q, 1), size(Q, 2));
    disp(Q);
    fprintf('R matrix (%dx%d):\n', size(R, 1), size(R, 2));
    disp(R);
    % Verify decomposition
    QR = Q * R;
    fprintf('Verification: ||A - QR|| = %.2e\n', norm(A - QR, 'fro'));
    % Verify Q orthogonality
    QtQ = Q.' * Q;
    ISmall = eye(size(Q, 2));
    fprintf('Q orthogonality: ||Q^T Q - I|| = %.2e\n', norm(QtQ - ISmall, 'fro'));
    % Application: Least squares problem
    fprintf('\n1. Least Squares Line Fitting\n');
    fprintf('----------------------------\n');
    % Generate noisy data for line fitting
    xData = linspace(0, 5, 10).';
    trueSlope = 2.5;
    trueIntercept = 1.2;
    rng(42);
    noise = 0.3 * randn(length(xData), 1);
    yData = trueSlope * xData + trueIntercept + noise;
    % Set up least squares problem: find line y = mx + c
    % Design matrix: [1, x1; 1, x2; ...; 1, xn]
    AFit = [ones(length(xData), 1), xData];
    fprintf('True line: y = %.1fx + %.1f\n', trueSlope, trueIntercept);
    fprintf('Data points with noise added\n');
    % Solve using QR
    result = LinearSystems.qrSolve(AFit, yData);
    if result.success
        interceptFit = result.solution(1);
        slopeFit = result.solution(2);
        fprintf('Fitted line: y = %.3fx + %.3f\n', slopeFit, interceptFit);
        fprintf('Residual norm: %.3f\n', result.residualNorm);
        % Plot results
        figure('Name', 'QR Least Squares', 'Position', [100, 100, 1000, 400]);
        subplot(1, 2, 1);
        scatter(xData, yData, 50, [0, 50, 98]/255, 'filled');
        hold on;
        xLine = linspace(-0.5, 5.5, 100);
        yTrue = trueSlope * xLine + trueIntercept;
        yFit = slopeFit * xLine + interceptFit;
        plot(xLine, yTrue, '--', 'Color', 'red', 'LineWidth', 2);
        plot(xLine, yFit, 'Color', [253, 181, 21]/255, 'LineWidth', 2);
        % Show residuals
        yPredicted = AFit * result.solution;
        for i = 1:length(xData)
            plot([xData(i), xData(i)], [yData(i), yPredicted(i)], 'Color', [0.5, 0.5, 0.5]);
        end
        xlabel('x');
        ylabel('y');
        title('QR-based Least Squares Fitting');
        legend({'Noisy data', 'True line', 'Fitted line'}, 'Location', 'best');
        grid on;
        % Condition number analysis
        subplot(1, 2, 2);
        condNumbers = [];
        for k = 2:min(9, length(xData))
            ASub = AFit(1:k, :);
            condNumbers(end+1) = cond(ASub); %#ok<AGROW>
        end
        semilogy(2:2+length(condNumbers)-1, condNumbers, 'o-', 'Color', [0, 50, 98]/255);
        xlabel('Number of data points');
        ylabel('Condition number');
        title('Condition Number vs Data Size');
        grid on;
    end
end
function demonstrateCholeskyDecomposition()
    % Demonstrate Cholesky decomposition for positive definite matrices
    fprintf('\n\nCholesky Decomposition\n');
    fprintf('======================\n');
    % Create positive definite matrix
    rng(42);
    ABase = randn(4, 4);
    A = ABase.' * ABase + 0.1 * eye(4);  % Ensure positive definite
    fprintf('Positive definite matrix A:\n');
    disp(A);
    % Verify positive definiteness
    eigenvals = eig(A);
    fprintf('Eigenvalues: [%.3f, %.3f, %.3f, %.3f]\n', eigenvals(1), eigenvals(2), eigenvals(3), eigenvals(4));
    fprintf('All positive: %s\n', mat2str(all(eigenvals > 0)));
    % Cholesky decomposition
    L = MatrixDecompositions.choleskyDecomposition(A, true);
    fprintf('\nCholesky factor L (lower triangular):\n');
    disp(L);
    % Verify decomposition
    LLt = L * L.';
    fprintf('Verification: ||A - LL^T|| = %.2e\n', norm(A - LLt, 'fro'));
    % Application: Efficient solving for positive definite systems
    fprintf('\n1. Efficient System Solving\n');
    fprintf('--------------------------\n');
    b = [1; 2; 3; 4];
    fprintf('Right-hand side: [%g; %g; %g; %g]\n', b(1), b(2), b(3), b(4));
    % Solve using Cholesky
    resultChol = LinearSystems.choleskySolve(A, b);
    % Compare with LU
    resultLU = LinearSystems.luSolve(A, b);
    if resultChol.success && resultLU.success
        fprintf('Cholesky solution: [%.6f; %.6f; %.6f; %.6f]\n', ...
                resultChol.solution(1), resultChol.solution(2), resultChol.solution(3), resultChol.solution(4));
        fprintf('LU solution:       [%.6f; %.6f; %.6f; %.6f]\n', ...
                resultLU.solution(1), resultLU.solution(2), resultLU.solution(3), resultLU.solution(4));
        fprintf('Difference: %.2e\n', norm(resultChol.solution - resultLU.solution));
        fprintf('\nCholesky residual: %.2e\n', resultChol.residualNorm);
        fprintf('LU residual:       %.2e\n', resultLU.residualNorm);
    end
    % Application: Generating correlated random variables
    fprintf('\n2. Generating Correlated Random Variables\n');
    fprintf('-----------------------------------------\n');
    % Desired correlation matrix
    desiredCorr = [1.0, 0.7, 0.3; 0.7, 1.0, 0.5; 0.3, 0.5, 1.0];
    fprintf('Desired correlation matrix:\n');
    disp(desiredCorr);
    % Cholesky factor of correlation matrix
    LCorr = MatrixDecompositions.choleskyDecomposition(desiredCorr, true);
    % Generate uncorrelated samples and transform
    nSamples = 1000;
    rng(42);
    uncorrSamples = randn(3, nSamples);
    corrSamples = LCorr * uncorrSamples;
    % Compute sample correlation
    sampleCorr = corrcoef(corrSamples.');
    fprintf('Sample correlation matrix (%d samples):\n', nSamples);
    disp(sampleCorr);
    fprintf('Error: %.3f\n', norm(sampleCorr - desiredCorr, 'fro'));
    % Visualize
    figure('Name', 'Correlated Random Variables', 'Position', [200, 200, 900, 300]);
    for i = 1:3
        subplot(1, 3, i);
        histogram(corrSamples(i, :), 30, 'FaceColor', [0, 50, 98]/255, 'EdgeColor', 'black', 'FaceAlpha', 0.7);
        title(sprintf('Variable %d', i));
        xlabel('Value');
        ylabel('Frequency');
        grid on;
    end
end
function demonstrateSVDApplications()
    % Demonstrate SVD and its applications
    fprintf('\n\nSingular Value Decomposition (SVD)\n');
    fprintf('==================================\n');
    % Create test matrix
    rng(42);
    A = randn(5, 3);
    fprintf('Matrix A (5x3):\n');
    disp(A);
    % SVD decomposition
    [U, S, V] = MatrixDecompositions.svd(A, false);
    fprintf('SVD: A = U @ diag(s) @ V^T\n');
    fprintf('U matrix (%dx%d):\n', size(U, 1), size(U, 2));
    disp(U);
    fprintf('Singular values: [%.3f, %.3f, %.3f]\n', S(1), S(2), S(3));
    fprintf('V^T matrix (%dx%d):\n', size(V, 1), size(V, 2));
    disp(V.');
    % Verify decomposition
    AReconstructed = U * diag(S) * V.';
    fprintf('Verification: ||A - U*S*V^T|| = %.2e\n', norm(A - AReconstructed, 'fro'));
    % Verify orthogonality
    fprintf('U orthogonality: ||U^T*U - I|| = %.2e\n', norm(U.' * U - eye(size(U, 2)), 'fro'));
    fprintf('V orthogonality: ||V*V^T - I|| = %.2e\n', norm(V * V.' - eye(size(V, 1)), 'fro'));
    % Application 1: Low-rank approximation
    fprintf('\n1. Low-rank Approximation\n');
    fprintf('------------------------\n');
    % Create a matrix with known low-rank structure
    rank2Matrix = [1; 2; 3; 4; 5] * [1, 1, 2] + [2; 1; 1; 2; 1] * [1, 2, 1];
    % Add small amount of noise
    rng(42);
    noisyMatrix = rank2Matrix + 0.1 * randn(size(rank2Matrix));
    fprintf('Original rank-2 matrix + noise:\n');
    disp(noisyMatrix);
    % SVD for denoising
    [UNoise, SNoise, VNoise] = MatrixDecompositions.svd(noisyMatrix, false);
    fprintf('Singular values: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', ...
            SNoise(1), SNoise(2), SNoise(3), SNoise(4), SNoise(5));
    % Reconstruct using only largest singular values
    k = 2;  % Keep only 2 components
    ADenoised = UNoise(:, 1:k) * diag(SNoise(1:k)) * VNoise(:, 1:k).';
    fprintf('Denoised matrix (rank %d):\n', k);
    disp(ADenoised);
    fprintf('Original rank: %d\n', rank(noisyMatrix));
    fprintf('Denoised rank: %d\n', rank(ADenoised));
    fprintf('Frobenius error: %.3f\n', norm(rank2Matrix - ADenoised, 'fro'));
    % Application 2: Principal Component Analysis (PCA)
    fprintf('\n2. Principal Component Analysis\n');
    fprintf('------------------------------\n');
    % Generate 2D data with correlation
    nPoints = 200;
    rng(42);
    % Original data in 2D
    theta = pi / 6;  % 30 degree rotation
    rotation = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    % Uncorrelated data
    dataUncorr = randn(2, nPoints);
    dataUncorr(1, :) = dataUncorr(1, :) * 3;  % Different variances
    dataUncorr(2, :) = dataUncorr(2, :) * 1;
    % Rotate to create correlation
    data = rotation * dataUncorr;
    % Center the data
    dataCentered = data - mean(data, 2);
    fprintf('Data shape: %dx%d\n', size(dataCentered, 1), size(dataCentered, 2));
    fprintf('Data covariance matrix:\n');
    covMatrix = cov(dataCentered.');
    disp(covMatrix);
    % PCA using SVD
    [UPCA, SPCA, VtPCA] = MatrixDecompositions.svd(dataCentered, false);
    % Principal components are columns of Vt.'
    principalComponents = VtPCA.';
    explainedVariance = SPCA.^2 / (nPoints - 1);
    fprintf('Principal components:\n');
    disp(principalComponents);
    fprintf('Explained variance: [%.3f, %.3f]\n', explainedVariance(1), explainedVariance(2));
    fprintf('Explained variance ratio: [%.3f, %.3f]\n', ...
            explainedVariance(1)/sum(explainedVariance), explainedVariance(2)/sum(explainedVariance));
    % Visualize PCA
    figure('Name', 'Principal Component Analysis', 'Position', [300, 300, 800, 400]);
    subplot(1, 2, 1);
    scatter(dataCentered(1, :), dataCentered(2, :), 30, [0, 50, 98]/255, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    % Plot principal components
    center = mean(dataCentered, 2);
    scale = 2 * sqrt(explainedVariance);
    for i = 1:2
        direction = principalComponents(:, i) * scale(i);
        quiver(center(1), center(2), direction(1), direction(2), 0, ...
               'Color', [253, 181, 21]/255, 'LineWidth', 2, 'MaxHeadSize', 0.3);
        text(center(1) + direction(1)*1.1, center(2) + direction(2)*1.1, ...
             sprintf('PC%d', i), 'FontSize', 12, 'Color', [253, 181, 21]/255, 'FontWeight', 'bold');
    end
    xlabel('X');
    ylabel('Y');
    title('Original Data with Principal Components');
    grid on;
    axis equal;
    % Project data onto principal components
    projectedData = VtPCA * dataCentered;
    subplot(1, 2, 2);
    scatter(projectedData(1, :), projectedData(2, :), 30, [0, 50, 98]/255, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('First Principal Component');
    ylabel('Second Principal Component');
    title('Data in Principal Component Space');
    grid on;
    axis equal;
end
function demonstrateEigenvalueDecomposition()
    % Demonstrate eigenvalue decomposition and applications
    fprintf('\n\nEigenvalue Decomposition\n');
    fprintf('========================\n');
    % Create symmetric matrix for real eigenvalues
    rng(42);
    ABase = randn(4, 4);
    A = ABase + ABase.';  % Make symmetric
    fprintf('Symmetric matrix A:\n');
    disp(A);
    % Eigenvalue decomposition
    [eigenvals, eigenvecs] = MatrixDecompositions.symmetricEigendecomposition(A);
    fprintf('Eigenvalues (ascending order): [%.3f, %.3f, %.3f, %.3f]\n', ...
            eigenvals(1), eigenvals(2), eigenvals(3), eigenvals(4));
    fprintf('Eigenvectors:\n');
    disp(eigenvecs);
    % Verify decomposition: A * v = λ * v
    fprintf('Verification (A*v = λ*v for each eigenpair):\n');
    for i = 1:length(eigenvals)
        v = eigenvecs(:, i);
        Av = A * v;
        lambdaV = eigenvals(i) * v;
        error = norm(Av - lambdaV);
        fprintf('  Eigenpair %d: ||A*v - λ*v|| = %.2e\n', i, error);
    end
    % Verify orthogonality of eigenvectors
    VtV = eigenvecs.' * eigenvecs;
    I = eye(4);
    fprintf('Eigenvector orthogonality: ||V^T*V - I|| = %.2e\n', norm(VtV - I, 'fro'));
    % Application: Matrix powers using eigendecomposition
    fprintf('\n1. Efficient Matrix Powers\n');
    fprintf('-------------------------\n');
    % Compute A^10 using eigendecomposition
    n = 10;
    APowerEigen = eigenvecs * diag(eigenvals.^n) * eigenvecs.';
    APowerDirect = A^n;
    fprintf('A^%d using eigendecomposition:\n', n);
    disp(APowerEigen);
    fprintf('A^%d using direct computation:\n', n);
    disp(APowerDirect);
    fprintf('Difference: %.2e\n', norm(APowerEigen - APowerDirect, 'fro'));
    % Application: Quadratic forms and definiteness
    fprintf('\n2. Quadratic Forms Analysis\n');
    fprintf('--------------------------\n');
    fprintf('Matrix eigenvalues: [%.3f, %.3f, %.3f, %.3f]\n', ...
            eigenvals(1), eigenvals(2), eigenvals(3), eigenvals(4));
    if all(eigenvals > 0)
        definiteness = 'positive definite';
    elseif all(eigenvals >= 0)
        definiteness = 'positive semidefinite';
    elseif all(eigenvals < 0)
        definiteness = 'negative definite';
    elseif all(eigenvals <= 0)
        definiteness = 'negative semidefinite';
    else
        definiteness = 'indefinite';
    end
    fprintf('Matrix is: %s\n', definiteness);
    % Visualize quadratic form x^T A x = c (for 2D case)
    A2d = A(1:2, 1:2);  % Take 2x2 submatrix
    [eigenvals2d, eigenvecs2d] = eig(A2d);
    eigenvals2d = diag(eigenvals2d);
    fprintf('2D submatrix eigenvalues: [%.3f, %.3f]\n', eigenvals2d(1), eigenvals2d(2));
    % Visualization
    figure('Name', 'Eigenvalue Analysis', 'Position', [400, 400, 800, 400]);
    subplot(1, 2, 1);
    % Create grid for contour plot
    x = linspace(-3, 3, 100);
    y = linspace(-3, 3, 100);
    [X, Y] = meshgrid(x, y);
    % Compute quadratic form
    Z = zeros(size(X));
    for i = 1:length(x)
        for j = 1:length(y)
            vec = [X(i, j); Y(i, j)];
            Z(i, j) = vec.' * A2d * vec;
        end
    end
    contour(X, Y, Z, 20, 'Color', [0, 50, 98]/255);
    hold on;
    % Plot eigenvector directions
    center = [0, 0];
    for i = 1:2
        direction = eigenvecs2d(:, i) * 2;
        quiver(center(1), center(2), direction(1), direction(2), 0, ...
               'Color', [253, 181, 21]/255, 'LineWidth', 2, 'MaxHeadSize', 0.3);
        text(direction(1)*1.2, direction(2)*1.2, sprintf('λ=%.2f', eigenvals2d(i)), ...
             'FontSize', 10, 'Color', [253, 181, 21]/255, 'FontWeight', 'bold');
    end
    xlabel('x_1');
    ylabel('x_2');
    title('Quadratic Form: x^T A x = c');
    grid on;
    axis equal;
    % Eigenvalue visualization
    subplot(1, 2, 2);
    bar(1:length(eigenvals), eigenvals, 'FaceColor', [0, 50, 98]/255, 'FaceAlpha', 0.7);
    hold on;
    yline(0, 'r--', 'LineWidth', 1);
    xlabel('Eigenvalue Index');
    ylabel('Eigenvalue');
    title('Eigenvalue Spectrum');
    grid on;
end
function imageCompressionExample()
    % Demonstrate image compression using SVD
    fprintf('\n\nPractical Application: Image Compression with SVD\n');
    fprintf('=================================================\n');
    % Create a simple synthetic "image" (2D pattern)
    x = linspace(-2, 2, 64);
    y = linspace(-2, 2, 64);
    [X, Y] = meshgrid(x, y);
    % Create interesting pattern
    image = exp(-(X.^2 + Y.^2)/2) .* cos(4*X) .* sin(3*Y) + ...
            0.3 * exp(-((X-1).^2 + (Y+0.5).^2)/0.5);
    fprintf('Original image size: %dx%d\n', size(image, 1), size(image, 2));
    fprintf('Original rank: %d\n', rank(image));
    % SVD compression
    [U, S, V] = MatrixDecompositions.svd(image, false);
    fprintf('Number of singular values: %d\n', length(S));
    fprintf('Largest singular values: [%.3f, %.3f, %.3f, %.3f, %.3f, ...]\n', ...
            S(1), S(2), S(3), S(4), S(5));
    % Compress with different ranks
    ranks = [1, 5, 10, 20, length(S)];
    figure('Name', 'SVD Image Compression', 'Position', [100, 100, 1200, 800]);
    for i = 1:length(ranks)
        k = ranks(i);
        if k <= length(S)
            % Reconstruct using first k singular values
            imageCompressed = U(:, 1:k) * diag(S(1:k)) * V(:, 1:k).';
            % Compute compression ratio and error
            originalElements = numel(image);
            compressedElements = k * (size(U, 1) + size(V, 1)) + k;
            compressionRatio = originalElements / compressedElements;
            relativeError = norm(image - imageCompressed, 'fro') / norm(image, 'fro');
            subplot(2, 3, i);
            imagesc(x, y, imageCompressed);
            colorbar;
            if k == length(S)
                titleStr = sprintf('Original\n(rank %d)', k);
            else
                titleStr = sprintf('Rank %d\nCompression: %.1fx\nError: %.3f', ...
                                 k, compressionRatio, relativeError);
            end
            title(titleStr);
            xlabel('x');
            ylabel('y');
            axis equal tight;
        end
    end
    % Singular value decay
    subplot(2, 3, 6);
    semilogy(1:length(S), S, 'o-', 'Color', [0, 50, 98]/255, 'MarkerSize', 4);
    xlabel('Singular Value Index');
    ylabel('Singular Value');
    title('Singular Value Decay');
    grid on;
    % Mark compression points
    for i = 1:length(ranks)-1
        k = ranks(i);
        if k <= length(S)
            xline(k, 'Color', [253, 181, 21]/255, 'LineStyle', '--');
            text(k+1, S(k), sprintf('%d', k), 'Color', [253, 181, 21]/255, 'FontWeight', 'bold');
        end
    end
    % Compression analysis
    fprintf('\nCompression Analysis:\n');
    fprintf('Rank\tComp.Ratio\tRel.Error\tStorage%%\n');
    fprintf('----------------------------------------\n');
    for i = 1:length(ranks)-1
        k = ranks(i);
        if k <= length(S)
            imageCompressed = U(:, 1:k) * diag(S(1:k)) * V(:, 1:k).';
            originalElements = numel(image);
            compressedElements = k * (size(U, 1) + size(V, 1)) + k;
            compressionRatio = originalElements / compressedElements;
            storagePercent = (compressedElements / originalElements) * 100;
            relativeError = norm(image - imageCompressed, 'fro') / norm(image, 'fro');
            fprintf('%4d\t%8.1f\t%8.3f\t%7.1f%%\n', k, compressionRatio, relativeError, storagePercent);
        end
    end
end