function test_matrix_operations()
    % TEST_MATRIX_OPERATIONS Comprehensive test suite for matrix operations
    %
    % This function tests all matrix operation functionality including
    % basic arithmetic, decompositions, properties, and special matrices.
    fprintf('Running Matrix Operations Tests\n');
    fprintf('===============================\n\n');
    % Add path to core functions
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'core'));
    % Run test suites
    testBasicOperations();
    testMatrixDecompositions();
    testMatrixProperties();
    testErrorHandling();
    fprintf('\n===============================\n');
    fprintf('All Matrix Operations Tests PASSED!\n');
end
function testBasicOperations()
    % Test basic matrix operations
    fprintf('Testing Basic Matrix Operations...\n');
    % Test matrices
    A = [2, 1; 1, 3];
    B = [1, 2; 3, 1];
    C = [1, 2, 1; 0, 1, 2];  % Non-square
    v = [1; 2];
    % Test matrix validation
    validatedA = MatrixOperations.validateMatrix(A, 'A');
    assert(isequal(validatedA, A), 'Matrix validation failed');
    % Test matrix multiplication
    resultMM = MatrixOperations.matrixMultiply(A, B);
    expected = A * B;
    assert(norm(resultMM - expected, 'fro') < 1e-12, 'Matrix multiplication failed');
    % Test matrix-vector multiplication
    resultMV = MatrixOperations.matrixMultiply(A, v);
    expected = A * v;
    assert(norm(resultMV - expected) < 1e-12, 'Matrix-vector multiplication failed');
    % Test matrix addition
    resultAdd = MatrixOperations.matrixAdd(A, B);
    expected = A + B;
    assert(norm(resultAdd - expected, 'fro') < 1e-12, 'Matrix addition failed');
    % Test matrix subtraction
    resultSub = MatrixOperations.matrixSubtract(A, B);
    expected = A - B;
    assert(norm(resultSub - expected, 'fro') < 1e-12, 'Matrix subtraction failed');
    % Test matrix power
    resultPow = MatrixOperations.matrixPower(A, 2);
    expected = A^2;
    assert(norm(resultPow - expected, 'fro') < 1e-12, 'Matrix power failed');
    % Test matrix power A^0 = I
    resultPow0 = MatrixOperations.matrixPower(A, 0);
    expected = eye(2);
    assert(norm(resultPow0 - expected, 'fro') < 1e-12, 'Matrix power A^0 failed');
    % Test transpose
    resultTranspose = MatrixOperations.transpose(A);
    expected = A.';
    assert(norm(resultTranspose - expected, 'fro') < 1e-12, 'Matrix transpose failed');
    % Test conjugate transpose
    A_complex = [1+2i, 3-1i; 2+1i, 4];
    resultConjTranspose = MatrixOperations.conjugateTranspose(A_complex);
    expected = A_complex';
    assert(norm(resultConjTranspose - expected, 'fro') < 1e-12, 'Conjugate transpose failed');
    % Test trace
    resultTrace = MatrixOperations.trace(A);
    expected = trace(A);
    assert(abs(resultTrace - expected) < 1e-12, 'Matrix trace failed');
    % Test determinant
    resultDet = MatrixOperations.determinant(A);
    expected = det(A);
    assert(abs(resultDet - expected) < 1e-12, 'Matrix determinant failed');
    % Test rank
    resultRank = MatrixOperations.rank(A);
    expected = rank(A);
    assert(resultRank == expected, 'Matrix rank failed');
    % Test condition number
    resultCond = MatrixOperations.conditionNumber(A);
    expected = cond(A);
    assert(abs(resultCond - expected) < 1e-10, 'Condition number failed');
    % Test norms
    resultFrob = MatrixOperations.frobeniusNorm(A);
    expected = norm(A, 'fro');
    assert(abs(resultFrob - expected) < 1e-12, 'Frobenius norm failed');
    resultSpec = MatrixOperations.spectralNorm(A);
    expected = norm(A, 2);
    assert(abs(resultSpec - expected) < 1e-12, 'Spectral norm failed');
    resultNuc = MatrixOperations.nuclearNorm(A);
    s = svd(A);
    expected = sum(s);
    assert(abs(resultNuc - expected) < 1e-12, 'Nuclear norm failed');
    fprintf('  Basic operations: PASSED\n');
end
function testMatrixDecompositions()
    % Test matrix decomposition algorithms
    fprintf('Testing Matrix Decompositions...\n');
    % Test data
    rng(42);
    A = randn(4, 4);
    A_spd = A.' * A + 0.1 * eye(4);  % Symmetric positive definite
    A_rect = randn(5, 3);
    % Test LU decomposition
    [P, L, U] = MatrixDecompositions.luDecomposition(A);
    % Check dimensions
    assert(isequal(size(P), [4, 4]), 'LU: P dimension incorrect');
    assert(isequal(size(L), [4, 4]), 'LU: L dimension incorrect');
    assert(isequal(size(U), [4, 4]), 'LU: U dimension incorrect');
    % Check decomposition: PA = LU
    assert(norm(P * A - L * U, 'fro') < 1e-12, 'LU decomposition failed');
    % Check L is lower triangular with unit diagonal
    assert(norm(triu(L, 1), 'fro') < 1e-14, 'LU: L not lower triangular');
    assert(norm(diag(L) - ones(4, 1)) < 1e-14, 'LU: L diagonal not unit');
    % Check U is upper triangular
    assert(norm(tril(U, -1), 'fro') < 1e-14, 'LU: U not upper triangular');
    % Test QR decomposition
    [Q, R] = MatrixDecompositions.qrDecomposition(A_rect, 'full');
    % Check decomposition: A = QR
    assert(norm(A_rect - Q * R, 'fro') < 1e-12, 'QR decomposition failed');
    % Check Q is orthogonal
    assert(norm(Q.' * Q - eye(size(Q, 2)), 'fro') < 1e-12, 'QR: Q not orthogonal');
    % Check R is upper triangular
    assert(norm(tril(R, -1), 'fro') < 1e-14, 'QR: R not upper triangular');
    % Test economic QR
    [Q_econ, R_econ] = MatrixDecompositions.qrDecomposition(A_rect, 'economic');
    assert(isequal(size(Q_econ), [5, 3]), 'Economic QR: Q dimension incorrect');
    assert(isequal(size(R_econ), [3, 3]), 'Economic QR: R dimension incorrect');
    assert(norm(A_rect - Q_econ * R_econ, 'fro') < 1e-12, 'Economic QR decomposition failed');
    % Test Cholesky decomposition
    L_chol = MatrixDecompositions.choleskyDecomposition(A_spd, true);
    % Check decomposition: A = LL'
    assert(norm(A_spd - L_chol * L_chol.', 'fro') < 1e-12, 'Cholesky decomposition failed');
    % Check L is lower triangular
    assert(norm(triu(L_chol, 1), 'fro') < 1e-14, 'Cholesky: L not lower triangular');
    % Test upper triangular Cholesky
    U_chol = MatrixDecompositions.choleskyDecomposition(A_spd, false);
    assert(norm(A_spd - U_chol.' * U_chol, 'fro') < 1e-12, 'Upper Cholesky decomposition failed');
    % Test SVD
    [U_svd, S_svd, V_svd] = MatrixDecompositions.svd(A_rect, false);
    % Check decomposition: A = U * diag(S) * V'
    assert(norm(A_rect - U_svd * diag(S_svd) * V_svd.', 'fro') < 1e-12, 'SVD decomposition failed');
    % Check orthogonality
    assert(norm(U_svd.' * U_svd - eye(size(U_svd, 2)), 'fro') < 1e-12, 'SVD: U not orthogonal');
    assert(norm(V_svd * V_svd.' - eye(size(V_svd, 1)), 'fro') < 1e-12, 'SVD: V not orthogonal');
    % Check singular values are non-negative and sorted
    assert(all(S_svd >= 0), 'SVD: Singular values not non-negative');
    assert(all(S_svd(1:end-1) >= S_svd(2:end)), 'SVD: Singular values not sorted');
    % Test singular values only
    S_only = MatrixDecompositions.svd(A_rect, false, false);
    assert(norm(S_svd - S_only) < 1e-12, 'SVD singular values only failed');
    % Test eigendecomposition
    [eigenvals, eigenvecs] = MatrixDecompositions.eigendecomposition(A);
    % Check decomposition: A * v = λ * v for each eigenpair
    for i = 1:length(eigenvals)
        lhs = A * eigenvecs(:, i);
        rhs = eigenvals(i) * eigenvecs(:, i);
        assert(norm(lhs - rhs) < 1e-10, sprintf('Eigendecomposition failed for eigenpair %d', i));
    end
    % Check that eigenvalues are sorted by magnitude
    mags = abs(eigenvals);
    assert(all(mags(1:end-1) >= mags(2:end)), 'Eigenvalues not sorted by magnitude');
    % Test symmetric eigendecomposition
    [eigenvals_sym, eigenvecs_sym] = MatrixDecompositions.symmetricEigendecomposition(A_spd);
    % Check decomposition
    for i = 1:length(eigenvals_sym)
        lhs = A_spd * eigenvecs_sym(:, i);
        rhs = eigenvals_sym(i) * eigenvecs_sym(:, i);
        assert(norm(lhs - rhs) < 1e-10, sprintf('Symmetric eigendecomposition failed for eigenpair %d', i));
    end
    % Check that eigenvalues are real and sorted
    assert(all(isreal(eigenvals_sym)), 'Symmetric eigenvalues not real');
    assert(all(eigenvals_sym(1:end-1) <= eigenvals_sym(2:end)), 'Symmetric eigenvalues not sorted ascending');
    % Check that eigenvectors are orthonormal
    assert(norm(eigenvecs_sym.' * eigenvecs_sym - eye(4), 'fro') < 1e-12, 'Symmetric eigenvectors not orthonormal');
    % For SPD matrix, all eigenvalues should be positive
    assert(all(eigenvals_sym > 0), 'SPD matrix eigenvalues not all positive');
    % Test Schur decomposition
    [T, Z] = MatrixDecompositions.schurDecomposition(A);
    % Check decomposition: A = Z * T * Z'
    assert(norm(A - Z * T * Z.', 'fro') < 1e-12, 'Schur decomposition failed');
    % Check Z is orthogonal
    assert(norm(Z.' * Z - eye(4), 'fro') < 1e-12, 'Schur: Z not orthogonal');
    fprintf('  Matrix decompositions: PASSED\n');
end
function testMatrixProperties()
    % Test matrix property detection
    fprintf('Testing Matrix Properties...\n');
    rng(42);
    % Symmetric matrix
    A = randn(3, 3);
    A_sym = A + A.';
    % Hermitian matrix
    A_complex = randn(3, 3) + 1i * randn(3, 3);
    A_herm = A_complex + A_complex';
    % Orthogonal matrix
    [Q, ~] = qr(randn(3, 3));
    % Positive definite matrix
    A_pd = A.' * A + 0.1 * eye(3);
    % Positive semidefinite matrix (rank deficient)
    A_psd = randn(3, 2) * randn(2, 3);
    A_psd = A_psd * A_psd.';
    % Test symmetry detection
    assert(SpecialMatrices.isSymmetric(A_sym), 'Symmetric detection failed');
    assert(~SpecialMatrices.isSymmetric(randn(3, 3)), 'Non-symmetric incorrectly detected as symmetric');
    assert(~SpecialMatrices.isSymmetric(randn(3, 2)), 'Non-square matrix incorrectly detected as symmetric');
    % Test Hermitian detection
    assert(SpecialMatrices.isHermitian(A_herm), 'Hermitian detection failed');
    assert(SpecialMatrices.isHermitian(A_sym), 'Real symmetric not detected as Hermitian');
    assert(~SpecialMatrices.isHermitian(randn(3, 3) + 1i * randn(3, 3)), 'Non-Hermitian incorrectly detected as Hermitian');
    % Test orthogonality detection
    assert(SpecialMatrices.isOrthogonal(Q), 'Orthogonal detection failed');
    assert(SpecialMatrices.isOrthogonal(eye(3)), 'Identity not detected as orthogonal');
    assert(~SpecialMatrices.isOrthogonal(randn(3, 3)), 'Non-orthogonal incorrectly detected as orthogonal');
    % Test unitary detection
    assert(SpecialMatrices.isUnitary(Q), 'Real orthogonal not detected as unitary');
    % Complex unitary matrix
    [Q_complex, ~] = qr(randn(3, 3) + 1i * randn(3, 3));
    assert(SpecialMatrices.isUnitary(Q_complex), 'Complex unitary detection failed');
    % Test positive definiteness detection
    assert(SpecialMatrices.isPositiveDefinite(A_pd), 'Positive definite detection failed');
    assert(~SpecialMatrices.isPositiveDefinite(A_psd), 'Positive semidefinite incorrectly detected as positive definite');
    assert(~SpecialMatrices.isPositiveDefinite(randn(3, 3)), 'Random matrix incorrectly detected as positive definite');
    % Test positive semidefiniteness detection
    assert(SpecialMatrices.isPositiveSemidefinite(A_pd), 'Positive definite not detected as positive semidefinite');
    assert(SpecialMatrices.isPositiveSemidefinite(A_psd), 'Positive semidefinite detection failed');
    % Negative definite matrix
    negative_definite = -A_pd;
    assert(~SpecialMatrices.isPositiveSemidefinite(negative_definite), 'Negative definite incorrectly detected as positive semidefinite');
    % Test matrix symmetrization
    A_nonsym = randn(3, 3);
    A_made_sym = SpecialMatrices.makeSymmetric(A_nonsym);
    assert(SpecialMatrices.isSymmetric(A_made_sym), 'Matrix symmetrization failed');
    expected_sym = (A_nonsym + A_nonsym.') / 2;
    assert(norm(A_made_sym - expected_sym, 'fro') < 1e-12, 'Matrix symmetrization incorrect');
    % Test matrix Hermitian-ization
    A_nonherm = randn(3, 3) + 1i * randn(3, 3);
    A_made_herm = SpecialMatrices.makeHermitian(A_nonherm);
    assert(SpecialMatrices.isHermitian(A_made_herm), 'Matrix Hermitian-ization failed');
    expected_herm = (A_nonherm + A_nonherm') / 2;
    assert(norm(A_made_herm - expected_herm, 'fro') < 1e-12, 'Matrix Hermitian-ization incorrect');
    % Test Gram-Schmidt orthogonalization
    A_gs = randn(4, 3);
    Q_gs = SpecialMatrices.gramSchmidt(A_gs, true);
    % Check orthonormality
    assert(norm(Q_gs.' * Q_gs - eye(3), 'fro') < 1e-12, 'Gram-Schmidt orthonormality failed');
    % Check that column space is preserved
    for i = 1:3
        % Each column of A should be expressible as linear combination of Q columns
        coeffs = Q_gs \ A_gs(:, i);
        reconstructed = Q_gs * coeffs;
        assert(norm(reconstructed - A_gs(:, i)) < 1e-12, sprintf('Gram-Schmidt column space preservation failed for column %d', i));
    end
    % Test Householder reflector
    x = [3; 4; 0];
    H = SpecialMatrices.householderReflector(x);
    % Check that H is orthogonal
    assert(SpecialMatrices.isOrthogonal(H), 'Householder reflector not orthogonal');
    % Check that Hx = ||x|| e_1
    Hx = H * x;
    expected = [norm(x); 0; 0];
    assert(norm(abs(Hx) - abs(expected)) < 1e-12, 'Householder reflector transformation failed');
    % Test Givens rotation
    a = 3.0; b = 4.0;
    [c, s, G] = SpecialMatrices.givensRotation(a, b);
    % Check rotation properties
    assert(abs(c^2 + s^2 - 1) < 1e-14, 'Givens rotation normalization failed');
    % Check that rotation zeros out b
    result = G * [a; b];
    assert(abs(result(2)) < 1e-14, 'Givens rotation zeroing failed');
    % Check that G is orthogonal
    assert(SpecialMatrices.isOrthogonal(G), 'Givens rotation matrix not orthogonal');
    fprintf('  Matrix properties: PASSED\n');
end
function testErrorHandling()
    % Test error handling and edge cases
    fprintf('Testing Error Handling...\n');
    % Test invalid inputs
    try
        MatrixOperations.validateMatrix([]);
        assert(false, 'Empty matrix validation should fail');
    catch ME
        assert(contains(ME.message, 'cannot be empty'), 'Wrong error message for empty matrix');
    end
    try
        MatrixOperations.validateMatrix(rand(1, 1, 2));
        assert(false, '3D array validation should fail');
    catch ME
        assert(contains(ME.message, '2-dimensional'), 'Wrong error message for 3D array');
    end
    % Test dimension mismatches
    A = rand(2, 3);
    B = rand(4, 2);
    try
        MatrixOperations.matrixMultiply(A, B);
        assert(false, 'Incompatible matrix multiplication should fail');
    catch ME
        assert(contains(ME.message, 'Incompatible dimensions'), 'Wrong error message for incompatible multiplication');
    end
    try
        MatrixOperations.matrixAdd(A, B);
        assert(false, 'Incompatible matrix addition should fail');
    catch ME
        assert(contains(ME.message, 'Incompatible shapes'), 'Wrong error message for incompatible addition');
    end
    % Test non-square matrix operations
    try
        MatrixOperations.trace(A);
        assert(false, 'Trace of non-square matrix should fail');
    catch ME
        assert(contains(ME.message, 'square'), 'Wrong error message for non-square trace');
    end
    try
        MatrixOperations.determinant(A);
        assert(false, 'Determinant of non-square matrix should fail');
    catch ME
        assert(contains(ME.message, 'square'), 'Wrong error message for non-square determinant');
    end
    try
        MatrixOperations.matrixPower(A, 2);
        assert(false, 'Matrix power of non-square matrix should fail');
    catch ME
        assert(contains(ME.message, 'square'), 'Wrong error message for non-square matrix power');
    end
    % Test negative matrix power
    try
        MatrixOperations.matrixPower(eye(2), -1);
        assert(false, 'Negative matrix power should fail');
    catch ME
        assert(contains(ME.message, 'non-negative'), 'Wrong error message for negative power');
    end
    % Test Cholesky on non-positive definite matrix
    A_not_pd = [1, 2; 2, 1];
    try
        MatrixDecompositions.choleskyDecomposition(A_not_pd);
        assert(false, 'Cholesky of non-positive definite matrix should fail');
    catch ME
        assert(contains(ME.message, 'positive definite'), 'Wrong error message for non-positive definite Cholesky');
    end
    fprintf('  Error handling: PASSED\n');
end
function testCreateTestMatrices()
    % Test test matrix creation function
    fprintf('Testing Test Matrix Creation...\n');
    matrices = MatrixOperations.createTestMatrices();
    expectedFields = {'random_3x3', 'symmetric_4x4', 'positive_definite_3x3', ...
                     'orthogonal_4x4', 'singular_3x3', 'hilbert_5x5'};
    for i = 1:length(expectedFields)
        assert(isfield(matrices, expectedFields{i}), sprintf('Missing test matrix: %s', expectedFields{i}));
    end
    % Check properties
    assert(SpecialMatrices.isSymmetric(matrices.symmetric_4x4), 'Test symmetric matrix not symmetric');
    assert(SpecialMatrices.isPositiveDefinite(matrices.positive_definite_3x3), 'Test positive definite matrix not positive definite');
    assert(SpecialMatrices.isOrthogonal(matrices.orthogonal_4x4), 'Test orthogonal matrix not orthogonal');
    % Singular matrix should have zero determinant
    assert(abs(det(matrices.singular_3x3)) < 1e-10, 'Test singular matrix not singular');
    % Hilbert matrix should be ill-conditioned
    assert(cond(matrices.hilbert_5x5) > 1e5, 'Test Hilbert matrix not ill-conditioned');
    fprintf('  Test matrix creation: PASSED\n');
end