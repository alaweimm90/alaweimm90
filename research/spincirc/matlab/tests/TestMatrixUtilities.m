classdef TestMatrixUtilities < matlab.unittest.TestCase
    % TESTMATRIXUTILITIES - Unit tests for matrix_utilities
    %
    % This test class validates matrix operations including Pauli matrices,
    % rotation matrices, tensor products, and specialized operations.
    %
    % Author: Meshal Alawein <meshal@berkeley.edu>
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    % License: MIT
    
    methods(Test)
        function testPauliMatrices(testCase)
            % Test Pauli matrix generation
            
            [sx, sy, sz] = matrix_utilities('pauli');
            
            % Check dimensions
            testCase.verifySize(sx, [2, 2]);
            testCase.verifySize(sy, [2, 2]);
            testCase.verifySize(sz, [2, 2]);
            
            % Verify Pauli matrix values
            expected_sx = [0, 1; 1, 0];
            expected_sy = [0, -1i; 1i, 0];
            expected_sz = [1, 0; 0, -1];
            
            testCase.verifyEqual(sx, expected_sx, 'AbsTol', 1e-15);
            testCase.verifyEqual(sy, expected_sy, 'AbsTol', 1e-15);
            testCase.verifyEqual(sz, expected_sz, 'AbsTol', 1e-15);
            
            % Test anticommutation relations: {σᵢ, σⱼ} = 2δᵢⱼI
            I2 = eye(2);
            testCase.verifyEqual(sx*sx, I2, 'AbsTol', 1e-15);
            testCase.verifyEqual(sy*sy, I2, 'AbsTol', 1e-15);
            testCase.verifyEqual(sz*sz, I2, 'AbsTol', 1e-15);
            
            % Test commutation relations: [σₓ, σᵧ] = 2iσᵤ
            commxy = sx*sy - sy*sx;
            expected_comm = 2i*sz;
            testCase.verifyEqual(commxy, expected_comm, 'AbsTol', 1e-15);
        end
        
        function testPauliMatricesAsStruct(testCase)
            % Test Pauli matrices returned as structure
            
            pauli_struct = matrix_utilities('pauli', 'struct');
            
            testCase.verifyTrue(isstruct(pauli_struct));
            testCase.verifyTrue(isfield(pauli_struct, 'x'));
            testCase.verifyTrue(isfield(pauli_struct, 'y'));
            testCase.verifyTrue(isfield(pauli_struct, 'z'));
            testCase.verifyTrue(isfield(pauli_struct, 'I'));
            
            % Verify values
            expected_sx = [0, 1; 1, 0];
            testCase.verifyEqual(pauli_struct.x, expected_sx, 'AbsTol', 1e-15);
        end
        
        function testRotationMatrices(testCase)
            % Test rotation matrix generation
            
            % Test rotation around z-axis
            angle = pi/4;
            Rz = matrix_utilities('rotation', 'z', angle);
            
            testCase.verifySize(Rz, [3, 3]);
            
            % Verify rotation matrix properties
            testCase.verifyEqual(det(Rz), 1, 'AbsTol', 1e-15); % Proper rotation
            testCase.verifyEqual(Rz * Rz', eye(3), 'AbsTol', 1e-15); % Orthogonal
            
            % Test specific rotation
            expected_Rz = [cos(angle), -sin(angle), 0;
                          sin(angle), cos(angle), 0;
                          0, 0, 1];
            testCase.verifyEqual(Rz, expected_Rz, 'AbsTol', 1e-15);
            
            % Test rotation around x-axis
            Rx = matrix_utilities('rotation', 'x', pi/2);
            expected_Rx = [1, 0, 0;
                          0, 0, -1;
                          0, 1, 0];
            testCase.verifyEqual(Rx, expected_Rx, 'AbsTol', 1e-15);
        end
        
        function testArbitraryAxisRotation(testCase)
            % Test rotation around arbitrary axis
            
            axis = [1, 1, 1];  % Will be normalized
            angle = pi/3;
            R = matrix_utilities('rotation', axis, angle);
            
            testCase.verifySize(R, [3, 3]);
            testCase.verifyEqual(det(R), 1, 'AbsTol', 1e-15);
            testCase.verifyEqual(R * R', eye(3), 'AbsTol', 1e-15);
        end
        
        function testRotationWithQuaternion(testCase)
            % Test rotation matrix with quaternion output
            
            angle = pi/6;
            [R, q] = matrix_utilities('rotation', 'z', angle);
            
            testCase.verifySize(R, [3, 3]);
            testCase.verifySize(q, [4, 1]);
            
            % Verify quaternion normalization
            testCase.verifyEqual(norm(q), 1, 'AbsTol', 1e-15);
            
            % Verify quaternion values for z-rotation
            expected_q = [cos(angle/2); 0; 0; sin(angle/2)];
            testCase.verifyEqual(q, expected_q, 'AbsTol', 1e-15);
        end
        
        function testTensorProduct(testCase)
            % Test tensor (Kronecker) product
            
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            
            result = matrix_utilities('tensor_product', A, B);
            expected = kron(A, B);
            
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-15);
            
            % Test with Pauli matrices
            [sx, sy, sz] = matrix_utilities('pauli');
            result = matrix_utilities('tensor_product', sx, sy);
            expected = kron(sx, sy);
            
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-15);
        end
        
        function testCommutator(testCase)
            % Test matrix commutator [A, B] = AB - BA
            
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            
            result = matrix_utilities('commutator', A, B);
            expected = A * B - B * A;
            
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-15);
            
            % Test Pauli matrix commutators
            [sx, sy, sz] = matrix_utilities('pauli');
            
            % [σₓ, σᵧ] = 2iσᵤ
            comm_xy = matrix_utilities('commutator', sx, sy);
            expected = 2i * sz;
            testCase.verifyEqual(comm_xy, expected, 'AbsTol', 1e-15);
            
            % Commutator with itself should be zero
            comm_xx = matrix_utilities('commutator', sx, sx);
            testCase.verifyEqual(comm_xx, zeros(2, 2), 'AbsTol', 1e-15);
        end
        
        function testAnticommutator(testCase)
            % Test matrix anticommutator {A, B} = AB + BA
            
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            
            result = matrix_utilities('anticommutator', A, B);
            expected = A * B + B * A;
            
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-15);
            
            % Test Pauli matrix anticommutators
            [sx, sy, sz] = matrix_utilities('pauli');
            
            % {σᵢ, σᵢ} = 2I
            anticomm_xx = matrix_utilities('anticommutator', sx, sx);
            expected = 2 * eye(2);
            testCase.verifyEqual(anticomm_xx, expected, 'AbsTol', 1e-15);
            
            % {σₓ, σᵧ} = 0 (for different Pauli matrices)
            anticomm_xy = matrix_utilities('anticommutator', sx, sy);
            testCase.verifyEqual(anticomm_xy, zeros(2, 2), 'AbsTol', 1e-15);
        end
        
        function testMatrixExponential(testCase)
            % Test matrix exponential computation
            
            % Test with simple diagonal matrix
            A = diag([1, 2, 3]);
            result = matrix_utilities('matrix_exp', A);
            expected = diag([exp(1), exp(2), exp(3)]);
            testCase.verifyEqual(result, expected, 'RelTol', 1e-12);
            
            % Test with Pauli matrix
            [sx, sy, sz] = matrix_utilities('pauli');
            theta = pi/4;
            result = matrix_utilities('matrix_exp', 1i * theta * sz);
            expected = cos(theta) * eye(2) + 1i * sin(theta) * sz;
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-12);
        end
        
        function testMatrixExponentialMethods(testCase)
            % Test different matrix exponential methods
            
            A = [1, 0.5; 0.5, 1];
            
            result_pade = matrix_utilities('matrix_exp', A, 'Method', 'pade');
            result_series = matrix_utilities('matrix_exp', A, 'Method', 'series');
            result_eigen = matrix_utilities('matrix_exp', A, 'Method', 'eigen');
            
            % All methods should give similar results
            testCase.verifyEqual(result_pade, result_series, 'RelTol', 1e-10);
            testCase.verifyEqual(result_pade, result_eigen, 'RelTol', 1e-10);
        end
        
        function testMatrixLogarithm(testCase)
            % Test matrix logarithm computation
            
            % Test with positive definite matrix
            A = [2, 1; 1, 2];
            log_A = matrix_utilities('matrix_log', A);
            
            % Verify: exp(log(A)) = A
            exp_log_A = matrix_utilities('matrix_exp', log_A);
            testCase.verifyEqual(exp_log_A, A, 'RelTol', 1e-12);
        end
        
        function testSpectralRadius(testCase)
            % Test spectral radius computation
            
            A = [2, 1; 0, 3];
            rho = matrix_utilities('spectral_radius', A);
            
            % Spectral radius should be the largest eigenvalue magnitude
            eigenvals = eig(A);
            expected = max(abs(eigenvals));
            testCase.verifyEqual(rho, expected, 'RelTol', 1e-12);
            
            % Test with symmetric matrix
            A_sym = [3, 1; 1, 2];
            rho_sym = matrix_utilities('spectral_radius', A_sym);
            eigenvals_sym = eig(A_sym);
            expected_sym = max(abs(eigenvals_sym));
            testCase.verifyEqual(rho_sym, expected_sym, 'RelTol', 1e-12);
        end
        
        function testConditionNumberEstimation(testCase)
            % Test condition number estimation
            
            A = [1, 2; 3, 4];
            kappa = matrix_utilities('condition_est', A);
            
            testCase.verifyGreaterThan(kappa, 0);
            testCase.verifyClass(kappa, 'double');
            
            % Test different norms
            kappa_2 = matrix_utilities('condition_est', A, 'Norm', '2');
            kappa_inf = matrix_utilities('condition_est', A, 'Norm', 'inf');
            
            testCase.verifyGreaterThan(kappa_2, 0);
            testCase.verifyGreaterThan(kappa_inf, 0);
        end
        
        function testBlockDiagonal(testCase)
            % Test block diagonal matrix creation
            
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            C = [9];
            
            result = matrix_utilities('block_diag', A, B, C);
            expected = blkdiag(A, B, C);
            
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-15);
            
            % Test with cell array input
            result_cell = matrix_utilities('block_diag', {A, B, C});
            testCase.verifyEqual(result_cell, expected, 'AbsTol', 1e-15);
        end
        
        function testKroneckerSum(testCase)
            % Test Kronecker sum A ⊕ B = A ⊗ I + I ⊗ B
            
            A = [1, 2; 3, 4];
            B = [5, 6; 7, 8];
            
            result = matrix_utilities('kron_sum', A, B);
            expected = kron(A, eye(size(B))) + kron(eye(size(A)), B);
            
            testCase.verifyEqual(result, expected, 'AbsTol', 1e-15);
        end
        
        function testVectorization(testCase)
            % Test matrix vectorization operations
            
            A = [1, 2, 3; 4, 5, 6];
            
            % Test vec operation
            vec_A = matrix_utilities('vectorize', 'vec', A);
            expected_vec = A(:);
            testCase.verifyEqual(vec_A, expected_vec);
            
            % Test unvec operation
            dims = [2, 3];
            unvec_A = matrix_utilities('vectorize', 'unvec', vec_A, dims);
            testCase.verifyEqual(unvec_A, A);
            
            % Test vech (lower triangular vectorization)
            A_square = [1, 2, 3; 2, 4, 5; 3, 5, 6]; % Symmetric
            vech_A = matrix_utilities('vectorize', 'vech', A_square);
            expected_vech = [1; 2; 3; 4; 5; 6]; % Lower triangular elements
            testCase.verifyEqual(vech_A, expected_vech);
        end
        
        function testSpinMatrices(testCase)
            % Test spin-S matrix generation
            
            % Test spin-1/2 (default)
            [Sx, Sy, Sz] = matrix_utilities('spin_matrices');
            
            % Should match Pauli matrices divided by 2
            [sx, sy, sz] = matrix_utilities('pauli');
            testCase.verifyEqual(Sx, sx/2, 'AbsTol', 1e-15);
            testCase.verifyEqual(Sy, sy/2, 'AbsTol', 1e-15);
            testCase.verifyEqual(Sz, sz/2, 'AbsTol', 1e-15);
            
            % Test spin-1
            spin_1 = matrix_utilities('spin_matrices', 1);
            testCase.verifySize(spin_1.x, [3, 3]);
            testCase.verifySize(spin_1.y, [3, 3]);
            testCase.verifySize(spin_1.z, [3, 3]);
            
            % Sz should be diagonal with eigenvalues -1, 0, 1
            eigenvals_z = sort(eig(spin_1.z));
            expected_eigenvals = [-1; 0; 1];
            testCase.verifyEqual(eigenvals_z, expected_eigenvals, 'AbsTol', 1e-15);
        end
        
        function testSU2Generators(testCase)
            % Test SU(2) group generators
            
            [J1, J2, J3] = matrix_utilities('su2_generators');
            
            % Should be Pauli matrices divided by 2
            [sx, sy, sz] = matrix_utilities('pauli');
            testCase.verifyEqual(J1, sx/2, 'AbsTol', 1e-15);
            testCase.verifyEqual(J2, sy/2, 'AbsTol', 1e-15);
            testCase.verifyEqual(J3, sz/2, 'AbsTol', 1e-15);
            
            % Test commutation relations: [Jᵢ, Jⱼ] = iεᵢⱼₖJₖ
            comm_12 = J1*J2 - J2*J1;
            expected_comm = 1i * J3;
            testCase.verifyEqual(comm_12, expected_comm, 'AbsTol', 1e-15);
        end
        
        function testCliffordMatrices2D(testCase)
            % Test 2D Clifford algebra (Pauli matrices)
            
            [gamma1, gamma2] = matrix_utilities('clifford', 2);
            
            % Should match Pauli matrices σₓ and σᵧ
            [sx, sy, sz] = matrix_utilities('pauli');
            testCase.verifyEqual(gamma1, sx, 'AbsTol', 1e-15);
            testCase.verifyEqual(gamma2, sy, 'AbsTol', 1e-15);
            
            % Test anticommutation: {γᵢ, γⱼ} = 2δᵢⱼI
            anticomm_11 = gamma1*gamma1 + gamma1*gamma1;
            testCase.verifyEqual(anticomm_11, 4*eye(2), 'AbsTol', 1e-15);
            
            anticomm_12 = gamma1*gamma2 + gamma2*gamma1;
            testCase.verifyEqual(anticomm_12, zeros(2, 2), 'AbsTol', 1e-15);
        end
        
        function testCliffordMatrices4D(testCase)
            % Test 4D Clifford algebra (Dirac gamma matrices)
            
            gammas = matrix_utilities('clifford', 4);
            
            testCase.verifyLength(gammas, 5); % γ₀, γ₁, γ₂, γ₃, γ₅
            
            % Each gamma matrix should be 4x4
            for i = 1:5
                testCase.verifySize(gammas{i}, [4, 4]);
            end
            
            % Test anticommutation relations for first few matrices
            gamma0 = gammas{1};
            gamma1 = gammas{2};
            
            % {γ₀, γ₀} = 2I
            anticomm_00 = gamma0*gamma0 + gamma0*gamma0;
            testCase.verifyEqual(anticomm_00, 4*eye(4), 'AbsTol', 1e-15);
            
            % {γ₀, γ₁} = 0
            anticomm_01 = gamma0*gamma1 + gamma1*gamma0;
            testCase.verifyEqual(anticomm_01, zeros(4, 4), 'AbsTol', 1e-15);
        end
        
        function testInvalidOperations(testCase)
            % Test error handling for invalid operations
            
            testCase.verifyError(@() matrix_utilities('invalid_operation'), ...
                                'MATLAB:error');
            
            % Test insufficient arguments
            testCase.verifyError(@() matrix_utilities('commutator', [1, 2]), ...
                                'MATLAB:error');
            
            testCase.verifyError(@() matrix_utilities('tensor_product', [1, 2]), ...
                                'MATLAB:error');
        end
        
        function testComplexMatrices(testCase)
            % Test operations with complex matrices
            
            A = [1+1i, 2; 3, 4-1i];
            B = [1-1i, 0; 1i, 2];
            
            % Commutator with complex matrices
            comm = matrix_utilities('commutator', A, B);
            expected = A * B - B * A;
            testCase.verifyEqual(comm, expected, 'AbsTol', 1e-15);
            
            % Matrix exponential with complex matrix
            exp_A = matrix_utilities('matrix_exp', A);
            testCase.verifyClass(exp_A, 'double'); % MATLAB returns double for complex exp
        end
        
        function testLargeMatrices(testCase)
            % Test operations with larger matrices
            
            n = 10;
            A = rand(n, n);
            B = rand(n, n);
            
            % Tensor product
            result = matrix_utilities('tensor_product', A, B);
            testCase.verifySize(result, [n*n, n*n]);
            
            % Commutator
            comm = matrix_utilities('commutator', A, B);
            testCase.verifySize(comm, [n, n]);
            
            % Block diagonal
            block = matrix_utilities('block_diag', A, B);
            testCase.verifySize(block, [2*n, 2*n]);
        end
    end
    
    methods(TestMethodSetup)
        function addUtilsPath(testCase)
            % Add utils directory to path for testing
            current_dir = fileparts(mfilename('fullpath'));
            utils_dir = fullfile(current_dir, '..', 'core', 'utils');
            addpath(utils_dir);
        end
    end
    
    methods(TestMethodTeardown)
        function removeUtilsPath(testCase)
            % Remove utils directory from path after testing
            current_dir = fileparts(mfilename('fullpath'));
            utils_dir = fullfile(current_dir, '..', 'core', 'utils');
            rmpath(utils_dir);
        end
    end
end