classdef ConductanceMatrix < handle
% CONDUCTANCEMATRIX - 4x4 conductance matrix generation utilities
%
% This class generates conductance matrices for spin transport in F/N/F
% heterostructures following the formalism in:
% "Circuit Models for Spintronic Devices Subject to Electric and Magnetic Fields"
% (Alawein & Fariborzi, IEEE J-XCDC 2018)
%
% Key Features:
%   - Nonmagnet: diagonal G^N_se, G^N_sh matrices
%   - Ferromagnet: coupled G^F_se matrices with β, β' asymmetry
%   - Interface: G^F/N tensors with mixing conductances η_R, η_I
%   - Rotation matrices for arbitrary magnetization directions
%   - Frequency-domain AC analysis capabilities
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (Constant)
        % Physical constants
        e = 1.602176634e-19;    % Elementary charge (C)
        hbar = 1.054571817e-34; % Reduced Planck constant (J⋅s)
        kb = 1.380649e-23;      % Boltzmann constant (J/K)
        mu_B = 9.2740100783e-24; % Bohr magneton (J/T)
    end
    
    methods (Static)
        
        function G = buildSystemMatrix(materials, magnetization, geometry, temperature, B_field)
            % Build complete 4x4 conductance matrix for the system
            %
            % Inputs:
            %   materials - Array of material structures
            %   magnetization - Magnetization vectors for FM regions
            %   geometry - Geometry parameters
            %   temperature - Operating temperature (K)
            %   B_field - External magnetic field (T)
            %
            % Outputs:
            %   G - 4x4 system conductance matrix
            
            n_regions = length(materials);
            
            % Calculate individual region conductances
            G_regions = cell(n_regions, 1);
            for i = 1:n_regions
                material = materials(i);
                
                if strcmp(material.type, 'ferromagnet')
                    m_vec = magnetization(sum(strcmp({materials(1:i).type}, 'ferromagnet')), :);
                    G_regions{i} = ConductanceMatrix.ferromagnetConductance(...
                        material, m_vec, geometry, temperature, B_field);
                else
                    G_regions{i} = ConductanceMatrix.nonmagnetConductance(...
                        material, geometry, temperature, B_field);
                end
            end
            
            % Calculate interface conductances
            G_interfaces = cell(n_regions-1, 1);
            fm_count = 0;
            for i = 1:n_regions-1
                mat1 = materials(i);
                mat2 = materials(i+1);
                
                % Get magnetization vectors if needed
                m1 = [];
                m2 = [];
                if strcmp(mat1.type, 'ferromagnet')
                    fm_count = fm_count + 1;
                    m1 = magnetization(fm_count, :);
                end
                if strcmp(mat2.type, 'ferromagnet')
                    if ~strcmp(mat1.type, 'ferromagnet')
                        fm_count = fm_count + 1;
                    else
                        fm_count = fm_count + 1;
                    end
                    m2 = magnetization(fm_count, :);
                end
                
                G_interfaces{i} = ConductanceMatrix.interfaceConductance(...
                    mat1, mat2, m1, m2, geometry, temperature);
            end
            
            % Assemble system matrix
            G = ConductanceMatrix.assembleSystemMatrix(G_regions, G_interfaces);
        end
        
        function G_nm = nonmagnetConductance(material, geometry, temperature, B_field)
            % Generate 4x4 conductance matrix for nonmagnetic material
            %
            % For nonmagnets, the conductance matrix is diagonal:
            % G_nm = diag([G_se, G_sh, G_sh, G_sh])
            
            % Extract material parameters
            rho = material.rho;              % Resistivity (Ω⋅m)
            lambda_sf = material.lambda_sf;  % Spin diffusion length (m)
            D = material.D;                  % Spin diffusion coefficient
            
            % Geometry parameters
            L = geometry.length;
            A = geometry.area;
            
            % Temperature correction
            if isfield(material, 'rho_T_coeff')
                rho = rho * (1 + material.rho_T_coeff * (temperature - 300));
            end
            
            % Calculate conductances
            G_se = A / (rho * L);  % Charge conductance
            
            % Spin conductance (includes diffusion effects)
            if L < lambda_sf
                % Short channel: ballistic regime
                G_sh = G_se;
            else
                % Long channel: diffusive regime
                G_sh = G_se * (lambda_sf / L) * tanh(L / lambda_sf);
            end
            
            % Apply magnetic field effects (Hanle precession)
            if norm(B_field) > 0
                omega_L = ConductanceMatrix.mu_B * norm(B_field) / ConductanceMatrix.hbar;
                tau_sf = lambda_sf^2 / D;
                
                % Hanle reduction factor
                hanle_factor = 1 / (1 + (omega_L * tau_sf)^2);
                G_sh = G_sh * hanle_factor;
            end
            
            % Build 4x4 matrix
            G_nm = diag([G_se, G_sh, G_sh, G_sh]);
        end
        
        function G_fm = ferromagnetConductance(material, m_vec, geometry, temperature, B_field)
            % Generate 4x4 conductance matrix for ferromagnetic material
            %
            % For ferromagnets, includes spin-dependent transport:
            % G_fm includes coupling between charge and spin channels
            
            % Extract material parameters
            rho = material.rho;
            lambda_sf = material.lambda_sf;
            beta = material.beta;      % Spin asymmetry parameter
            beta_prime = material.beta_prime;  % Interface asymmetry
            alpha_damping = material.alpha;    % Gilbert damping
            
            % Geometry parameters
            L = geometry.length;
            A = geometry.area;
            
            % Temperature corrections
            if isfield(material, 'rho_T_coeff')
                rho = rho * (1 + material.rho_T_coeff * (temperature - 300));
            end
            
            % Base conductances
            G_se = A / (rho * L);
            
            % Spin-dependent conductances
            G_up = G_se * (1 + beta) / 2;
            G_down = G_se * (1 - beta) / 2;
            
            % Normalize magnetization
            m_hat = m_vec / norm(m_vec);
            
            % Build rotation matrix for arbitrary magnetization direction
            R = ConductanceMatrix.buildRotationMatrix(m_hat);
            
            % Base 4x4 matrix in magnetization frame
            G_base = zeros(4, 4);
            
            % Charge-charge coupling
            G_base(1, 1) = G_se;
            
            % Charge-spin coupling
            G_base(1, 4) = G_se * beta_prime;  % Coupling to z-spin
            G_base(4, 1) = G_se * beta_prime;
            
            % Spin-spin couplings
            G_base(2, 2) = G_up;   % x-spin
            G_base(3, 3) = G_up;   % y-spin
            G_base(4, 4) = G_down; % z-spin (along magnetization)
            
            % Apply magnetization rotation
            R_4x4 = blkdiag(1, R);  % Charge channel unchanged
            G_fm = R_4x4' * G_base * R_4x4;
            
            % Add dynamic effects for AC analysis
            if isfield(material, 'frequency') && material.frequency > 0
                omega = 2 * pi * material.frequency;
                gamma = 1.760859644e11;  % Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
                
                % Add frequency-dependent terms
                G_dynamic = ConductanceMatrix.buildDynamicTerms(omega, gamma, alpha_damping, m_hat);
                G_fm = G_fm + G_dynamic;
            end
        end
        
        function G_int = interfaceConductance(mat1, mat2, m1, m2, geometry, temperature)
            % Generate 4x4 interface conductance matrix
            %
            % Handles F/N, N/F, and F/F interfaces with mixing conductances
            
            A = geometry.area;
            
            % Determine interface type
            is_fm1 = strcmp(mat1.type, 'ferromagnet');
            is_fm2 = strcmp(mat2.type, 'ferromagnet');
            
            if is_fm1 && is_fm2
                % F/F interface
                G_int = ConductanceMatrix.buildFFInterface(mat1, mat2, m1, m2, A, temperature);
            elseif is_fm1 && ~is_fm2
                % F/N interface
                G_int = ConductanceMatrix.buildFNInterface(mat1, mat2, m1, A, temperature);
            elseif ~is_fm1 && is_fm2
                % N/F interface
                G_int = ConductanceMatrix.buildNFInterface(mat1, mat2, m2, A, temperature);
            else
                % N/N interface (simple resistive)
                R_int = (mat1.rho + mat2.rho) / (2 * A);
                G_int = diag([1/R_int, 1/R_int, 1/R_int, 1/R_int]);
            end
        end
        
        function G_ff = buildFFInterface(mat1, mat2, m1, m2, area, temperature)
            % Build F/F interface conductance matrix
            
            % Interface parameters
            if isfield(mat1, 'G_interface')
                G_base = mat1.G_interface * area;
            else
                G_base = area / (mat1.rho + mat2.rho);  % Default
            end
            
            % Mixing conductances
            if isfield(mat1, 'eta_R')
                eta_R = mat1.eta_R;
                eta_I = mat1.eta_I;
            else
                eta_R = 0.1 * G_base;  % Default values
                eta_I = 0.05 * G_base;
            end
            
            % Magnetization alignment
            cos_theta = dot(m1, m2);
            sin_theta = sqrt(1 - cos_theta^2);
            
            % Build interface matrix
            G_ff = zeros(4, 4);
            
            % Charge conductance
            G_ff(1, 1) = G_base;
            
            % Spin mixing terms
            if sin_theta > 1e-6  % Non-collinear case
                % Real mixing conductance (pumping)
                G_ff(2:4, 2:4) = G_base * eye(3) + eta_R * (eye(3) - m1' * m2);
                
                % Imaginary mixing conductance (precession)
                cross_product = cross(m1, m2);
                G_mixing = eta_I * [0, -cross_product(3), cross_product(2); ...
                                   cross_product(3), 0, -cross_product(1); ...
                                   -cross_product(2), cross_product(1), 0];
                G_ff(2:4, 2:4) = G_ff(2:4, 2:4) + G_mixing;
            else
                % Collinear case
                G_ff(2:4, 2:4) = G_base * eye(3);
            end
        end
        
        function G_fn = buildFNInterface(mat1, mat2, m_vec, area, temperature)
            % Build F/N interface conductance matrix
            
            % Sharvin conductance
            if isfield(mat1, 'G_sharvin')
                G_sharvin = mat1.G_sharvin * area;
            else
                % Default Sharvin conductance
                k_F = sqrt(2 * 9.109e-31 * 10 * 1.602e-19) / 1.055e-34;  % Fermi wavevector
                G_sharvin = (1.602e-19)^2 / (2 * pi * 1.055e-34) * k_F * area;
            end
            
            % Spin polarization
            P = mat1.polarization;
            
            G_fn = zeros(4, 4);
            
            % Charge conductance
            G_fn(1, 1) = G_sharvin;
            
            % Spin-dependent terms
            m_hat = m_vec / norm(m_vec);
            
            % Polarized injection/detection
            for i = 2:4
                G_fn(i, i) = G_sharvin * (1 + P * m_hat(i-1)^2);
                G_fn(1, i) = G_sharvin * P * m_hat(i-1);  % Charge-spin coupling
                G_fn(i, 1) = G_fn(1, i);  % Symmetry
            end
        end
        
        function G_nf = buildNFInterface(mat1, mat2, m_vec, area, temperature)
            % Build N/F interface conductance matrix (transpose of F/N)
            
            G_nf = ConductanceMatrix.buildFNInterface(mat2, mat1, m_vec, area, temperature)';
        end
        
        function R = buildRotationMatrix(m_hat)
            % Build 3x3 rotation matrix to align z-axis with magnetization
            %
            % Input: m_hat - normalized magnetization vector
            % Output: R - 3x3 rotation matrix
            
            % Handle special cases
            if abs(m_hat(3)) > 0.999  % Nearly along z
                R = eye(3);
                if m_hat(3) < 0
                    R(3, 3) = -1;
                end
                return;
            end
            
            % Rodrigues' rotation formula
            % Rotate z-axis to align with m_hat
            z_axis = [0; 0; 1];
            v = cross(z_axis, m_hat);  % Rotation axis
            s = norm(v);               % sin(angle)
            c = dot(z_axis, m_hat);    % cos(angle)
            
            if s < 1e-10  % Parallel vectors
                R = eye(3);
                return;
            end
            
            v = v / s;  % Normalize rotation axis
            
            % Rodrigues' formula: R = I + sin(θ)[v]× + (1-cos(θ))[v]×²
            v_cross = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
            
            R = eye(3) + s * v_cross + (1 - c) * (v_cross * v_cross);
        end
        
        function G_dynamic = buildDynamicTerms(omega, gamma, alpha, m_hat)
            % Build frequency-dependent terms for AC analysis
            
            % Precession matrix
            L_matrix = [0, -m_hat(3), m_hat(2); ...
                       m_hat(3), 0, -m_hat(1); ...
                       -m_hat(2), m_hat(1), 0];
            
            % Dynamic susceptibility
            chi_inv = (alpha + 1i * omega / (gamma * 1)) * eye(3) + 1i * L_matrix;
            chi = inv(chi_inv);
            
            % Build dynamic conductance terms
            G_dynamic = zeros(4, 4);
            G_dynamic(2:4, 2:4) = 1i * omega * chi;
        end
        
        function G_system = assembleSystemMatrix(G_regions, G_interfaces)
            % Assemble complete system conductance matrix
            %
            % This function combines regional and interface conductances
            % into a single system matrix using nodal analysis
            
            n_regions = length(G_regions);
            n_interfaces = length(G_interfaces);
            
            % Total number of nodes (regions + interfaces)
            n_nodes = n_regions + n_interfaces;
            
            % Each node has 4 variables (charge, spin_x, spin_y, spin_z)
            matrix_size = 4 * n_nodes;
            G_system = zeros(matrix_size, matrix_size);
            
            % Add regional conductances (diagonal blocks)
            for i = 1:n_regions
                row_idx = (i-1)*4 + 1:i*4;
                col_idx = row_idx;
                G_system(row_idx, col_idx) = G_regions{i};
            end
            
            % Add interface conductances (off-diagonal coupling)
            for i = 1:n_interfaces
                % Interface connects region i to region i+1
                row_idx1 = (i-1)*4 + 1:i*4;
                col_idx1 = row_idx1;
                row_idx2 = i*4 + 1:(i+1)*4;
                col_idx2 = row_idx2;
                
                G_int = G_interfaces{i};
                
                % Add interface conductance
                G_system(row_idx1, col_idx1) = G_system(row_idx1, col_idx1) + G_int;
                G_system(row_idx2, col_idx2) = G_system(row_idx2, col_idx2) + G_int;
                
                % Add coupling terms
                G_system(row_idx1, col_idx2) = G_system(row_idx1, col_idx2) - G_int;
                G_system(row_idx2, col_idx1) = G_system(row_idx2, col_idx1) - G_int;
            end
        end
        
        function validateConductanceMatrix(G)
            % Validate properties of conductance matrix
            %
            % Checks:
            %   - Hermitian symmetry (for passive systems)
            %   - Current conservation (sum of rows = 0)
            %   - Positive definiteness
            
            [m, n] = size(G);
            
            % Check square matrix
            if m ~= n
                error('Conductance matrix must be square');
            end
            
            % Check current conservation (Kirchhoff's current law)
            row_sums = sum(G, 2);
            if max(abs(row_sums)) > 1e-10
                warning('Current conservation violated. Max row sum: %.2e', max(abs(row_sums)));
            end
            
            % Check symmetry for passive systems
            if max(max(abs(G - G'))) > 1e-10
                warning('Conductance matrix is not symmetric');
            end
            
            % Check positive definiteness (for stability)
            eigenvals = eig(G);
            if min(real(eigenvals)) < 0
                warning('Conductance matrix has negative eigenvalues');
            end
            
            fprintf('Conductance matrix validation completed.\n');
            fprintf('Matrix size: %dx%d\n', m, n);
            fprintf('Condition number: %.2e\n', cond(G));
            fprintf('Minimum eigenvalue: %.2e\n', min(real(eigenvals)));
        end
        
    end
end