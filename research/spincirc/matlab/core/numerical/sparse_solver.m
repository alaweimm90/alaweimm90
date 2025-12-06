function [x, info] = sparse_solver(A, b, varargin)
% SPARSE_SOLVER - High-performance sparse linear system solver
%
% Advanced sparse linear system solver optimized for spintronic device
% simulations. Provides multiple direct and iterative solution methods
% with automatic method selection and preconditioning.
%
% Syntax:
%   x = sparse_solver(A, b)
%   [x, info] = sparse_solver(A, b, 'param', value, ...)
%
% Inputs:
%   A - Sparse coefficient matrix
%   b - Right-hand side vector(s)
%
% Optional Parameters:
%   'Method'      - Solution method ('auto', 'direct', 'iterative', 'multigrid')
%   'Solver'      - Specific solver ('LU', 'Cholesky', 'QR', 'PCG', 'GMRES', 'BiCGSTAB')
%   'Tolerance'   - Convergence tolerance for iterative methods (default: 1e-9)
%   'MaxIter'     - Maximum iterations (default: min(n, 1000))
%   'Restart'     - Restart parameter for GMRES (default: 30)
%   'Precond'     - Preconditioner ('none', 'jacobi', 'ilu', 'ssor', 'amg')
%   'Reordering'  - Matrix reordering ('none', 'amd', 'colamd', 'rcm', 'symrcm')
%   'Verbose'     - Display iteration progress (default: false)
%   'CheckCond'   - Check condition number (default: true)
%   'Refine'      - Iterative refinement steps (default: 0)
%
% Outputs:
%   x    - Solution vector(s)
%   info - Information structure with solver statistics
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Input validation
    if ~issparse(A)
        warning('Matrix is not sparse. Converting to sparse format.');
        A = sparse(A);
    end
    
    if size(A, 1) ~= size(A, 2)
        error('Matrix must be square');
    end
    
    if size(A, 1) ~= size(b, 1)
        error('Matrix and RHS dimensions must match');
    end
    
    % Parse input arguments
    p = inputParser;
    addRequired(p, 'A', @(x) issparse(x) && size(x,1) == size(x,2));
    addRequired(p, 'b', @isnumeric);
    
    addParameter(p, 'Method', 'auto', @(x) any(strcmpi(x, {'auto', 'direct', 'iterative', 'multigrid'})));
    addParameter(p, 'Solver', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Tolerance', 1e-9, @(x) isscalar(x) && x > 0);
    addParameter(p, 'MaxIter', min(size(A,1), 1000), @(x) isscalar(x) && x > 0);
    addParameter(p, 'Restart', 30, @(x) isscalar(x) && x > 0);
    addParameter(p, 'Precond', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Reordering', 'auto', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Verbose', false, @islogical);
    addParameter(p, 'CheckCond', true, @islogical);
    addParameter(p, 'Refine', 0, @(x) isscalar(x) && x >= 0);
    
    parse(p, A, b, varargin{:});
    
    % Extract parameters
    method = p.Results.Method;
    solver = p.Results.Solver;
    tol = p.Results.Tolerance;
    max_iter = p.Results.MaxIter;
    restart = p.Results.Restart;
    precond = p.Results.Precond;
    reordering = p.Results.Reordering;
    verbose = p.Results.Verbose;
    check_cond = p.Results.CheckCond;
    refine_steps = p.Results.Refine;
    
    % Initialize info structure
    info = struct();
    info.method = '';
    info.solver = '';
    info.iterations = 0;
    info.residual = 0;
    info.condition_number = NaN;
    info.solve_time = 0;
    info.setup_time = 0;
    info.success = false;
    
    n = size(A, 1);
    nnz_A = nnz(A);
    
    if verbose
        fprintf('Sparse solver: n = %d, nnz = %d (density = %.2e)\n', ...
            n, nnz_A, nnz_A / n^2);
    end
    
    tic;
    
    % Matrix analysis and automatic method selection
    if strcmpi(method, 'auto') || strcmpi(solver, 'auto')
        [method, solver] = select_method(A, verbose);
    end
    
    % Matrix reordering for better performance
    if strcmpi(reordering, 'auto')
        if issymmetric(A)
            reordering = 'symrcm';
        else
            reordering = 'colamd';
        end
    end
    
    [A_reorder, perm, perm_inv] = apply_reordering(A, reordering);
    b_reorder = b(perm, :);
    
    setup_time = toc;
    info.setup_time = setup_time;
    
    if verbose
        fprintf('Using method: %s, solver: %s, reordering: %s\n', ...
            method, solver, reordering);
    end
    
    % Solve the system
    tic;
    
    switch lower(method)
        case 'direct'
            [x_reorder, solve_info] = solve_direct(A_reorder, b_reorder, solver, check_cond, verbose);
            
        case 'iterative'
            [x_reorder, solve_info] = solve_iterative(A_reorder, b_reorder, solver, ...
                tol, max_iter, restart, precond, verbose);
            
        case 'multigrid'
            [x_reorder, solve_info] = solve_multigrid(A_reorder, b_reorder, tol, max_iter, verbose);
            
        otherwise
            error('Unknown solution method: %s', method);
    end
    
    solve_time = toc;
    
    % Reorder solution back
    x = x_reorder(perm_inv, :);
    
    % Iterative refinement
    if refine_steps > 0
        [x, refine_info] = iterative_refinement(A, b, x, refine_steps, verbose);
        solve_info.residual_after_refinement = refine_info.final_residual;
        solve_info.refinement_steps = refine_info.steps_taken;
    end
    
    % Update info structure
    info.method = method;
    info.solver = solver;
    info.iterations = solve_info.iterations;
    info.residual = solve_info.residual;
    info.condition_number = solve_info.condition_number;
    info.solve_time = solve_time;
    info.success = solve_info.success;
    info.reordering = reordering;
    
    if isfield(solve_info, 'residual_after_refinement')
        info.residual_after_refinement = solve_info.residual_after_refinement;
        info.refinement_steps = solve_info.refinement_steps;
    end
    
    % Final residual check
    final_residual = norm(A*x - b) / norm(b);
    info.final_residual = final_residual;
    
    if verbose
        fprintf('Solution completed in %.3f seconds\n', solve_time + setup_time);
        fprintf('Final residual: %.2e\n', final_residual);
        if ~isnan(info.condition_number)
            fprintf('Condition number: %.2e\n', info.condition_number);
        end
    end
end

function [method, solver] = select_method(A, verbose)
    % Automatic method selection based on matrix properties
    
    n = size(A, 1);
    nnz_A = nnz(A);
    density = nnz_A / n^2;
    
    is_symmetric = issymmetric(A);
    is_positive_definite = false;
    
    if is_symmetric
        % Check positive definiteness (approximate)
        try
            min_eig = eigs(A, 1, 'smallestreal');
            is_positive_definite = (min_eig > 1e-12);
        catch
            % If eigs fails, assume not positive definite
        end
    end
    
    % Decision logic
    if n < 5000 && density > 1e-3
        % Small/dense systems - use direct methods
        method = 'direct';
        if is_positive_definite
            solver = 'Cholesky';
        else
            solver = 'LU';
        end
    elseif n < 50000 && is_symmetric
        % Medium symmetric systems
        method = 'direct';
        if is_positive_definite
            solver = 'Cholesky';
        else
            solver = 'LU';
        end
    elseif is_positive_definite
        % Large positive definite systems - use PCG
        method = 'iterative';
        solver = 'PCG';
    elseif n > 100000
        % Very large systems - consider memory constraints
        method = 'iterative';
        solver = 'BiCGSTAB';  % More memory efficient than GMRES
    else
        % Large general systems - use GMRES
        method = 'iterative';
        solver = 'GMRES';
    end
    
    if verbose
        fprintf('Matrix analysis:\n');
        fprintf('  Size: %d x %d\n', n, n);
        fprintf('  Density: %.2e\n', density);
        fprintf('  Symmetric: %s\n', string(is_symmetric));
        fprintf('  Positive definite: %s\n', string(is_positive_definite));
        fprintf('Selected method: %s, solver: %s\n', method, solver);
    end
end

function [A_reorder, perm, perm_inv] = apply_reordering(A, reordering_method)
    % Apply matrix reordering for better performance
    
    n = size(A, 1);
    
    switch lower(reordering_method)
        case 'none'
            perm = 1:n;
            
        case 'amd'
            perm = amd(A);
            
        case 'colamd'
            perm = colamd(A);
            
        case 'rcm'
            perm = symrcm(A);
            
        case 'symrcm'
            if issymmetric(A)
                perm = symrcm(A);
            else
                perm = symrcm(A + A');
            end
            
        otherwise
            warning('Unknown reordering method: %s. Using no reordering.', reordering_method);
            perm = 1:n;
    end
    
    perm_inv(perm) = 1:n;
    A_reorder = A(perm, perm);
end

function [x, info] = solve_direct(A, b, solver, check_cond, verbose)
    % Solve using direct methods
    
    info = struct();
    info.iterations = 0;
    info.success = false;
    info.condition_number = NaN;
    
    try
        switch upper(solver)
            case 'LU'
                [L, U, P] = lu(A);
                if check_cond
                    info.condition_number = condest(A);
                end
                x = U \ (L \ (P * b));
                
            case 'CHOLESKY'
                if ~issymmetric(A)
                    error('Cholesky factorization requires symmetric matrix');
                end
                R = chol(A);
                if check_cond
                    info.condition_number = condest(A);
                end
                x = R \ (R' \ b);
                
            case 'QR'
                [Q, R] = qr(A);
                if check_cond
                    info.condition_number = condest(A);
                end
                x = R \ (Q' * b);
                
            case 'BACKSLASH'
                if check_cond
                    info.condition_number = condest(A);
                end
                x = A \ b;
                
            otherwise
                error('Unknown direct solver: %s', solver);
        end
        
        info.success = true;
        info.residual = norm(A*x - b) / norm(b);
        
    catch ME
        warning('Direct solver failed: %s', ME.message);
        % Fallback to MATLAB's backslash
        x = A \ b;
        info.success = true;
        info.residual = norm(A*x - b) / norm(b);
    end
    
    if verbose && info.success
        fprintf('Direct solve completed. Residual: %.2e\n', info.residual);
    end
end

function [x, info] = solve_iterative(A, b, solver, tol, max_iter, restart, precond, verbose)
    % Solve using iterative methods
    
    info = struct();
    info.success = false;
    info.condition_number = NaN;
    
    % Set up preconditioner
    [M1, M2] = setup_preconditioner(A, precond, verbose);
    
    % Initial guess
    x0 = zeros(size(b));
    
    switch upper(solver)
        case 'PCG'
            if ~issymmetric(A)
                warning('PCG requires symmetric matrix. Using BiCGSTAB instead.');
                solver = 'BiCGSTAB';
            end
    end
    
    try
        switch upper(solver)
            case 'PCG'
                if ~isempty(M1)
                    [x, flag, relres, iter] = pcg(A, b, tol, max_iter, M1, M2, x0);
                else
                    [x, flag, relres, iter] = pcg(A, b, tol, max_iter, [], [], x0);
                end
                
            case 'GMRES'
                if ~isempty(M1)
                    [x, flag, relres, iter] = gmres(A, b, restart, tol, max_iter/restart, M1, M2, x0);
                else
                    [x, flag, relres, iter] = gmres(A, b, restart, tol, max_iter/restart, [], [], x0);
                end
                
            case 'BICGSTAB'
                if ~isempty(M1)
                    [x, flag, relres, iter] = bicgstab(A, b, tol, max_iter, M1, M2, x0);
                else
                    [x, flag, relres, iter] = bicgstab(A, b, tol, max_iter, [], [], x0);
                end
                
            case 'CGS'
                if ~isempty(M1)
                    [x, flag, relres, iter] = cgs(A, b, tol, max_iter, M1, M2, x0);
                else
                    [x, flag, relres, iter] = cgs(A, b, tol, max_iter, [], [], x0);
                end
                
            otherwise
                error('Unknown iterative solver: %s', solver);
        end
        
        % Interpret results
        info.iterations = iter;
        info.residual = relres;
        info.success = (flag == 0);
        
        if flag ~= 0 && verbose
            fprintf('Warning: Iterative solver flag = %d\n', flag);
        end
        
    catch ME
        warning('Iterative solver failed: %s', ME.message);
        x = zeros(size(b));
        info.iterations = 0;
        info.residual = inf;
        info.success = false;
    end
    
    if verbose
        fprintf('Iterative solve: %d iterations, residual: %.2e\n', ...
            info.iterations, info.residual);
    end
end

function [M1, M2] = setup_preconditioner(A, precond_type, verbose)
    % Set up preconditioner
    
    M1 = [];
    M2 = [];
    
    switch lower(precond_type)
        case {'none', 'auto'}
            if verbose
                fprintf('No preconditioner used\n');
            end
            return;
            
        case 'jacobi'
            M1 = diag(diag(A));
            
        case 'ilu'
            try
                [L, U] = ilu(A);
                M1 = L;
                M2 = U;
            catch
                if verbose
                    fprintf('ILU factorization failed. Using no preconditioner.\n');
                end
            end
            
        case 'ssor'
            % Symmetric SOR preconditioner
            D = diag(diag(A));
            L = tril(A, -1);
            omega = 1.0;  % SOR parameter
            
            M1 = (D + omega*L) / sqrt(omega*(2-omega));
            M2 = M1';
            
        case 'amg'
            % Algebraic multigrid (simplified implementation)
            try
                M1 = setup_simple_amg(A);
            catch
                if verbose
                    fprintf('AMG setup failed. Using no preconditioner.\n');
                end
            end
            
        otherwise
            warning('Unknown preconditioner: %s', precond_type);
    end
    
    if verbose && ~isempty(M1)
        fprintf('Preconditioner: %s\n', precond_type);
    end
end

function M = setup_simple_amg(A)
    % Simplified algebraic multigrid preconditioner
    % This is a very basic implementation - real AMG is much more complex
    
    n = size(A, 1);
    
    if n < 100
        % Base case - use direct solve
        [L, U] = lu(A);
        M = struct('L', L, 'U', U, 'type', 'lu');
    else
        % Coarsening - simple approach
        coarse_indices = 1:2:n;  % Take every other point
        fine_indices = 2:2:n;
        
        % Restriction matrix
        R = sparse(fine_indices, 1:length(fine_indices), 1, n, length(fine_indices));
        
        % Prolongation matrix (transpose of restriction)
        P = R';
        
        % Coarse matrix
        A_coarse = R * A * P;
        
        % Recursive setup
        M_coarse = setup_simple_amg(A_coarse);
        
        M = struct('A', A, 'R', R, 'P', P, 'M_coarse', M_coarse, 'type', 'amg');
    end
end

function [x, info] = solve_multigrid(A, b, tol, max_iter, verbose)
    % Multigrid solver (simplified implementation)
    
    info = struct();
    info.iterations = 0;
    info.success = false;
    
    % For now, fall back to iterative solver
    % A full multigrid implementation would be quite extensive
    
    [x, info] = solve_iterative(A, b, 'BiCGSTAB', tol, max_iter, 30, 'ilu', verbose);
    
    if verbose
        fprintf('Multigrid solver (simplified) completed\n');
    end
end

function [x_refined, info] = iterative_refinement(A, b, x, max_steps, verbose)
    % Iterative refinement for improved accuracy
    
    info = struct();
    info.steps_taken = 0;
    info.initial_residual = norm(A*x - b) / norm(b);
    
    x_refined = x;
    
    for step = 1:max_steps
        % Compute residual
        r = b - A * x_refined;
        residual_norm = norm(r) / norm(b);
        
        if residual_norm < 1e-14
            break;  % Already at machine precision
        end
        
        % Solve correction equation
        try
            delta_x = A \ r;
            x_refined = x_refined + delta_x;
            info.steps_taken = step;
        catch
            break;  % Refinement failed
        end
        
        if verbose
            fprintf('Refinement step %d: residual = %.2e\n', step, residual_norm);
        end
    end
    
    info.final_residual = norm(A*x_refined - b) / norm(b);
    
    if verbose
        fprintf('Iterative refinement: %d steps, final residual: %.2e\n', ...
            info.steps_taken, info.final_residual);
    end
end