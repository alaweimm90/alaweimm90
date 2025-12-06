classdef LinearRegression < handle
    % LinearRegression - Advanced linear regression for scientific computing
    %
    % This class implements linear regression with multiple solvers, uncertainty
    % quantification, and scientific computing features designed for the Berkeley
    % SciComp framework.
    %
    % Features:
    %   - Multiple solvers (SVD, normal equation, iterative)
    %   - Uncertainty quantification with confidence intervals
    %   - Regularization support (Ridge regression)
    %   - Statistical inference and diagnostics
    %   - Berkeley-themed visualizations
    %
    % Example:
    %   model = supervised.LinearRegression('Solver', 'svd', 'FitIntercept', true);
    %   model.fit(X, y);
    %   predictions = model.predict(X_test);
    %   score = model.score(X_test, y_test);
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties (Access = private)
        coefficients_
        intercept_
        covarianceMatrix_
        isFitted_ = false
        nFeatures_
    end
    properties
        FitIntercept = true         % Include intercept term
        Solver = 'svd'              % Solver type: 'svd', 'normal', 'iterative'
        Regularization = 0.0        % Regularization parameter
        UncertaintyEstimation = true % Compute uncertainty estimates
    end
    methods
        function obj = LinearRegression(varargin)
            % Constructor for LinearRegression
            %
            % Parameters:
            %   'FitIntercept' - Include intercept (default: true)
            %   'Solver' - Solver type (default: 'svd')
            %   'Regularization' - Regularization parameter (default: 0.0)
            %   'UncertaintyEstimation' - Compute uncertainties (default: true)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'FitIntercept', true, @islogical);
            addParameter(p, 'Solver', 'svd', @(x) ismember(x, {'svd', 'normal', 'iterative'}));
            addParameter(p, 'Regularization', 0.0, @(x) isnumeric(x) && x >= 0);
            addParameter(p, 'UncertaintyEstimation', true, @islogical);
            parse(p, varargin{:});
            obj.FitIntercept = p.Results.FitIntercept;
            obj.Solver = p.Results.Solver;
            obj.Regularization = p.Results.Regularization;
            obj.UncertaintyEstimation = p.Results.UncertaintyEstimation;
        end
        function obj = fit(obj, X, y)
            % Fit linear regression model
            %
            % Parameters:
            %   X - Feature matrix (n_samples x n_features)
            %   y - Target values (n_samples x 1)
            %
            % Returns:
            %   obj - Fitted model object
            % Validate inputs
            [X, y] = obj.validateInput(X, y);
            obj.nFeatures_ = size(X, 2);
            % Add intercept column if needed
            if obj.FitIntercept
                X_aug = [ones(size(X, 1), 1), X];
            else
                X_aug = X;
            end
            % Solve using specified method
            switch obj.Solver
                case 'svd'
                    obj.fitSVD(X_aug, y);
                case 'normal'
                    obj.fitNormalEquation(X_aug, y);
                case 'iterative'
                    obj.fitIterative(X_aug, y);
                otherwise
                    error('Unknown solver: %s', obj.Solver);
            end
            % Compute uncertainty estimates
            if obj.UncertaintyEstimation
                obj.computeUncertainty(X_aug, y);
            end
            obj.isFitted_ = true;
        end
        function predictions = predict(obj, X, varargin)
            % Make predictions using trained model
            %
            % Parameters:
            %   X - Feature matrix
            %   'ReturnUncertainty' - Return prediction uncertainties (default: false)
            %
            % Returns:
            %   predictions - Predicted values
            %   uncertainties - Prediction uncertainties (if requested)
            if ~obj.isFitted_
                error('Model must be fitted before making predictions');
            end
            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'ReturnUncertainty', false, @islogical);
            parse(p, varargin{:});
            X = obj.validateInput(X);
            predictions = X * obj.coefficients_ + obj.intercept_;
            if p.Results.ReturnUncertainty && ~isempty(obj.covarianceMatrix_)
                % Add intercept column for uncertainty calculation
                if obj.FitIntercept
                    X_aug = [ones(size(X, 1), 1), X];
                else
                    X_aug = X;
                end
                % Prediction variance
                predVar = sum((X_aug * obj.covarianceMatrix_) .* X_aug, 2);
                uncertainties = sqrt(predVar);
                predictions = {predictions, uncertainties};
            end
        end
        function ci = confidenceIntervals(obj, X, varargin)
            % Compute confidence intervals for predictions
            %
            % Parameters:
            %   X - Feature matrix
            %   'ConfidenceLevel' - Confidence level (default: 0.95)
            %
            % Returns:
            %   ci - Confidence intervals [lower, upper]
            if ~obj.isFitted_ || isempty(obj.covarianceMatrix_)
                error('Model must be fitted with uncertainty estimation');
            end
            p = inputParser;
            addParameter(p, 'ConfidenceLevel', 0.95, @(x) x > 0 && x < 1);
            parse(p, varargin{:});
            predResult = obj.predict(X, 'ReturnUncertainty', true);
            predictions = predResult{1};
            uncertainties = predResult{2};
            alpha = 1 - p.Results.ConfidenceLevel;
            tVal = tinv(1 - alpha/2, length(predictions) - obj.nFeatures_ - 1);
            margin = tVal * uncertainties;
            ci = [predictions - margin, predictions + margin];
        end
        function r2 = score(obj, X, y)
            % Calculate R² score
            %
            % Parameters:
            %   X - Feature matrix
            %   y - True values
            %
            % Returns:
            %   r2 - R² score
            yPred = obj.predict(X);
            ssRes = sum((y - yPred).^2);
            ssTot = sum((y - mean(y)).^2);
            r2 = 1 - (ssRes / ssTot);
        end
        function summary = getSummary(obj)
            % Get model summary statistics
            %
            % Returns:
            %   summary - Structure with model information
            if ~obj.isFitted_
                error('Model must be fitted first');
            end
            summary.intercept = obj.intercept_;
            summary.coefficients = obj.coefficients_;
            summary.nFeatures = obj.nFeatures_;
            summary.solver = obj.Solver;
            summary.regularization = obj.Regularization;
            if ~isempty(obj.covarianceMatrix_)
                stdErrors = sqrt(diag(obj.covarianceMatrix_));
                if obj.FitIntercept
                    summary.interceptStdError = stdErrors(1);
                    summary.coefficientStdErrors = stdErrors(2:end);
                else
                    summary.coefficientStdErrors = stdErrors;
                end
            end
        end
        function fig = plot(obj, X, y, varargin)
            % Plot regression results with Berkeley styling
            %
            % Parameters:
            %   X - Feature matrix
            %   y - Target values
            %   'Title' - Plot title (default: 'Linear Regression Results')
            %
            % Returns:
            %   fig - Figure handle
            p = inputParser;
            addParameter(p, 'Title', 'Linear Regression Results', @ischar);
            parse(p, varargin{:});
            yPred = obj.predict(X);
            residuals = y - yPred;
            % Berkeley colors
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            fig = figure('Position', [100, 100, 1200, 500]);
            % Plot 1: Predictions vs Actual
            subplot(1, 2, 1);
            scatter(y, yPred, 50, berkeleyBlue, 'filled', 'MarkerFaceAlpha', 0.6);
            hold on;
            % Perfect prediction line
            minVal = min([min(y), min(yPred)]);
            maxVal = max([max(y), max(yPred)]);
            plot([minVal, maxVal], [minVal, maxVal], '--', 'Color', californiaGold, 'LineWidth', 2);
            xlabel('Actual Values');
            ylabel('Predicted Values');
            title('Predictions vs Actual');
            grid on;
            grid minor;
            % Plot 2: Residuals
            subplot(1, 2, 2);
            scatter(yPred, residuals, 50, berkeleyBlue, 'filled', 'MarkerFaceAlpha', 0.6);
            hold on;
            yline(0, '--', 'Color', californiaGold, 'LineWidth', 2);
            xlabel('Predicted Values');
            ylabel('Residuals');
            title('Residual Plot');
            grid on;
            grid minor;
            sgtitle(p.Results.Title, 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
    methods (Access = private)
        function [X, y] = validateInput(obj, X, y)
            % Validate input data
            if nargin < 3
                y = [];
            end
            X = double(X);
            if ndims(X) ~= 2
                error('X must be a 2D matrix');
            end
            if ~isempty(y)
                y = double(y(:));
                if size(X, 1) ~= length(y)
                    error('X and y must have the same number of samples');
                end
            end
        end
        function fitSVD(obj, X, y)
            % Fit using SVD (most stable)
            if obj.Regularization > 0
                % Ridge regression via SVD
                [U, S, V] = svd(X, 'econ');
                s = diag(S);
                d = s ./ (s.^2 + obj.Regularization);
                coeffs = V * (d .* (U' * y));
            else
                coeffs = pinv(X) * y;
            end
            obj.extractCoefficients(coeffs);
        end
        function fitNormalEquation(obj, X, y)
            % Fit using normal equation
            XTX = X' * X;
            if obj.Regularization > 0
                XTX = XTX + obj.Regularization * eye(size(XTX));
            end
            coeffs = XTX \ (X' * y);
            obj.extractCoefficients(coeffs);
        end
        function fitIterative(obj, X, y)
            % Fit using iterative solver (LSQR)
            if obj.Regularization > 0
                % Add regularization term
                X_reg = [X; sqrt(obj.Regularization) * eye(size(X, 2))];
                y_reg = [y; zeros(size(X, 2), 1)];
                coeffs = lsqr(X_reg, y_reg);
            else
                coeffs = lsqr(X, y);
            end
            obj.extractCoefficients(coeffs);
        end
        function extractCoefficients(obj, coeffs)
            % Extract intercept and coefficients
            if obj.FitIntercept
                obj.intercept_ = coeffs(1);
                obj.coefficients_ = coeffs(2:end);
            else
                obj.intercept_ = 0.0;
                obj.coefficients_ = coeffs;
            end
        end
        function computeUncertainty(obj, X, y)
            % Compute uncertainty estimates
            try
                % Compute covariance matrix
                if obj.FitIntercept
                    allCoeffs = [obj.intercept_; obj.coefficients_];
                else
                    allCoeffs = obj.coefficients_;
                end
                residuals = y - X * allCoeffs;
                sigmaSquared = sum(residuals.^2) / (length(y) - size(X, 2));
                XTX_inv = inv(X' * X + obj.Regularization * eye(size(X, 2)));
                obj.covarianceMatrix_ = sigmaSquared * XTX_inv;
            catch
                warning('Could not compute uncertainty estimates due to singular matrix');
                obj.covarianceMatrix_ = [];
            end
        end
    end
end