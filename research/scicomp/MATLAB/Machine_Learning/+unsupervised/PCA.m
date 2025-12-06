classdef PCA < handle
    % PCA - Principal Component Analysis for Scientific Computing
    %
    % This class implements Principal Component Analysis with multiple algorithms
    % and comprehensive analysis capabilities for scientific computing applications.
    %
    % Features:
    %   - SVD and eigenvalue decomposition algorithms
    %   - Explained variance analysis
    %   - Data reconstruction capabilities
    %   - Berkeley-themed visualizations
    %   - Scientific computing integration
    %
    % Example:
    %   pca = unsupervised.PCA(2, 'Algorithm', 'svd', 'Whiten', true);
    %   X_transformed = pca.fitTransform(X);
    %   variance_ratio = pca.getExplainedVarianceRatio();
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties (Access = private)
        components_
        explainedVariance_
        explainedVarianceRatio_
        singularValues_
        mean_
        nSamples_
        nFeatures_
        isFitted_ = false
    end
    properties
        NComponents = []        % Number of components to keep
        Algorithm = 'svd'      % Algorithm ('svd', 'eigen')
        Whiten = false          % Whiten the components
        Copy = true             % Copy input data
        RandomState = 42        % Random seed
    end
    methods
        function obj = PCA(nComponents, varargin)
            % Constructor for PCA
            %
            % Parameters:
            %   nComponents - Number of components to keep (default: min(n_samples, n_features))
            %   'Algorithm' - Decomposition algorithm ('svd', 'eigen') (default: 'svd')
            %   'Whiten' - Whiten the components (default: false)
            %   'Copy' - Copy input data (default: true)
            %   'RandomState' - Random seed (default: 42)
            % Parse input arguments
            p = inputParser;
            addOptional(p, 'nComponents', [], @(x) isempty(x) || (isnumeric(x) && x > 0 && mod(x,1) == 0));
            addParameter(p, 'Algorithm', 'svd', @(x) ismember(x, {'svd', 'eigen'}));
            addParameter(p, 'Whiten', false, @islogical);
            addParameter(p, 'Copy', true, @islogical);
            addParameter(p, 'RandomState', 42, @(x) isnumeric(x) && x >= 0);
            parse(p, nComponents, varargin{:});
            obj.NComponents = p.Results.nComponents;
            obj.Algorithm = p.Results.Algorithm;
            obj.Whiten = p.Results.Whiten;
            obj.Copy = p.Results.Copy;
            obj.RandomState = p.Results.RandomState;
        end
        function obj = fit(obj, X)
            % Fit PCA to data
            %
            % Parameters:
            %   X - Data matrix (n_samples x n_features)
            %
            % Returns:
            %   obj - Fitted PCA object
            if ~ismatrix(X)
                error('Input must be a 2D matrix');
            end
            if obj.Copy
                X = X; % Already copied by MATLAB
            end
            [obj.nSamples_, obj.nFeatures_] = size(X);
            % Set default number of components
            if isempty(obj.NComponents)
                obj.NComponents = min(obj.nSamples_, obj.nFeatures_);
            end
            if obj.NComponents > min(obj.nSamples_, obj.nFeatures_)
                error('Number of components cannot exceed min(n_samples, n_features)');
            end
            % Center the data
            obj.mean_ = mean(X, 1);
            X_centered = bsxfun(@minus, X, obj.mean_);
            % Perform decomposition
            switch obj.Algorithm
                case 'svd'
                    obj.fitSVD(X_centered);
                case 'eigen'
                    obj.fitEigen(X_centered);
            end
            obj.isFitted_ = true;
        end
        function X_transformed = transform(obj, X)
            % Transform data to lower dimensional space
            %
            % Parameters:
            %   X - Data matrix (n_samples x n_features)
            %
            % Returns:
            %   X_transformed - Transformed data
            if ~obj.isFitted_
                error('PCA must be fitted before transforming data');
            end
            if size(X, 2) ~= obj.nFeatures_
                error('Number of features must match training data');
            end
            % Center the data
            X_centered = bsxfun(@minus, X, obj.mean_);
            % Project onto principal components
            X_transformed = X_centered * obj.components_';
            % Whiten if requested
            if obj.Whiten
                X_transformed = bsxfun(@rdivide, X_transformed, sqrt(obj.explainedVariance_'));
            end
        end
        function X_transformed = fitTransform(obj, X)
            % Fit PCA and transform data in one step
            %
            % Parameters:
            %   X - Data matrix (n_samples x n_features)
            %
            % Returns:
            %   X_transformed - Transformed data
            obj.fit(X);
            X_transformed = obj.transform(X);
        end
        function X_reconstructed = inverseTransform(obj, X_transformed)
            % Transform data back to original space
            %
            % Parameters:
            %   X_transformed - Transformed data
            %
            % Returns:
            %   X_reconstructed - Reconstructed data
            if ~obj.isFitted_
                error('PCA must be fitted before inverse transforming');
            end
            % Unwhiten if necessary
            if obj.Whiten
                X_transformed = bsxfun(@times, X_transformed, sqrt(obj.explainedVariance_'));
            end
            % Project back to original space
            X_reconstructed = X_transformed * obj.components_ + obj.mean_;
        end
        function components = getComponents(obj)
            % Get principal components
            if ~obj.isFitted_
                error('PCA must be fitted first');
            end
            components = obj.components_;
        end
        function variance = getExplainedVariance(obj)
            % Get explained variance for each component
            if ~obj.isFitted_
                error('PCA must be fitted first');
            end
            variance = obj.explainedVariance_;
        end
        function ratio = getExplainedVarianceRatio(obj)
            % Get explained variance ratio for each component
            if ~obj.isFitted_
                error('PCA must be fitted first');
            end
            ratio = obj.explainedVarianceRatio_;
        end
        function values = getSingularValues(obj)
            % Get singular values
            if ~obj.isFitted_
                error('PCA must be fitted first');
            end
            values = obj.singularValues_;
        end
        function meanVec = getMean(obj)
            % Get mean vector
            if ~obj.isFitted_
                error('PCA must be fitted first');
            end
            meanVec = obj.mean_;
        end
        function fig = plot(obj, varargin)
            % Plot PCA results with Berkeley styling
            %
            % Parameters:
            %   'Type' - Plot type ('variance', 'components', 'biplot') (default: 'variance')
            %   'Title' - Plot title (default: auto-generated)
            %
            % Returns:
            %   fig - Figure handle
            if ~obj.isFitted_
                error('PCA must be fitted first');
            end
            p = inputParser;
            addParameter(p, 'Type', 'variance', @(x) ismember(x, {'variance', 'components', 'biplot'}));
            addParameter(p, 'Title', '', @ischar);
            parse(p, varargin{:});
            % Berkeley colors
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            fig = figure('Position', [100, 100, 800, 600]);
            switch p.Results.Type
                case 'variance'
                    obj.plotVarianceExplained(p.Results.Title);
                case 'components'
                    obj.plotComponents(p.Results.Title);
                case 'biplot'
                    obj.plotBiplot(p.Results.Title);
            end
        end
    end
    methods (Access = private)
        function fitSVD(obj, X_centered)
            % Fit PCA using SVD
            if obj.nSamples_ >= obj.nFeatures_
                % Standard SVD
                [U, S, V] = svd(X_centered, 'econ');
                obj.components_ = V(:, 1:obj.NComponents);
                obj.singularValues_ = diag(S(1:obj.NComponents, 1:obj.NComponents));
            else
                % Compact SVD for wide matrices
                [U, S, V] = svd(X_centered', 'econ');
                obj.components_ = U(:, 1:obj.NComponents);
                obj.singularValues_ = diag(S(1:obj.NComponents, 1:obj.NComponents));
            end
            % Compute explained variance
            obj.explainedVariance_ = (obj.singularValues_.^2) / (obj.nSamples_ - 1);
            obj.explainedVarianceRatio_ = obj.explainedVariance_ / sum(obj.explainedVariance_);
        end
        function fitEigen(obj, X_centered)
            % Fit PCA using eigenvalue decomposition
            % Compute covariance matrix
            if obj.nSamples_ >= obj.nFeatures_
                % Standard covariance
                C = (X_centered' * X_centered) / (obj.nSamples_ - 1);
                [V, D] = eig(C);
            else
                % Use trick for wide matrices
                C = (X_centered * X_centered') / (obj.nSamples_ - 1);
                [U, D] = eig(C);
                V = X_centered' * U / sqrt(obj.nSamples_ - 1);
                % Normalize
                V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 1)));
            end
            % Sort by eigenvalues (descending)
            [eigenvalues, idx] = sort(diag(D), 'descend');
            eigenvectors = V(:, idx);
            % Keep requested number of components
            obj.components_ = eigenvectors(:, 1:obj.NComponents);
            obj.explainedVariance_ = eigenvalues(1:obj.NComponents);
            obj.explainedVarianceRatio_ = obj.explainedVariance_ / sum(eigenvalues);
            obj.singularValues_ = sqrt(obj.explainedVariance_ * (obj.nSamples_ - 1));
        end
        function plotVarianceExplained(obj, titleStr)
            % Plot explained variance
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            components = 1:length(obj.explainedVarianceRatio_);
            yyaxis left;
            bar(components, obj.explainedVarianceRatio_, 'FaceColor', berkeleyBlue, 'EdgeColor', 'none');
            ylabel('Explained Variance Ratio');
            ylim([0, max(obj.explainedVarianceRatio_) * 1.1]);
            yyaxis right;
            plot(components, cumsum(obj.explainedVarianceRatio_), 'o-', ...
                 'Color', californiaGold, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', californiaGold);
            ylabel('Cumulative Explained Variance');
            ylim([0, 1]);
            xlabel('Principal Component');
            if isempty(titleStr)
                title('PCA: Explained Variance Analysis');
            else
                title(titleStr);
            end
            grid on;
            grid minor;
        end
        function plotComponents(obj, titleStr)
            % Plot principal components
            berkeleyBlue = [0, 50, 98]/255;
            nPlots = min(obj.NComponents, 4);
            for i = 1:nPlots
                subplot(2, 2, i);
                bar(1:obj.nFeatures_, obj.components_(:, i), 'FaceColor', berkeleyBlue, 'EdgeColor', 'none');
                xlabel('Feature Index');
                ylabel('Component Weight');
                title(sprintf('PC %d (%.1f%% variance)', i, obj.explainedVarianceRatio_(i) * 100));
                grid on;
                grid minor;
            end
            if isempty(titleStr)
                sgtitle('Principal Components');
            else
                sgtitle(titleStr);
            end
        end
        function plotBiplot(obj, titleStr)
            % Plot biplot (if 2D projection available)
            if obj.NComponents < 2
                error('Biplot requires at least 2 components');
            end
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            % Plot component vectors
            scale = 2;
            for i = 1:min(obj.nFeatures_, 10) % Limit to 10 features for clarity
                arrow = scale * [obj.components_(i, 1), obj.components_(i, 2)];
                plot([0, arrow(1)], [0, arrow(2)], 'Color', californiaGold, 'LineWidth', 2);
                hold on;
                text(arrow(1) * 1.1, arrow(2) * 1.1, sprintf('F%d', i), ...
                     'FontSize', 10, 'Color', californiaGold, 'FontWeight', 'bold');
            end
            xlabel(sprintf('PC 1 (%.1f%% variance)', obj.explainedVarianceRatio_(1) * 100));
            ylabel(sprintf('PC 2 (%.1f%% variance)', obj.explainedVarianceRatio_(2) * 100));
            if isempty(titleStr)
                title('PCA Biplot');
            else
                title(titleStr);
            end
            grid on;
            grid minor;
            axis equal;
        end
    end
end