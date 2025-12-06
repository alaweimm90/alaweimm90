classdef KMeans < handle
    % KMEANS - K-Means Clustering for Scientific Computing
    %
    % This class implements the K-means clustering algorithm with scientific
    % computing focus, including k-means++ initialization, multiple distance
    % metrics, and convergence monitoring.
    %
    % Features:
    %   - K-means++ initialization for better convergence
    %   - Multiple distance metrics (euclidean, manhattan, cosine)
    %   - Convergence monitoring and early stopping
    %   - Berkeley-themed visualizations
    %   - Scientific computing integration
    %
    % Example:
    %   kmeans = unsupervised.KMeans(3, 'Init', 'k-means++', 'MaxIter', 300);
    %   labels = kmeans.fit(X);
    %   centroids = kmeans.getCentroids();
    %   inertia = kmeans.getInertia();
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties (Access = private)
        centroids_
        labels_
        inertia_
        nIter_
        converged_
        isFitted_ = false
    end
    properties
        NClusters = 3           % Number of clusters
        Init = 'k-means++'      % Initialization method
        MaxIter = 300           % Maximum iterations
        Tolerance = 1e-4        % Convergence tolerance
        RandomState = 42        % Random seed
        Metric = 'euclidean'    % Distance metric
    end
    methods
        function obj = KMeans(nClusters, varargin)
            % Constructor for KMeans
            %
            % Parameters:
            %   nClusters - Number of clusters
            %   'Init' - Initialization method ('k-means++', 'random') (default: 'k-means++')
            %   'MaxIter' - Maximum iterations (default: 300)
            %   'Tolerance' - Convergence tolerance (default: 1e-4)
            %   'RandomState' - Random seed (default: 42)
            %   'Metric' - Distance metric ('euclidean', 'manhattan', 'cosine') (default: 'euclidean')
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'nClusters', @(x) isnumeric(x) && x > 0 && mod(x,1) == 0);
            addParameter(p, 'Init', 'k-means++', @(x) ismember(x, {'k-means++', 'random'}));
            addParameter(p, 'MaxIter', 300, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Tolerance', 1e-4, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'RandomState', 42, @(x) isnumeric(x) && x >= 0);
            addParameter(p, 'Metric', 'euclidean', @(x) ismember(x, {'euclidean', 'manhattan', 'cosine'}));
            parse(p, nClusters, varargin{:});
            obj.NClusters = p.Results.nClusters;
            obj.Init = p.Results.Init;
            obj.MaxIter = p.Results.MaxIter;
            obj.Tolerance = p.Results.Tolerance;
            obj.RandomState = p.Results.RandomState;
            obj.Metric = p.Results.Metric;
        end
        function labels = fit(obj, X)
            % Fit K-means clustering to data
            %
            % Parameters:
            %   X - Data matrix (n_samples x n_features)
            %
            % Returns:
            %   labels - Cluster labels
            if ~ismatrix(X) || size(X, 1) < obj.NClusters
                error('Data must be a matrix with at least %d samples', obj.NClusters);
            end
            % Set random seed
            rng(obj.RandomState);
            [nSamples, nFeatures] = size(X);
            % Initialize centroids
            obj.centroids_ = obj.initializeCentroids(X);
            % Initialize labels
            obj.labels_ = zeros(nSamples, 1);
            % K-means iterations
            for iter = 1:obj.MaxIter
                prevCentroids = obj.centroids_;
                % Assign points to clusters
                obj.assignClusters(X);
                % Update centroids
                obj.updateCentroids(X);
                % Check convergence
                centroidChange = obj.computeDistance(obj.centroids_, prevCentroids);
                if max(centroidChange(:)) < obj.Tolerance
                    obj.converged_ = true;
                    obj.nIter_ = iter;
                    break;
                end
                obj.nIter_ = iter;
            end
            % Compute final inertia
            obj.computeInertia(X);
            obj.isFitted_ = true;
            labels = obj.labels_;
        end
        function labels = predict(obj, X)
            % Predict cluster labels for new data
            %
            % Parameters:
            %   X - Data matrix (n_samples x n_features)
            %
            % Returns:
            %   labels - Predicted cluster labels
            if ~obj.isFitted_
                error('Model must be fitted before making predictions');
            end
            nSamples = size(X, 1);
            labels = zeros(nSamples, 1);
            for i = 1:nSamples
                distances = obj.computeDistance(X(i, :), obj.centroids_);
                [~, labels(i)] = min(distances);
            end
        end
        function centroids = getCentroids(obj)
            % Get cluster centroids
            if ~obj.isFitted_
                error('Model must be fitted first');
            end
            centroids = obj.centroids_;
        end
        function inertia = getInertia(obj)
            % Get within-cluster sum of squares (inertia)
            if ~obj.isFitted_
                error('Model must be fitted first');
            end
            inertia = obj.inertia_;
        end
        function nIter = getNIterations(obj)
            % Get number of iterations until convergence
            if ~obj.isFitted_
                error('Model must be fitted first');
            end
            nIter = obj.nIter_;
        end
        function converged = hasConverged(obj)
            % Check if algorithm converged
            if ~obj.isFitted_
                error('Model must be fitted first');
            end
            converged = obj.converged_;
        end
        function fig = plot(obj, X, varargin)
            % Plot clustering results with Berkeley styling
            %
            % Parameters:
            %   X - Data matrix
            %   'Title' - Plot title (default: 'K-Means Clustering')
            %   'ShowCentroids' - Show centroids (default: true)
            %
            % Returns:
            %   fig - Figure handle
            if ~obj.isFitted_
                error('Model must be fitted first');
            end
            if size(X, 2) > 2
                warning('Plotting only first two dimensions of high-dimensional data');
                X = X(:, 1:2);
            end
            p = inputParser;
            addParameter(p, 'Title', 'K-Means Clustering', @ischar);
            addParameter(p, 'ShowCentroids', true, @islogical);
            parse(p, varargin{:});
            % Berkeley colors
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            colors = [berkeleyBlue; californiaGold; [0.8, 0.2, 0.2]; [0.2, 0.8, 0.2]; [0.8, 0.2, 0.8]; [0.2, 0.8, 0.8]];
            fig = figure('Position', [100, 100, 800, 600]);
            % Plot data points colored by cluster
            for k = 1:obj.NClusters
                clusterData = X(obj.labels_ == k, :);
                if ~isempty(clusterData)
                    color = colors(mod(k-1, size(colors, 1)) + 1, :);
                    scatter(clusterData(:, 1), clusterData(:, 2), 40, color, 'filled', ...
                           'MarkerFaceAlpha', 0.7, 'DisplayName', sprintf('Cluster %d', k));
                    hold on;
                end
            end
            % Plot centroids
            if p.Results.ShowCentroids && size(obj.centroids_, 2) >= 2
                scatter(obj.centroids_(:, 1), obj.centroids_(:, 2), 200, 'k', 'x', ...
                       'LineWidth', 3, 'DisplayName', 'Centroids');
            end
            xlabel('Feature 1');
            ylabel('Feature 2');
            title(p.Results.Title);
            legend('Location', 'best');
            grid on;
            grid minor;
        end
    end
    methods (Access = private)
        function centroids = initializeCentroids(obj, X)
            % Initialize centroids using specified method
            [nSamples, nFeatures] = size(X);
            centroids = zeros(obj.NClusters, nFeatures);
            switch obj.Init
                case 'k-means++'
                    % K-means++ initialization
                    centroids(1, :) = X(randi(nSamples), :);
                    for k = 2:obj.NClusters
                        distances = inf(nSamples, 1);
                        % Compute distance to nearest centroid
                        for i = 1:nSamples
                            minDist = inf;
                            for j = 1:(k-1)
                                dist = obj.computeDistance(X(i, :), centroids(j, :));
                                minDist = min(minDist, dist);
                            end
                            distances(i) = minDist;
                        end
                        % Choose next centroid with probability proportional to squared distance
                        probabilities = distances.^2 / sum(distances.^2);
                        cumulativeProb = cumsum(probabilities);
                        r = rand();
                        nextCentroid = find(cumulativeProb >= r, 1);
                        centroids(k, :) = X(nextCentroid, :);
                    end
                case 'random'
                    % Random initialization
                    indices = randperm(nSamples, obj.NClusters);
                    centroids = X(indices, :);
            end
        end
        function assignClusters(obj, X)
            % Assign each point to nearest cluster
            nSamples = size(X, 1);
            for i = 1:nSamples
                distances = obj.computeDistance(X(i, :), obj.centroids_);
                [~, obj.labels_(i)] = min(distances);
            end
        end
        function updateCentroids(obj, X)
            % Update cluster centroids
            for k = 1:obj.NClusters
                clusterPoints = X(obj.labels_ == k, :);
                if ~isempty(clusterPoints)
                    obj.centroids_(k, :) = mean(clusterPoints, 1);
                end
            end
        end
        function distances = computeDistance(obj, x1, x2)
            % Compute distance between points or point to multiple points
            switch obj.Metric
                case 'euclidean'
                    if size(x1, 1) == 1
                        % Single point to multiple points
                        differences = bsxfun(@minus, x2, x1);
                        distances = sqrt(sum(differences.^2, 2));
                    else
                        % Multiple points
                        differences = x1 - x2;
                        distances = sqrt(sum(differences.^2, 2));
                    end
                case 'manhattan'
                    if size(x1, 1) == 1
                        differences = bsxfun(@minus, x2, x1);
                        distances = sum(abs(differences), 2);
                    else
                        differences = x1 - x2;
                        distances = sum(abs(differences), 2);
                    end
                case 'cosine'
                    if size(x1, 1) == 1
                        % Cosine distance = 1 - cosine similarity
                        x1_norm = x1 / norm(x1);
                        x2_norm = bsxfun(@rdivide, x2, sqrt(sum(x2.^2, 2)));
                        cosine_sim = x2_norm * x1_norm';
                        distances = 1 - cosine_sim;
                    else
                        x1_norm = bsxfun(@rdivide, x1, sqrt(sum(x1.^2, 2)));
                        x2_norm = bsxfun(@rdivide, x2, sqrt(sum(x2.^2, 2)));
                        cosine_sim = sum(x1_norm .* x2_norm, 2);
                        distances = 1 - cosine_sim;
                    end
            end
        end
        function computeInertia(obj, X)
            % Compute within-cluster sum of squares
            obj.inertia_ = 0;
            for k = 1:obj.NClusters
                clusterPoints = X(obj.labels_ == k, :);
                if ~isempty(clusterPoints)
                    centroid = obj.centroids_(k, :);
                    distances = obj.computeDistance(clusterPoints, repmat(centroid, size(clusterPoints, 1), 1));
                    obj.inertia_ = obj.inertia_ + sum(distances.^2);
                end
            end
        end
    end
end