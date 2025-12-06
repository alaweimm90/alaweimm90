classdef DataProcessor < handle
    % DATAPROCESSOR - Comprehensive Data Processing Utilities
    %
    % This class provides comprehensive data preprocessing capabilities
    % for machine learning workflows in scientific computing applications.
    %
    % Features:
    %   - Data cleaning and validation
    %   - Feature scaling and normalization
    %   - Missing value handling
    %   - Feature engineering utilities
    %   - Train/validation/test splitting
    %   - Berkeley-themed visualizations
    %
    % Example:
    %   processor = utils.DataProcessor();
    %   [X_clean, y_clean] = processor.cleanData(X, y);
    %   X_scaled = processor.standardScale(X_clean);
    %   [X_train, X_test, y_train, y_test] = processor.trainTestSplit(X_scaled, y_clean, 0.8);
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties (Access = private)
        scalers_
        encoders_
        statistics_
    end
    properties
        HandleMissing = 'mean'     % Missing value strategy
        OutlierMethod = 'iqr'      % Outlier detection method
        ScalingMethod = 'standard' % Feature scaling method
        RandomState = 42           % Random seed
    end
    methods
        function obj = DataProcessor(varargin)
            % Constructor for DataProcessor
            %
            % Parameters:
            %   'HandleMissing' - Missing value strategy ('mean', 'median', 'mode', 'drop') (default: 'mean')
            %   'OutlierMethod' - Outlier detection ('iqr', 'zscore', 'isolation') (default: 'iqr')
            %   'ScalingMethod' - Feature scaling ('standard', 'minmax', 'robust') (default: 'standard')
            %   'RandomState' - Random seed (default: 42)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'HandleMissing', 'mean', @(x) ismember(x, {'mean', 'median', 'mode', 'drop'}));
            addParameter(p, 'OutlierMethod', 'iqr', @(x) ismember(x, {'iqr', 'zscore', 'isolation'}));
            addParameter(p, 'ScalingMethod', 'standard', @(x) ismember(x, {'standard', 'minmax', 'robust'}));
            addParameter(p, 'RandomState', 42, @(x) isnumeric(x) && x >= 0);
            parse(p, varargin{:});
            obj.HandleMissing = p.Results.HandleMissing;
            obj.OutlierMethod = p.Results.OutlierMethod;
            obj.ScalingMethod = p.Results.ScalingMethod;
            obj.RandomState = p.Results.RandomState;
            % Initialize storage
            obj.scalers_ = struct();
            obj.encoders_ = struct();
            obj.statistics_ = struct();
        end
        function [X_clean, y_clean, indices] = cleanData(obj, X, y, varargin)
            % Clean data by handling missing values and outliers
            %
            % Parameters:
            %   X - Feature matrix
            %   y - Target vector
            %   'RemoveOutliers' - Remove outliers (default: true)
            %   'Verbose' - Show cleaning statistics (default: false)
            %
            % Returns:
            %   X_clean - Cleaned feature matrix
            %   y_clean - Cleaned target vector
            %   indices - Indices of kept samples
            p = inputParser;
            addRequired(p, 'X', @ismatrix);
            addRequired(p, 'y', @isvector);
            addParameter(p, 'RemoveOutliers', true, @islogical);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, X, y, varargin{:});
            [nSamples, nFeatures] = size(X);
            y = y(:); % Ensure column vector
            if p.Results.Verbose
                fprintf('Data Cleaning Report\n');
                fprintf('===================\n');
                fprintf('Initial data: %d samples, %d features\n', nSamples, nFeatures);
            end
            % Handle missing values in X
            X_clean = obj.handleMissingValues(X);
            % Handle missing values in y
            missingY = isnan(y) | isinf(y);
            if strcmp(obj.HandleMissing, 'drop')
                validSamples = ~any(isnan(X_clean) | isinf(X_clean), 2) & ~missingY;
            else
                y(missingY) = nanmean(y); % Replace with mean
                validSamples = true(nSamples, 1);
            end
            X_clean = X_clean(validSamples, :);
            y_clean = y(validSamples);
            indices = find(validSamples);
            if p.Results.Verbose
                fprintf('After missing value handling: %d samples\n', size(X_clean, 1));
            end
            % Remove outliers
            if p.Results.RemoveOutliers
                outliers = obj.detectOutliers(X_clean, y_clean);
                nonOutliers = ~outliers;
                X_clean = X_clean(nonOutliers, :);
                y_clean = y_clean(nonOutliers);
                indices = indices(nonOutliers);
                if p.Results.Verbose
                    fprintf('After outlier removal: %d samples\n', size(X_clean, 1));
                    fprintf('Removed %d outliers\n', sum(outliers));
                end
            end
            % Store statistics
            obj.statistics_.originalSamples = nSamples;
            obj.statistics_.cleanSamples = size(X_clean, 1);
            obj.statistics_.removedSamples = nSamples - size(X_clean, 1);
            obj.statistics_.cleaningRatio = size(X_clean, 1) / nSamples;
        end
        function X_scaled = standardScale(obj, X)
            % Standardize features to zero mean and unit variance
            %
            % Parameters:
            %   X - Feature matrix
            %
            % Returns:
            %   X_scaled - Standardized feature matrix
            obj.scalers_.mean = mean(X, 1);
            obj.scalers_.std = std(X, 0, 1);
            % Avoid division by zero
            obj.scalers_.std(obj.scalers_.std == 0) = 1;
            X_scaled = bsxfun(@minus, X, obj.scalers_.mean);
            X_scaled = bsxfun(@rdivide, X_scaled, obj.scalers_.std);
        end
        function X_scaled = minMaxScale(obj, X, varargin)
            % Scale features to specified range
            %
            % Parameters:
            %   X - Feature matrix
            %   'Range' - Target range [min, max] (default: [0, 1])
            %
            % Returns:
            %   X_scaled - Scaled feature matrix
            p = inputParser;
            addRequired(p, 'X', @ismatrix);
            addParameter(p, 'Range', [0, 1], @(x) isnumeric(x) && length(x) == 2);
            parse(p, X, varargin{:});
            obj.scalers_.min = min(X, [], 1);
            obj.scalers_.max = max(X, [], 1);
            obj.scalers_.range = p.Results.Range;
            % Avoid division by zero
            dataRange = obj.scalers_.max - obj.scalers_.min;
            dataRange(dataRange == 0) = 1;
            X_normalized = bsxfun(@minus, X, obj.scalers_.min);
            X_normalized = bsxfun(@rdivide, X_normalized, dataRange);
            targetRange = p.Results.Range(2) - p.Results.Range(1);
            X_scaled = X_normalized * targetRange + p.Results.Range(1);
        end
        function X_scaled = robustScale(obj, X)
            % Scale features using robust statistics (median and IQR)
            %
            % Parameters:
            %   X - Feature matrix
            %
            % Returns:
            %   X_scaled - Robust scaled feature matrix
            obj.scalers_.median = median(X, 1);
            obj.scalers_.q25 = quantile(X, 0.25, 1);
            obj.scalers_.q75 = quantile(X, 0.75, 1);
            obj.scalers_.iqr = obj.scalers_.q75 - obj.scalers_.q25;
            % Avoid division by zero
            obj.scalers_.iqr(obj.scalers_.iqr == 0) = 1;
            X_scaled = bsxfun(@minus, X, obj.scalers_.median);
            X_scaled = bsxfun(@rdivide, X_scaled, obj.scalers_.iqr);
        end
        function X_transformed = applyScaling(obj, X)
            % Apply previously fitted scaling to new data
            %
            % Parameters:
            %   X - Feature matrix
            %
            % Returns:
            %   X_transformed - Scaled feature matrix
            if isempty(obj.scalers_)
                error('No scaler has been fitted');
            end
            switch obj.ScalingMethod
                case 'standard'
                    X_transformed = bsxfun(@minus, X, obj.scalers_.mean);
                    X_transformed = bsxfun(@rdivide, X_transformed, obj.scalers_.std);
                case 'minmax'
                    dataRange = obj.scalers_.max - obj.scalers_.min;
                    X_normalized = bsxfun(@minus, X, obj.scalers_.min);
                    X_normalized = bsxfun(@rdivide, X_normalized, dataRange);
                    targetRange = obj.scalers_.range(2) - obj.scalers_.range(1);
                    X_transformed = X_normalized * targetRange + obj.scalers_.range(1);
                case 'robust'
                    X_transformed = bsxfun(@minus, X, obj.scalers_.median);
                    X_transformed = bsxfun(@rdivide, X_transformed, obj.scalers_.iqr);
            end
        end
        function [X_train, X_test, y_train, y_test, indices] = trainTestSplit(obj, X, y, trainRatio, varargin)
            % Split data into training and testing sets
            %
            % Parameters:
            %   X - Feature matrix
            %   y - Target vector
            %   trainRatio - Fraction of data for training
            %   'Stratify' - Stratified sampling for classification (default: false)
            %   'Shuffle' - Shuffle data before splitting (default: true)
            %
            % Returns:
            %   X_train, X_test, y_train, y_test - Split datasets
            %   indices - Structure with training and test indices
            p = inputParser;
            addRequired(p, 'X', @ismatrix);
            addRequired(p, 'y', @isvector);
            addRequired(p, 'trainRatio', @(x) isnumeric(x) && x > 0 && x < 1);
            addParameter(p, 'Stratify', false, @islogical);
            addParameter(p, 'Shuffle', true, @islogical);
            parse(p, X, y, trainRatio, varargin{:});
            nSamples = size(X, 1);
            y = y(:); % Ensure column vector
            % Set random seed
            rng(obj.RandomState);
            if p.Results.Stratify
                % Stratified sampling
                [trainIdx, testIdx] = obj.stratifiedSplit(y, trainRatio);
            else
                % Random sampling
                if p.Results.Shuffle
                    idx = randperm(nSamples);
                else
                    idx = 1:nSamples;
                end
                nTrain = floor(trainRatio * nSamples);
                trainIdx = idx(1:nTrain);
                testIdx = idx((nTrain+1):end);
            end
            % Split data
            X_train = X(trainIdx, :);
            X_test = X(testIdx, :);
            y_train = y(trainIdx);
            y_test = y(testIdx);
            % Return indices
            indices.train = trainIdx;
            indices.test = testIdx;
        end
        function X_poly = polynomialFeatures(obj, X, degree, varargin)
            % Generate polynomial features
            %
            % Parameters:
            %   X - Feature matrix
            %   degree - Polynomial degree
            %   'InteractionOnly' - Only interaction features (default: false)
            %   'IncludeBias' - Include bias term (default: true)
            %
            % Returns:
            %   X_poly - Polynomial feature matrix
            p = inputParser;
            addRequired(p, 'X', @ismatrix);
            addRequired(p, 'degree', @(x) isnumeric(x) && x > 0 && mod(x,1) == 0);
            addParameter(p, 'InteractionOnly', false, @islogical);
            addParameter(p, 'IncludeBias', true, @islogical);
            parse(p, X, degree, varargin{:});
            [nSamples, nFeatures] = size(X);
            % Start with original features
            X_poly = X;
            % Add bias term if requested
            if p.Results.IncludeBias
                X_poly = [ones(nSamples, 1), X_poly];
            end
            % Generate polynomial terms
            for d = 2:degree
                if p.Results.InteractionOnly
                    % Only interaction terms (no pure powers)
                    newFeatures = obj.generateInteractionFeatures(X, d);
                else
                    % All polynomial terms including powers
                    newFeatures = obj.generatePolynomialFeatures(X, d);
                end
                X_poly = [X_poly, newFeatures];
            end
        end
        function stats = getStatistics(obj)
            % Get data processing statistics
            stats = obj.statistics_;
        end
        function fig = plotDataQuality(obj, X, y, varargin)
            % Plot data quality analysis with Berkeley styling
            %
            % Parameters:
            %   X - Feature matrix
            %   y - Target vector
            %   'Title' - Plot title (default: 'Data Quality Analysis')
            %
            % Returns:
            %   fig - Figure handle
            p = inputParser;
            addRequired(p, 'X', @ismatrix);
            addRequired(p, 'y', @isvector);
            addParameter(p, 'Title', 'Data Quality Analysis', @ischar);
            parse(p, X, y, varargin{:});
            % Berkeley colors
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            fig = figure('Position', [100, 100, 1200, 800]);
            % Missing values heatmap
            subplot(2, 2, 1);
            missingData = isnan(X) | isinf(X);
            imagesc(missingData);
            colormap(gca, [1 1 1; berkeleyBlue]);
            xlabel('Features');
            ylabel('Samples');
            title('Missing Values (blue = missing)');
            % Feature distributions
            subplot(2, 2, 2);
            if size(X, 2) <= 10
                boxplot(X, 'Colors', berkeleyBlue);
                xlabel('Features');
                ylabel('Values');
                title('Feature Distributions');
            else
                histogram(X(:), 'FaceColor', berkeleyBlue, 'EdgeColor', 'none');
                xlabel('Feature Values');
                ylabel('Frequency');
                title('Overall Feature Distribution');
            end
            % Target distribution
            subplot(2, 2, 3);
            histogram(y, 'FaceColor', californiaGold, 'EdgeColor', 'none');
            xlabel('Target Values');
            ylabel('Frequency');
            title('Target Distribution');
            % Correlation matrix (sample of features)
            subplot(2, 2, 4);
            if size(X, 2) <= 20
                corrMatrix = corrcoef(X, 'rows', 'complete');
                imagesc(corrMatrix);
                colorbar;
                xlabel('Features');
                ylabel('Features');
                title('Feature Correlation Matrix');
            else
                % Sample features for large datasets
                sampleFeatures = randperm(size(X, 2), 20);
                corrMatrix = corrcoef(X(:, sampleFeatures), 'rows', 'complete');
                imagesc(corrMatrix);
                colorbar;
                xlabel('Sample Features');
                ylabel('Sample Features');
                title('Sample Feature Correlation');
            end
            sgtitle(p.Results.Title, 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
    methods (Access = private)
        function X_filled = handleMissingValues(obj, X)
            % Handle missing values based on strategy
            switch obj.HandleMissing
                case 'mean'
                    X_filled = fillmissing(X, 'linear');
                    for j = 1:size(X, 2)
                        nanIdx = isnan(X(:, j)) | isinf(X(:, j));
                        if any(nanIdx)
                            X_filled(nanIdx, j) = nanmean(X(:, j));
                        end
                    end
                case 'median'
                    X_filled = X;
                    for j = 1:size(X, 2)
                        nanIdx = isnan(X(:, j)) | isinf(X(:, j));
                        if any(nanIdx)
                            X_filled(nanIdx, j) = nanmedian(X(:, j));
                        end
                    end
                case 'mode'
                    X_filled = X;
                    for j = 1:size(X, 2)
                        nanIdx = isnan(X(:, j)) | isinf(X(:, j));
                        if any(nanIdx)
                            X_filled(nanIdx, j) = mode(X(~nanIdx, j));
                        end
                    end
                case 'drop'
                    X_filled = X; % Will be handled in cleanData
            end
        end
        function outliers = detectOutliers(obj, X, y)
            % Detect outliers using specified method
            [nSamples, nFeatures] = size(X);
            outliers = false(nSamples, 1);
            switch obj.OutlierMethod
                case 'iqr'
                    for j = 1:nFeatures
                        Q1 = quantile(X(:, j), 0.25);
                        Q3 = quantile(X(:, j), 0.75);
                        IQR = Q3 - Q1;
                        lowerBound = Q1 - 1.5 * IQR;
                        upperBound = Q3 + 1.5 * IQR;
                        outliers = outliers | (X(:, j) < lowerBound) | (X(:, j) > upperBound);
                    end
                case 'zscore'
                    zScores = abs(zscore(X));
                    outliers = any(zScores > 3, 2);
                case 'isolation'
                    % Simplified isolation forest (would use proper implementation)
                    scores = zeros(nSamples, 1);
                    for i = 1:nSamples
                        scores(i) = sum(abs(X(i, :) - mean(X, 1)) ./ std(X, 0, 1));
                    end
                    threshold = quantile(scores, 0.95);
                    outliers = scores > threshold;
            end
        end
        function [trainIdx, testIdx] = stratifiedSplit(obj, y, trainRatio)
            % Stratified sampling for classification
            uniqueClasses = unique(y);
            trainIdx = [];
            testIdx = [];
            for i = 1:length(uniqueClasses)
                classIdx = find(y == uniqueClasses(i));
                nClassSamples = length(classIdx);
                nTrain = floor(trainRatio * nClassSamples);
                shuffledIdx = classIdx(randperm(nClassSamples));
                trainIdx = [trainIdx; shuffledIdx(1:nTrain)];
                testIdx = [testIdx; shuffledIdx((nTrain+1):end)];
            end
        end
        function features = generateInteractionFeatures(obj, X, degree)
            % Generate interaction features only
            [nSamples, nFeatures] = size(X);
            % Generate all combinations of degree features
            combinations = nchoosek(1:nFeatures, degree);
            nCombinations = size(combinations, 1);
            features = zeros(nSamples, nCombinations);
            for i = 1:nCombinations
                features(:, i) = prod(X(:, combinations(i, :)), 2);
            end
        end
        function features = generatePolynomialFeatures(obj, X, degree)
            % Generate all polynomial features of given degree
            [nSamples, nFeatures] = size(X);
            % This is a simplified version - full implementation would use
            % multivariate polynomial expansion
            features = [];
            for i = 1:nFeatures
                features = [features, X(:, i).^degree];
            end
        end
    end
end