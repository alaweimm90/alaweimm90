classdef Mesh < handle
    % Mesh - Finite element mesh representation
    %
    % Features:
    % - Node and element management
    % - Mesh generation utilities
    % - Quality assessment
    % - Boundary detection
    % - Visualization
    %
    % Example:
    %   mesh = Mesh();
    %   mesh.addNode(Node(1, [0, 0]));
    %   mesh.addNode(Node(2, [1, 0]));
    %   mesh.addElement('bar1d', [1, 2], material);
    properties (Access = private)
        nodes           % Map of Node objects (nodeId -> Node)
        elements        % Map of Element objects (elementId -> Element)
        dimension       % Mesh dimension (1, 2, or 3)
        nodeCounter     % Counter for auto-generating node IDs
        elementCounter  % Counter for auto-generating element IDs
        boundaryNodes   % Map of boundary node groups
    end
    methods
        function obj = Mesh()
            % Constructor
            obj.nodes = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            obj.elements = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            obj.dimension = 2;  % Default to 2D
            obj.nodeCounter = 1;
            obj.elementCounter = 1;
            obj.boundaryNodes = containers.Map();
        end
        function nodeId = addNode(obj, node)
            % Add node to mesh
            %
            % Parameters:
            %   node: Node object or coordinates array
            %
            % Returns:
            %   nodeId: Node identifier
            if isa(node, 'Node')
                nodeId = node.getId();
                obj.nodes(nodeId) = node;
            elseif isnumeric(node)
                % Create node from coordinates
                nodeId = obj.nodeCounter;
                obj.nodes(nodeId) = Node(nodeId, node);
                obj.nodeCounter = obj.nodeCounter + 1;
            else
                error('Input must be Node object or coordinate array');
            end
            % Update mesh dimension
            nodeObj = obj.nodes(nodeId);
            if nodeObj.getDimension() > obj.dimension
                obj.dimension = nodeObj.getDimension();
            end
        end
        function node = getNode(obj, nodeId)
            % Get node by ID
            %
            % Parameters:
            %   nodeId: Node identifier
            %
            % Returns:
            %   node: Node object
            if obj.nodes.isKey(nodeId)
                node = obj.nodes(nodeId);
            else
                error('Node %d not found', nodeId);
            end
        end
        function removeNode(obj, nodeId)
            % Remove node from mesh
            %
            % Parameters:
            %   nodeId: Node identifier
            if obj.nodes.isKey(nodeId)
                % Check if node is used by any elements
                node = obj.nodes(nodeId);
                if node.getNumConnectedElements() > 0
                    error('Cannot remove node %d: still connected to elements', nodeId);
                end
                obj.nodes.remove(nodeId);
            else
                warning('Node %d not found', nodeId);
            end
        end
        function elementId = addElement(obj, elementType, nodeIds, material, varargin)
            % Add element to mesh
            %
            % Parameters:
            %   elementType: Type of element ('bar1d', 'triangle2d', etc.)
            %   nodeIds: Array of node IDs
            %   material: Material object
            %   varargin: Additional element-specific parameters
            %
            % Returns:
            %   elementId: Element identifier
            elementId = obj.elementCounter;
            % Validate that all nodes exist
            for i = 1:length(nodeIds)
                if ~obj.nodes.isKey(nodeIds(i))
                    error('Node %d not found', nodeIds(i));
                end
            end
            % Create element based on type
            switch lower(elementType)
                case 'bar1d'
                    if length(varargin) < 1
                        error('Bar element requires cross-sectional area');
                    end
                    area = varargin{1};
                    nodes = {obj.nodes(nodeIds(1)), obj.nodes(nodeIds(2))};
                    element = LinearBar1D(elementId, nodes, material, area);
                case 'triangle2d'
                    thickness = 1.0;  % Default thickness
                    if length(varargin) >= 1
                        thickness = varargin{1};
                    end
                    nodes = {obj.nodes(nodeIds(1)), obj.nodes(nodeIds(2)), obj.nodes(nodeIds(3))};
                    element = LinearTriangle2D(elementId, nodes, material, thickness);
                case 'quad2d'
                    thickness = 1.0;  % Default thickness
                    if length(varargin) >= 1
                        thickness = varargin{1};
                    end
                    nodes = {obj.nodes(nodeIds(1)), obj.nodes(nodeIds(2)), obj.nodes(nodeIds(3)), obj.nodes(nodeIds(4))};
                    element = LinearQuad2D(elementId, nodes, material, thickness);
                otherwise
                    error('Unknown element type: %s', elementType);
            end
            obj.elements(elementId) = element;
            % Update node connectivity
            for i = 1:length(nodeIds)
                obj.nodes(nodeIds(i)).addConnectedElement(elementId);
            end
            obj.elementCounter = obj.elementCounter + 1;
        end
        function element = getElement(obj, elementId)
            % Get element by ID
            %
            % Parameters:
            %   elementId: Element identifier
            %
            % Returns:
            %   element: Element object
            if obj.elements.isKey(elementId)
                element = obj.elements(elementId);
            else
                error('Element %d not found', elementId);
            end
        end
        function removeElement(obj, elementId)
            % Remove element from mesh
            %
            % Parameters:
            %   elementId: Element identifier
            if obj.elements.isKey(elementId)
                element = obj.elements(elementId);
                % Remove from connected nodes
                nodes = element.getNodes();
                for i = 1:length(nodes)
                    nodes{i}.removeConnectedElement(elementId);
                end
                obj.elements.remove(elementId);
            else
                warning('Element %d not found', elementId);
            end
        end
        function coords = getNodeCoordinates(obj)
            % Get all node coordinates
            %
            % Returns:
            %   coords: Matrix of node coordinates
            nodeIds = cell2mat(obj.nodes.keys);
            numNodes = length(nodeIds);
            if numNodes == 0
                coords = [];
                return;
            end
            % Get first node to determine dimension
            firstNode = obj.nodes(nodeIds(1));
            dim = firstNode.getDimension();
            coords = zeros(numNodes, dim);
            for i = 1:numNodes
                node = obj.nodes(nodeIds(i));
                coords(i, :) = node.getCoordinates();
            end
        end
        function connectivity = getElementConnectivity(obj)
            % Get element connectivity
            %
            % Returns:
            %   connectivity: Cell array of node ID arrays
            elementIds = cell2mat(obj.elements.keys);
            numElements = length(elementIds);
            connectivity = cell(numElements, 1);
            for i = 1:numElements
                element = obj.elements(elementIds(i));
                nodes = element.getNodes();
                nodeIds = zeros(1, length(nodes));
                for j = 1:length(nodes)
                    nodeIds(j) = nodes{j}.getId();
                end
                connectivity{i} = nodeIds;
            end
        end
        function boundaryNodes = findBoundaryNodes(obj, tolerance)
            % Find boundary nodes automatically
            %
            % Parameters:
            %   tolerance: Tolerance for boundary detection (default: 1e-10)
            %
            % Returns:
            %   boundaryNodes: Map of boundary node groups
            if nargin < 2
                tolerance = 1e-10;
            end
            coords = obj.getNodeCoordinates();
            if isempty(coords)
                boundaryNodes = containers.Map();
                return;
            end
            boundaryNodes = containers.Map();
            if obj.dimension == 2
                % Find bounding box
                xMin = min(coords(:, 1));
                xMax = max(coords(:, 1));
                yMin = min(coords(:, 2));
                yMax = max(coords(:, 2));
                % Initialize boundary groups
                leftNodes = [];
                rightNodes = [];
                bottomNodes = [];
                topNodes = [];
                nodeIds = cell2mat(obj.nodes.keys);
                for i = 1:length(nodeIds)
                    node = obj.nodes(nodeIds(i));
                    coords_i = node.getCoordinates();
                    x = coords_i(1);
                    y = coords_i(2);
                    if abs(x - xMin) < tolerance
                        leftNodes(end+1) = nodeIds(i);
                    elseif abs(x - xMax) < tolerance
                        rightNodes(end+1) = nodeIds(i);
                    end
                    if abs(y - yMin) < tolerance
                        bottomNodes(end+1) = nodeIds(i);
                    elseif abs(y - yMax) < tolerance
                        topNodes(end+1) = nodeIds(i);
                    end
                end
                if ~isempty(leftNodes), boundaryNodes('left') = leftNodes; end
                if ~isempty(rightNodes), boundaryNodes('right') = rightNodes; end
                if ~isempty(bottomNodes), boundaryNodes('bottom') = bottomNodes; end
                if ~isempty(topNodes), boundaryNodes('top') = topNodes; end
            elseif obj.dimension == 1
                % For 1D, boundary nodes are endpoints
                xMin = min(coords(:, 1));
                xMax = max(coords(:, 1));
                leftNodes = [];
                rightNodes = [];
                nodeIds = cell2mat(obj.nodes.keys);
                for i = 1:length(nodeIds)
                    node = obj.nodes(nodeIds(i));
                    x = node.getX();
                    if abs(x - xMin) < tolerance
                        leftNodes(end+1) = nodeIds(i);
                    elseif abs(x - xMax) < tolerance
                        rightNodes(end+1) = nodeIds(i);
                    end
                end
                if ~isempty(leftNodes), boundaryNodes('left') = leftNodes; end
                if ~isempty(rightNodes), boundaryNodes('right') = rightNodes; end
            end
            obj.boundaryNodes = boundaryNodes;
        end
        function metrics = meshQualityMetrics(obj)
            % Compute mesh quality metrics
            %
            % Returns:
            %   metrics: Structure with quality metrics
            numNodes = obj.nodes.Count;
            numElements = obj.elements.Count;
            metrics.numNodes = numNodes;
            metrics.numElements = numElements;
            metrics.dimension = obj.dimension;
            if numElements == 0
                return;
            end
            % Element-specific quality metrics
            aspectRatios = [];
            elementIds = cell2mat(obj.elements.keys);
            for i = 1:length(elementIds)
                element = obj.elements(elementIds(i));
                if isa(element, 'LinearTriangle2D')
                    quality = obj.triangleQuality(element);
                    aspectRatios(end+1) = quality.aspectRatio;
                elseif isa(element, 'LinearBar1D')
                    % For bar elements, aspect ratio is not meaningful
                    aspectRatios(end+1) = 1.0;
                end
            end
            if ~isempty(aspectRatios)
                metrics.minAspectRatio = min(aspectRatios);
                metrics.maxAspectRatio = max(aspectRatios);
                metrics.avgAspectRatio = mean(aspectRatios);
                metrics.stdAspectRatio = std(aspectRatios);
            end
        end
        function quality = triangleQuality(obj, element)
            % Calculate quality metrics for triangular element
            %
            % Parameters:
            %   element: Triangle element
            %
            % Returns:
            %   quality: Structure with quality metrics
            nodes = element.getNodes();
            coords = zeros(3, 2);
            for i = 1:3
                coords(i, :) = nodes{i}.getCoordinates();
            end
            % Calculate edge lengths
            edges = [
                norm(coords(2, :) - coords(1, :));
                norm(coords(3, :) - coords(2, :));
                norm(coords(1, :) - coords(3, :))
            ];
            % Area using cross product
            v1 = coords(2, :) - coords(1, :);
            v2 = coords(3, :) - coords(1, :);
            area = 0.5 * abs(v1(1)*v2(2) - v1(2)*v2(1));
            % Aspect ratio (longest edge / shortest edge)
            aspectRatio = max(edges) / min(edges);
            % Skewness (deviation from equilateral triangle)
            optimalArea = (sqrt(3) / 4) * min(edges)^2;
            skewness = abs(area - optimalArea) / optimalArea;
            quality.aspectRatio = aspectRatio;
            quality.skewness = skewness;
            quality.area = area;
            quality.minEdgeLength = min(edges);
            quality.maxEdgeLength = max(edges);
        end
        function numNodes = getNumNodes(obj)
            % Get number of nodes
            numNodes = obj.nodes.Count;
        end
        function numElements = getNumElements(obj)
            % Get number of elements
            numElements = obj.elements.Count;
        end
        function dim = getDimension(obj)
            % Get mesh dimension
            dim = obj.dimension;
        end
        function setDimension(obj, dimension)
            % Set mesh dimension
            validateattributes(dimension, {'numeric'}, {'scalar', 'integer', 'positive', '<=', 3});
            obj.dimension = dimension;
        end
        function plotMesh(obj, varargin)
            % Plot mesh
            %
            % Optional parameters:
            %   'showNodeIds': Show node IDs (default: false)
            %   'showElementIds': Show element IDs (default: false)
            %   'nodeColor': Node color (default: 'red')
            %   'elementColor': Element color (default: 'blue')
            p = inputParser;
            addParameter(p, 'showNodeIds', false, @islogical);
            addParameter(p, 'showElementIds', false, @islogical);
            addParameter(p, 'nodeColor', 'red');
            addParameter(p, 'elementColor', 'blue');
            parse(p, varargin{:});
            figure;
            hold on;
            % Plot elements
            elementIds = cell2mat(obj.elements.keys);
            for i = 1:length(elementIds)
                element = obj.elements(elementIds(i));
                obj.plotElement(element, p.Results.elementColor);
                if p.Results.showElementIds
                    obj.plotElementId(element);
                end
            end
            % Plot nodes
            nodeIds = cell2mat(obj.nodes.keys);
            for i = 1:length(nodeIds)
                node = obj.nodes(nodeIds(i));
                node.plotNode('color', p.Results.nodeColor, 'showId', p.Results.showNodeIds);
            end
            xlabel('X');
            ylabel('Y');
            if obj.dimension == 3
                zlabel('Z');
            end
            title('Finite Element Mesh');
            grid on;
            axis equal;
        end
        function plotElement(obj, element, color)
            % Plot individual element
            %
            % Parameters:
            %   element: Element object
            %   color: Element color
            nodes = element.getNodes();
            coords = zeros(length(nodes), obj.dimension);
            for i = 1:length(nodes)
                nodeCoords = nodes{i}.getCoordinates();
                coords(i, 1:length(nodeCoords)) = nodeCoords;
            end
            if isa(element, 'LinearBar1D')
                if obj.dimension == 1
                    plot(coords(:, 1), zeros(size(coords, 1), 1), '-', 'Color', color, 'LineWidth', 2);
                else
                    plot(coords(:, 1), coords(:, 2), '-', 'Color', color, 'LineWidth', 2);
                end
            elseif isa(element, 'LinearTriangle2D') || isa(element, 'LinearQuad2D')
                % Close the polygon
                coords_closed = [coords; coords(1, :)];
                plot(coords_closed(:, 1), coords_closed(:, 2), '-', 'Color', color, 'LineWidth', 1);
                fill(coords(:, 1), coords(:, 2), color, 'FaceAlpha', 0.1, 'EdgeColor', color);
            end
        end
        function plotElementId(obj, element)
            % Plot element ID at centroid
            %
            % Parameters:
            %   element: Element object
            nodes = element.getNodes();
            coords = zeros(length(nodes), obj.dimension);
            for i = 1:length(nodes)
                nodeCoords = nodes{i}.getCoordinates();
                coords(i, 1:length(nodeCoords)) = nodeCoords;
            end
            centroid = mean(coords, 1);
            if obj.dimension >= 2
                text(centroid(1), centroid(2), sprintf('%d', element.getId()), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'Color', 'blue', 'FontWeight', 'bold');
            else
                text(centroid(1), 0, sprintf('%d', element.getId()), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                     'Color', 'blue', 'FontWeight', 'bold');
            end
        end
        function disp(obj)
            % Display mesh information
            fprintf('Finite Element Mesh:\n');
            fprintf('  Dimension: %dD\n', obj.dimension);
            fprintf('  Nodes: %d\n', obj.nodes.Count);
            fprintf('  Elements: %d\n', obj.elements.Count);
            if obj.boundaryNodes.Count > 0
                fprintf('  Boundary Groups:\n');
                boundaryNames = obj.boundaryNodes.keys;
                for i = 1:length(boundaryNames)
                    nodes = obj.boundaryNodes(boundaryNames{i});
                    fprintf('    %s: %d nodes\n', boundaryNames{i}, length(nodes));
                end
            end
        end
    end
end