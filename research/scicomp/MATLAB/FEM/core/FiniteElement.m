classdef FiniteElement < handle
    % FiniteElement - Base class for finite elements in MATLAB
    %
    % Features:
    % - Abstract base class for all element types
    % - Shape function evaluation
    % - Numerical integration
    % - Stiffness and mass matrix computation
    %
    % Example:
    %   % Cannot instantiate directly - use derived classes
    %   element = LinearBar1D(nodes, material, area);
    properties (Access = protected)
        elementId       % Element identifier
        nodes          % Array of Node objects
        material       % Material property object
        dimension      % Element dimension (1, 2, or 3)
    end
    methods (Abstract)
        % Abstract methods that must be implemented by derived classes
        N = shapeFunction(obj, xi)
        dN = shapeFunctionDerivatives(obj, xi)
        K = stiffnessMatrix(obj)
        M = massMatrix(obj)
    end
    methods
        function obj = FiniteElement(elementId, nodes, material)
            % Constructor
            %
            % Parameters:
            %   elementId: Unique element identifier
            %   nodes: Array of Node objects
            %   material: Material property object
            validateattributes(elementId, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(nodes, {'cell'}, {'vector'});
            obj.elementId = elementId;
            obj.nodes = nodes;
            obj.material = material;
            obj.dimension = length(nodes{1}.coordinates);
            % Validate all nodes have same dimension
            for i = 1:length(nodes)
                if length(nodes{i}.coordinates) ~= obj.dimension
                    error('All nodes must have same dimensionality');
                end
            end
        end
        function J = jacobianMatrix(obj, xi)
            % Compute Jacobian matrix for coordinate transformation
            %
            % Parameters:
            %   xi: Natural coordinates
            %
            % Returns:
            %   J: Jacobian matrix dx/dxi
            dN_dxi = obj.shapeFunctionDerivatives(xi);
            nodeCoords = obj.getNodeCoordinates();
            % J = sum(x_i * dN_i/dxi)
            J = nodeCoords' * dN_dxi;
        end
        function detJ = jacobianDeterminant(obj, xi)
            % Compute Jacobian determinant
            %
            % Parameters:
            %   xi: Natural coordinates
            %
            % Returns:
            %   detJ: Jacobian determinant
            J = obj.jacobianMatrix(xi);
            if size(J, 1) == 1
                detJ = J(1);  % 1D case
            elseif size(J, 1) == size(J, 2)
                detJ = det(J);  % Square matrix
            else
                % Non-square Jacobian (e.g., 2D element in 3D space)
                detJ = sqrt(det(J * J'));
            end
        end
        function coords = globalCoordinates(obj, xi)
            % Map natural coordinates to global coordinates
            %
            % Parameters:
            %   xi: Natural coordinates
            %
            % Returns:
            %   coords: Global coordinates
            N = obj.shapeFunction(xi);
            nodeCoords = obj.getNodeCoordinates();
            coords = N * nodeCoords;
        end
        function [points, weights] = gaussQuadrature(obj, order)
            % Get Gauss quadrature points and weights
            %
            % Parameters:
            %   order: Quadrature order
            %
            % Returns:
            %   points: Integration points
            %   weights: Integration weights
            switch order
                case 1
                    points = 0;
                    weights = 2;
                case 2
                    points = [-1/sqrt(3), 1/sqrt(3)];
                    weights = [1, 1];
                case 3
                    points = [-sqrt(3/5), 0, sqrt(3/5)];
                    weights = [5/9, 8/9, 5/9];
                otherwise
                    error('Quadrature order %d not implemented', order);
            end
            % For higher dimensions, use tensor products
            if obj.dimension > 1
                points1D = points;
                weights1D = weights;
                if obj.dimension == 2
                    [xiGrid, etaGrid] = meshgrid(points1D, points1D);
                    points = [xiGrid(:), etaGrid(:)];
                    [wXi, wEta] = meshgrid(weights1D, weights1D);
                    weights = (wXi .* wEta)';
                    weights = weights(:);
                elseif obj.dimension == 3
                    [xiGrid, etaGrid, zetaGrid] = meshgrid(points1D, points1D, points1D);
                    points = [xiGrid(:), etaGrid(:), zetaGrid(:)];
                    [wXi, wEta, wZeta] = meshgrid(weights1D, weights1D, weights1D);
                    weights = (wXi .* wEta .* wZeta);
                    weights = weights(:);
                end
            end
        end
        function nodeCoords = getNodeCoordinates(obj)
            % Get coordinates of all nodes
            %
            % Returns:
            %   nodeCoords: Matrix of node coordinates
            numNodes = length(obj.nodes);
            nodeCoords = zeros(numNodes, obj.dimension);
            for i = 1:numNodes
                nodeCoords(i, :) = obj.nodes{i}.coordinates;
            end
        end
        function id = getId(obj)
            % Get element ID
            id = obj.elementId;
        end
        function nodes = getNodes(obj)
            % Get element nodes
            nodes = obj.nodes;
        end
        function mat = getMaterial(obj)
            % Get element material
            mat = obj.material;
        end
        function dim = getDimension(obj)
            % Get element dimension
            dim = obj.dimension;
        end
        function disp(obj)
            % Display element information
            fprintf('Finite Element (ID: %d)\n', obj.elementId);
            fprintf('  Type: %s\n', class(obj));
            fprintf('  Nodes: %d\n', length(obj.nodes));
            fprintf('  Dimension: %d\n', obj.dimension);
            fprintf('  Material: %s\n', obj.material.name);
        end
    end
end