classdef Node < handle
    % Node - Finite element node representation
    %
    % Features:
    % - Node coordinates in 1D, 2D, or 3D
    % - DOF management
    % - Boundary condition storage
    % - Connectivity tracking
    %
    % Example:
    %   node = Node(1, [0.0, 0.0]);
    %   node.addBoundaryCondition(1, 0.0);  % Fix y-displacement
    properties (Access = private)
        nodeId          % Node identifier
        coordinates     % Node coordinates
        dofIds          % Global DOF identifiers
        boundaryConditions  % Boundary conditions {dof: value}
        connectedElements   % List of connected element IDs
    end
    methods
        function obj = Node(nodeId, coordinates)
            % Constructor
            %
            % Parameters:
            %   nodeId: Unique node identifier
            %   coordinates: Node coordinates [x] or [x,y] or [x,y,z]
            validateattributes(nodeId, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(coordinates, {'numeric'}, {'vector'});
            if length(coordinates) < 1 || length(coordinates) > 3
                error('Coordinates must be 1D, 2D, or 3D');
            end
            obj.nodeId = nodeId;
            obj.coordinates = coordinates(:)';  % Ensure row vector
            obj.dofIds = [];
            obj.boundaryConditions = containers.Map('KeyType', 'int32', 'ValueType', 'double');
            obj.connectedElements = [];
        end
        function id = getId(obj)
            % Get node ID
            id = obj.nodeId;
        end
        function coords = getCoordinates(obj)
            % Get node coordinates
            coords = obj.coordinates;
        end
        function x = getX(obj)
            % Get x-coordinate
            x = obj.coordinates(1);
        end
        function y = getY(obj)
            % Get y-coordinate (0 if 1D)
            if length(obj.coordinates) >= 2
                y = obj.coordinates(2);
            else
                y = 0;
            end
        end
        function z = getZ(obj)
            % Get z-coordinate (0 if 1D or 2D)
            if length(obj.coordinates) >= 3
                z = obj.coordinates(3);
            else
                z = 0;
            end
        end
        function setCoordinates(obj, coordinates)
            % Set node coordinates
            validateattributes(coordinates, {'numeric'}, {'vector'});
            if length(coordinates) ~= length(obj.coordinates)
                error('New coordinates must have same dimension as original');
            end
            obj.coordinates = coordinates(:)';
        end
        function setDofIds(obj, dofIds)
            % Set global DOF identifiers
            validateattributes(dofIds, {'numeric'}, {'vector', 'integer', 'nonnegative'});
            obj.dofIds = dofIds(:)';  % Ensure row vector
        end
        function dofIds = getDofIds(obj)
            % Get global DOF identifiers
            dofIds = obj.dofIds;
        end
        function addBoundaryCondition(obj, localDof, value)
            % Add boundary condition
            %
            % Parameters:
            %   localDof: Local DOF number (1=x, 2=y, 3=z)
            %   value: Prescribed value
            validateattributes(localDof, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(value, {'numeric'}, {'scalar'});
            if localDof > length(obj.coordinates)
                error('Local DOF %d exceeds node dimension %d', localDof, length(obj.coordinates));
            end
            obj.boundaryConditions(localDof) = value;
        end
        function removeBoundaryCondition(obj, localDof)
            % Remove boundary condition
            %
            % Parameters:
            %   localDof: Local DOF number to remove
            if obj.boundaryConditions.isKey(localDof)
                obj.boundaryConditions.remove(localDof);
            end
        end
        function bc = getBoundaryConditions(obj)
            % Get all boundary conditions
            %
            % Returns:
            %   bc: Map of boundary conditions
            bc = obj.boundaryConditions;
        end
        function tf = hasBoundaryCondition(obj, localDof)
            % Check if node has boundary condition on specified DOF
            %
            % Parameters:
            %   localDof: Local DOF number
            %
            % Returns:
            %   tf: True if boundary condition exists
            tf = obj.boundaryConditions.isKey(localDof);
        end
        function value = getBoundaryConditionValue(obj, localDof)
            % Get boundary condition value
            %
            % Parameters:
            %   localDof: Local DOF number
            %
            % Returns:
            %   value: Boundary condition value (empty if not set)
            if obj.boundaryConditions.isKey(localDof)
                value = obj.boundaryConditions(localDof);
            else
                value = [];
            end
        end
        function addConnectedElement(obj, elementId)
            % Add connected element ID
            %
            % Parameters:
            %   elementId: Element ID to add
            validateattributes(elementId, {'numeric'}, {'scalar', 'integer', 'nonnegative'});
            if ~ismember(elementId, obj.connectedElements)
                obj.connectedElements(end+1) = elementId;
            end
        end
        function removeConnectedElement(obj, elementId)
            % Remove connected element ID
            %
            % Parameters:
            %   elementId: Element ID to remove
            obj.connectedElements(obj.connectedElements == elementId) = [];
        end
        function elements = getConnectedElements(obj)
            % Get list of connected element IDs
            elements = obj.connectedElements;
        end
        function numElements = getNumConnectedElements(obj)
            % Get number of connected elements
            numElements = length(obj.connectedElements);
        end
        function dimension = getDimension(obj)
            % Get node dimension
            dimension = length(obj.coordinates);
        end
        function distance = distanceTo(obj, otherNode)
            % Calculate distance to another node
            %
            % Parameters:
            %   otherNode: Another Node object
            %
            % Returns:
            %   distance: Euclidean distance
            if ~isa(otherNode, 'Node')
                error('Input must be a Node object');
            end
            if length(obj.coordinates) ~= length(otherNode.coordinates)
                error('Nodes must have same dimension');
            end
            distance = norm(obj.coordinates - otherNode.coordinates);
        end
        function tf = isEqual(obj, otherNode, tolerance)
            % Check if two nodes are equal within tolerance
            %
            % Parameters:
            %   otherNode: Another Node object
            %   tolerance: Coordinate tolerance (default: 1e-10)
            %
            % Returns:
            %   tf: True if nodes are equal
            if nargin < 3
                tolerance = 1e-10;
            end
            if ~isa(otherNode, 'Node')
                tf = false;
                return;
            end
            if obj.nodeId ~= otherNode.nodeId
                tf = false;
                return;
            end
            if length(obj.coordinates) ~= length(otherNode.coordinates)
                tf = false;
                return;
            end
            tf = all(abs(obj.coordinates - otherNode.coordinates) < tolerance);
        end
        function node_copy = copy(obj)
            % Create a copy of the node
            %
            % Returns:
            %   node_copy: Copy of this node
            node_copy = Node(obj.nodeId, obj.coordinates);
            node_copy.setDofIds(obj.dofIds);
            % Copy boundary conditions
            keys = obj.boundaryConditions.keys;
            for i = 1:length(keys)
                node_copy.addBoundaryCondition(keys{i}, obj.boundaryConditions(keys{i}));
            end
            % Copy connected elements
            for i = 1:length(obj.connectedElements)
                node_copy.addConnectedElement(obj.connectedElements(i));
            end
        end
        function plotNode(obj, varargin)
            % Plot node
            %
            % Optional parameters:
            %   'showId': Show node ID (default: true)
            %   'markerSize': Marker size (default: 8)
            %   'color': Marker color (default: 'blue')
            p = inputParser;
            addParameter(p, 'showId', true, @islogical);
            addParameter(p, 'markerSize', 8, @(x) isnumeric(x) && isscalar(x) && x > 0);
            addParameter(p, 'color', 'blue');
            parse(p, varargin{:});
            if length(obj.coordinates) == 1
                % 1D plot
                plot(obj.coordinates(1), 0, 'o', 'Color', p.Results.color, 'MarkerSize', p.Results.markerSize, 'MarkerFaceColor', p.Results.color);
                if p.Results.showId
                    text(obj.coordinates(1), 0.1, sprintf('%d', obj.nodeId), 'HorizontalAlignment', 'center');
                end
            elseif length(obj.coordinates) == 2
                % 2D plot
                plot(obj.coordinates(1), obj.coordinates(2), 'o', 'Color', p.Results.color, 'MarkerSize', p.Results.markerSize, 'MarkerFaceColor', p.Results.color);
                if p.Results.showId
                    text(obj.coordinates(1), obj.coordinates(2), sprintf('  %d', obj.nodeId));
                end
            else
                % 3D plot
                plot3(obj.coordinates(1), obj.coordinates(2), obj.coordinates(3), 'o', 'Color', p.Results.color, 'MarkerSize', p.Results.markerSize, 'MarkerFaceColor', p.Results.color);
                if p.Results.showId
                    text(obj.coordinates(1), obj.coordinates(2), obj.coordinates(3), sprintf('  %d', obj.nodeId));
                end
            end
        end
        function disp(obj)
            % Display node information
            fprintf('Node %d:\n', obj.nodeId);
            if length(obj.coordinates) == 1
                fprintf('  Coordinates: [%.6f]\n', obj.coordinates(1));
            elseif length(obj.coordinates) == 2
                fprintf('  Coordinates: [%.6f, %.6f]\n', obj.coordinates(1), obj.coordinates(2));
            else
                fprintf('  Coordinates: [%.6f, %.6f, %.6f]\n', obj.coordinates(1), obj.coordinates(2), obj.coordinates(3));
            end
            if ~isempty(obj.dofIds)
                fprintf('  DOF IDs: ');
                fprintf('%d ', obj.dofIds);
                fprintf('\n');
            end
            if obj.boundaryConditions.Count > 0
                fprintf('  Boundary Conditions:\n');
                keys = obj.boundaryConditions.keys;
                for i = 1:length(keys)
                    fprintf('    DOF %d: %.6f\n', keys{i}, obj.boundaryConditions(keys{i}));
                end
            end
            if ~isempty(obj.connectedElements)
                fprintf('  Connected Elements: ');
                fprintf('%d ', obj.connectedElements);
                fprintf('\n');
            end
        end
    end
end