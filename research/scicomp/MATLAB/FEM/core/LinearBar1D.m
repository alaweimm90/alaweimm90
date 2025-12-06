classdef LinearBar1D < FiniteElement
    % LinearBar1D - 1D linear bar element for truss analysis
    %
    % Features:
    % - Axial deformation only
    % - Linear shape functions
    % - 2 nodes, 1 DOF per node
    % - Analytical stiffness and mass matrices
    %
    % Example:
    %   nodes = {Node(1, [0]), Node(2, [1])};
    %   material = Steel();
    %   area = 0.01;  % m²
    %   element = LinearBar1D(1, nodes, material, area);
    properties (Access = private)
        area            % Cross-sectional area
        length          % Element length
    end
    methods
        function obj = LinearBar1D(elementId, nodes, material, area)
            % Constructor
            %
            % Parameters:
            %   elementId: Element identifier
            %   nodes: Cell array of 2 Node objects
            %   material: Material property object
            %   area: Cross-sectional area
            if length(nodes) ~= 2
                error('Bar element requires exactly 2 nodes');
            end
            obj@FiniteElement(elementId, nodes, material);
            validateattributes(area, {'numeric'}, {'scalar', 'positive'});
            obj.area = area;
            % Calculate element length
            coords = obj.getNodeCoordinates();
            obj.length = norm(coords(2, :) - coords(1, :));
            if obj.length <= 0
                error('Element has zero or negative length');
            end
        end
        function N = shapeFunction(obj, xi)
            % Linear shape functions for bar element
            %
            % Parameters:
            %   xi: Natural coordinate in range [-1, 1]
            %
            % Returns:
            %   N: Shape function values [N1, N2]
            xi = xi(:);  % Ensure column vector
            numPoints = length(xi);
            N = zeros(numPoints, 2);
            N(:, 1) = 0.5 * (1 - xi);  % N1
            N(:, 2) = 0.5 * (1 + xi);  % N2
        end
        function dN_dxi = shapeFunctionDerivatives(obj, xi)
            % Shape function derivatives w.r.t. natural coordinates
            %
            % Parameters:
            %   xi: Natural coordinate(s)
            %
            % Returns:
            %   dN_dxi: Shape function derivatives
            xi = xi(:);  % Ensure column vector
            numPoints = length(xi);
            dN_dxi = zeros(numPoints, 2);
            dN_dxi(:, 1) = -0.5;  % dN1/dxi
            dN_dxi(:, 2) = 0.5;   % dN2/dxi
        end
        function K = stiffnessMatrix(obj)
            % Compute element stiffness matrix using analytical integration
            %
            % Returns:
            %   K: 2x2 element stiffness matrix
            E = obj.material.getYoungsModulus();
            A = obj.area;
            L = obj.length;
            % Analytical stiffness matrix for bar element
            k = (E * A / L) * [1, -1; -1, 1];
            K = k;
        end
        function M = massMatrix(obj)
            % Compute element mass matrix
            %
            % Returns:
            %   M: 2x2 element mass matrix (consistent mass)
            rho = obj.material.getDensity();
            A = obj.area;
            L = obj.length;
            % Consistent mass matrix
            m = (rho * A * L / 6) * [2, 1; 1, 2];
            M = m;
        end
        function M_lumped = lumpedMassMatrix(obj)
            % Compute lumped mass matrix
            %
            % Returns:
            %   M_lumped: 2x2 diagonal mass matrix
            rho = obj.material.getDensity();
            A = obj.area;
            L = obj.length;
            % Lumped mass (half total mass at each node)
            nodeMass = rho * A * L / 2;
            M_lumped = diag([nodeMass, nodeMass]);
        end
        function stress = computeStress(obj, displacement)
            % Compute axial stress from nodal displacements
            %
            % Parameters:
            %   displacement: [u1, u2] nodal displacements
            %
            % Returns:
            %   stress: Axial stress (constant throughout element)
            validateattributes(displacement, {'numeric'}, {'vector', 'numel', 2});
            E = obj.material.getYoungsModulus();
            L = obj.length;
            % Strain = du/dx = (u2 - u1) / L
            strain = (displacement(2) - displacement(1)) / L;
            % Stress = E * strain
            stress = E * strain;
        end
        function strain = computeStrain(obj, displacement)
            % Compute axial strain from nodal displacements
            %
            % Parameters:
            %   displacement: [u1, u2] nodal displacements
            %
            % Returns:
            %   strain: Axial strain
            validateattributes(displacement, {'numeric'}, {'vector', 'numel', 2});
            L = obj.length;
            strain = (displacement(2) - displacement(1)) / L;
        end
        function force = internalForce(obj, displacement)
            % Compute internal force from displacement
            %
            % Parameters:
            %   displacement: [u1, u2] nodal displacements
            %
            % Returns:
            %   force: Internal axial force
            stress = obj.computeStress(displacement);
            force = stress * obj.area;
        end
        function area = getArea(obj)
            % Get cross-sectional area
            area = obj.area;
        end
        function length = getLength(obj)
            % Get element length
            length = obj.length;
        end
        function setArea(obj, newArea)
            % Set cross-sectional area
            validateattributes(newArea, {'numeric'}, {'scalar', 'positive'});
            obj.area = newArea;
        end
        function plotElement(obj, varargin)
            % Plot element
            %
            % Optional parameters:
            %   'deformed': Show deformed shape
            %   'displacement': Displacement vector [u1, u2]
            %   'scaleFactor': Scale factor for deformed shape
            p = inputParser;
            addParameter(p, 'deformed', false, @islogical);
            addParameter(p, 'displacement', [0, 0], @(x) isnumeric(x) && length(x) == 2);
            addParameter(p, 'scaleFactor', 1.0, @(x) isnumeric(x) && isscalar(x));
            parse(p, varargin{:});
            coords = obj.getNodeCoordinates();
            x = coords(:, 1);
            if obj.dimension == 1
                y = zeros(size(x));
            else
                y = coords(:, 2);
            end
            % Plot undeformed
            plot(x, y, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Undeformed');
            hold on;
            % Plot deformed if requested
            if p.Results.deformed
                disp = p.Results.displacement * p.Results.scaleFactor;
                x_def = x + [disp(1); disp(2)];
                plot(x_def, y, 'r--s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Deformed');
            end
            xlabel('X Position');
            ylabel('Y Position');
            title(sprintf('Bar Element %d', obj.elementId));
            legend;
            grid on;
            axis equal;
        end
        function disp(obj)
            % Display element information
            fprintf('Linear Bar Element (ID: %d)\n', obj.elementId);
            fprintf('  Length: %.6f\n', obj.length);
            fprintf('  Area: %.6f\n', obj.area);
            fprintf('  Material: %s\n', obj.material.name);
            fprintf('  Nodes: %d\n', length(obj.nodes));
        end
    end
end