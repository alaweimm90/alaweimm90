classdef Ray < handle
    %RAY Optical ray for ray tracing
    %   Represents a light ray with position, direction, wavelength, and intensity
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        position        % Ray position [x, y, z] (meters)
        direction       % Ray direction [dx, dy, dz] (normalized)
        wavelength      % Wavelength (meters)
        intensity       % Ray intensity (W/m²)
        optical_path    % Accumulated optical path length (meters)
        medium_index    % Current medium refractive index
    end
    properties (Constant)
        SPEED_OF_LIGHT = 2.99792458e8  % m/s
    end
    methods
        function obj = Ray(position, direction, wavelength, varargin)
            %RAY Constructor
            %   ray = Ray(position, direction, wavelength, ...)
            %
            %   Inputs:
            %       position - Starting position [x, y, z] (meters)
            %       direction - Direction vector [dx, dy, dz]
            %       wavelength - Wavelength (meters)
            %
            %   Parameters:
            %       'Intensity' - Ray intensity (W/m², default: 1.0)
            %       'MediumIndex' - Medium refractive index (default: 1.0)
            p = inputParser;
            addRequired(p, 'position', @(x) isnumeric(x) && length(x) == 3);
            addRequired(p, 'direction', @(x) isnumeric(x) && length(x) == 3);
            addRequired(p, 'wavelength', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Intensity', 1.0, @(x) isnumeric(x) && x >= 0);
            addParameter(p, 'MediumIndex', 1.0, @(x) isnumeric(x) && x >= 1);
            parse(p, position, direction, wavelength, varargin{:});
            % Set properties
            obj.position = p.Results.position(:)';  % Ensure row vector
            obj.wavelength = p.Results.wavelength;
            obj.intensity = p.Results.Intensity;
            obj.optical_path = 0;
            obj.medium_index = p.Results.MediumIndex;
            % Normalize and set direction
            direction_vec = p.Results.direction(:)';  % Ensure row vector
            direction_norm = norm(direction_vec);
            if direction_norm < 1e-12
                error('Ray:InvalidDirection', 'Direction vector cannot be zero');
            end
            obj.direction = direction_vec / direction_norm;
            obj.validateInput();
        end
        function validateInput(obj)
            %VALIDATEINPUT Validate ray parameters
            if ~all(isfinite(obj.position))
                error('Ray:InvalidPosition', 'Position must contain finite values');
            end
            if ~all(isfinite(obj.direction))
                error('Ray:InvalidDirection', 'Direction must contain finite values');
            end
            if obj.wavelength <= 0 || ~isfinite(obj.wavelength)
                error('Ray:InvalidWavelength', 'Wavelength must be positive and finite');
            end
            if obj.intensity < 0 || ~isfinite(obj.intensity)
                error('Ray:InvalidIntensity', 'Intensity must be non-negative and finite');
            end
        end
        function new_ray = propagateDistance(obj, distance)
            %PROPAGATEDISTANCE Propagate ray by specified distance
            %   new_ray = propagateDistance(distance)
            %
            %   Input:
            %       distance - Propagation distance (meters)
            %
            %   Output:
            %       new_ray - New Ray object at propagated position
            if distance < 0
                warning('Ray:NegativeDistance', 'Negative propagation distance');
            end
            % New position
            new_position = obj.position + distance * obj.direction;
            % Create new ray
            new_ray = Ray(new_position, obj.direction, obj.wavelength, ...
                'Intensity', obj.intensity, 'MediumIndex', obj.medium_index);
            % Update optical path
            new_ray.optical_path = obj.optical_path + distance * obj.medium_index;
        end
        function new_ray = propagateToPlane(obj, plane_normal, plane_point)
            %PROPAGATETOPLANE Propagate ray to a plane
            %   new_ray = propagateToPlane(plane_normal, plane_point)
            %
            %   Inputs:
            %       plane_normal - Normal vector to plane
            %       plane_point - Point on plane
            %
            %   Output:
            %       new_ray - Ray at plane intersection (empty if no intersection)
            % Normalize plane normal
            plane_normal = plane_normal(:)' / norm(plane_normal);
            plane_point = plane_point(:)';
            % Check if ray is parallel to plane
            denominator = dot(obj.direction, plane_normal);
            if abs(denominator) < 1e-12
                % Ray is parallel to plane
                new_ray = [];
                return;
            end
            % Calculate intersection distance
            numerator = dot(plane_point - obj.position, plane_normal);
            distance = numerator / denominator;
            if distance < 1e-12
                % Intersection behind ray start
                new_ray = [];
                return;
            end
            % Propagate to intersection
            new_ray = obj.propagateDistance(distance);
        end
        function new_ray = propagateToZ(obj, z_position)
            %PROPAGATETOZ Propagate ray to specific z-coordinate
            %   new_ray = propagateToZ(z_position)
            %
            %   Input:
            %       z_position - Target z-coordinate (meters)
            %
            %   Output:
            %       new_ray - Ray at z-position (empty if no forward intersection)
            if abs(obj.direction(3)) < 1e-12
                % Ray parallel to z-plane
                new_ray = [];
                return;
            end
            % Calculate propagation distance
            distance = (z_position - obj.position(3)) / obj.direction(3);
            if distance < 1e-12
                % Target behind ray
                new_ray = [];
                return;
            end
            new_ray = obj.propagateDistance(distance);
        end
        function angle = getAngleWithDirection(obj, reference_direction)
            %GETANGLEWITHDIRECTION Get angle between ray and reference direction
            %   angle = getAngleWithDirection(reference_direction)
            %
            %   Input:
            %       reference_direction - Reference direction vector
            %
            %   Output:
            %       angle - Angle in radians
            reference_direction = reference_direction(:)' / norm(reference_direction);
            cos_angle = dot(obj.direction, reference_direction);
            % Clamp to valid range for acos
            cos_angle = max(-1, min(1, cos_angle));
            angle = acos(cos_angle);
        end
        function angle = getAngleWithAxis(obj, axis)
            %GETANGLEWITHAXIS Get angle between ray and coordinate axis
            %   angle = getAngleWithAxis(axis)
            %
            %   Input:
            %       axis - Axis ('x', 'y', 'z', or axis index 1-3)
            %
            %   Output:
            %       angle - Angle in radians
            if ischar(axis)
                switch lower(axis)
                    case 'x'
                        axis_vector = [1, 0, 0];
                    case 'y'
                        axis_vector = [0, 1, 0];
                    case 'z'
                        axis_vector = [0, 0, 1];
                    otherwise
                        error('Ray:InvalidAxis', 'Axis must be ''x'', ''y'', or ''z''');
                end
            elseif isnumeric(axis) && isscalar(axis) && axis >= 1 && axis <= 3
                axis_vector = zeros(1, 3);
                axis_vector(axis) = 1;
            else
                error('Ray:InvalidAxis', 'Invalid axis specification');
            end
            angle = obj.getAngleWithDirection(axis_vector);
        end
        function distance = getDistanceToPoint(obj, point)
            %GETDISTANCETOPOINT Get perpendicular distance from ray to point
            %   distance = getDistanceToPoint(point)
            %
            %   Input:
            %       point - Point coordinates [x, y, z]
            %
            %   Output:
            %       distance - Perpendicular distance (meters)
            point = point(:)';
            % Vector from ray origin to point
            to_point = point - obj.position;
            % Project onto ray direction
            projection_length = dot(to_point, obj.direction);
            projection = projection_length * obj.direction;
            % Perpendicular component
            perpendicular = to_point - projection;
            distance = norm(perpendicular);
        end
        function [closest_point, distance_along_ray] = getClosestPointToPoint(obj, point)
            %GETCLOSESTPOINTTOPOINT Get closest point on ray to given point
            %   [closest_point, distance_along_ray] = getClosestPointToPoint(point)
            %
            %   Input:
            %       point - Point coordinates [x, y, z]
            %
            %   Outputs:
            %       closest_point - Closest point on ray
            %       distance_along_ray - Distance along ray to closest point
            point = point(:)';
            % Vector from ray origin to point
            to_point = point - obj.position;
            % Project onto ray direction
            distance_along_ray = dot(to_point, obj.direction);
            closest_point = obj.position + distance_along_ray * obj.direction;
        end
        function phase = getPhase(obj, reference_position)
            %GETPHASE Get optical phase relative to reference position
            %   phase = getPhase(reference_position)
            %
            %   Input:
            %       reference_position - Reference position [x, y, z]
            %
            %   Output:
            %       phase - Phase in radians
            reference_position = reference_position(:)';
            % Path difference
            path_difference = obj.optical_path - norm(reference_position - obj.position) * obj.medium_index;
            % Wave number in medium
            k = 2 * pi / (obj.wavelength / obj.medium_index);
            phase = k * path_difference;
        end
        function new_ray = changeDirection(obj, new_direction)
            %CHANGEDIRECTION Create new ray with different direction
            %   new_ray = changeDirection(new_direction)
            %
            %   Input:
            %       new_direction - New direction vector
            %
            %   Output:
            %       new_ray - New Ray object with same position but new direction
            new_ray = Ray(obj.position, new_direction, obj.wavelength, ...
                'Intensity', obj.intensity, 'MediumIndex', obj.medium_index);
            new_ray.optical_path = obj.optical_path;
        end
        function new_ray = changeMedium(obj, new_medium_index)
            %CHANGEMEDIUM Create new ray in different medium
            %   new_ray = changeMedium(new_medium_index)
            %
            %   Input:
            %       new_medium_index - New medium refractive index
            %
            %   Output:
            %       new_ray - New Ray object in new medium
            new_ray = Ray(obj.position, obj.direction, obj.wavelength, ...
                'Intensity', obj.intensity, 'MediumIndex', new_medium_index);
            new_ray.optical_path = obj.optical_path;
        end
        function info = getInfo(obj)
            %GETINFO Get ray information structure
            %   info = getInfo()
            info = struct();
            info.position = obj.position;
            info.direction = obj.direction;
            info.wavelength = obj.wavelength;
            info.wavelength_nm = obj.wavelength * 1e9;
            info.frequency = obj.SPEED_OF_LIGHT / obj.wavelength;
            info.intensity = obj.intensity;
            info.optical_path = obj.optical_path;
            info.medium_index = obj.medium_index;
            % Direction angles
            info.angle_x = obj.getAngleWithAxis('x') * 180/pi;  % degrees
            info.angle_y = obj.getAngleWithAxis('y') * 180/pi;
            info.angle_z = obj.getAngleWithAxis('z') * 180/pi;
        end
        function plotRay(obj, varargin)
            %PLOTRAY Plot ray in 3D
            %   plotRay(...)
            %
            %   Parameters:
            %       'Length' - Ray length to plot (meters, default: 0.1)
            %       'Color' - Ray color (default: 'blue')
            %       'LineWidth' - Line width (default: 2)
            %       'ShowDirection' - Show direction arrow (default: true)
            p = inputParser;
            addParameter(p, 'Length', 0.1, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Color', 'blue', @(x) ischar(x) || isnumeric(x));
            addParameter(p, 'LineWidth', 2, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'ShowDirection', true, @islogical);
            parse(p, varargin{:});
            % End position
            end_position = obj.position + p.Results.Length * obj.direction;
            % Plot ray line
            hold on;
            plot3([obj.position(1), end_position(1)], ...
                  [obj.position(2), end_position(2)], ...
                  [obj.position(3), end_position(3)], ...
                  'Color', p.Results.Color, 'LineWidth', p.Results.LineWidth);
            % Show direction arrow
            if p.Results.ShowDirection
                arrow_start = obj.position + 0.7 * p.Results.Length * obj.direction;
                arrow_end = obj.position + 0.9 * p.Results.Length * obj.direction;
                % Simple arrow using quiver3
                quiver3(arrow_start(1), arrow_start(2), arrow_start(3), ...
                       (arrow_end(1) - arrow_start(1)), ...
                       (arrow_end(2) - arrow_start(2)), ...
                       (arrow_end(3) - arrow_start(3)), ...
                       0, 'Color', p.Results.Color, 'LineWidth', p.Results.LineWidth, ...
                       'MaxHeadSize', 0.3);
            end
            hold off;
        end
        function displayInfo(obj)
            %DISPLAYINFO Display ray information
            fprintf('Ray Information:\n');
            fprintf('================\n');
            fprintf('Position: [%.3f, %.3f, %.3f] mm\n', obj.position * 1000);
            fprintf('Direction: [%.3f, %.3f, %.3f]\n', obj.direction);
            fprintf('Wavelength: %.1f nm\n', obj.wavelength * 1e9);
            fprintf('Intensity: %.2e W/m²\n', obj.intensity);
            fprintf('Medium index: %.6f\n', obj.medium_index);
            fprintf('Optical path: %.3f mm\n', obj.optical_path * 1000);
            % Direction angles
            fprintf('\nDirection angles:\n');
            fprintf('With x-axis: %.1f°\n', obj.getAngleWithAxis('x') * 180/pi);
            fprintf('With y-axis: %.1f°\n', obj.getAngleWithAxis('y') * 180/pi);
            fprintf('With z-axis: %.1f°\n', obj.getAngleWithAxis('z') * 180/pi);
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstrate ray functionality
            fprintf('Ray Demo\n');
            fprintf('========\n\n');
            % Create test ray
            position = [0.001, 0, -0.05];      % 1 mm off-axis, 50 mm before origin
            direction = [0.02, 0, 1];          % Slight angle toward axis
            wavelength = 633e-9;               % HeNe laser
            intensity = 1000;                  % W/m²
            ray = Ray(position, direction, wavelength, 'Intensity', intensity);
            fprintf('Created ray:\n');
            ray.displayInfo();
            % Test propagation
            fprintf('\n1. Ray Propagation Test\n');
            fprintf('-----------------------\n');
            % Propagate 100 mm
            distance = 0.1;
            propagated_ray = ray.propagateDistance(distance);
            fprintf('After propagating %.0f mm:\n', distance * 1000);
            fprintf('New position: [%.3f, %.3f, %.3f] mm\n', propagated_ray.position * 1000);
            fprintf('Optical path: %.3f mm\n', propagated_ray.optical_path * 1000);
            % Propagate to z-plane
            z_target = 0.05;  % 50 mm
            ray_at_z = ray.propagateToZ(z_target);
            if ~isempty(ray_at_z)
                fprintf('\nPropagated to z = %.0f mm:\n', z_target * 1000);
                fprintf('Position: [%.3f, %.3f, %.3f] mm\n', ray_at_z.position * 1000);
            end
            % Test angles
            fprintf('\n2. Angle Calculations\n');
            fprintf('--------------------\n');
            angle_z = ray.getAngleWithAxis('z') * 180/pi;
            fprintf('Angle with z-axis: %.2f°\n', angle_z);
            % Distance to point
            test_point = [0, 0, 0];
            distance_to_origin = ray.getDistanceToPoint(test_point);
            fprintf('Distance to origin: %.3f mm\n', distance_to_origin * 1000);
            % Plot ray
            fprintf('\n3. Ray Visualization\n');
            fprintf('-------------------\n');
            figure('Position', [100, 100, 800, 600]);
            ray.plotRay('Length', 0.15, 'Color', 'red', 'ShowDirection', true);
            % Add coordinate system
            hold on;
            plot3([0, 0.02], [0, 0], [0, 0], 'r-', 'LineWidth', 2);  % x-axis
            plot3([0, 0], [0, 0.02], [0, 0], 'g-', 'LineWidth', 2);  % y-axis
            plot3([0, 0], [0, 0], [0, 0.02], 'b-', 'LineWidth', 2);  % z-axis
            text(0.021, 0, 0, 'X', 'FontSize', 12, 'Color', 'red');
            text(0, 0.021, 0, 'Y', 'FontSize', 12, 'Color', 'green');
            text(0, 0, 0.021, 'Z', 'FontSize', 12, 'Color', 'blue');
            xlabel('X (m)');
            ylabel('Y (m)');
            zlabel('Z (m)');
            title('Ray Visualization');
            grid on;
            axis equal;
            view(45, 30);
            hold off;
            fprintf('Ray plotted in 3D\n');
            fprintf('\nDemo completed!\n');
        end
    end
end