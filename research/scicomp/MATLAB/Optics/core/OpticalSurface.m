classdef (Abstract) OpticalSurface < handle
    %OPTICALSURFACE Abstract base class for optical surfaces
    %   Defines interface for optical surface interactions including
    %   ray intersections, refraction, and reflection
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = public)
        position        % Z position of surface (meters)
        diameter        % Clear aperture diameter (meters)
        material_before % Material before surface
        material_after  % Material after surface
    end
    methods
        function obj = OpticalSurface(position, diameter, varargin)
            %OPTICALSURFACE Constructor
            %   surface = OpticalSurface(position, diameter, ...)
            %
            %   Parameters:
            %       'MaterialBefore' - Material before surface (default: 'air')
            %       'MaterialAfter' - Material after surface (default: 'air')
            p = inputParser;
            addRequired(p, 'position', @isnumeric);
            addRequired(p, 'diameter', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'MaterialBefore', 'air', @ischar);
            addParameter(p, 'MaterialAfter', 'air', @ischar);
            parse(p, position, diameter, varargin{:});
            obj.position = p.Results.position;
            obj.diameter = p.Results.diameter;
            obj.material_before = p.Results.MaterialBefore;
            obj.material_after = p.Results.MaterialAfter;
        end
    end
    methods (Abstract)
        % Find intersection point with ray
        [intersection_point, hit] = findIntersection(obj, ray)
        % Calculate surface normal at point
        normal = surfaceNormal(obj, point)
    end
    methods
        function new_ray = refractRay(obj, ray, intersection_point, wavelength)
            %REFRACTRAY Refract ray at surface using Snell's law
            %   new_ray = refractRay(ray, intersection_point, wavelength)
            % Get refractive indices
            n1 = obj.getRefractiveIndex(obj.material_before, wavelength);
            n2 = obj.getRefractiveIndex(obj.material_after, wavelength);
            % Surface normal
            normal = obj.surfaceNormal(intersection_point);
            % Incident direction
            incident_dir = ray.direction;
            % Apply Snell's law in vector form
            cos_theta1 = -dot(incident_dir, normal);
            if cos_theta1 < 0
                % Ray hitting from behind, flip normal
                normal = -normal;
                cos_theta1 = -cos_theta1;
                % Also swap refractive indices
                temp = n1;
                n1 = n2;
                n2 = temp;
            end
            % Check for total internal reflection
            sin_theta1_squared = 1 - cos_theta1^2;
            discriminant = 1 - (n1/n2)^2 * sin_theta1_squared;
            if discriminant < 0
                % Total internal reflection
                new_direction = incident_dir - 2 * dot(incident_dir, normal) * normal;
            else
                % Refraction
                cos_theta2 = sqrt(discriminant);
                new_direction = (n1/n2) * incident_dir + ...
                               ((n1/n2) * cos_theta1 - cos_theta2) * normal;
            end
            % Normalize direction
            new_direction = new_direction / norm(new_direction);
            % Create new ray
            new_ray = Ray(intersection_point, new_direction, wavelength, ray.intensity);
        end
        function new_ray = reflectRay(obj, ray, intersection_point)
            %REFLECTRAY Reflect ray at surface
            %   new_ray = reflectRay(ray, intersection_point)
            % Surface normal
            normal = obj.surfaceNormal(intersection_point);
            % Incident direction
            incident_dir = ray.direction;
            % Ensure normal points toward incident ray
            if dot(incident_dir, normal) > 0
                normal = -normal;
            end
            % Reflection: r = d - 2(d·n)n
            reflected_dir = incident_dir - 2 * dot(incident_dir, normal) * normal;
            % Normalize direction
            reflected_dir = reflected_dir / norm(reflected_dir);
            % Create new ray
            new_ray = Ray(intersection_point, reflected_dir, ray.wavelength, ray.intensity);
        end
        function n = getRefractiveIndex(obj, material, wavelength)
            %GETREFRACTIVEINDEX Get refractive index of material
            %   n = getRefractiveIndex(material, wavelength)
            % Simplified material database
            switch lower(material)
                case 'air'
                    n = 1.000293;
                case 'vacuum'
                    n = 1.0;
                case 'water'
                    n = 1.333;
                case 'bk7'
                    % Simplified BK7 dispersion
                    lambda_um = wavelength * 1e6;
                    n_squared = 1 + 1.03961212 * lambda_um^2 / (lambda_um^2 - 0.00600069867) + ...
                               0.231792344 * lambda_um^2 / (lambda_um^2 - 0.0200179144) + ...
                               1.01046945 * lambda_um^2 / (lambda_um^2 - 103.560653);
                    n = sqrt(n_squared);
                case 'silica'
                    % Simplified fused silica
                    lambda_um = wavelength * 1e6;
                    n_squared = 1 + 0.6961663 * lambda_um^2 / (lambda_um^2 - 0.00467914826) + ...
                               0.4079426 * lambda_um^2 / (lambda_um^2 - 0.0135120631) + ...
                               0.8974794 * lambda_um^2 / (lambda_um^2 - 97.9340025);
                    n = sqrt(n_squared);
                otherwise
                    n = 1.5;  % Default glass
            end
        end
        function is_inside = checkAperture(obj, point)
            %CHECKAPERTURE Check if point is within clear aperture
            %   is_inside = checkAperture(point)
            % Distance from optical axis
            r = sqrt(point(1)^2 + point(2)^2);
            is_inside = r <= obj.diameter / 2;
        end
    end
end