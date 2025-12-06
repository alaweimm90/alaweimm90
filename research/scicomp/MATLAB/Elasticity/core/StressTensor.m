classdef StressTensor < handle
    % StressTensor - 3D stress tensor representation and operations
    %
    % Features:
    % - Stress tensor creation and manipulation
    % - Principal stress calculation
    % - Invariant computation
    % - Coordinate transformations
    % - Von Mises stress calculation
    %
    % Example:
    %   stress = StressTensor(100e6, 50e6, 25e6, 10e6, 5e6, 15e6);
    %   principalStresses = stress.principalStresses();
    %   vonMises = stress.vonMisesStress();
    properties (Access = private)
        tensor  % 3x3 stress tensor matrix
    end
    methods
        function obj = StressTensor(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz)
            % Constructor
            %
            % Parameters:
            %   sigma_xx, sigma_yy, sigma_zz: Normal stress components (Pa)
            %   sigma_xy, sigma_xz, sigma_yz: Shear stress components (Pa)
            if nargin < 3
                sigma_zz = 0;
            end
            if nargin < 4
                sigma_xy = 0;
            end
            if nargin < 5
                sigma_xz = 0;
            end
            if nargin < 6
                sigma_yz = 0;
            end
            obj.tensor = [sigma_xx, sigma_xy, sigma_xz; ...
                         sigma_xy, sigma_yy, sigma_yz; ...
                         sigma_xz, sigma_yz, sigma_zz];
        end
        function tensor = getTensor(obj)
            % Get stress tensor matrix
            tensor = obj.tensor;
        end
        function voigt = toVoigt(obj)
            % Convert to Voigt notation [σxx, σyy, σzz, σyz, σxz, σxy]
            voigt = [obj.tensor(1,1); obj.tensor(2,2); obj.tensor(3,3); ...
                    obj.tensor(2,3); obj.tensor(1,3); obj.tensor(1,2)];
        end
        function [principalValues, principalDirections] = principalStresses(obj)
            % Calculate principal stresses and directions
            %
            % Returns:
            %   principalValues: Principal stress values (sorted descending)
            %   principalDirections: Principal stress directions
            [eigenvectors, eigenvalues] = eig(obj.tensor);
            eigenvalues = diag(eigenvalues);
            % Sort in descending order
            [principalValues, idx] = sort(eigenvalues, 'descend');
            principalDirections = eigenvectors(:, idx);
        end
        function [I1, I2, I3] = invariants(obj)
            % Calculate stress tensor invariants
            %
            % Returns:
            %   I1, I2, I3: First, second, and third stress invariants
            s = obj.tensor;
            I1 = trace(s);
            I2 = 0.5 * (trace(s)^2 - trace(s^2));
            I3 = det(s);
        end
        function devStress = deviatoric(obj)
            % Calculate deviatoric stress tensor
            meanStress = trace(obj.tensor) / 3;
            deviatoricTensor = obj.tensor - meanStress * eye(3);
            devStress = StressTensor.fromArray(deviatoricTensor);
        end
        function vonMises = vonMisesStress(obj)
            % Calculate von Mises equivalent stress
            dev = obj.deviatoric();
            devTensor = dev.getTensor();
            vonMises = sqrt(1.5 * sum(sum(devTensor .* devTensor)));
        end
        function maxShear = maximumShearStress(obj)
            % Calculate maximum shear stress
            principalStresses = obj.principalStresses();
            maxShear = (principalStresses(1) - principalStresses(3)) / 2;
        end
        function octShear = octaheralShearStress(obj)
            % Calculate octahedral shear stress
            principalStresses = obj.principalStresses();
            s1 = principalStresses(1);
            s2 = principalStresses(2);
            s3 = principalStresses(3);
            octShear = sqrt((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2) / 3;
        end
        function transformedStress = transform(obj, rotationMatrix)
            % Transform stress tensor to new coordinate system
            %
            % Parameters:
            %   rotationMatrix: 3x3 rotation matrix
            %
            % Returns:
            %   transformedStress: StressTensor in new coordinate system
            validateattributes(rotationMatrix, {'numeric'}, {'size', [3 3]});
            R = rotationMatrix;
            transformedTensor = R * obj.tensor * R';
            transformedStress = StressTensor.fromArray(transformedTensor);
        end
        function result = plus(obj, other)
            % Add stress tensors
            result = StressTensor.fromArray(obj.tensor + other.getTensor());
        end
        function result = minus(obj, other)
            % Subtract stress tensors
            result = StressTensor.fromArray(obj.tensor - other.getTensor());
        end
        function result = mtimes(obj, scalar)
            % Multiply stress tensor by scalar
            result = StressTensor.fromArray(scalar * obj.tensor);
        end
        function disp(obj)
            % Display stress tensor
            fprintf('StressTensor:\n');
            disp(obj.tensor);
        end
    end
    methods (Static)
        function obj = fromArray(tensor)
            % Create StressTensor from 3x3 array
            validateattributes(tensor, {'numeric'}, {'size', [3 3]});
            obj = StressTensor(tensor(1,1), tensor(2,2), tensor(3,3), ...
                              tensor(1,2), tensor(1,3), tensor(2,3));
        end
        function obj = fromVoigt(voigt)
            % Create StressTensor from Voigt notation
            validateattributes(voigt, {'numeric'}, {'numel', 6});
            obj = StressTensor(voigt(1), voigt(2), voigt(3), ...
                              voigt(6), voigt(5), voigt(4));
        end
    end
end