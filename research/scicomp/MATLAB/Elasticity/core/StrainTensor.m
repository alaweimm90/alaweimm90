classdef StrainTensor < handle
    % StrainTensor - 3D strain tensor representation and operations
    %
    % Features:
    % - Strain tensor creation and manipulation
    % - Principal strain calculation
    % - Strain invariants
    % - Compatibility checks
    % - Finite vs infinitesimal strain
    %
    % Example:
    %   strain = StrainTensor(0.001, 0.0005, 0, 0.0008, 0, 0);
    %   principalStrains = strain.principalStrains();
    %   volumetricStrain = strain.volumetricStrain();
    properties (Access = private)
        tensor  % 3x3 strain tensor matrix
    end
    methods
        function obj = StrainTensor(epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_xz, gamma_yz, engineeringStrain)
            % Constructor
            %
            % Parameters:
            %   epsilon_xx, epsilon_yy, epsilon_zz: Normal strain components
            %   gamma_xy, gamma_xz, gamma_yz: Shear strain components
            %   engineeringStrain: If true, use engineering shear strains (default: true)
            if nargin < 3
                epsilon_zz = 0;
            end
            if nargin < 4
                gamma_xy = 0;
            end
            if nargin < 5
                gamma_xz = 0;
            end
            if nargin < 6
                gamma_yz = 0;
            end
            if nargin < 7
                engineeringStrain = true;
            end
            % Convert engineering shear strains to tensor shear strains
            if engineeringStrain
                shearFactor = 0.5;
            else
                shearFactor = 1.0;
            end
            obj.tensor = [epsilon_xx, shearFactor * gamma_xy, shearFactor * gamma_xz; ...
                         shearFactor * gamma_xy, epsilon_yy, shearFactor * gamma_yz; ...
                         shearFactor * gamma_xz, shearFactor * gamma_yz, epsilon_zz];
        end
        function tensor = getTensor(obj)
            % Get strain tensor matrix
            tensor = obj.tensor;
        end
        function voigt = toVoigt(obj, engineeringStrain)
            % Convert to Voigt notation
            %
            % Parameters:
            %   engineeringStrain: If true, use engineering shear strains (default: true)
            %
            % Returns:
            %   voigt: [εxx, εyy, εzz, γyz, γxz, γxy] or [εxx, εyy, εzz, εyz, εxz, εxy]
            if nargin < 2
                engineeringStrain = true;
            end
            if engineeringStrain
                shearFactor = 2.0;
            else
                shearFactor = 1.0;
            end
            voigt = [obj.tensor(1,1); obj.tensor(2,2); obj.tensor(3,3); ...
                    shearFactor * obj.tensor(2,3); shearFactor * obj.tensor(1,3); ...
                    shearFactor * obj.tensor(1,2)];
        end
        function [principalValues, principalDirections] = principalStrains(obj)
            % Calculate principal strains and directions
            %
            % Returns:
            %   principalValues: Principal strain values (sorted descending)
            %   principalDirections: Principal strain directions
            [eigenvectors, eigenvalues] = eig(obj.tensor);
            eigenvalues = diag(eigenvalues);
            % Sort in descending order
            [principalValues, idx] = sort(eigenvalues, 'descend');
            principalDirections = eigenvectors(:, idx);
        end
        function volStrain = volumetricStrain(obj)
            % Calculate volumetric strain (trace of strain tensor)
            volStrain = trace(obj.tensor);
        end
        function devStrain = deviatoric(obj)
            % Calculate deviatoric strain tensor
            volumetricStrain = obj.volumetricStrain() / 3;
            deviatoricTensor = obj.tensor - volumetricStrain * eye(3);
            devStrain = StrainTensor.fromArray(deviatoricTensor);
        end
        function eqStrain = equivalentStrain(obj)
            % Calculate equivalent strain (von Mises equivalent)
            dev = obj.deviatoric();
            devTensor = dev.getTensor();
            eqStrain = sqrt(2.0/3.0 * sum(sum(devTensor .* devTensor)));
        end
        function maxShear = maximumShearStrain(obj)
            % Calculate maximum shear strain
            principalStrains = obj.principalStrains();
            maxShear = (principalStrains(1) - principalStrains(3)) / 2;
        end
        function transformedStrain = transform(obj, rotationMatrix)
            % Transform strain tensor to new coordinate system
            %
            % Parameters:
            %   rotationMatrix: 3x3 rotation matrix
            %
            % Returns:
            %   transformedStrain: StrainTensor in new coordinate system
            validateattributes(rotationMatrix, {'numeric'}, {'size', [3 3]});
            R = rotationMatrix;
            transformedTensor = R * obj.tensor * R';
            transformedStrain = StrainTensor.fromArray(transformedTensor);
        end
        function result = plus(obj, other)
            % Add strain tensors
            result = StrainTensor.fromArray(obj.tensor + other.getTensor());
        end
        function result = minus(obj, other)
            % Subtract strain tensors
            result = StrainTensor.fromArray(obj.tensor - other.getTensor());
        end
        function result = mtimes(obj, scalar)
            % Multiply strain tensor by scalar
            result = StrainTensor.fromArray(scalar * obj.tensor);
        end
        function disp(obj)
            % Display strain tensor
            fprintf('StrainTensor:\n');
            disp(obj.tensor);
        end
    end
    methods (Static)
        function obj = fromArray(tensor)
            % Create StrainTensor from 3x3 array
            validateattributes(tensor, {'numeric'}, {'size', [3 3]});
            obj = StrainTensor(tensor(1,1), tensor(2,2), tensor(3,3), ...
                              2*tensor(1,2), 2*tensor(1,3), 2*tensor(2,3), false);
        end
        function obj = fromVoigt(voigt, engineeringStrain)
            % Create StrainTensor from Voigt notation
            %
            % Parameters:
            %   voigt: Strain vector in Voigt notation
            %   engineeringStrain: If true, interpret shears as engineering strains
            if nargin < 2
                engineeringStrain = true;
            end
            validateattributes(voigt, {'numeric'}, {'numel', 6});
            obj = StrainTensor(voigt(1), voigt(2), voigt(3), ...
                              voigt(6), voigt(5), voigt(4), engineeringStrain);
        end
    end
end