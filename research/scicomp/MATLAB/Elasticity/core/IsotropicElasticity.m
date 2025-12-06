classdef IsotropicElasticity < handle
    % IsotropicElasticity - Isotropic linear elasticity relationships
    %
    % Features:
    % - Stress-strain constitutive relationships
    % - Compliance and stiffness matrices
    % - Elastic wave velocities
    % - Energy calculations
    %
    % Example:
    %   elasticity = IsotropicElasticity(200e9, 0.3, 7850);
    %   stress = elasticity.stressFromStrain(strainTensor);
    %   strain = elasticity.strainFromStress(stressTensor);
    properties (Access = private)
        youngsModulus     % Young's modulus (Pa)
        poissonsRatio     % Poisson's ratio
        density           % Material density (kg/m³)
        shearModulus      % Shear modulus (Pa)
        bulkModulus       % Bulk modulus (Pa)
        lameFirst         % First Lamé parameter (Pa)
        lameSecond        % Second Lamé parameter (Pa)
        stiffnessMatrix   % 6x6 stiffness matrix
        complianceMatrix  % 6x6 compliance matrix
    end
    methods
        function obj = IsotropicElasticity(youngsModulus, poissonsRatio, density)
            % Constructor
            %
            % Parameters:
            %   youngsModulus: Young's modulus (Pa)
            %   poissonsRatio: Poisson's ratio
            %   density: Material density (kg/m³, default: 7850)
            if nargin < 3
                density = 7850;
            end
            % Validate inputs
            validateattributes(youngsModulus, {'numeric'}, {'scalar', 'positive'});
            validateattributes(poissonsRatio, {'numeric'}, {'scalar', '>', -1, '<', 0.5});
            validateattributes(density, {'numeric'}, {'scalar', 'positive'});
            obj.youngsModulus = youngsModulus;
            obj.poissonsRatio = poissonsRatio;
            obj.density = density;
            % Calculate derived constants
            obj.calculateDerivedConstants();
            % Pre-compute matrices
            obj.computeStiffnessMatrix();
            obj.computeComplianceMatrix();
        end
        function calculateDerivedConstants(obj)
            % Calculate derived elastic constants
            E = obj.youngsModulus;
            nu = obj.poissonsRatio;
            obj.shearModulus = E / (2 * (1 + nu));
            obj.bulkModulus = E / (3 * (1 - 2 * nu));
            obj.lameFirst = (E * nu) / ((1 + nu) * (1 - 2 * nu));
            obj.lameSecond = obj.shearModulus;
        end
        function computeStiffnessMatrix(obj)
            % Compute 6x6 stiffness matrix in Voigt notation
            E = obj.youngsModulus;
            nu = obj.poissonsRatio;
            factor = E / ((1 + nu) * (1 - 2 * nu));
            C = zeros(6, 6);
            % Diagonal terms
            C(1,1) = factor * (1 - nu);
            C(2,2) = factor * (1 - nu);
            C(3,3) = factor * (1 - nu);
            C(4,4) = factor * (1 - 2 * nu) / 2;
            C(5,5) = factor * (1 - 2 * nu) / 2;
            C(6,6) = factor * (1 - 2 * nu) / 2;
            % Off-diagonal terms
            offDiag = factor * nu;
            C(1,2) = offDiag; C(1,3) = offDiag;
            C(2,1) = offDiag; C(2,3) = offDiag;
            C(3,1) = offDiag; C(3,2) = offDiag;
            obj.stiffnessMatrix = C;
        end
        function computeComplianceMatrix(obj)
            % Compute 6x6 compliance matrix in Voigt notation
            E = obj.youngsModulus;
            nu = obj.poissonsRatio;
            G = obj.shearModulus;
            S = zeros(6, 6);
            % Diagonal terms
            S(1,1) = 1 / E;
            S(2,2) = 1 / E;
            S(3,3) = 1 / E;
            S(4,4) = 1 / G;
            S(5,5) = 1 / G;
            S(6,6) = 1 / G;
            % Off-diagonal terms
            S(1,2) = -nu / E; S(1,3) = -nu / E;
            S(2,1) = -nu / E; S(2,3) = -nu / E;
            S(3,1) = -nu / E; S(3,2) = -nu / E;
            obj.complianceMatrix = S;
        end
        function C = getStiffnessMatrix(obj)
            % Get stiffness matrix
            C = obj.stiffnessMatrix;
        end
        function S = getComplianceMatrix(obj)
            % Get compliance matrix
            S = obj.complianceMatrix;
        end
        function stress = stressFromStrain(obj, strainTensor)
            % Calculate stress from strain using Hooke's law
            %
            % Parameters:
            %   strainTensor: StrainTensor object
            %
            % Returns:
            %   stress: StressTensor object
            strainVoigt = strainTensor.toVoigt(true); % Engineering strain
            stressVoigt = obj.stiffnessMatrix * strainVoigt;
            stress = StressTensor.fromVoigt(stressVoigt);
        end
        function strain = strainFromStress(obj, stressTensor)
            % Calculate strain from stress using compliance
            %
            % Parameters:
            %   stressTensor: StressTensor object
            %
            % Returns:
            %   strain: StrainTensor object
            stressVoigt = stressTensor.toVoigt();
            strainVoigt = obj.complianceMatrix * stressVoigt;
            strain = StrainTensor.fromVoigt(strainVoigt, true);
        end
        function [vP, vS] = elasticWaveVelocities(obj)
            % Calculate elastic wave velocities
            %
            % Returns:
            %   vP: Longitudinal wave velocity (m/s)
            %   vS: Transverse wave velocity (m/s)
            rho = obj.density;
            lambda = obj.lameFirst;
            mu = obj.lameSecond;
            % Longitudinal wave velocity
            vP = sqrt((lambda + 2 * mu) / rho);
            % Transverse (shear) wave velocity
            vS = sqrt(mu / rho);
        end
        function energy = elasticEnergyDensity(obj, stressTensor, strainTensor)
            % Calculate elastic energy density
            %
            % Parameters:
            %   stressTensor: StressTensor object
            %   strainTensor: StrainTensor object
            %
            % Returns:
            %   energy: Elastic energy density (J/m³)
            stressVoigt = stressTensor.toVoigt();
            strainVoigt = strainTensor.toVoigt(true);
            energy = 0.5 * dot(stressVoigt, strainVoigt);
        end
        function K = bulkModulusFromStress(obj, hydrostaticStress)
            % Calculate bulk modulus from hydrostatic stress state
            K = obj.bulkModulus;
        end
        function G = shearModulusFromStress(obj, shearStress, shearStrain)
            % Calculate shear modulus from pure shear state
            %
            % Parameters:
            %   shearStress: Applied shear stress (Pa)
            %   shearStrain: Resulting shear strain
            %
            % Returns:
            %   G: Shear modulus (Pa)
            if abs(shearStrain) < 1e-12
                warning('Shear strain is very small, calculation may be inaccurate');
                G = obj.shearModulus;
            else
                G = shearStress / shearStrain;
            end
        end
        % Getter methods for material properties
        function E = getYoungsModulus(obj)
            E = obj.youngsModulus;
        end
        function nu = getPoissonsRatio(obj)
            nu = obj.poissonsRatio;
        end
        function rho = getDensity(obj)
            rho = obj.density;
        end
        function G = getShearModulus(obj)
            G = obj.shearModulus;
        end
        function K = getBulkModulus(obj)
            K = obj.bulkModulus;
        end
        function lambda = getLameFirst(obj)
            lambda = obj.lameFirst;
        end
        function mu = getLameSecond(obj)
            mu = obj.lameSecond;
        end
        function disp(obj)
            % Display material properties
            fprintf('Isotropic Elastic Material Properties:\n');
            fprintf('  Young''s Modulus: %.2e Pa (%.1f GPa)\n', ...
                    obj.youngsModulus, obj.youngsModulus/1e9);
            fprintf('  Poisson''s Ratio: %.3f\n', obj.poissonsRatio);
            fprintf('  Shear Modulus: %.2e Pa (%.1f GPa)\n', ...
                    obj.shearModulus, obj.shearModulus/1e9);
            fprintf('  Bulk Modulus: %.2e Pa (%.1f GPa)\n', ...
                    obj.bulkModulus, obj.bulkModulus/1e9);
            fprintf('  Density: %.0f kg/m³\n', obj.density);
            [vP, vS] = obj.elasticWaveVelocities();
            fprintf('  P-wave velocity: %.0f m/s\n', vP);
            fprintf('  S-wave velocity: %.0f m/s\n', vS);
        end
    end
end