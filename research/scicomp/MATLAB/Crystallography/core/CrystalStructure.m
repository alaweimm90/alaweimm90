classdef CrystalStructure < handle
    % CrystalStructure - Crystal structure representation and analysis
    %
    % Features:
    % - Lattice parameter calculations
    % - Unit cell volume and density
    % - Coordinate transformations (fractional ↔ Cartesian)
    % - Interatomic distance calculations
    % - Miller indices operations
    % - Structure factor calculations
    %
    % Example:
    %   lattice = struct('a', 5.0, 'b', 5.0, 'c', 5.0, 'alpha', 90, 'beta', 90, 'gamma', 90);
    %   atoms = struct('element', {'Si', 'Si'}, 'x', {0.0, 0.5}, 'y', {0.0, 0.5}, 'z', {0.0, 0.5});
    %   crystal = CrystalStructure(lattice, atoms);
    %   volume = crystal.unitCellVolume();
    properties (Access = private)
        lattice_params      % Lattice parameters structure
        atoms_list         % Cell array of atomic positions
        direct_matrix      % Direct lattice matrix (3x3)
        reciprocal_matrix  % Reciprocal lattice matrix (3x3)
        metric_tensor      % Metric tensor (3x3)
    end
    properties (Dependent)
        lattice            % Lattice parameters (read-only)
        atoms             % Atomic positions (read-only)
        directMatrix      % Direct lattice matrix (read-only)
        reciprocalMatrix  % Reciprocal lattice matrix (read-only)
        metricTensor      % Metric tensor (read-only)
    end
    methods
        function obj = CrystalStructure(lattice_parameters, atomic_positions)
            % Constructor
            %
            % Parameters:
            %   lattice_parameters - Structure with fields: a, b, c, alpha, beta, gamma
            %   atomic_positions - Structure array with fields: element, x, y, z, occupancy, thermal_factor
            % Validate inputs
            obj.validateLatticeParameters(lattice_parameters);
            obj.validateAtomicPositions(atomic_positions);
            obj.lattice_params = lattice_parameters;
            obj.atoms_list = atomic_positions;
            % Compute fundamental matrices
            obj.computeMatrices();
        end
        % Dependent property getters
        function lattice = get.lattice(obj)
            lattice = obj.lattice_params;
        end
        function atoms = get.atoms(obj)
            atoms = obj.atoms_list;
        end
        function matrix = get.directMatrix(obj)
            matrix = obj.direct_matrix;
        end
        function matrix = get.reciprocalMatrix(obj)
            matrix = obj.reciprocal_matrix;
        end
        function tensor = get.metricTensor(obj)
            tensor = obj.metric_tensor;
        end
        function volume = unitCellVolume(obj)
            % Calculate unit cell volume in Å³
            volume = abs(det(obj.direct_matrix));
        end
        function rho = density(obj, molecular_weight, z)
            % Calculate crystal density
            %
            % Parameters:
            %   molecular_weight - Molecular weight (g/mol)
            %   z - Number of formula units per unit cell (default: 1)
            %
            % Returns:
            %   rho - Density in g/cm³
            if nargin < 3
                z = 1;
            end
            validateattributes(molecular_weight, {'numeric'}, {'scalar', 'positive'});
            validateattributes(z, {'numeric'}, {'scalar', 'positive', 'integer'});
            avogadro = 6.02214076e23;  % mol⁻¹
            volume_cm3 = obj.unitCellVolume() * 1e-24;  % Å³ to cm³
            rho = (z * molecular_weight) / (avogadro * volume_cm3);
        end
        function cartesian_coords = fractionalToCartesian(obj, fractional_coords)
            % Convert fractional coordinates to Cartesian coordinates
            %
            % Parameters:
            %   fractional_coords - Fractional coordinates (Nx3 or 3x1)
            %
            % Returns:
            %   cartesian_coords - Cartesian coordinates in Å
            if size(fractional_coords, 2) == 3
                % Nx3 format
                cartesian_coords = (obj.direct_matrix * fractional_coords')';
            else
                % 3x1 or 3xN format
                cartesian_coords = obj.direct_matrix * fractional_coords;
            end
        end
        function fractional_coords = cartesianToFractional(obj, cartesian_coords)
            % Convert Cartesian coordinates to fractional coordinates
            %
            % Parameters:
            %   cartesian_coords - Cartesian coordinates in Å (Nx3 or 3x1)
            %
            % Returns:
            %   fractional_coords - Fractional coordinates
            inv_direct = inv(obj.direct_matrix);
            if size(cartesian_coords, 2) == 3
                % Nx3 format
                fractional_coords = (inv_direct * cartesian_coords')';
            else
                % 3x1 or 3xN format
                fractional_coords = inv_direct * cartesian_coords;
            end
        end
        function distance = interatomicDistance(obj, atom1_idx, atom2_idx, include_symmetry)
            % Calculate distance between two atoms
            %
            % Parameters:
            %   atom1_idx - Index of first atom
            %   atom2_idx - Index of second atom
            %   include_symmetry - Whether to consider periodic boundary conditions (default: false)
            %
            % Returns:
            %   distance - Distance in Å
            if nargin < 4
                include_symmetry = false;
            end
            validateattributes(atom1_idx, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(atom2_idx, {'numeric'}, {'scalar', 'integer', 'positive'});
            if atom1_idx > length(obj.atoms_list) || atom2_idx > length(obj.atoms_list)
                error('Atom index out of range');
            end
            atom1 = obj.atoms_list(atom1_idx);
            atom2 = obj.atoms_list(atom2_idx);
            % Fractional coordinate difference
            df = [atom2.x - atom1.x; atom2.y - atom1.y; atom2.z - atom1.z];
            if include_symmetry
                % Apply minimum image convention
                df = df - round(df);
            end
            % Distance using metric tensor
            distance_squared = df' * obj.metric_tensor * df;
            distance = sqrt(distance_squared);
        end
        function d = dSpacing(obj, h, k, l)
            % Calculate d-spacing for Miller indices (hkl)
            %
            % Parameters:
            %   h, k, l - Miller indices
            %
            % Returns:
            %   d - d-spacing in Å
            validateattributes(h, {'numeric'}, {'scalar', 'integer'});
            validateattributes(k, {'numeric'}, {'scalar', 'integer'});
            validateattributes(l, {'numeric'}, {'scalar', 'integer'});
            % Reciprocal lattice vector
            hkl = [h; k; l];
            % Calculate |G|² where G is reciprocal lattice vector
            reciprocal_metric = inv(obj.metric_tensor);
            g_squared = hkl' * reciprocal_metric * hkl;
            if g_squared <= 0
                error('Invalid Miller indices result in zero d-spacing');
            end
            d = 1.0 / sqrt(g_squared);
        end
        function theta = braggAngle(obj, h, k, l, wavelength)
            % Calculate Bragg angle for reflection (hkl)
            %
            % Parameters:
            %   h, k, l - Miller indices
            %   wavelength - X-ray wavelength in Å
            %
            % Returns:
            %   theta - Bragg angle in degrees
            d = obj.dSpacing(h, k, l);
            sin_theta = wavelength / (2 * d);
            if sin_theta > 1
                error('No diffraction possible for hkl=(%d,%d,%d) at λ=%g Å', h, k, l, wavelength);
            end
            theta = asind(sin_theta);
        end
        function coord_num = coordinationNumber(obj, atom_idx, cutoff_radius)
            % Calculate coordination number for an atom
            %
            % Parameters:
            %   atom_idx - Index of central atom
            %   cutoff_radius - Maximum distance for coordination (Å) (default: 3.0)
            %
            % Returns:
            %   coord_num - Coordination number
            if nargin < 3
                cutoff_radius = 3.0;
            end
            if atom_idx > length(obj.atoms_list)
                error('Atom index out of range');
            end
            coord_num = 0;
            % Check all other atoms
            for i = 1:length(obj.atoms_list)
                if i == atom_idx
                    continue;
                end
                distance = obj.interatomicDistance(atom_idx, i, true);
                if distance <= cutoff_radius
                    coord_num = coord_num + 1;
                end
            end
        end
        function supercell_crystal = supercell(obj, nx, ny, nz)
            % Create supercell structure
            %
            % Parameters:
            %   nx, ny, nz - Supercell dimensions
            %
            % Returns:
            %   supercell_crystal - New CrystalStructure representing supercell
            validateattributes(nx, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(ny, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(nz, {'numeric'}, {'scalar', 'integer', 'positive'});
            % New lattice parameters
            new_lattice = obj.lattice_params;
            new_lattice.a = new_lattice.a * nx;
            new_lattice.b = new_lattice.b * ny;
            new_lattice.c = new_lattice.c * nz;
            % Replicate atoms
            new_atoms = [];
            atom_count = 0;
            for i = 0:nx-1
                for j = 0:ny-1
                    for k = 0:nz-1
                        for atom_idx = 1:length(obj.atoms_list)
                            atom = obj.atoms_list(atom_idx);
                            atom_count = atom_count + 1;
                            new_atom.element = atom.element;
                            new_atom.x = (atom.x + i) / nx;
                            new_atom.y = (atom.y + j) / ny;
                            new_atom.z = (atom.z + k) / nz;
                            if isfield(atom, 'occupancy')
                                new_atom.occupancy = atom.occupancy;
                            else
                                new_atom.occupancy = 1.0;
                            end
                            if isfield(atom, 'thermal_factor')
                                new_atom.thermal_factor = atom.thermal_factor;
                            else
                                new_atom.thermal_factor = 0.0;
                            end
                            new_atoms = [new_atoms, new_atom];
                        end
                    end
                end
            end
            supercell_crystal = CrystalStructure(new_lattice, new_atoms);
        end
    end
    methods (Access = private)
        function validateLatticeParameters(~, lattice)
            % Validate lattice parameters structure
            required_fields = {'a', 'b', 'c', 'alpha', 'beta', 'gamma'};
            for i = 1:length(required_fields)
                if ~isfield(lattice, required_fields{i})
                    error('Lattice parameters must contain field: %s', required_fields{i});
                end
                value = lattice.(required_fields{i});
                if ~isnumeric(value) || ~isscalar(value)
                    error('Lattice parameter %s must be a numeric scalar', required_fields{i});
                end
                if strcmp(required_fields{i}(1), 'a') || strcmp(required_fields{i}(1), 'b') || strcmp(required_fields{i}(1), 'c')
                    if value <= 0
                        error('Lattice parameters a, b, c must be positive');
                    end
                else
                    if value <= 0 || value >= 180
                        error('Lattice angles must be between 0 and 180 degrees');
                    end
                end
            end
        end
        function validateAtomicPositions(~, atoms)
            % Validate atomic positions structure array
            if isempty(atoms)
                error('At least one atom must be provided');
            end
            required_fields = {'element', 'x', 'y', 'z'};
            for atom_idx = 1:length(atoms)
                atom = atoms(atom_idx);
                for i = 1:length(required_fields)
                    if ~isfield(atom, required_fields{i})
                        error('Atom %d must contain field: %s', atom_idx, required_fields{i});
                    end
                end
                if ~ischar(atom.element) && ~isstring(atom.element)
                    error('Atom %d element must be a string', atom_idx);
                end
                coords = [atom.x, atom.y, atom.z];
                if ~isnumeric(coords) || length(coords) ~= 3
                    error('Atom %d coordinates must be numeric values', atom_idx);
                end
                % Set default values for optional fields
                if ~isfield(atom, 'occupancy')
                    atoms(atom_idx).occupancy = 1.0;
                end
                if ~isfield(atom, 'thermal_factor')
                    atoms(atom_idx).thermal_factor = 0.0;
                end
            end
        end
        function computeMatrices(obj)
            % Compute fundamental crystallographic matrices
            a = obj.lattice_params.a;
            b = obj.lattice_params.b;
            c = obj.lattice_params.c;
            alpha = deg2rad(obj.lattice_params.alpha);
            beta = deg2rad(obj.lattice_params.beta);
            gamma = deg2rad(obj.lattice_params.gamma);
            % Trigonometric values
            cos_alpha = cos(alpha);
            cos_beta = cos(beta);
            cos_gamma = cos(gamma);
            sin_alpha = sin(alpha);
            sin_beta = sin(beta);
            sin_gamma = sin(gamma);
            % Volume calculation
            volume = a * b * c * sqrt(1 - cos_alpha^2 - cos_beta^2 - cos_gamma^2 + ...
                                     2*cos_alpha*cos_beta*cos_gamma);
            % Direct lattice matrix
            obj.direct_matrix = [
                a, b*cos_gamma, c*cos_beta;
                0, b*sin_gamma, c*(cos_alpha - cos_beta*cos_gamma)/sin_gamma;
                0, 0, volume/(a*b*sin_gamma)
            ];
            % Metric tensor (dot products of lattice vectors)
            obj.metric_tensor = [
                a^2, a*b*cos_gamma, a*c*cos_beta;
                a*b*cos_gamma, b^2, b*c*cos_alpha;
                a*c*cos_beta, b*c*cos_alpha, c^2
            ];
            % Reciprocal lattice matrix
            obj.reciprocal_matrix = 2*pi * inv(obj.direct_matrix)';
        end
    end
end