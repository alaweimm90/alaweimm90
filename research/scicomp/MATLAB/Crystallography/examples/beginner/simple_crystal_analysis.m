function simple_crystal_analysis()
    % Simple Crystal Structure Analysis (MATLAB)
    %
    % This example demonstrates basic crystallographic calculations using simple
    % crystal structures like sodium chloride (NaCl) and diamond.
    %
    % Learning Objectives:
    % - Understand crystal lattice parameters
    % - Calculate unit cell volume and density
    % - Compute interatomic distances
    % - Determine d-spacings and Bragg angles
    fprintf('Simple Crystal Structure Analysis - MATLAB\n');
    fprintf('==========================================\n');
    % Add path to core Crystallography classes
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'core'));
    % Set Berkeley visual defaults
    setBerkeleyDefaults();
    % Create and analyze NaCl structure
    nacl_crystal = create_nacl_structure();
    % Analyze interatomic distances
    analyze_interatomic_distances(nacl_crystal);
    % Calculate diffraction data
    nacl_diffraction = calculate_diffraction_data(nacl_crystal, 'NaCl');
    % Coordination analysis
    coordination_analysis(nacl_crystal, 'NaCl');
    % Create and analyze diamond structure
    diamond_crystal = create_diamond_structure();
    diamond_diffraction = calculate_diffraction_data(diamond_crystal, 'Diamond');
    coordination_analysis(diamond_crystal, 'Diamond');
    % Compare structures
    compare_structures(nacl_crystal, diamond_crystal);
    % Plot comparison
    plot_comparison(nacl_diffraction, diamond_diffraction);
    % Demonstrate supercell creation
    supercell_example(nacl_crystal);
end
function crystal = create_nacl_structure()
    % Create sodium chloride (NaCl) crystal structure
    fprintf('\nCreating NaCl (Sodium Chloride) Structure\n');
    fprintf('-----------------------------------------\n');
    % NaCl has a face-centered cubic structure with a = 5.64 Å
    lattice = struct(...
        'a', 5.64, 'b', 5.64, 'c', 5.64, ...
        'alpha', 90, 'beta', 90, 'gamma', 90 ...
    );
    % Atomic positions in the unit cell
    atoms = struct('element', {}, 'x', {}, 'y', {}, 'z', {});
    % Sodium atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    atoms(1) = struct('element', 'Na', 'x', 0.0, 'y', 0.0, 'z', 0.0);
    atoms(2) = struct('element', 'Na', 'x', 0.5, 'y', 0.5, 'z', 0.0);
    atoms(3) = struct('element', 'Na', 'x', 0.5, 'y', 0.0, 'z', 0.5);
    atoms(4) = struct('element', 'Na', 'x', 0.0, 'y', 0.5, 'z', 0.5);
    % Chlorine atoms at (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0.5)
    atoms(5) = struct('element', 'Cl', 'x', 0.5, 'y', 0.0, 'z', 0.0);
    atoms(6) = struct('element', 'Cl', 'x', 0.0, 'y', 0.5, 'z', 0.0);
    atoms(7) = struct('element', 'Cl', 'x', 0.0, 'y', 0.0, 'z', 0.5);
    atoms(8) = struct('element', 'Cl', 'x', 0.5, 'y', 0.5, 'z', 0.5);
    crystal = CrystalStructure(lattice, atoms);
    fprintf('Lattice parameters: a = b = c = %.2f Å\n', lattice.a);
    fprintf('Number of atoms in unit cell: %d\n', length(atoms));
    fprintf('Unit cell volume: %.2f Å³\n', crystal.unitCellVolume());
    % Calculate density
    % NaCl molecular weight = 22.99 (Na) + 35.45 (Cl) = 58.44 g/mol
    % 4 formula units per unit cell (FCC structure)
    molecular_weight = 58.44;  % g/mol
    z = 4;  % formula units per unit cell
    density = crystal.density(molecular_weight, z);
    fprintf('Calculated density: %.2f g/cm³\n', density);
    fprintf('Experimental density: ~2.16 g/cm³\n');
end
function analyze_interatomic_distances(crystal)
    % Analyze interatomic distances in the crystal
    fprintf('\nInteratomic Distance Analysis\n');
    fprintf('-----------------------------\n');
    % Calculate Na-Cl distances (nearest neighbors)
    atoms = crystal.atoms;
    na_indices = [];
    cl_indices = [];
    for i = 1:length(atoms)
        if strcmp(atoms(i).element, 'Na')
            na_indices = [na_indices, i];
        elseif strcmp(atoms(i).element, 'Cl')
            cl_indices = [cl_indices, i];
        end
    end
    na_cl_distances = [];
    for i = 1:length(na_indices)
        for j = 1:length(cl_indices)
            distance = crystal.interatomicDistance(na_indices(i), cl_indices(j), true);
            na_cl_distances = [na_cl_distances, distance];
        end
    end
    % Find unique distances (within tolerance)
    unique_distances = [];
    tolerance = 0.01;
    for distance = na_cl_distances
        is_unique = true;
        for unique_dist = unique_distances
            if abs(distance - unique_dist) < tolerance
                is_unique = false;
                break;
            end
        end
        if is_unique
            unique_distances = [unique_distances, distance];
        end
    end
    unique_distances = sort(unique_distances);
    fprintf('Na-Cl distances:\n');
    for i = 1:min(3, length(unique_distances))
        fprintf('  Distance %d: %.3f Å\n', i, unique_distances(i));
    end
    % Nearest neighbor distance should be a/2 = 2.82 Å
    theoretical_nn = crystal.lattice.a / 2;
    fprintf('Theoretical nearest neighbor distance: %.3f Å\n', theoretical_nn);
end
function diffraction_data = calculate_diffraction_data(crystal, name)
    % Calculate basic diffraction information
    fprintf('\nX-ray Diffraction Analysis for %s\n', name);
    fprintf('----------------------------------\n');
    % Common X-ray wavelength (Cu Kα)
    wavelength = 1.54056;  % Å
    % Calculate d-spacings and Bragg angles for low-index reflections
    reflections = [
        1, 0, 0; 1, 1, 0; 1, 1, 1;
        2, 0, 0; 2, 1, 0; 2, 1, 1;
        2, 2, 0; 3, 1, 0; 2, 2, 2
    ];
    fprintf('Miller Indices | d-spacing (Å) | 2θ (degrees)\n');
    fprintf('---------------------------------------------\n');
    diffraction_data = struct('hkl', {}, 'd_spacing', {}, 'two_theta', {});
    count = 0;
    for i = 1:size(reflections, 1)
        h = reflections(i, 1);
        k = reflections(i, 2);
        l = reflections(i, 3);
        try
            d = crystal.dSpacing(h, k, l);
            theta = crystal.braggAngle(h, k, l, wavelength);
            two_theta = 2 * theta;
            fprintf('    (%d,%d,%d)     |    %.3f     |    %.2f\n', h, k, l, d, two_theta);
            count = count + 1;
            diffraction_data(count).hkl = [h, k, l];
            diffraction_data(count).d_spacing = d;
            diffraction_data(count).two_theta = two_theta;
        catch
            fprintf('    (%d,%d,%d)     |   forbidden   |     N/A\n', h, k, l);
        end
    end
end
function crystal = create_diamond_structure()
    % Create diamond crystal structure for comparison
    fprintf('\nCreating Diamond Structure\n');
    fprintf('--------------------------\n');
    % Diamond cubic structure with a = 3.567 Å
    lattice = struct(...
        'a', 3.567, 'b', 3.567, 'c', 3.567, ...
        'alpha', 90, 'beta', 90, 'gamma', 90 ...
    );
    % Carbon atoms in diamond structure
    atoms = struct('element', {}, 'x', {}, 'y', {}, 'z', {});
    atoms(1) = struct('element', 'C', 'x', 0.0, 'y', 0.0, 'z', 0.0);
    atoms(2) = struct('element', 'C', 'x', 0.25, 'y', 0.25, 'z', 0.25);
    atoms(3) = struct('element', 'C', 'x', 0.5, 'y', 0.5, 'z', 0.0);
    atoms(4) = struct('element', 'C', 'x', 0.75, 'y', 0.75, 'z', 0.25);
    atoms(5) = struct('element', 'C', 'x', 0.5, 'y', 0.0, 'z', 0.5);
    atoms(6) = struct('element', 'C', 'x', 0.75, 'y', 0.25, 'z', 0.75);
    atoms(7) = struct('element', 'C', 'x', 0.0, 'y', 0.5, 'z', 0.5);
    atoms(8) = struct('element', 'C', 'x', 0.25, 'y', 0.75, 'z', 0.75);
    crystal = CrystalStructure(lattice, atoms);
    fprintf('Lattice parameter: a = %.3f Å\n', lattice.a);
    fprintf('Unit cell volume: %.2f Å³\n', crystal.unitCellVolume());
    % Calculate density
    % Carbon atomic weight = 12.01 g/mol
    % 8 atoms per unit cell
    molecular_weight = 12.01;
    z = 8;
    density = crystal.density(molecular_weight, z);
    fprintf('Calculated density: %.2f g/cm³\n', density);
    fprintf('Experimental density: ~3.52 g/cm³\n');
end
function coordination_analysis(crystal, structure_name)
    % Analyze coordination numbers
    fprintf('\nCoordination Analysis for %s\n', structure_name);
    fprintf('--------------------------------\n');
    % Calculate coordination numbers for first few atoms
    cutoff_radius = 3.5;  % Å
    atoms = crystal.atoms;
    for i = 1:min(4, length(atoms))
        coord_num = crystal.coordinationNumber(i, cutoff_radius);
        atom = atoms(i);
        fprintf('Atom %d (%s): coordination number = %d\n', i, atom.element, coord_num);
        fprintf('  Position: (%.3f, %.3f, %.3f)\n', atom.x, atom.y, atom.z);
    end
end
function compare_structures(nacl_crystal, diamond_crystal)
    % Compare structures
    fprintf('\nStructure Comparison\n');
    fprintf('====================\n');
    fprintf('NaCl unit cell volume: %.2f Å³\n', nacl_crystal.unitCellVolume());
    fprintf('Diamond unit cell volume: %.2f Å³\n', diamond_crystal.unitCellVolume());
    fprintf('\nNaCl coordination: 6 (octahedral)\n');
    fprintf('Diamond coordination: 4 (tetrahedral)\n');
end
function plot_comparison(nacl_data, diamond_data)
    % Plot comparison of diffraction patterns
    figure('Position', [100, 100, 1200, 500]);
    % NaCl diffraction pattern
    subplot(1, 2, 1);
    nacl_angles = [nacl_data.two_theta];
    nacl_intensities = 100 ./ (1:length(nacl_data));  % Mock intensities
    stem(nacl_angles, nacl_intensities, 'Color', [0, 0.2, 0.38], 'LineWidth', 2, 'MarkerFaceColor', [0, 0.2, 0.38]);
    xlabel('2θ (degrees)');
    ylabel('Relative Intensity');
    title('NaCl Diffraction Pattern');
    grid on;
    xlim([0, 80]);
    % Add Miller indices labels
    for i = 1:min(5, length(nacl_angles))
        hkl = nacl_data(i).hkl;
        text(nacl_angles(i), nacl_intensities(i) + 5, ...
             sprintf('%d%d%d', hkl(1), hkl(2), hkl(3)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    % Diamond diffraction pattern
    subplot(1, 2, 2);
    diamond_angles = [diamond_data.two_theta];
    diamond_intensities = 100 ./ (1:length(diamond_data));
    stem(diamond_angles, diamond_intensities, 'Color', [0.99, 0.71, 0.08], 'LineWidth', 2, 'MarkerFaceColor', [0.99, 0.71, 0.08]);
    xlabel('2θ (degrees)');
    ylabel('Relative Intensity');
    title('Diamond Diffraction Pattern');
    grid on;
    xlim([0, 80]);
    % Add Miller indices labels
    for i = 1:min(5, length(diamond_angles))
        hkl = diamond_data(i).hkl;
        text(diamond_angles(i), diamond_intensities(i) + 5, ...
             sprintf('%d%d%d', hkl(1), hkl(2), hkl(3)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    sgtitle('Crystal Structure Diffraction Comparison', 'FontSize', 16, 'FontWeight', 'bold');
end
function supercell_example(crystal)
    % Demonstrate supercell creation
    fprintf('\nSupercell Example\n');
    fprintf('-----------------\n');
    % Create 2x2x2 supercell of NaCl
    supercell = crystal.supercell(2, 2, 2);
    original_atoms = length(crystal.atoms);
    supercell_atoms = length(supercell.atoms);
    fprintf('Original unit cell: %d atoms\n', original_atoms);
    fprintf('2x2x2 supercell: %d atoms\n', supercell_atoms);
    fprintf('Supercell volume: %.2f Å³\n', supercell.unitCellVolume());
    fprintf('Volume ratio: %.0f\n', supercell.unitCellVolume() / crystal.unitCellVolume());
end