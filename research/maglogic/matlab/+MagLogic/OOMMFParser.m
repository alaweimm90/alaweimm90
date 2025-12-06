classdef OOMMFParser < handle
    %OOMMFPARSER Parse OOMMF simulation output files
    %   Comprehensive parser for OOMMF OVF and ODT files with Berkeley styling
    %   
    %   Author: Meshal Alawein
    %   Email: meshal@berkeley.edu
    %   Institution: University of California, Berkeley
    %   
    %   Example:
    %       parser = MagLogic.OOMMFParser();
    %       data = parser.readOVF('magnetization.ovf');
    %       parser.plotMagnetization(data);
    
    properties (Access = private)
        verbose = false;
        berkeley_colors;
    end
    
    methods
        function obj = OOMMFParser(varargin)
            %OOMMFPARSER Constructor
            %   parser = OOMMFParser('verbose', true)
            
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'verbose', false, @islogical);
            parse(p, varargin{:});
            
            obj.verbose = p.Results.verbose;
            
            % Initialize Berkeley color scheme
            obj.berkeley_colors = obj.initializeBerkeleyColors();
            
            if obj.verbose
                fprintf('MagLogic OOMMF Parser initialized\n');
            end
        end
        
        function data = readOVF(obj, filename)
            %READOVF Read OOMMF Vector Field (OVF) file
            %   data = readOVF(filename)
            %   
            %   Returns struct with fields:
            %       mx, my, mz - magnetization components
            %       x, y, z - coordinate arrays  
            %       header - file header information
            %       metadata - simulation parameters
            
            if obj.verbose
                fprintf('Reading OVF file: %s\n', filename);
            end
            
            % Check file exists
            if ~exist(filename, 'file')
                error('MagLogic:FileNotFound', 'OVF file not found: %s', filename);
            end
            
            % Open file
            fid = fopen(filename, 'r');
            if fid == -1
                error('MagLogic:FileAccess', 'Cannot open OVF file: %s', filename);
            end
            
            try
                % Read header
                header = obj.parseOVFHeader(fid);
                
                % Read data based on format
                if strcmpi(header.format, 'binary')
                    magdata = obj.readOVFBinary(fid, header);
                else
                    magdata = obj.readOVFText(fid, header);
                end
                
                % Create coordinate grids
                [x, y, z] = obj.createCoordinateGrids(header);
                
                % Package data
                data = struct();
                data.mx = reshape(magdata(:,1), header.znodes, header.ynodes, header.xnodes);
                data.my = reshape(magdata(:,2), header.znodes, header.ynodes, header.xnodes);
                data.mz = reshape(magdata(:,3), header.znodes, header.ynodes, header.xnodes);
                data.x = x;
                data.y = y;
                data.z = z;
                data.header = header;
                data.metadata = obj.extractMetadata(header);
                data.filename = filename;
                
                if obj.verbose
                    fprintf('Successfully read OVF file: %dx%dx%d grid\n', ...
                            header.xnodes, header.ynodes, header.znodes);
                end
                
            catch ME
                fclose(fid);
                rethrow(ME);
            end
            
            fclose(fid);
        end
        
        function data = readODT(obj, filename)
            %READODT Read OOMMF Data Table (ODT) file
            %   data = readODT(filename)
            %   
            %   Returns struct with time series data
            
            if obj.verbose
                fprintf('Reading ODT file: %s\n', filename);
            end
            
            % Check file exists
            if ~exist(filename, 'file')
                error('MagLogic:FileNotFound', 'ODT file not found: %s', filename);
            end
            
            % Read using MATLAB's table reader
            try
                % Read header to get column names
                header_info = obj.parseODTHeader(filename);
                
                % Read numeric data
                raw_data = readmatrix(filename, 'CommentStyle', '#');
                
                % Create structured data
                data = struct();
                for i = 1:length(header_info.columns)
                    col_name = header_info.columns{i};
                    % Clean column name for MATLAB field names
                    clean_name = regexprep(col_name, '[^a-zA-Z0-9_]', '_');
                    clean_name = regexprep(clean_name, '^_+|_+$', '');
                    if isempty(clean_name)
                        clean_name = sprintf('column_%d', i);
                    end
                    
                    if i <= size(raw_data, 2)
                        data.(clean_name) = raw_data(:, i);
                    end
                end
                
                data.filename = filename;
                data.header = header_info;
                
                if obj.verbose
                    fprintf('Successfully read ODT file: %d rows, %d columns\n', ...
                            size(raw_data, 1), size(raw_data, 2));
                end
                
            catch ME
                error('MagLogic:ODTError', 'Error reading ODT file: %s', ME.message);
            end
        end
        
        function fig = plotMagnetization(obj, data, varargin)
            %PLOTMAGNETIZATION Create Berkeley-styled magnetization plot
            %   fig = plotMagnetization(data, 'component', 'mz', 'slice', 'middle')
            
            % Parse arguments
            p = inputParser;
            addParameter(p, 'component', 'mz', @(x) ismember(x, {'mx', 'my', 'mz', 'magnitude'}));
            addParameter(p, 'slice', 'middle', @(x) ischar(x) || isnumeric(x));
            addParameter(p, 'colormap', 'magnetization', @ischar);
            addParameter(p, 'title', '', @ischar);
            parse(p, varargin{:});
            
            component = p.Results.component;
            slice_spec = p.Results.slice;
            cmap_name = p.Results.colormap;
            plot_title = p.Results.title;
            
            % Extract slice data
            if strcmp(component, 'magnitude')
                plot_data = sqrt(data.mx.^2 + data.my.^2 + data.mz.^2);
            else
                plot_data = data.(component);
            end
            
            % Handle 3D data slicing
            if ndims(plot_data) == 3
                if ischar(slice_spec) && strcmp(slice_spec, 'middle')
                    slice_idx = round(size(plot_data, 1) / 2);
                elseif isnumeric(slice_spec)
                    slice_idx = slice_spec;
                else
                    slice_idx = 1;
                end
                plot_data = squeeze(plot_data(slice_idx, :, :));
                x_coords = squeeze(data.x(slice_idx, :, :));
                y_coords = squeeze(data.y(slice_idx, :, :));
            else
                x_coords = data.x;
                y_coords = data.y;
            end
            
            % Create figure with Berkeley styling
            fig = figure('Color', 'white', 'Position', [100, 100, 800, 600]);
            
            % Create surface plot
            surf(x_coords * 1e9, y_coords * 1e9, zeros(size(plot_data)), plot_data, ...
                 'EdgeColor', 'none', 'FaceColor', 'interp');
            view(2);
            
            % Apply Berkeley colormap
            obj.applyBerkeleyColormap(cmap_name);
            
            % Styling
            xlabel('x (nm)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('y (nm)', 'FontSize', 12, 'FontWeight', 'bold');
            
            if isempty(plot_title)
                title_str = sprintf('Magnetization %s', upper(component));
            else
                title_str = plot_title;
            end
            title(title_str, 'FontSize', 16, 'FontWeight', 'bold', ...
                  'Color', obj.berkeley_colors.primary.berkeley_blue);
            
            % Colorbar
            cb = colorbar;
            if strcmp(component, 'magnitude')
                cb.Label.String = '|M| (normalized)';
            else
                cb.Label.String = sprintf('M_%s (normalized)', component(2));
            end
            cb.Label.FontSize = 12;
            cb.Label.FontWeight = 'bold';
            
            % Set color limits
            if ~strcmp(component, 'magnitude')
                clim([-1, 1]);
            end
            
            % Grid and appearance
            axis equal tight;
            box on;
            grid off;
            
            % Apply Berkeley styling
            obj.applyBerkeleyStyling(fig);
        end
        
        function fig = plotTimeSeries(obj, data, varargin)
            %PLOTTIMESERIES Plot time series data with Berkeley styling
            %   fig = plotTimeSeries(data, 'columns', {'mx', 'my', 'mz'})
            
            % Parse arguments
            p = inputParser;
            addParameter(p, 'columns', {'mx', 'my', 'mz'}, @iscell);
            addParameter(p, 'time_column', 'Time', @ischar);
            addParameter(p, 'title', 'Magnetization Dynamics', @ischar);
            parse(p, varargin{:});
            
            columns = p.Results.columns;
            time_col = p.Results.time_column;
            plot_title = p.Results.title;
            
            % Get time data
            time_field = obj.findField(data, time_col);
            if isempty(time_field)
                error('MagLogic:TimeColumnNotFound', 'Time column not found');
            end
            time_data = data.(time_field) * 1e9; % Convert to ns
            
            % Create figure
            fig = figure('Color', 'white', 'Position', [100, 100, 1000, 600]);
            hold on;
            
            % Plot each column
            colors = obj.getBerkeleyColorCycle();
            for i = 1:length(columns)
                col_field = obj.findField(data, columns{i});
                if ~isempty(col_field)
                    plot(time_data, data.(col_field), ...
                         'LineWidth', 2.5, 'Color', colors{mod(i-1, length(colors))+1}, ...
                         'DisplayName', strrep(columns{i}, '_', ' '));
                end
            end
            
            % Styling
            xlabel('Time (ns)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Magnetization (normalized)', 'FontSize', 12, 'FontWeight', 'bold');
            title(plot_title, 'FontSize', 16, 'FontWeight', 'bold', ...
                  'Color', obj.berkeley_colors.primary.berkeley_blue);
            
            legend('show', 'Location', 'best', 'FontSize', 10);
            grid on;
            grid minor;
            
            % Apply Berkeley styling
            obj.applyBerkeleyStyling(fig);
        end
        
        function exportData(obj, data, filename, varargin)
            %EXPORTDATA Export data to various formats
            %   exportData(data, 'output.mat', 'format', 'mat')
            %   exportData(data, 'output.h5', 'format', 'hdf5')
            
            p = inputParser;
            addParameter(p, 'format', 'mat', @(x) ismember(x, {'mat', 'hdf5', 'csv'}));
            parse(p, varargin{:});
            
            format = p.Results.format;
            
            switch format
                case 'mat'
                    save(filename, 'data', '-v7.3');
                    
                case 'hdf5'
                    obj.exportToHDF5(data, filename);
                    
                case 'csv'
                    obj.exportToCSV(data, filename);
            end
            
            if obj.verbose
                fprintf('Data exported to: %s (format: %s)\n', filename, format);
            end
        end
    end
    
    methods (Access = private)
        function berkeley_colors = initializeBerkeleyColors(obj)
            %Initialize Berkeley color scheme
            berkeley_colors = struct();
            berkeley_colors.primary = struct(...
                'berkeley_blue', [0, 0.196, 0.384], ...
                'california_gold', [0.992, 0.710, 0.082]);
            berkeley_colors.secondary = struct(...
                'green_dark', [0, 0.333, 0.227], ...
                'red_dark', [0.549, 0.082, 0.082], ...
                'purple_dark', [0.263, 0.067, 0.439]);
        end
        
        function header = parseOVFHeader(obj, fid)
            %Parse OVF file header
            header = struct();
            
            % Read until data section
            while ~feof(fid)
                line = fgetl(fid);
                if contains(line, '# Begin: Data')
                    break;
                end
                
                % Parse header fields
                if startsWith(line, '# xnodes:')
                    header.xnodes = str2double(extractAfter(line, ':'));
                elseif startsWith(line, '# ynodes:')
                    header.ynodes = str2double(extractAfter(line, ':'));
                elseif startsWith(line, '# znodes:')
                    header.znodes = str2double(extractAfter(line, ':'));
                elseif startsWith(line, '# xstepsize:')
                    header.xstepsize = str2double(extractAfter(line, ':'));
                elseif startsWith(line, '# ystepsize:')
                    header.ystepsize = str2double(extractAfter(line, ':'));
                elseif startsWith(line, '# zstepsize:')
                    header.zstepsize = str2double(extractAfter(line, ':'));
                elseif contains(line, 'binary')
                    header.format = 'binary';
                elseif contains(line, 'text')
                    header.format = 'text';
                end
            end
        end
        
        function magdata = readOVFBinary(obj, fid, header)
            %Read binary OVF data
            total_points = header.xnodes * header.ynodes * header.znodes;
            
            % Skip binary marker
            fread(fid, 1, 'float32');
            
            % Read magnetization data
            magdata = fread(fid, [3, total_points], 'float32')';
        end
        
        function magdata = readOVFText(obj, fid, header)
            %Read text OVF data
            total_points = header.xnodes * header.ynodes * header.znodes;
            magdata = zeros(total_points, 3);
            
            for i = 1:total_points
                line = fgetl(fid);
                values = str2num(line); %#ok<ST2NM>
                magdata(i, :) = values;
            end
        end
        
        function [x, y, z] = createCoordinateGrids(obj, header)
            %Create coordinate grids from header info
            x_vec = (0:header.xnodes-1) * header.xstepsize;
            y_vec = (0:header.ynodes-1) * header.ystepsize;
            z_vec = (0:header.znodes-1) * header.zstepsize;
            
            [x, y, z] = meshgrid(x_vec, y_vec, z_vec);
            x = permute(x, [3, 1, 2]);
            y = permute(y, [3, 1, 2]);
            z = permute(z, [3, 1, 2]);
        end
        
        function metadata = extractMetadata(obj, header)
            %Extract simulation metadata
            metadata = struct();
            metadata.grid_size = [header.xnodes, header.ynodes, header.znodes];
            metadata.cell_size = [header.xstepsize, header.ystepsize, header.zstepsize];
            metadata.total_cells = prod(metadata.grid_size);
        end
        
        function header_info = parseODTHeader(obj, filename)
            %Parse ODT file header
            fid = fopen(filename, 'r');
            header_info = struct();
            header_info.columns = {};
            
            try
                while ~feof(fid)
                    line = fgetl(fid);
                    if startsWith(line, '# Columns:')
                        % Extract column names
                        col_line = extractAfter(line, 'Columns:');
                        header_info.columns = strsplit(strtrim(col_line));
                        break;
                    end
                end
            catch
                % Fallback: generic column names
                header_info.columns = {'Time', 'Energy', 'mx', 'my', 'mz'};
            end
            
            fclose(fid);
        end
        
        function field_name = findField(obj, data, target)
            %Find field name matching target (case-insensitive)
            fields = fieldnames(data);
            field_name = '';
            
            for i = 1:length(fields)
                if strcmpi(fields{i}, target) || contains(lower(fields{i}), lower(target))
                    field_name = fields{i};
                    break;
                end
            end
        end
        
        function applyBerkeleyColormap(obj, cmap_name)
            %Apply Berkeley-themed colormap
            switch cmap_name
                case 'magnetization'
                    % Red-white-blue for magnetization
                    colors = [obj.berkeley_colors.secondary.red_dark; 1, 1, 1; obj.berkeley_colors.primary.berkeley_blue];
                case 'energy'
                    % Blue-white-gold for energy
                    colors = [obj.berkeley_colors.primary.berkeley_blue; 1, 1, 1; obj.berkeley_colors.primary.california_gold];
                otherwise
                    colors = [obj.berkeley_colors.primary.berkeley_blue; 1, 1, 1; obj.berkeley_colors.primary.california_gold];
            end
            
            colormap(colors);
        end
        
        function colors = getBerkeleyColorCycle(obj)
            %Get Berkeley color cycle for line plots
            colors = {
                obj.berkeley_colors.primary.berkeley_blue;
                obj.berkeley_colors.primary.california_gold;
                obj.berkeley_colors.secondary.green_dark;
                obj.berkeley_colors.secondary.red_dark;
                obj.berkeley_colors.secondary.purple_dark
            };
        end
        
        function applyBerkeleyStyling(obj, fig)
            %Apply Berkeley styling to figure
            set(fig, 'Color', 'white');
            
            % Style all axes
            axes_handles = findall(fig, 'Type', 'axes');
            for i = 1:length(axes_handles)
                ax = axes_handles(i);
                set(ax, 'FontSize', 11, 'FontName', 'Arial');
                set(ax, 'GridColor', [0.8, 0.8, 0.8], 'GridAlpha', 0.7);
                set(ax, 'MinorGridColor', [0.9, 0.9, 0.9], 'MinorGridAlpha', 0.5);
            end
        end
        
        function exportToHDF5(obj, data, filename)
            %Export data to HDF5 format
            if exist(filename, 'file')
                delete(filename);
            end
            
            % Save magnetization data
            h5create(filename, '/magnetization/mx', size(data.mx));
            h5write(filename, '/magnetization/mx', data.mx);
            h5create(filename, '/magnetization/my', size(data.my));
            h5write(filename, '/magnetization/my', data.my);
            h5create(filename, '/magnetization/mz', size(data.mz));
            h5write(filename, '/magnetization/mz', data.mz);
            
            % Save coordinates
            h5create(filename, '/coordinates/x', size(data.x));
            h5write(filename, '/coordinates/x', data.x);
            h5create(filename, '/coordinates/y', size(data.y));
            h5write(filename, '/coordinates/y', data.y);
            h5create(filename, '/coordinates/z', size(data.z));
            h5write(filename, '/coordinates/z', data.z);
        end
        
        function exportToCSV(obj, data, filename)
            %Export time series data to CSV
            if isstruct(data) && isfield(data, 'Time')
                % Time series data
                table_data = struct2table(data);
                writetable(table_data, filename);
            else
                warning('MagLogic:CSVExport', 'CSV export only supports time series data');
            end
        end
    end
end