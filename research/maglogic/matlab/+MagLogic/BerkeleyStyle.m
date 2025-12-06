classdef BerkeleyStyle < handle
    %BERKELEYSTYLE UC Berkeley themed plotting styles for MATLAB
    %   Comprehensive styling class for professional scientific visualizations
    %   following UC Berkeley's official visual identity guidelines.
    %   
    %   Author: Meshal Alawein
    %   Email: meshal@berkeley.edu
    %   Institution: University of California, Berkeley
    %   
    %   Example:
    %       style = MagLogic.BerkeleyStyle();
    %       style.setup();  % Apply Berkeley styling globally
    %       plot(x, y);     % Plot with Berkeley styling
    
    properties (Constant)
        % UC Berkeley Official Colors
        BERKELEY_BLUE = [0, 0.196, 0.384];      % #003262
        CALIFORNIA_GOLD = [0.992, 0.710, 0.082]; % #FDB515
        
        % Secondary Colors
        BLUE_DARK = [0.004, 0.075, 0.200];      % #010133
        GOLD_DARK = [0.988, 0.576, 0.075];      % #FC9313
        GREEN_DARK = [0, 0.333, 0.227];         % #00553A
        ROSE_DARK = [0.467, 0.027, 0.278];      % #770747
        PURPLE_DARK = [0.263, 0.067, 0.439];    % #431170
        RED_DARK = [0.549, 0.082, 0.082];       % #8C1515
        ORANGE_DARK = [0.824, 0.412, 0.118];    % #D2691E
        TEAL_DARK = [0, 0.298, 0.353];          % #004C5A
        
        % Neutral Colors
        GREY_LIGHT = [0.851, 0.851, 0.851];     % #D9D9D9
        GREY_MEDIUM = [0.600, 0.600, 0.600];    % #999999
        GREY_DARK = [0.400, 0.400, 0.400];      % #666666
        BLACK = [0, 0, 0];                       % #000000
        WHITE = [1, 1, 1];                       % #FFFFFF
    end
    
    properties (Access = private)
        original_defaults;
        style_applied = false;
    end
    
    methods
        function obj = BerkeleyStyle()
            %BERKELEYSTYLE Constructor
            %   Initialize Berkeley style manager
            
            % Store original MATLAB defaults
            obj.original_defaults = obj.getCurrentDefaults();
        end
        
        function setup(obj)
            %SETUP Apply Berkeley styling to MATLAB globally
            %   Sets default colors, fonts, and layout parameters
            
            if obj.style_applied
                return;
            end
            
            % Set default color order
            color_cycle = obj.getColorCycle();
            set(groot, 'defaultAxesColorOrder', color_cycle);
            
            % Figure defaults
            set(groot, 'defaultFigureColor', obj.WHITE);
            set(groot, 'defaultFigurePosition', [100, 100, 800, 600]);
            
            % Axes defaults
            set(groot, 'defaultAxesFontSize', 11);
            set(groot, 'defaultAxesFontName', 'Arial');
            set(groot, 'defaultAxesFontWeight', 'normal');
            set(groot, 'defaultAxesLineWidth', 1.5);
            set(groot, 'defaultAxesXColor', obj.BLACK);
            set(groot, 'defaultAxesYColor', obj.BLACK);
            set(groot, 'defaultAxesZColor', obj.BLACK);
            set(groot, 'defaultAxesGridColor', obj.GREY_LIGHT);
            set(groot, 'defaultAxesGridAlpha', 0.7);
            set(groot, 'defaultAxesMinorGridColor', obj.GREY_MEDIUM);
            set(groot, 'defaultAxesMinorGridAlpha', 0.5);
            
            % Title defaults
            set(groot, 'defaultAxesTitleFontSize', 16);
            set(groot, 'defaultAxesTitleFontWeight', 'bold');
            set(groot, 'defaultAxesTitleColor', obj.BERKELEY_BLUE);
            
            % Label defaults
            set(groot, 'defaultAxesLabelFontSize', 12);
            set(groot, 'defaultAxesLabelFontWeight', 'bold');
            
            % Line defaults
            set(groot, 'defaultLineLineWidth', 2.5);
            set(groot, 'defaultLineMarkerSize', 8);
            
            % Legend defaults
            set(groot, 'defaultLegendFontSize', 10);
            set(groot, 'defaultLegendLocation', 'best');
            set(groot, 'defaultLegendBox', 'on');
            set(groot, 'defaultLegendColor', obj.WHITE);
            set(groot, 'defaultLegendEdgeColor', obj.GREY_DARK);
            
            % Text defaults
            set(groot, 'defaultTextFontSize', 12);
            set(groot, 'defaultTextFontName', 'Arial');
            
            obj.style_applied = true;
            fprintf('Berkeley styling applied to MATLAB\n');
        end
        
        function reset(obj)
            %RESET Restore original MATLAB defaults
            %   Removes Berkeley styling and restores original settings
            
            if ~obj.style_applied
                return;
            end
            
            % Restore original defaults
            fields = fieldnames(obj.original_defaults);
            for i = 1:length(fields)
                try
                    set(groot, fields{i}, obj.original_defaults.(fields{i}));
                catch
                    % Skip if property cannot be set
                end
            end
            
            obj.style_applied = false;
            fprintf('MATLAB defaults restored\n');
        end
        
        function colors = getColorCycle(obj)
            %GETCOLORCYCLE Get Berkeley color cycle for plots
            %   Returns matrix of RGB colors in preferred order
            
            colors = [
                obj.BERKELEY_BLUE;
                obj.CALIFORNIA_GOLD;
                obj.GREEN_DARK;
                obj.RED_DARK;
                obj.PURPLE_DARK;
                obj.ORANGE_DARK;
                obj.TEAL_DARK;
                obj.ROSE_DARK
            ];
        end
        
        function cmap = getColormap(obj, name, varargin)
            %GETCOLORMAP Generate Berkeley-themed colormaps
            %   cmap = getColormap('berkeley')
            %   cmap = getColormap('magnetization', 'levels', 256)
            
            p = inputParser;
            addRequired(p, 'name', @ischar);
            addParameter(p, 'levels', 256, @isnumeric);
            parse(p, name, varargin{:});
            
            levels = p.Results.levels;
            
            switch lower(name)
                case 'berkeley'
                    colors = [obj.BERKELEY_BLUE; obj.WHITE; obj.CALIFORNIA_GOLD];
                    
                case 'berkeley_blue'
                    colors = [obj.WHITE; obj.BERKELEY_BLUE];
                    
                case 'berkeley_gold'
                    colors = [obj.WHITE; obj.CALIFORNIA_GOLD];
                    
                case 'magnetization'
                    colors = [obj.RED_DARK; obj.WHITE; obj.BERKELEY_BLUE];
                    
                case 'energy'
                    colors = [obj.BERKELEY_BLUE; obj.WHITE; obj.CALIFORNIA_GOLD];
                    
                case 'phase'
                    colors = [obj.RED_DARK; obj.ORANGE_DARK; obj.CALIFORNIA_GOLD; ...
                             obj.GREEN_DARK; obj.BERKELEY_BLUE; obj.PURPLE_DARK];
                    
                otherwise
                    error('BerkeleyStyle:UnknownColormap', ...
                          'Unknown colormap: %s', name);
            end
            
            % Interpolate to requested number of levels
            cmap = obj.interpolateColors(colors, levels);
        end
        
        function fig = createFigure(obj, varargin)
            %CREATEFIGURE Create figure with Berkeley styling
            %   fig = createFigure('Title', 'My Plot', 'Size', [800, 600])
            
            p = inputParser;
            addParameter(p, 'Title', '', @ischar);
            addParameter(p, 'Size', [800, 600], @isnumeric);
            addParameter(p, 'Position', [100, 100], @isnumeric);
            parse(p, varargin{:});
            
            title_str = p.Results.Title;
            fig_size = p.Results.Size;
            fig_pos = p.Results.Position;
            
            % Create figure
            fig = figure('Color', obj.WHITE, ...
                        'Position', [fig_pos, fig_size], ...
                        'Units', 'pixels');
            
            % Apply title if provided
            if ~isempty(title_str)
                sgtitle(title_str, 'FontSize', 18, 'FontWeight', 'bold', ...
                       'Color', obj.BERKELEY_BLUE);
            end
            
            % Ensure Berkeley styling is applied to this specific figure
            obj.applyToFigure(fig);
        end
        
        function applyToFigure(obj, fig)
            %APPLYTOFIGURE Apply Berkeley styling to specific figure
            %   applyToFigure(fig)
            
            % Set figure properties
            set(fig, 'Color', obj.WHITE);
            
            % Find all axes in figure
            ax_handles = findall(fig, 'Type', 'axes');
            
            for i = 1:length(ax_handles)
                obj.applyToAxes(ax_handles(i));
            end
        end
        
        function applyToAxes(obj, ax)
            %APPLYTOAXES Apply Berkeley styling to specific axes
            %   applyToAxes(ax)
            
            % Basic styling
            set(ax, 'FontSize', 11, 'FontName', 'Arial');
            set(ax, 'LineWidth', 1.5);
            set(ax, 'XColor', obj.BLACK, 'YColor', obj.BLACK);
            
            % Grid styling
            set(ax, 'GridColor', obj.GREY_LIGHT, 'GridAlpha', 0.7);
            set(ax, 'MinorGridColor', obj.GREY_MEDIUM, 'MinorGridAlpha', 0.5);
            
            % Title styling
            title_handle = get(ax, 'Title');
            if ~isempty(title_handle) && ~isempty(get(title_handle, 'String'))
                set(title_handle, 'FontSize', 16, 'FontWeight', 'bold', ...
                    'Color', obj.BERKELEY_BLUE);
            end
            
            % Label styling
            xlabel_handle = get(ax, 'XLabel');
            ylabel_handle = get(ax, 'YLabel');
            set(xlabel_handle, 'FontSize', 12, 'FontWeight', 'bold');
            set(ylabel_handle, 'FontSize', 12, 'FontWeight', 'bold');
        end
        
        function saveFigure(obj, fig, filename, varargin)
            %SAVEFIGURE Save figure with Berkeley styling and high quality
            %   saveFigure(fig, 'plot.png', 'dpi', 300, 'format', 'png')
            
            p = inputParser;
            addRequired(p, 'fig');
            addRequired(p, 'filename', @ischar);
            addParameter(p, 'dpi', 300, @isnumeric);
            addParameter(p, 'format', 'png', @ischar);
            addParameter(p, 'quality', 95, @isnumeric);  % For JPEG
            parse(p, fig, filename, varargin{:});
            
            dpi_val = p.Results.dpi;
            format_str = p.Results.format;
            quality_val = p.Results.quality;
            
            % Ensure proper file extension
            [~, ~, ext] = fileparts(filename);
            if isempty(ext)
                filename = [filename, '.', format_str];
            end
            
            % Save with high quality
            switch lower(format_str)
                case 'png'
                    print(fig, filename, '-dpng', sprintf('-r%d', dpi_val));
                    
                case 'jpg'
                    print(fig, filename, '-djpeg', sprintf('-r%d', dpi_val), ...
                          sprintf('-q%d', quality_val));
                    
                case 'pdf'
                    print(fig, filename, '-dpdf', '-painters');
                    
                case 'eps'
                    print(fig, filename, '-depsc', '-painters');
                    
                case 'svg'
                    print(fig, filename, '-dsvg', '-painters');
                    
                otherwise
                    error('BerkeleyStyle:UnsupportedFormat', ...
                          'Unsupported format: %s', format_str);
            end
            
            fprintf('Figure saved: %s\n', filename);
        end
        
        function color = getColor(obj, name)
            %GETCOLOR Get specific Berkeley color by name
            %   color = getColor('berkeley_blue')
            
            switch lower(name)
                case {'berkeley_blue', 'blue'}
                    color = obj.BERKELEY_BLUE;
                case {'california_gold', 'gold'}
                    color = obj.CALIFORNIA_GOLD;
                case 'green_dark'
                    color = obj.GREEN_DARK;
                case 'red_dark'
                    color = obj.RED_DARK;
                case 'purple_dark'
                    color = obj.PURPLE_DARK;
                case 'orange_dark'
                    color = obj.ORANGE_DARK;
                case 'teal_dark'
                    color = obj.TEAL_DARK;
                case 'rose_dark'
                    color = obj.ROSE_DARK;
                case 'grey_light'
                    color = obj.GREY_LIGHT;
                case 'grey_medium'
                    color = obj.GREY_MEDIUM;
                case 'grey_dark'
                    color = obj.GREY_DARK;
                case 'black'
                    color = obj.BLACK;
                case 'white'
                    color = obj.WHITE;
                otherwise
                    error('BerkeleyStyle:UnknownColor', 'Unknown color: %s', name);
            end
        end
        
        function printColors(obj)
            %PRINTCOLORS Display all available colors
            %   Prints color names and RGB values
            
            fprintf('\nUC Berkeley Color Palette\n');
            fprintf('========================\n\n');
            
            fprintf('PRIMARY COLORS:\n');
            fprintf('  berkeley_blue:   [%.3f, %.3f, %.3f]\n', obj.BERKELEY_BLUE);
            fprintf('  california_gold: [%.3f, %.3f, %.3f]\n', obj.CALIFORNIA_GOLD);
            
            fprintf('\nSECONDARY COLORS:\n');
            fprintf('  green_dark:      [%.3f, %.3f, %.3f]\n', obj.GREEN_DARK);
            fprintf('  red_dark:        [%.3f, %.3f, %.3f]\n', obj.RED_DARK);
            fprintf('  purple_dark:     [%.3f, %.3f, %.3f]\n', obj.PURPLE_DARK);
            fprintf('  orange_dark:     [%.3f, %.3f, %.3f]\n', obj.ORANGE_DARK);
            fprintf('  teal_dark:       [%.3f, %.3f, %.3f]\n', obj.TEAL_DARK);
            fprintf('  rose_dark:       [%.3f, %.3f, %.3f]\n', obj.ROSE_DARK);
            
            fprintf('\nNEUTRAL COLORS:\n');
            fprintf('  grey_light:      [%.3f, %.3f, %.3f]\n', obj.GREY_LIGHT);
            fprintf('  grey_medium:     [%.3f, %.3f, %.3f]\n', obj.GREY_MEDIUM);
            fprintf('  grey_dark:       [%.3f, %.3f, %.3f]\n', obj.GREY_DARK);
            fprintf('  black:           [%.3f, %.3f, %.3f]\n', obj.BLACK);
            fprintf('  white:           [%.3f, %.3f, %.3f]\n', obj.WHITE);
        end
    end
    
    methods (Access = private)
        function defaults = getCurrentDefaults(obj)
            %Get current MATLAB default settings
            defaults = struct();
            
            % Get current default properties
            prop_names = {
                'defaultAxesColorOrder',
                'defaultFigureColor',
                'defaultAxesFontSize',
                'defaultAxesFontName',
                'defaultAxesLineWidth',
                'defaultLineLineWidth',
                'defaultAxesTitleFontSize'
            };
            
            for i = 1:length(prop_names)
                try
                    defaults.(prop_names{i}) = get(groot, prop_names{i});
                catch
                    % Property may not exist in older MATLAB versions
                end
            end
        end
        
        function cmap = interpolateColors(obj, colors, levels)
            %Interpolate between colors to create colormap
            n_colors = size(colors, 1);
            
            if n_colors == 1
                cmap = repmat(colors, levels, 1);
                return;
            end
            
            % Create interpolation points
            x_orig = linspace(0, 1, n_colors);
            x_new = linspace(0, 1, levels);
            
            % Interpolate each color channel
            cmap = zeros(levels, 3);
            for channel = 1:3
                cmap(:, channel) = interp1(x_orig, colors(:, channel), x_new, 'linear');
            end
            
            % Ensure values are in valid range
            cmap = max(0, min(1, cmap));
        end
    end
    
    methods (Static)
        function demo()
            %DEMO Demonstrate Berkeley styling capabilities
            %   BerkeleyStyle.demo()
            
            % Create Berkeley style instance
            style = MagLogic.BerkeleyStyle();
            style.setup();
            
            % Create demo plots
            fig = style.createFigure('Title', 'Berkeley Style Demo', 'Size', [1200, 800]);
            
            % Subplot 1: Line plot with color cycle
            subplot(2, 3, 1);
            x = linspace(0, 4*pi, 100);
            for i = 1:5
                plot(x, sin(x + i*pi/4), 'DisplayName', sprintf('Series %d', i));
                hold on;
            end
            title('Color Cycle Demo');
            xlabel('x');
            ylabel('sin(x + \phi)');
            legend('show');
            grid on;
            
            % Subplot 2: Magnetization colormap
            subplot(2, 3, 2);
            [X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
            Z = sin(X).*cos(Y);
            imagesc(Z);
            colormap(gca, style.getColormap('magnetization'));
            colorbar;
            title('Magnetization Colormap');
            
            % Subplot 3: Energy colormap
            subplot(2, 3, 3);
            Z2 = exp(-(X.^2 + Y.^2));
            imagesc(Z2);
            colormap(gca, style.getColormap('energy'));
            colorbar;
            title('Energy Colormap');
            
            % Subplot 4: Bar plot
            subplot(2, 3, 4);
            data = [3.2, 2.8, 4.1, 3.9, 2.3];
            colors = style.getColorCycle();
            bar(1:5, data, 'FaceColor', 'flat', 'CData', colors(1:5, :));
            title('Bar Plot with Berkeley Colors');
            xlabel('Category');
            ylabel('Value');
            
            % Subplot 5: Scatter plot
            subplot(2, 3, 5);
            n_points = 100;
            scatter_x = randn(n_points, 1);
            scatter_y = randn(n_points, 1);
            scatter(scatter_x, scatter_y, 50, style.getColor('berkeley_blue'), 'filled');
            title('Scatter Plot');
            xlabel('X Data');
            ylabel('Y Data');
            grid on;
            
            % Subplot 6: Phase colormap
            subplot(2, 3, 6);
            theta = atan2(Y, X);
            imagesc(theta);
            colormap(gca, style.getColormap('phase'));
            colorbar;
            title('Phase Colormap');
            
            % Apply styling to entire figure
            style.applyToFigure(fig);
            
            fprintf('Berkeley Style Demo completed.\n');
            fprintf('Try: style.printColors() to see available colors.\n');
        end
    end
end