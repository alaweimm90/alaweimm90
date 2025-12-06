function berkeley()
% BERKELEY - Set Berkeley-themed plotting style for scientific figures
%
% This function configures MATLAB's default plotting properties to use
% the UC Berkeley color scheme and professional formatting suitable for
% scientific publications.
%
% Color Scheme:
%   - Berkeley Blue: [0, 50, 98] (Primary)
%   - California Gold: [253, 181, 21] (Primary)
%   - Secondary colors: Blue Dark, Gold Dark, Green Dark, Rose Dark, Purple Dark
%   - Neutral grays for backgrounds and text
%
% Usage:
%   berkeley()  % Apply Berkeley style to all future plots
%   berkeley('reset')  % Reset to MATLAB defaults
%
% Example:
%   berkeley();
%   figure;
%   plot(1:10, rand(10,1), 'LineWidth', 2);
%   xlabel('Time (ns)');
%   ylabel('Magnetization');
%   title('LLG Dynamics');
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    if nargin > 0 && strcmp(varargin{1}, 'reset')
        % Reset to MATLAB defaults
        set(groot, 'factory');
        return;
    end
    
    % Get Berkeley color palette
    colors = getBerkeleyColors();
    
    % Set default color order
    set(groot, 'defaultAxesColorOrder', colors);
    
    % Line and marker properties
    set(groot, 'defaultLineLineWidth', 2);
    set(groot, 'defaultLineMarkerSize', 8);
    
    % Font properties
    set(groot, 'defaultAxesFontSize', 12);
    set(groot, 'defaultAxesTitleFontSize', 16);
    set(groot, 'defaultTextFontSize', 11);
    set(groot, 'defaultLegendFontSize', 10);
    set(groot, 'defaultAxesFontName', 'Arial');
    
    % Axes properties
    set(groot, 'defaultAxesLineWidth', 1.5);
    set(groot, 'defaultAxesBox', 'on');
    set(groot, 'defaultAxesTickDir', 'in');
    set(groot, 'defaultAxesTickLength', [0.01 0.025]);
    set(groot, 'defaultAxesGridAlpha', 0.3);
    set(groot, 'defaultAxesMinorGridAlpha', 0.1);
    
    % Figure properties
    set(groot, 'defaultFigurePosition', [100, 100, 800, 600]);
    set(groot, 'defaultFigureColor', 'white');
    set(groot, 'defaultFigurePaperPositionMode', 'auto');
    
    % Publication quality settings
    set(groot, 'defaultFigureRenderer', 'painters');
    set(groot, 'defaultFigurePaperType', 'usletter');
    
end

function colors = getBerkeleyColors()
% GETBERKELEYCOLORS - Return Berkeley color palette as RGB triplets
%
% Returns a matrix where each row is an RGB color (values 0-1)
% Colors are ordered by visual hierarchy and contrast

    colors = [
        0/255,   50/255,  98/255;    % Berkeley Blue (Primary)
        253/255, 181/255, 21/255;    % California Gold (Primary)
        1/255,   1/255,   51/255;    % Blue Dark
        252/255, 147/255, 19/255;    % Gold Dark
        0/255,   85/255,  58/255;    % Green Dark
        119/255, 7/255,   71/255;    % Rose Dark
        67/255,  17/255,  112/255;   % Purple Dark
        102/255, 102/255, 102/255;   % Gray Dark
        153/255, 153/255, 153/255;   % Gray Medium
        217/255, 217/255, 217/255;   % Gray Light
    ];
end

function colors = getBerkeleyColorMap(name)
% GETBERKELEYCOLORMAP - Get specific Berkeley color maps
%
% Inputs:
%   name - Color map name ('primary', 'secondary', 'neutral', 'thermal')
%
% Outputs:
%   colors - Color map matrix (Nx3)

    switch lower(name)
        case 'primary'
            colors = [
                0/255,   50/255,  98/255;    % Berkeley Blue
                253/255, 181/255, 21/255;    % California Gold
            ];
            
        case 'secondary'
            colors = [
                1/255,   1/255,   51/255;    % Blue Dark
                252/255, 147/255, 19/255;    % Gold Dark
                0/255,   85/255,  58/255;    % Green Dark
                119/255, 7/255,   71/255;    % Rose Dark
                67/255,  17/255,  112/255;   % Purple Dark
            ];
            
        case 'neutral'
            colors = [
                0/255,   0/255,   0/255;     % Black
                102/255, 102/255, 102/255;   % Gray Dark
                153/255, 153/255, 153/255;   % Gray Medium
                217/255, 217/255, 217/255;   % Gray Light
                255/255, 255/255, 255/255;   % White
            ];
            
        case 'thermal'
            % Temperature-based colormap from blue to gold
            n_colors = 64;
            blue = [0/255, 50/255, 98/255];
            gold = [253/255, 181/255, 21/255];
            colors = zeros(n_colors, 3);
            for i = 1:n_colors
                t = (i-1)/(n_colors-1);
                colors(i,:) = (1-t)*blue + t*gold;
            end
            
        otherwise
            error('Unknown colormap name: %s', name);
    end
end

function fig = berkeleyFigure(varargin)
% BERKELEYFIGURE - Create figure with Berkeley styling
%
% Inputs:
%   varargin - Figure properties (Name-Value pairs)
%
% Outputs:
%   fig - Figure handle
%
% Examples:
%   fig = berkeleyFigure();
%   fig = berkeleyFigure('Position', [100 100 1200 900]);
%   fig = berkeleyFigure('Publication', true);

    % Parse inputs
    p = inputParser;
    addParameter(p, 'Position', [100, 100, 800, 600], @isnumeric);
    addParameter(p, 'Publication', false, @islogical);
    addParameter(p, 'Title', '', @ischar);
    parse(p, varargin{:});
    
    % Create figure
    fig = figure('Position', p.Results.Position, ...
                 'Color', 'white', ...
                 'PaperPositionMode', 'auto', ...
                 'Renderer', 'painters');
    
    % Publication settings
    if p.Results.Publication
        set(fig, 'Position', [100, 100, 1200, 900]);
        set(fig, 'PaperSize', [12, 9]);
        set(fig, 'PaperPosition', [0, 0, 12, 9]);
    end
    
    % Set title
    if ~isempty(p.Results.Title)
        sgtitle(p.Results.Title, 'FontSize', 16, 'FontWeight', 'bold');
    end
end

function saveBerkeleyFigure(fig, filename, varargin)
% SAVEBERKEYFIGURE - Save figure with Berkeley formatting
%
% Inputs:
%   fig - Figure handle
%   filename - Output filename (without extension)
%   varargin - Additional options
%
% Options:
%   'Format' - Output format ('png', 'pdf', 'eps', 'svg')
%   'DPI' - Resolution for raster formats (default: 300)
%   'Transparent' - Transparent background (default: false)

    % Parse inputs
    p = inputParser;
    addParameter(p, 'Format', 'png', @ischar);
    addParameter(p, 'DPI', 300, @isnumeric);
    addParameter(p, 'Transparent', false, @islogical);
    parse(p, varargin{:});
    
    % Prepare filename
    [filepath, name, ~] = fileparts(filename);
    if isempty(filepath)
        filepath = pwd;
    end
    
    % Set figure properties for export
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'InvertHardcopy', 'off');
    
    % Export based on format
    switch lower(p.Results.Format)
        case 'png'
            if p.Results.Transparent
                print(fig, fullfile(filepath, [name '.png']), '-dpng', ...
                      sprintf('-r%d', p.Results.DPI), '-transparent');
            else
                print(fig, fullfile(filepath, [name '.png']), '-dpng', ...
                      sprintf('-r%d', p.Results.DPI));
            end
            
        case 'pdf'
            print(fig, fullfile(filepath, [name '.pdf']), '-dpdf', ...
                  '-painters', '-fillpage');
            
        case 'eps'
            print(fig, fullfile(filepath, [name '.eps']), '-depsc', ...
                  '-painters', '-tiff');
            
        case 'svg'
            print(fig, fullfile(filepath, [name '.svg']), '-dsvg', ...
                  '-painters');
            
        otherwise
            error('Unsupported format: %s', p.Results.Format);
    end
end