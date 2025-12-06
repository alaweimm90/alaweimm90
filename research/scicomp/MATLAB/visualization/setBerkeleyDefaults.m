function setBerkeleyDefaults()
%SETBERKELEYDEFAULTS Configure MATLAB plotting with UC Berkeley visual identity
%
% Sets default plotting parameters to match UC Berkeley's official brand
% guidelines, creating publication-quality figures with Berkeley Blue and
% California Gold color scheme.
%
% Usage:
%   setBerkeleyDefaults()
%
% Features:
%   - UC Berkeley official colors (Berkeley Blue #003262, California Gold #FDB515)
%   - Publication-quality typography and sizing
%   - Professional axis styling with inward-pointing ticks
%   - High-resolution output settings
%   - Consistent legend and colorbar styling
%
% The function modifies MATLAB's default figure properties globally,
% affecting all subsequent plots until MATLAB is restarted or defaults
% are reset.
%
% Examples:
%   setBerkeleyDefaults();
%   x = linspace(0, 2*pi, 100);
%   plot(x, sin(x));
%   title('Berkeley-Styled Plot');
%
% See also: BERKELEYPLOTSTYLE, PLOTQUANTUMSTATES
%
% Author: Meshal Alawein (meshal@berkeley.edu)
% Institution: University of California, Berkeley
% License: MIT
% Copyright © 2025 Meshal Alawein — All rights reserved.
%% UC Berkeley Official Colors
% Primary colors
berkeleyBlue = [0, 50, 98] / 255;          % #003262
californiaGold = [253, 181, 21] / 255;     % #FDB515
% Secondary colors for multi-line plots
blueDark = [1, 1, 51] / 255;               % #010133
goldDark = [252, 147, 19] / 255;           % #FC9313
greenDark = [0, 85, 58] / 255;             % #00553A
roseDark = [119, 7, 71] / 255;             % #770747
purpleDark = [67, 17, 112] / 255;          % #431170
redDark = [140, 21, 21] / 255;             % #8C1515
orangeDark = [210, 105, 30] / 255;         % #D2691E
tealDark = [0, 76, 90] / 255;              % #004C5A
% Neutral colors
greyLight = [217, 217, 217] / 255;         % #D9D9D9
greyMedium = [153, 153, 153] / 255;        % #999999
greyDark = [102, 102, 102] / 255;          % #666666
%% Color Order (for multiple line plots)
berkeleyColorOrder = [
    berkeleyBlue;
    californiaGold;
    greenDark;
    roseDark;
    purpleDark;
    redDark;
    orangeDark;
    tealDark
];
%% Figure Properties
set(groot, 'DefaultFigureColor', 'white');
set(groot, 'DefaultFigurePosition', [100, 100, 800, 500]);
set(groot, 'DefaultFigurePaperPositionMode', 'auto');
set(groot, 'DefaultFigureRenderer', 'painters');
% High-resolution settings
set(groot, 'DefaultFigurePaperUnits', 'inches');
set(groot, 'DefaultFigurePaperSize', [10, 6]);
%% Axes Properties
set(groot, 'DefaultAxesBox', 'off');
set(groot, 'DefaultAxesColor', 'white');
set(groot, 'DefaultAxesXColor', 'black');
set(groot, 'DefaultAxesYColor', 'black');
set(groot, 'DefaultAxesZColor', 'black');
set(groot, 'DefaultAxesLineWidth', 1.5);
% Grid settings (Berkeley style avoids grids unless needed)
set(groot, 'DefaultAxesXGrid', 'off');
set(groot, 'DefaultAxesYGrid', 'off');
set(groot, 'DefaultAxesZGrid', 'off');
set(groot, 'DefaultAxesGridColor', greyLight);
set(groot, 'DefaultAxesGridLineStyle', '-');
set(groot, 'DefaultAxesGridAlpha', 0.8);
% Tick properties (inward-pointing ticks)
set(groot, 'DefaultAxesTickDir', 'in');
set(groot, 'DefaultAxesTickLength', [0.01, 0.025]);
% Color order for multiple lines
set(groot, 'DefaultAxesColorOrder', berkeleyColorOrder);
%% Line Properties
set(groot, 'DefaultLineLineWidth', 2.0);
set(groot, 'DefaultLineMarkerSize', 8);
%% Text and Font Properties
% Berkeley standard: clean, professional fonts
if ispc
    defaultFont = 'Arial';
elseif ismac
    defaultFont = 'Helvetica';
else
    defaultFont = 'DejaVu Sans';
end
set(groot, 'DefaultTextFontName', defaultFont);
set(groot, 'DefaultAxesFontName', defaultFont);
set(groot, 'DefaultTextFontSize', 12);
set(groot, 'DefaultAxesFontSize', 12);
% Title formatting
set(groot, 'DefaultTextColor', berkeleyBlue);
%% Legend Properties
set(groot, 'DefaultLegendBox', 'on');
set(groot, 'DefaultLegendColor', 'white');
set(groot, 'DefaultLegendEdgeColor', 'black');
set(groot, 'DefaultLegendLineWidth', 1.0);
set(groot, 'DefaultLegendFontSize', 10);
set(groot, 'DefaultLegendLocation', 'best');
%% Colorbar Properties
set(groot, 'DefaultColorbarColor', 'black');
set(groot, 'DefaultColorbarLineWidth', 1.0);
set(groot, 'DefaultColorbarFontSize', 10);
set(groot, 'DefaultColorbarTickDirection', 'in');
%% Surface and Patch Properties
set(groot, 'DefaultSurfaceEdgeColor', 'none');
set(groot, 'DefaultPatchEdgeColor', 'black');
set(groot, 'DefaultPatchLineWidth', 1.0);
%% Histogram Properties
set(groot, 'DefaultHistogramEdgeColor', 'black');
set(groot, 'DefaultHistogramLineWidth', 1.0);
set(groot, 'DefaultHistogramFaceAlpha', 0.7);
%% Scatter Properties
set(groot, 'DefaultScatterLineWidth', 1.5);
set(groot, 'DefaultScatterMarkerEdgeColor', 'flat');
%% Save default colormap
% Create Berkeley-themed colormap
berkeleyMap = [
    linspace(1, berkeleyBlue(1), 128)';
    linspace(1, berkeleyBlue(2), 128)';
    linspace(1, berkeleyBlue(3), 128)'
]';
% Alternative: Blue to Gold gradient
blueGoldMap = [
    linspace(berkeleyBlue(1), californiaGold(1), 256)';
    linspace(berkeleyBlue(2), californiaGold(2), 256)';
    linspace(berkeleyBlue(3), californiaGold(3), 256)'
]';
% Set default colormap
set(groot, 'DefaultFigureColormap', blueGoldMap);
%% LaTeX Interpreter Settings
% Enable LaTeX for mathematical expressions
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultColorbarTickLabelInterpreter', 'latex');
%% Export Settings for Publication
% Configure print options for high-quality output
set(groot, 'DefaultFigureInvertHardCopy', 'off');
set(groot, 'DefaultFigurePaperPositionMode', 'auto');
%% Custom Berkeley Functions
% Store Berkeley colors in global variables for easy access
assignin('base', 'BERKELEY_BLUE', berkeleyBlue);
assignin('base', 'CALIFORNIA_GOLD', californiaGold);
assignin('base', 'BERKELEY_COLORS', struct(...
    'berkeleyBlue', berkeleyBlue, ...
    'californiaGold', californiaGold, ...
    'blueDark', blueDark, ...
    'goldDark', goldDark, ...
    'greenDark', greenDark, ...
    'roseDark', roseDark, ...
    'purpleDark', purpleDark, ...
    'redDark', redDark, ...
    'orangeDark', orangeDark, ...
    'tealDark', tealDark, ...
    'greyLight', greyLight, ...
    'greyMedium', greyMedium, ...
    'greyDark', greyDark ...
));
%% Success message
fprintf('🎨 Berkeley visual identity configured successfully!\n');
fprintf('   Primary colors: Berkeley Blue and California Gold\n');
fprintf('   Typography: %s font family\n', defaultFont);
fprintf('   Resolution: High-DPI ready\n');
fprintf('   Style: Publication quality\n');
%% Display color palette
if nargout == 0
    % Create a sample plot showing the color scheme
    figure('Name', 'Berkeley Color Scheme', 'NumberTitle', 'off');
    % Sample data
    x = linspace(0, 2*pi, 100);
    y = [sin(x); cos(x); sin(x+pi/4); cos(x+pi/4); ...
         sin(x+pi/2); cos(x+pi/2); sin(x+3*pi/4); cos(x+3*pi/4)];
    % Plot with Berkeley colors
    hold on;
    for i = 1:size(y, 1)
        plot(x, y(i, :) + (i-1)*0.5, 'LineWidth', 2);
    end
    hold off;
    title('UC Berkeley Color Scheme Demo', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('x', 'FontSize', 12);
    ylabel('y + offset', 'FontSize', 12);
    % Add color legend
    colorNames = {'Berkeley Blue', 'California Gold', 'Green Dark', ...
                  'Rose Dark', 'Purple Dark', 'Red Dark', 'Orange Dark', 'Teal Dark'};
    legend(colorNames, 'Location', 'eastoutside', 'FontSize', 10);
    % Add Berkeley branding
    text(0.02, 0.98, '🐻💙💛 UC Berkeley', 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontSize', 10, 'Color', berkeleyBlue);
end
end
%% Helper Functions
function rgb = hex2rgb(hexColor)
%HEX2RGB Convert hex color to RGB
    hexColor = strrep(hexColor, '#', '');
    r = hex2dec(hexColor(1:2)) / 255;
    g = hex2dec(hexColor(3:4)) / 255;
    b = hex2dec(hexColor(5:6)) / 255;
    rgb = [r, g, b];
end