% MATLAB Code to Visualize 8-Coloring of a 2D Grid

% Grid dimensions
Nx = 16; % Number of cells in the x-direction
Ny = 16; % Number of cells in the y-direction

% Initialize the grid to store color indices
grid_colors = zeros(Ny, Nx);

% Number of colors
num_colors = 4;

% Assign colors to the grid cells
for i = 1:Nx
    for j = 1:Ny
        % Coloring formula to ensure no adjacent cells share the same color
        % Including edge and diagonal adjacency
        grid_colors(j, i) = mod( ...
            (mod(i - 1, 2) * 4) + ...      % Based on i coordinate
            (mod(j - 1, 2) * 2) + ...      % Based on j coordinate
            mod(i + j - 2, 2), ...         % Combined parity of i and j
            num_colors) + 1;               % Ensure color indices start from 1
    end
end

% Create a colormap with 8 distinct colors
cmap = [
    1, 0, 0;        % Red
    0, 1, 0;        % Green
    0, 0, 1;        % Blue
    1, 1, 0;        % Yellow
];

% Display the grid with colored cells
figure;
imagesc(grid_colors);
colormap(cmap);
colorbar('Ticks', 1:num_colors, 'TickLabels', 1:num_colors);
axis equal tight;
title('8-Coloring of a 2D Grid (No Adjacent Cells Share the Same Color)');
xlabel('X-axis');
ylabel('Y-axis');
set(gca, 'XTick', 1:Nx, 'YTick', 1:Ny, 'TickLength', [0 0]);

% Overlay grid lines for better visualization
hold on;
for k = 0.5:Nx
    plot([k, k], [0.5, Ny+0.5], 'k-');
end
for k = 0.5:Ny
    plot([0.5, Nx+0.5], [k, k], 'k-');
end
hold off;
