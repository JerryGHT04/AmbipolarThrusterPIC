# Grid dimensions
Nx = 16  # Number of cells in the x-direction
Ny = 16  # Number of cells in the y-direction

# Initialize the grid to store color indices
grid_colors = Array{Int}(undef, Ny, Nx)

# Number of colors
num_colors = 8

# Assign colors to the grid cells
for i in 1:Nx
    for j in 1:Ny
        # Adjust indices to start from 0
        ii = i - 1
        jj = j - 1

        # Coloring formula to ensure no adjacent cells share the same color
        # Including edge and diagonal adjacency
        color = mod( (mod(ii, 2) * 4) +   # Based on i coordinate parity
                     (mod(jj, 2) * 2) +   # Based on j coordinate parity
                     mod(ii + jj, 2),     # Combined parity of i and j
                     num_colors ) + 1     # Ensure color indices start from 1

        grid_colors[j, i] = color
    end
end

# Create a colormap with 8 distinct colors
using Colors, Plots

# Define 8 distinct colors
cmap = [
    RGB(1, 0, 0),       # Red
    RGB(0, 1, 0),       # Green
    RGB(0, 0, 1),       # Blue
    RGB(1, 1, 0),       # Yellow
    RGB(1, 0, 1),       # Magenta
    RGB(0, 1, 1),       # Cyan
    RGB(0.5, 0.5, 0.5), # Gray
    RGB(1, 0.5, 0),     # Orange
]

# Display the grid with colored cells
p = heatmap(
    grid_colors;
    color = cmap,
    colorbar = false,
    aspect_ratio = :equal,
    xlims = (0.5, Nx + 0.5),
    ylims = (0.5, Ny + 0.5),
    xticks = 1:Nx,
    yticks = 1:Ny,
    xlabel = "X-axis",
    ylabel = "Y-axis",
    title = "8-Coloring of a 2D Grid (No Adjacent Cells Share the Same Color)",
    legend = false,
    border = :none,
)

# Add grid lines for better visualization
hline!(p, 0.5:Ny+0.5; color = :black, lw = 0.5)
vline!(p, 0.5:Nx+0.5; color = :black, lw = 0.5)

# Display the plot
display(p)
