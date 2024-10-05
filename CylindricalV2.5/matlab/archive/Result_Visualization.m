% Visualization
save_figures = false; % Set to true if you want to save the figures
foldername = ["100eV4A.mat"];

% Save each frame of Figure 1 and 2 to the following save_path.
% Each picture is named as xxx_num, where num represents the timestep (the index of Tspace)
save_path = fullfile('..', 'result_field', foldername);

% Create the directory if it doesn't exist
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

% Assuming you have the following variables:
% - ni, uiz, ne, uez, potential: 3D arrays of size (Nr x Nz x Nt)
% - Tspace: vector of time steps, length Nt
% - zspace: vector of z positions, length Nz
% - rspace: vector of r positions, length Nr
% - R: maximum radial coordinate

%% Figure 1: Plot ni, uiz, ne, uez animation
figure(1);
clf;
figureHandle = figure;
set(figureHandle, 'Position', [0, 300, 2000, 700]); % [left, bottom, width, height]

for t = 1:length(Tspace)
    % Extract data at time t
    ni_t = squeeze(ni(:, :, t));   % Size: (Nr x Nz)
    uiz_t = squeeze(uiz(:, :, t));
    ne_t = squeeze(ne(:, :, t));
    uez_t = squeeze(uez(:, :, t));
    
    % Subplot 1: ni
    subplot(2,3,1)
    surf(zspace, rspace, ni_t, 'EdgeColor', 'none'); % Disable grid bezel
    view(2); % View from above (2D view)
    shading interp;
    xlabel('Z-axis');
    ylabel('R-axis');
    title(sprintf('ni at t = %.2f ns', Tspace(t)*1e9));
    colorbar;
    xlim([0 max(zspace)])
    % Subplot 2: uiz
    subplot(2,3,2)
    surf(zspace, rspace, uiz_t, 'EdgeColor', 'none');
    view(2);
    shading interp;
    xlabel('Z-axis');
    ylabel('R-axis');
    title(sprintf('uiz at t = %.2f ns', Tspace(t)*1e9));
    colorbar;
    xlim([0 max(zspace)])
    % Subplot 3: ne
    subplot(2,3,4)
    surf(zspace, rspace, ne_t, 'EdgeColor', 'none');
    view(2);
    shading interp;
    xlabel('Z-axis');
    ylabel('R-axis');
    title(sprintf('ne at t = %.2f ns', Tspace(t)*1e9));
    colorbar;
    xlim([0 max(zspace)])
    % Subplot 4: uez
    subplot(2,3,5)
    surf(zspace, rspace, uez_t, 'EdgeColor', 'none');
    view(2);
    shading interp;
    xlabel('Z-axis');
    ylabel('R-axis');
    title(sprintf('uez at t = %.2f ns', Tspace(t)*1e9));
    colorbar;
    xlim([0 max(zspace)])
    subplot(2,3,3)
    surf(zspace, rspace, ni_t.*uiz_t, 'EdgeColor', 'none'); % Disable grid bezel
    view(2); % View from above (2D view)
    shading interp;
    xlabel('Z-axis');
    ylabel('R-axis');
    title(sprintf('ni*uiz at t = %.2f ns', Tspace(t)*1e9));
    colorbar;
    xlim([0 max(zspace)])
    potential_t = squeeze(potential(:, :, t)); % Size: (Nr x Nz)
    % Subplot 4: uez
    subplot(2,3,6)
    surf(zspace, rspace, potential(:, :, t), 'EdgeColor', 'none');
    view(2);
    shading interp;
    xlabel('Z-axis');
    ylabel('R-axis');
    title(sprintf('potential at t = %.2f ns', Tspace(t)*1e9));
    colorbar;
    xlim([0 max(zspace)])
    
    % Adjust layout
    drawnow;
    
    % Save each frame if required
    if save_figures
        filename = sprintf('figure1_%d.png', t);
        saveas(gcf, fullfile(save_path, strcat(foldername,filename)));
    end
end
close
%% Figure 3: Plot time-averaged ni, uiz, ne, uez at four different r
figure(3);
clf;

RNum = 4;
rPos = linspace(0, R, RNum); % Positions in R to plot
[~, rIndices] = min(abs(rspace - rPos'), [], 2); % Find closest indices in rspace

% Time-averaged data
ni_time_avg = mean(ni, 3);   % Size: (Nr x Nz)
uiz_time_avg = mean(uiz, 3);
ne_time_avg = mean(ne, 3);
uez_time_avg = mean(uez, 3);

% Subplot 1: ni
subplot(2,2,1)
hold on
for i = 1:RNum
    plot(zspace, ni_time_avg(rIndices(i), :), 'DisplayName', sprintf('r = %.4f', rPos(i)));
end
hold off
xlabel('Z-axis');
ylabel('ni');
title('Time-Averaged ni at Different r');
legend;
grid on;

% Subplot 2: uiz
subplot(2,2,2)
hold on
for i = 1:RNum
    plot(zspace, uiz_time_avg(rIndices(i), :), 'DisplayName', sprintf('r = %.4f', rPos(i)));
end
hold off
xlabel('Z-axis');
ylabel('uiz');
title('Time-Averaged uiz at Different r');
legend;
grid on;

% Subplot 3: ne
subplot(2,2,3)
hold on
for i = 1:RNum
    plot(zspace, ne_time_avg(rIndices(i), :), 'DisplayName', sprintf('r = %.4f', rPos(i)));
end
hold off
xlabel('Z-axis');
ylabel('ne');
title('Time-Averaged ne at Different r');
legend;
grid on;

% Subplot 4: uez
subplot(2,2,4)
hold on
for i = 1:RNum
    plot(zspace, uez_time_avg(rIndices(i), :), 'DisplayName', sprintf('r = %.4f', rPos(i)));
end
hold off
xlabel('Z-axis');
ylabel('uez');
title('Time-Averaged uez at Different r');
legend;
grid on;
if save_figures
        filename = sprintf('figure3_%d.png', t);
        saveas(gcf, fullfile(save_path, strcat(foldername,filename)));
    end
%% Figure 4: Plot potential at four different r
figure(4);
clf;
hold on

% Time-averaged potential
potential_time_avg = mean(potential, 3); % Size: (Nr x Nz)

for i = 1:RNum
    plot(zspace, potential_time_avg(rIndices(i), :), 'DisplayName', sprintf('r = %.4f', rPos(i)));
end
hold off
xlabel('Z-axis');
ylabel('Potential');
title('Time-Averaged Potential at Different r');
legend;
grid on;
if save_figures
        filename = sprintf('figure4_%d.png', t);
        saveas(gcf, fullfile(save_path, strcat(foldername,filename)));
    end
%% Figure 4: Plot potential at four different r
figure(5);
clf;
hold on

% Time-averaged potential
fe_time_avg = mean(ni./ne, 3); % Size: (Nr x Nz)

for i = 1:RNum
    plot(zspace, fe_time_avg(rIndices(i), :), 'DisplayName', sprintf('r = %.4f', rPos(i)));
end
hold off
xlabel('Z-axis');
ylabel('fe');
title('Time-Averaged fe at Different r');
legend;
grid on;
if save_figures
        filename = sprintf('figure5_%d.png', t);
        saveas(gcf, fullfile(save_path, strcat(foldername,filename)));
end