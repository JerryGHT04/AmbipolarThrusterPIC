%% Time-dependent properties
%Mean ion velocity
%compare ni with time at each node with theoretical
Ue = zeros(size(uex));
E = zeros(size(uex));
sigma = zeros(size(uex));

Ni_sum = zeros(1,length(Tspace));
Ne_sum = zeros(1,length(Tspace));
Nn_sum = zeros(1,length(Tspace));
Ue_mean = zeros(1,length(Tspace));
Uez_mean = zeros(1,length(Tspace));
Uer_mean = zeros(1,length(Tspace));
Ui_mean = zeros(1,length(Tspace));
Uiz_mean = zeros(1,length(Tspace));
Uir_mean = zeros(1,length(Tspace));
Uiz_center_mean = zeros(1,length(Tspace));
Q_beam = zeros(1,length(Tspace));
for i = 1:length(Tspace)
    Ue(:,:,i) = sqrt(uex(:,:,i).^2 + uey(:,:,i).^2 + uez(:,:,i).^2);
    Ui(:,:,i) = sqrt(uix(:,:,i).^2 + uiy(:,:,i).^2 + uiz(:,:,i).^2);

    Ni_sum(i) = sum(sum(ni(:,:,i).*nodeVolume,1),2);
    Ne_sum(i) = sum(sum(ne(:,:,i).*nodeVolume,1),2);
    Nn_sum(i) = sum(sum(nn(:,:,i).*nodeVolume,1),2);

    Ue_mean(i) = mean(mean(Ue(:,:,i),1),2);
    Uez_mean(i) = (mean(mean(uez(:,:,i),1),2));
    Uer_mean(i) = mean(mean(sqrt(uex(:,:,i).^2 + uey(:,:,i).^2),1),2);
    Ui_mean(i) = mean(mean(Ui(:,:,i),1),2);
    Uiz_mean(i) = (mean(mean(uiz(:,:,i),1),2));
    Uir_mean(i) = mean(mean(sqrt(uix(:,:,i).^2 + uiy(:,:,i).^2),1),2);
    Uiz_center_mean(i) = mean(mean(Ui(1,:,i),1),2);

    Q_beam(i) = sum(sum((ni(:,end,i) - ne(:,end,i)).*nodeVolume,1),2);
end

%% 1. thrust with time
figure(1)
clf;
scatter(Tspace, thrust) 
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 12)
ylabel("Thrust (N)")

%% 2. average ion velocity with time
figure(2)
clf;
plot(Tspace,Ui_mean)
hold on
plot(Tspace,Uiz_mean)
hold on
plot(Tspace,Uir_mean)
legend("Ui","Uiz","Uir")
title("Average ion speed")
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 12)

%% 3. Average Ion Velocity at Different Radial Positions with time
figure(3)
clf; % Clear the current figure

% Define the number of radial positions to plot
RNum = 3;

% Define the radial positions to plot, from 0 to R (ensure R is defined in your workspace)
rPos = linspace(0, R, RNum); % Positions in R to plot (e.g., meters)

% Find the closest indices in rspace to the desired radial positions
% Assuming rspace is a vector containing radial positions corresponding to the first dimension of uiz
% The 'min' function returns the minimum value and its index; we only need the index
[~, rIndices] = min(abs(rspace - rPos'), [], 2); % rIndices is a column vector of size [RNum x 1]

% Initialize a cell array for legend entries
legend_entries = cell(1, RNum);

% Activate hold to plot multiple lines on the same figure
hold on

% Define a colormap for different plots (optional, for better visualization)
colors = lines(RNum); % 'lines' is a default MATLAB colormap with distinct colors

% Loop over each radial position and plot Uiz vs Tspace
for j = 1:RNum
    rIdx = rIndices(j); % Index in rspace closest to rPos(j)
    
    % Extract Uiz at the given radial index across all z and time
    % Assuming uiz has dimensions [r, z, time]
    % Average over the z-dimension to get Z-averaged Uiz
    % Resulting in a vector of size [1 x 1 x time], squeeze to [time x 1]
    Uiz_rj = squeeze(mean(uiz(rIdx, :, :), 2)); % [time x 1]
    
    % Plot Uiz vs Tspace for the current radial position
    % Customize the line style, color, and width as desired
    plot(Tspace, Uiz_rj, 'Color', colors(j, :), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('r = %.4f m', rPos(j)))
    
    % Optionally, store legend entries if not using 'DisplayName'
    % legend_entries{j} = sprintf('r = %.2f m', rPos(j));
end

% Add labels and title with LaTeX interpreter for better formatting
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$U_{iz}$ (m/s)', 'Interpreter', 'latex', 'FontSize', 12)
title('Average Ion Velocity $U_{iz}$ at Different Radial Positions', 'Interpreter', 'latex', 'FontSize', 14)

% Add legend with LaTeX interpreter and place it optimally
legend('Interpreter', 'latex', 'Location', 'best')

% Enhance plot readability
grid on
hold off


%% 4. Number of particle with time
figure(4)
clf;
hold on  % Move hold on here to avoid redundant hold on commands

% Plot the data with DisplayName properties for the legend
plot(Tspace, Ni_sum - Ni_sum(1), 'DisplayName', '$\Delta N_i$')
plot(Tspace, Ne_sum - Ne_sum(1), 'DisplayName', '$\Delta N_e$')
plot(Tspace, -(Nn_sum - Nn_sum(1)), 'DisplayName', '$-\Delta N_n$')

% Perform polyfit on Nn and plot the fitted line
coef = polyfit(Tspace, -(Nn_sum - Nn_sum(1)), 1);
fitted_line = polyval(coef, Tspace);
plot(Tspace, fitted_line, '--', 'DisplayName', sprintf('$y = (%.3e) x + (%.3e)$', coef(1), coef(2)))

% Add title and configure legend
title("Number Balance")
legend('Interpreter', 'latex', 'Location', 'northwest')
hold off
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 12)

%% 5. Charge neutrality in beanm

figure(5)
clf;
plot(Tspace, Q_beam)
ylabel("Charge in beam [e]")
xlabel("Time (s)")

