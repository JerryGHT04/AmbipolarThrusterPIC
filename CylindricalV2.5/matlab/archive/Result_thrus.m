mi = 2.1802e-25;  % Ion mass
fileNames = ["100eV4A.mat","300eV4A.mat","500eV4A.mat"]

%R = 0.05;  % Assuming the radius is defined

% Load the first file to determine the size of Tspace
load(fileNames(1), 'Tspace');

% Preallocate matrices based on the number of files and time steps
thrustMat = zeros(length(fileNames), length(Tspace));
maxphiMat = zeros(length(fileNames), length(Tspace));
maxuizMat = zeros(length(fileNames), length(Tspace));
for i = 1:length(fileNames)
    % Load only the variables you need
    load(fileNames(i), 'ni', 'uiz', 'potential', 'Tspace');
    
    thrust = zeros(1, length(Tspace));  % Preallocate thrust array
    maxpotential = zeros(1, length(Tspace));  % Preallocate max potential array
    
    % Loop over time steps to calculate thrust at each time step
    for t = 1:length(Tspace)
        % Get the ion density and velocity at time t
        ni_t = ni(:, :, t);  % Ion density at time t
        uiz_t = uiz(:, :, t);  % Ion velocity at time t
        
        % Calculate thrust for the current time step
        thrust(t) = sum(sum(ni_t .* mi .* uiz_t.^2.*sign(uiz_t))) * pi * R^2;
    end
    
    thrustMat(i, :) = thrust';  % Store thrust for the current file
    
    % Loop over time steps to calculate maximum potential at each time step
    for t = 1:length(Tspace)
        potential_t = potential(:, :, t);  % Potential at time t
        
        % Find the maximum and minimum potential values
        maxval = max(potential_t(:));
        minval = min(potential_t(:));
        
        % Store the potential with the largest absolute magnitude
        if abs(maxval) > abs(minval)
            maxpotential(t) = maxval;
        else
            maxpotential(t) = minval;
        end
    end
    
    maxphiMat(i, :) = maxpotential';  % Store potential for the current file


    uizArray=[];
    % Loop over time steps to calculate maximum potential at each time step
    for t = 1:length(Tspace)
        uiz_t = uiz(:, :, t);  % Potential at time t
        
        % Find the maximum and minimum potential values
        maxval = max(uiz_t(:));
        minval = min(uiz_t(:));
        
        % Store the potential with the largest absolute magnitude
        if abs(maxval) > abs(minval)
            uizArray(t) = maxval;
        else
            uizArray(t) = minval;
        end
    end
    
    maxuizMat(i,:) = uizArray';
    % Plotting thrust vs time
    figure(1)
    plot(Tspace * 1e9, thrust, 'LineWidth', 2);
    hold on
    xlabel('Time [ns]');
    ylabel('Thrust (N)');
    title('Thrust = ni * mi * uiz^2 * pi * R^2');
    grid on;
    legend(fileNames)
end



% Plotting potential vs time
figure(2)
clf;
for i = 1:length(fileNames)
    plot(Tspace * 1e9, maxphiMat(i, :), 'LineWidth', 2);
    hold on;
end
xlabel('Time [ns]');
ylabel('Potential (V)');
title('Maximum Potential');
grid on;
legend(fileNames)
% Plotting uiz vs time
figure(3)
clf;
for i = 1:length(fileNames)
    plot(Tspace * 1e9, maxuizMat(i, :), 'LineWidth', 2);
    hold on;
end
xlabel('Time [ns]');
ylabel('uiz (m/s)');
title('Maximum uiz');
grid on;
legend(fileNames)

figure(4)
clf;
%change of number

plot(Tspace*1e9, Ni)
hold on

ylabel("Total change in number of particles")
xlabel("time [ns]")
legend("$\Delta Ni$", "interpreter", "latex", "Location","northwest")
%% 