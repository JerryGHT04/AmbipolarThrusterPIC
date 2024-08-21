% MATLAB code to read Parameters.txt and create an Excel file with a table

% Define the meanings and units for each parameter
parameter_meanings = {
    'Output Folder', '';
    'Max Time (s)', 's';
    'Time Step (s)', 's';
    'Number of Cells', '';
    'Number of Superparticles', '';
    'Electron Density (m^-3)', 'm^-3';
    'Neutral Density (m^-3)', 'm^-3';
    'Ionization Efficiency', '';
    'Reynolds Number', '';
    'Initial Electron Temperature (K)', 'K';
    'Ion Temperature (K)', 'K';
    'Spatial Step dY (m)', 'm';
    'Spatial Step dZ (m)', 'm';
    'Ionization Chamber Length X2 (m)', 'm';
    'Total Simulation Length X3 (m)', 'm';
    'X4', '';
    'Anode Voltage (V)', 'V';
    'Magnetic Field B0z (T)', 'T';
    'UeX', '';
    'Rate of Electron Generation Ne_dot (s^-1)', 's^-1';
    'Particle Spawn Time (s)', 's'
};

% Filename of the parameters file
filename = 'Parameters.txt';

% Read the Parameters.txt file
fileID = fopen(filename, 'r');
if fileID == -1
    error('Cannot open file: %s', filename);
end

parameters = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);
parameters = parameters{1};

% Convert numerical values from strings to appropriate types
for i = 2:length(parameters)
    if ~isempty(parameters{i}) && all(isstrprop(parameters{i}, 'digit') | isstrprop(parameters{i}, 'punct'))
        parameters{i} = str2double(parameters{i});
    end
end

% Create a table
parameter_table = table(parameter_meanings(:, 1), parameters, 'VariableNames', {'Description', 'Value'});

% Write the table to an Excel file
writetable(parameter_table, 'parameters.xlsx');

disp('Excel file "parameters.xlsx" created successfully.');
