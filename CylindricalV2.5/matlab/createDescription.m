%Create a description for each tests using params
% List of .mat files
mat_files = {'EB_Neumann.mat', 'EB_Neumann_Source.mat', 'EB_Dirichlet.mat'}; % Replace with your actual filenames

% Initialize an empty cell array to store the data
data = {};

% Load the first .mat file to get the field names
load(mat_files{1}, 'params');
field_names = fieldnames(params);

% Add 'Filename' to the beginning of the field names
all_field_names = ['Filename'; field_names];

% Initialize the data cell array with headers
data = [all_field_names'];

% Loop through each .mat file
for i = 1:length(mat_files)
    % Load the params variable from the .mat file
    load(mat_files{i}, 'params');
    
    % Initialize a cell array to hold the values for this row
    row_data = cell(1, length(all_field_names));
    
    % First column is the filename
    row_data{1} = mat_files{i};
    
    % Loop through each field and get the corresponding value
    for j = 1:length(field_names)
        field_value = params.(field_names{j});
        row_data{j+1} = field_value;
    end
    
    % Append the row data to the data cell array
    data = [data; row_data];
end

% Write the data to an Excel file called 'Tests.xlsx'
writecell(data, 'TestDescription.xlsx');