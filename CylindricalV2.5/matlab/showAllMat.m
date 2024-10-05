function showAllMat()
    % Get a list of all .mat files in the current folder
    matFiles = dir('*.mat');
    
    % Loop through each file and print the name
    for i = 1:length(matFiles)
        disp(matFiles(i).name)
    end
end