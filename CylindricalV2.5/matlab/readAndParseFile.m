function data = readAndParseFile(filename)
    % Open the file for reading
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Initialize variables
    data = [];
    idx = 1;
    
    % Read the file line by line
    while ~feof(fid)
        % Read the timestamp line (but ignore it in this function)
        timestamp_line = fgetl(fid);
        if ~ischar(timestamp_line)
            continue;
        end
        
        % Read the 2D array line
        array_line = fgetl(fid);
        if ~ischar(array_line)
            continue;
        end
        
        array_data = str2num(array_line); %#ok<ST2NM>
        
        % Store the 2D array in the 3D array
        if isempty(data)
            % Initialize the data array with the size of the first array
            [rows, cols] = size(array_data);
            data = zeros(rows, cols, 0); % Initialize as empty along 3rd dimension
        end
        
        data(:, :, idx) = array_data;
        idx = idx + 1;
    end
    
    % Close the file
    fclose(fid);
end
