function timestamps = readTimestamps(filename)
    % Open the file for reading
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    timestamps = [];
    idx = 1;
    
    % Read the file line by line
    while ~feof(fid)
        % Read the timestamp line
        timestamp_line = fgetl(fid);
        if ischar(timestamp_line)
            timestamp = str2double(timestamp_line);
        else
            continue;
        end
        
        % Read the 2D array line (but discard it)
        array_line = fgetl(fid);
        
        % Store the timestamp
        timestamps(idx) = timestamp;
        idx = idx + 1;
    end
    
    % Close the file
    fclose(fid);
end
