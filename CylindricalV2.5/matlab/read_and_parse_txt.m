function dataArray = read_and_parse_txt(filename)
    % Open the file for reading
    fid = fopen(filename, 'r');
    
    % Check if the file was successfully opened
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Initialize arrays to store timestamps and data
    timestamps = [];
    data = [];
    
    % Read the file line by line
    while ~feof(fid)
        % Read the timestamp
        timestampLine = fgetl(fid);
        if ischar(timestampLine)
            timestamp = str2double(timestampLine);
            timestamps = [timestamps, timestamp];  % Append timestamp
        end
        
        % Read the data line
        dataLine = fgetl(fid);
        if ischar(dataLine)
            dataValue = str2double(dataLine(2:end-1));  % Remove square brackets and convert to double
            data = [data, dataValue];  % Append data value
        end
    end
    
    % Close the file
    fclose(fid);
    
    % Combine the timestamps and data into a 2xN array
    dataArray = [timestamps; data];
end