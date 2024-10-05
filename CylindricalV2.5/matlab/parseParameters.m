function params = parseParameters(filename)
    % Function to parse parameters from a text file and return them as fields in a struct
    %
    % Usage:
    %   params = parseParameters('path/to/your/file.txt');
    %
    % The function returns a struct 'params' with the following fields:
    %   - output_folder
    %   - maxTime
    %   - timeStep
    %   - Nr
    %   - Nz
    %   - numParticle
    %   - edensity
    %   - ndensity
    %   - Telectron
    %   - Tion
    %   - dR
    %   - dZ
    %   - V_z0
    %   - V_zmax
    %   - BeamEnergy
    %   - BeamCurrent
    %   - BeamSuperParticle
    %   - rb
    %   - tr
    %   - neutralRate
    %   - recover
    %   - save
    %   - save_path
    %   - Nmax
    %   - writeTimeStep

    % Open the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    % Read the file line by line into a cell array
    parameters = {};
    idx = 1;
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            parameters{idx} = strtrim(line); % Remove leading/trailing whitespace
            idx = idx + 1;
        end
    end
    fclose(fid);

    % Check that we have at least 25 parameters
    if length(parameters) < 25
        error('Expected at least 25 parameters, but got %d', length(parameters));
    end

    % Initialize struct to hold parameters
    params = struct();

    % Parse and assign each parameter
    params.output_folder       = parameters{1};                         % String
    params.maxTime             = str2double(parameters{2});             % Float
    params.timeStep            = str2double(parameters{3});             % Float
    params.Nr                  = str2double(parameters{4});             % Integer
    params.Nz                  = str2double(parameters{5});             % Integer
    params.numParticle         = str2double(parameters{6});             % Integer
    params.edensity            = str2double(parameters{7});             % Float
    params.ndensity            = str2double(parameters{8});             % Float
    params.Telectron           = str2double(parameters{9});             % Float
    params.Tion                = str2double(parameters{10});            % Float
    params.dR                  = str2double(parameters{11});            % Float
    params.dZ                  = str2double(parameters{12});            % Float
    params.BeamEnergy          = str2double(parameters{13});            % Float
    params.BeamCurrent          = str2double(parameters{14});            % Float
    params.BeamSuperParticle   = str2double(parameters{15});            % Integer
    params.rb                  = str2double(parameters{16});            % Float
    params.tr                  = str2double(parameters{17});            % Float
    params.neutralRate      = str2double(parameters{18});            % Float
    params.recover             = str2double(parameters{19});            % Integer
    params.save                = str2double(parameters{20});            % Integer
    params.save_path           = parameters{21};                        % String
    params.Nmax                = str2double(parameters{22});            % Integer
    params.writeTimeStep       = str2double(parameters{23});            % Float
    params.Bz       = str2double(parameters{24});            % Float
    params.boundaryCondition       = str2double(parameters{25});            % Float
    params.saveFrequency       = str2double(parameters{26});            % Float
end
