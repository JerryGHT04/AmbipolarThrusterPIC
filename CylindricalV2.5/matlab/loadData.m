function loadData(parName, saveName)
    params = parseParameters('../' + "" +  parName);
    path = string(params.output_folder);
     if nargin == 1
        saveName = path(1:end-1);  % No parameter passed, set flag to true
     elseif nargin == 2
        saveName = saveName;  % One parameter passed, set flag to false
     else
        error('Too many input arguments');
     end
    
    potential = readAndParseFile('../' + path + 'potential.txt');
    ne = readAndParseFile('../' + path + 'ne.txt');
    ni = readAndParseFile('../' + path + 'ni.txt');
    nn =  readAndParseFile('../' + path + 'nn.txt');
    uiz = readAndParseFile('../' + path + 'uiz.txt');
    uix = readAndParseFile('../' + path + 'uix.txt');
    uiy = readAndParseFile('../' + path + 'uiy.txt');
    uez = readAndParseFile('../' + path + 'uez.txt');
    uex = readAndParseFile('../' + path + 'uex.txt');
    uey = readAndParseFile('../' + path + 'uey.txt');
    energy = read_and_parse_txt('../' + path + 'energy.txt');
    energy = energy(2,:);
    thrust = read_and_parse_txt('../' + path + 'thrust.txt');
    thrust = thrust(2,:);
    Tspace = readTimestamps('../' + path + 'ni.txt');
    
    
    
    Nr = params.Nr;
    Nz = params.Nz;
    R = params.dR;
    L = params.dZ;
    rspace = linspace(0,R,Nr);
    zspace = linspace(0,L,Nz);
    
    dz = L/(Nz-1);
    dr = R/(Nr-1);
    
    me =  9.10938356E-31;
    qe =  1.6021766208e-19;
    mi = 2.1802e-25;
    
    nodeVolume = [];
    for i = 1:Nr
        for j = 1:Nz
            if i == 1
                if j == 1 || j == Nz
                    nodeVolume(i, j) = 1/2 * dz * pi * ((rspace(i) + dr/2)^2 - rspace(i)^2);
                else
                    nodeVolume(i, j) = dz * pi * ((rspace(i) + dr/2)^2 - rspace(i)^2);
                end
            elseif i == Nr
                if j == 1 || j == Nz
                    nodeVolume(i, j) = 1/2 * dz * pi * (rspace(i)^2 - (rspace(i) - dr/2)^2);
                else
                    nodeVolume(i, j) = dz * pi * (rspace(i)^2 - (rspace(i) - dr/2)^2);
                end
            else
                if j == 1 || j == Nz
                    nodeVolume(i, j) = 1/2 * dz * pi * ((rspace(i) + dr/2)^2 - (rspace(i) - dr/2)^2);
                else
                    nodeVolume(i, j) = dz * pi * ((rspace(i) + dr/2)^2 - (rspace(i) - dr/2)^2);
                end
            end
        end
    end
    
    Nt = length(Tspace);
    Ni = [];
    Ne = [];
    Nn = [];
    Ee = [];
    EeR = [];
    c = 3e8;
    me =  9.10938356E-31;
    qe =  1.6021766208e-19;
    mi = 2.1802e-25; %Xenon atom
    kB = 1.380649e-23;
    for t = 1:Nt
        % Element-wise multiplication and summation over spatial dimensions
        Ni(t) = sum(sum(ni(:, :, t) .* nodeVolume));
        Ne(t) = sum(sum(ne(:, :, t) .* nodeVolume));
        Nn(t) = sum(sum(nn(:, :, t) .* nodeVolume));
        
        %calculate mean electron energy
        % Calculate velocity magnitude
        V = sqrt(uex(:, :, t).^2 + uey(:, :, t).^2 + uez(:, :, t).^2);
        Ee(t) = sum(sum(0.5*me.*V.^2 .*ne(:, :, t) .* nodeVolume)) / Ne(t);
        EeR(t) =  sum(sum( (1./sqrt(1-V.^2/c^2)-1)*me*c^2.*ne(:, :, t) .* nodeVolume )) / Ne(t);
    end
    
    tr = params.tr * 1e-9;
    Ne_dot_max = params.BeamCurrent/qe;
    % Initialize Ne_dot array
    Ne_dot = zeros(size(Tspace));
    if tr == 0
        % If tr is zero, Ne_dot is Ne_dot_max for all times
        Ne_dot(:) = Ne_dot_max;
    else
        % Compute Ne_dot for tr > 0
        Ne_dot = Ne_dot_max * min(Tspace / tr, 1);
    end
    
    Nn_dot = zeros(size(Tspace));
    Nn_dot =  params.neutralRate;
    

    % Store variables to the workspace
    varsToSave = {'potential', 'ne', 'ni', 'nn', 'uiz', 'uix', 'uiy', 'uez', 'uex', 'uey', 'energy', 'Tspace', ...
                  'params', 'Nr', 'Nz', 'R', 'L', 'rspace', 'zspace', 'dz', 'dr', 'nodeVolume', 'Nt', 'Ni', 'Ne', ...
                  'Nn', 'Ee', 'EeR', 'c', 'me', 'qe', 'mi', 'kB', 'tr', 'Ne_dot_max', 'Ne_dot', 'Nn_dot'};

    for i = 1:length(varsToSave)
        assignin('base', varsToSave{i}, eval(varsToSave{i}));
    end
    save(saveName)
    disp('All variables saved to workspace.');
end