%% Enter requried sim names here
simNames = ["100eV4A","300eV4A","500eV4A"];

%% 
setFigurePreset('publication')
load("experimentsigma.mat")
load("RBEB.mat")
me =  9.10938356E-31;
qe =  1.6021766208e-19;
ERBEB = T/qe;
sigmaRBEB = sigmaRBEBav;
clf;

for idx = 1:length(simNames)
    %if .mat does not exit, loadData the sim
    matName = strcat(simNames(idx),".mat");
    parName = strcat("params_",simNames(idx), ".txt");
    if exist(matName, 'file') == 2
        disp('Found .mat, loading');
        load(matName)
    else
        disp('.mat file does not exist, load data');
        loadData(parName,matName)
    end
    
    %1. ionization(Ni) vs t
    % independent graphs
    plotVerifyIonization(uex, uey, uez, ni, ne, nn, Tspace, nodeVolume, energy, Ecombined, sigmacombined, ERBEB, sigmaRBEB, simNames, idx)
    
    %% 
    %6. Beam neutrality vs t
    % independent
    % this is to verify concept: a thruster has benefit of an automatic
    % charge-neutral beam
    Qb = zeros(1,length(Tspace));
    rbidx = floor(params.rb/dr);
    for i = 1:length(Tspace)
        Ni_sum = sum(ni(:,end,i).*nodeVolume(:,end),1);
        Ne_sum = sum(ne(:,end,i).*nodeVolume(:,end),1);
        Qb(i) = Ni_sum - Ne_sum;
    end
    figure(6)
    clf;
    plot(Tspace, Qb)
    hold on
    xlabel("Time")
    ylabel("Beam charge (e)")
    saveFigureToFolder(simNames(idx), "figure6", "png")
    
    %% 
    %7. Thrust vs t
    % independent
    figure(7)
    clf;
    plot(Tspace, smooth(thrust))
   
    xlabel("Time")
    ylabel("Thrust (N)")
    saveFigureToFolder(simNames(idx), "figure7", "png")
    %% 
    %8. neutrality averaged in beam, surf plot t and fe
    % independent graphs
    % this is to ensure a moving potential well to accelerate ion, compare with
    % theory
    ni_c = zeros(size(ni,2),length(Tspace));
    ne_c = zeros(size(ni,2),length(Tspace));
    for i = 1:length(Tspace)
        ni_c(:,i) = mean(ni(1:rbidx,:,i),1);
        ne_c(:,i) = mean(ne(1:rbidx,:,i),1);
    end
    figure(8)
    clf;
    fe = ni_c./ne_c;
    surf(zspace,Tspace,(fe)', 'EdgeColor', 'none')
    colorbar;
    view(2);
    xlabel("Z")
    ylabel("Time")
    shading interp;
    saveFigureToFolder(simNames(idx), "figure8", "png")
    %% 
    %9. potential averaged in beam, surf plot t and phi
    % independent graphs
    % to show moving potential well
    potential_c = zeros(size(potential,2),length(Tspace));
    for i = 1:length(Tspace)
        potential_c(:,i) = mean(potential(1:rbidx,:,i),1);
    end
    figure(9)
    clf;
    surf(zspace,Tspace,(potential_c)', 'EdgeColor', 'none')
    colorbar;
    view(2);
    xlabel("Z")
    ylabel("Time")
    shading interp;
    saveFigureToFolder(simNames(idx), "figure9", "png")
    
end

%% 
%animation

%% 
function setFigurePreset(mode)
    % Set default figure properties based on the mode: 'presentation' or 'publication'
    switch lower(mode)
        case 'presentation'
            set(0, 'DefaultLineLineWidth', 2); % Line width for publication
            set(0, 'DefaultAxesFontSize', 18); % Font size for axes
            set(0, 'DefaultTextFontSize', 18); % Font size for text (xlabel, ylabel)
            set(0, 'DefaultLegendLocation', 'northwest')  % Set location to northeast
        case 'publication'
            set(0, 'DefaultLineLineWidth', 2); % Line width for publication
            set(0, 'DefaultAxesFontSize', 18); % Font size for axes
            set(0, 'DefaultTextFontSize', 18); % Font size for text (xlabel, ylabel)
            set(0, 'DefaultAxesTitleFontSize', 0.00001)
            set(0, 'DefaultLegendLocation', 'northwest')  % Set location to northeast
            
        otherwise
            error('Invalid mode. Use "presentation" or "publication".');
    end
end


%% 
function saveFigureToFolder(simName, fileName, format)
    % Check if there's a current figure
    if isempty(get(0, 'Children'))
        error('No figure exists to save. Create a figure before calling saveFigureToFolder.');
    end

    % Convert inputs to character vectors if they're not already
    simName = char(simName);
    fileName = char(fileName);
    format = char(format);

    % Check if the folder exists, if not, create it
    folderName = fullfile('..', ['figures_' simName]);
    folderName = char(folderName);  % Convert to character vector
    if ~exist(folderName, 'dir')
        [status, msg] = mkdir(folderName);
        if ~status
            error('Failed to create folder: %s. Error: %s', folderName, msg);
        end
    end

    % Define the full path where the image will be saved
    fullFilePath = fullfile(folderName, [fileName '.' format]);
    fullFilePath = char(fullFilePath);  % Convert to character vector

    % Save the current figure to the specified path in the desired format
    try
        print(gcf, fullFilePath, ['-d' format]);
        disp(['Figure saved to ', fullFilePath]);
    catch ME
        error('Failed to save figure. Error: %s', ME.message);
    end
end

%% verify ionization function
function plotVerifyIonization(uex, uey, uez, ni, ne, nn, Tspace, nodeVolume, energy, Ecombined, sigmacombined, ERBEB, sigmaRBEB, simNames, k)
    %compare ni with time at each node with theoretical
    Ue = zeros(size(uex));
    E = zeros(size(uex));
    sigma = zeros(size(uex));
    sigma2 = zeros(size(uex));
    K = zeros(size(uex));
    KRBEB = zeros(size(uex));
    ni_sum = zeros(1,length(Tspace));
    ne_sum = zeros(1,length(Tspace));
    nn_sum = zeros(1,length(Tspace));
    Ni_sum = zeros(1,length(Tspace));
    Ne_sum = zeros(1,length(Tspace));
    Nn_sum = zeros(1,length(Tspace));
    R_theory = zeros(1,length(Tspace));
    dNidt_theory = zeros(1,length(Tspace));
    R_theory_RBEB = zeros(1,length(Tspace));
    dNidt_theory_RBEB = zeros(1,length(Tspace));
    Ue_mean = zeros(1,length(Tspace));
    Uez_mean = zeros(1,length(Tspace));
    Uer_mean = zeros(1,length(Tspace));
    
    me =  9.10938356E-31;
    qe =  1.6021766208e-19;

    for i = 1:length(Tspace)
        Ue(:,:,i) = sqrt(uex(:,:,i).^2 + uey(:,:,i).^2 + uez(:,:,i).^2);
        E(:,:,i) = 1/2*me.*Ue(:,:,i).^2./qe;
        sigma(:,:,i) = interp1(Ecombined,sigmacombined,E(:,:,i),'spline');
        sigma2(:,:,i) = interp1(ERBEB,sigmaRBEB,E(:,:,i),'spline');
        K(:,:,i) = sigma(:,:,i).*Ue(:,:,i);
        KRBEB(:,:,i) = sigma2(:,:,i).*Ue(:,:,i);
        R_theory(i) = sum(sum(nn(:,:,i).*ne(:,:,i).*K(:,:,i),1),2);
        R_theory_RBEB(i) = sum(sum(nn(:,:,i).*ne(:,:,i).*KRBEB(:,:,i),1),2);
        ni_sum(i) = sum(sum(ni(:,:,i),1),2);
        ne_sum(i) = sum(sum(ne(:,:,i),1),2);
        nn_sum(i) = sum(sum(nn(:,:,i),1),2);
        Ni_sum(i) = sum(sum(ni(:,:,i).*nodeVolume,1),2);
        Ne_sum(i) = sum(sum(ne(:,:,i).*nodeVolume,1),2);
        Nn_sum(i) = sum(sum(nn(:,:,i).*nodeVolume,1),2);
        dNidt_theory(i) = sum(sum(nn(:,:,i).*ne(:,:,i).*K(:,:,i).*nodeVolume,1),2);
        dNidt_theory_RBEB(i) = sum(sum(nn(:,:,i).*ne(:,:,i).*KRBEB(:,:,i).*nodeVolume,1),2);
        Ue_mean(i) = mean(mean(Ue(:,:,i),1),2);
        Uez_mean(i) = abs(mean(mean(uez(:,:,i),1),2));
        Uer_mean(i) = mean(mean(sqrt(uex(:,:,i).^2 + uey(:,:,i).^2),1),2);
    end

    fig = figure(1);
    clf(fig);
    plot(Tspace, Ne_sum)
    hold on
    plot(Tspace, Ni_sum)
    legend("Electrons","Ions")
    xlabel("Time [s]")
    ylabel("number")
    
    grid on
    saveFigureToFolder(simNames(k), "figure1", "png")
    
    figure(2)
    clf;
    plot(Tspace, Ni_sum)
    hold on
    plot(Tspace, Tspace.*dNidt_theory+Ni_sum(1))
    hold on
    plot(Tspace, Tspace.*dNidt_theory_RBEB+Ni_sum(1))
    legend("Actual = $\Sigma (n_iV_{node})$","Theoretical","Theoretical_RBEB","Interpreter","latex")
    title("Sum of number of ions")
    xlabel("Time [s]")
    ylabel("number")
    
    grid on
    saveFigureToFolder(simNames(k), "figure2", "png")
    y1 = Tspace.*dNidt_theory+Ni_sum(1);
    y2 = Tspace.*dNidt_theory_RBEB+Ni_sum(1);
    err1 = (Ni_sum(end) - y1(end))/y1(end);
    err2 = (Ni_sum(end) - y2(end))/y2(end);
    string = strcat(simNames(k),"_errorwithExp: ",num2str(err1));
    disp(string); % To display the result
    string = strcat(simNames(k),"_errorwithRBEB: ",num2str(err2));
    disp(string); % To display the result

    figure(3)
    clf;
    plot(Tspace, Ni_sum - Ni_sum(1))
    hold on
    plot(Tspace, Ne_sum - Ne_sum(1))
    hold on
    plot(Tspace, -(Nn_sum - Nn_sum(1)))
    title("Number balance")
    legend("$\Delta N_i$","$\Delta N_e$","$-\Delta N_n$","Interpreter","latex")
    
    grid on
    saveFigureToFolder(simNames(k), "figure3", "png")
    
    figure(4)
    clf;
    plot(Tspace,Ue_mean)
    hold on
    plot(Tspace,Uez_mean)
    hold on
    plot(Tspace,Uer_mean)
    legend("Ue","Uez","Uer","location","northeast")
    title("Mean electron velocity")
    
    grid on
    saveFigureToFolder(simNames(k), "figure4", "png")
    
    figure(5)
    clf;
    plot(Tspace, energy/qe)
    ylabel("Total energy [eV]")
    xlabel("time [s]")
    grid on
    saveFigureToFolder(simNames(k), "figure5", "png")
end

