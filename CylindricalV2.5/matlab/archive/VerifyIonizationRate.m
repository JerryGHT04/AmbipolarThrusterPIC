%load("IonizationNew1e-12.mat")
simNames = ["100eV4A","300eV4A","500eV4A"];
load("experimentsigma.mat")
load("RBEB.mat")

for k = 1:3
    
    loadData(strcat("params_",simNames(k), ".txt"), strcat(simNames(k),".mat"))
    %load(strcat(simNames(k),".mat"))
    qe = 1.6021766208e-19;
    
    ERBEB = T/qe;
    sigmaRBEB = sigmaRBEBav;
    
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

    figure(1)
    clf;
    plot(Tspace, Ne_sum)
    hold on
    plot(Tspace, Ni_sum)
    legend("Electrons","Ions")
    xlabel("Time [s]")
    ylabel("number")
    
    grid on
    string = strcat("verifyIon_",simNames(k),"_1");
    print('-dpng', string);
    
    figure(2)
    clf;
    plot(Tspace, Ni_sum)
    hold on
    plot(Tspace, Tspace.*dNidt_theory+Ni_sum(1))
    hold on
    plot(Tspace, Tspace.*dNidt_theory_RBEB+Ni_sum(1))
    legend("Actual = $\Sigma (n_iV_{node})$","Theoretical","Theoretical_RBEB","Interpreter","latex","Location","northwest")
    title("Sum of number of ions")
    xlabel("Time [s]")
    ylabel("number")
    
    grid on
    string = strcat("verifyIon_",simNames(k),"_2");
    print('-dpng', string);
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
    legend("$\Delta N_i$","$\Delta N_e$","$-\Delta N_n$","Interpreter","latex","Location","northwest")
    
    grid on
    string = strcat("verifyIon_",simNames(k),"_3");
    print('-dpng', string);
    
    figure(4)
    clf;
    plot(Tspace,Ue_mean)
    hold on
    plot(Tspace,Uez_mean)
    hold on
    plot(Tspace,Uer_mean)
    legend("Ue","Uez","Uer")
    title("Mean electron velocity")
    
    grid on
    string = strcat("verifyIon_",simNames(k),"_4");
    print('-dpng', string);
    
    figure(5)
    clf;
    plot(Tspace, energy/qe)
    ylabel("Total energy [eV]")
    xlabel("time [s]")
    grid on
    string = strcat("verifyIon_",simNames(k),"_5");

    print('-dpng', string);
    
end