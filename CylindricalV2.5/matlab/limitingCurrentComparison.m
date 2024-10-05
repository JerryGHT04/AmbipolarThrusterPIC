%Compares limiting current
R = 0.005;
r = 0.001;
me = 9.109383700000000e-31;
e = 1.602176630000000e-19;
c = 3e8;

Ue = linspace(0,2.8e7,20000);

gamma = @(U) (1-U.^2/c^2).^(-1/2);
G = @(R,r) 1+2*log(R/r);

eps0 = 8.85418782e-12;
%Vacuum
IBR = @(U) me*c^3*4*pi*eps0/e/(1+2*log(R/r))*( gamma(U).^(2/3)-1).^(3/2);
IGP = @(U) IBR(U).*gamma(U).^(2/3).*( ((gamma(U).^(2/3) + G(R,r)).^2 - gamma(U).^(2/3)).^(1/2)- G(R,r)).^(-1);

%Compensated
IO = @(U,fe) 4*pi*eps0* U/c.*(gamma(U)-1).*me*c^3/e./G(R,r)./(1-fe);
IBRCr = @(U) 4*pi*eps0* me.*U.^3/(4*e)*2/G(R,r).*gamma(U).^3;

figure(1)
clf;
set(gca, 'fontsize', 18)

plot((gamma(Ue)-1)*me*c^2/e,IBR(Ue), 'LineWidth', 2)
hold on
plot((gamma(Ue)-1)*me*c^2/e,IGP(Ue),'--', 'LineWidth', 2)

legend("BR [12]","GP [15]", 'Interpreter', 'latex', 'Location', 'northwest')

ylabel("Limiting current [A]", 'Interpreter', 'latex', 'fontsize', 18)
xlabel('Electron beam energy [eV]', 'Interpreter', 'latex', 'fontsize', 18)
xlim([(gamma(0)-1)*me*c^2/e (gamma(Ue(end))-1)*me*c^2/e])
title(strcat("Vacuum propagation, R/r = ",num2str(R/r)))
grid on

print -dpng Fig_LimitingCurrentVacuum


figure(2)
clf;
set(gca, 'fontsize', 18)

fe = [0 0.25 0.5];

for i = 1:length(fe)
    k = fe(i); 
    y = IO(Ue,k);  
    x = (gamma(Ue)-1)*me*c^2/e;
    % Plot the line
    plot(x, y, 'black', 'DisplayName', ['fe = ' num2str(k)], 'LineWidth', 2);
    hold on
    positionCoeff = 0.98;
    % Find a position on the line to place the label (e.g., near the end)
    x_label_pos = x(floor(length(x)*positionCoeff));  % Use the last point in x for label
    y_label_pos = y(floor(length(x)*positionCoeff));  % Corresponding y-value for label
    
    % Add label using the text function
    text(x_label_pos, y_label_pos, [ num2str(k)], 'fontsize', 9, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

plot(Ue, IBRCr(Ue),'--')
ylabel("Limiting current [A]", 'Interpreter', 'latex', 'fontsize', 18)
xlabel('Electron beam energy [eV]', 'Interpreter', 'latex', 'fontsize', 18)
xlim([(gamma(0)-1)*me*c^2/e (gamma(Ue(end))-1)*me*c^2/e])
title(strcat("Compensated beam, R/r = ",num2str(R/r)))
legend("Olson [3]", "", "", "", "IBRcr",'Location', 'northwest')
grid on
print -dpng Fig_LimitingCurrentCompensated

figure(3)
clf;
fe_space = linspace(0,1,1000);
E = [100, 300, 500];
Ue = sqrt(2*E.*e/me);
for i = 1:length(Ue)
    plot(fe_space,IO(Ue(i),fe_space),"LineWidth",2)
    hold on
end
legend("100eV","300eV","500eV")

set(gca, 'YScale', 'log'); % Set x-axis to logarithmic scale