%Compare ionization change of particle with theoretical ionization rate
ne = 5e19;
nn = 5e20;

Te = 116050;

kB = 1.380649e-23;
me = 9.109383700000000e-31;
e = 1.602176630000000e-19;
c = 3e8;

Ee = 3/2*kB*Te;
load experimentsigma.mat
sigma2 = sigmacombined;
T2 = Ecombined;
gammafromE = @(Ee) Ee ./ (me*c^2) + 1;
UefromE = @(Ee) c.*sqrt(1-gammafromE(Ee).^(-2));
%rate coefficient K = mean(sigma * vel) over all energy
sigma_e2 = interp1(T2, sigma2, Ee/e);

Rion = nn*ne*sigma_e2*UefromE(Ee);

figure(1)
clf;
%change of number

plot(Tspace*1e9, Ni - Ni(1))
hold on
plot(Tspace*1e9, Ne - Ne(1),'--')
hold on

hold on
plot(Tspace*1e9, -Nn + Nn(1))
hold on
plot(Tspace*1e9, Rion.*Tspace*pi*R^2*L,'LineWidth',2)


ylabel("Total change in number of particles")
xlabel("time [ns]")
legend("$\Delta Ni$","$\Delta Ne$", "$-\Delta Nn$","$R_{collision theory}$", "interpreter", "latex", "Location","northwest")


%electron energy
figure(2)
clf;
plot(Tspace*1e9,Ee/qe)
ylabel("Average Electron energy [eV]")
xlabel("time [ns]")


figure(3)
clf;
plot(Tspace*1e9, energy/qe)
ylabel("Total energy [eV]")
xlabel("time [ns]")