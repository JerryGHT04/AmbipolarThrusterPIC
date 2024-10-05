%B = [35755.82, 5509.33, 5161.43, 4835.57, 1170.37, 1024.78, 961.25, 708.13, 694.90, 229.39, 175.58, 162.8, 73.78, 71.67, 27.49, 13.40, 12.1298]
B = [3.925e4, 9.33e3, 8.617e3, 7.224e3, 3.296e3, 3.001e3, 2.548e3, 1.987e3, 1.394e3, 8.527e2, 7.26e2, 5.725e2, 3.583e2, 1.708e2, 9.805e1, 5.969e1, 1.197e1]
U = [42281.58, 9173.91, 9135.67, 8324.23, 2722.32, 2642.38, 2453.45, 2311.18, 2261.96, 757.48, 690.54, 643.34, 497.77, 485.43, 122.59, 88.90, 79.42]
N = [2, 2, 2, 4, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 2, 2, 4]
me = 9.10938356E-31
c = 3e8
e = 1.6021766208e-19
a0 = 5.2917721054482e-11
alpha = 1/137
B = B*e
U = U*e

T = 1e4*e

t = T./B
u = U./B

tp = T/me/c^2
bp = B/me/c^2
up = U/me/c^2

Bt2 = 1 - 1./(1+tp).^2
Bb2 = 1 - 1./(1+bp).^2
Bu2 = 1 - 1./(1+up).^2

A = 2*pi*a0^2*alpha^4.*N./(Bt2 + Bu2 + Bb2)./bp
B = 1/2*(log(Bt2./(1-Bt2)) - Bt2 - log(2*bp)).*(1-1./t.^2)
C = 1- 1./t - log(t)./(t+1).*(1+2.*tp)./(1+tp/2).^2 + bp.^2./(1 + tp/2).^2.*(t-1)/2

RBEB = A.*(B + C)
%plot(linspace(0,1,length(B)),RBEB)
RBEBav = 1/2*(1 + (Bt2 + Bu2 + Bb2)./Bt2).*RBEB

sum(RBEBav)

function [result, resultR] = getSigma(T)
B = [35755.82, 5509.33, 5161.43, 4835.57, 1170.37, 1024.78, 961.25, 708.13, 694.90, 229.39, 175.58, 162.8, 73.78, 71.67, 27.49, 13.40, 12.1298];

U = [42281.58, 9173.91, 9135.67, 8324.23, 2722.32, 2642.38, 2453.45, 2311.18, 2261.96, 757.48, 690.54, 643.34, 497.77, 485.43, 122.59, 88.90, 79.42];
N = [2, 2, 2, 4, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 2, 2, 4];
me = 9.10938356E-31;
c = 3e8;
e = 1.6021766208e-19;
a0 = 5.2917721054482e-11;
alpha = 1/137;
B = B*e;
U = U*e;


t = T./B;
u = U./B;

tp = T/me/c^2;
bp = B/me/c^2;
up = U/me/c^2;

Bt2 = 1 - 1./(1+tp).^2;
Bb2 = 1 - 1./(1+bp).^2;
Bu2 = 1 - 1./(1+up).^2;

%A = 2*pi*a0^2*alpha^4./(Bt2 + Bu2 + Bb2)./bp;
%B = 1/2*(log(Bt2./(1-Bt2)) - Bt2 - log(2*bp));
%C = 1- 1./t - log(t)./(t+1).*(1+2.*tp)./(1+tp/2).^2 + bp.^2.*(t-1)./(1 + tp/2).^2 ./2;

A1 = 2*pi*a0^2*alpha^4./(Bt2 + Bu2 + Bb2)./bp;
A2 = 1/2*(log(Bt2./(1-Bt2)) - Bt2 - log(2*bp));
A3 = 1- 1./t - log(t)./(t+1).*(1+2.*tp)./(1+tp/2).^2 + bp.^2.*(t-1)./(1 + tp/2).^2 ./2;

S =  4*pi*a0^2.*(13.6*e./B).^2;
BEB = S./(t+(u+1)/5).*(log(t)./2.*(1-(1./t).^2) + 1 - 1./t - log(t)./(t+1));
BEBav = S./2.*(t.^(-1) + (t+u+1).^(-1)).*(log(t)/2 .*(1 - t.^(-2)) + 1 - t.^(-1) - log(t).*(t+1).^(-1));
RBEB = A1.*(A2.*(1-1./t.^2) + A3);
%plot(linspace(0,1,length(B)),RBEB)
RBEBav = 1/2*(1 + (Bt2 + Bu2 + Bb2)./Bt2).*RBEB;
%result = 3*RBEBav(17)
%occupation number * target orbital ionization cross section, summed for
%first 4 ionizations
result = 3*BEB(17) + 2*BEB(17) + BEB(17) + BEB(16)*2
resultR = 3*RBEBav(17) + 2*RBEBav(17) + RBEBav(17) + RBEBav(16)*2
%sigmak = flip(sum(RBEBav.*N));
%resultR = sigmak(1)*3 + sigmak(2)*3
%sigmak
%result = BEB(17)
end

function [w_bar,e_bar] = getw(T)
B = [35755.82, 5509.33, 5161.43, 4835.57, 1170.37, 1024.78, 961.25, 708.13, 694.90, 229.39, 175.58, 162.8, 73.78, 71.67, 27.49, 13.40, 12.1298];

U = [42281.58, 9173.91, 9135.67, 8324.23, 2722.32, 2642.38, 2453.45, 2311.18, 2261.96, 757.48, 690.54, 643.34, 497.77, 485.43, 122.59, 88.90, 79.42];
N = [2, 2, 2, 4, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 2, 2, 4];
me = 9.10938356E-31;
c = 3e8;
e = 1.6021766208e-19;
a0 = 5.2917721054482e-11;
alpha = 1/137;
B = B*e;
U = U*e;


t = T./B;
u = U./B;

tp = T/me/c^2;
bp = B/me/c^2;
up = U/me/c^2;

Bt2 = 1 - 1./(1+tp).^2;
Bb2 = 1 - 1./(1+bp).^2;
Bu2 = 1 - 1./(1+up).^2;

K1 = 2*pi*a0^2*alpha^4./(Bt2 + Bu2 + Bb2)./bp;
K2 = 1/2*(log(Bt2./(1-Bt2)) - Bt2 - log(2*bp));
K3 = 1- 1./t - log(t)./(t+1).*(1+2.*tp)./(1+tp/2).^2 + bp.^2.*(t-1)./(1 + tp/2).^2 ./2;
RBEB = K1.*(K2.*(1-1./t.^2) + K3);
RBEBav = 1/2*(1 + (Bt2 + Bu2 + Bb2)./Bt2).*RBEB;
sigma_bar = 3*RBEBav(17) + 2*RBEBav(17) + RBEBav(17) + 2*RBEBav(16);

sigma0 = 2*pi*a0^2*alpha^4./(Bt2 + Bu2 + Bb2)./bp;
A1 = (1+t).^(-1).*(1+2*tp).*(1+tp/2).^(-2);
A2 = (t-1)./2.*(bp.^2).*(1+tp./2).^(-2);
A3 = log(Bt2./(1-Bt2)) - Bt2 - log(2.*bp);

wk = sigma0./RBEBav.*(A3/2.*(t-1).^2.*(t.*(t+1)).^(-1) + 2*log((t+1)/2) - log(t) + A2.*(t-1)/4 - A1.*(log(t) - (t+1).*log((t+1)/2)));

w_bar = 3*wk(17)*B(17)*RBEBav(17)/sigma_bar + 2*wk(17)*B(17)*RBEBav(17)/sigma_bar + wk(17)*B(17)*RBEBav(17)/sigma_bar + 2*wk(16)*B(16)*RBEBav(16)/sigma_bar;
e_bar = 3*(wk(17)+1)*B(17)*RBEBav(17)/sigma_bar + 2*(wk(17)+1)*B(17)*RBEBav(17)/sigma_bar +(wk(17)+1)*B(17)*RBEBav(17)/sigma_bar + 2*(wk(16)+1)*B(16)*RBEBav(16)/sigma_bar;
end



%T = linspace(10,1e6,70000)*e;
T1 = linspace(13,110,300)*e;
T2 = linspace(110 ,1e4, 1500)*e;
T = [T1, T2(2:end)];
T = linspace(20,4e3,1000)
T = T*e;
sigma = [];
sigmaR = [];
w_bar = [];
t_bar = [];
for i = 1:length(T)
    [val1, val2]= getSigma(T(i));
    sigma(i) = val1;
    sigmaR(i) = val2;
    [val1, val2] = getw(T(i));
    w_bar(i) = val1;
    t_bar(i) = val2;
end



figure(1)
clf;
plot(T/e,sigma*1e20,'--')
hold on
plot(T/e,sigmaR*1e20)
ylim([0 max(sigmaR*1e20)*1.2] )
xlim([0 max(T/e)*1.2])
set(gca, 'XScale', 'log'); % Set x-axis to logarithmic scale
set(gca, 'fontsize', 18)
E = [140 160 180 200 225 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2500 3000 3500 4000];
sigma = [445.5,422.5,401.9,383.0,361.3,342.7,311.5,286.7,265.7,247.9,231.2,217.9,205.8,194.9,184.9,176.0,168.1,161.2,154.9,149.2,143.7,134.1,125.8,118.5,112.0,106.5,101.6,96.8,92.6,88.7,85.3,71.9,62.26,55.14,49.72];
sigma = sigma*1e6*1e-28;
hold on
scatter(E, sigma*1e20)

E2 = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0,130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0];
sigma2 = [1.15e-20, 2.42e-20, 3.81e-20, 4.17e-20, 4.17e-20,4.30e-20, 4.31e-20, 4.29e-20, 4.27e-20, 4.37e-20, 4.47e-20, 4.54e-20,4.57e-20, 4.59e-20, 4.55e-20, 4.48e-20, 4.42e-20, 4.31e-20, 4.26e-20, 4.21e-20, 4.13e-20, 4.06e-20, 3.99e-20, 3.97e-20, 3.92e-20,3.87e-20, 3.85e-20, 3.82e-20, 3.78e-20, 3.74e-20, 3.73e-20, 3.67e-20, 3.63e-20, 3.58e-20];

E31 = 12:1:40;
E32 = 45:5:200;
E3 = [E31 E32];
sigma31 = [0.1 0.31 0.63 1.0 1.35 1.65 1.93 2.17 2.42 2.67 2.91 3.12 3.3 3.46 3.59 3.73 3.89 4.04 4.13 4.23 4.33 4.43 4.51 4.59 4.62 4.67 4.72 4.74 4.8 4.82 4.84 4.86 4.89 4.93 4.96 4.98 4.95 4.95 4.92 4.88 4.84 4.78 4.77 4.69 4.63 4.59 4.55 4.5 4.46 4.44 4.38 4.33 4.3 4.26 4.24 4.19 4.16 4.12 4.08 4.05 4.03]
sigma32 = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.002 0.004 0.001 0.008 0.016 0.016 0.034 0.046 0.065 0.157 0.26 0.323 0.363 0.377 0.389 0.4 0.411 0.436 0.456 0.482 0.51 0.531 0.545 0.552 0.553 0.552 0.542 0.534 0.525 0.504 0.494 0.477 0.467 0.458 0.446 0.434 0.426 0.413 0.406 0.394 0.391]
sigma33 = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.003 0.004 0.009 0.021 0.033 0.047 0.065 0.087 0.109 0.129 0.147 0.164 0.174 0.178 0.182 0.186 0.186 0.180 0.179 0.175 0.170 0.169 0.166 0.162 0.160 0.155 0.156]
sigma3 = sigma33+sigma32+sigma31;
hold on
scatter(E2, sigma2*1e20,'x')
hold on
scatter(E3,sigma3,'.')
legend("BEB","RBEB","Experiment K. Stephan","Experiment A. A. Sorokin",'R.C. Wetzel')
%title("Total ionization cross section")
xlabel("Incident electron energy (eV)", 'fontsize', 18)
ylabel("Ionization cross section (10^{-20} m^2)", 'fontsize', 18)
print -dpng Fig_IonizationCrossSection

figure(2)
clf;
plot(T/e, w_bar./e)
hold on
plot(T/e, t_bar./e,'--')
set(gca, 'fontsize', 18)
legend({"$\overline{W}$", "$\overline{\delta E}$"}, 'Interpreter', 'latex', 'Location', 'southeast')
set(gca, 'XScale', 'log');
xlabel("Incident electron energy [eV]", 'fontsize', 18)
ylabel("Energy [eV]", 'fontsize', 18)
ylim([0 max(t_bar./e)*1.2])
xlim([0 max(T/e)*1.2])
print -dpng Fig_EnergyTransfer

writematrix(T/e, 'T.csv');
writematrix(sigmaR,'sigma.csv')
writematrix(w_bar,'W.csv')
writematrix(t_bar,'E.csv')
