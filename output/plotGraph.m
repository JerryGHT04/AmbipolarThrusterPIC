dX = 0.5;
dY = 0.1;
dZ = 0.1;
N_dot = 1e14;
ionefficiency = 0.22;
X1 = 1;
X2 = 400;
X3 = 600;
X4 = 1000;
numCell = 2000;
tf = 5e-5;
me =  9.10938356E-31;
mi = 2.1802e-25;

Xspace = linspace(0,dX,numCell);
Tspace = linspace(0,tf,length(tspace));
figure(1)
clf;
h = surf(Xspace,Tspace,uix);
colormap('parula');
set(h,'LineStyle','none')
xlabel("X [m]")
ylabel("time")
zlabel("uix*ni")
view(0, 90)


ne1 = ne(:,X2);
ve1 = uex(:,X2);
ni1 = ni(:,X2);
vi1 = uix(:,X2);

ne2 = ne(:,X3);
ve2 = uex(:,X3);
ni2 = ni(:,X3);
vi2 = uix(:,X3);

%ideal gas equation to find pressure
Ti = 300;
Te = 10000;
k = 1.380649e-23;
pe1 = ne1*k*Te;
pe2 = ne2*k*Te;
pi1 = ni1*k*Ti;
pi2 = ne2*k*Ti;

Ft = dY*dZ*(me*(ne2.*ve2.^2 - ne1.*ve1.^2) + mi*(ni2.*vi2.^2 - ni1.*vi1.^2) + (pe2 + pi2- pe1 - pi1));
figure(2)
clf;
plot(Tspace,Ft*1000)
xlabel("t [s]")
ylabel("Thrust [mN]")
xlim([0,Tspace(end)])


figure(3)
clf;
netotal = sum(ne(:,X2:X3),2);
nitotal = sum(ni(:,X2:X3),2);
plot(Tspace,netotal)
hold on
plot(Tspace,nitotal)
legend("ne","ni")
target = uex;
dnidt =  polyfit(Tspace(end-2000:end), nitotal(end-2000:end),1);
dnidt = dnidt(1);

video_file_name = 'uex_animation.mp4';
video = VideoWriter(video_file_name, 'MPEG-4');
open(video);
figure(4)
for i = 1:100:length(tspace)
    plot(Xspace,target(i,:))
    ylim([-max(max(abs(target))), max(max(abs(target)))])
    title(["t = ", Tspace(i)])
    ylabel("UeX")
    xlabel("X")
    hold on  % Hold the plot to overlay subsequent plots
    frame = getframe(gcf); % Capture the current frame
    writeVideo(video, frame); % Write the frame to the video
    clf; % Clear the figure for the next plot
end
close(video);
