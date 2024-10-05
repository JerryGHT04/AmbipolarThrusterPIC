%vary grid res, time step and boundary condition
listGrid = ["conserveN40.mat","conserveN70.mat","conserveN100.mat"]
figure(1)
clf;
for i = 1:length(listGrid)
    clearvars -except listGrid i
    load(listGrid(i))
    plot(Tspace, energy)
    hold on
end
legend("Nr=Nz= 40", "Nr=Nz= 70", "Nr=Nz= 100",'Location', 'northwest')
title("timeStep = 1e-12, DirichletBC")
xlabel("Time (s)", 'fontsize', 18)
ylabel("Total energy", 'fontsize', 18)
print -dpng Fig_1

listdt = ["conserveN100.mat","conserve5e-13.mat"]
figure(2)
clf;
for i = 1:length(listdt)
    clearvars -except listdt i
    load(listdt(i))
    plot(Tspace, energy)
    hold on
end
legend("step=1e-12", "step=5e-13",'Location', 'northwest')
title("DirichletBC, N=100")
xlabel("Time (s)", 'fontsize', 18)
ylabel("Total energy", 'fontsize', 18)
print -dpng Fig_2

listBC = ["conserveN100.mat","conserveN100Neumann"]
figure(3)
clf;
for i = 1:length(listdt)
    clearvars -except listBC i
    load(listBC(i))
    plot(Tspace, energy)
    hold on
end
legend("Dirichlet", "Neumann",'Location', 'northwest')
title("timeStep =1e-12 ,Nr=Nz=100 , varying Z-BC")
xlabel("Time (s)", 'fontsize', 18)
ylabel("Total energy", 'fontsize', 18)
print -dpng Fig_3