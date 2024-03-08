include("ParticlePusherJuliaTranslation.jl")
include("Poisson1DJuliaTranslation.jl")
include("StaticMagneticFieldJuliaTranslation.jl")
include("write_to_file.jl")
include("classdefinition.jl")

using Dates#for timer

tstart = now()

#= take parameters: 
1output path
2maxTime:1 mius
3timestep: 1ns
4number of cells
5number of particles
6initial electron density
7initial neutral density
8ionisation ratio
9electron source rate per spawn time step
10electron temperature
11ion temperature
12dY of volumn 0.1m
13dZ of volumn 0.1m
14dX of chamber 0.1m
15dX of total length(beam+chamber), set to 0.5(5 times chamber length)
16Vx = 0 anode voltage
17B0z = 0
18UeX =  the horizontal velocity added to electron
19Spawn time step
=#

#read file
file = open("Parameters.txt")
parameters = []
for line in eachline(file)
    content = line
    push!(parameters, content)
end
close(file)

root_directory = @__DIR__
output_folder = parameters[1]
outputPath = joinpath(root_directory, output_folder)
maxTime = parse(Float64, parameters[2])
timeStep =  parse(Float64, parameters[3])
numCells = parse(Int64, parameters[4])
numParticle = parse(Float64, parameters[5]) #number of superparticles
edensity = parse(Float64, parameters[6])
ndensity = parse(Float64, parameters[7])
IonEfficiency = parse(Float64, parameters[8])
Re = parse(Float64, parameters[9])
Telectron = parse(Float64, parameters[10]) #initial electron temperature
Tion = parse(Float64, parameters[11])
dY = parse(Float64, parameters[12])
dZ = parse(Float64, parameters[13])
dXchamber = parse(Float64, parameters[14]) #ionisation chamber length
dXbeam = parse(Float64, parameters[15]) #total simulation length
VAnode = parse(Float64, parameters[16]) #anode voltage
B0z = parse(Float64, parameters[17])
UeX = parse(Float64, parameters[18])
Ne_dot = parse(Float64, parameters[19])
SpawnTime = parse(Float64, parameters[20])

#initialization
writeFrequency = 1e-9;
me =  9.10938356E-31;
qe =  1.6021766208e-19;
mi = 2.1802e-25#Xenon atom
N = numParticle;
Ne = edensity *dXchamber*dY*dZ;
Ni = Ne;#same number of ions and electrons
Nn = ndensity *dXchamber*dY*dZ;

electronSuperSize = Ne / N;
ionSuperSize = Ni / N;
neutralSuperSize = Nn/ N;

electronSpawnSuperSize = Ne_dot*SpawnTime/Re;
ionSpawnSuperSize = electronSpawnSuperSize ;

#initialize particles
MyParticles =  Particle()
#push electrons to the ionisation chamber
createParticles_jl(MyParticles, N, me, Telectron, -qe, electronSuperSize, dXbeam, dXbeam+dXchamber, 0.0, dY, 0.0, dZ, 0.0,0.0)
#push ions
createParticles_jl(MyParticles, N, mi, Tion, qe, ionSuperSize, dXbeam, dXbeam+dXchamber, 0.0, dY, 0.0, dZ, 216.0e-12,0.0)
#createParticles_jl(MyParticles, Nn, mi, 300, 0, NeutralSuperSize, 0.0, dXchamber, 0.0, dY, 0.0, dZ, 216.0e-12)


#create volume for simulation
MyDomain = RectangularDomain()
createRectangularDomain_jl(MyDomain, dXbeam*2+dXchamber, dY, dZ, VAnode, 0.0, numCells)
#initialize Magnetic Field
MyMagneticField = MagneticField()
createRectangularRegion_jl(MyMagneticField, 0.0, dXbeam*2+dXchamber, 0.0, dY, 0.0, dZ, 0.0, 0.0, B0z)

timer = now()
step_count = 0
timeStamp = 0.0
currentTime = 0.0
writeTimer = 0.0

t0 = 0;

for t in 0:timeStep:maxTime
    global t0;
    global SpawnTime;
    particleCountRectangular_jl(MyParticles, MyDomain)
    solvePoissonEquation_jl(MyDomain)
    electricFieldRectangular_jl(MyParticles,MyDomain)
    setMagneticField_jl(MyMagneticField,MyParticles)
    pushParticlesBoris_jl(MyParticles,timeStep)
    #boundary condition
    Nilost, Nelost = boundaryConditionsRectangularNoWallLoss_jl(MyParticles,MyDomain)#lost at x = dX, otherwise periodic
    print("NiLost:")
    print(Nilost)
    print("")
    print("Nelost:")
    print(Nelost)
    println(write_to_file())
    if t-t0 > SpawnTime
        println("ParticleSpawned!")
        createParticles_jl(MyParticles, Re, me, Telectron, -qe, electronSpawnSuperSize,0.0,(dXbeam*2+dXchamber)/numCells, 0.0, dY, 0.0, dZ, 0.0,UeX)#create electrons at x = 0
        createParticles_jl(MyParticles, Re, mi, Tion, qe, ionSpawnSuperSize, dXbeam, dXbeam+dXchamber, 0.0, dY, 0.0, dZ, 216.0e-12,0.0)#create ions in the chamber
        t0 = t;
    end
end

print(now() - tstart)