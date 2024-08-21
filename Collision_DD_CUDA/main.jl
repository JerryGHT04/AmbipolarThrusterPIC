include("ParticlePusherJuliaTranslation.jl")
include("Poisson1DJuliaTranslation.jl")
include("StaticMagneticFieldJuliaTranslation.jl")
include("write_to_file.jl")
include("classdefinition.jl")
include("GPUkernel.jl")
using Dates#for timer
using Printf
using CUDA

tstart = now()

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
mass_flow_rate = parse(Float64, parameters[9])
Telectron = parse(Float64, parameters[10]) #initial electron temperature
Tion = parse(Float64, parameters[11])
dY = parse(Float64, parameters[12])
dZ = parse(Float64, parameters[13])
X2 = parse(Float64, parameters[14]) #ionisation chamber length
X3 = parse(Float64, parameters[15]) #total simulation length
X4 = parse(Float64, parameters[16])
VAnode = parse(Float64, parameters[17]) #anode voltage
B0z = parse(Float64, parameters[18])
UeX = parse(Float64, parameters[19])
Ne_dot = parse(Float64, parameters[20])

#initialization
writeFrequency = 1e-10;
me =  9.10938356E-31;
qe =  1.6021766208e-19;
mi = 2.1802e-25#Xenon atom
N = numParticle;
Ne = edensity *(X3 - X2)*dY*dZ;
Ni = Ne;#same number of ions and electrons
Nn = ndensity *(X3 - X2)*dY*dZ;

electronSuperSize = Ne / N;
ionSuperSize = Ni / N;
neutralSuperSize = Nn/ N;

#electronSource_SuperSize = Ne_dot*SpawnTime/numCells;
#neutralSource_SuperSize = mass_flow_rate/mi*SpawnTime/numCells;
#ionSpawnSuperSize = Nn_dot*SpawnTime;

Electron_SpawnTime = timeStep
IREB_SuperSize = Ne_dot*timeStep/1000;
#numSuperParticle = Ne_dot*timeStep/electronSuperSize
Neutral_SpawnTime = neutralSuperSize/(mass_flow_rate/mi)

#initialize particles
MyParticles =  Particle()
#push electrons to the ionisation chamber
createParticles_jl(MyParticles, N, me, Telectron, -qe, electronSuperSize, X2, X3, 0.0, dY, 0.0, dZ, 0.0,0.0)
#push ions
createParticles_jl(MyParticles, N, mi, Tion, qe, ionSuperSize, X2, X3, 0.0, dY, 0.0, dZ, 216.0e-12,0.0)
#push neutrals
createParticles_jl(MyParticles, N, mi, 300.0, 0.0, neutralSuperSize, X2, X3, 0.0, dY, 0.0, dZ, 216.0e-12,0.0)


#create volume for simulation
MyDomain = RectangularDomain()
createRectangularDomain_jl(MyDomain, X4, dY, dZ, VAnode, 0.0, numCells)#Gives A as Dirichlet on LHS, Neumann on RHS

#initialize Magnetic Field
MyMagneticField = MagneticField()
createRectangularRegion_jl(MyMagneticField, 0.0, X4, 0.0, dY, 0.0, dZ, 0.0, 0.0, B0z)

timer = now()
step_count = 0
timeStamp = 0.0
currentTime = 0.0
writeTimer = 0.0

t0 = 0;
rb = 0.00025;
E2 = 0.0;
time = 0.0;
#next_Spawn_Time = SpawnTime;
next_Electron_Spawn_Time = Electron_SpawnTime
next_Neutral_Spawn_Time = Neutral_SpawnTime

while time < maxTime
    global time
    global next_Electron_Spawn_Time
    global next_Neutral_Spawn_Time
    global E2;
    particleCountRectangular_jl(MyParticles, MyDomain)

    #Collision starts here
    solvePoissonEquation_GPU_jl(MyDomain)
    fixedTemperatureXenonIonization_jl(MyParticles,MyDomain,Telectron,Tion,timeStep)
    reduceNeutralMassFromArray_jl(MyParticles,MyDomain)
    neutralCollisions_jl(MyParticles,2.1802e-25,216.0e-12,timeStep)

    electricFieldRectangular_jl(MyParticles,MyDomain)
    setMagneticField_jl(MyMagneticField,MyParticles)
    pushParticlesBoris_jl(MyParticles,timeStep)

    #boundary condition
    boundaryConditionsRectangularNoWallLoss_jl(MyParticles,MyDomain)#lost at x = dX, otherwise periodic
    println(write_to_file())

    # Check if it's time to spawn new electrons
    if time > next_Electron_Spawn_Time
        print("Electrons Spawned!")
        
        # Create electrons
        createParticles_jl(MyParticles, 1000, me, Telectron, -qe, IREB_SuperSize, 0.0, 0.0, dY/2-rb, dY/2+rb, dZ/2-rb, dZ/2+rb, 0.0, UeX)
        
        # Update the next electron spawn time
        next_Electron_Spawn_Time += Electron_SpawnTime
    end

    # Check if it's time to spawn new neutrals
    if time > next_Neutral_Spawn_Time
        print("Neutrals Spawned!")
        
        # Create neutrals
        createParticles_jl(MyParticles, 1, mi, 300.0, 0.0, neutralSuperSize, X2, X3, 0.0, dY, 0.0, dZ, 216.0e-12, 0.0)
        
        # Update the next neutral spawn time
        next_Neutral_Spawn_Time += Neutral_SpawnTime
    end

    anodeMassFlow_jl(MyDomain, mass_flow_rate, timeStep)
    # Advance simulation time
    time += timeStep
end

print(now() - tstart)