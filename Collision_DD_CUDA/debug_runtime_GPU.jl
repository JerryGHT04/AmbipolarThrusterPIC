include("ParticlePusherJuliaTranslation.jl")
include("Poisson1DJuliaTranslation.jl")
include("StaticMagneticFieldJuliaTranslation.jl")
include("write_to_file.jl")
include("classdefinition.jl")
include("GPUkernel.jl")
using TimerOutputs
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
IREB_SuperSize = Ne_dot*timeStep;
numSuperParticle = Ne_dot*timeStep/electronSuperSize
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



# Create a TimerOutput object
const to = TimerOutput()

while time < maxTime
    # Global variables
    global time
    global next_Electron_Spawn_Time
    global next_Neutral_Spawn_Time
    global E2

    # Start timing the sections of your code

    @timeit to "Particle Count" begin
        particleCountRectangular_jl(MyParticles, MyDomain)
    end

    # Collision and Field calculations
    @timeit to "Poisson Equation" begin
        solvePoissonEquation_GPU_jl(MyDomain)
    end

    @timeit to "Xenon Ionization" begin
        fixedTemperatureXenonIonization_jl(MyParticles, MyDomain, Telectron, Tion, timeStep)
    end

    @timeit to "Reduce Neutral Mass" begin
        reduceNeutralMassFromArray_jl(MyParticles, MyDomain)
    end

    @timeit to "Neutral Collisions" begin
        neutralCollisions_jl(MyParticles, 2.1802e-25, 216.0e-12, timeStep)
    end

    @timeit to "Electric Field" begin
        electricFieldRectangular_jl(MyParticles, MyDomain)
    end

    @timeit to "Set Magnetic Field" begin
        setMagneticField_jl(MyMagneticField, MyParticles)
    end

    @timeit to "Push Particles (Boris)" begin
        pushParticlesBoris_jl(MyParticles, timeStep)
    end

    # Boundary conditions
    @timeit to "Boundary Conditions" begin
        boundaryConditionsRectangularNoWallLoss_jl(MyParticles, MyDomain)  # Lost at x = dX, otherwise periodic
    end

    @timeit to "Write to File" begin
        print(write_to_file())
    end

    # Check if it's time to spawn new electrons
    @timeit to "Electron Spawning" begin
        if time > next_Electron_Spawn_Time
            print("Electrons Spawned!")
            @timeit to "createElectron" begin
            # Create electrons
                createParticles_jl(MyParticles, numSuperParticle, me, Telectron, -qe, electronSuperSize, 0.0, 0.0, dY/2-rb, dY/2+rb, dZ/2-rb, dZ/2+rb, 0.0, UeX)
            end
            # Update the next electron spawn time
            next_Electron_Spawn_Time += Electron_SpawnTime
        end
    end

    # Check if it's time to spawn new neutrals
    @timeit to "Neutral Spawning" begin
        if time > next_Neutral_Spawn_Time
            print("Neutrals Spawned!")
            @timeit to "createNeutral" begin
            # Create neutrals
                createParticles_jl(MyParticles, 1, mi, 300.0, 0.0, neutralSuperSize, X2, X3, 0.0, dY, 0.0, dZ, 216.0e-12, 0.0)
            end
            # Update the next neutral spawn time
            next_Neutral_Spawn_Time += Neutral_SpawnTime
        end
    end

    @timeit to "Anode Mass Flow" begin
        anodeMassFlow_jl(MyDomain, mass_flow_rate, timeStep)
    end

    # Advance simulation time
    @timeit to "Advance Time" begin
        time += timeStep
    end

    # Print the timing results at the end of the simulation
    show(to)
end
# After the loop finishes, save the final timing results to a file

open("final_runtime_results.txt", "w") do file
    show(file, to)
end  # This 'end' closes the 'do' block

print(now() - tstart)