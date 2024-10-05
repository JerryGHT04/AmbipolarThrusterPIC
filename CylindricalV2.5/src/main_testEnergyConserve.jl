#2D3C PIC in axisymmetric cylindrical coordinate, using fully GPU computation with CUDA
#created by Haitian Gao
#last updateL 3/9/2024

include("classDefinition.jl")
include("library.jl")
include("auxlibrary.jl")
include("writeToFile.jl")
include("energy2D.jl")
#include("ionization.jl")
include("MCCIonization_Relativistic.jl")
include("neutralCollisions.jl")
include("save_restoreFiles.jl")
#include("IonizationNEW.jl")
using CUDA
using Dates#for timer
using Printf
using BenchmarkTools
using DataFrames
using CSV
using SparseArrays

using DelimitedFiles
# Read precalculated data
array = readdlm("src/sigma.csv", ',')
σList = CuArray(array)
σLength = length(σList)
# energy lost by incident electron
array = readdlm("src/E.csv", ',')
EList = CuArray(array)

#Incident electron energy in eV
array = readdlm("src/T.csv", ',')
TList = CuArray(array)
stepT = TList[2] - TList[1]

array = readdlm("src/W.csv", ',')
WList = CuArray(array)
array = nothing

#Ionization power in eV
IP = CuArray([11.97 23.54 35.11 46.68])
#---------------------------------------------------------------------------#
#read file
paramsName = "params_testIonization.txt"
file = open(joinpath(@__DIR__,"..", paramsName),"r")
parameters = []
for line in eachline(file)
    content = line
    push!(parameters, content)
end
close(file)

root_directory = @__DIR__
output_folder = parameters[1]
outputPath = joinpath(root_directory, output_folder)
#delete_all_files(outputPath)
maxTime = parse(Float64, parameters[2])
timeStep =  parse(Float64, parameters[3])
Nr = parse(Int, parameters[4])
Nz = parse(Int, parameters[5])

# Plasma properties
numParticle = parse(Int64, parameters[6]) #number of superparticles
edensity = parse(Float64, parameters[7])
ndensity = parse(Float64, parameters[8])
Telectron = parse(Float64, parameters[9]) #initial electron temperature
Tion = parse(Float64, parameters[10])

#Volume size
dR = parse(Float64, parameters[11])
dZ = parse(Float64, parameters[12])

# Beam properties
BeamEnergy = parse(Float64, parameters[13]) #in eV
BeamCurrent = parse(Float64, parameters[14])
BeamSuperParticle = parse(Int, parameters[15]) #Num of super particles per timestep
rb = parse(Float64, parameters[16]) #Beam radius
tr = parse(Float64, parameters[17]) #Beam risetime in ns

# Source rate
neutralRate = parse(Float64, parameters[18]) #mass flow rate for neutrals in kg/s

# Recover config
recover = parse(Int, parameters[19]) #1 for recovering from savepoint
save = parse(Int, parameters[20]) #1 for save
save_path = parameters[21]

Nmax = parse(Int, parameters[22]) #total number of particles
writeTimeStep = parse(Float64, parameters[23]) #sampling frequency = time step * sampling factor
Bz =  parse(Float64, parameters[24])

boundaryCondition = parameters[25]

#--------------------------------------------------------------------------#
# constant
me =  9.10938356E-31;
qe =  1.6021766208e-19;
mi = 2.1802e-25 # Xenon atom
c = 3e8
kB = 1.380649e-23 #J/K
maxCounter = Int(floor(writeTimeStep / timeStep))
cross_section_energy_eV = CuArray([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
        55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0,
        130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0]);
cross_section_area = CuArray([1.15e-20, 2.42e-20, 3.81e-20, 4.17e-20, 4.17e-20,
        4.30e-20, 4.31e-20, 4.29e-20, 4.27e-20, 4.37e-20, 4.47e-20, 4.54e-20,
        4.57e-20, 4.59e-20, 4.55e-20, 4.48e-20, 4.42e-20, 4.31e-20, 4.26e-20, 
        4.21e-20, 4.13e-20, 4.06e-20, 3.99e-20, 3.97e-20, 3.92e-20,
        3.87e-20, 3.85e-20, 3.82e-20, 3.78e-20, 3.74e-20, 3.73e-20, 3.67e-20, 3.63e-20, 3.58e-20]);

#---------------------------------------------------------------------------#
N = numParticle;
Ne = edensity * (π*dR^2*dZ)
Ni = Ne; # quasi-neutrality
Nn = ndensity * (π*dR^2*dZ)

Ne_dot_max = BeamCurrent/qe
BeamEnergy *= qe
γ = 1 + BeamEnergy/(me*c^2)
Ue = c*sqrt(1 - γ^(-2))
neb = BeamCurrent/(qe*pi*rb^2*Ue)
# initial super sizes
electronSuperSize = Ne / N;
ionSuperSize = Ni / N;
neutralSuperSize = Nn/ N;

#Estimate neutral source particle
if neutralRate > 0
    neutralRateSuperSize = Ne_dot_max*timeStep/BeamSuperParticle
    neutralSuperParticle = neutralSuperSize
    if neutralSuperParticle < 1
        neutralSuperParticle = 1
        neutralRateSuperSize = neutralRate*timeStep/neutralSuperParticle
    else
        neutralSuperParticle = Int(round(neutralRate*timeStep/neutralRateSuperSize))
        neutralRateSuperSize = neutralRate*timeStep/neutralSuperParticle
    end
end
#---------------------------------------------------------------------------#
#---------------------------------------------------------------------------#
#Initialization
myParticle = Particle_GPU(Nmax)
myDomain = RZDomain_GPU(Nr,Nz,dR,dZ,"Dirichlet")
#Only support define one continuous region of magnetic field

myMagneticField = MagneticField_GPU([0.0,dR],[0.0,dR],[0.0,dZ],[0.0,0.0,Bz])
#output storage
myData = DataLogger(Nr, Nz)
if recover == 1
    timepath = joinpath(@__DIR__, "..", save_path, "time.txt")
    lines = open(timepath, "r") do file
        readlines(file)
    end
    time = parse(Float64, lines[1])
    restore_object_from_csv(myParticle, save_path)
    restore_object_from_csv(myDomain, save_path)
    restore_object_from_csv(myMagneticField, save_path)
    global Nmax = length(myParticle.alive)
    println("Restoring from ", time*1e9, " ns")
else
    if Ne > 0
        createParticle_GPU(N, me, Telectron, -qe, electronSuperSize, 0.0, CuArray([0,dR]), CuArray([0,dZ]), myParticle)
    end
    if Nn > 0
        createParticle_GPU(N, mi, 300.0, 0.0, neutralSuperSize, 216.0e-12, CuArray([0,dR]), CuArray([0,dZ]), myParticle)
    end
    if Ni > 0
        createParticle_GPU(N, mi, Tion, qe ,ionSuperSize, 216.0e-12, CuArray([0,dR]), CuArray([0,dZ]), myParticle)
    end
    time = 0.0;
end

if boundaryCondition == "Neumann"
    v_0 = 0.0
elseif boundaryCondition =="Dirichlet"
    v_0 = BeamEnergy/qe
end

counter = Int(0)

while time < maxTime
    global time
    global counter
    global dR
    global dZ
    global rb
    global Nmax
    NumElectron = 0
    #check if particle size need to be increased
    if CUDA.sum(myParticle.alive) >= 0.9*length(myParticle.alive)
        enlargeParticleArraySize!(myParticle)
    end
    particleCountCylindrical(myDomain,myParticle)
    
    PoissonCylindricalCUDSSNEW(myDomain, v_0)

    #Numionized = XenonNeutralCollisionalIonization_optimized(myParticle, myDomain, timeStep, EList, TList, WList, σList, σLength, stepT)
    #println("ionization: ", Numionized)
    #NumCollided = neutralCollisions(myParticle, timeStep, 2.1802e-25, 216.0e-12)
    #println("collision: ", NumCollided)
    setMagneticField(myParticle,myMagneticField)

    electricFieldCylindrical(myParticle, myDomain)
    #ImplicitPusher(myParticle, timeStep)
    BorisPusher(myParticle,timeStep)
    #reflectiveBC(myParticle,myDomain, ioncounter, electroncounter)
    AllreflectiveBC(myParticle,myDomain)
    #(ioncounter, electroncounter) = absorbBE_reflectPlasma_BC_countThrust(myParticle, myDomain)
    totalEnergy(myParticle,myDomain,myData)

    #print("ionlost", ioncounter)
    #println(", electronlost", electroncounter)
    #println(myData.energy)
    #print("ionlost", ioncounter)
    #println(", electronlost", electroncounter)
    #=
    if Ne_dot_max > 0
        if tr>0
            BeamSuperSize = clamp(Ne_dot_max/(tr*1e-9)*time, 0, Ne_dot_max)*timeStep/BeamSuperParticle
        else
            BeamSuperSize = Ne_dot_max*timeStep/BeamSuperParticle
        end
        if BeamSuperSize > 0
            createBeamParticle_GPU(BeamSuperParticle, me, 0.0, -qe, BeamSuperSize, 0.0, CuArray([0,rb]), CuArray([dZ/Nz/1000,dZ/Nz/999]), myParticle, Ue)
        end
        #print("Beam electron created = ")
        #println(BeamSuperParticle);
    end
      
    if neutralRate > 0
        #create neutral
        neutralSource(neutralRate,timeStep,myParticle)
    end
    =#
    #IO
    #------------------------------------------------------------------------------------------------------#
    accumualteOutputData(myDomain, myData)
    if counter >= maxCounter
        writeToFile(myData, counter, output_folder, time, timeStep)
        print("File written!")
        counter = 0
    else
        counter += 1
    end
    
    aliveidx = findall(x-> x==1, myParticle.alive)
    electronsubidx = findall(x->x<0,myParticle.qArray[aliveidx])
    electronidx = aliveidx[electronsubidx]
    NumElectron = CUDA.sum(myParticle.superParticleSizeArray[electronidx])
    
    println("Num of electron: ", NumElectron)
    println("Energy = ", myData.energy)
    time += timeStep
    println("time= ", time, ", Particle count: ", sum(myParticle.alive))
end
if save == 1
    timepath = joinpath(@__DIR__, "..", save_path, "time.txt")
    open(timepath, "w") do file
        write(file, string(time))  # Convert the number to a string and write it
    end
    save_object_to_csv(myParticle, save_path)
    save_object_to_csv(myDomain, save_path)
    save_object_to_csv(myMagneticField, save_path)
    println("Saved to ", save_path, ", time = ", time*1e9, " ns")
end