include("verifyParameter.jl")

#1. Define parameters
#Define common parameters
maxTime = 1e-7
timeStep = 5e-13
Nz = 120
numParticle = 20000
Telectron = 0.0
Tion = 0.0
dR = 0.005
dZ = 0.020
Nr = Int(round(dR/(dZ/Nz)))
Nmax = 40000
writeTimeStep = 1e-11
Bz = 4000
recover = 0
save = 1
saveFrequency = 5e-8

#Define varying parameters
BeamEnergy = [100,300,500]
BeamCurrent = [4.0,4.0,4.0]
BeamSuperParticle = [25,25,25]
rb = [0.001,0.001,0.001]
tr = [0.0,0.0,0.0]
neutralRate = [0.0,0.0,0.0]
boundaryCondition = ["Neumann","Neumann","Neumann"]
edensity = [0.0,0.0,0.0]
ndensity = [5e19,5e19,5e19]

#Define name of each sim
simNames = ["100eV4A","300eV4A","500eV4A"]
numofSims = length(simNames)

#Define a unique save_path and output_folder, according to simNames
output_folder =  Vector{String}(undef, numofSims)
save_path =  Vector{String}(undef, numofSims)

for i = 2:numofSims
    output_folder[i] = "output_"*simNames[i]*"/"
    save_path[i] = "save_"*simNames[i]
end
#2. Verify each parameter
for i = 2:numofSims
    flag = verifyParams(BeamEnergy[i], BeamCurrent[i], rb[i], edensity[i], Telectron, timeStep, Nz, dZ, Bz)
    println("Parameter "*string(i)*" checked")
end

#3. Create parameter txt file
parameterNames = Vector{String}(undef, numofSims)
for i = 2:numofSims
    parameterNames[i] = "params_"*simNames[i]*".txt"
    createParamsFile(maxTime, timeStep, Nr, Nz, numParticle, Telectron, Tion, dR, dZ, Nmax, writeTimeStep, Bz, recover, save
    , BeamEnergy[i],BeamCurrent[i], BeamSuperParticle[i], rb[i], tr[i], neutralRate[i], boundaryCondition[i], output_folder[i], save_path[i], parameterNames[i], edensity[i], ndensity[i], saveFrequency)
end

#4. run main_EB_NEW several times
for i = 2:numofSims
    # Set the parameter file name for this iteration
    global paramsName = parameterNames[i]
    
    # Include and run the main script
    include("src/main_EB_NEW.jl")
end


