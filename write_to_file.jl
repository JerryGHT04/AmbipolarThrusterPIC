include("ParticlePusherJuliaTranslation.jl")
include("Poisson1DJuliaTranslation.jl")
function write_to_file()
    global timeStep
    global timeStamp
    global writeFrequency
    global outputPath
    global currentTime
    global writeTimer
    numOfParticles = getNumberOfParticles_jl(MyParticles)
    currentTime = currentTime + timeStep
    timeStamp = timeStamp + 1
    writeTimer = writeTimer + timeStep

    if writeTimer > writeFrequency
        writeTimer = 0.0
        writeToFile_jl(MyDomain, currentTime, outputPath)       
    end

    str = string("time = ", currentTime*1000000000.0, " ns, particle count = ", numOfParticles)
    return str
end
