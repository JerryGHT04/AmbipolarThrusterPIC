include("ParticlePusher.jl")
include("classdefinition.jl")


function createParticles_jl(myParticle::Particle, N::Real, mass::Float64, Temp::Float64, q::Float64, superParticleSize::Float64, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64, z_min::Float64, z_max::Float64, van_der_waals_radius::Float64,UeX::Float64)
    createParticles(N, mass, Temp, q, superParticleSize, x_min, x_max,y_min, y_max, z_min, z_max, van_der_waals_radius, 
    myParticle.xArray, myParticle.yArray, myParticle.zArray, myParticle.VxArray, myParticle.VyArray, myParticle.VzArray, 
    myParticle.xArray_old, myParticle.yArray_old, myParticle.zArray_old, myParticle.VxArray_old, myParticle.VyArray_old, myParticle.VzArray_old, 
    myParticle.ExArray, myParticle.EyArray, myParticle.EzArray, myParticle.BxArray, myParticle.ByArray, myParticle.BzArray,
    myParticle.mArray, myParticle.qArray, myParticle.superParticleSizeArray, myParticle.vdwrArray, 
    myParticle.localNn, myParticle.localNe, myParticle.localNi, myParticle.localCn, myParticle.localCe, myParticle.localCi, 
    myParticle.cellNumber, UeX
    )
end

function pushParticlesBoris_jl(myParticle::Particle, timeStep::Float64)
    par1 = myParticle.xArray
    par2 = myParticle.yArray
    par3 = myParticle.zArray
    par4 = myParticle.VxArray
    par5 = myParticle.VyArray
    par6 = myParticle.VzArray
    par7 = myParticle.xArray_old
    par8 = myParticle.yArray_old
    par9 = myParticle.zArray_old
    par10 = myParticle.VxArray_old
    par11 = myParticle.VyArray_old
    par12 = myParticle.VzArray_old
    par13 = myParticle.ExArray
    par14 = myParticle.EyArray
    par15 = myParticle.EzArray
    par16 = myParticle.BxArray
    par17 = myParticle.ByArray
    par18 = myParticle.BzArray
    par19 = myParticle.mArray
    par20 = myParticle.qArray
    par21 = timeStep
    pushParticlesBoris(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14, par15,par16, par17, par18, par19, par20, par21)
    
end

function getNumberOfParticles_jl(myParticle::Particle)
    return Int32(length(myParticle.xArray))
end

function writeParticlesToFile_jl(myParticle::Particle, path::String)
    par1 = myParticle.xArray
    par2 = myParticle.yArray
    par3 = myParticle.zArray
    par4 = myParticle.VxArray
    par5 = myParticle.VyArray
    par6 = myParticle.VzArray
    par7 = myParticle.mArray
    par8 = myParticle.qArray
    par9 = myParticle.superParticleSizeArray
    par10 = path

    writeParticlesToFile(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10)

end