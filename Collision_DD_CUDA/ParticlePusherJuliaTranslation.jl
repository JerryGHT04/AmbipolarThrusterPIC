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

function fixedTemperatureXenonIonization_jl(myParticle::Particle,myDomain::RectangularDomain,electron_temperature::Float64,ion_temperature::Float64,timeStep::Float64)
    par1 = electron_temperature
    par2 = ion_temperature
    par3 = myDomain.neutral_mass_store
    par4 = myParticle.xArray
    par5 = myParticle.yArray
    par6 = myParticle.zArray
    par7 = myParticle.VxArray
    par8 = myParticle.VyArray
    par9 = myParticle.VzArray
    par10 = myParticle.xArray_old
    par11 = myParticle.yArray_old
    par12 = myParticle.zArray_old
    par13 = myParticle.VxArray_old
    par14 = myParticle.VyArray_old
    par15 = myParticle.VzArray_old
    par16 = myParticle.ExArray
    par17 = myParticle.EyArray
    par18 = myParticle.EzArray
    par19 = myParticle.BxArray
    par20 = myParticle.ByArray
    par21 = myParticle.BzArray
    par22 = myParticle.mArray
    par23 = myParticle.qArray
    par24 = myParticle.superParticleSizeArray
    par25 = myParticle.vdwrArray
    par26 = myParticle.localNn
    par27 = myParticle.localNe
    par28 = myParticle.localNi
    par29 = myParticle.localCn
    par30 = myParticle.localCe
    par31 = myParticle.localCi
    par32 = myParticle.cellNumber
    par33 = timeStep
    
    
    fixedTemperatureXenonIonization(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11,par12,par13,par14,par15,par16,par17,par18,par19,par20,par21,par22,par23,par24,par25,par26,par27,par28,par29,par30,par31,par32,par33)


end

function reduceNeutralMassFromArray_jl(myParticle::Particle,myDomain::RectangularDomain)
    par1 = myDomain.neutral_mass_store
    par2 = myParticle.qArray
    par3 = myParticle.cellNumber
    par4 = myParticle.mArray
    par5 = myParticle.superParticleSizeArray
    
    reduceNeutralMassFromArray(par1, par2, par3, par4, par5)
end

function neutralCollisions_jl(myParticle::Particle, neutral_particle_mass::Float64,neutral_van_der_waals_radius::Float64,timeStep::Float64)
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
    par13 = myParticle.mArray
    par14= myParticle.vdwrArray
    par15 = myParticle.localNn
    par16 = myParticle.localCn
    par17 = myParticle.superParticleSizeArray
    par18 = neutral_particle_mass
    par19 = neutral_van_der_waals_radius
    par20 = timeStep
    
    neutralCollisions(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14, par15, par16,par17,par18,par19,par20)

end
