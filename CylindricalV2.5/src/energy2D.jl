using CUDA
include("classDefinition.jl")
include("auxlibrary.jl")

function totalEnergy_kernel(Energy,mArray,qArray,VelArray,phi,rCellNum,zCellNum, alive, superParticleSizeArray)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    # loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            CUDA.@atomic Energy[1] += superParticleSizeArray[idx]*mArray[idx]*(VelArray[1,idx]^2 + VelArray[2,idx]^2 + VelArray[3,idx]^2) + superParticleSizeArray[idx]*qArray[idx]*gather(phi,rCellNum[idx],zCellNum[idx])
        end
    end
    return
end

function totalEnergy(myParticle, myDomain, myData)
    myData.energy .= 0
    @cuda(
        threads=256,
        blocks=cld(length(myParticle.alive),256),
        totalEnergy_kernel(myData.energy,myParticle.mArray,myParticle.qArray,myParticle.VelArray,myDomain.phi,myParticle.rCellNum,myParticle.zCellNum, myParticle.alive, myParticle.superParticleSizeArray))
end