#using IterativeSolvers
include("classDefinition.jl")
include("auxlibrary.jl")
#using LinearSolve
using CUDA, CUDA.CUSPARSE
using CUDSS
using SparseArrays, LinearAlgebra

function createParticle_kernel(N, prefixsum, mass, Temp, charge, superSize, Rrange, Zrange, van_der_waals_radius, PosArray, PosArray_old, VelArray, VelArray_old, mArray, qArray, superParticleSizeArray, vdwrArray, alive, isBeam)
    idx = threadIdx().x + (blockIdx().x-1) * blockDim().x
    
    if idx <= length(prefixsum)
        if alive[idx] == 0 && prefixsum[idx] <= N
            # Set this element as alive, so it will be looped when needed
            alive[idx] = 1
            isBeam[idx] = 0

            #r = sqrt(rand() * (Rrange[2]^2 - Rrange[1]^2) + Rrange[1]^2)
            r = Rrange[2] + 1
            while r >= Rrange[2]
                @inbounds PosArray[1,idx] = (rand() * (Rrange[2] - Rrange[1])) + Rrange[1]
                @inbounds PosArray[2,idx] =  (rand() * (Rrange[2] - Rrange[1])) + Rrange[1]
                r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            end

            @inbounds PosArray[3,idx] = (rand() * (Zrange[2] - Zrange[1])) + Zrange[1]
            #
            y1 = rand();
            while (y1 == 0.0)
                y1 = rand();
            end
            y2 = rand();
            @inbounds VelArray[1,idx] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp/mass)

            y1 = rand();
            while (y1 == 0.0)
                y1 = rand();
            end
            y2 = rand();
            @inbounds VelArray[2,idx] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp/mass)

            y1 = rand();
            while (y1 == 0.0)
                y1 = rand();
            end
            y2 = rand();
            @inbounds VelArray[3,idx] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp/mass)

            @inbounds PosArray_old[1,idx] = PosArray[1,idx]
            @inbounds PosArray_old[2,idx] = PosArray[2,idx]
            @inbounds PosArray_old[3,idx] = PosArray[3,idx]

            @inbounds VelArray_old[1,idx] = VelArray[1,idx]
            @inbounds VelArray_old[2,idx] = VelArray[2,idx]
            @inbounds VelArray_old[3,idx] = VelArray[3,idx]

            @inbounds mArray[idx] = mass
            @inbounds qArray[idx] = charge
            @inbounds superParticleSizeArray[idx] = superSize
            @inbounds vdwrArray[idx] = van_der_waals_radius
        end
    end
    return
end

function createBeamParticle_kernel(N, prefixsum, mass, Temp, charge, superSize, Rrange, Zrange, van_der_waals_radius, PosArray, PosArray_old, VelArray, VelArray_old, mArray, qArray, superParticleSizeArray, vdwrArray, alive, BArray, Ue, isBeam)
    idx = threadIdx().x + (blockIdx().x-1) * blockDim().x
    
    if idx <= length(prefixsum)
        if alive[idx] == 0 && prefixsum[idx] <= N
            # Set this element as alive, so it will be looped when needed
            alive[idx] = 1
            isBeam[idx] = 1

            #r = sqrt(rand() * (Rrange[2]^2 - Rrange[1]^2) + Rrange[1]^2)
            r = Rrange[2] + 1
            while r >= Rrange[2]
                @inbounds PosArray[1,idx] = (rand() * (Rrange[2] - Rrange[1])) + Rrange[1]
                @inbounds PosArray[2,idx] =  (rand() * (Rrange[2] - Rrange[1])) + Rrange[1]
                r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            end

            @inbounds PosArray[3,idx] = (rand() * (Zrange[2] - Zrange[1])) + Zrange[1]

            @inbounds VelArray[1,idx] = 0.0
            @inbounds VelArray[2,idx] = 0.0
            @inbounds VelArray[3,idx] = Ue

            @inbounds PosArray_old[1,idx] = PosArray[1,idx]
            @inbounds PosArray_old[2,idx] = PosArray[2,idx]
            @inbounds PosArray_old[3,idx] = PosArray[3,idx]

            @inbounds VelArray_old[1,idx] = VelArray[1,idx]
            @inbounds VelArray_old[2,idx] = VelArray[2,idx]
            @inbounds VelArray_old[3,idx] = VelArray[3,idx]

            @inbounds mArray[idx] = mass
            @inbounds qArray[idx] = charge
            @inbounds superParticleSizeArray[idx] = superSize
            @inbounds vdwrArray[idx] = van_der_waals_radius
        end
    end
    return
end


function createParticle_GPU(N, mass, Temp, charge, superSize, van_der_waals_radius, Rrange, Zrange, myParticle::Particle_GPU)
    empty_flags = CUDA.zeros(Int32, length(myParticle.alive))

    CUDA.@sync @cuda(
        threads=256,
        blocks=cld(length(myParticle.alive),256),
        mark_empty_spaces!(empty_flags, myParticle.alive))
    #find prefixsum. eg. [1 0 0 1] gives [1 1 1 2], this means there is at most 1 particles can be created between index 1 and 3, at most 2 particles at index 4
    prefixsum = CUDA.zeros(Int32, length(myParticle.alive))
    CUDA.scan!(+, prefixsum, empty_flags, dims=1)

    if reduce(+,myParticle.alive ) + N > length(myParticle.alive)
        error("maximum number of particle reached! Attempted to add " * string(N) * "particles to " * string(length(myParticle.alive)) * "in function createBeamParticle_GPU")
    end

    CUDA.@sync @cuda(
        threads=256,
        blocks=cld(length(myParticle.alive),256),
        createParticle_kernel(N, prefixsum, mass, Temp, charge, superSize, Rrange, Zrange, van_der_waals_radius, myParticle.PosArray, myParticle.PosArray_old, myParticle.VelArray, myParticle.VelArray_old, myParticle.mArray, myParticle.qArray, myParticle.superParticleSizeArray, myParticle.vdwrArray, myParticle.alive, myParticle.isBeam)
        )
    empty_flags = nothing
    prefixsum = nothing
end


function createBeamParticle_GPU(N, mass, Temp, charge, superSize, van_der_waals_radius, Rrange, Zrange, myParticle::Particle_GPU ,Ue)
    empty_flags = CUDA.zeros(Int32, length(myParticle.alive))

    CUDA.@sync @cuda(
        threads=256,
        blocks=cld(length(myParticle.alive),256),
        mark_empty_spaces!(empty_flags, myParticle.alive))

    prefixsum = CUDA.zeros(Int32, length(myParticle.alive))
    CUDA.scan!(+, prefixsum, empty_flags, dims=1)

    if reduce(+,myParticle.alive ) + N > length(myParticle.alive)
        error("maximum number of particle reached! Attempted to add " * string(N) * "particles to " * string(length(myParticle.alive)) * "in function createBeamParticle_GPU")
    end

    CUDA.@sync @cuda(
        threads=256,
        blocks=cld(length(myParticle.alive),256),
        createBeamParticle_kernel(N, prefixsum, mass, Temp, charge, superSize, Rrange, Zrange, van_der_waals_radius, myParticle.PosArray, myParticle.PosArray_old, myParticle.VelArray, myParticle.VelArray_old, myParticle.mArray, myParticle.qArray, myParticle.superParticleSizeArray, myParticle.vdwrArray, myParticle.alive, myParticle.BArray,Ue, myParticle.isBeam)
        )
    empty_flags = nothing
    prefixsum = nothing
end

function particleDomainMapping_kernel(
    PosArray, qArray, alive, rCellNum, zCellNum, r_MAX, z_MAX, charge_bin, 
    superParticleSizeArray, VelArray, CeArray, CiArray, CnArray, 
    neArray, niArray, nnArray, BxBinArray, ByBinArray, BzBinArray, 
    uexArray, ueyArray, uezArray, uixArray, uiyArray, uizArray, Nr, Nz, BArray, dr, dz, nodeVolume, mean_neutralSuperSize
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            # Calculate the local node indices
            r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            z = PosArray[3,idx]

            i0 = r/dr[1] + 1
            j0 = z/dz[1] + 1
            # Ensure i0, j are within bounds
            if i0 <= Nr[1] && j0 <= Nz[1]
                # Set local cell number
                @inbounds rCellNum[idx] = i0
                @inbounds zCellNum[idx] = j0
                
                # Accumulate charge using atomic operations to prevent race conditions
                scatter(charge_bin,i0,j0,superParticleSizeArray[idx] * qArray[idx])

                if qArray[idx] < 0.0
                    # Electron
                    #@cuprint("electron")
                    scatter(neArray,i0,j0,superParticleSizeArray[idx])
                    scatter(uexArray,i0,j0,superParticleSizeArray[idx]* VelArray[1, idx])
                    scatter(ueyArray,i0,j0,superParticleSizeArray[idx] * VelArray[2, idx])
                    scatter(uezArray,i0,j0,superParticleSizeArray[idx] * VelArray[3, idx])
                    scatter(CeArray,i0,j0,sqrt(VelArray[1, idx]^2 + VelArray[2, idx]^2 + VelArray[3, idx]^2) * superParticleSizeArray[idx])
                    scatter(BxBinArray,i0,j0, superParticleSizeArray[idx] * BArray[1,idx])
                    scatter(ByBinArray,i0,j0, superParticleSizeArray[idx] * BArray[2,idx])
                    scatter(BzBinArray,i0,j0, superParticleSizeArray[idx] * BArray[3,idx])
                elseif qArray[idx] > 0.0
                    # Ion
                    scatter(niArray,i0,j0,superParticleSizeArray[idx])
                    scatter(uixArray,i0,j0,superParticleSizeArray[idx]* VelArray[1, idx])
                    scatter(uiyArray,i0,j0,superParticleSizeArray[idx] * VelArray[2, idx])
                    scatter(uizArray,i0,j0,superParticleSizeArray[idx] * VelArray[3, idx])
                    scatter(CiArray,i0,j0,sqrt(VelArray[1, idx]^2 + VelArray[2, idx]^2 + VelArray[3, idx]^2) * superParticleSizeArray[idx])
                else
                    # Neutrals
                    scatter(nnArray,i0,j0,superParticleSizeArray[idx])
                    scatter(CnArray,i0,j0,sqrt(VelArray[1, idx]^2 + VelArray[2, idx]^2 + VelArray[3, idx]^2) * superParticleSizeArray[idx])
                    scatter(mean_neutralSuperSize, i0, j0, 1) #record number of neutrals in this cell
                end
            end
        end
    end
    return nothing
end


function normalizeCell_kernel(
    Nr, Nz, CeArray, CiArray, CnArray, neArray, niArray, nnArray,
    BxBinArray, ByBinArray, BzBinArray, uexArray, ueyArray, uezArray,
    uixArray, uiyArray, uizArray, nodeVolume, mean_neutralSuperSize
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    # Loop through all nodes
    # Calculate particle avearged value
    if i <= Nr[1] && j <= Nz[1]
        if neArray[i,j] > 0.0
            @inbounds uexArray[i,j] /= neArray[i,j]
            @inbounds ueyArray[i,j] /= neArray[i,j]
            @inbounds uezArray[i,j] /= neArray[i,j]
            @inbounds CeArray[i,j] /= neArray[i,j]
            @inbounds BxBinArray[i,j] /= neArray[i,j]
            @inbounds ByBinArray[i,j] /= neArray[i,j]
            @inbounds BzBinArray[i,j] /= neArray[i,j]
            @inbounds neArray[i,j] /= nodeVolume[i,j]
        else
            @inbounds uexArray[i,j] = 0.0
            @inbounds ueyArray[i,j] = 0.0
            @inbounds uezArray[i,j] = 0.0
            @inbounds CeArray[i,j] = 0.0
            @inbounds BxBinArray[i,j] = 0.0
            @inbounds ByBinArray[i,j] = 0.0
            @inbounds BzBinArray[i,j] = 0.0
            @inbounds neArray[i,j] = 0.0
        end

        if niArray[i,j] > 0.0
            @inbounds uixArray[i,j] /= niArray[i,j]
            @inbounds uiyArray[i,j] /= niArray[i,j]
            @inbounds uizArray[i,j] /= niArray[i,j]
            @inbounds CiArray[i,j] /= niArray[i,j]
            @inbounds niArray[i,j] /= nodeVolume[i,j]
        else
            @inbounds uixArray[i,j] = 0.0
            @inbounds uiyArray[i,j] = 0.0
            @inbounds uizArray[i,j] = 0.0
            @inbounds CiArray[i,j] = 0.0
            @inbounds niArray[i,j] = 0.0
        end

        if nnArray[i,j] > 0.0
            @inbounds CnArray[i,j] /= nnArray[i,j]
            if mean_neutralSuperSize[i,j] > 0
                temp = nnArray[i,j] / mean_neutralSuperSize[i,j]
                @inbounds mean_neutralSuperSize[i,j] = temp
            else
                mean_neutralSuperSize = 0
            end
            @inbounds nnArray[i,j] /= nodeVolume[i,j]
        else
            @inbounds CnArray[i,j] = 0.0
            @inbounds nnArray[i,j] = 0.0
        end
    end

    

    return nothing
end

function setLocalProperties_kernel(
    localNn, localNe, localNi, localCn, localCe, localCi,
    nnArray, neArray, niArray, CnArray, CeArray, CiArray,
    alive, rCellNum, zCellNum, localColor, color
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            #distribute grid point values to particle location
            @inbounds localNn[idx] = gather(nnArray, rCellNum[idx], zCellNum[idx])
            @inbounds localNe[idx] = gather(neArray, rCellNum[idx], zCellNum[idx])
            @inbounds localNi[idx] = gather(niArray, rCellNum[idx], zCellNum[idx])
            @inbounds localCn[idx] = gather(CnArray, rCellNum[idx], zCellNum[idx])
            @inbounds localCe[idx] = gather(CeArray, rCellNum[idx], zCellNum[idx])
            @inbounds localCi[idx] = gather(CiArray, rCellNum[idx], zCellNum[idx])

            i = CUDA.Int(CUDA.trunc(rCellNum[idx]))
            j = CUDA.Int(CUDA.trunc(zCellNum[idx]))

            @inbounds localColor[idx] = color[i,j]
        end
    end

    return nothing
end

function particleCountCylindrical(myDomain::RZDomain_GPU,myParticle::Particle_GPU)
    #Fix the function by scatter the charge of a particle to nearby grid points
    myDomain.charge_bin .= 0.0
    myDomain.neArray .= 0.0
    myDomain.niArray .= 0.0
    myDomain.nnArray .= 0.0
    myDomain.uexArray .= 0.0
    myDomain.ueyArray .= 0.0
    myDomain.uezArray .= 0.0
    myDomain.uixArray .= 0.0
    myDomain.uiyArray .= 0.0
    myDomain.uizArray .= 0.0
    myDomain.CnArray .= 0.0
    myDomain.CiArray .= 0.0
    myDomain.CeArray .= 0.0
    myDomain.BxBinArray .= 0.0
    myDomain.ByBinArray .= 0.0
    myDomain.BzBinArray .= 0.0
    myDomain.mean_neutralSuperSize .= 0.0
    myDomain.neutral_change .= 0.0


    CUDA.@sync @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    particleDomainMapping_kernel(myParticle.PosArray, myParticle.qArray, myParticle.alive, myParticle.rCellNum, myParticle.zCellNum, myDomain.r_MAX, myDomain.z_MAX, myDomain.charge_bin, myParticle.superParticleSizeArray, myParticle.VelArray, myDomain.CeArray, myDomain.CiArray, myDomain.CnArray, 
    myDomain.neArray, myDomain.niArray, myDomain.nnArray, myDomain.BxBinArray, myDomain.ByBinArray, myDomain.BzBinArray, myDomain.uexArray, myDomain.ueyArray, myDomain.uezArray, myDomain.uixArray, myDomain.uiyArray, myDomain.uizArray,
    myDomain.Nr, myDomain.Nz, myParticle.BArray, myDomain.dr, myDomain.dz, myDomain.nodeVolume, myDomain.mean_neutralSuperSize)
    )
    
    sz = size(myDomain.phi)

    CUDA.@sync @cuda(
    threads=(16,16),
    blocks=(cld(sz[1], 16), cld(sz[2],16)),
    normalizeCell_kernel(myDomain.Nr, myDomain.Nz, myDomain.CeArray, myDomain.CiArray, myDomain.CnArray, myDomain.neArray, myDomain.niArray, myDomain.nnArray, myDomain.BxBinArray, myDomain.ByBinArray, myDomain.BzBinArray, myDomain.uexArray
    , myDomain.ueyArray, myDomain.uezArray, myDomain.uixArray, myDomain.uiyArray, myDomain.uizArray, myDomain.nodeVolume, myDomain.mean_neutralSuperSize)
    )

    CUDA.@sync @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    setLocalProperties_kernel(
        myParticle.localNn, myParticle.localNe, myParticle.localNi, myParticle.localCn, myParticle.localCe, myParticle.localCi,
        myDomain.nnArray, myDomain.neArray, myDomain.niArray, myDomain.CnArray, myDomain.CeArray, myDomain.CiArray,
        myParticle.alive, myParticle.rCellNum, myParticle.zCellNum, myParticle.localColor, myDomain.color
    ))

end


function construct_B(rho, eps0, phi, phiz_first, phiz_last)
    sz = size(phi)  # Get the size of the input phi, which is also the size of rho
    Nr, Nz = sz  # Extract the dimensions
    
    B = CUDA.zeros(Float64,Nr * Nz)  # Initialize B as a 1D array
    
    # Kernel function to fill B
    function fill_B!(B, rho, eps0, phiz_first, phiz_last, Nr, Nz)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        
        if i <= Nr && j <= Nz
            idx = (i-1) * Nz + j  # Linear index
            
            if i == 1
                # Neumann BC at r = 0: Adjust B by setting it to 0
                @inbounds B[idx] = 0
            
            elseif i == Nr
                # Dirichlet BC at r = rMax: Set B to the boundary potential value
                @inbounds B[idx] = 0.0
            
            elseif j == 1
                # Dirichlet BC at z = 0: Set B to the boundary potential value
                @inbounds B[idx] = phiz_first
            
            elseif j == Nz
                # Dirichlet BC at z = zMax: Set B to the boundary potential value
                @inbounds B[idx] = phiz_last
            
            else
                # Interior points: Set B based on charge density
                @inbounds B[idx] = -rho[i, j] / eps0
            end
        end
        return
    end
    
    CUDA.@sync @cuda(
    threads=(16,16),
    blocks=(cld(Nr, 16), cld(Nz,16)),
    fill_B!(B, rho, eps0, phiz_first, phiz_last, Nr, Nz))
    
    return B
end

function construct_BNEW(rho, eps0, phi, phiz_first, phiz_last)
    sz = size(phi)  # Get the size of the input phi, which is also the size of rho
    Nr, Nz = sz  # Extract the dimensions
    
    B = CUDA.zeros(Float64,Nr * Nz)  # Initialize B as a 1D array
    
    # Kernel function to fill B
    function fill_B!(B, rho, eps0, phiz_first, phiz_last, Nr, Nz)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        
        if i <= Nr && j <= Nz
            idx = (i-1) * Nz + j  # Linear index
            
            if i == 1
                # Neumann BC at r = 0: Adjust B by setting it to 0
                @inbounds B[idx] = 0.0
            
            elseif i == Nr
                # Dirichlet BC at r = rMax: Set B to the boundary potential value
                @inbounds B[idx] = 0.0
            
            elseif j == 1
                # Dirichlet BC at z = 0: Set B to the boundary potential value
                @inbounds B[idx] = 0.0
            
            elseif j == Nz
                # Neumann BC at z = zMax: Set B to the boundary potential value
                @inbounds B[idx] = 0.0
            
            else
                # Interior points: Set B based on charge density
                @inbounds B[idx] = -rho[i, j] / eps0
            end
        end
        return
    end
    
    CUDA.@sync @cuda(
    threads=(16,16),
    blocks=(cld(Nr, 16), cld(Nz,16)),
    fill_B!(B, rho, eps0, phiz_first, phiz_last, Nr, Nz))
    
    return B
end


function PoissonCylindrical(myDomain, phiz_first::Float64, phiz_last::Float64)

    sz = size(myDomain.phi)
    N =  sz[1]* sz[2]
    eps0 = 8.85418782e-12
    qe =  1.6021766208e-19;
    #construct B
    #B = construct_B(qe*(myDomain.niArray - myDomain.neArray), eps0, myDomain.phi, phiz_first, phiz_last)
    B = construct_B(myDomain.charge_bin./myDomain.nodeVolume, eps0, myDomain.phi, phiz_first, phiz_last)

    phi_vector = CuArray(myDomain.A) \ B
    phi = reshape(phi_vector, Nz, Nr)
    myDomain.phi = phi'

    return B
end

function PoissonCylindricalLU(myDomain, phiz_first::Float64, phiz_last::Float64)

    sz = size(myDomain.phi)
    N =  sz[1]* sz[2]
    eps0 = 8.85418782e-12
    qe =  1.6021766208e-19;

    B = construct_B(myDomain.charge_bin./myDomain.nodeVolume , eps0, myDomain.phi, phiz_first, phiz_last)
    lu_fact = lu(myDomain.A)
    phi_vector = lu_fact \ B
    #phi_vector2 = CuArray(myDomain.A) \ B
    #println("LU 2-norm:",sqrt(sum((Array(phi_vector)-Array(phi_vector2)).^2)))

    phi = reshape(phi_vector, Nz, Nr)
    
    #println("transposing matrix")
    myDomain.phi = phi'

    return B
end

function PoissonCylindricalKLU(myDomain, phiz_first::Float64, phiz_last::Float64)

    sz = size(myDomain.phi)
    N =  sz[1]* sz[2]
    eps0 = 8.85418782e-12
    qe =  1.6021766208e-19;

    B = construct_B(myDomain.charge_bin./myDomain.nodeVolume , eps0, myDomain.phi, phiz_first, phiz_last)
    prob = LinearProblem(myDomain.A_regular, Array(B))
    sol = solve(prob, KLUFactorization())
    phi_vector = CuArray(sol.u)
    phi = reshape(phi_vector, Nz, Nr)
    
    #println("transposing matrix")
    myDomain.phi = phi'

    return B
end

function PoissonCylindricalCUDSS(myDomain, phiz_first::Float64, phiz_last::Float64)

    sz = size(myDomain.phi)
    N =  sz[1]* sz[2]
    eps0 = 8.85418782e-12
    qe =  1.6021766208e-19;

    B = construct_B(myDomain.charge_bin./myDomain.nodeVolume , eps0, myDomain.phi, phiz_first, phiz_last)
    phi_vector = CUDA.zeros(Float64, Nr*Nz)
    solver = CudssSolver(myDomain.A, "G", 'F')
    cudss("analysis", solver, phi_vector, B)
    cudss("factorization", solver, phi_vector, B)
    cudss("solve", solver, phi_vector, B)
    phi = reshape(phi_vector, Nz, Nr)
    #println("transposing matrix")
    myDomain.phi = phi'
    return B
end

function PoissonCylindricalCUDSSNEW(myDomain, phiz_first::Float64, phiz_last::Float64)

    sz = size(myDomain.phi)
    N =  sz[1]* sz[2]
    eps0 = 8.85418782e-12
    qe =  1.6021766208e-19;

    B = construct_BNEW(myDomain.charge_bin./myDomain.nodeVolume , eps0, myDomain.phi, phiz_first, phiz_last)
    phi_vector = CUDA.zeros(Float64, Nr*Nz)
    solver = CudssSolver(myDomain.A, "G", 'F')
    cudss("analysis", solver, phi_vector, B)
    cudss("factorization", solver, phi_vector, B)
    cudss("solve", solver, phi_vector, B)
    phi = reshape(phi_vector, Nz, Nr)
    #println("transposing matrix")
    myDomain.phi = phi'
    return B
end

function calculateNodeEField_kernel(Nr,Nz, dr, dz, ErArray, EzArray, phi)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #use a finite difference method
    #loop through all node
    if i <= Nr[1] && j <= Nz[1]
        # Handle the electric field in the r direction
        if i > 1
            if i == Nr[1]
                #No field at rmax
                ErArray[i, j] = 0.0
            elseif i == 2
                # 4th order Forward difference
                ErArray[i, j] = - (-phi[i+2,j] + 4*phi[i+1,j] - 3*phi[i,j]) / (2*dr[1])
            elseif i == Nr[1] - 1
                # 4th orde Backward difference
                ErArray[i, j] = - (3*phi[i,j] - 4*phi[i-1,j] + phi[i-2,j]) / (2 * dr[1])
            else
                # forth order central difference
                ErArray[i, j] = - (-phi[i+2,j] + 8*phi[i+1,j] - 8*phi[i-1,j] + phi[i-2,j])/(12*dr[1])
            end
        elseif i == 1
            #Neumann BC
            ErArray[i, j] = 0.0        
        end

        # Handle the electric field in the z direction
        if j > 2
            if (j == Nz[1]) || (j == Nz[1] - 1)
                #Backward difference for boundary at z = zMax
                EzArray[i, j] = - (3*phi[i,j] - 4*phi[i,j-1] + phi[i,j-2]) / (2 * dz[1])
            else
                #forth order central difference
                EzArray[i, j] = -(-phi[i,j+2] + 8*phi[i,j+1] - 8*phi[i,j-1] + phi[i,j-2])/(12*dz[1])
            end
        elseif (j == 1) || (j == 2)
            # Forward difference for boundary at z = 0
            EzArray[i, j] = - (-phi[i,j+2] + 4*phi[i,j+1] - 3*phi[i,j]) / (2*dz[1])
        end
    end
    return
end


function gatherEField_kernel(zCellNum, rCellNum, alive, localE, ErArray, EzArray, PosArray)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            #Use central difference for interior points, forwawrd at 0, backward at end
            #Calculate electric field in r,z direction, then decompose it to xyz directions
            localEr = gather(ErArray, rCellNum[idx], zCellNum[idx])
            localEz = gather(EzArray, rCellNum[idx], zCellNum[idx])

            r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            
            #convert from cylindrical to cartesian
            cos = PosArray[1,idx] / r
            sin = PosArray[2,idx] / r
            localE[1,idx] = localEr * cos
            localE[2,idx] = localEr * sin
            localE[3,idx] = localEz
        end
    end
    return nothing
end



function electricFieldCylindrical(myParticle, myDomain)
    sz = size(myDomain.phi)

    CUDA.@sync @cuda(
    threads=(16,16),
    blocks=(cld(sz[1], 16), cld(sz[2],16)),
    calculateNodeEField_kernel(myDomain.Nr, myDomain.Nz, myDomain.dr, myDomain.dz, myDomain.ErArray, myDomain.EzArray, myDomain.phi)
    )

    CUDA.@sync @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    gatherEField_kernel(myParticle.zCellNum, myParticle.rCellNum, myParticle.alive, myParticle.localE, myDomain.ErArray, myDomain.EzArray, myParticle.PosArray))
    
end


function setMagneticField_kernel(alive, PosArray, BArray, regionXRange, regionYRange, regionZRange, regionB, gaussian_profile_flag, store_B_max, store_B_max_x, store_B_half_x)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    # loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            #check if particle is within the effect of magnetic field
            if (PosArray[1,idx] >= regionXRange[1] && PosArray[1,idx] <= regionXRange[2]) && (PosArray[2,idx] >= regionYRange[1] && PosArray[2,idx] <= regionYRange[2])&& (PosArray[3,idx] >= regionZRange[1] && PosArray[3,idx] <= regionZRange[2])
                @inbounds BArray[1,idx] = regionB[1]
                @inbounds BArray[2,idx] = regionB[2]
                @inbounds BArray[3,idx] = regionB[3]
            else
                @inbounds BArray[1,idx] = 0.0
                @inbounds BArray[2,idx] = 0.0
                @inbounds BArray[3,idx] = 0.0
            end
        end
    end
    return nothing
end

function setMagneticField(myParticle::Particle_GPU, myMagneticField::MagneticField_GPU)
    CUDA.@sync @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    setMagneticField_kernel(myParticle.alive, myParticle.PosArray, myParticle.BArray, myMagneticField.regionXRange, myMagneticField.regionYRange, myMagneticField.regionZRange, myMagneticField.regionB, myMagneticField.gaussian_profile_flag, myMagneticField.store_B_max, myMagneticField.store_B_max_x, myMagneticField.store_B_half_x)
   )
end

function BorisPusher(myParticle, timeStep)
    mArray = repeat(reshape(myParticle.mArray, 1, Nmax), 3, 1)  # Broadcast to 3xNmax
    qArray = repeat(reshape(myParticle.qArray, 1, Nmax), 3, 1)
    vm = myParticle.VelArray + qArray.*myParticle.localE*timeStep./(mArray*2);
    
    t = (qArray.*myParticle.BArray./mArray)*timeStep/2
    t_squared = sum(t .^ 2, dims=1)
    
    u = vm
    v = t
    vprime = v + (mycross(u,v))

    s = 2*t ./ (1 .+ t_squared)

    u = vprime
    v = s
    vp = vm + (mycross(u,v))

    v_new = vp + qArray.*myParticle.localE*timeStep./(mArray*2)

    myParticle.VelArray_old = myParticle.VelArray
    myParticle.VelArray = v_new
    myParticle.PosArray_old = myParticle.PosArray
    myParticle.PosArray .+= timeStep.*v_new
end

function ImplicitPusher(myParticle, timeStep)
    
    CUDA.@sync @cuda(
        threads=256,
        blocks=cld(length(myParticle.alive),256),
        ImplicitPusherKernel(
            myParticle.VelArray, myParticle.PosArray, myParticle.VelArray_old, myParticle.PosArray_old,
            myParticle.qArray, myParticle.mArray, timeStep, myParticle.localE, myParticle.alive, myParticle.BArray
        )
       )


end

function ImplicitPusherKernel(
    VelArray, PosArray, VelArray_old, PosArray_old,
    qArray, mArray, timeStep, localE, alive, BArray
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(alive) && alive[idx] == 1
        # Extract particle properties
        q = qArray[idx]
        m = mArray[idx]
        Δt = timeStep
        c = 3e8

        # Access particle velocity and position components
        vpn1 = VelArray[1, idx]
        vpn2 = VelArray[2, idx]
        vpn3 = VelArray[3, idx]

        xpn1 = PosArray[1, idx]
        xpn2 = PosArray[2, idx]
        xpn3 = PosArray[3, idx]

        # Access local electric and magnetic fields
        Ep1 = localE[1, idx]
        Ep2 = localE[2, idx]
        Ep3 = localE[3, idx]

        Bp1 = BArray[1, idx]
        Bp2 = BArray[2, idx]
        Bp3 = BArray[3, idx]

        # Compute magnitude of Bp
        BpMag = sqrt(Bp1^2 + Bp2^2 + Bp3^2)

        # Update velocity (vp1)
        vp1_1 = vpn1 + (q * Δt) / (2 * m) * Ep1
        vp1_2 = vpn2 + (q * Δt) / (2 * m) * Ep2
        vp1_3 = vpn3 + (q * Δt) / (2 * m) * Ep3

        # Compute cross product vp1 × Bp
        cross1 = vp1_2 * Bp3 - vp1_3 * Bp2
        cross2 = vp1_3 * Bp1 - vp1_1 * Bp3
        cross3 = vp1_1 * Bp2 - vp1_2 * Bp1

        # Compute dot product vp1 ⋅ Bp
        dot = vp1_1 * Bp1 + vp1_2 * Bp2 + vp1_3 * Bp3

        # Compute numerator and denominator
        factor = q * Δt / (2 * m * c)
        numerator1 = vp1_1 + factor * (cross1 + factor * dot * Bp1)
        numerator2 = vp1_2 + factor * (cross2 + factor * dot * Bp2)
        numerator3 = vp1_3 + factor * (cross3 + factor * dot * Bp3)

        denominator = 1 + ((q * Δt * BpMag) / (2 * m * c))^2

        # Compute vpbar components
        vpbar1 = numerator1 / denominator
        vpbar2 = numerator2 / denominator
        vpbar3 = numerator3 / denominator

        # Update velocities
        vpnew1 = 2 * vpbar1 - vpn1
        vpnew2 = 2 * vpbar2 - vpn2
        vpnew3 = 2 * vpbar3 - vpn3

        # Update positions
        xpnew1 = xpn1 + vpbar1 * Δt
        xpnew2 = xpn2 + vpbar2 * Δt
        xpnew3 = xpn3 + vpbar3 * Δt

        # Save old velocities and positions
        VelArray_old[1, idx] = vpn1
        VelArray_old[2, idx] = vpn2
        VelArray_old[3, idx] = vpn3

        PosArray_old[1, idx] = xpn1
        PosArray_old[2, idx] = xpn2
        PosArray_old[3, idx] = xpn3

        # Update velocities and positions
        VelArray[1, idx] = vpnew1
        VelArray[2, idx] = vpnew2
        VelArray[3, idx] = vpnew3

        PosArray[1, idx] = xpnew1
        PosArray[2, idx] = xpnew2
        PosArray[3, idx] = xpnew3
    end
    return
end

function BorisPusherRelativistic(myParticle, timeStep)
    Nmax = length(myParticle.alive)
    
    # Constants
    c = 3.0e8  # Speed of light in m/s

    # Step 1: First, the particle is accelerated half a time step with the electric field E only
    mArray = repeat(reshape(myParticle.mArray, 1, Nmax), 3, 1)  # Broadcast to 3xNmax
    qArray = repeat(reshape(myParticle.qArray, 1, Nmax), 3, 1)
    u1 = myParticle.VelArray + (qArray .* timeStep ./ (2 * mArray)) .* myParticle.localE
    #println("u1 = ", u1)
    # Step 2: Calculate the relativistic factor gamma
    gamma = sqrt.(1 .+ sum(u1 .^ 2, dims=1) ./ c^2)
    gamma = repeat(reshape(gamma, 1, Nmax), 3, 1)  # Broadcast to 3xNmax
    
    # Step 3: Compute Omega (Ω = qB/mc)
    Omega = qArray .* myParticle.BArray ./ mArray / c

    # Step 4: Compute the magnitude of Omega (Ω)
    omega_scal = sqrt.(sum(Omega .^ 2, dims=1))
    omega_scal = repeat(omega_scal, 3, 1)  # Broadcast to 3xNmax

    # Step 5: Compute u2 based on the provided formula
    term1 = u1 .* (1 .- (omega_scal .* timeStep ./ (2 * gamma)) .^ 2)
    term2 = mycross(u1, Omega .* timeStep)./ gamma
    term3 = 0.5*(u1 .* Omega) .* Omega .* ((timeStep ./ gamma) .^ 2)

    u2 = (term1 + term2 + term3)./ (1 .+ (omega_scal .* timeStep ./ (2 * gamma)) .^ 2)

    # Step 6: The particle is accelerated another half a time step with the electric field E only
    u = u2 + (qArray .* timeStep ./ (2 * mArray)) .* myParticle.localE

    myParticle.VelArray_old = myParticle.VelArray
    myParticle.VelArray = u

    myParticle.PosArray_old = myParticle.PosArray
    myParticle.PosArray = myParticle.PosArray .+ timeStep.*u
    
end

#r = 0 is not treated since it is within a cell, at r=max electrons/ions are absorbed, neutral is reflected
#at z boundaries any particle is removed
function reflectiveBC_kernel(PosArray,PosArray_old, VelArray, VelArray_old, alive, r_MAX, z_MAX, qArray, ioncounter, electroncounter, localNn, localNe, localNi, localCn, localCe, localCi, rCellNum, zCellNum, localE, BArray)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
     # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            removeflag = false
            r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            z = PosArray[3,idx]
            q = qArray[idx]

            if z > z_MAX[1] || z < 0.0
                #remove anyways
                if q > 0
                    #remove ions
                    @CUDA.atomic ioncounter[1] +=1
                else
                    #remove electron
                    @CUDA.atomic electroncounter[1] +=1
                end
                removeflag = true
                
            elseif r > r_MAX[1]
                    @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                    @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                    @inbounds PosArray[3,idx] = PosArray_old[3,idx]
                    @inbounds VelArray[1,idx] *= -1
                    @inbounds VelArray[2,idx] *= -1
                    @inbounds VelArray_old[1,idx] *= -1
                    @inbounds VelArray_old[2,idx] *= -1
            end
            if removeflag == true
                @inbounds alive[idx] = 0
                @inbounds localNn[idx] = 0.0
                @inbounds localNe[idx] = 0.0
                @inbounds localNi[idx] = 0.0
                @inbounds localCn[idx] = 0.0
                @inbounds localCe[idx] = 0.0
                @inbounds localCi[idx] = 0.0
                @inbounds rCellNum[idx] = 0.0
                @inbounds zCellNum[idx] = 0.0
                @inbounds localE[1,idx] = 0.0
                @inbounds localE[2,idx] = 0.0
                @inbounds localE[3,idx] = 0.0
                @inbounds BArray[1,idx] = 0.0
                @inbounds BArray[2,idx] = 0.0
                @inbounds BArray[3,idx] = 0.0
                @inbounds rCellNum[idx] = 0.0
                @inbounds zCellNum[idx] = 0.0
            end
        end
    end
    return nothing
end

function reflectiveBC(myParticle, myDomain, ioncounter, electroncounter)

    @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    reflectiveBC_kernel(myParticle.PosArray,myParticle.PosArray_old, myParticle.VelArray, myParticle.VelArray_old, myParticle.alive, myDomain.r_MAX, myDomain.z_MAX, myParticle.qArray, ioncounter, electroncounter, myParticle.localNn, myParticle.localNe, myParticle.localNi, myParticle.localCn, myParticle.localCe, myParticle.localCi, myParticle.rCellNum, myParticle.zCellNum, myParticle.localE, myParticle.BArray)
    )

end

#r = 0 is not treated since it is within a cell, at r=max electrons/ions are absorbed, neutral is reflected
#at z boundaries any particle is removed
function AllreflectiveBC_kernel(PosArray,PosArray_old, VelArray, VelArray_old, alive, r_MAX, z_MAX, qArray, ioncounter, electroncounter)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
     # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            z = PosArray[3,idx]
            q = qArray[idx]

            if z > z_MAX[1] || z < 0.0
                if z > z_MAX[1]
                #periodic
                    @inbounds PosArray_old[3,idx] = PosArray[3,idx]
                    @inbounds PosArray[3,idx] = z_MAX[1] - (PosArray[3,idx] - z_MAX[1]);
                else
                    @inbounds PosArray_old[3,idx] = PosArray[3,idx]
                    @inbounds PosArray[3,idx] *= -1
                end

            elseif r > r_MAX[1]
                #=
                PosArray[1,idx] = PosArray_old[1,idx]
                PosArray[2,idx] = PosArray_old[2,idx]
                x, y, vx, vy = reflect_boundary!(PosArray[1,idx], PosArray[2,idx], VelArray[1,idx], VelArray[2,idx], r_MAX[1])
                VelArray[1,idx] = vx
                VelArray[2,idx] = vy
                =#
                
                @inbounds PosArray_old[1,idx] = PosArray[1,idx]
                @inbounds PosArray_old[2,idx] = PosArray[2,idx]
                @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                x, y, vx, vy = reflect_boundary!(PosArray[1,idx], PosArray[2,idx], VelArray[1,idx], VelArray[2,idx], r_MAX[1])
                PosArray[1,idx] = x
                PosArray[2,idx] = y
                VelArray[1,idx] = vx
                VelArray[2,idx] = vy
                
            end
        end
    end
    return nothing
end

function reflectVel(vx::Float64, vy::Float64, nx::Float64, ny::Float64)
    # Normalize the normal vector to make it a unit vector
    norm_n = sqrt(nx^2 + ny^2)
    nx /= norm_n
    ny /= norm_n

    # Velocity vector dot normal vector
    dot_product = vx * nx + vy * ny

    # Reflected velocity formula
    vx_ref = vx - 2 * dot_product * nx
    vy_ref = vy - 2 * dot_product * ny

    return vx_ref, vy_ref
end

function reflect_boundary!(x::Float64, y::Float64, vx::Float64, vy::Float64, r_max::Float64)
    # Calculate radial position r
    r = sqrt(x^2 + y^2)

    # If the particle is beyond the outer wall
    if r > r_max
        # Calculate the radial and angular components of velocity
        vr = (x * vx + y * vy) / r  # Radial velocity
        vtheta = (x * vy - y * vx) / r  # Angular velocity

        # Reflect the radial velocity (invert its sign)
        vr = -vr

        # Convert back to Cartesian coordinates
        vx = (vr * x / r) - (vtheta * y / r)
        vy = (vr * y / r) + (vtheta * x / r)

        # Reposition the particle to the boundary (r = r_max)
        scaling_factor = r_max / r * 0.9999999
        x *= scaling_factor
        y *= scaling_factor
    end

    return x, y, vx, vy
end

function polar_to_cartesian_velocity(vr::Float64, vtheta::Float64, theta::Float64)
    # Compute the Cartesian velocity components v_x and v_y
    vx = vr * cos(theta) - vtheta * sin(theta)
    vy = vr * sin(theta) + vtheta * cos(theta)
    
    return vx, vy
end

function cartesian_to_polar_velocity(vx::Float64, vy::Float64)
    # Compute the polar angle theta
    θ = atan(vy, vx)
    
    # Compute the radial velocity v_r
    v_r = vx * cos(θ) + vy * sin(θ)
    
    # Compute the angular velocity v_θ
    v_θ = -vx * sin(θ) + vy * cos(θ)
    
    return v_r, v_θ, θ
end

function reflectVelTrig(vx::Float64, vy::Float64)
    v_r, v_θ, θ = cartesian_to_polar_velocity(vx, vy)
    v_r *= -1
    return polar_to_cartesian_velocity(v_r, v_θ, θ)
end

function AllreflectiveBC(myParticle, myDomain)
    electroncounter = CUDA.zeros(Int, 1)
    ioncounter = CUDA.zeros(Int, 1)

    @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    AllreflectiveBC_kernel(myParticle.PosArray, myParticle.PosArray_old, myParticle.VelArray, myParticle.VelArray_old, myParticle.alive, myDomain.r_MAX, myDomain.z_MAX, myParticle.qArray, ioncounter, electroncounter))

end

function absorbBE_reflectPlasma_BC_kernel(PosArray,PosArray_old, VelArray, VelArray_old, alive, r_MAX, z_MAX, qArray, ioncounter, electroncounter, isBeam)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
     # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            z = PosArray[3,idx]
            q = qArray[idx]

            if z >= z_MAX[1]
                alive[idx] = 0
                isBeam[idx] = 0

                if qArray[idx] < 0 
                    @CUDA.atomic electroncounter[1] += 1
                elseif qArray[idx] > 0 
                    @CUDA.atomic ioncounter[1] += 1
                end

            elseif z <= 0.0
                if isBeam[idx] == 1
                    alive[idx] = 0
                    isBeam[idx] = 0
                    @CUDA.atomic electroncounter[1] += 1
                else
                    @inbounds PosArray[3,idx] *= -1
                    @inbounds VelArray_old[3,idx] = VelArray[3,idx]
                    @inbounds VelArray[3,idx] *= -1
                end
            elseif r >= r_MAX[1]
                if isBeam[idx] == 1
                    alive[idx] = 0
                    isBeam[idx] = 0
                    @CUDA.atomic electroncounter[1] += 1
                else
                    if q == 0.0
                        #reflect a neutral
                        @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                        @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                        @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                        @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                        @inbounds VelArray[1,idx] *= -1
                        @inbounds VelArray[2,idx] *= -1
                    elseif q > 0.0
                        #reflect an ion
                        @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                        @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                        @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                        @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                        @inbounds VelArray[1,idx] *= -1
                        @inbounds VelArray[2,idx] *= -1
                    elseif q < 0.0
                        #reflect an electron
                        @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                        @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                        @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                        @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                        @inbounds VelArray[1,idx] *= -1
                        @inbounds VelArray[2,idx] *= -1
                    end
                end
            end

        end
    end



    return nothing
end

function absorbBE_reflectPlasma_BC(myParticle, myDomain)
    electroncounter = CUDA.zeros(Int, 1)
    ioncounter = CUDA.zeros(Int, 1)

    @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    absorbBE_reflectPlasma_BC_kernel(myParticle.PosArray, myParticle.PosArray_old, myParticle.VelArray, myParticle.VelArray_old, myParticle.alive, myDomain.r_MAX, myDomain.z_MAX, myParticle.qArray, ioncounter, electroncounter, myParticle.isBeam))
    return ioncounter, electroncounter
end

function absorbBE_reflectPlasma_BC_countThrust(myParticle, myDomain)
    #This BC is absoprtion for beam electrons at all boundaries, and reflective for plasma particles except RHS wall
    electroncounter = CUDA.zeros(Int, 1)
    ioncounter = CUDA.zeros(Int, 1)
    myDomain.momentumOut .= 0.0
    @cuda(
    threads=256,
    blocks=cld(length(myParticle.alive),256),
    absorbBE_reflectPlasma_BC_Thrust_kernel(myParticle.PosArray, myParticle.PosArray_old, myParticle.VelArray, myParticle.VelArray_old, myParticle.alive, myDomain.r_MAX, myDomain.z_MAX, myParticle.qArray, ioncounter, electroncounter, myParticle.isBeam, myDomain.momentumOut, myParticle.mArray, myParticle.superParticleSizeArray))
    return ioncounter, electroncounter
end

function absorbBE_reflectPlasma_BC_Thrust_kernel(PosArray,PosArray_old, VelArray, VelArray_old, alive, r_MAX, z_MAX, qArray, ioncounter, electroncounter, isBeam, momentumOut, mArray,superParticleSizeArray)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
     # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            r = sqrt(PosArray[1,idx]^2 + PosArray[2,idx]^2)
            z = PosArray[3,idx]
            q = qArray[idx]

            if z >= z_MAX[1]
                @inbounds alive[idx] = 0
                @inbounds isBeam[idx] = 0

                if qArray[idx] < 0 
                    @CUDA.atomic electroncounter[1] += 1
                elseif qArray[idx] > 0 
                    @CUDA.atomic ioncounter[1] += 1
                    @CUDA.atomic momentumOut[1] += mArray[idx]*VelArray[3,idx]*superParticleSizeArray[idx]
                end

            elseif z <= 0.0
                if isBeam[idx] == 1
                    @inbounds alive[idx] = 0
                    @inbounds isBeam[idx] = 0
                    @CUDA.atomic electroncounter[1] += 1
                else
                    @inbounds PosArray[3,idx] *= -1
                    @inbounds VelArray_old[3,idx] = VelArray[3,idx]
                    @inbounds VelArray[3,idx] *= -1
                end
                if qArray[idx] > 0 
                    @CUDA.atomic momentumOut[1] += -mArray[idx]*VelArray[3,idx]*superParticleSizeArray[idx]
                end
            elseif r >= r_MAX[1]
                if isBeam[idx] == 1
                    @inbounds alive[idx] = 0
                    @inbounds isBeam[idx] = 0
                    @CUDA.atomic electroncounter[1] += 1
                else
                    if q == 0.0
                        #reflect a neutral
                        @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                        @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                        @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                        @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                        @inbounds VelArray[1,idx] *= -1
                        @inbounds VelArray[2,idx] *= -1
                    elseif q > 0.0
                        #reflect an ion
                        @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                        @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                        @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                        @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                        @inbounds VelArray[1,idx] *= -1
                        @inbounds VelArray[2,idx] *= -1
                    elseif q < 0.0
                        #reflect an electron
                        @inbounds PosArray[1,idx] = PosArray_old[1,idx]
                        @inbounds PosArray[2,idx] = PosArray_old[2,idx]
                        @inbounds VelArray_old[1,idx] = VelArray[1,idx]
                        @inbounds VelArray_old[2,idx] = VelArray[2,idx]
                        @inbounds VelArray[1,idx] *= -1
                        @inbounds VelArray[2,idx] *= -1
                    end
                end
            end

        end
    end



    return nothing
end