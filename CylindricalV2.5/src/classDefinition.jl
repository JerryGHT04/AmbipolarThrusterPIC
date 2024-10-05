using CUDA
using CUDA.CUSPARSE
using SparseArrays
# define particle class using CUArray
mutable struct Particle_GPU
    PosArray::CuArray{Float64,2} #x, y, z each row, each column represents different time
    VelArray::CuArray{Float64,2} #Vx, Vy, Vz each row, each column represents different time
    PosArray_old::CuArray{Float64,2} 
    VelArray_old::CuArray{Float64,2} 
    localE::CuArray{Float64,2} #Er, Ez each row, each column represents different time
    BArray::CuArray{Float64,2} #Bx, By, Bz each row, each column represents different time
    mArray::CuArray{Float64}
    qArray::CuArray{Float64}
    superParticleSizeArray::CuArray{Float64}
    vdwrArray::CuArray{Float64} #van_der_waals_radius
    localNn::CuArray{Float64} #Number of neutrals this superparticle represents
    localNe::CuArray{Float64}
    localNi::CuArray{Float64}
    localCn::CuArray{Float64}
    localCe::CuArray{Float64}
    localCi::CuArray{Float64}
    rCellNum::CuArray{Float64}
    zCellNum::CuArray{Float64}
    alive::CuArray{Int32} # element set to 1 if the particle is looped
    isBeam::CuArray{Int32}
    localColor::CuArray{Int32}


    #pairMatrix: stores 0 or 1. 1 represents this particle is being paired.
    #eg. if pairMatrix[1,4] == 1, it means the first particle is paired with the forth particle
    #collision will occur if pairMatrix[1,4] = pairMatrix[4,1] == 1.
    #After collision, both of the position are reset to 0
    #pairMatrix::SparseMatrixCSC{Int32, Int32}
    #numPairs::CuArray{Int32}

    function Particle_GPU(N_max::Int)
        new(CUDA.zeros(3, N_max), CUDA.zeros(3, N_max), CUDA.zeros(3, N_max),
            CUDA.zeros(3, N_max), CUDA.zeros(3, N_max), CUDA.zeros(3, N_max),
            CUDA.zeros(N_max), CUDA.zeros(N_max), CUDA.zeros(N_max),
            CUDA.zeros(N_max), CUDA.zeros(N_max), CUDA.zeros(N_max),
            CUDA.zeros(N_max), CUDA.zeros(N_max), CUDA.zeros(N_max),
            CUDA.zeros(N_max), CUDA.zeros(N_max), CUDA.zeros(N_max),
            CUDA.fill(0, N_max),CUDA.fill(0, N_max),CUDA.fill(0, N_max))
    end
end

mutable struct RZDomain_GPU #Axisymmetric cylindrical coordinate
    Nr::CuArray{Int32, 1}
    Nz::CuArray{Int32, 1}
    r_MAX::CuArray{Float64, 1}
    z_MAX::CuArray{Float64, 1}
    dr::CuArray{Float64, 1}
    dz::CuArray{Float64, 1}
    charge_bin::CuArray{Float64, 2}
    phi::CuArray{Float64, 2}
    neArray::CuArray{Float64, 2}
    niArray::CuArray{Float64, 2}
    nnArray::CuArray{Float64, 2}
    CeArray::CuArray{Float64, 2}
    CiArray::CuArray{Float64, 2}
    CnArray::CuArray{Float64, 2}
    nodeVolume::CuArray{Float64, 2}
    rArray::CuArray{Float64,1}
    BxBinArray::CuArray{Float64, 2}
    ByBinArray::CuArray{Float64, 2}
    BzBinArray::CuArray{Float64, 2}
    uixArray::CuArray{Float64, 2}
    uiyArray::CuArray{Float64, 2}
    uizArray::CuArray{Float64, 2}
    uexArray::CuArray{Float64, 2}
    ueyArray::CuArray{Float64, 2}
    uezArray::CuArray{Float64, 2}
    A::CuSparseMatrixCSR{}
    ErArray::CuArray{Float64, 2}
    EzArray::CuArray{Float64, 2}
    neutral_change::CuArray{Float64, 2}
    mean_neutralSuperSize::CuArray{Float64, 2}
    momentumOut::CuArray{Float64, 1}
    color::CuArray{Int32, 2}

    function RZDomain_GPU(Nr,Nz,r_max,z_max,BCType)
        dr = r_max/(Nr-1)
        dz = z_max/(Nz-1)
        rArray = collect(LinRange(0,r_max,Nr))
        #println("rArray = ", rArray)
        zArray = collect(LinRange(0,z_max,Nz))
        #println("dr = ", dr)
        #println("dz = ", dz)
        #Calculate cell volume
        nodeVolume = zeros(Float64, Nr, Nz)

        
        for i = 1:Nr
            for j = 1:Nz
                if i == 1
                    if (j == 1 || j == Nz)
                        nodeVolume[i, j] = 1/2*dz*π*((rArray[i]+dr/2)^2 - (rArray[i])^2)
                    else
                        nodeVolume[i, j] = dz*π*((rArray[i]+dr/2)^2 - (rArray[i])^2)
                    end
                elseif i == Nr
                    if (j == 1 || j == Nz)
                        nodeVolume[i, j] = 1/2*dz*π*((rArray[i])^2 - (rArray[i]-dr/2)^2)
                    else
                        nodeVolume[i, j] = dz*π*((rArray[i])^2 - (rArray[i]-dr/2)^2)
                    end
                else
                    if (j == 1 || j == Nz)
                        nodeVolume[i, j] = 1/2*dz*π*((rArray[i]+dr/2)^2 - (rArray[i]-dr/2)^2)
                    else
                        nodeVolume[i, j] = dz*π*((rArray[i]+dr/2)^2 - (rArray[i]-dr/2)^2)
                    end
                end
            end
        end
        
        d_nodeVolume = CuArray(nodeVolume)
        d_rArray = CuArray(rArray)

        N = Nr*Nz
        A = zeros(N,N)
        #Create A matrix for solving Poisson's equation
        # Fill the matrix A and vector B
        if BCType == "Neumann"
            for i in 1:Nr
                for j in 1:Nz
                    idx = (i-1)*Nz + j  # Linear index in the sparse matrix
                    
                    if i == 1  
                        # Neumann BC at r = 0
                        A[idx, idx] = -1
                        A[idx, idx + Nz] = 1
                    
                    elseif i == Nr
                        #Dirichlet BC at r = rmax
                        A[idx, idx] = 1;
                        
                    elseif j == 1  # Neumann BC at z = 0
                        A[idx, idx] = -1
                        A[idx, idx+1] = 1
                    
                    elseif j == Nz  # Neumann BC at z = zMax
                        A[idx, idx] = 1 
                        A[idx, idx-1] = -1
                    
                    else  # Interior points
                        A[idx, idx] = -2 / dr^2 - 2 / dz^2
                        A[idx, idx - 1] = 1 / dz^2
                        A[idx, idx + 1] = 1 / dz^2
                        A[idx, idx - Nz] = 1 / dr^2 - 1 / (2 * dr * rArray[i])
                        A[idx, idx + Nz] = 1 / dr^2 + 1 / (2 * dr * rArray[i])
                    end
                end
            end
        elseif BCType == "Dirichlet"
            # Set Dirichlet BC at z = 0, everything else is the same
            for i in 1:Nr
                for j in 1:Nz
                    idx = (i - 1) * Nz + j  # Linear index in the sparse matrix
                    
                    if i == 1  
                        # Neumann BC at r = 0
                        A[idx, idx] = -1
                        A[idx, idx + Nz] = 1
                    
                    elseif i == Nr
                        # Dirichlet BC at r = rmax
                        A[idx, idx] = 1
                        
                    elseif j == 1  # Dirichlet BC at z = 0
                        A[idx, idx] = 1  # Apply Dirichlet condition
                        
                    elseif j == Nz 
                        # Dirichlet BC at z = zMax
                        A[idx, idx] = 1

                        #=
                        # Neumann BC at z = zMax
                        A[idx, idx] = 1 
                        A[idx, idx - 1] = -1
                        =#
                    else  # Interior points
                        A[idx, idx] = -2 / dr^2 - 2 / dz^2
                        A[idx, idx - 1] = 1 / dz^2
                        A[idx, idx + 1] = 1 / dz^2
                        A[idx, idx - Nz] = 1 / dr^2 - 1 / (2 * dr * rArray[i])
                        A[idx, idx + Nz] = 1 / dr^2 + 1 / (2 * dr * rArray[i])
                    end
                end
            end
        end
        A_sparse = sparse(A)
        d_A = CuSparseMatrixCSR(A_sparse);

        #assign colors
        color = zeros(Nr,Nz)
        num_colors = 4
        for i in 1:Nr
            for j in 1:Nz
                 # Adjust indices to start from 0
                ii = i - 1
                jj = j - 1

                # Coloring formula to ensure no adjacent cells share the same color
                # Including edge and diagonal adjacency
                color[i,j] = mod( (mod(ii, 2) * 4) +   # Based on i coordinate parity
                            (mod(jj, 2) * 2) +   # Based on j coordinate parity
                            mod(ii + jj, 2),     # Combined parity of i and j
                            num_colors ) + 1     # Ensure color indices start from 1
            end
        end
        d_color = CuArray(color);

        new(CuArray([Nr]), CuArray([Nz]), CuArray([r_max]),
        CuArray([z_max]), CuArray([dr]), CuArray([dz]),
        CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), 
        CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
         CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), d_nodeVolume,
        d_rArray, CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz), d_A, CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz),CUDA.zeros(Nr,Nz), CuArray([0]), d_color)
    end
end


mutable struct MagneticField_GPU
    regionXRange::CuArray{Float64,1}
    regionYRange::CuArray{Float64,1}
    regionZRange::CuArray{Float64,1}
    regionB::CuArray{Float64,1} #contains Bx, By, Bz
    gaussian_profile_flag::CuArray{Int32,1}
    store_B_max::CuArray{Float64,1}
    store_B_max_x::CuArray{Float64,1}
    store_B_half_x::CuArray{Float64,1}

    function MagneticField_GPU(xrange, yrange, zrange, B)
        new(CuArray(xrange), CuArray(yrange), CuArray(zrange), 
        CuArray(B), CuArray([0]), CuArray([0.0]),
        CuArray([0.0]), CuArray([0.0]))
    end
end

mutable struct DataLogger
    phi_output::CuArray{Float64, 2}
    neArray_output::CuArray{Float64, 2}
    niArray_output::CuArray{Float64, 2}
    nnArray_output::CuArray{Float64, 2}
    uixArray_output::CuArray{Float64, 2}
    uiyArray_output::CuArray{Float64, 2}
    uizArray_output::CuArray{Float64, 2}
    uexArray_output::CuArray{Float64, 2}
    ueyArray_output::CuArray{Float64, 2}
    uezArray_output::CuArray{Float64, 2}
    energy::CuArray{Float64, 1}
    energy_output::CuArray{Float64, 1}
    thrust::CuArray{Float64, 1}

    function DataLogger(Nr, Nz)
        new(CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz), CUDA.zeros(Nr,Nz),
        CUDA.zeros(Nr,Nz), CuArray([0]),CuArray([0]),CuArray([0]))
    end
end