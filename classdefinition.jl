#define class
#in this version of code, all functions instantiate or operate class
mutable struct Particle #corresponds to ParticleArrayName
    xArray::Vector{Float64}
    yArray::Vector{Float64}
    zArray::Vector{Float64}
    VxArray::Vector{Float64}
    VyArray::Vector{Float64}
    VzArray::Vector{Float64}
    xArray_old::Vector{Float64}
    yArray_old::Vector{Float64}
    zArray_old::Vector{Float64}
    VxArray_old::Vector{Float64}
    VyArray_old::Vector{Float64}
    VzArray_old::Vector{Float64}
    ExArray::Vector{Float64}
    EyArray::Vector{Float64}
    EzArray::Vector{Float64}
    BxArray::Vector{Float64}
    ByArray::Vector{Float64}
    BzArray::Vector{Float64}
    mArray::Vector{Float64}
    qArray::Vector{Float64}
    superParticleSizeArray::Vector{Float64}
    vdwrArray::Vector{Float64}
    localNn::Vector{Float64}
    localNe::Vector{Float64}
    localNi::Vector{Float64}
    localCn::Vector{Float64}
    localCe::Vector{Float64}
    localCi::Vector{Float64}
    cellNumber::Vector{Int}

    function Particle()
        new(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], 
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], 
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], 
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
            Float64[], Float64[], Float64[], Float64[],
            Int[])
    end
end

mutable struct RectangularDomain
    N::Int
    X_MAX::Float64
    Y_MAX::Float64
    Z_MAX::Float64
    Area::Matrix{Float64}
    dx::Matrix{Float64}
    chagre_bin::Matrix{Float64}
    neArray::Matrix{Float64}
    uexArray::Matrix{Float64}
    ueyArray::Matrix{Float64}
    uezArray::Matrix{Float64}
    niArray::Matrix{Float64}
    uixArray::Matrix{Float64}
    phi::Matrix{Float64}
    nnArray::Matrix{Float64}
    CeArray::Matrix{Float64}
    CiArray::Matrix{Float64}
    CnArray::Matrix{Float64}
    BxBinArray::Matrix{Float64}
    ByBinArray::Matrix{Float64}
    BzBinArray::Matrix{Float64}
    neutral_mass_store::Matrix{Float64}
    
    neArray_output::Matrix{Float64}
    uexArray_output::Matrix{Float64}
    ueyArray_output::Matrix{Float64}
    uezArray_output::Matrix{Float64}
    niArray_output::Matrix{Float64}
    uixArray_output::Matrix{Float64}
    phi_output::Matrix{Float64}
    nnArray_output::Matrix{Float64}
    CeArray_output::Matrix{Float64}
    CiArray_output::Matrix{Float64}
    CnArray_output::Matrix{Float64}
    BxBinArray_output::Matrix{Float64}
    ByBinArray_output::Matrix{Float64}
    BzBinArray_output::Matrix{Float64}
    output_counter::Int

    V_first::Float64
    V_last::Float64

    A::Matrix{Float64}

    electric_constant::Float64

    function RectangularDomain()
        new(0, 0.0, 0.0, 0.0,  Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 
        Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 
        Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 
        Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 
        Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0), 0, 0.0, 0.0, Array{Float64}(undef, 0, 0), 0.0)

    end
end

mutable struct MagneticField
    regionXMin::Vector{Float64}
    regionXMax::Vector{Float64}
    regionYMin::Vector{Float64}
    regionYMax::Vector{Float64}
    regionZMin::Vector{Float64}
    regionZMax::Vector{Float64}
    regionBx::Vector{Float64}
    regionBy::Vector{Float64}
    regionBz::Vector{Float64}
    gaussian_profile_flag::Int
    store_B_max::Float64
    store_B_max_x::Float64
    store_B_half_x::Float64
    function MagneticField()
        new(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], 0, 0.0, 0.0, 0.0)

    end


end