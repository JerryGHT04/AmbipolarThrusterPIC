using LinearAlgebra
include("Poisson1D.jl")
include("classdefinition.jl")


function createRectangularDomain_jl(MyRectangularDomain::RectangularDomain, set_x_max::Float64, set_y_max::Float64, set_z_max::Float64, V_at_x_min::Float64, V_at_x_max::Float64, N_cells_x::Int)
    #Set up the global memory structure in Julia for the new 1D rectangular domain
    MyRectangularDomain.N = N_cells_x
    MyRectangularDomain.X_MAX = set_x_max
    MyRectangularDomain.Y_MAX = set_y_max
    MyRectangularDomain.Z_MAX = set_z_max
    MyRectangularDomain.Area = zeros(Float64, ( 1 ,N_cells_x + 1 )) .+ (MyRectangularDomain.Y_MAX * MyRectangularDomain.Z_MAX);
    MyRectangularDomain.dx = zeros(Float64, ( 1, N_cells_x - 1 )) .+ (MyRectangularDomain.X_MAX / MyRectangularDomain.N);
    MyRectangularDomain.chagre_bin = zeros(Float64, ( 1,N_cells_x));

    #Initialize the simulation arrays
    MyRectangularDomain.neArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.uexArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.ueyArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.uezArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.niArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.uixArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.phi = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.nnArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.CeArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.CiArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.CnArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.BxBinArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.ByBinArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.BzBinArray = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.neutral_mass_store = zeros(Float64, ( 1,N_cells_x));

    #initialize the simulation output arrays
    MyRectangularDomain.neArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.uexArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.ueyArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.uezArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.niArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.uixArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.phi_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.nnArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.CeArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.CiArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.CnArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.BxBinArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.ByBinArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.BzBinArray_output = zeros(Float64, ( 1,N_cells_x));
    MyRectangularDomain.output_counter = 0;

    #Set the voltage of the anode and cathode
    MyRectangularDomain.V_first = V_at_x_min
    MyRectangularDomain.V_last = V_at_x_max

    #Define the governing poisson equation
    DL = zeros(Float64, (1, N_cells_x-1)) .- (MyRectangularDomain.Area[1]/MyRectangularDomain.dx[1]);
    DL[N_cells_x - 1] = 0.0;
    DU = zeros(Float64, (1, N_cells_x-1)) .- (MyRectangularDomain.Area[1]/MyRectangularDomain.dx[1]);
    DU[1] = 0.0;
    DD = zeros(Float64, (1, N_cells_x)) .+ (2.0*MyRectangularDomain.Area[1]/MyRectangularDomain.dx[1]);
    DD[1] = 1.0;
    DD[N_cells_x] = 1.0;
    MyRectangularDomain.A = Tridiagonal(vec(DL), vec(DD), vec(DU));

    #Set the value of the electric constant
    MyRectangularDomain.electric_constant = 8.85418782e-12;
end

function particleCountRectangular_jl(myParticle::Particle, myDomain::RectangularDomain)
    par1 = myParticle.xArray
    par2 = myParticle.VxArray
    par3 = myParticle.VyArray
    par4 = myParticle.VzArray
    par5 = myParticle.qArray
    par6 = myParticle.localNn
    par7 = myParticle.localNe
    par8 = myParticle.localNi
    par9 = myParticle.localCn
    par10 = myParticle.localCe
    par11 = myParticle.localCi
    par12 = myParticle.BxArray
    par13 = myParticle.ByArray
    par14 = myParticle.BzArray
    par15 = myParticle.superParticleSizeArray
    par16 = myParticle.cellNumber

    par17 = myDomain.chagre_bin
    par18 = myDomain.neArray
    par19 = myDomain.uexArray
    par20 = myDomain.ueyArray
    par21 = myDomain.uezArray
    par22 = myDomain.niArray
    par23 = myDomain.uixArray
    par24 = myDomain.nnArray
    par25 = myDomain.CnArray
    par26 = myDomain.CeArray
    par27 = myDomain.CiArray
    par28 = myDomain.BxBinArray
    par29 = myDomain.ByBinArray
    par30 = myDomain.BzBinArray
    
    par31 = myDomain.neArray_output
    par32 = myDomain.uexArray_output
    par33 = myDomain.ueyArray_output
    par34 = myDomain.uezArray_output
    par35 = myDomain.niArray_output
    par36 = myDomain.uixArray_output
    par37 = myDomain.nnArray_output
    par38 = myDomain.CnArray_output
    par39 = myDomain.CeArray_output
    par40 = myDomain.CiArray_output
    par41 = myDomain.BxBinArray_output
    par42 = myDomain.ByBinArray_output
    par43 = myDomain.BzBinArray_output

    par44 = myDomain.Area
    par45 = myDomain.dx
    par46 = myDomain.N
    par47 = myDomain.X_MAX
    par48 = myDomain.output_counter
    
    particleCountRectangular(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11,par12,par13,par14,par15,par16,par17,par18,par19,par20,par21,par22,par23,par24,par25,par26,
    par27,par28,par29,par30,par31,par32,par33,par34,par35,par36,par37,par38,par39,par40,par41,par42,par43,par44,par45,par46,par47,par48)

end


function solvePoissonEquation_jl(myDomain::RectangularDomain)
    par1= myDomain.N
    par2 = myDomain.chagre_bin
    par3 = myDomain.V_first
    par4 = myDomain.V_last
    par5 = myDomain.A
    par6 = myDomain.phi
    par7 = myDomain.phi_output
    par8 = myDomain.electric_constant
    
    solvePoissonEquation(
    par1,par2,par3,par4,par5,par6,par7,par8)
end

function electricFieldRectangular_jl(myParticle::Particle, myDomain::RectangularDomain)
    par1 = myParticle.xArray
    par2 = myParticle.ExArray
    par3 = myParticle.EyArray
    par4 = myParticle.EzArray
    par5 = myDomain.X_MAX
    par6 = myDomain.phi
    par7 = myDomain.dx
    par8 = myDomain.N
    electricFieldRectangular(par1, par2, par3, par4, par5, par6, par7, par8)
end

function boundaryConditionsRectangularNoWallLoss_jl(myParticle::Particle, myDomain::RectangularDomain)
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

    par21 = myParticle.vdwrArray
    par22 = myParticle.localNn
    par23 = myParticle.localNe
    par24 = myParticle.localNi
    par25 = myParticle.localCn
    par26 = myParticle.localCe
    par27 = myParticle.localCi
    par28 = myParticle.cellNumber
    par29 = myParticle.superParticleSizeArray
    par30 = myDomain.X_MAX
    par31 = myDomain.Y_MAX
    par32 = myDomain.Z_MAX
    
    return boundaryConditionsRectangularNoWallLoss(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14,
    par15, par16, par17, par18, par19, par20, par21, par22, par23, par24, par25, par26, par27, par28, par29, par30, par31, par32)

end

function writeToFile_jl(myDomain::RectangularDomain, time_stamp::Float64, path::String) 
    par1 = myDomain.phi
    par2 = myDomain.phi_output

    par3 = myDomain.neArray
    par4 = myDomain.uexArray
    par5 = myDomain.ueyArray
    par6 = myDomain.uezArray
    par7 = myDomain.niArray
    par8 = myDomain.uixArray
    par9 = myDomain.nnArray
    par10 = myDomain.CnArray
    par11 = myDomain.CeArray
    par12 = myDomain.CiArray
    par13 = myDomain.BxBinArray
    par14 = myDomain.ByBinArray
    par15 = myDomain.BzBinArray

    par16 = myDomain.neArray_output
    par17 = myDomain.uexArray_output
    par18 = myDomain.ueyArray_output
    par19 = myDomain.uezArray_output
    par20 = myDomain.niArray_output
    par21 = myDomain.uixArray_output
    par22 = myDomain.nnArray_output
    par23 = myDomain.CnArray_output
    par24 = myDomain.CeArray_output
    par25 = myDomain.CiArray_output
    par26 = myDomain.BxBinArray_output
    par27 = myDomain.ByBinArray_output
    par28 = myDomain.BzBinArray_output

    par29 = myDomain.N
    par30 = myDomain.output_counter
    par31 = time_stamp
    par32 = path
    writeToFile(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14, par15, par16, par17, par18, par19, par20, par21, par22, par23, par24, par25, par26, par27, par28, par29, par30, par31, par32)
end