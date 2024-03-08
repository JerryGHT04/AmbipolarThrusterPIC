include("StaticMagneticField.jl")
include("classdefinition.jl")
function initializeMagneticField_jl()
    #Set up the global memory structure in Julia for the new magnetic field
    a = MagneticField(
        Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
        Float64[], Float64[], Float64[], 0, 0.0, 0.0, 0.0
    )
    return a
end

function createRectangularRegion_jl(MyMagneticField::MagneticField, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64, z_min::Float64, z_max::Float64, Bx::Float64, By::Float64, Bz::Float64)
    #Call the create rectangular region function in StaticMagneticField.jl
    createRectangularRegion(x_min, x_max, y_min, y_max, z_min, z_max, Bx, By, Bz, 
    MyMagneticField.regionXMin, MyMagneticField.regionXMax, MyMagneticField.regionYMin, MyMagneticField.regionYMax, MyMagneticField.regionZMin, MyMagneticField.regionZMax,
    MyMagneticField.regionBx, MyMagneticField.regionBy, MyMagneticField.regionBz
    )
end

function setMagneticField_jl(MyMagneticField::MagneticField, myParticle::Particle)
    par1 = myParticle.xArray
    par2 = myParticle.yArray
    par3 = myParticle.zArray
    par4 = myParticle.BxArray
    par5 = myParticle.ByArray
    par6 = myParticle.BzArray
    par7 = MyMagneticField.regionXMin
    par8 = MyMagneticField.regionXMax
    par9 = MyMagneticField.regionYMin
    par10 = MyMagneticField.regionYMax
    par11 = MyMagneticField.regionZMin
    par12 = MyMagneticField.regionZMax
    par13 = MyMagneticField.regionBx
    par14 = MyMagneticField.regionBy
    par15 = MyMagneticField.regionBz
    par16 = MyMagneticField.gaussian_profile_flag
    par17 = MyMagneticField.store_B_max
    par18 = MyMagneticField.store_B_max_x
    par19 = MyMagneticField.store_B_half_x
    setMagneticField(par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12, par13, par14, par15, par16, par17, par18, par19)
end