function verifyParams(BeamEnergy, BeamCurrent, rb,ne, Telectron, timeStep, Nz, dZ, Bz)
    #Define comstants
    me =  9.10938356E-31;
    qe =  1.6021766208e-19;
    eps0 = 8.85418782e-12;
    mi = 2.1802e-25 # Xenon atom
    kB = 1.380649e-23;
    c = 3e8;
    Ee = BeamEnergy*qe
    Ie = BeamCurrent

    gamma = 1 + Ee*qe/(me*c^2)
    if gamma > 1.2
        Ue = sqrt(1-gamma^(-2))*c
    else
        Ue = sqrt(2*Ee/me)
    end
    
    Ne_dot_max = Ie/qe
    #println(Ue)
    flag = true

    #1. verify if magnatic field is big enough
    neb =  Ie/(qe*pi*rb^2*Ue)
    #println(neb)
    magFactor = 2;
    B0 = sqrt(magFactor*8*pi*neb*me*c^2*gamma)
    if Bz < B0
        flag = false
        error("Bz too small, expected "*string(B0))
        return flag
    end
    
    #2. verify if time step is small enough:
    #plasma frequency of beam electron
    ωb = sqrt(qe^2*neb/(me*eps0))
    #plasma frequency of bulk electron
    ω = sqrt(qe^2*ne/(me*eps0))
    if ne == 0.0
        ω = Inf
    end
    #gyro frequency of electron
    ωgyro = qe*B0/me;
    #time step is ten times smaller than higest frequency
    #WIKIPEDIA
    dt = 0.1 * 1/min(min(ωgyro,ω), ωb)
    if timeStep > dt
        flag = false
        error("time step too big, expected "*string(dt))
        return flag
    end

    #3. verify if cell size is small enough
    #3.1 Debye length of electron beam
    Teb = 2/3*Ee/kB
    if Telectron == 0 
        Telectron = Inf
    end
    λ_D = sqrt(eps0*kB*Teb/(neb*qe^2))

    Δz = dZ / Nz
    if Δz >= 3.4*λ_D
        flag = false
        error("cell size too big, expected N = "*string(dZ / (3.4*λ_D)))
        return flag
    end
    #3.2 Using CFL condition, dt < dx/c
    #WIKIPEDIA
    if timeStep >= Δz/c
        flag = false
        error("time step too big, expected "*string(Δz/c))
        return flag
    end

    #4. cell size should be small enough that beam electron cannot pass more than one cell each step
    if Ue*dt > Δz
        flag = false
        error("cell size too big, expected "*string(Ue*dt))
        return flag
    end

    return flag
end

function createParamsFile(maxTime, timeStep, Nr, Nz, numParticle, Telectron, Tion, dR, dZ, Nmax, writeTimeStep, Bz, recover, save, BeamEnergy,
    BeamCurrent, BeamSuperParticle, rb, tr, neutralRate, boundaryCondition, output_folder, save_path, parameterName, edensity, ndensity,saveFrequency)
    io = open(parameterName, "w")

    println(io,output_folder)
    println(io,maxTime)
    println(io,timeStep)
    println(io,Nr)
    println(io,Nz)
    println(io,numParticle)
    println(io,edensity)
    println(io,ndensity)
    println(io,Telectron)
    println(io,Tion)
    println(io,dR)
    println(io,dZ)
    println(io,BeamEnergy)
    println(io,BeamCurrent)
    println(io,BeamSuperParticle)
    println(io,rb)
    println(io,tr)
    println(io,neutralRate)
    println(io,recover)
    println(io,save)
    println(io,save_path)
    println(io,Nmax)
    println(io,writeTimeStep)
    println(io,Bz)
    println(io,boundaryCondition)
    println(io,saveFrequency)

    close(io)

    if !isdir(output_folder)
        mkdir(output_folder)
    end
    if !isdir(save_path)
        mkdir(save_path)
    end
end