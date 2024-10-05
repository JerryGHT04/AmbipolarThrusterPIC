using CUDA
include("classDefinition.jl")

#numberofCollisions should be a CuArray with one element
function neutralCollisions(myParticle, timeStep, neutral_particle_mass, neutral_van_der_waals_radius)
    numberofCollisions = CUDA.zeros(1)
    CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(myParticle.alive),256),
        neutralCollisions_kernel(numberofCollisions, neutral_particle_mass, neutral_van_der_waals_radius, timeStep, myParticle.vdwrArray,  myParticle.VelArray,  myParticle.VelArray_old,  myParticle.PosArray,  myParticle.PosArray_old,  myParticle.localCn,  myParticle.localNn, myParticle.mArray, myParticle.alive, myParticle.superParticleSizeArray)
    )
    return numberofCollisions
end

function neutralCollisions_kernel(number_of_collisions, neutral_particle_mass, neutral_van_der_waals_radius, timeStep, vdwrArray, VelArray, VelArray_old, PosArray, PosArray_old, localCn, localNn, mArray, alive, superParticleSizeArray)
    #loop through all particles
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
     # Loop through all particles
    if idx <= length(alive)
        if alive[idx] == 1
            # Calculate the collision cross section
		    cross_section = pi*(neutral_van_der_waals_radius + vdwrArray[idx])^2;
            
            # Calculate the collision frequency
            V_squared = VelArray[1,idx]^2 + VelArray[2,idx]^2 + VelArray[3,idx]^2
            collision_frequency = cross_section*sqrt(V_squared + localCn[idx]^2)*localNn[idx];
            
            # Determine statistically if a collision has occured
		    probability = 1.0 - exp(-timeStep*collision_frequency);
            if( rand() <= probability )
                # Record the number of collisions
                number_of_collisions .+= Float32(superParticleSizeArray[idx]);
    
                # Find a collision partner using the DSMC approach
                v_max = sqrt( V_squared + localCn[idx]^2)*2.0;
                Vx = 0.0;
                Vy = 0.0;
                Vz = 0.0;
                v_rel = 0.0;
                done = 0;
                while (done == 0)
                    # Generate a random neutral particle
                    y1 = rand();
                    while (y1 == 0.0)
                        y1 = rand();
                    end
                    y2 = rand();
                    Vx = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt((localCn[idx]^2)/3.0);
                    y1 = rand();
                    while (y1 == 0.0)
                        y1 = rand();
                    end
                    y2 = rand();
                    Vy = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt((localCn[idx]^2)/3.0);
                    y1 = rand();
                    while (y1 == 0.0)
                        y1 = rand();
                    end
                    y2 = rand();
                    Vz = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt((localCn[idx]^2)/3.0);
    
                    # Statistical test to see if the two particles should collide
                    z = rand();
                    v_rel = sqrt((VelArray[1, idx] - Vx)^2 + (VelArray[2,idx] - Vy)^2 +
                        (VelArray[3,idx] - Vz)^2);
                    if (z <= v_rel / v_max)
                        done = 1;
                    end
                end
    
                # Pick a random unit vector using Marsaglia's recipe
                done = 0;
                y1 = 0.0;
                y2 = 0.0;
                while (done == 0)
                    y1 = 2.0*rand() - 1.0;
                    y2 = 2.0*rand() - 1.0;
                    if (y1^2 + y2^2 <= 1.0)
                        done = 1;
                    end
                end
                r = sqrt(y1^2 + y2^2);
                x1 = 2.0*y1*sqrt(1.0 - r^2);
                x2 = 2.0*y2*sqrt(1.0 - r^2);
                x3 = 1.0 - 2.0*(r^2);
    
                # Calculate the new particle velocity after the collision
                new_vx = (mArray[idx]*VelArray[1,idx] + neutral_particle_mass*Vx - neutral_particle_mass*v_rel*x1) / (mArray[idx] + neutral_particle_mass);
                new_vy = (mArray[idx]*VelArray[2,idx] + neutral_particle_mass*Vy - neutral_particle_mass*v_rel*x2) / (mArray[idx] + neutral_particle_mass);
                new_vz = (mArray[idx]*VelArray[3,idx] + neutral_particle_mass*Vz - neutral_particle_mass*v_rel*x3) / (mArray[idx] + neutral_particle_mass);
    
                # Update the particle velocity
                PosArray_old[1,idx] = PosArray[1,idx]
                PosArray_old[2,idx] = PosArray[2,idx]
                PosArray_old[3,idx] = PosArray[3,idx]
                VelArray[1,idx] = new_vx
                VelArray[2,idx] = new_vy
                VelArray[3,idx] = new_vz
                VelArray_old[1,idx] = new_vx
                VelArray_old[2,idx] = new_vy
                VelArray_old[3,idx] = new_vz
            end
        end
    end

    return
end