# Filename: ParticlePusher.jl
# Programmed by: Aaron K Knoll
# Date: September 17th, 2020

# Add new particles to the particle array
function createParticles(N, mass, Temp, q, superParticleSize, x_min, x_max,
    y_min, y_max, z_min, z_max, van_der_waals_radius,
    xArray, yArray, zArray, VxArray, VyArray, VzArray, xArray_old, yArray_old,
    zArray_old, VxArray_old, VyArray_old, VzArray_old, ExArray, EyArray,
    EzArray, BxArray, ByArray, BzArray, mArray, qArray, superParticleSizeArray,
    vdwrArray, localNn, localNe, localNi, localCn, localCe, localCi,
    cellNumber,UeX)

    # Set the default return value
    return_value = 1;

    # Loop through all particles to add
    for index=1:N
		# Randomize the particles location within the domain
        push!(xArray, (rand()*(x_max - x_min)) + x_min);
        push!(yArray, (rand()*(y_max - y_min)) + y_min);
        push!(zArray, (rand()*(z_max - z_min)) + z_min);

		# Randomize the velocity based on a Maxwellian temperature distribution
        # (Box-Muller technique)
		y1 = rand();
		while (y1 == 0.0)
			y1 = rand();
		end
		y2 = rand();
		#Assign electron with initial velocity UeX
		VX = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp/mass);
		if q<0
			VX += UeX;
		end
        push!(VxArray, VX);
        y1 = rand();
		while (y1 == 0.0)
			y1 = rand();
		end
		y2 = rand();
        push!(VyArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp/mass));
        y1 = rand();
		while (y1 == 0.0)
			y1 = rand();
		end
		y2 = rand();
        push!(VzArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp/mass));

		

		# Set the values for the previous time step
		index2 = length(xArray);
        push!(xArray_old, xArray[index2]);
        push!(yArray_old, yArray[index2]);
        push!(zArray_old, zArray[index2]);
        push!(VxArray_old, VxArray[index2]);
        push!(VyArray_old, VyArray[index2]);
        push!(VzArray_old, VzArray[index2]);

		# Set the default values of the electric and magnetic field
        push!(ExArray, 0.0);
        push!(EyArray, 0.0);
        push!(EzArray, 0.0);
        push!(BxArray, 0.0);
        push!(ByArray, 0.0);
        push!(BzArray, 0.0);

		# Set the mass, charge and super particle size
        push!(mArray, mass);
        push!(qArray, q);
        push!(superParticleSizeArray, superParticleSize);

		# Set the particle collision parameters
        push!(vdwrArray, van_der_waals_radius);
        push!(localNn, 0.0);
        push!(localNe, 0.0);
        push!(localNi, 0.0);
        push!(localCn, 0.0);
        push!(localCe, 0.0);
        push!(localCi, 0.0);
        push!(cellNumber, 1);
	end

    # Return a default value
    return_value;
end

# Add new particles to the particle array, z pinch variation
function createParticlesZPinch(N, mass, Temp, q, x_max,
    y_max, z_max, n_0, x_0, x_start, xArray, yArray, zArray, VxArray, VyArray,
	VzArray, xArray_old, yArray_old, zArray_old, VxArray_old,
	VyArray_old, VzArray_old, ExArray, EyArray, EzArray, BxArray, ByArray,
	BzArray, mArray, qArray, superParticleSizeArray, vdwrArray,
	localNn, localNe, localNi, localCn, localCe, localCi,
    cellNumber)

    # Set the default return value
    return_value = 1;

    # Loop through all particles to add
	volume = x_max*y_max*z_max;
    for index=1:N
		# Randomize the particles location within the domain
        push!(xArray, rand()*x_max);
        push!(yArray, rand()*y_max);
        push!(zArray, rand()*z_max);

		# Randomize the velocity based on a Maxwellian temperature distribution
        # (Box-Muller technique)
		index2 = length(xArray);
		Temp2 = Temp;
		if( xArray[index2] + x_start > x_0 )
			Temp2 = 1160.45;
		end
		y1 = rand();
		while (y1 == 0.0)
			y1 = rand();
		end
		y2 = rand();
        push!(VxArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp2/mass));
        y1 = rand();
		while (y1 == 0.0)
			y1 = rand();
		end
		y2 = rand();
        push!(VyArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp2/mass));
        y1 = rand();
		while (y1 == 0.0)
			y1 = rand();
		end
		y2 = rand();
        push!(VzArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Temp2/mass));

		# Set the values for the previous time step
        push!(xArray_old, xArray[index2]);
        push!(yArray_old, yArray[index2]);
        push!(zArray_old, zArray[index2]);
        push!(VxArray_old, VxArray[index2]);
        push!(VyArray_old, VyArray[index2]);
        push!(VzArray_old, VzArray[index2]);

		# Set the default values of the electric and magnetic field
        push!(ExArray, 0.0);
        push!(EyArray, 0.0);
        push!(EzArray, 0.0);
        push!(BxArray, 0.0);
        push!(ByArray, 0.0);
        push!(BzArray, 0.0);

		# Calculate the super particle size based on its location
		x_loc = xArray[index2] + x_start;
		if( x_loc < x_0 )
			superParticleSize = n_0*(1.0 - (x_loc/x_0)^2)*volume/Float64(N);
		else
			superParticleSize = 1.0e18*volume/Float64(N);
		end

		# Set the mass, charge and super particle size
        push!(mArray, mass);
        push!(qArray, q);
        push!(superParticleSizeArray, superParticleSize);

		# Set the particle collision parameters
        push!(vdwrArray, 0.0);
        push!(localNn, 0.0);
        push!(localNe, 0.0);
        push!(localNi, 0.0);
        push!(localCn, 0.0);
        push!(localCe, 0.0);
        push!(localCi, 0.0);
        push!(cellNumber, 1);
	end

    # Return a default value
    return_value;
end

# Create particles from the mass storage array
function createParticlesFromArray(N, mass, Temp, q, superParticleSize,
	x_min, x_max, y_min, y_max, z_min, z_max, van_der_waals_radius,
	mass_array, xArray, yArray, zArray, VxArray, VyArray, VzArray,
    xArray_old, yArray_old, zArray_old, VxArray_old, VyArray_old,
    VzArray_old, ExArray, EyArray, EzArray, BxArray, ByArray, BzArray,
    mArray, qArray, superParticleSizeArray, vdwrArray, localNn,
    localNe, localNi, localCn, localCe, localCi, cellNumber)

	# Set the default return value
    my_return = 1;

	# Loop through all cells in the mass array
    for index=1:N
		# Introduce new particles while there is available mass in the array
		while (mass_array[index] >= mass*superParticleSize)
			# Reduce the mass stored in the mass array by a single superparticle
			mass_array[index] = mass_array[index] - mass*superParticleSize;

			# Introduce a new superparticle
			# Randomize the particles location within the domain
			local_x_min = ((x_max - x_min) / Float64(N))*Float64(index) + x_min;
			local_x_max = ((x_max - x_min) / Float64(N))*Float64(index + 1) + x_min;
            push!(xArray, (rand()*(local_x_max - local_x_min)) + local_x_min);
            push!(yArray, (rand()*(y_max - y_min)) + y_min);
            push!(zArray, (rand()*(z_max - z_min)) + z_min);

			# Randomize the velocity based on a Maxwellian temperature distribution (Box-Muller technique)
			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
            push!(VxArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*
                sqrt(1.38064852e-23*Temp/mass));
			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
            push!(VyArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*
                sqrt(1.38064852e-23*Temp/mass));
			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
            push!(VzArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*
                sqrt(1.38064852e-23*Temp/mass));

			# Set the values for the previous time step
            index2 = length(xArray);
            push!(xArray_old, xArray[index2]);
            push!(yArray_old, yArray[index2]);
            push!(zArray_old, zArray[index2]);
            push!(VxArray_old, VxArray[index2]);
            push!(VyArray_old, VyArray[index2]);
            push!(VzArray_old, VzArray[index2]);

			# Set the default values of the electric and magnetic field
            push!(ExArray, 0.0);
            push!(EyArray, 0.0);
            push!(EzArray, 0.0);
            push!(BxArray, 0.0);
            push!(ByArray, 0.0);
            push!(BzArray, 0.0);

			# Set the mass, charge and super particle size
            push!(mArray, mass);
            push!(qArray, q);
            push!(superParticleSizeArray, superParticleSize);

			# Set the particle collision parameters
            push!(vdwrArray, van_der_waals_radius);
            push!(localNn, 0.0);
            push!(localNe, 0.0);
            push!(localNi, 0.0);
            push!(localCn, 0.0);
            push!(localCe, 0.0);
            push!(localCi, 0.0);
            push!(cellNumber, 1);
		end
	end

    # Return a default value
    return_value;
end

# Remove neutral particles according to lost mass array
function reduceNeutralMassFromArray(mass_array, qArray, cellNumber, mArray,
    superParticleSizeArray )

    # Set the default return value
    my_return = 1;

	# Loop through all neutral particles until we've reduced the mass
    # sufficiently in all cells of the mass array
    done = 0;
    i = Int(floor(rand()*length(qArray))) + 1;
	while (done == 0)
		# Check if we're already done
		done = 1;
        for index=1:length(mass_array)
			if (mass_array[index] < 0.0)
				# Not done yet
				done = 0;
			end
		end

		# If we're not done, then find a neutral particle and reduce its mass
		if (done == 0)
			while (qArray[i] != 0.0)
				i = Int(floor(rand()*length(qArray))) + 1;
			end
			j = cellNumber[i];
			if (mArray[i]*superParticleSizeArray[i] > -mass_array[j] &&
                mass_array[j] < 0.0 )
				superParticleSizeArray[i] -= (-mass_array[j]/mArray[i]);
				mass_array[j] = 0.0;
			end
			i = Int(floor(rand()*length(qArray))) + 1;
		end
	end

    # Return the default value
    my_return;
end

# Boris particle pusher
function pushParticlesBoris(xArray, yArray, zArray, VxArray, VyArray, VzArray,
    xArray_old, yArray_old, zArray_old, VxArray_old, VyArray_old, VzArray_old,
    ExArray, EyArray, EzArray, BxArray, ByArray, BzArray, mArray, qArray, timeStep)

    # Set the default return value
    my_return = 1;

	# Loop through all particles in the array
    for index=1:length(xArray)
		# Step #1: Obtain "v minus" by adding half acceleration to the initial velocity
		vm_x = VxArray[index] + qArray[index] * ExArray[index] * timeStep / (mArray[index] * 2.0);
		vm_y = VyArray[index] + qArray[index] * EyArray[index] * timeStep / (mArray[index] * 2.0);
		vm_z = VzArray[index] + qArray[index] * EzArray[index] * timeStep / (mArray[index] * 2.0);

		# Define the vector t
		tx = (qArray[index] * BxArray[index] / mArray[index])*timeStep / 2.0;
		ty = (qArray[index] * ByArray[index] / mArray[index])*timeStep / 2.0;
		tz = (qArray[index] * BzArray[index] / mArray[index])*timeStep / 2.0;
		t_squared = (tx^2 + ty^2 + tz^2);

		# Perform the first half rotation
		ux = vm_x;
		uy = vm_y;
		uz = vm_z;
		vx = tx;
		vy = ty;
		vz = tz;
		vprime_x = vm_x + (uy*vz - uz*vy);
		vprime_y = vm_y + (uz*vx - ux*vz);
		vprime_z = vm_z + (ux*vy - uy*vx);

		# Define the vector s
		sx = 2.0*tx / (1.0 + t_squared);
		sy = 2.0*ty / (1.0 + t_squared);
		sz = 2.0*tz / (1.0 + t_squared);

		# Perform the second half rotation
		ux = vprime_x;
		uy = vprime_y;
		uz = vprime_z;
		vx = sx;
		vy = sy;
		vz = sz;
		vp_x = vm_x + (uy*vz - uz*vy);
		vp_y = vm_y + (uz*vx - ux*vz);
		vp_z = vm_z + (ux*vy - uy*vx);

		# Add another half acceleration
		vx_new = vp_x + qArray[index] * ExArray[index] * timeStep / (mArray[index] * 2.0);
		vy_new = vp_y + qArray[index] * EyArray[index] * timeStep / (mArray[index] * 2.0);
		vz_new = vp_z + qArray[index] * EzArray[index] * timeStep / (mArray[index] * 2.0);

		# Update the particle velocity
		VxArray_old[index] = VxArray[index];
		VyArray_old[index] = VyArray[index];
		VzArray_old[index] = VzArray[index];
		VxArray[index] = vx_new;
		VyArray[index] = vy_new;
		VzArray[index] = vz_new;

		# Update the particle position
		xArray_old[index] = xArray[index];
		yArray_old[index] = yArray[index];
		zArray_old[index] = zArray[index];
		xArray[index] = xArray[index] + timeStep*vx_new;
		yArray[index] = yArray[index] + timeStep*vy_new;
		zArray[index] = zArray[index] + timeStep*vz_new;
    end

    # Return the default value
    my_return;
end

# Model neutral elastic scattering collisions
function neutralCollisions(xArray, yArray, zArray, VxArray, VyArray, VzArray,
    xArray_old, yArray_old, zArray_old, VxArray_old, VyArray_old, VzArray_old,
    mArray, vdwrArray, localNn, localCn, superParticleSizeArray, neutral_particle_mass,
    neutral_van_der_waals_radius, timeStep)

    # Set the default return value
    number_of_collisions = Float32(0.0);

	# Loop through all particles
    for index=1:length(xArray)
		# Stage 1: Anything + neutral collisions

		# Calculate the collision cross section
		cross_section = pi*(neutral_van_der_waals_radius + vdwrArray[index])^2;

		# Calculate the collision frequency
		collision_frequency = cross_section*sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2 +
			localCn[index]^2)*localNn[index];

		# Determine statistically if a collision has occured
		probability = 1.0 - exp(-timeStep*collision_frequency);
		if( probability > rand() )
			# Record the number of collisions
			number_of_collisions += Float32(superParticleSizeArray[index]);

			# Find a collision partner using the DSMC approach
			v_max = sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2 +
				localCn[index]^2)*2.0;
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
                Vx = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt((localCn[index]^2)/3.0);
                y1 = rand();
        		while (y1 == 0.0)
        			y1 = rand();
        		end
        		y2 = rand();
                Vy = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt((localCn[index]^2)/3.0);
                y1 = rand();
        		while (y1 == 0.0)
        			y1 = rand();
        		end
        		y2 = rand();
                Vz = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt((localCn[index]^2)/3.0);

				# Statistical test to see if the two particles should collide
				z = rand();
				v_rel = sqrt((VxArray[index] - Vx)^2 + (VyArray[index] - Vy)^2 +
                    (VzArray[index] - Vz)^2);
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
			new_vx = (mArray[index]*VxArray[index] + neutral_particle_mass*Vx - neutral_particle_mass*v_rel*x1) / (mArray[index] + neutral_particle_mass);
			new_vy = (mArray[index]*VyArray[index] + neutral_particle_mass*Vy - neutral_particle_mass*v_rel*x2) / (mArray[index] + neutral_particle_mass);
			new_vz = (mArray[index]*VzArray[index] + neutral_particle_mass*Vz - neutral_particle_mass*v_rel*x3) / (mArray[index] + neutral_particle_mass);

			# Update the particle velocity
			xArray_old[index] = xArray[index];
			yArray_old[index] = yArray[index];
			zArray_old[index] = zArray[index];
			VxArray[index] = new_vx;
			VyArray[index] = new_vy;
			VzArray[index] = new_vz;
			VxArray_old[index] = new_vx;
			VyArray_old[index] = new_vy;
			VzArray_old[index] = new_vz;
		end
	end

    # Return the number of collisions
	number_of_collisions;
end

# Model the Xenon ionization assuming fixed temperatures of electrons and ions
function fixedTemperatureXenonIonization(electron_temperature, ion_temperature,
	neutral_mass_array, xArray, yArray, zArray, VxArray, VyArray, VzArray,
    xArray_old, yArray_old, zArray_old, VxArray_old, VyArray_old, VzArray_old,
    ExArray, EyArray, EzArray, BxArray, ByArray, BzArray, mArray, qArray,
    superParticleSizeArray, vdwrArray, localNn, localNe, localNi, localCn,
    localCe, localCi, cellNumber, timeStep)

    # Set the default return value
    my_return = 1;

	# Xenon ionization cross section data from: Stephen and Mark, “Absolute Partial Electron Impact Ionization
	# Cross Sections of Xe from Threshold up to 180 eV, ” Journal of Chemical
	# Physics, vol. 81, pp.3116–3117, 1984
    cross_section_energy_eV = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
        55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0];
    cross_section_area = [1.15e-20, 2.42e-20, 3.81e-20, 4.17e-20, 4.17e-20,
        4.30e-20, 4.31e-20, 4.29e-20, 4.27e-20, 4.37e-20, 4.47e-20, 4.54e-20,
        4.57e-20, 4.59e-20, 4.55e-20, 4.48e-20, 4.42e-20, 4.31e-20];

	# Loop through all particles
    index_max = length(xArray);
    for index=1:index_max
		# Calculate the ionization cross section (zero for anything that's not an electron)
		if (qArray[index] < 0.0)
			kinetic_energy_eV = (0.5*(VxArray[index]^2 + VyArray[index]^2 +
                VzArray[index]^2 + localCn[index]^2)*mArray[index]) / 1.602176621e-19;
		else
			kinetic_energy_eV = 0.0;
		end
        j = Int(floor(kinetic_energy_eV/5.0)) - 2;
		if (j < 1)
			cross_section = 0.0;
		elseif (j >= 18)
			# Set the cross section to the limit value
			cross_section = cross_section_area[18];
		else
			# Linearly interpolate the cross section
			cross_section = cross_section_area[j] + (((kinetic_energy_eV / 5.0) -
                2.0) - Float64(j))*(cross_section_area[j+1] - cross_section_area[j]);
		end

		# Calculate the collision frequency
		collision_frequency = cross_section*sqrt(VxArray[index]^2 +
            VyArray[index]^2 + VzArray[index]^2 + localCn[index]^2)*localNn[index];

		# Determine statistically if an ionization has occured
		probability = (timeStep*collision_frequency)^0.1;
		done = 0;
		count = 0;
		while (done == 0)
			if (probability < rand())
				done = 1;
			else
				count += 1;
			end
			if (count == 10)
				done = 1;
			end
		end
		if (count == 10)
			# Record the mass lost from the background neutral population
			j = cellNumber[index];
			neutral_mass_array[j] -= superParticleSizeArray[index]*2.180171366e-25;

			# Create a new ion
            push!(xArray, xArray[index]);
            push!(yArray, yArray[index]);
            push!(zArray, zArray[index]);

			# Randomize the ion velocity based on a Maxwellian temperature distribution (Box-Muller technique)
			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
            push!(VxArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*
                ion_temperature/2.180171366e-25));
            y1 = rand();
    		while (y1 == 0.0)
    			y1 = rand();
    		end
    		y2 = rand();
            push!(VyArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*
                ion_temperature/2.180171366e-25));
            y1 = rand();
    		while (y1 == 0.0)
    			y1 = rand();
    		end
    		y2 = rand();
            push!(VzArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*
                ion_temperature/2.180171366e-25));

			# Set the values for the previous time step
			index2 = length(xArray);
            push!(VxArray_old, VxArray[index2]);
            push!(VyArray_old, VyArray[index2]);
            push!(VzArray_old, VzArray[index2]);
            push!(xArray_old, xArray[index2]);
            push!(yArray_old, yArray[index2]);
            push!(zArray_old, zArray[index2]);

            # Set the default values of the electric and magnetic field
            push!(ExArray, 0.0);
            push!(EyArray, 0.0);
            push!(EzArray, 0.0);
            push!(BxArray, 0.0);
            push!(ByArray, 0.0);
            push!(BzArray, 0.0);

    		# Set the ion mass, charge and super particle size
            push!(mArray, 2.180171366e-25);
            push!(qArray, 1.6021766208e-19);
            push!(superParticleSizeArray, superParticleSizeArray[index]);

    		# Set the particle collision parameters
            push!(vdwrArray, 2.16e-10);
            push!(localNn, 0.0);
            push!(localNe, 0.0);
            push!(localNi, 0.0);
            push!(localCn, 0.0);
            push!(localCe, 0.0);
            push!(localCi, 0.0);
            push!(cellNumber, j);

			# Create a new electron
            push!(xArray, xArray[index]);
            push!(yArray, yArray[index]);
            push!(zArray, zArray[index]);

			# Randomize the electron velocity based on a Maxwellian temperature distribution (Box-Muller technique)
            y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
            push!(VxArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*
                electron_temperature/9.10938356e-31));
            y1 = rand();
    		while (y1 == 0.0)
    			y1 = rand();
    		end
    		y2 = rand();
            push!(VyArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*
                electron_temperature/9.10938356e-31));
            y1 = rand();
    		while (y1 == 0.0)
    			y1 = rand();
    		end
    		y2 = rand();
            push!(VzArray, sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*
                electron_temperature/9.10938356e-31));

			# Set the values for the previous time step
            index2 = length(xArray);
            push!(VxArray_old, VxArray[index2]);
            push!(VyArray_old, VyArray[index2]);
            push!(VzArray_old, VzArray[index2]);
            push!(xArray_old, xArray[index2]);
            push!(yArray_old, yArray[index2]);
            push!(zArray_old, zArray[index2]);

			# Set the default values of the electric and magnetic field
            push!(ExArray, 0.0);
            push!(EyArray, 0.0);
            push!(EzArray, 0.0);
            push!(BxArray, 0.0);
            push!(ByArray, 0.0);
            push!(BzArray, 0.0);

			# Set the electron mass, charge and super particle size
            push!(mArray, 9.10938356e-31);
            push!(qArray, -1.6021766208e-19);
            push!(superParticleSizeArray, superParticleSizeArray[index]);

			# Set the particle collision parameters
            push!(vdwrArray, 0.0);
            push!(localNn, 0.0);
            push!(localNe, 0.0);
            push!(localNi, 0.0);
            push!(localCn, 0.0);
            push!(localCe, 0.0);
            push!(localCi, 0.0);
            push!(cellNumber, j);
		end
	end

    # Return the default value
    my_return;
end

# Write the particle arrays to file
function writeParticlesToFile(xArray, yArray, zArray, VxArray, VyArray, VzArray,
    mArray, qArray, superParticleSizeArray, path)

    # Set the default return value
    my_return = 1;

	# Open the output files
    stream1 = open("$(path)ions.txt", "w");
	stream2 = open("$(path)electrons.txt", "w");
	stream3 = open("$(path)neutrals.txt", "w");

	# Loop through all particles
    for index=1:length(xArray)
		if (qArray[index] > 0.0)
			write(stream1, "$(xArray[index]) $(yArray[index]) $(zArray[index]) $(VxArray[index]) $(VyArray[index]) $(VzArray[index]) $(mArray[index]) $(qArray[index]) $(superParticleSizeArray[index])\n");
		elseif (qArray[index] < 0.0)
			write(stream2, "$(xArray[index]) $(yArray[index]) $(zArray[index]) $(VxArray[index]) $(VyArray[index]) $(VzArray[index]) $(mArray[index]) $(qArray[index]) $(superParticleSizeArray[index])\n");
		else
			write(stream3, "$(xArray[index]) $(yArray[index]) $(zArray[index]) $(VxArray[index]) $(VyArray[index]) $(VzArray[index]) $(mArray[index]) $(qArray[index]) $(superParticleSizeArray[index])\n");
		end
	end

	# Close the output file
	close(stream1);
	close(stream2);
	close(stream3);

    # Return the default value
    my_return;
end

function logParticleLocations(time_stamp, xArray, yArray, zArray)

	# Set the default return value
    my_return = 1;

	# Open the output files
    stream = open("particle_locations.txt", "a");

	# Log the particle locations
	write(stream, "$(time_stamp) $(length(xArray)) ");
	for index=1:length(xArray)
		write(stream, "$(xArray[index]) $(yArray[index]) $(zArray[index]) ");
	end
	write(stream, "\n");

	# Close the output file
	close(stream);

	# Return the default value
    my_return;
end
