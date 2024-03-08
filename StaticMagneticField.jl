# Filename: StaticMagneticField.jl
# Programmed by: Aaron K Knoll
# Date: September 18th, 2020

# Create a rectangular region of constant magnetic field
function createRectangularRegion(x_min, x_max, y_min, y_max, z_min, z_max,
    Bx, By, Bz, regionXMin, regionXMax, regionYMin, regionYMax,
    regionZMin, regionZMax, regionBx, regionBy, regionBz)

    # Set the default return value
    my_return = 1;

    # Define a new rectangular magnetic field region
    push!(regionXMin, x_min);
    push!(regionXMax, x_max);
    push!(regionYMin, y_min);
    push!(regionYMax, y_max);
    push!(regionZMin, z_min);
    push!(regionZMax, z_max);
    push!(regionBx, Bx);
    push!(regionBy, By);
    push!(regionBz, Bz);

    # Return the default value
    my_return;
end

# Set the local magnetic field for each particle
function setMagneticField(xArray, yArray, zArray, BxArray, ByArray, BzArray,
    regionXMin, regionXMax, regionYMin, regionYMax, regionZMin, regionZMax,
    regionBx, regionBy, regionBz, gaussian_profile_flag, store_B_max,
    store_B_max_x, store_B_half_x)

    # Set the default return value
    my_return = 1;

	# Loop through all particles
    for index=1:length(xArray)
		# Set the magnetic field to zero (default condition)
		BxArray[index] = 0.0;
		ByArray[index] = 0.0;
		BzArray[index] = 0.0;

		# Check if we're inside a rectangular field region
        for j=1:length(regionXMax)
			# Define the x, y and z location of the particle
			x = xArray[index];
			y = yArray[index];
			z = zArray[index];

			if (x >= regionXMin[j] && x <= regionXMax[j] &&
				y >= regionYMin[j] && y <= regionYMax[j] &&
				z >= regionZMin[j] && z <= regionZMax[j])
				# Set the magentic field strength for the particle
					BxArray[index] = regionBx[j];
					ByArray[index] = regionBy[j];
					BzArray[index] = regionBz[j];
			end
		end
	end

    # Check if there is a Gaussian field set up
	if (gaussian_profile_flag == 1)
		# Set the magnetic field according to the Gaussian field strength in the y-direction
        for index=1:length(xArray)
            ByArray[index] = store_B_max*exp((log(0.5)*((xArray[index] - store_B_max_x)^2)) /
				((store_B_max_x - store_B_half_x)^2));
        end
	end

    # Return the default value
    my_return;
end

# Set the local magnetic field for each particle
function setMagneticFieldZPinch(xArray, yArray, zArray, BxArray, ByArray,
	BzArray, B_0, x_0, x_start)

    # Set the default return value
    my_return = 1;

	# Loop through all particles
    for index=1:length(xArray)
		# Set the magnetic field to zero for Bx and Bz (default condition)
		BxArray[index] = 0.0;
		BzArray[index] = 0.0;

		# Set the By magnetic field
		if( xArray[index] + x_start < x_0 )
			ByArray[index] = B_0*(xArray[index] + x_start)/x_0;
		else
			ByArray[index] = B_0;
		end
	end

    # Return the default value
    my_return;
end
