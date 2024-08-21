# Filename: Poisson1D.jl
# Programmed by: Aaron K Knoll
# Date: September 17th, 2020

include("GPUkernel.jl")

# Poisson equation solver
function solvePoissonEquation( N, charge_bin, V_first, V_last, A, phi,
    phi_output, electric_constant )
    # Set the default return value
    my_return = 1;

    # Define the right hand side of the Poisson equation
    B = zeros(Float64, (N,1));
    B[1] = V_first;
    for index=2:N-1
        B[index] = charge_bin[index] / electric_constant;
    end
    B[N] = V_last;

    # Solve the governing equation
    y = A\B;#A is the Tridiagonal matrix of second order difference approximation

    # Return the calculated phi array
    phi .= y';#transpose

    # Update the phi_output array
    for index=1:N
        phi_output[index] += phi[index];
    end

    # Return a default value
    my_return;
end

# Poisson equation solver
function solvePoissonEquation_GPU(N, charge_bin, V_first, V_last, A, phi,
    phi_output, electric_constant )

    # Copy arrays to GPU
    d_charge_bin = CuArray(charge_bin)
    d_A = CuArray(A)
    d_phi_output = CuArray(phi_output)
	#println("sizeof phi_output:")
	#println(size(d_phi_output))
	d_B = d_charge_bin'./electric_constant
	#println("sizeof B:")
	#println(size(d_B))
	# Launch kernel to set boundary conditions
	PoissonEquation_kernel(V_first,V_last,d_B)

    # Solve the governing equation on the GPU
    d_y = d_A \ d_B
	#println("sizeof y:")
	#println(size(d_y))
    # Return calculated phi array
    phi .= Array(d_y')
    # Update the phi_output array on the GPU
    d_phi_output += d_y'

    # Retrive data from GPU
    phi_output .= Array(d_phi_output)

    # Return default value
    return 1
end



function solvePoissonEquation_2Neumann(N, charge_bin, E1,E2, A, phi,
    phi_output, electric_constant,deltaX)
	#E1 and E2 are the electric field at i = 1 and i = N grid points (boundaries)
	#E = -dphi/dx
    # Set the default return value
    my_return = 1;

	#Modify Dirilet tridiagonal matrix to Neumann BC type
	A_N = copy(A);
	M,N = size(A);
	A_N[M,N] = 0.5*A[M,N];
	A_N[1,1] = 0.5*A[1,1];

    # Define the right hand side of the Poisson equation
    B = zeros(Float64, (N,1));
    B[1] = E1/deltaX;
    for index=2:N-1
        B[index] = charge_bin[index] / electric_constant;
    end
    B[N] = -E2/deltaX;

    # Solve the governing equation
    y = A_N\B;#A is the Tridiagonal matrix of second order difference approximation

    # Return the calculated phi array
    phi .= y';#transpose

    # Update the phi_output array
    for index=1:N
        phi_output[index] += phi[index];
    end

    # Return a default value
    my_return;
end

function solvePoissonEquation_DN(N, charge_bin, V_first ,E2, A, phi,
    phi_output, electric_constant,deltaX)
	# Set the default return value
    my_return = 1;

    # Define the right hand side of the Poisson equation
    B = zeros(Float64, (N,1));
    B[1] = V_first;
    for index=2:N-1
        B[index] = charge_bin[index] / electric_constant;
    end
    B[N] = -E2*deltaX;

    # Solve the governing equation
    y = A\B;#A is the Tridiagonal matrix of second order difference approximation

    # Return the calculated phi array
    phi .= y';#transpose

    # Update the phi_output array
    for index=1:N
        phi_output[index] += phi[index];
    end

    # Return a default value
    my_return;
end

# Poisson equation solver - Z pinch variation
function solvePoissonEquationZPinch( N, charge_bin, A, phi, phi_output,
	electric_constant )
    # Set the default return value
    my_return = 1;

    # Define the right hand side of the Poisson equation
    B = zeros(Float64, (N,1));
    B[1] = 0.0;
    for index=2:N-1
        B[index] = charge_bin[index] / electric_constant;
    end
    B[N] = 0.0;

    # Solve the governing equation
    y = A\B;

    # Return the calculated phi array
    phi .= y';

    # Update the phi_output array
    for index=1:N
        phi_output[index] += phi[index];
    end

    # Return a default value
    my_return;
end

# Poisson equation bi-directional solver
function solvePoissonEquationBidirectional( N, charge_bin, charge_bin_2, V_first,
    V_last, A, A_2, dx, dy, phi, phi_2, phi_output, electric_constant )
    # Set the default return value
    my_return = 1;

	# Define the right hand side of the first Poisson equation
    B = zeros(Float64, (N,1));
    B[1] = V_first;
    for index=2:N-1
        B[index] = (charge_bin[index] / electric_constant)*
            (dy[index] / (dx[index] + dy[index]));
    end
    B[N] = V_last;

	# Solve the governing equation
    y = A\B;

	# Define the right hand side of the second Poisson equation
    B_2 = zeros(Float64, (N,1));
    B_2[1] = 0.0;
    for index=2:N-1
        B_2[index] = (charge_bin_2[index] / electric_constant)*
            (dx[index] / (dx[index] + dy[index]));
    end
    B_2[N] = 0.0;

	# Solve the governing equation
    y_2 = A_2\B_2;

    # Return the calculated phi array
    phi .= y';
    phi_2 .= y_2';

    # Update the phi_output array
    for index=1:N
        phi_output[index] += phi[index];
    end

    # Return a default value
    my_return;
end

# Calculate the statistical properties within each cell
function particleCountRectangular(xArray, VxArray, VyArray, VzArray, qArray,
    localNn, localNe, localNi, localCn, localCe, localCi, BxArray, ByArray,
    BzArray, superParticleSizeArray, cellNumber, charge_bin, neArray, uexArray,
    ueyArray, uezArray, niArray, uixArray, nnArray, CnArray, CeArray, CiArray,
    BxBinArray, ByBinArray, BzBinArray, neArray_output, uexArray_output,
    ueyArray_output, uezArray_output, niArray_output, uixArray_output,
    nnArray_output, CnArray_output, CeArray_output, CiArray_output,
    BxBinArray_output, ByBinArray_output, BzBinArray_output,
    Area, dx, N, X_MAX, output_counter)

	# Zero the charge bins
    charge_bin .= 0.0;
    neArray .= 0.0;
    uexArray .= 0.0;
    ueyArray .= 0.0;
    uezArray .= 0.0;
    niArray .= 0.0;
    uixArray .= 0.0;
    nnArray .= 0.0;
    CnArray .= 0.0;
    CeArray .= 0.0;
    CiArray .= 0.0;
    BxBinArray .= 0.0;
    ByBinArray .= 0.0;
    BzBinArray .= 0.0;

	# Loop through all particles
	volume = Area[1]*dx[1];
    for index=1:length(xArray)
		# Calculate the cell location of the particle
        j = Int(floor(Float64(N)*(xArray[index]/X_MAX))) + 1;
		if (j > N)
			j = N;
		end
		if (j < 1)
			j = 1;
		end

		# Set the local cell number
		cellNumber[index] = j;

		# Calculate the charge contribution to the bin
		charge_bin[j] += (qArray[index] * superParticleSizeArray[index]);
		if (qArray[index] < 0.0)
			neArray[j] += superParticleSizeArray[index];
			uexArray[j] += (superParticleSizeArray[index] * VxArray[index]);
			ueyArray[j] += (superParticleSizeArray[index] * VyArray[index]);
			uezArray[j] += (superParticleSizeArray[index] * VzArray[index]);
			CeArray[j] += sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2)*superParticleSizeArray[index];
			BxBinArray[j] += (superParticleSizeArray[index] * BxArray[index]);
			ByBinArray[j] += (superParticleSizeArray[index] * ByArray[index]);
			BzBinArray[j] += (superParticleSizeArray[index] * BzArray[index]);
		elseif (qArray[index] > 0.0)
			niArray[j] += superParticleSizeArray[index];
			uixArray[j] += (superParticleSizeArray[index] * VxArray[index]);
			CiArray[j] += sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2)*superParticleSizeArray[index];
		else
			nnArray[j] += superParticleSizeArray[index];
			CnArray[j] += sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2)*superParticleSizeArray[index];
		end
	end

	# Loop through all cells
    for index=1:N
		# Normalize the simulation output parameters
		if (neArray[index] > 0.0)
			uexArray[index] /= neArray[index];
			ueyArray[index] /= neArray[index];
			uezArray[index] /= neArray[index];
			CeArray[index] /= neArray[index];
			BxBinArray[index] /= neArray[index];
			ByBinArray[index] /= neArray[index];
			BzBinArray[index] /= neArray[index];
		else
			uexArray[index] = 0.0;
			ueyArray[index] = 0.0;
			uezArray[index] = 0.0;
			CeArray[index] = 0.0;
			BxBinArray[index] = 0.0;
			ByBinArray[index] = 0.0;
			BzBinArray[index] = 0.0;
		end
		if (niArray[index] > 0.0)
			uixArray[index] /= niArray[index];
			CiArray[index] /= niArray[index];
		else
			uixArray[index] = 0.0;
			CiArray[index] = 0.0;
		end
		if (nnArray[index] > 0.0)
			CnArray[index] /= nnArray[index];
		else
			CnArray[index] = 0.0;
		end
		neArray[index] /= volume;
		niArray[index] /= volume;
		nnArray[index] /= volume;
	end

	# Update the local collision properties for each particle
    for index=1:length(xArray)
		# Calculate the cell location of the particle
        j = Int(floor(Float64(N)*(xArray[index]/X_MAX))) + 1;
		if (j > N)
			j = N;
		end
		if (j < 1)
			j = 1;
		end

		# Set the local properties
		localNn[index] = nnArray[j];
		localNe[index] = neArray[j];
		localNi[index] = niArray[j];
		localCn[index] = CnArray[j];
		localCe[index] = CeArray[j];
		localCi[index] = CiArray[j];
	end

	# Record the output data
	neArray_output += neArray;
    uexArray_output += uexArray;
	ueyArray_output += ueyArray;
	uezArray_output += uezArray;
	niArray_output += niArray;
	uixArray_output += uixArray;
	nnArray_output += nnArray;
	CnArray_output += CnArray;
    CeArray_output += CeArray;
	CiArray_output += CiArray;
	BxBinArray_output += BxBinArray;
	ByBinArray_output += ByBinArray;
	BzBinArray_output += BzBinArray;

	output_counter += 1;
end

# Calculate the statistical properties within each cell
function particleCountRectangularBidirectional(xArray, yArray,
	VxArray, VyArray, VzArray, qArray, localNn, localNe,
	localNi, localCn, localCe, localCi, BxArray, ByArray, BzArray,
	superParticleSizeArray, cellNumber, charge_bin, charge_bin_2,
    neArray, uexArray, ueyArray, uezArray, niArray, uixArray, nnArray,
    CnArray, CeArray, CiArray, BxBinArray, ByBinArray, BzBinArray,
    neArray_output, uexArray_output, ueyArray_output, uezArray_output,
    niArray_output, uixArray_output, nnArray_output, CnArray_output,
    CeArray_output, CiArray_output, BxBinArray_output, ByBinArray_output,
    BzBinArray_output, Area, dx, N, X_MAX, Y_MAX, output_counter)

	# Zero the charge bins
    charge_bin .= 0.0;
    charge_bin_2 .= 0.0;
    neArray .= 0.0;
    uexArray .= 0.0;
    ueyArray .= 0.0;
    uezArray .= 0.0;
    niArray .= 0.0;
    uixArray .= 0.0;
    nnArray .= 0.0;
    CnArray .= 0.0;
    CeArray .= 0.0;
    CiArray .= 0.0;
    BxBinArray .= 0.0;
    ByBinArray .= 0.0;
    BzBinArray .= 0.0;

	# Loop through all particles
	volume = Area[1]*dx[1];
    for index=1:length(xArray)
        # Calculate the cell location of the particle in the first direction
        j = Int(floor(Float64(N)*(xArray[index]/X_MAX))) + 1;
		if (j > N)
			j = N;
		end
		if (j < 1)
			j = 1;
		end

        # Calculate the cell location of the particle in the second direction
        j2 = Int(floor(Float64(N)*(yArray[index]/Y_MAX))) + 1;
		if (j2 > N)
			j2 = N;
		end
		if (j2 < 1)
			j2 = 1;
		end

		# Set the local cell number
		cellNumber[index] = j;

		# Calculate the charge contribution to the bins
		charge_bin[j] += (qArray[index] * superParticleSizeArray[index]);
		charge_bin_2[j2] += (qArray[index] * superParticleSizeArray[index]);
		if (qArray[index] < 0.0)
			neArray[j] += superParticleSizeArray[index];
			uexArray[j] += (superParticleSizeArray[index] * VxArray[index]);
			ueyArray[j] += (superParticleSizeArray[index] * VyArray[index]);
			uezArray[j] += (superParticleSizeArray[index] * VzArray[index]);
			CeArray[j] += sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2)*superParticleSizeArray[index];
			BxBinArray[j] += (superParticleSizeArray[index] * BxArray[index]);
			ByBinArray[j] += (superParticleSizeArray[index] * ByArray[index]);
			BzBinArray[j] += (superParticleSizeArray[index] * BzArray[index]);
		elseif (qArray[index] > 0.0)
			niArray[j] += superParticleSizeArray[index];
			uixArray[j] += (superParticleSizeArray[index] * VxArray[index]);
			CiArray[j] += sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2)*superParticleSizeArray[index];
		else
			nnArray[j] += superParticleSizeArray[index];
			CnArray[j] += sqrt(VxArray[index]^2 + VyArray[index]^2 + VzArray[index]^2)*superParticleSizeArray[index];
		end
	end

	# Loop through all cells
    for index=1:N
		# Normalize the simulation output parameters
		if (neArray[index] > 0.0)
			uexArray[index] /= neArray[index];
			ueyArray[index] /= neArray[index];
			uezArray[index] /= neArray[index];
			CeArray[index] /= neArray[index];
			BxBinArray[index] /= neArray[index];
			ByBinArray[index] /= neArray[index];
			BzBinArray[index] /= neArray[index];
		else
			uexArray[index] = 0.0;
			ueyArray[index] = 0.0;
			uezArray[index] = 0.0;
			CeArray[index] = 0.0;
			BxBinArray[index] = 0.0;
			ByBinArray[index] = 0.0;
			BzBinArray[index] = 0.0;
		end
		if (niArray[index] > 0.0)
			uixArray[index] /= niArray[index];
			CiArray[index] /= niArray[index];
		else
			uixArray[index] = 0.0;
			CiArray[index] = 0.0;
		end
		if (nnArray[index] > 0.0)
			CnArray[index] /= nnArray[index];
		else
			CnArray[index] = 0.0;
		end
		neArray[index] /= volume;
		niArray[index] /= volume;
		nnArray[index] /= volume;
	end

	# Update the local collision properties for each particle
    for index=1:length(xArray)
		# Calculate the cell location of the particle
        j = Int(floor(Float64(N)*(xArray[index]/X_MAX))) + 1;
		if (j > N)
			j = N;
		end
		if (j < 1)
			j = 1;
		end

		# Set the local properties
		localNn[index] = nnArray[j];
		localNe[index] = neArray[j];
		localNi[index] = niArray[j];
		localCn[index] = CnArray[j];
		localCe[index] = CeArray[j];
		localCi[index] = CiArray[j];
	end

	# Record the output data
	neArray_output += neArray;
    uexArray_output += uexArray;
	ueyArray_output += ueyArray;
	uezArray_output += uezArray;
	niArray_output += niArray;
	uixArray_output += uixArray;
	nnArray_output += nnArray;
	CnArray_output += CnArray;
    CeArray_output += CeArray;
	CiArray_output += CiArray;
	BxBinArray_output += BxBinArray;
	ByBinArray_output += ByBinArray;
	BzBinArray_output += BzBinArray;

	output_counter += 1;
end

# Calculate the electric field experienced by each particle
function electricFieldRectangular(xArray, ExArray, EyArray, EzArray, X_MAX,
    phi, dx, N)
    # Set the default return value
    my_return = 1;

	# Loop through all particles
    for index=1:length(xArray)
		# Calculate the cell location of the particle
        j = Int(round(Float64(N)*(xArray[index]/X_MAX)));

		# Calculate the electric field
		if (j < 1)
			j = 1;
			ExArray[index] = (-phi[j + 1] + phi[j]) / dx[j];
		elseif (j >= N)
			j = N;
			ExArray[index] = (-phi[j] + phi[j - 1]) / dx[j - 1];
		else
			ExArray[index] = (-phi[j + 1] + phi[j]) / dx[j];
		end

		# Set the electric field in the y and z directions to zero
		EyArray[index] = 0.0;
		EzArray[index] = 0.0;
	end

	# Return the default value
	my_return;
end

function electricFieldRectangularBidirectional(xArray, yArray,
	ExArray, EyArray, EzArray, X_MAX, Y_MAX, phi, phi_2, dx, dy, N )

    # Set the default return value
    my_return = 1;

	# Loop through all particles
    for index=1:length(xArray)
		# Calculate the cell location of the particle
        j = Int(round(Float64(N)*(xArray[index]/X_MAX)));
        j2 = Int(round(Float64(N)*(yArray[index]/Y_MAX)));

		# Calculate the electric field in the x direction
        if (j < 1)
			j = 1;
			ExArray[index] = (-phi[j + 1] + phi[j]) / dx[j];
		elseif (j >= N)
			j = N;
			ExArray[index] = (-phi[j] + phi[j - 1]) / dx[j - 1];
		else
			ExArray[index] = (-phi[j + 1] + phi[j]) / dx[j];
		end


		# Calculate the electric field in the y direction
        if (j2 < 1)
			j2 = 1;
			EyArray[index] = (-phi_2[j2 + 1] + phi_2[j2]) / dy[j2];
		elseif (j2 >= N)
			j2 = N;
			EyArray[index] = (-phi_2[j2] + phi_2[j2 - 1]) / dy[j2 - 1];
		else
			EyArray[index] = (-phi_2[j2 + 1] + phi_2[j2]) / dy[j2];
		end

		# Set the electric field in the z direction to zero
		EzArray[index] = 0.0;
	end

    # Return the default value
	my_return;
end

# Rectangular boundary conditions with periodic conditions in y and z
function boundaryConditionsRectangularNoWallLoss(xArray, yArray, zArray,
	VxArray, VyArray, VzArray, xArray_old, yArray_old, zArray_old,
	VxArray_old, VyArray_old, VzArray_old, ExArray, EyArray, EzArray,
	BxArray, ByArray, BzArray, mArray, qArray, vdwrArray, localNn, localNe,
	localNi, localCn, localCe, localCi, cellNumber, superParticleSizeArray,
    X_MAX, Y_MAX, Z_MAX)

	# Clear the ion counter
	ion_counter = 0;

	# electron counter
	electron_counter = 0;
	# Loop through all particles
    index = 1;
	while (index <= length(xArray))
		# Set the initial state of the remove ion and electron flags
		remove_ion_flag = 0;
		remove_electron_flag = 0;

		# Implement the channel wall boundary conditions as periodic
		if (yArray[index] < 0.0)
			yArray[index] = yArray[index] + Y_MAX;
			yArray_old[index] = yArray_old[index] + Y_MAX;
		end
		if (yArray[index] > Y_MAX)
			yArray[index] = yArray[index] - Y_MAX;
			yArray_old[index] = yArray_old[index] - Y_MAX;
		end
		if (zArray[index] < 0.0)
			zArray[index] = zArray[index] + Z_MAX;
			zArray_old[index] = zArray_old[index] + Z_MAX;
		end
		if (zArray[index] > Z_MAX)
			zArray[index] = zArray[index] - Z_MAX;
			zArray_old[index] = zArray_old[index] - Z_MAX;
		end

		# Implement the boundary conditions in the x-direction
		if (qArray[index] != 0.0)
			if (xArray[index] < 0.0 || xArray[index] > X_MAX)
				if (qArray[index] > 0.0)
					# Remove the particle
					remove_ion_flag = 1;
				end
				if (qArray[index] < 0.0)
					# Remove the particle
					remove_electron_flag = 1;
				end
			end
		else
			if (xArray[index] < 0.0 || xArray[index] > X_MAX)
				# Reflect the particle
				xArray[index] = xArray_old[index];
				VxArray[index] = -VxArray[index];
				VxArray_old[index] = -VxArray_old[index];
			end
		end

		if (remove_ion_flag == 1)
            # Remove the ions
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);

			# Increment the ion counter
			ion_counter += 1;
		elseif (remove_electron_flag == 1)
            # Remove the electrons
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);

			#increment electron counter
			electron_counter += 1;
		else
            index += 1;
        end
	end

	# Return the number of ions lost from the domain
	return ion_counter, electron_counter

end


function boundaryConditionsRectangularNoWallLoss_returndE(xArray, yArray, zArray,
	VxArray, VyArray, VzArray, xArray_old, yArray_old, zArray_old,
	VxArray_old, VyArray_old, VzArray_old, ExArray, EyArray, EzArray,
	BxArray, ByArray, BzArray, mArray, qArray, vdwrArray, localNn, localNe,
	localNi, localCn, localCe, localCi, cellNumber, superParticleSizeArray,
    X_MAX, Y_MAX, Z_MAX, electric_constant,Aexit)

	# Clear the ion counter
	Qi_lost = 0.0;

	# electron counter
	Qe_lost = 0.0;
	# Loop through all particles
    index = 1;
	while (index <= length(xArray))
		# Set the initial state of the remove ion and electron flags
		remove_ion_flag = 0;
		remove_electron_flag = 0;
		escape_from_RHS_flag = 0;

		# Implement the channel wall boundary conditions as periodic
		if (yArray[index] < 0.0)
			yArray[index] = yArray[index] + Y_MAX;
			yArray_old[index] = yArray_old[index] + Y_MAX;
		end
		if (yArray[index] > Y_MAX)
			yArray[index] = yArray[index] - Y_MAX;
			yArray_old[index] = yArray_old[index] - Y_MAX;
		end
		if (zArray[index] < 0.0)
			zArray[index] = zArray[index] + Z_MAX;
			zArray_old[index] = zArray_old[index] + Z_MAX;
		end
		if (zArray[index] > Z_MAX)
			zArray[index] = zArray[index] - Z_MAX;
			zArray_old[index] = zArray_old[index] - Z_MAX;
		end

		# Implement the boundary conditions in the x-direction
		if (qArray[index] != 0.0)
			if (xArray[index] < 0.0 || xArray[index] > X_MAX)
				if (qArray[index] > 0.0)
					# Remove the particle
					remove_ion_flag = 1;
				end
				if (qArray[index] < 0.0)
					# Remove the particle
					remove_electron_flag = 1;
				end
			end
			if xArray[index] > X_MAX
				escape_from_RHS_flag = 1;
			end
		else
			if (xArray[index] < 0.0 || xArray[index] > X_MAX)
				# Reflect the particle
				xArray[index] = xArray_old[index];
				VxArray[index] = -VxArray[index];
				VxArray_old[index] = -VxArray_old[index];
			end
		end

		if (remove_ion_flag == 1)
            # Remove the ions
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            
			
			if (escape_from_RHS_flag == 1)
				# Increment the ion counter
				Qi_lost += qArray[index]*superParticleSizeArray[index];
			end
			deleteat!(qArray, index);
			deleteat!(superParticleSizeArray, index);
		elseif (remove_electron_flag == 1)
            # Remove the electrons
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            
			if (escape_from_RHS_flag == 1)
				#increment electron counter
				Qe_lost += qArray[index]*superParticleSizeArray[index];
			end
			deleteat!(qArray, index);
			deleteat!(superParticleSizeArray, index);

		else
            index += 1;
        end
	end

	# Return the number of ions lost from the domain
	dE = -(Qi_lost + Qe_lost)/(electric_constant*Aexit)
	return dE

end

# Rectangular boundary conditions with periodic conditions in all directions
function boundaryConditionsRectangularFullyPeriodic(xArray, yArray, zArray,
	xArray_old, yArray_old, zArray_old, X_MAX, Y_MAX, Z_MAX)

	# Set the default return value
    my_return = 1;

	# Loop through all particles
    index = 1;
	while (index <= length(xArray))
		# Implement the periodic boundary conditions
		if (xArray[index] < 0.0)
			xArray[index] = xArray[index] + X_MAX;
			xArray_old[index] = xArray_old[index] + X_MAX;
		end
		if (xArray[index] > X_MAX)
			xArray[index] = xArray[index] - X_MAX;
			xArray_old[index] = xArray_old[index] - X_MAX;
		end
		if (yArray[index] < 0.0)
			yArray[index] = yArray[index] + Y_MAX;
			yArray_old[index] = yArray_old[index] + Y_MAX;
		end
		if (yArray[index] > Y_MAX)
			yArray[index] = yArray[index] - Y_MAX;
			yArray_old[index] = yArray_old[index] - Y_MAX;
		end
		if (zArray[index] < 0.0)
			zArray[index] = zArray[index] + Z_MAX;
			zArray_old[index] = zArray_old[index] + Z_MAX;
		end
		if (zArray[index] > Z_MAX)
			zArray[index] = zArray[index] - Z_MAX;
			zArray_old[index] = zArray_old[index] - Z_MAX;
		end

		index += 1;
	end

	# Return the default value
	my_return;
end

# Set the boundary conditions for a Hall Effect Thruster
function boundaryConditionsHallThruster(xArray, yArray, zArray,
	VxArray, VyArray, VzArray, xArray_old, yArray_old, zArray_old,
	VxArray_old, VyArray_old, VzArray_old, ExArray, EyArray, EzArray,
	BxArray, ByArray, BzArray, mArray, qArray, vdwrArray, localNn,
    localNe, localNi, localCn, localCe, localCi, cellNumber,
	superParticleSizeArray, X_MAX, Y_MAX, Z_MAX, N, neutral_mass_store,
    cathodeRegionLength)

	# Clear the ion counter
	ion_counter = 0;

	# Loop through all particles
    index = 1;
	while (index <= length(xArray))
		# Set the initial state of the remove ion, electron and neutral flags
		remove_ion_flag = 0;
		remove_electron_flag = 0;
		remove_neutral_flag = 0;

		# Implement the channel wall boundary condition
		if (yArray[index] < 0.0)
			if (qArray[index] > 0.0)
				# Remove the particle
				remove_ion_flag = 1;

				if (xArray[index] >= cathodeRegionLength)
					# Calculate the cell location of the particle
					j = Int(floor(Float64(N)*(xArray[index]/X_MAX)));
					if (j > N)
						j = N;
					end
					if (j < 1)
						j = 1;
					end

					# Store the lost mass
					neutral_mass_store[j] += (mArray[index]*superParticleSizeArray[index]);
				end
			elseif (qArray[index] < 0.0)
				if (xArray[index] >= cathodeRegionLength)
					# Remove the particle
					remove_electron_flag = 1;
				else
					# Reflect the particle
					VyArray[index] = -VyArray[index];
					VyArray_old[index] = -VyArray_old[index];
					yArray[index] = yArray_old[index];
				end
			else
				if (xArray[index] >= cathodeRegionLength)
					# Periodic condition
					yArray[index] = yArray[index] + Y_MAX;
					yArray_old[index] = yArray_old[index] + Y_MAX;
				else
					# Remove the particle
					remove_neutral_flag = 1;
				end
			end
		end
		if ( yArray[index] > Y_MAX )
			if (qArray[index] > 0.0)
				# Remove the particle
				remove_ion_flag = 1;

				if (xArray[index] >= cathodeRegionLength)
                    # Calculate the cell location of the particle
					j = Int(floor(Float64(N)*(xArray[index]/X_MAX)));
					if (j > N)
						j = N;
					end
					if (j < 1)
						j = 1;
					end

					# Store the lost mass
					neutral_mass_store[j] += (mArray[index]*superParticleSizeArray[index]);
				end
			elseif (qArray[index] < 0.0)
				if (xArray[index] >= cathodeRegionLength)
					# Remove the particle
					remove_electron_flag = 1;
				else
					#Reflect the particle
					VyArray[index] = -VyArray[index];
					VyArray_old[index] = -VyArray_old[index];
					yArray[index] = yArray_old[index];
				end
			else
				if (xArray[index] >= cathodeRegionLength)
					# Periodic condition
					yArray[index] = yArray[index] - Y_MAX;
					yArray_old[index] = yArray_old[index] - Y_MAX;
				else
					# Remove the particle
					remove_neutral_flag = 1;
				end
			end
		end

		# Implement the periodic boundary condition
		if (zArray[index] < 0.0)
			zArray[index] = zArray[index] + Z_MAX;
			zArray_old[index] = zArray_old[index] + Z_MAX;
		end
		if (zArray[index] > Z_MAX)
			zArray[index] = zArray[index] - Z_MAX;
			zArray_old[index] = zArray_old[index] - Z_MAX;
		end

		# Implement the boundary conditions in the x-direction
		if (qArray[index] != 0.0)
			if (xArray[index] < 0.0)
				if (qArray[index] > 0.0)
					# Remove the particle
					remove_ion_flag = 1;
				end
				if (qArray[index] < 0.0)
					# Mirror the particle
					VxArray[index] = -VxArray[index];
					VxArray_old[index] = -VxArray_old[index];
					xArray[index] = xArray_old[index];
				end
			end
			if (xArray[index] > X_MAX)
				if (qArray[index] > 0.0)
					#Remove the particle
					remove_ion_flag = 1;
				end
				if (qArray[index] < 0.0)
					# Reset the particle at the cathode plane
					xArray[index] = xArray[index] - X_MAX;
					xArray_old[index] = xArray_old[index] - X_MAX;

					# Randomize the velocity based on a Maxwellian temperature distribution (Box-Muller technique)
					y1 = rand();
					while (y1 == 0.0)
						y1 = rand();
					end
					y2 = rand();
					VxArray[index] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*
                        sqrt(1.38064852e-23*4.0*11604.505/mArray[index]);
					y1 = rand();
					while (y1 == 0.0)
						y1 = rand();
					end
					y2 = rand();
					VyArray[index] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*
                        sqrt(1.38064852e-23*4.0*11604.505/mArray[index]);
					y1 = rand();
					while (y1 == 0.0)
						y1 = rand();
					end
					y2 = rand();
					VzArray[index] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*
                        sqrt(1.38064852e-23*4.0*11604.505/mArray[index]);

					# Make sure the particle is travelling into the domain
					if (VxArray[index] < 0.0)
						VxArray[index] = -VxArray[index];
					end

					VxArray_old[index] = VxArray[index];
					VyArray_old[index] = VyArray[index];
					VzArray_old[index] = VzArray[index];
				end
			end
		else
			if (xArray[index] < 0.0 || xArray[index] > X_MAX)
				# Reflect the particle
				xArray[index] = xArray_old[index];
				VxArray[index] = -VxArray[index];
				VxArray_old[index] = -VxArray_old[index];
			end
		end

		# Remove the ions
		if (remove_ion_flag == 1)
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);

			# Increment the ion counter
			ion_counter += 1;
		elseif ( remove_electron_flag == 1 )
            # Remove the electrons
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);
		elseif ( remove_neutral_flag == 1 )
            # Remove the neutrals
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);
        else
            index += 1;
        end
	end

    # Return the number of ions lost from the domain
	Int32(ion_counter);
end

# Rectangular boundary conditions for z pinch
function boundaryConditionsRectangularZPinch(xArray, yArray, zArray,
	VxArray, VyArray, VzArray, xArray_old, yArray_old, zArray_old,
	VxArray_old, VyArray_old, VzArray_old, ExArray, EyArray, EzArray,
	BxArray, ByArray, BzArray, mArray, qArray, vdwrArray, localNn, localNe,
	localNi, localCn, localCe, localCi, cellNumber, superParticleSizeArray,
    X_MAX, Y_MAX, Z_MAX)

	# Clear the ion counter
	ion_counter = 0;

	# Loop through all particles
    index = 1;
	while (index <= length(xArray))
		# Set the initial state of the remove ion and electron flags
		remove_ion_flag = 0;
		remove_electron_flag = 0;

		# Implement the y and z boundary conditions as periodic
		if (yArray[index] < 0.0)
			yArray[index] = yArray[index] + Y_MAX;
			yArray_old[index] = yArray_old[index] + Y_MAX;
		end
		if (yArray[index] > Y_MAX)
			yArray[index] = yArray[index] - Y_MAX;
			yArray_old[index] = yArray_old[index] - Y_MAX;
		end
		if (zArray[index] < 0.0)
			zArray[index] = zArray[index] + Z_MAX;
			zArray_old[index] = zArray_old[index] + Z_MAX;
		end
		if (zArray[index] > Z_MAX)
			zArray[index] = zArray[index] - Z_MAX;
			zArray_old[index] = zArray_old[index] - Z_MAX;
		end

		# Implement the boundary conditions in the x-direction
		if( xArray[index] < 0.0 )
			# Reflect the particle
			xArray[index] = xArray_old[index];
			VxArray[index] = -VxArray[index];
			VxArray_old[index] = -VxArray_old[index];
		end
		if( xArray[index] > X_MAX )
			if (qArray[index] > 0.0)
				# Remove the particle
				remove_ion_flag = 1;
			end
			if (qArray[index] < 0.0)
				# Remove the particle
				remove_electron_flag = 1;
			end
		end

		if (remove_ion_flag == 1)
            # Remove the ions
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);

			# Increment the ion counter
			ion_counter += 1;
		elseif (remove_electron_flag == 1)
            # Remove the electrons
            deleteat!(xArray, index);
            deleteat!(yArray, index);
            deleteat!(zArray, index);
            deleteat!(VxArray, index);
            deleteat!(VyArray, index);
            deleteat!(VzArray, index);
            deleteat!(xArray_old, index);
            deleteat!(yArray_old, index);
            deleteat!(zArray_old, index);
            deleteat!(VxArray_old, index);
            deleteat!(VyArray_old, index);
            deleteat!(VzArray_old, index);
            deleteat!(ExArray, index);
            deleteat!(EyArray, index);
            deleteat!(EzArray, index);
            deleteat!(BxArray, index);
            deleteat!(ByArray, index);
            deleteat!(BzArray, index);
            deleteat!(mArray, index);
            deleteat!(qArray, index);
            deleteat!(vdwrArray, index);
            deleteat!(localNn, index);
            deleteat!(localNe, index);
            deleteat!(localNi, index);
            deleteat!(localCn, index);
            deleteat!(localCe, index);
            deleteat!(localCi, index);
            deleteat!(cellNumber, index);
            deleteat!(superParticleSizeArray, index);
		else
            index += 1;
        end
	end

	# Return the number of ions lost from the domain
	Int32(ion_counter);
end

# Calculate the ideal number of cells based on the current
# plasma properties
function calculateNeededNumberOfCells(electric_constant, neArray,
    CeArray, N, X_MAX, Y_MAX)
	# Calculate the minimum Debye length
	electron_temperature = (CeArray[1]^2)*9.10938356e-31/
        (3.0*1.38064852e-23);
	if (electron_temperature == 0.0)
		electron_temperature = 10000.0;
	end
	electron_density = neArray[1];
	if (electron_density == 0.0)
		electron_density = 1.0;
	end
	debye_length = sqrt(electric_constant*1.3806485e-23*electron_temperature/
        (electron_density*(1.602176621e-19^2)));
    for index=2:N
        electron_temperature = (CeArray[index]^2)*9.10938356e-31/
            (3.0*1.38064852e-23);
		if (electron_temperature == 0.0)
			electron_temperature = 10000.0;
		end
		electron_density = neArray[index];
		if (electron_density == 0.0)
			electron_density = 1.0;
		end
		new_debye_length = sqrt(electric_constant*1.3806485e-23*electron_temperature/
            (electron_density*(1.602176621e-19^2)));
		if (new_debye_length < debye_length && new_debye_length >= 0.0)
			debye_length = new_debye_length;
		end
	end

	# Calculate the needed number of cells
	if (X_MAX > Y_MAX)
		number_of_cells = Int32(round(6.0*X_MAX/debye_length));
	else
		number_of_cells = Int32(round(6.0*Y_MAX/debye_length));
	end

	# Return the number of cells
	number_of_cells;
end

# Calculate the ideal time step based on the current plasma properties
function calculateNeededTimeStep(electric_constant, neArray, ByBinArray, N)

	# Calculate the maximum plasma frequency
	electron_density = neArray[1];
	if (electron_density == 0.0)
		electron_density = 1.0;
	end
	plasma_frequency = sqrt(electron_density*(1.602176621e-19^2)/
        (9.10938356e-31*electric_constant))/(2.0*3.14159265359);
    for index=2:N
		electron_density = neArray[index];
		if (electron_density == 0.0)
			electron_density = 1.0;
		end
		new_plasma_frequency = sqrt(electron_density*(1.602176621e-19^2)/
            (9.10938356e-31*electric_constant))/(2.0*3.14159265359);

		if (new_plasma_frequency > plasma_frequency)
			plasma_frequency = new_plasma_frequency;
		end
	end

	# Calculate the maximum electron gyrofrequency
	magnetic_field = abs(ByBinArray[1]);
	if (magnetic_field == 0.0)
		magnetic_field = 1.0e-10;
	end
	gyrofrequency = 1.602176621e-19*magnetic_field/
        (9.10938356e-31*2.0*3.14159265359);
    for index=2:N
		magnetic_field = abs(ByBinArray[index]);
		if (magnetic_field == 0.0)
			magnetic_field = 1.0e-10;
		end
		new_gyrofrequency = 1.602176621e-19*magnetic_field/
            (9.10938356e-31*2.0*3.14159265359);

		if (new_gyrofrequency > gyrofrequency)
			gyrofrequency = new_gyrofrequency;
		end
	end

	# Calculate the needed time step
	if (gyrofrequency > plasma_frequency)
		needed_time_step = 1.0 / (25.0*gyrofrequency);
    else
		needed_time_step = 1.0 / (25.0*plasma_frequency);
	end

	# Return the needed time step
	Float32(needed_time_step);
end

# Write the results to a text file
function writeToFile( phi, phi_output, neArray, uexArray, ueyArray, uezArray,
    niArray, uixArray, nnArray, CnArray, CeArray, CiArray, BxBinArray,
    ByBinArray, BzBinArray, neArray_output, uexArray_output, ueyArray_output,
    uezArray_output, niArray_output, uixArray_output, nnArray_output,
    CnArray_output, CeArray_output, CiArray_output, BxBinArray_output,
    ByBinArray_output, BzBinArray_output, N, output_counter,
    time_stamp, path )
    # Set the default return value
    my_return = 1;

	# Normalize the output data
	if (output_counter != 0)
		neArray_output ./= Float64(output_counter);
		uexArray_output ./= Float64(output_counter);
		ueyArray_output ./= Float64(output_counter);
		uezArray_output ./= Float64(output_counter);
		niArray_output ./= Float64(output_counter);
		uixArray_output ./= Float64(output_counter);
		nnArray_output ./= Float64(output_counter);
		CnArray_output ./= Float64(output_counter);
		CeArray_output ./= Float64(output_counter);
		CiArray_output ./= Float64(output_counter);
		BxBinArray_output ./= Float64(output_counter);
		ByBinArray_output ./= Float64(output_counter);
		BzBinArray_output ./= Float64(output_counter);
		phi_output ./= Float64(output_counter);
    else
		neArray_output .= neArray;
		uexArray_output .= uexArray;
		ueyArray_output .= ueyArray;
		uezArray_output .= uezArray;
		niArray_output .= niArray;
		uixArray_output .= uixArray;
		nnArray_output .= nnArray;
		CnArray_output .= CnArray;
		CeArray_output .= CeArray;
		CiArray_output .= CiArray;
		BxBinArray_output .= BxBinArray;
		ByBinArray_output .= ByBinArray;
		BzBinArray_output .= BzBinArray;
		phi_output .= phi;
	end

	# Open the output files
	stream1 = open("$(path)potential.txt", "a");
	stream2 = open("$(path)ne.txt", "a");
	stream3 = open("$(path)uex.txt", "a");
	stream4 = open("$(path)uey.txt", "a");
	stream5 = open("$(path)uez.txt", "a");
	stream6 = open("$(path)ni.txt", "a");
	stream7 = open("$(path)uix.txt", "a");
	stream8 = open("$(path)nn.txt", "a");
	stream9 = open("$(path)Cn.txt", "a");
	stream10 = open("$(path)Ce.txt", "a");
	stream11 = open("$(path)Ci.txt", "a");
	stream12 = open("$(path)Bx.txt", "a");
	stream13 = open("$(path)By.txt", "a");
	stream14 = open("$(path)Bz.txt", "a");

	# Write the results to file
	write(stream1, "$(time_stamp) ");
	write(stream2, "$(time_stamp) ");
	write(stream3, "$(time_stamp) ");
	write(stream4, "$(time_stamp) ");
	write(stream5, "$(time_stamp) ");
	write(stream6, "$(time_stamp) ");
	write(stream7, "$(time_stamp) ");
	write(stream8, "$(time_stamp) ");
	write(stream9, "$(time_stamp) ");
	write(stream10, "$(time_stamp) ");
	write(stream11, "$(time_stamp) ");
	write(stream12, "$(time_stamp) ");
	write(stream13, "$(time_stamp) ");
	write(stream14, "$(time_stamp) ");
    for index=1:N
		write(stream1, "$(phi_output[index]) ");
		write(stream2, "$(neArray_output[index]) ");
		write(stream3, "$(uexArray_output[index]) ");
		write(stream4, "$(ueyArray_output[index]) ");
		write(stream5, "$(uezArray_output[index]) ");
		write(stream6, "$(niArray_output[index]) ");
		write(stream7, "$(uixArray_output[index]) ");
		write(stream8, "$(nnArray_output[index]) ");
		write(stream9, "$(CnArray_output[index]) ");
		write(stream10, "$(CeArray_output[index]) ");
		write(stream11, "$(CiArray_output[index]) ");
		write(stream12, "$(BxBinArray_output[index]) ");
		write(stream13, "$(ByBinArray_output[index]) ");
		write(stream14, "$(BzBinArray_output[index]) ");
	end
	write(stream1, "\n");
	write(stream2, "\n");
	write(stream3, "\n");
	write(stream4, "\n");
	write(stream5, "\n");
	write(stream6, "\n");
	write(stream7, "\n");
	write(stream8, "\n");
	write(stream9, "\n");
	write(stream10, "\n");
	write(stream11, "\n");
	write(stream12, "\n");
	write(stream13, "\n");
	write(stream14, "\n");

	# Zero the output arrays
	neArray_output .= 0.0;
	uexArray_output .= 0.0;
	ueyArray_output .= 0.0;
	uezArray_output .= 0.0;
	niArray_output .= 0.0;
	uixArray_output .= 0.0;
	nnArray_output .= 0.0;
	CnArray_output .= 0.0;
	CeArray_output .= 0.0;
	CiArray_output .= 0.0;
	BxBinArray_output .= 0.0;
	ByBinArray_output .= 0.0;
	BzBinArray_output .= 0.0;
	phi_output .= 0.0;
	output_counter = 0;

	# Close the output file
	close(stream1);
	close(stream2);
	close(stream3);
	close(stream4);
	close(stream5);
	close(stream6);
	close(stream7);
	close(stream8);
	close(stream9);
	close(stream10);
	close(stream11);
	close(stream12);
	close(stream13);
	close(stream14);

    # Return the default value
    my_return;
end

# Set the electric field for all particles
function setElectricField( Ex, Ey, Ez, ExArray, EyArray, EzArray )
	# Set the default return value
    my_return = 1;

	# Set the field value for all partcles within the array
	ExArray .= Ex;
	EyArray .= Ey;
	EzArray .= Ez;

	# Return the default value
	my_return;
end

function temp()
    io = open("temp.txt", "w");
    write(io, "Hello World!");
    close(io);
end
