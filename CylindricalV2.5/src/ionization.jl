using CUDA
include("classDefinition.jl")
include("library.jl")
include("auxlibrary.jl")


function determineCollision_kernel(ionization_flags,cross_section_energy_eV, cross_section_area, alive, VelArray, mArray, localCn, timeStep, qArray, localNn)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	
    #Loop through all particles
    if idx <= length(alive)
		if alive[idx] == 1
			# Calculate the ionization cross section (zero for anything that's not an electron)

			V_squared = VelArray[1,idx]^2 + VelArray[2,idx]^2 + VelArray[3,idx]^2
			if (qArray[idx] < 0.0)
				kinetic_energy_eV = (0.5*(V_squared + localCn[idx]^2)*mArray[idx]) / 1.602176621e-19;
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
			collision_frequency = cross_section*sqrt(V_squared + localCn[idx]^2)*localNn[idx];

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
				#Create a particle
				ionization_flags[idx] = 1
			end
		end
    end
    return
end



function fixedTemperatureXenonIonization(myParticle, myDomain, timeStep, cross_section_energy_eV, cross_section_area, Ti, Te)
	#1. initialize flag to store ionization position
	ionization_flags = CUDA.zeros(Int32, length(myParticle.alive))
	
	#2. determine where ionization happened
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(myParticle.alive),256),
		determineCollision_kernel(ionization_flags,cross_section_energy_eV, cross_section_area, myParticle.alive, myParticle.VelArray, myParticle.mArray, myParticle.localCn, timeStep, myParticle.qArray, myParticle.localNn))
	NumIonized = reduce(+,ionization_flags)
	
	#3. find index of 1 in ionization_flags
	ionization_index = findall(x -> x == 1, ionization_flags)
	if NumIonized > 0
		#4. create electron-ion pair
		createIon_ElectronPair(ionization_index, myParticle.alive, NumIonized, myParticle.PosArray,myParticle.VelArray, myParticle.PosArray_old, myParticle.VelArray_old,myParticle.mArray,myParticle.qArray,myParticle.superParticleSizeArray, myParticle.vdwrArray, myParticle.rCellNum, myParticle.zCellNum,Te, Ti)
	
		#5. reduce neutral superparticle around ionized particle
		#first add reduction to node
		CUDA.@sync @cuda(
			threads=256,
			blocks=cld(length(ionization_index), 256),
			scatterReductionToNodes(myParticle.rCellNum, myParticle.zCellNum, myParticle.superParticleSizeArray, ionization_index, myDomain.neutral_change)
		)

		#index of all alive
		aliveIdx = findall(x->x==1, myParticle.alive)

		#index of all neutrals
		neutralIdx = findall(x->x==0.0, myParticle.qArray[aliveIdx])

		reduceNeutralSuperParticle(neutralIdx, myDomain.neutral_change, myParticle.rCellNum, myParticle.zCellNum, myParticle.superParticleSizeArray, myParticle.alive)
	end
	ionization_index = nothing
	ionization_flags = nothing
	return NumIonized
end

function scatterReductionToNodes(rCellNum, zCellNum, superParticleSizeArray, ionization_index,neutral_change)
	#1. loop through ionization_index
	idx = threadIdx().x + (blockIdx().x-1) * blockDim().x
	if idx <= length(ionization_index)
		#2. get ionization location at current index
		rCell = rCellNum[idx]
		zCell = zCellNum[idx]
		#3. calculate reduction in superparticle at this ionization location
		# change_in_num_of_neutrals = -superparticle[idx]
		Δparticle = -superParticleSizeArray[idx]
		#4. scatter superparticle reduction to surrounding nodes
		scatter(neutral_change, rCell, zCell, Δparticle)
	end
	return
end

function reduceNeutralSuperParticle(idxloop, neutral_change, rCellNum, zCellNum, superParticleSizeArray, alive)
	#1. loop through neutral particles
	i = 1
	while sum(neutral_change) < 0 && i <= length(idxloop)
		idx  = idxloop[i]
		CUDA.@sync @cuda(
		threads=1,
		blocks=1,
		reduceNeutralSuperPaticle_kernel(idx, neutral_change, rCellNum, zCellNum, superParticleSizeArray, alive)
		)
		i += 1
	end
	println("sum(neutral_change)", sum(neutral_change))
end

function reduceNeutralSuperPaticle_kernel(idx, neutral_change, rCellNum, zCellNum, superParticleSizeArray, alive)
	#2. gather neutral_change from surrounding nodes
	Δparticle = gather(neutral_change, rCellNum[idx], zCellNum[idx])
	#3. clamp reduction by superparticle size
	Δparticle = clamp(Δparticle, -superParticleSizeArray[idx], 0)
	#4. reduce superparticle size
	#@cuprintln("Δparticle = ",  Δparticle)
	if Δparticle < 0
		#if superparticle size equals to number of particle to be reduced, remove this superparticle
		@inbounds superParticleSizeArray[idx] += Δparticle
		
		#delete neutral particles with super size 0:
		if superParticleSizeArray[idx] == 0
			alive[idx] = 0
		end
		#update neutral change at the node, increment by the num of particle removed
		scatter(neutral_change, rCellNum[idx], zCellNum[idx], -Δparticle)
	end
	return
end

function createIon_kernel(alive, prefixsum, ionization_index,NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, Ti)
	j = threadIdx().x + (blockIdx().x-1) * blockDim().x#Corresponds to the length of alive

	#Loop through all particles
	if j <= length(alive)
		if alive[j] == 0 && prefixsum[j] <= NumIonized
			#loop through ionization flag
			#index i is the index of the ionized particle, new particle is assigned with same position as the ionized particle

			i = ionization_index[prefixsum[j]]
			#Create one ion
			alive[j] = 1
			@inbounds PosArray[1,j] = PosArray[1,i]
			@inbounds PosArray[2,j] = PosArray[2,i]
			@inbounds PosArray[3,j] = PosArray[3,i]
			#
			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
			@inbounds VelArray[1,j] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Ti/2.1802e-25)

			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
			VelArray[2,j] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Ti/2.1802e-25)

			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
			VelArray[3,j] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Ti/2.1802e-25)

			@inbounds PosArray_old[1,j] = PosArray[1,j]
			@inbounds PosArray_old[2,j] = PosArray[2,j]
			@inbounds PosArray_old[3,j] = PosArray[3,j]

			@inbounds VelArray_old[1,j] = VelArray[1,j]
			@inbounds VelArray_old[2,j] = VelArray[2,j]
			@inbounds VelArray_old[3,j] = VelArray[3,j]

			@inbounds mArray[j] = 2.1802e-25
			@inbounds qArray[j] = 1.6021766208e-19
			@inbounds superParticleSizeArray[j] = superParticleSizeArray[i]
			@inbounds vdwrArray[j] = 216.0e-12

			@inbounds rCellNum[j] = rCellNum[i]
			@inbounds zCellNum[j] = zCellNum[i]
		end
	end
	return
end

function createElectron_kernel(alive, prefixsum, ionization_index,NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, Te)
	j = threadIdx().x + (blockIdx().x-1) * blockDim().x#Corresponds to the length of alive

	#Loop through all particles
	if j <= length(alive)
		if alive[j] == 0 && prefixsum[j] <= NumIonized
			#loop through ionization flag
			i = ionization_index[prefixsum[j]]
			#Create one ion
			alive[j] = 1
			@inbounds PosArray[1,j] = PosArray[1,i]
			@inbounds PosArray[2,j] = PosArray[2,i]
			@inbounds PosArray[3,j] = PosArray[3,i]
			#
			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
			@inbounds VelArray[1,j] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Te/2.1802e-25)

			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
			VelArray[2,j] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Te/9.10938356E-31)

			y1 = rand();
			while (y1 == 0.0)
				y1 = rand();
			end
			y2 = rand();
			VelArray[3,j] = sqrt(-2.0*log(y1))*cos(2.0*pi*y2)*sqrt(1.38064852e-23*Te/9.10938356E-31)

			@inbounds PosArray_old[1,j] = PosArray[1,j]
			@inbounds PosArray_old[2,j] = PosArray[2,j]
			@inbounds PosArray_old[3,j] = PosArray[3,j]

			@inbounds VelArray_old[1,j] = VelArray[1,j]
			@inbounds VelArray_old[2,j] = VelArray[2,j]
			@inbounds VelArray_old[3,j] = VelArray[3,j]

			@inbounds mArray[j] = 9.10938356E-31
			@inbounds qArray[j] = -1.6021766208e-19
			@inbounds superParticleSizeArray[j] = superParticleSizeArray[i]
			@inbounds vdwrArray[j] = 0.0

			@inbounds rCellNum[j] = rCellNum[i]
			@inbounds zCellNum[j] = zCellNum[i]
		end
	end
	return
end

function createIon_ElectronPair(ionization_index, alive, NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum,Te, Ti)
	#1. initializa empty flag
	empty_flags = CUDA.zeros(Int32, length(alive))
	

	#3. mask alive array
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(alive),256),
		mark_empty_spaces!(empty_flags, alive))

	#4. calculate prefixsum
	prefixsum = CUDA.zeros(Int32, length(alive))
	CUDA.scan!(+, prefixsum, empty_flags, dims=1)

	#5.create electrons
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(alive),256),
    createElectron_kernel(alive, prefixsum, ionization_index,NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum,Te)
    )

	#6.recalculate prefixsum using updated alive
	empty_flags = CUDA.zeros(Int32, length(alive))
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(alive),256),
		mark_empty_spaces!(empty_flags, alive))
	prefixsum = CUDA.zeros(Int32, length(alive))
	CUDA.scan!(+, prefixsum, empty_flags, dims=1)

	#7.create ions
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(alive),256),
    createIon_kernel(alive, prefixsum, ionization_index, NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum,Ti)
    )

	prefixsum = nothing
	empty_flags = nothing
end