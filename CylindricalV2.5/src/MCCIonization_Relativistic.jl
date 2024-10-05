using CUDA
include("classDefinition.jl")
include("library.jl")
include("auxlibrary.jl")


function determineCollision_kernel(ionization_flags, alive, VelArray, VelArray_old, mArray, localCn, timeStep, qArray, localNn, WArray, EArray, EList, TList, WList, σList)
	idx = threadIdx().x + (blockIdx().x-1) * blockDim().x
	qe = 1.6021766208e-19
	c = 3e8
	me =  9.10938356E-31
    #Loop through all particles
    if idx <= length(alive)
		if alive[idx] == 1 
			# Calculate the ionization cross section (zero for anything that's not an electron)
			if localNn[idx] > 0.0
				if (qArray[idx] < 0.0) # if its an electron and neutral density is not zero
					Vmag = sqrt(VelArray[1,idx]^2 + VelArray[2,idx]^2 + VelArray[3,idx]^2)
					if Vmag >= 3e8
						Vmag *= 0.999999
					end
					γ = 1/ sqrt((1 - Vmag*Vmag / (3e8)^2))
					Ein = (γ -1)*mArray[idx]*(3e8)^2 #incident electron energy
					Ein = 1/2*mArray[idx]*Vmag^2;
					#2. find ionization cross-section
					Ein_eV = Ein/qe
					#ionization only possible for electron energy above first ionization energy
					if Ein_eV > 12.1298
						if isnan(Ein_eV)
							Ein_eV = 0.0
						end
						# Linearly interpolate the cross section
						σ = interp(σList,TList,Ein_eV)
						W = interp(WList,TList,Ein_eV)
						E = interp(EList,TList,Ein_eV)
						vrel = sqrt(VelArray[1,idx]^2 + VelArray[2,idx]^2 + VelArray[3,idx]^2 + localCn[idx]^2)

						# Determine statistically if an ionization has occured
						#probability = (1 - exp(- localNn[idx]*σ*vrel*timeStep))^0.1
						probability = 1 - exp(- localNn[idx]*σ*vrel*timeStep)			
						if rand() <= probability
							#Create a particle
							@inbounds ionization_flags[idx] = 1
							@inbounds WArray[idx] = W
							@inbounds EArray[idx] = E
							
							#update velocity
							vold1 = VelArray[1,idx]
							vold2 = VelArray[2,idx]
							vold3 = VelArray[3,idx]
							@inbounds VelArray_old[1,idx] = vold1
							@inbounds VelArray_old[2,idx] = vold2
							@inbounds VelArray_old[3,idx] = vold3

							#incident electron loses energy
							Ein -= E
							Ein = clamp(Ein, 0, Ein)
							θ = acos(1 - 2 * rand())  # Polar angle (0 to π)
							φ = 2 * π * rand()        # Azimuthal angle (0 to 2π)
							γ = Ein/(me*c^2) + 1
							VmagNew = sqrt(1 - 1 / γ^2) * c
							@inbounds VelArray[1, idx] = VmagNew * sin(θ) * cos(φ)
							@inbounds VelArray[2, idx] = VmagNew * sin(θ) * sin(φ)
							@inbounds VelArray[3, idx] = VmagNew * cos(θ)

						end
					end
				end
			end
		end
    end
    return
end

function XenonNeutralCollisionalIonization(myParticle, myDomain, timeStep, EList, TList, WList, σList, σLength, stepT)
	#1. initialize flag to store ionization position
	ionization_flags = CUDA.zeros(Int32, length(myParticle.alive))
	WArray = CUDA.zeros(Float64, length(myParticle.alive))
	EArray = CUDA.zeros(Float64, length(myParticle.alive))
	#myDomain.neutral_change .= 0.0
	#2. determine where ionization happened
	#println("determine collision: ")
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(myParticle.alive),256),
		determineCollision_kernel(ionization_flags, myParticle.alive, myParticle.VelArray, myParticle.VelArray_old, myParticle.mArray, myParticle.localCn, timeStep, myParticle.qArray, myParticle.localNn, WArray, EArray, EList, TList, WList, σList, σLength, stepT))
	
	NumIonized = reduce(+,ionization_flags)
	
	#3. find index of 1 in ionization_flags
	ionization_index = findall(x -> x == 1, ionization_flags)
	if NumIonized > 0
		#println("Numionized: ", NumIonized)
		#4. create electron-ion pair
		#the new particles have the super size of smaller super size between mean neutral size in cell and electron super size
		#println("creating ion electron pair")
		createIon_ElectronPair(ionization_index, myParticle.alive, NumIonized, myParticle.PosArray,myParticle.VelArray, myParticle.PosArray_old, myParticle.VelArray_old,myParticle.mArray,myParticle.qArray,myParticle.superParticleSizeArray, myParticle.vdwrArray, myParticle.rCellNum, myParticle.zCellNum,WArray, myParticle.localCn, myDomain.mean_neutralSuperSize)
	
		#5. reduce neutral superparticle around ionized particle
		#first add reduction to node
		#println("scatter reduction to nodes ")
		CUDA.@sync @cuda(
			threads=256,
			blocks=cld(length(ionization_flags), 256),
			scatterReductionToNodes(myParticle.rCellNum, myParticle.zCellNum, myParticle.superParticleSizeArray, ionization_flags, myDomain.neutral_change, myDomain.mean_neutralSuperSize)
		)

		#index of all alive
		aliveIdx = CUDA.findall(x->x==1, myParticle.alive)

		#index of all neutrals
		neutral_subIdx = CUDA.findall(x->x==0.0, myParticle.qArray[aliveIdx])
		neutralIdx = aliveIdx[neutral_subIdx]
		#println("reducing neutral super particle sizd")
		reduceNeutralSuperParticle(neutralIdx, myDomain.neutral_change, myParticle.rCellNum, myParticle.zCellNum, myParticle.superParticleSizeArray, myParticle.alive)

	end
	ionization_index = nothing
	ionization_flags = nothing
	return NumIonized
end

function scatterReductionToNodes(rCellNum, zCellNum, superParticleSizeArray, ionization_flags,neutral_change,mean_neutralSuperSize)
	#1. loop through ionization_index
	idx = threadIdx().x + (blockIdx().x-1) * blockDim().x
	if idx <= length(ionization_flags)
		if ionization_flags[idx] == 1
			#ionization happens at idx
			#2. get ionization location at current index
			rCell = rCellNum[idx]
			zCell = zCellNum[idx]
			#3. calculate reduction in superparticle at this ionization location
			Wn = gather(mean_neutralSuperSize, rCell, zCell)
			# change_in_num_of_neutrals = -superparticle[idx]
			# reduce the smaller weight between incident electron and background neutral
			Δparticle = -min(superParticleSizeArray[idx], Wn)
			#@cuprintln("superparticle size: ", superParticleSizeArray[idx])
			#@cuprintln("Wn: ", Wn)
			#4. scatter superparticle reduction to surrounding nodes
			scatter(neutral_change, rCell, zCell, Δparticle)
		end
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
	#println("sum(neutral_change)", sum(neutral_change))
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

function XenonNeutralCollisionalIonization_optimized(myParticle, myDomain, timeStep, EList, TList, WList, σList)
	#1. initialize flag to store ionization position
	ionization_flags = CUDA.zeros(Int32, length(myParticle.alive))
	WArray = CUDA.zeros(Float64, length(myParticle.alive))
	EArray = CUDA.zeros(Float64, length(myParticle.alive))
	#myDomain.neutral_change .= 0.0
	#2. determine where ionization happened
	#println("determine collision: ")
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(myParticle.alive),256),
		determineCollision_kernel(ionization_flags, myParticle.alive, myParticle.VelArray, myParticle.VelArray_old, myParticle.mArray, myParticle.localCn, timeStep, myParticle.qArray, myParticle.localNn, WArray, EArray, EList, TList, WList, σList))
	
	NumIonized = CUDA.sum(+,ionization_flags)
	
	#3. find index of 1 in ionization_flags
	ionization_index = CUDA.findall(x -> x == 1, ionization_flags)
	if NumIonized > 0
		#println("Numionized: ", NumIonized)
		#4. create electron-ion pair
		#the new particles have the super size of smaller super size between mean neutral size in cell and electron super size
		#println("creating ion electron pair")
		createIon_ElectronPair(ionization_index, myParticle.alive, NumIonized, myParticle.PosArray,myParticle.VelArray, myParticle.PosArray_old, myParticle.VelArray_old,myParticle.mArray,myParticle.qArray,myParticle.superParticleSizeArray, myParticle.vdwrArray, myParticle.rCellNum, myParticle.zCellNum,WArray, myParticle.localCn, myDomain.mean_neutralSuperSize)
		
		#5. reduce neutral superparticle around ionized particle
		#first add reduction to node
		#println("scatter reduction to nodes ")
		CUDA.@sync @cuda(
			threads=256,
			blocks=cld(length(ionization_flags), 256),
			scatterReductionToNodes(myParticle.rCellNum, myParticle.zCellNum, myParticle.superParticleSizeArray, ionization_flags, myDomain.neutral_change, myDomain.mean_neutralSuperSize)
		)

		#reduce by each color of cell:
		for color = 1:4
			if sum(myDomain.neutral_change) <0
				#println(sum(myDomain.neutral_change))
				#6. bin particles into each cell
				aliveidx = CUDA.findall(x->x==1,myParticle.alive)
				if color == 1
					color_subidx = CUDA.findall(x->x==1, myParticle.localColor[aliveidx])
				elseif color == 2
					color_subidx = CUDA.findall(x->x==2, myParticle.localColor[aliveidx])
				elseif color == 3
					color_subidx = CUDA.findall(x->x==3, myParticle.localColor[aliveidx])
				elseif color ==4
					color_subidx = CUDA.findall(x->x==4, myParticle.localColor[aliveidx])
				end
				#println("Running color = ", color)
				coloridx = aliveidx[color_subidx]
				neutral_subidx = CUDA.findall(x->x==0,myParticle.qArray[coloridx])
				neutralidx = coloridx[neutral_subidx]
				if length(neutralidx) > 0 # if there are neutrals to be reduced
					#println("number of neutrals = ", length(neutralidx))
					#proceed only if there are neutrals to handle
					unified_k = (floor.(myParticle.rCellNum[neutralidx]).-1).*myDomain.Nz .+ floor.(myParticle.zCellNum[neutralidx])
					
					#sort neutralIdx based on k value:

					# Convert unified_k to Int64 if not already
					unified_k = convert.(Int64, unified_k)

					# Get the sorting permutation
					permutation = CUDA.sortperm(unified_k)

					# Apply permutation to unified_k and neutralidx
					sorted_unified_k = unified_k[permutation]
					sorted_neutralidx = neutralidx[permutation]

					# Get unique cell indices
					unique_cells = CUDA.unique(sorted_unified_k)

					# Initialize cell_start and cell_end arrays
					num_cells = maximum(unique_cells) + 1 
					#cell_start[k] = idx of the first particle in cell k, in sorted_neutralidx array
					cell_start = CuArray(zeros(Int64, num_cells))
					#cell_end[k] = idx of the last particle in cell k, in sorted_neutralidx array
					cell_end = CuArray(zeros(Int64, num_cells))

					# Launch kernels
					num_blocks = cld(length(sorted_unified_k), 256)

					@cuda threads=256 blocks=num_blocks find_cell_start!(cell_start, sorted_unified_k)
					@cuda threads=256 blocks=num_blocks find_cell_end!(cell_end, sorted_unified_k)
					#7. parallel loop each cell, sequentially loop each particle in each cell to gather and scatter neutral change
					exitFlag = CuArray([0])
					@cuda threads=256 blocks=cld(length(cell_start), 256) applyNeutralReduction_kernel(cell_start, cell_end, sorted_neutralidx, myDomain.neutral_change, myParticle.rCellNum, myParticle.zCellNum, myParticle.superParticleSizeArray, myParticle.alive, exitFlag)
					#println(sum(myDomain.neutral_change))
				end
			end
		end
	end
	return NumIonized
end

# Kernel to find cell_start positions
function find_cell_start!(cell_start, sorted_unified_k)
	idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	if idx <= length(sorted_unified_k)
		if idx == 1 || sorted_unified_k[idx] != sorted_unified_k[idx - 1]
			cell_idx = sorted_unified_k[idx]
			@inbounds cell_start[cell_idx + 1] = idx
		end
	end
	return
end

# Similarly for cell_end positions
function find_cell_end!(cell_end, sorted_unified_k)
	idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	if idx <= length(sorted_unified_k)
		if idx == length(sorted_unified_k) || sorted_unified_k[idx] != sorted_unified_k[idx + 1]
			cell_idx = sorted_unified_k[idx]
			@inbounds cell_end[cell_idx + 1] = idx 
		end
	end
	return
end

function applyNeutralReduction_kernel(cell_start, cell_end, sorted_neutralidx, neutral_change, rCellNum, zCellNum, superParticleSizeArray, alive, exitFlag)
	#loop through each cell, only length of cellstart or cellend need to be looped.
	Kidx = threadIdx().x + (blockIdx().x-1) * blockDim().x
	if Kidx <= length(cell_start) && exitFlag[1] == 0
		#if there is a starting index in sorted_neutralidx
		if cell_start[Kidx] > 0 && exitFlag[1] == 0
			for idx in cell_start[Kidx]:cell_end[Kidx]
				if exitFlag[1] == 1
					break
				end
				particleIdx = sorted_neutralidx[idx]
				#start reducing neutral mass
				Δparticle = gather(neutral_change, rCellNum[particleIdx], zCellNum[particleIdx])
				#3. clamp reduction by superparticle size
				Δparticle = clamp(Δparticle, -superParticleSizeArray[particleIdx], 0)
				#4. reduce superparticle size
				#@cuprintln("Δparticle = ",  Δparticle)
				if Δparticle < 0
					#if superparticle size equals to number of particle to be reduced, remove this superparticle
					@inbounds CUDA.@atomic superParticleSizeArray[particleIdx] += Δparticle
					
					#delete neutral particles with super size 0:
					if superParticleSizeArray[particleIdx] == 0
						@inbounds alive[particleIdx] = 0
					end
					#update neutral change at the node, increment by the num of particle removed
					scatter(neutral_change, rCellNum[particleIdx], zCellNum[particleIdx], -Δparticle)
				end
				if CUDA.sum(neutral_change) >= 0
					CUDA.@atomic exitFlag[1] += 1
					break
				end
			end
		end
	end
	return
end


function createIon_kernel(alive, prefixsum, ionization_index,NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, localCn, ve2Vector, mean_neutralSuperSize)
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
			
			theta = acos(2*rand()-1)
			phi = 2*pi*rand()

			#velocity of neutral before collision, a random direction
			vn1 = sin(theta)*cos(phi)*localCn[i]
			vn2 = sin(theta)*sin(phi)*localCn[i]
			vn3 = cos(theta)*localCn[i]
				
			#velocity of incident electron after collision
			ve1 = VelArray[1,i]
			ve2 = VelArray[2,i]
			ve3 = VelArray[3,i]

			#velocity of incident electron before collision
			ve1_old = VelArray_old[1,i]
			ve2_old = VelArray_old[2,i]
			ve3_old = VelArray_old[3,i]

			#velocity of secondary elecctron
			ve21 = ve2Vector[1,i]
			ve22 = ve2Vector[2,i]
			ve23 = ve2Vector[3,i]

			#use momentum conservation to calculate velocity after collision
			me =  9.10938356E-31;
			mi = 2.1802e-25;

			vi1 = me/mi*(ve1_old - ve1 -ve21) + vn1
			vi2 = me/mi*(ve2_old - ve2 -ve22) + vn2
			vi3 = me/mi*(ve3_old - ve3 -ve23) + vn3

			@inbounds VelArray[1,j] = vi1
			@inbounds VelArray[2,j] = vi2
			@inbounds VelArray[3,j] = vi3

			@inbounds PosArray_old[1,j] = PosArray[1,j]
			@inbounds PosArray_old[2,j] = PosArray[2,j]
			@inbounds PosArray_old[3,j] = PosArray[3,j]

			@inbounds VelArray_old[1,j] = VelArray[1,j]
			@inbounds VelArray_old[2,j] = VelArray[2,j]
			@inbounds VelArray_old[3,j] = VelArray[3,j]

			@inbounds mArray[j] = 2.1802e-25
			@inbounds qArray[j] = 1.6021766208e-19
			Wn = gather(mean_neutralSuperSize, rCellNum[i], zCellNum[i])
			@inbounds superParticleSizeArray[j] = min(superParticleSizeArray[i], Wn)
			@inbounds vdwrArray[j] = 216.0e-12

			@inbounds rCellNum[j] = rCellNum[i]
			@inbounds zCellNum[j] = zCellNum[i]
		end
	end
	return
end

function createElectron_kernel(alive, prefixsum, ionization_index,NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, WArray, ve2Vector, mean_neutralSuperSize)
	j = threadIdx().x + (blockIdx().x-1) * blockDim().x#Corresponds to the length of alive

	#Loop through all particles
	if j <= length(alive)
		if alive[j] == 0 && prefixsum[j] <= NumIonized
			#loop through ionization flag
			i = ionization_index[prefixsum[j]]
			#Create one electron at the location of ionization
			alive[j] = 1
			@inbounds PosArray[1,j] = PosArray[1,i]
			@inbounds PosArray[2,j] = PosArray[2,i]
			@inbounds PosArray[3,j] = PosArray[3,i]

			@inbounds PosArray_old[1,j] = PosArray[1,j]
			@inbounds PosArray_old[2,j] = PosArray[2,j]
			@inbounds PosArray_old[3,j] = PosArray[3,j]

			me =  9.10938356E-31;
			#secondary electron energy
			W = WArray[i]
			VMagNew = sqrt(2*W/me)

			theta = acos(2*rand()-1)
			phi = 2*pi*rand()

			v1 = sin(theta)*cos(phi)
			v2 = sin(theta)*sin(phi)
			v3 = cos(theta)

			#store secondary electron vector
			ve2Vector[1,i] = v1*VMagNew
			ve2Vector[2,i] = v2*VMagNew
			ve2Vector[3,i] = v3*VMagNew

			@inbounds VelArray[1,j] = v1*VMagNew
			@inbounds VelArray[2,j] = v2*VMagNew
			@inbounds VelArray[3,j] = v3*VMagNew

			@inbounds VelArray_old[1,j] = VelArray[1,j]
			@inbounds VelArray_old[2,j] = VelArray[2,j]
			@inbounds VelArray_old[3,j] = VelArray[3,j]


			@inbounds mArray[j] = 9.10938356E-31
			@inbounds qArray[j] = -1.6021766208e-19

			Wn = gather(mean_neutralSuperSize, rCellNum[i], zCellNum[i])

			@inbounds superParticleSizeArray[j] = min(superParticleSizeArray[i], Wn)
			@inbounds vdwrArray[j] = 0.0

			@inbounds rCellNum[j] = rCellNum[i]
			@inbounds zCellNum[j] = zCellNum[i]
		end
	end
	return
end

function createIon_ElectronPair(ionization_index, alive, NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, WArray, localCn, mean_neutralSuperSize)
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

	ve2Vector= CUDA.zeros(Float64, 3, length(alive))
	#5.create electrons with random vector
	CUDA.@sync @cuda(
		threads=256,
		blocks=cld(length(alive),256),
    createElectron_kernel(alive, prefixsum, ionization_index,NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, WArray, ve2Vector, mean_neutralSuperSize)
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
    createIon_kernel(alive, prefixsum, ionization_index, NumIonized,PosArray,VelArray, PosArray_old, VelArray_old,mArray,qArray,superParticleSizeArray, vdwrArray, rCellNum, zCellNum, localCn, ve2Vector, mean_neutralSuperSize)
    )
end