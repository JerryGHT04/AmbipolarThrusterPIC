using CUDA
include("classDefinition.jl")

function find_nth_zero_index(alive, N::Int)
    M = length(alive)
    
    # Create a mask where 1 corresponds to zeros in the original array
    zero_mask = CUDA.map(x -> 1 - x, alive)

    # Perform a cumulative sum (prefix sum) on the mask
    prefix_sum = cumsum(zero_mask)

    # Find the first index where the prefix sum equals N
    idx_array = CUDA.map(x -> x == N ? 1 : 0, prefix_sum)

    # Reduce the index array to find the first occurrence
    idx = findfirst(x -> x == 1, idx_array)
    
    return idx
end

function gather(Array, i0, j0)
    #accept field array and index (location normalized by cell step), 
    #interpolate the scalar field at (i0,j0)

    #calculate node index
    i = CUDA.Int(CUDA.trunc(i0))
    j = CUDA.Int(CUDA.trunc(j0))
    #calculate distance between lc and node normalized by dr and dz
    di = i0 - i
    dj = j0 - j
    #interpolate    
   return (Array[i,j]*(1-di)*(1-dj) +
          Array[i+1,j]*(di)*(1-dj) + 
          Array[i,j+1]*(1-di)*(dj) + 
          Array[i+1,j+1]*(di)*(dj))
end

function scatter(Array, i0, j0, value)
    #calculatpe node index
    i = CUDA.Int(CUDA.trunc(i0))
    j = CUDA.Int(CUDA.trunc(j0))#calculate distance between lc and node normalized by dr and dz
    #calculate distance between lc and node normalized by dr and dz
    di = i0 - i
    dj = j0 - j
    
    CUDA.@atomic Array[i,j] += (1-di)*(1-dj)*value
    CUDA.@atomic Array[i+1,j] += (di)*(1-dj)*value
    CUDA.@atomic Array[i,j+1] += (1-di)*(dj)*value
    CUDA.@atomic Array[i+1,j+1] += (di)*(dj)*value
end

function scatter_kernel(Array, i0, j0, value)
    #calculatpe node index
    i = CUDA.Int(CUDA.trunc(i0))
    j = CUDA.Int(CUDA.trunc(j0))#calculate distance between lc and node normalized by dr and dz
    #calculate distance between lc and node normalized by dr and dz
    di = i0 - i
    dj = j0 - j
    
    CUDA.@atomic Array[i,j] += (1-di)*(1-dj)*value
    CUDA.@atomic Array[i+1,j] += (di)*(1-dj)*value
    CUDA.@atomic Array[i,j+1] += (1-di)*(dj)*value
    CUDA.@atomic Array[i+1,j+1] += (di)*(dj)*value
end


function cross_kernel(A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if i <= size(A, 2)
        C[1, i] = A[2, i] * B[3, i] - A[3, i] * B[2, i]
        C[2, i] = A[3, i] * B[1, i] - A[1, i] * B[3, i]
        C[3, i] = A[1, i] * B[2, i] - A[2, i] * B[1, i]
    end
    return nothing
end

# Wrapper function to launch the kernel
function mycross(A::CuArray, B::CuArray)
    # A and B are expected to be 3xN matrices, where each column is a 3D vector
    @assert size(A, 1) == 3 && size(B, 1) == 3 "Input arrays must be 3xN matrices"

    N = size(A, 2)
    C = CUDA.zeros(3, N)  # Initialize the result array

    # Define block and grid dimensions
    threads_per_block = 256
    blocks_per_grid = ceil(Int, N / threads_per_block)

    # Launch the kernel
    @cuda threads=threads_per_block blocks=blocks_per_grid cross_kernel(A, B, C)

    return C
end

function mark_empty_spaces!(empty_flags, alive)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(alive)
        empty_flags[idx] = 1 - alive[idx]  # 1 if alive[idx] is 0, else 0
    end
    return
end

@inline function interp(yList, xList, xval)
    # Assume xList is sorted and has uniform spacing
    stepT = xList[2] - xList[1]
    
    # Compute the index 'j' for interpolation
    j = Int(floor((xval - xList[1]) / stepT)) + 1
    listLength = length(xList);
    # Handle edge cases
    if j < 1
        return yList[1]
    elseif j >= listLength
        return yList[listLength]
    else
        # Perform linear interpolation
        x0 = xList[j]
        x1 = xList[j + 1]
        y0 = yList[j]
        y1 = yList[j + 1]
        t = (xval - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    end
end