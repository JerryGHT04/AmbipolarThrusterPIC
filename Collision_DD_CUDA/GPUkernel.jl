using CUDA

# Define the CUDA kernel
function assign_first_last!(d_B, V_first, V_last)
    # Calculate the global thread index
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    # Assign the value to the first element
    if i == 1
        d_B[1] = V_first
    end

    # Assign the value to the last element
    if i == length(d_B)
        d_B[end] = V_last
    end
    return nothing
end

# Example usage
function PoissonEquation_kernel(V_first::Float64, V_last::Float64, d_B::CuArray)
    N = length(d_B)  # Size of the CuArray

    # Define the number of threads per block and number of blocks
    threads_per_block = 256
    num_blocks = ceil(Int, N / threads_per_block)

    # Launch the kernel
    @cuda threads=threads_per_block blocks=num_blocks assign_first_last!(d_B, V_first, V_last)
end


# CUDA Kernel for boundary conditions
function boundary_conditions_kernel!(
    xArray, yArray, zArray, VxArray, VyArray, VzArray,
    xArray_old, yArray_old, zArray_old, VxArray_old, VyArray_old, VzArray_old,
    qArray, ion_remove_flags, electron_remove_flags, X_MAX, Y_MAX, Z_MAX
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    # Ensure within bounds
    if idx <= length(xArray)
        # Apply periodic boundary conditions for y and z directions
        if yArray[idx] < 0.0
            yArray[idx] += Y_MAX
            yArray_old[idx] += Y_MAX
        elseif yArray[idx] > Y_MAX
            yArray[idx] -= Y_MAX
            yArray_old[idx] -= Y_MAX
        end

        if zArray[idx] < 0.0
            zArray[idx] += Z_MAX
            zArray_old[idx] += Z_MAX
        elseif zArray[idx] > Z_MAX
            zArray[idx] -= Z_MAX
            zArray_old[idx] -= Z_MAX
        end

        # Implement boundary conditions in the x-direction
        if qArray[idx] != 0.0
            if xArray[idx] < 0.0 || xArray[idx] > X_MAX
                if qArray[idx] > 0.0
                    ion_remove_flags[idx] = 1  # Mark ion for removal
                elseif qArray[idx] < 0.0
                    electron_remove_flags[idx] = 1  # Mark electron for removal
                end
            end
        else
            if xArray[idx] < 0.0 || xArray[idx] > X_MAX
                # Reflect the particle
                xArray[idx] = xArray_old[idx]
                VxArray[idx] = -VxArray[idx]
                VxArray_old[idx] = -VxArray_old[idx]
            end
        end
    end
    return nothing
end