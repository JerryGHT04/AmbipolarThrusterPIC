using CUDA
include("classDefinition.jl")

function accumualteOutputData(myDomain, myData)
    myData.phi_output += myDomain.phi
    myData.neArray_output += myDomain.neArray
    myData.niArray_output += myDomain.niArray
    myData.nnArray_output += myDomain.nnArray
    myData.uixArray_output += myDomain.uixArray
    myData.uiyArray_output += myDomain.uiyArray
    myData.uizArray_output += myDomain.uizArray
    myData.uexArray_output += myDomain.uexArray
    myData.ueyArray_output += myDomain.ueyArray
    myData.uezArray_output += myDomain.uezArray
    myData.energy_output += myData.energy
    myData.thrust += myDomain.momentumOut
end

function writeToFile(myData::DataLogger, output_counter, path, time_stamp, timeStep)
    myData.phi_output ./= output_counter
    myData.neArray_output ./= output_counter
    myData.niArray_output ./= output_counter
    myData.nnArray_output ./= output_counter
    myData.uixArray_output ./= output_counter
    myData.uiyArray_output ./= output_counter
    myData.uizArray_output ./= output_counter
    myData.uexArray_output ./= output_counter
    myData.ueyArray_output ./= output_counter
    myData.uezArray_output ./= output_counter
    myData.energy_output ./= output_counter
    myData.thrust ./= (output_counter*timeStep)

   # Define a function to write a 1D or 2D CuArray to a file in MATLAB format
function write_cuarray_to_file_matlab(stream, cuarray)
    CUDA.@sync begin
        dims = size(cuarray)
        if length(dims) == 1  # Handling 1D CuArray
            n_elements = dims[1]
            write(stream, "[")  # Start the MATLAB array notation
            cuarray_host = Vector{eltype(cuarray)}(undef, n_elements)
            CUDA.copyto!(cuarray_host, cuarray)
            write(stream, join(cuarray_host, ", "))
            write(stream, "]\n")  # End the MATLAB array notation
        else  # Handling 2D CuArray
            n_rows, n_cols = dims
            write(stream, "[")  # Start the MATLAB matrix notation
            for i in 1:n_rows
                row = cuarray[i, :]  # This creates a view on the GPU
                row_host = Vector{eltype(cuarray)}(undef, n_cols)
                CUDA.copyto!(row_host, row)
                write(stream, join(row_host, ", "))
                if i < n_rows
                    write(stream, "; ")
                end
            end
            write(stream, "]\n")  # End the MATLAB matrix notation
        end
    end
end

    # Define a function to write each 2D CuArray to a file in MATLAB format
    function write_cuarray_to_file_matlab1D(stream, cuarray)
        CUDA.@sync begin

        end
    end

    stream1 = open("$(path)potential.txt", "a")
    stream2 = open("$(path)ne.txt", "a")
    stream3 = open("$(path)ni.txt", "a")
    stream4 = open("$(path)nn.txt", "a")
    stream5 = open("$(path)uix.txt", "a")
    stream6 = open("$(path)uiy.txt","a")
    stream7 = open("$(path)uiz.txt", "a")
    stream8 = open("$(path)uex.txt","a")
    stream9 = open("$(path)uey.txt", "a")
    stream10 = open("$(path)uez.txt", "a")
    stream11 = open("$(path)energy.txt", "a")
    stream12 = open("$(path)thrust.txt", "a")

    # Write the timestamp
    println(stream1, time_stamp)
    println(stream2, time_stamp)
    println(stream3, time_stamp)
    println(stream4, time_stamp)
    println(stream5, time_stamp)
    println(stream6, time_stamp)
    println(stream7, time_stamp)
    println(stream8, time_stamp)
    println(stream9, time_stamp)
    println(stream10, time_stamp)
    println(stream11, time_stamp)
    println(stream12, time_stamp)


    # Write the CuArrays to the corresponding files in MATLAB format
    write_cuarray_to_file_matlab(stream1, myData.phi_output)
    write_cuarray_to_file_matlab(stream2, myData.neArray_output)
    write_cuarray_to_file_matlab(stream3, myData.niArray_output)
    write_cuarray_to_file_matlab(stream4, myData.nnArray_output)
    write_cuarray_to_file_matlab(stream5, myData.uixArray_output)
    write_cuarray_to_file_matlab(stream6, myData.uiyArray_output)
    write_cuarray_to_file_matlab(stream7, myData.uizArray_output)
    write_cuarray_to_file_matlab(stream8, myData.uexArray_output)
    write_cuarray_to_file_matlab(stream9, myData.ueyArray_output)
    write_cuarray_to_file_matlab(stream10, myData.uezArray_output)
    write_cuarray_to_file_matlab(stream11, myData.energy_output)
    write_cuarray_to_file_matlab(stream12, myData.thrust)

    # Close the files
    close(stream1)
    close(stream2)
    close(stream3)
    close(stream4)
    close(stream5)
    close(stream6)
    close(stream7)
    close(stream8)
    close(stream9)
    close(stream10)
    close(stream11)
    close(stream12)
    myData.phi_output .= 0.0
    myData.neArray_output .= 0.0
    myData.niArray_output .= 0.0
    myData.nnArray_output .= 0.0
    myData.uixArray_output .= 0.0
    myData.uiyArray_output .= 0.0
    myData.uizArray_output .= 0.0
    myData.uexArray_output .= 0.0
    myData.ueyArray_output .= 0.0
    myData.uezArray_output .= 0.0
    myData.energy_output .= 0.0
    myData.thrust .= 0.0
end




#function to delete all files under a folder
function delete_all_files(folder_path::String; recursive::Bool=false)
    # Get the list of all files and directories in the folder
    files = readdir(folder_path)

    # Loop through each item in the folder
    for file in files
        full_path = joinpath(folder_path, file)

        if isfile(full_path)
            # If it's a file, remove it
            rm(full_path)
        elseif isdir(full_path) && recursive
            # If it's a directory and recursive is true, remove it and its contents
            rm(full_path, recursive=true)
        end
    end
    println("All files deleted from $folder_path.")
end

function saveData(save_path, time, myParticle, myDomain, myMagneticField)
    timepath = joinpath(@__DIR__, "..", save_path, "time.txt")
    open(timepath, "w") do file
        write(file, string(time))  # Convert the number to a string and write it
    end
    save_object_to_csv(myParticle, save_path)
    save_object_to_csv(myDomain, save_path)
    save_object_to_csv(myMagneticField, save_path)
    println("Saved to ", save_path, ", time = ", time*1e9, " ns")
end