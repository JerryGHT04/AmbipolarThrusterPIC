using CSV
using DataFrames
using CUDA


# Function to save the object's fields to CSV
function save_object_to_csv(obj, save_path)
    for field in fieldnames(typeof(obj))
        field_value = getfield(obj, field)

        # Check if the field is a CuArray
        if field_value isa CuArray
            # Convert CuArray to regular Array
            array_value = Array(field_value)

            # Handle both 1D and 2D arrays
            if ndims(array_value) == 1
                # Convert 1D array to a DataFrame with one column
                df = DataFrame(:Column1 => array_value)
            elseif ndims(array_value) == 2
                # Convert 2D array to DataFrame using :auto for column names
                df = DataFrame(array_value, :auto)
            else
                error("Unsupported number of dimensions for field $field")
            end

            # Save the DataFrame as a CSV file
            CSV.write(joinpath(@__DIR__, "..", save_path, string(field) * ".csv"), df)
        end
    end
end

# Function to restore the object's fields from CSV
function restore_object_from_csv(obj, save_path)
    for field in fieldnames(typeof(obj))
        file_path = joinpath(@__DIR__, "..", save_path, string(field) * ".csv")
        
        # Check if the CSV file exists for the given field
        if isfile(file_path)
            # Read the CSV file into a DataFrame
            df = CSV.read(file_path, DataFrame)

            # Convert the DataFrame back to an array
            if ncol(df) == 1
                # If the DataFrame has one column, convert it back to a 1D array
                array_value = convert(Array{Float64}, df[:, 1])  # Extract the first column as an array
            else
                # If the DataFrame has multiple columns, convert it to a 2D array
                array_value = convert(Array{Float64}, Matrix(df))
            end

            # Detect the expected field type (e.g., CuArray{Int32})
            expected_type = fieldtype(typeof(obj), field)
            
            # Convert the array to the expected CuArray type
            cu_array_value = convert(expected_type, CuArray(array_value))

            # Assign the CuArray back to the corresponding field in the object
            setfield!(obj, field, cu_array_value)
        end
    end
end