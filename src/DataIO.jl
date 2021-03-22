module DataIO

    # using ..BasicRoutines
    using JLD

    """
        write_output_file(U::Array{Float64,3}, ε::Array{Float64,1}, mesh::Array{Float64,2}, k::Int64)

    This function takes as input the global array of conservative variables `U`, the global vector of
    artificial viscosities `ε`, the mesh array `mesh`, and the number of the output file `k`. It writes
    all input data into a JLD file.
    """
    function write_output_file(U::Array{Float64,3}, ε::Array{Float64,1}, mesh::Array{Float64,2}, k::Int64)

        nₑ, n, N = size(U)
        W = zeros(nₑ, n, N)

        for j in 1:N, i in 1:n
            W[:,i,j] = primitive_variables(U[:,i,j])
        end

        save("output_"*lpad(k, 5, "0")*".jld", "X", mesh, "PrimitiveVariables", W, "ArtificialViscosity", ε)

    end

end
