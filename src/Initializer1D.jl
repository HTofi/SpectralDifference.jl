module Initializer1D

    using InputParameters
    using BasicRoutines

    export steady_shock

    """
        steady_shock(M::T, ρ₀::T, u₀::T, p₀::T, xₛ::T, mesh::Array{T,2}) where T <: Float64

    This function initializes the flow field for the simulation of a stationary shock wave. The input arguments are the following:

    * the shock Mach number `M`
    * the upstream flow density, velocity and pressure `ρ₀`, `u₀` and `p₀` respectively
    * the shock position in the comuptational domain `xₛ`
    * the `mesh` array containing the positions of all solution points

    The function outputs a tuple `(U, Uₗ, Uᵣ)`. `U` is the solution array of size `3*n*N`, where `n*N` are the solution points in the mesh, and the `3` correponds to the three conservative variables at each point `[ρ, ρu, ρE]`. `Uₗ` and `Uᵣ` are respectively the left and right boundary values for the solution.

    The initial state corresponds to a heaviside function with a jump in the flow variables at position `xₛ` given by the Rankine-Hugoniot relationships.

    """
    function steady_shock(M::T, ρ₀::T, u₀::T, p₀::T, xₛ::T, mesh::Array{T,2}) where T <: Float64

        n, N = size(mesh)
        U = zeros(3, n, N)

        ρ₁ = (γ+1)*M^2 / (2 + (γ-1)*M^2) * ρ₀
        u₁ = ρ₁/ρ₀ * u₀
        p₁ = (1 + 2*γ/(γ+1) * (M^2 - 1)) * p₀

        U₀ = conservative_variables([ρ₀, u₀, p₀])
        U₁ = conservative_variables([ρ₁, u₁, p₁])

        for j in 1:N, i in 1:n
            if mesh[i, j] < xₛ
                U[:, i, j] = U₀
            else
                U[:, i, j] = U₁
            end
        end

        Uₗ = U₀
        Uᵣ = U₁

        return U, Uₗ, Uᵣ
    end

end
