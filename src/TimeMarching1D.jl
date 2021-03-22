module TimeMarching1D

    using FluxReconstruction1D
    using ArtificialViscosity1D
    using InputParameters

    export 𝓛, RK45_SSP

    """
        𝓛(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, L::Array{T,2}, D₁::Array{T,2}, D₂::Array{T,2}, sₚ::Array{T,1}, V::Array{T,2}) where T <: Float64

    This function takes as input the following:

    * The global array of conservative variables `U`
    * The left and right boundary vectors `Uₗ` and `Uᵣ`
    * The Lagrange interpolation matrix from solution to flux points `L`
    * The differentiation matrix from solution to solution points `D₁`
    * The differentiation matrix from flux to solution points `D₂`
    * The vector of solution points in a standard element `sₚ`
    * The Vandermonde matrix associated to solution points `V`

    It calculates the global array `Lₕ` used for time-stepping the solution in the Runge Kutta method:

    ``\\frac{duₕ}{dt} = Lₕ(uₕ,t)``

    as well as the global artificial viscosity `ε`.
    """
    function 𝓛(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, L::Array{T,2}, D₁::Array{T,2}, D₂::Array{T,2}, sₚ::Array{T,1}, V::Array{T,2}) where T <: Float64

        _, n, N = size(U)
        F = global_flux(U, Uₗ, Uᵣ, L)
        Lₕ = -flux_divergence(F, D₂)

        ε = global_artificial_viscosity(U, ε₀, s₀, κ, sₚ, V)
        # ε[:,:] .= 0.0

        for j in 1:N, i in 1:3
            Lₕ[i,:,j] .+= D₁*(ε[:,j] .* (D₁*U[i,:,j]))
        end

        return Lₕ, ε
    end

    """
        RK45_SSP(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, L::Array{T,2}, D₁::Array{T,2}, D₂::Array{T,2}, V::Array{T,2}, Δt::T) where T <: Float64

    This function takes as input the following:

    * The global array of conservative variables `U`
    * The left and right boundary vectors `Uₗ` and `Uᵣ`
    * The Lagrange interpolation matrix from solution to flux points `L`
    * The differentiation matrix from solution to solution points `D₁`
    * The differentiation matrix from flux to solution points `D₂`
    * The vector of solution points in a standard element `sₚ`
    * The Vandermonde matrix associated to solution points `V`
    * The time step `Δt`

    It then calculates the global vector of conservative variables `U` at the next time step using
    the fourth-order, five-stage, strong stability preserving Runge-Kutta method (RK45-SSP), and it
    returns a tuple containing `U` and the global artificial viscosity `ε`.
    """
    function RK45_SSP(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, L::Array{T,2}, D₁::Array{T,2}, D₂::Array{T,2}, sₚ::Array{T,1}, V::Array{T,2}, Δt::T) where T <: Float64

        v₁ = U + 0.39175222700392*Δt*𝓛(U,Uₗ,Uᵣ,L,D₁,D₂,sₚ,V)[1]
        v₂ = 0.44437049406734*U + 0.55562950593266*v₁ + 0.36841059262959*Δt*𝓛(v₁,Uₗ,Uᵣ,L,D₁,D₂,sₚ,V)[1]
        v₃ = 0.62010185138540*U + 0.37989814861460*v₂ + 0.25189177424738*Δt*𝓛(v₂,Uₗ,Uᵣ,L,D₁,D₂,sₚ,V)[1]

        Lₕ₃ = 𝓛(v₃,Uₗ,Uᵣ,L,D₁,D₂,sₚ,V)[1]

        v₄ = 0.17807995410773*U + 0.82192004589227*v₃ + 0.54497475021237*Δt*Lₕ₃

        𝓛₄ = 𝓛(v₄,Uₗ,Uᵣ,L,D₁,D₂,sₚ,V)
        ε = 𝓛₄[2]

        v₅ = 0.00683325884039*U + 0.51723167208978*v₂ + 0.12759831133288*v₃ + 0.34833675773694*v₄ + 0.08460416338212*Δt*Lₕ₃ + 0.22600748319395*Δt*𝓛₄[1]

        return v₅, ε
    end
end
