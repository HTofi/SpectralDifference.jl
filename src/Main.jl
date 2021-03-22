using InputParameters
using Meshing1D
using Initializer1D
using ArtificialViscosity1D
using TimeMarching1D
using BasicRoutines
using FluxReconstruction1D

using Plots



function main()

    # define useful matrices
    sₚ, _, fₚ = standard_element(nₛ)
    L = interpolation_matrix(sₚ, fₚ)
    D₂ = differentiation_matrix(fₚ, sₚ)
    D₁ = D₂*L
    V = vandermonde_matrix(sₚ)

    # create the mesh
    mesh = regular_mesh(x₀, x₁, N, nₛ)

    # set the initial state
    U, Uₗ, Uᵣ = steady_shock(M, ρ₀, u₀, p₀, xₛ, mesh)

    # set the value of artificial viscosity coefficients
    ρ₁, u₁, p₁ = primitive_variables(Uᵣ)
    λₘ = u₁ + √(γ*p₁/ρ₁)
    h = (x₁ - x₀)/N
    set_ε₀(λₘ, h, nₛ, C)
    set_s₀(sensor_threshold(sₚ, V))
    ε = zeros(nₛ, N)

    # set the value of the time step
    xₘ = (sₚ[1] + 1)/(2*N)
    set_Δt(xₘ, α, λₘ)

    # time marching
    for i in 1:10000
        U, ε = RK45_SSP(U, Uₗ, Uᵣ, L, D₁, D₂, sₚ, V, Δt)
        println("iteration ", i)

        if i%100 == 0
            display(plot(reshape(mesh,(100nₛ,1)), reshape(U[1, :, :],(100nₛ,1)), legend=false, ylims=(0.98,1.2)))
        end
    end

    return mesh, U, ε
end

mesh, U, ε = main()
println("SIMULATION COMPLETED!")
# plot(reshape(mesh,(100nₛ,1)), reshape(U[1, :, :],(100nₛ,1)), legend=false)
