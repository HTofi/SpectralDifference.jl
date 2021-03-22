module InputParameters

    export γ, nₛ, N, M, ρ₀, u₀, p₀, xₛ, ε₀, s₀, κ, x₀, x₁, C, Δt, α, set_ε₀, set_s₀, set_Δt

    # specific heat ratio
    const γ = 1.4

    # simulation order
    nₛ = 5

    # number of elements in mesh
    const N = 100

    # shockwave parameters
    const M = 1.1
    const ρ₀ = 1.0
    const u₀ = 1.0
    const p₀ = 1.0
    const xₛ = 0.5

    # artificial viscosity parameters
    ε₀ = 0.011588
    const κ = 1.0
    const C = 1.0
    s₀ = 1.0

    # domain boundaries
    const x₀ = 0.0
    const x₁ = 1.0

    # time step
    Δt = 0.1
    const α = 1.0

    """
        set_ε₀(λₘ::Float64, h::Float64, n::Int64, C::Float64)

    This function takes as input the spectral radius `λₘ`, the cell width `h`, the simulation order `n`,
    and the viscosity coefficient `C`. It then calculates and sets the coefficient `ε₀`.
    """
    function set_ε₀(λₘ::Float64, h::Float64, n::Int64, C::Float64)
        global ε₀ = C*λₘ*h/(n-1)
    end

    """
        set_s₀(s)

    This function takes as input the value of the sensor of the sensor threshold `s`and assigns it to the input parameter `s₀`.
    """
    function set_s₀(s::Float64)
        global s₀ = s
    end

    """
        set_Δt(xₘ::Float64, α::Float64, λₘ::Float64)

    This function calculates the time step given by the CFL condition. The function takes as input:

    * the minimum distance form the solution point to the boundary `xₘ`
    * the coefficient `α` that sets the value of the time step to the estimated limit of stability
    * the spectral radius of the flux Jacobian `λ = u + a` where `u` is the flow velocity and `a` is the speed of sound.

    It then calculates and sets the value of the time step `Δt`.
    """
    function set_Δt(xₘ::Float64, α::Float64, λₘ::Float64)
        global Δt = α*xₘ/λₘ
    end

end
