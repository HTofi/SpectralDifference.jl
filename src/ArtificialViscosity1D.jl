module ArtificialViscosity1D

    using LegendrePolynomials
    using Interpolations
    # using ..BasicRoutines

    export modal_sensor, artificial_viscosity, sensor_threshold, global_artificial_viscosity

    """
        modal_sensor(u::Array{Float64,1}, V::Array{Float64,2})

    This function takes a set of nodal values of a variable u and the Vandermonde
    matrix V based on the same nodes as u. It outputs the value of the modal sensor
    relevant to the nodal values of u:

    ``s(u) = \\log\\left[\\frac{(u-\\overline u, u-\\overline u)}{(u, u)}\\right]``
    """
    function modal_sensor(u::Array{Float64,1}, V::Array{Float64,2})
        û = V\u
        return log10(û[end]^2/sum(û.^2))
    end

    """
        artificial_viscosity(s::Float64,ε₀::Float64,s₀::Float64,κ::Float64)

    This function reads in the value of a sensor `s` for an element as well as
    the three parameters `ε₀`, `s₀` and `κ`. It then calculates the value of artificial
    viscosity for the element as

    ``ε = \\begin{cases}0 & \\text{for }\\ s < s₀ - κ \\\\ \\frac{ε₀}{2}\\left[1 + \\sin\\frac{π(s-s₀)}{2κ} \\right] & \\text{for }\\ s₀ - κ ≤ s ≤ s₀ + κ\\\\ ε₀ & \\text{for }\\ s > s₀ + κ\\end{cases}``
    """
    function artificial_viscosity(s::Float64,ε₀::Float64,s₀::Float64,κ::Float64)

        if s < s₀ - κ
            return 0.0
        elseif s < s₀ + κ
            return ε₀/2 * (1 + sin(π*(s-s₀)/(2*κ)))
        end

        return ε₀
    end

    """
        ψₘ(n::Array{Float64,1})

    This function takes in a vector of nodes `n` and calculates the manufactured
    solution used to calculate the threshold parameter `s₀` for the artificial
    viscosity. The function `ψₘ` is defined as

    ``ψₘ(ξᵢ) = \\frac{1}{2}\\left[1 + \\tanh\\left(\\frac{⌊i-1-l/2⌋ + 1/2}{(l-1)/5}\\right) \\right]``

    where ``l`` is the total number of nodes.
    """
    function ψₘ(n::Array{Float64,1})

        l = length(n)
        ψ = zeros(l)
        for i in 1:l
            ψ[i] = 0.5 * (1 + tanh(5*(floor(i-1-l/2) + 0.5)/(l-1)))
        end

        return ψ
    end

    """
        sensor_threshold(n::Array{Float64,1}, V::Array{Float64,2})

    This function takes as input a set of nodes n and their associated Vandermonde
    matrix V. It outputs the value of the sensor threshold s₀ used as an input
    parameter for the function `artificial_viscosity`. The threshold `s₀` is defined
    as

    ``s₀ = sₘ - 3``

    where ``sₘ`` is the sensor value for the manufactured solution ``ψₘ``.
    """
    function sensor_threshold(n::Array{Float64,1}, V::Array{Float64,2})
        return modal_sensor(ψₘ(n), V) - 3
    end

    """
        global_artificial_viscosity(U::Array{T,3}, ε₀::T, s₀::T, κ::T, V::Array{T,2}) where T<:Float64

    This function takes as input the global array of conservative variables `U`, the artificial viscosity parameters `ε₀`, `s₀` and `κ`, the solution points in a standard element sₚ, and the Vandermonde matrix associated to the solution points `V`. It then outputs the global array of artificial viscosities for all solution points in the domain.

    The sensor variable used is density.
    """
    function global_artificial_viscosity(U::Array{T,3}, ε₀::T, s₀::T, κ::T, sₚ::Array{T,1}, V::Array{T,2}) where T<:Float64

        _, n, N = size(U)
        εₑ = zeros(N) # element-wise constant AV
        ε = zeros(n, N) # interpolated AV

        for i in 1:N
            s = modal_sensor(U[1,:,i], V)
            εₑ[i] = artificial_viscosity(s, ε₀, s₀, κ)
        end

        # linear interpolation
        xₚ = sₚ / 2
        e = extrapolate(interpolate(εₑ, BSpline(Linear())), Line())
        for j in 1:N, i in 1:n
            ε[i,j] = e(j + xₚ[i])
        end

        return ε
    end
end
