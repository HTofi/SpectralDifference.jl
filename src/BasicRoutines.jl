module BasicRoutines

    using InputParameters

    using LinearAlgebra
    using LegendrePolynomials

    export vandermonde_matrix, interpolation_matrix, differentiation_matrix, primitive_variables, conservative_variables

    """
        vandermonde_matrix(n::Array{Float64,1})

    This function takes as input a set of nodes n. It then calculates the Vandermonde
    matrix V relevant to the given nodes and the normalised Legendre polynomials.

    The Vandermonde matrix can be used to calculate the expansion coefficients of
    a function from its nodal values at the points of n:

    ``\\mathcal V \\hat{\\mathbf u} = \\mathbf u``

    where u is the vector of nodal values and û is the vector of modal coefficients.
    """
    function vandermonde_matrix(n::Array{Float64,1})

        l = length(n)
        V = zeros(l,l)

        for j in 1:l, i in 1:l
            V[i,j] = √((2*(j-1)+1)/2) * Pl(n[i], j-1)
        end

        return V
    end

    """
        interpolation_matrix(n₁::Array{Float64,1}, n₂::Array{Float64,1})

    This function takes as input two sets of points (or nodes) in an element n₁ and n₂.
    It then calculates the interpolation matrix L, which allows to calculate the Lagrange
    interpolation of a function f defined on n₁ at the points of n₂:

        fᵢₙₜ.(n₂) = L*f.(n₁)

    """
    function interpolation_matrix(n₁::Array{Float64,1}, n₂::Array{Float64,1})

        l = length(n₁)
        m = length(n₂)
        L = ones(m,l)

        for j in 1:l, i in 1:m, k in 1:l
            if (k ≠ j)
                L[i,j] *= (n₂[i] - n₁[k])/(n₁[j] - n₁[k])
            end
        end

        return L
    end

    """
        differentiation_matrix(n₁::Array{Float64,1}, n₂::Array{Flaot64,1})

    This function takes as input two sets of points (or nodes) `n₁` and `n₂`. It then
    calculates the differentiation matrix `D`, which allows to calculate the derivative
    of the Lagrange interpolation of a function `f` defined on `n₁` at the points of `n₂`:

        dfᵢₙₜ/dξ.(n₂) = D*f.(n₁)

    """
    function differentiation_matrix(n₁::Array{Float64,1}, n₂::Array{Float64,1})

        l = length(n₁)
        m = length(n₂)
        D = zeros(m,l)

        for j in 1:l, i in 1:m

            for r in 1:l
                if (r ≠ j)
                    D[i,j] += 1/(n₂[i] - n₁[r])
                end
            end

            for k in 1:l
                if (k ≠ j)
                    D[i,j] *= (n₂[i] - n₁[k])/(n₁[j] - n₁[k])
                end
            end
        end

        return D
    end

    """
        primitive_variables(U::Array{Float64,1})

    This function takes in the vector of conservative variables U = [ρ, ρu, ρE]
    and calculates the corresponding vector of primitive variables W = [ρ, u, p].

    The calculation of p makes use of the ideal gas law.
    """
    function primitive_variables(U::Array{Float64,1})

        W::Array{Float64,1} = zeros(3)

        W[1] = U[1]
        W[2] = U[2]/U[1]
        W[3] = (γ-1) * (U[3] - 0.5*U[1]*W[2]^2)

        return W
    end

    """
        conservative_variables(W::Array{Float64,1})

    This function takes in the vector of primitive variables W = [ρ, u, p]
    and calculates the corresponding vector of conservative variables U = [ρ, ρu, ρE].

    The calculation of ρE makes use of the ideal gas law.
    """
    function conservative_variables(W::Array{Float64,1})

        U::Array{Float64,1} = zeros(3)

        U[1] = W[1]
        U[2] = W[1]*W[2]
        U[3] = W[3]/(γ-1) + 0.5*W[1]*W[2]^2

        return U
    end

end
