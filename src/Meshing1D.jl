module Meshing1D

    using FastGaussQuadrature

    export standard_element, physical_location, regular_mesh

    """
        standard_element(n::Int)

    This function takes in the number of nodes in the standard element as an input
    and returns a tuple of three arrays containing the:

    * solution nodes `n_s`
    * associated weights `w`
    * flux nodes `n_f`
    """
    function standard_element(n::Int64)

        # solution nodes and weights in computational space
        n_s, w = gausslegendre(n)

        # flux nodes
        n_f, = gausslobatto(n+1)

        return n_s, w, n_f
    end

    """
        physical_location(ξ::Float64, xₗ::Float64, h::Float64)

    This function takes in the abscissa of a point in the standard element `ξ`, the position of the left boundary of the element in physical space `xₗ` and the element's width in physical space `h`. It then calculates the location of the point in physical space using the relation

    ``x = xₗ + \\frac{1+ξ}{2}h``
    """
    function physical_location(ξ::Float64, xₗ::Float64, h::Float64)
        return xₗ + h*(1+ξ)/2
    end

    """
        regular_mesh(x₀::Float64, x₁::Float64, N::Int64, n::Int64)

    This function takes the following as input:

    * the location of the left boundary of the computational domain `x₀`
    * the location of the right boundary of the computational domain `x₁`
    * the total number of mesh elements `N`
    * the order of the computation `n`

    It then outputs an `n`×`N` array containing the abscissas in physical space of all solution points in the computational domain.
    """
    function regular_mesh(x₀::Float64, x₁::Float64, N::Int64, n::Int64)

        mesh = zeros(n,N)
        xₗ = range(x₀, x₁, length = N+1)
        n_s, = standard_element(n)
        h = (x₁ - x₀)/N

        for j in 1:N, i in 1:n
            mesh[i,j] = physical_location(n_s[i], xₗ[j], h)
        end

        return mesh
    end

end
