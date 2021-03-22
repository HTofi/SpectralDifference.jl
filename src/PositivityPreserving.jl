module PositivityPreserving

    using LinearAlgebra
    using Roots
    using BasicRoutines

    export preserve_positivity!, zhang_shu!

    """
        zhang_shu!(U::Array{Float64,2}, Uf::Array{Float64,2}, w::Array{Float64,1})

    Thos function takes in the element-wise arrays of solution vectors at solution and flux points, `U` and `Uf` respectively, and the vector of quadrature weights `w`. It then applies the Zhang-Shu positivity preserving procedure, and changes `U` and `Uf` to ensure that density and pressure remain positive.

    Reference: Lodato, G. (2019). Characteristic modal shock detection for discontinuous finite element methods. Computers & Fluids, 179, 309-333.
    """
    function zhang_shu!(U::Array{Float64,2}, Uf::Array{Float64,2}, w::Array{Float64,1})

        ϵ = 10^-13

        nₛ = length(w)

        # minimum density
        ρₘ = min(minimum(U[1,:]), minimum(Uf[1,:]))

        # average density
        ρₐᵥ= U[1,:] ⋅ w / 2

        θ₁ = min(1.0, (ρₐᵥ - ϵ)/(ρₐᵥ - ρₘ))

        # calculate intermediate solution vector Û
        ρ = θ₁*(U[1,:] .- ρₐᵥ) .+ ρₐᵥ
        for i in 1:nₛ
            U[:,i] *= ρ[i] ./ U[1,i]
        end

        ρ = θ₁*(Uf[1,:] .- ρₐᵥ) .+ ρₐᵥ
        for i in 1:nₛ+1
            Uf[:,i] *= ρ[i] ./ Uf[1,i]
        end

        # pressure
        p = mapslices(primitive_variables, U, dims=(1))[3,:]

        Uₐᵥ = reshape(mapslices(x -> x⋅w/2, U, dims=(2)), 3)

        # tϵ for solution points
        tϵ = zeros(nₛ)
        tϵ[p .> ϵ] .= 1
        for i in 1:nₛ
            if tϵ[i] == 0
                tϵ[i] = fzero(x -> primitive_variables((1-x)*Uₐᵥ + x*U[:,i])[3] - ϵ, 0.5)
            end
        end

        # tϵ for flux points
        pf = mapslices(primitive_variables, Uf, dims=(1))[3,:]
        tϵf = zeros(nₛ+1)
        tϵf[pf .> ϵ] .= 1
        for i in 1:nₛ+1
            if tϵf[i] == 0
                tϵf[i] = fzero(x -> primitive_variables((1-x)*Uₐᵥ + x*Uf[:,i])[3] - ϵ, 0.5)
            end
        end

        # calculate corrected solution Ũ
        θ₂ = min(minimum(tϵ), minimum(tϵf))

        for i in 1:nₛ
            U[:,i] .*= θ₂*(U[:,i] .- Uₐᵥ) .+ Uₐᵥ
        end

        for i in 1:nₛ+1
            Uf[:,i] .*= θ₂*(Uf[:,i] .- Uₐᵥ) .+ Uₐᵥ
        end
    end

    """
        preserve_positivity!(U::Array{Float64,3}, Uf::Array{Float64,3}, w::Array{Float64,1})

    This function takes in the global arrays of conservative variables at solution and flux points, `U` and `Uf` resepectively, and the vector of quadrature weights `w`. It then applies the Zhang-Shu positivity preserving procedure to each mesh element by calling the function `zhang_shu!`.
    """
    function preserve_positivity!(U::Array{Float64,3}, Uf::Array{Float64,3}, w::Array{Float64,1})
        _, _, N = size(U)
        for i in 1:N
            zhang_shu!(U[:,:,i], Uf[:,:,i], w)
        end
    end

end
