module FluxReconstruction1D

    using BasicRoutines
    using LinearAlgebra
    using InputParameters
    using PositivityPreserving
    using Meshing1D

    export local_flux, intercell_flux, right_intercell_flux, global_flux, flux_divergence

    """
        local_flux(U::Array{Float64, 1})

    This function takes in the vector of conservative variables U = [ρ, ρu, ρE] at a point and calculates the flux vector F = [ρu, ρu²+p, u(ρE+p)] at the same point.
    """
    function local_flux(U::Array{Float64,1})

        # calculate primitive variables
        ρ, u, p = primitive_variables(U)

        # calculate the flux vector
        F = zeros(3)
        F[1] = ρ*u
        F[2] = ρ*u^2 + p
        F[3] = u*(U[3] + p)

        return F
    end

    """
        intercell_flux(UL::Array{Float64,1}, UR::Array{Float64,1}, FL::Array{Float64,1})

    This function takes in the vectors of conservative variables to the left and right of an intercell discontinuity, UL and UR respectively, and the flux vector to the left FL, and computes the intercell flux Fᵢ using the Roe Riemann solver.

    No entropy fix is included yet.

    Reference: Riemann Solvers and Numerical Methods for Fluid Dynamics A Practical Introduction, Third Edition by Eleuterio F. Toro.
    """
    function intercell_flux(UL::Array{Float64,1}, UR::Array{Float64,1}, FL::Array{Float64,1})

        # compute primitive variables
        ρL, uL, pL = primitive_variables(UL)
        ρR, uR, pR = primitive_variables(UR)
        HL = (UL[3] + pL)/ρL
        HR = (UR[3] + pR)/ρR

        # compute the Roe-average variables
        Roe_avg = [√ρL, √ρR]/(√ρL + √ρR)

        ũ = Roe_avg ⋅ [uL, uR]
        H̃ = Roe_avg ⋅ [HL, HR]
        Ṽ² = ũ^2
        ã = √((γ-1)*(H̃ - 0.5*Ṽ²))

        # compute the average eigenvalues λᵢ
        λ₁ = ũ - ã
        λ₂ = ũ
        λ₃ = ũ + ã

        # compute the average right eigenvectors K̃ᵢ
        K̃₁ = [1.0, ũ - ã, H̃ - ũ*ã]
        K̃₂ = [1.0, ũ, 0.5*Ṽ²]
        K̃₃ = [1.0, ũ + ã, H̃ + ũ*ã]

        # compute the coefficients αᵢ of the projection of ΔU on the K̃ᵢ's
        Δu₁, Δu₂, Δu₃ = UR - UL
        α₂ = (γ-1)/ã^2 * (Δu₁*(H̃-ũ^2) - ũ*Δu₂ - Δu₃)
        α₁ = 1/(2*ã) * (Δu₁*(ũ+ã) - Δu₂ - ã*α₂)
        α₃ = Δu₁ - (α₁ + α₂)

        # compute intercell flux
        Λ = (λ₁, λ₂, λ₃)
        Α = (α₁, α₂, α₃)
        K = (K̃₁, K̃₂, K̃₃)

        Fᵢ = FL
        for i in 1:3
            if Λ[i] < 0
                Fᵢ += Α[i]*Λ[i]*K[i]
            end
        end

        return Fᵢ
    end


    """
        global_flux(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, L::Array{T,2}) where T <: Float64

    This function takes as input the global array of conservative variables `U`, the conservative variable vectors at the left and right boundaries `Uₗ` and `Uᵣ` respectively, and the interpolation matrix `L` which maps variables from solution points to flux points. It outputs the global flux array `F` of size `3*(n+1)*N` where `3` correponds to the three conservative variables, `n+1` is the number of flux points per element, and `N` is the number of elements in the mesh.

    This function assumes a regular mesh, since `L` is the same for all elements.
    """
    function global_flux(U::Array{T,3}, Uₗ::Array{T,1}, Uᵣ::Array{T,1}, L::Array{T,2}) where T <: Float64

        _, n, N = size(U)
        _, w, _ = standard_element(n)

        # global flux array
        F = zeros(3, n+1, N)

        # solution vector at flux points
        Uf = zeros(3, n+1, N)

        # calculate flux at each point by interpolation
        for j in 1:N
            Uf[1,:,j] = L*U[1,:,j]
            Uf[2,:,j] = L*U[2,:,j]
            Uf[3,:,j] = L*U[3,:,j]
        end

        preserve_positivity!(U[:,:,:], Uf[:,:,:], w)

        for j in 1:N, i in 1:n+1
            F[:,i,j] = local_flux(Uf[:,i,j])
        end

        # calculate intercell fluxes
        for j in 1:N-1
            F[:,n+1,j] = intercell_flux(Uf[:,n+1,j], Uf[:,1,j+1], F[:,n+1,j])
            F[:,1,j+1] = F[:,n+1,j]
        end

        # calculate flux at boundaries
        F[:,1,1] = right_intercell_flux(Uₗ, Uf[:,1,1], F[:,1,1])
        F[:,n+1,N] = intercell_flux(Uf[:,n+1,N], Uᵣ, F[:,n+1,N])

        return F
    end

    """
        right_intercell_flux(UL::Array{Float64,1}, UR::Array{Float64,1}, FR::Array{Float64,1})

    This function takes in the vectors of conservative variables to the left and right of an intercell discontinuity, UL and UR respectively, and the flux vector to the right FR, and computes the intercell flux Fᵢ using the Roe Riemann solver.

    No entropy fix is included yet.

    Reference: Riemann Solvers and Numerical Methods for Fluid Dynamics A Practical Introduction, Third Edition by Eleuterio F. Toro.
    """
    function right_intercell_flux(UL::Array{Float64,1}, UR::Array{Float64,1}, FR::Array{Float64,1})

        # compute primitive variables
        ρL, uL, pL = primitive_variables(UL)
        ρR, uR, pR = primitive_variables(UR)
        HL = (UL[3] + pL)/ρL
        HR = (UR[3] + pR)/ρR

        # compute the Roe-average variables
        Roe_avg = [√ρL, √ρR]/(√ρL + √ρR)

        ũ = Roe_avg ⋅ [uL, uR]
        H̃ = Roe_avg ⋅ [HL, HR]
        Ṽ² = ũ^2
        ã = √((γ-1)*(H̃ - 0.5*Ṽ²))

        # compute the average eigenvalues λᵢ
        λ₁ = ũ - ã
        λ₂ = ũ
        λ₃ = ũ + ã

        # compute the average right eigenvectors K̃ᵢ
        K̃₁ = [1.0, ũ - ã, H̃ - ũ*ã]
        K̃₂ = [1.0, ũ, 0.5*Ṽ²]
        K̃₃ = [1.0, ũ + ã, H̃ + ũ*ã]

        # compute the coefficients αᵢ of the projection of ΔU on the K̃ᵢ's
        Δu₁, Δu₂, Δu₃ = UR - UL
        α₂ = (γ-1)/ã^2 * (Δu₁*(H̃-ũ^2) - ũ*Δu₂ - Δu₃)
        α₁ = 1/(2*ã) * (Δu₁*(ũ+ã) - Δu₂ - ã*α₂)
        α₃ = Δu₁ - (α₁ + α₂)

        # compute intercell flux
        Λ = (λ₁, λ₂, λ₃)
        Α = (α₁, α₂, α₃)
        K = (K̃₁, K̃₂, K̃₃)

        Fᵢ = FR
        for i in 1:3
            if Λ[i] > 0
                Fᵢ -= Α[i]*Λ[i]*K[i]
            end
        end

        return Fᵢ
    end

    """
        flux_divergence(F::Array{Float64,3}, D::Array{Float64,2})

    This function takes as input the global flux array `F` and the differentiation matrix `D` that maps the flux points on to the solution points. It then calculates the global flux divergence array of size `nₑ*n*N`, where `nₑ` is the number of equations (i.e number of conservative variables), `n` is the number of solution nodes per standard element, and `N` is the number of elements in the mesh.
    """
    function flux_divergence(F::Array{Float64,3}, D::Array{Float64,2})

        nₑ, n, N = size(F) .- (0, 1, 0)
        ∇F = zeros(nₑ, n, N)

        for j in 1:N, i in 1:nₑ
            ∇F[i,:,j] = D*F[i,:,j]
        end

        return ∇F
    end

end
