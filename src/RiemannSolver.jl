module RiemannSolver

    using LinearAlgebra

    export intercell_flux, primitive_variables

    # specific heat ratio
    const γ = 1.4

    """
        intercell_flux(UL::Array{Float64,1}, UR::Array{Float64,1}, FL::Array{Float64,1})

    This function takes in the vectors of conservative variables to the left and right
    of an intercell discontinuity, UL and UR respectively, and the flux vector to the
    left FL, and computes the intercell flux Fᵢ using the Roe Riemann solver.

    No entropy fix is included yet.

    Reference: Riemann Solvers and Numerical Methods for Fluid Dynamics A Practical
    Introduction, Third Edition by Eleuterio F. Toro.
    """
    function intercell_flux(UL::Array{Float64,1}, UR::Array{Float64,1}, FL::Array{Float64,1})

        # compute primitive variables
        ρL, uL, vL, wL, pL = primitive_variables(UL)
        ρR, uR, vR, wR, pR = primitive_variables(UR)
        HL = (UL[5] + pL)/ρL
        HR = (UR[5] + pR)/ρR

        # compute the Roe-average variables
        Roe_avg = [√ρL, √ρR]/(√ρL + √ρR)

        ũ = Roe_avg ⋅ [uL, uR]
        ṽ = Roe_avg ⋅ [vL, vR]
        w̃ = Roe_avg ⋅ [wL, wR]
        H̃ = Roe_avg ⋅ [HL, HR]

        Ṽ² = ũ^2 + ṽ^2 + w̃^2
        ã = √((γ-1)*(H̃ - 0.5*Ṽ²))

        # compute the average eigenvalues λᵢ
        λ₁ = ũ - ã
        λ₂ = λ₃ = λ₄ = ũ
        λ₅ = ũ + ã

        # compute the average right eigenvectors K̃ᵢ
        K̃₁ = [1.0, ũ - ã, ṽ, w̃, H̃ - ũ*ã]
        K̃₂ = [1.0, ũ, ṽ, w̃, 0.5*Ṽ²]
        K̃₃ = [0.0, 0.0, 1.0, 0.0, ṽ]
        K̃₄ = [0.0, 0.0, 0.0, 1.0, w̃]
        K̃₅ = [1.0, ũ + ã, ṽ, w̃, H̃ + ũ*ã]

        # compute the coefficients αᵢ of the projection of ΔU on the K̃ᵢ's
        Δu₁, Δu₂, Δu₃, Δu₄, Δu₅ = UR - UL
        α₃ = Δu₃ - ṽ*Δu₁
        α₄ = Δu₄ - w̃*Δu₁
        Δū₅ = Δu₅ - α₃*ṽ - α₄*w̃
        α₂ = (γ-1)/ã^2 * (Δu₁*(H̃-ũ^2) - ũ*Δu₂ - Δū₅)
        α₁ = 1/(2*ã) * (Δu₁*(ũ+ã) - Δu₂ - ã*α₂)
        α₅ = Δu₁ - (α₁ + α₂)

        # compute intercell flux
        Λ = (λ₁, λ₂, λ₃, λ₄, λ₅)
        Α = (α₁, α₂, α₃, α₄, α₅)
        K = (K̃₁, K̃₂, K̃₃, K̃₄, K̃₅)

        Fᵢ = FL
        for i in 1:5
            if Λ[i] < 0
                Fᵢ += Α[i]*Λ[i]*K[i]
            end
        end

        return Fᵢ
    end

    """
        primitive_variables(U::Array{Float64,1})

    This function takes in the vector of conservative variables U = [ρ, ρu, ρv, ρw, ρE]
    and calculates the corresponding vector of primitive variables W = [ρ, u, v, w, p].

    The calculation of p makes use of the ideal gas law.
    """
    function primitive_variables(U::Array{Float64,1})

        W::Array{Float64,1} = zeros(5)

        W[1] = U[1]
        W[2] = U[2]/U[1]
        W[3] = U[3]/U[1]
        W[4] = U[4]/U[1]
        W[5] = (γ-1) * (U[5] - 0.5*U[1]*(W[2]^2 + W[3]^2 + W[4]^2))

        return W
    end

end
