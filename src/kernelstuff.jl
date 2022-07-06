abstract type Kernelfunction end


mutable struct Precompute{T}
    kernelfunction::T
    Kpre::Matrix{Float64}           # Pre-computed matrix 
    K::Matrix{Float64}              # Pre-allocated kernel matrix
    idx::Vector{Vector{CartesianIndex{2}}} # Indices used for applying scaling per band
    L::Int64                        # Number of bands
end


function Precompute(; kf = kernelfunction, x = x, y = y, delays = delays)

    Nx = map(length, x)
    
    Ny = map(length, y)

    B = PseudoBlockArray(0.0*I, Nx, Ny)

    L = length(x) ; @assert(L == length(y))

    for i in 1:L

        for j in 1:L
            
            B[Block(i,j)] = [precompute(kf, x₁-delays[i], x₂-delays[j])  for x₁ in x[i], x₂ in y[j]]
            
        end
        
    end

    idx = calculateBlockIndices(Nx, Ny, L)
    
    return Precompute(kf, Matrix(B), zeros(size(Matrix(B))), idx, L)

end


function calculateBlockIndices(Nx, Ny, L)::Vector{Vector{CartesianIndex{2}}}

    A = [PseudoBlockArray(0.0*I, Nx, Ny) for l in 1:L]

    for l in 1:L

        for i in 1:L

            for j in 1:L
                
                (l == i || l == j) ? A[l][Block(i,j)] .= 1.0 : nothing
                
            end
            
        end

    end

    return [findall(Matrix(Aₗ).>0) for Aₗ in A]

end


function update!(P, α, ρ)
    
    for i in eachindex(P.K)
        @inbounds P.K[i] = postcompute(P, P.Kpre[i], ρ)
    end

    for (i, αᵢ) in enumerate(α)
        @views P.K[P.idx[i]] .*= αᵢ
    end

    P.K
    
end


#---------------------------------------------------


######
# OU #
######

struct OU <: Kernelfunction

end

function OUfunction(xᵢ, xⱼ ; ρ=1.0)

    ℓ = ρ

    r = abs(xᵢ - xⱼ)

    exp(-r/ℓ)

end


function precompute(k::OU, xᵢ, xⱼ)

    -abs(xᵢ - xⱼ)

end


function postcompute(::Precompute{OU}, r, ρ)

    exp(r/ρ)

end


#######
# RBF #
#######

function RBF(xᵢ, xⱼ ; ρ=1.0)

    exp(-((xᵢ-xⱼ)^2)/ρ)

end


function preRBF(xᵢ, xⱼ)

    -(xᵢ-xⱼ)^2

end


function postRBF(r, ρ)

    exp(r/ρ)

end


############
# Matern32 #
############

function Matern32(xᵢ, xⱼ ; ρ=1.0)

    ℓ = ρ

    r = abs(xᵢ - xⱼ)

    (1 + sqrt(3)*r/ℓ) * exp(- sqrt(3)*r/ℓ)

end


function preMatern32(xᵢ, xⱼ)

    sqrt(3) * abs(xᵢ - xⱼ)

end


function postMatern32(val, ρ)

    (1 + val/ρ) * exp(-val/ρ)

end

# function matern52(xᵢ,xⱼ ; ρ=1.0)

#     ℓ = ρ

#     r = abs(xᵢ - xⱼ)

#     (1 + sqrt(5)*r/ρ +( 5*r^2)/(3*ℓ^2)) * exp(- sqrt(5)*r/ℓ)

# end
#---------------------------------------------------
