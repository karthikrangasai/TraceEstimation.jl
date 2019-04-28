using Random
using LinearAlgebra

"""
Generating a random vector based on Rademacher Distribution
"""
function randomRademacherVector(x::Int64, T::Type)
    o = one(T)
    t = 2*o
    return rand(-o:t:o, x)
end

"""
m-step Lanczos function for eigen values
"""
function lanczos(A, x, m)
    q = (x/norm(x))
    r = similar(q)

    mul!(r,A,q)
    alpha = q' * r
    
    r = r - alpha .* q
    
    Alpha = Vector{Float64}(undef, (m+1))
    Beta = Vector{Float64}(undef, m)
    Alpha[1] = alpha

    for j = 2:(m+1)
        Beta[j-1] = norm(r)
        v = q
        q = r/(Beta[j-1])

        mul!(r, A, q)
        r .-= Beta[j-1] .* v
        alpha = q' * r

        r .= r .- alpha .* q
        
        Alpha[j] = alpha
    end
    T = SymTridiagonal(Alpha, Beta)
    return T
end

"""
    slq(A, f, m, nv)
# Arguments
- `A` : Symmetric Positive Definite Matrix
- `f` : Function used to compute f(A)
- `m` : Number of iterations of the Lanczos Algorithm
- `nv` : Number of different starting unit Rademacher Vectors
"""
function slq(A::AbstractMatrix, f::Function, m::Int64, nv::Int64)
    if !(isposdef(A))
        throw("Matrix passed is not a Positive Semi-Definite Martix")
    else
        trace = 0
        for i = 1:nv
            vl = randomRademacherVector(size(A,2), Float64)
            T = lanczos(A, vl, m)

            Y = eigvecs(T)
            theta = eigvals(T)

            # for k = 1:size(theta,1)
            for k = 1:(m+1)
                trace = trace + Y[1,k]^2 * f(theta[k])
            end
        end
        return (size(A, 1)/nv) * trace
    end
end