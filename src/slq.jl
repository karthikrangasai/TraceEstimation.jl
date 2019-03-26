using Random
using LinearAlgebra

"""
Basis Vector = ej
"""
function basisVec(l, i)
    return vcat(zeros(i-1), 1.0, zeros(l-i))
end

"""
Generating a random vector based on Rademacher Distribution
"""
function randomRademacherVector(x::Int64)
    return rand(-1.0:2.0:1.0, x)
end

"""
Lanczos function for eigen vlaues
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

        r = r - alpha .* q
        # beta = norm(r)
        
        Alpha[j] = alpha
        # if beta == 0
        #     break
        # end
    end
    T = SymTridiagonal(Alpha, Beta)
    return T
end

"""
Actual Algorithm Implementataion
"""
function slq(A::AbstractMatrix, f::Function, m::Int64, nv::Int64)
    if !(isposdef(A))
        throw("Matrix passed is not a Positive Semi-Definite Martix")
    else
        trace = 0
        for i = 1:nv
            vl = randomRademacherVector(size(A,2))
            T = lanczos(A, vl, m)

            Y = eigvecs(T)
            theta = eigvals(T)

            for k = 1:size(theta,1)
                trace = trace + ((basisVec(size(Y[:,k],1),1))' * Y[:,k])^2 * f(theta[k])
            end
        end
        return (size(A, 1)/nv) * trace
    end
end