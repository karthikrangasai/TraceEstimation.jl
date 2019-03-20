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
    mp = [-1 1];
    v = rand(1,x);

    for i = range(1, length = x)
        v[i] = mp[(v[i] < 0.5) + 1];
    end

    return v
end

"""
Lanczos function for eigen vlaues
"""
function lanczos(A, x, m)
    q = (x/norm(x))'
    Q = hcat(q)
    r = A*q
    alpha = q' * r
    r = r - alpha .* q
    beta = norm(r)

    Alpha = alpha
    Beta = [beta]

    for j = 2:(m+1)
        v = q
        q = r/beta
        Q = hcat(Q,q)
        Alpha = vcat(Alpha, alpha)
        Beta = vcat(Beta, beta)
        r = A*q - beta .* v
        alpha = q' * r
        r = r - alpha .* q
        beta = norm(r)
        if beta == 0
            break
        end
    end
    Alpha = vec(Alpha)
    T = diagm(-1=>Beta, 0=>Alpha, 1=>Beta)
    return T
end

"""
Actual Algorithm Implementataion
"""
function slq(A::AbstractMatrix, f::Function, m::Int64, nv::Int64)
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

    return trace
end
