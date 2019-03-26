using Test
using LinearAlgebra
using TraceEstimation
using SparseArrays


function error(computed::Float64, actual::Float64)
    if (abs(computed - actual)/actual) < 100
        return true
    else
        return false
    end
end

function signCheck(eMax::Float64, eMin::Float64)
    if sign(eMax) == sign(eMin)
        return true
    else
        retuen false
    end
end

eps = rand(1)
neta = rand(1)
f(x) = exp(x)
g(x) = x^2

@testset "Testing Stochastic Lanczos Quadrature" begin
    @testset "Testing for Dense Matrices" begin
        @testset "For Symmetric Matrices of size ($m, $n) with analytic function f(x) = exp(x)" for
            (m, n) in ((50, 50), (100, 100), (500, 500), (1000, 1000), (5000, 5000))
            
            A = rand(m, n)
            testMatrix = 0.5 * (A + A')
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            condNum = eigMax/eigMin
            Mp = f(eigMax)
            mp = f(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)

            m = ceil((sqrt(condNum) * log(K/eps)) / 4)
            nv = ceil((24 * log(2/neta)) / (eps^2))

            @time computed = slq(testMatrix, f, m, nv)
            @time actual = tr(inv(testMatrix))
            @test error(computed, actual)
        end

        @testset "For Symmetric Matrices of size ($m, $n) with analytic function f(x) = x^2" for
            (m, n) in ((50, 50), (100, 100), (500, 500), (1000, 1000), (5000, 5000))
            
            A = rand(m, n)
            testMatrix = 0.5 * (A + A')
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            while signCheck(eigMax, eigMin)
                A = rand(m, n)
                testMatrix = 0.5 * (A + A')
                eigMax = eigmax(testMatrix)
                eigMin = eigmin(testMatrix)
            end
            condNum = eigMax/eigMin
            Mp = g(eigMax)
            mp = g(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)

            m = ceil((sqrt(condNum) * log(K/eps)) / 4)
            nv = ceil((24 * log(2/neta)) / (eps^2))

            @time computed = slq(testMatrix, f, m, nv)
            @time actual = tr(inv(testMatrix))
            @test error(computed, actual)
        end

        @testset "For Matrices using function exp(x) for f(A) of size ($m, $n) with analytic function f(x) = exp(x)" for
            (m, n) in ((50, 50), (100, 100), (500, 500), (1000, 1000), (5000, 5000))
            
            A = rand(m, n)
            for i=1:m
                for j=1:n
                    A[i,j] = exp(A[i,j])
                end
            end
            eigMax = eigmax(A)
            eigMin = eigmin(A)
            condNum = eigMax/eigMin
            Mp = f(eigMax)
            mp = f(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)

            m = ceil((sqrt(condNum) * log(K/eps)) / 4)
            nv = ceil((24 * log(2/neta)) / (eps^2))

            @time computed = slq(A, f, m, nv)
            @time actual = tr(inv(A))
            @test error(computed, actual)
        end

        @testset "For Matrices using function exp(x) for f(A) of size ($m, $n) with analytic function f(x) = x^2" for
            (m, n) in ((50, 50), (100, 100), (500, 500), (1000, 1000), (5000, 5000))
            
            A = rand(m, n)
            for i=1:m
                for j=1:n
                    A[i,j] = exp(A[i,j])
                end
            end
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            while signCheck(eigMax, eigMin)
                A = rand(m, n)
                for i=1:m
                    for j=1:n
                        A[i,j] = exp(A[i,j])
                    end
                end
                eigMax = eigmax(testMatrix)
                eigMin = eigmin(testMatrix)
            end
            condNum = eigMax/eigMin
            Mp = g(eigMax)
            mp = g(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)

            m = ceil((sqrt(condNum) * log(K/eps)) / 4)
            nv = ceil((24 * log(2/neta)) / (eps^2))

            @time computed = slq(testMatrix, f, m, nv)
            @time actual = tr(inv(testMatrix))
            @test error(computed, actual)
        end
    end


    @testset "Testing for Sparse Matrices" begin
        @testset "For Symmetric Matrices of size ($m, $n) with analytic function f(x) = exp(x)" for
            (m, n) in ((50, 50), (100, 100), (500, 500), (1000, 1000), (5000, 5000))
            
            A = sprand(m, n)
            testMatrix = 0.5 * (A + A')
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            condNum = eigMax/eigMin
            Mp = f(eigMax)
            mp = f(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)

            m = ceil((sqrt(condNum) * log(K/eps)) / 4)
            nv = ceil((24 * log(2/neta)) / (eps^2))

            @time computed = slq(testMatrix, f, m, nv)
            @time actual = tr(inv(testMatrix))
            @test error(computed, actual)
        end
    end
end