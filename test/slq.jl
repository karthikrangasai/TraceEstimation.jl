using Test
using Random
using LinearAlgebra
using TraceEstimation
import TraceEstimation
using SparseArrays
using Arpack

Random.seed!(31415)

epsilon = 0.05
neta = 0.05

f(x) = 1/x
g(x) = x^2
h(x) = exp(x/10)

function error(computed::Float64, actual::Float64, limit::Float64)
    if abs(computed - actual) < limit
        return true
    else
        return false
    end
end

@testset "Testing Stochastic Lanczos Quadrature" begin
    @testset "Testing for Dense Matrices" begin
        @testset "For Symmetric Matrices of size ($n, $n) with analytic function f(x) = 1/x" for
            n in (500, 1000)
            
            A = rand(n, n)
            testMatrix = A + A' + n*I
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            condNum = eigMax/eigMin
            Mp = f(eigMax)
            mp = f(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)
            C = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(2 * sqrt(condNum))
            rho = (sqrt(condNum) + 1)/(sqrt(condNum) - 1)

            m = Int64(ceil((sqrt(condNum) * log(K/epsilon)) / 4))
            nv = Int64(ceil((24 * log(2/neta)) / (epsilon^2)))
            # nv = 2*m - 1
            # m = 50
            # nv = 100
            
            @time computed = TraceEstimation.slq(testMatrix, f, m, nv)
            @time actual = tr(inv(testMatrix))
            print("Trace computed using the algorithm :  ")
            print(computed)
            print("\n")
            print("Trace computed using tr(inv(A)) :  ")
            print(actual)
            print("\n")
            @test isapprox(computed, actual, rtol=1)
        end

        @testset "For Symmetric Matrices of size ($n, $n) with analytic function f(x) = x^2" for
            n in (500, 1000)
                
            A = rand(n, n)
            testMatrix = A + A' + n*I
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            condNum = eigMax/eigMin
            Mp = f(eigMax)
            mp = f(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)
            C = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(2 * sqrt(condNum))
            rho = (sqrt(condNum) + 1)/(sqrt(condNum) - 1)

            m = Int64(ceil((sqrt(condNum) * log(K/epsilon)) / 4))
            nv = Int64(ceil((24 * log(2/neta)) / (epsilon^2)))
            # nv = 2*m - 1
            # m = 50
            # nv = 100
            
            @time computed = TraceEstimation.slq(testMatrix, g, m, nv)
            @time actual = tr(g(testMatrix))

            print("Trace computed using the algorithm :  ")
            print(computed)
            print("\n")
            print("Trace computed using tr(f(A)) :  ")            
            print(actual)
            print("\n")
            @test isapprox(computed, actual, rtol=1)
        end

        @testset "For Symmetric Matrices of size ($n, $n) with analytic function f(x) = exp(x/10)" for
            n in (500, 1000)
                
            A = rand(n, n)
            testMatrix = A + A' + n*I
            eigMax = eigmax(testMatrix)
            eigMin = eigmin(testMatrix)
            condNum = eigMax/eigMin
            Mp = f(eigMax)
            mp = f(eigMin)
            K = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(sqrt(condNum) * mp)
            C = ((eigMax - eigMin) * ((sqrt(condNum) - 1)^2) * Mp)/(2 * sqrt(condNum))
            rho = (sqrt(condNum) + 1)/(sqrt(condNum) - 1)

            m = Int64(ceil((sqrt(condNum) * log(K/epsilon)) / 4))
            nv = Int64(ceil((24 * log(2/neta)) / (epsilon^2)))
            # nv = 2*m - 1
            # m = 50
            # nv = 100
            
            @time computed = TraceEstimation.slq(testMatrix, h, m, nv)
            @time actual = tr(h(testMatrix))

            print("Trace computed using the algorithm :  ")
            print(computed)
            print("\n")
            print("Trace computed using tr(f(A)) :  ")
            print(actual)
            print("\n")
            @test isapprox(computed, actual, rtol=1)
        end
    end
end