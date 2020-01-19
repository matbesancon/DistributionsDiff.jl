using DistributionsDiff
using Test
using Distributions
using ChainRulesCore
import ForwardDiff

@testset "Exponential gradlogpdf" begin
    # f(x, mean, std) = logpdf(Normal(mean, std), x))
    x = 2.0
    θ = 3.0
    E, E_pullback = rrule(Exponential, θ)
    lp, lp_pullback = rrule(logpdf, E, x)

    vm = 1
    (_, res1, res2) = lp_pullback(vm)

    grad = ForwardDiff.gradient(pair -> logpdf(Exponential(pair[2]), pair[1]), [x, θ])
    @test extern(res1) ≈ grad[1]
    @test extern(res2) ≈ grad[2]
    # broken (_, res3) = E_pullback(E, res1)
end