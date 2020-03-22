using DistributionsDiff
using Test
using Distributions
using ChainRulesCore
import ForwardDiff

@testset "gradlogpdf" begin
    @testset "Exponential" begin
        x = 2.0
        θ = 3.0
        E, E_pullback = rrule(Exponential, θ)
        lp, lp_pullback = rrule(logpdf, E, x)

        vm = 1
        (_, res1, res2) = lp_pullback(vm)

        grad = ForwardDiff.gradient(pair -> logpdf(Exponential(pair[2]), pair[1]), [x, θ])
        @test extern(res1) ≈ grad[1]
        @test extern(res2) ≈ grad[2]
    end

    @testset "Normal" begin
        x = 2.0
        μ = 3.0
        σ = 1.0
        N, N_pullback = rrule(Normal, μ, σ)
        lp, lp_pullback = rrule(logpdf, N, x)

        vm = 1
        (_, res1, res2, res3) = lp_pullback(vm)

        grad = ForwardDiff.gradient(pair -> logpdf(Normal(pair[2], pair[3]), pair[1]), [x, μ, σ])
        @test extern(res1) ≈ grad[1]
        @test extern(res2) ≈ grad[2]
        @test extern(res3) ≈ grad[3]
    end

    # @testset "Arcsine" begin
    #     x = 4.0
    #     a = 3.0
    #     b = 5.0
    #     N, N_pullback = rrule(Normal, μ, σ)
    #     lp, lp_pullback = rrule(logpdf, N, x)

    #     vm = 1
    #     (_, res1, res2, res3) = lp_pullback(vm)

    #     grad = ForwardDiff.gradient(pair -> logpdf(Normal(pair[2], pair[3]), pair[1]), [x, μ, σ])
    #     @test extern(res1) ≈ grad[1]
    #     @test extern(res2) ≈ grad[2]
    #     @test extern(res3) ≈ grad[3]
    # end

    @testset "Beta" begin
        x = 0.4
        α = 3.0
        β = 5.0
        B, B_pullback = rrule(Beta, α, β)
        lp, lp_pullback = rrule(logpdf, B, x)

        vm = 1
        (_, res1, res2, res3) = lp_pullback(vm)

        grad = ForwardDiff.gradient(pair -> logpdf(Beta(pair[2], pair[3]), pair[1]), [x, α, β])
        @test extern(res1) ≈ grad[1]
        @test extern(res2) ≈ grad[2]
        @test extern(res3) ≈ grad[3]
    end

end
