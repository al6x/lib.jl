include.(["./skewnormal.jl", "./Lib.jl"])
using Test, Distributions
import QuadGK
using .Lib

@testset "skewnormal_mean_exp" begin
  αs = [-2.0, 0.0, 2.0]

  skewnormal_mean_exp_numerical(ξ, ω, α) = begin
    f(x) = pdf(SkewNormal(ξ, ω, α), x) * exp(x)
    QuadGK.quadgk(f, -10, 10)[1]
  end

  for α in αs
    ξ, ω = 0.5, 1.2
    @test skewnormal_mean_exp(ξ, ω, α) ≈ skewnormal_mean_exp_numerical(ξ, ω, α) atol=1e-6
  end
end;