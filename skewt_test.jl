include("./skewt.jl")
using PyCall, Test

# begin
#   py"""
#   import numpy as np
#   from arch.univariate import SkewStudent

#   skewt = SkewStudent()

#   def skewt_pdf(η, λ, x):
#     ll = skewt.loglikelihood([η, λ], resids=np.asarray(x), sigma2=1, individual=True)
#     return np.exp(ll)

#   def skewt_cdf(η, λ, x):
#     return skewt.cdf(resids=x, parameters=[η, λ])

#   def skewt_quantile(η, λ, p):
#     return skewt.ppf(pits=p, parameters=[η, λ])
#   """
# end

@testset "SkewT pdf, cdf, quantile" begin
  νs = [2.1, 3, 5, 10, 30, 100]
  λs = [-0.97, -0.8, -0.5, -0.1, 0.0, 0.1, 0.5, 0.8, 0.97]

  function x_test_points()
    xs = [-5, -2, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 2, 5, 10]
    points = [(ν, λ, x) for ν in νs, λ in λs, x in xs]
    getindex.(points, 1), getindex.(points, 2), getindex.(points, 3)
  end

  let (νs, λs, xs) = x_test_points()
    @test pdf.(SkewT.(0.0, 1.0, νs, λs), xs) ≈ py"skewt_pdf".(νs, λs, xs)
    @test cdf.(SkewT.(0.0, 1.0, νs, λs), xs) ≈ py"skewt_cdf".(νs, λs, xs)
  end

  function q_test_points()
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    points = [(ν, λ, q) for ν in νs, λ in λs, q in qs]
    getindex.(points, 1), getindex.(points, 2), getindex.(points, 3)
  end

  let (νs, λs, qs) = q_test_points()
    @test quantile.(SkewT.(0.0, 1.0, νs, λs), qs) ≈ py"skewt_quantile".(νs, λs, qs)
  end
end

@testset "SkewT fit_mle" begin
  x = rand(SkewT(1.0, 2.0, 5.0, 0.5), 10_000);
  (; μ, σ, ν, λ) = fit_mle(SkewT, x)
  @test [μ, σ, ν, λ] ≈ [1.0, 2.0, 5.0, 0.5] rtol=0.1
end

@testset "SkewT fit_mle_fixed" begin
  x = rand(SkewT(1.0, 2.0, 5.0, 0.5), 10_000);
  (; μ, σ, ν, λ) = fit_mle(SkewT, x, (ν=5.0,))
  @test [μ, σ, ν, λ] ≈ [1.0, 2.0, 5.0, 0.5] rtol=0.1
end
