include.(["./skewt.jl", "./Lib.jl"])
using Test, PyCall, Statistics
using .Lib

py"""
import numpy as np
from arch.univariate import SkewStudent

skewt = SkewStudent()

def skewt_pdf(μ, σ, ν, λ, x):
  z = (np.asarray(x) - μ) / σ
  ll = skewt.loglikelihood([ν, λ], resids=z, sigma2=1, individual=True)
  return np.exp(ll) / σ

def skewt_cdf(μ, σ, ν, λ, x):
  z = (np.asarray(x) - μ) / σ
  return skewt.cdf(resids=z, parameters=[ν, λ])

def skewt_quantile(μ, σ, ν, λ, p):
  return μ + σ * skewt.ppf(pits=p, parameters=[ν, λ])
"""

@testset "SkewT pdf, cdf, quantile" begin
  νs = [2.1, 3, 5, 10, 30, 100]
  λs = [-0.97, -0.8, -0.5, -0.1, 0.0, 0.1, 0.5, 0.8, 0.97]

  x_test_points() = begin
    xs = [-5, -2, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 2, 5, 10]
    points = [(ν, λ, x) for ν in νs, λ in λs, x in xs]
    getindex.(points, 1), getindex.(points, 2), getindex.(points, 3)
  end

  let (νs, λs, xs) = x_test_points()
    @test pdf.(SkewT.(0.1, 1.2, νs, λs), xs) ≈ py"skewt_pdf".(0.1, 1.2, νs, λs, xs)
    @test cdf.(SkewT.(0.1, 1.2, νs, λs), xs) ≈ py"skewt_cdf".(0.1, 1.2, νs, λs, xs)

    @test skewt_logpdf.(Ref(0.1), Ref(1.2), νs, λs, xs) ≈ log.(py"skewt_pdf".(0.1, 1.2, νs, λs, xs))
  end

  q_test_points() = begin
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    points = [(ν, λ, q) for ν in νs, λ in λs, q in qs]
    getindex.(points, 1), getindex.(points, 2), getindex.(points, 3)
  end

  let (νs, λs, qs) = q_test_points()
    @test quantile.(SkewT.(0.1, 1.2, νs, λs), qs) ≈ py"skewt_quantile".(0.1, 1.2, νs, λs, qs)
  end

  @test skewt_logpdf(0.1, 1.2, 3.0, 0.0, 1.5) ≈ tdist_logpdf_std(0.1, 1.2, 3.0, 1.5)
end;

@testset "SkewT fit_mle" begin
  Random.seed!(0) do
    x = rand(SkewT(1.0, 2.0, 5.0, 0.5), 10_000);
    (; μ, σ, ν, λ) = fit_mle(SkewT, x)
    @test [μ, σ, ν, λ] ≈ [1.0, 2.0, 5.0, 0.5] rtol=0.1
  end
end

@testset "SkewT fit_mle_fixed" begin
  Random.seed!(0) do
    x = rand(SkewT(1.0, 2.0, 5.0, 0.5), 10_000);
    (; μ, σ, ν, λ) = fit_mle(SkewT, x, (ν=5.0,))
    @test [μ, σ, ν, λ] ≈ [1.0, 2.0, 5.0, 0.5] rtol=0.1
  end
end
