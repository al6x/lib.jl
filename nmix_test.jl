include.(["./nmix.jl", "./Lib.jl"])
using Test, Statistics
using .Lib
import QuadGK

@testset "nmix_le_mean_logpdf should match specified E[e^x]" begin
  f(x) = exp(nmix_le_mean_logpdf(0.1, 0.015, 3.0, 3.0, x)[1]) * exp(x)
  @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ exp(0.1)
end;

@testset "nmix_le_mean_logpdf skewed should match specified E[e^x]" begin
  f(x) = exp(nmix_le_mean_logpdf(0.1, 0.015, 3.0, 3.0, -1.0, x)[1]) * exp(x)
  @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ exp(0.1)
end;