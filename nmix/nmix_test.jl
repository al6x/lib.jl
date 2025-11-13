include.(["./nmix.jl", "./Lib.jl", "./SkewT.jl"])
using Test, Statistics, Distributions
using .Lib
import QuadGK

@testset "tmix_le_mean_logpdf should match specified E[e^x] and return E[x]" begin
  d = (0.1, 0.015, 3.0)
  for same_loc in (false)
    f_exp(x) = exp(tmix_le_mean_logpdf(d..., x; same_loc)[1]) * exp(x)
    @test QuadGK.quadgk(f_exp, log(1e-4), log(1e+4))[1] ≈ exp(0.1)

    f(x) = exp(tmix_le_mean_logpdf(d..., x; same_loc)[1]) * x
    @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ tmix_le_mean_logpdf(d..., 0.0; same_loc)[2]
  end
end;

@testset "tmix_le_mean_logpdf skewed should match specified E[e^x] and return E[x]" begin
  d = (0.1, 0.015, 3.0, -1.0)
  for same_loc in (false)
    f_exp(x) = exp(tmix_le_mean_logpdf(d..., x; same_loc)[1]) * exp(x)
    @test QuadGK.quadgk(f_exp, log(1e-4), log(1e+4))[1] ≈ exp(0.1)

    f(x) = exp(tmix_le_mean_logpdf(d..., x; same_loc)[1]) * x
    @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ tmix_le_mean_logpdf(d..., 0.0; same_loc)[2]
  end
end;

@testset "tmix_logpdf should match SkewT" begin
  d = (0.0, 1.0, 3.0)
  @assert isapprox(exp(tmix_logpdf(d..., 0.5)), pdf(SkewT(d..., 0.0), 0.5), rtol=0.03)
end;

@testset "nigmix_logpdf should match NIG" begin
  nig_pdf = 0.38633 # py"nig_pdf"(0.0, 1.0, 8.0, 0.0, 0.5)
  @assert isapprox(exp(nigmix_logpdf(0.0, 1.0, 8.0, 0.5)), nig_pdf, rtol=0.03)
end;