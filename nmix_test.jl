include.(["./nmix.jl", "./Lib.jl", "./SkewT.jl"])
using Test, Statistics, Distributions
using .Lib
import QuadGK

@testset "tmix_le_mean_logpdf should match specified E[e^x]" begin
  f(x) = exp(tmix_le_mean_logpdf(0.1, 0.015, 3.0, x)[1]) * exp(x)
  @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ exp(0.1)
end;

@testset "tmix_le_mean_logpdf skewed should match specified E[e^x]" begin
  f(x) = exp(tmix_le_mean_logpdf(0.1, 0.015, 3.0, -1.0, x)[1]) * exp(x)
  @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ exp(0.1)
end;

# @testset "tmix_logpdf should match SkewT" begin
  d = SkewT(0.0, 1.0, 3.0, 0.0);
  exp(logpdf(d, 0.5))
  exp(tmix_logpdf(0.0, 1.0, 5.0, 0.5))
  # @test QuadGK.quadgk(f, log(1e-4), log(1e+4))[1] ≈ exp(0.1)
# end;

d = SkewT(0.0, 1.0, 3.0, 0.0);
xs = collect(range(-5.0, 5.0; length=1000));

ν = 5.0
plot(xs, pdf.(Ref(SkewT(0.0, 1.0, ν, 0.0)), xs) ./ exp.(tmix_logpdf.(0.0, 1.0, ν, xs)), label="SkewT", )

plot(xs, pdf.(Ref(d), xs), label="SkewT", )
plot!(xs, exp.(tmix_logpdf.(0.0, 1.0, 3.0, xs)), label="TMix", linestyle=:dash)
