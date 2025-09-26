import Statistics, SpecialFunctions, StatsFuns, Optim, Distributions, Random
import Distributions: logpdf, pdf, cdf, quantile, fit_mle
import Random: rand
import Statistics: mean, std
import QuadGK, Interpolations

struct SMix3{T <: Real} <: Distributions.ContinuousUnivariateDistribution
  μ::T; λ::T;
  σ1::T, σ2::T, σ3::T;
  w1::T, w2::T;
end

function SMix3(μ::T, λ::T, σ1::T, σ2::T, σ3::T, w::T, w2::T) where {T <: Real}
  σ1 > 0 || throw(DomainError(σ1, "σ1 must be > 0"))
  σ2 > 0 || throw(DomainError(σ2, "σ2 must be > 0"))
  σ3 > 0 || throw(DomainError(σ3, "σ3 must be > 0"))
  w1 > 0 || throw(DomainError(w1, "w1 must be > 0"))
  w2 > 0 || throw(DomainError(w2, "w2 must be > 0"))
  SMix3(μ, λ, σ1, σ2, σ3, w1, w2)
end

logpdf(d::SMix3, x::Real) = begin
  (; μ, λ, σ1, σ2, σ3, w1, w2) = d

end

pdf(d::SMix3, x::Real) = exp(logpdf(d, x))

cdf(d::SMix3, x::Real) = begin
  (; μ, λ, σ1, σ2, σ3, w1, w2) = d

end

quantile(d::SMix3, p::Real) = begin
  (; μ, λ, σ1, σ2, σ3, w1, w2) = d
  0.0 < p < 1 || throw(DomainError(p, "p must be in (0,1)"))
end

rand(rng::Random.AbstractRNG, d::SMix3) = quantile(d, rand(rng))

fit_mle(::Type{SMix3}, x::AbstractVector{<:Real}) = begin
  min_ν = 2.051

  @inline decode(θ) = θ[1], exp(θ[2]), exp(θ[3]) + min_ν, tanh(θ[4])

  nll(θ) = begin
    μ, σ, ν, λ = decode(θ)
    if σ < 1e-12 || ν <= min_ν || abs(λ) >= 1 || ν > 100 return Inf end
    d = SkewT(μ, σ, ν, λ)
    -sum(logpdf.(Ref(d), x))
  end

  # Sometimes it doens't converge, trying various inits
  inits = [
    [mean(x), log(std(x) + 1e-6), log(5-min_ν), 0],
    [median(x), log(median(abs.(x .- median(x))) + 1e-6), log(15-min_ν), 0]
  ]

  for init in inits
    res = Optim.optimize(nll, init, Optim.BFGS(); autodiff = :forward)
    !Optim.converged(res) && continue
    θ = Optim.minimizer(res)
    return SkewT(decode(θ)...)
  end

  error("Can't estimate SkewT")
end

mean_exp(d::SkewT; l::Real, h::Real) = begin
  f(x) = pdf(d, x) * exp(x)
  QuadGK.quadgk(f, l, h)[1]
end