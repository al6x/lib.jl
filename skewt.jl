import Statistics, SpecialFunctions, StatsFuns, Optim, Distributions, Random
import Distributions: logpdf, pdf, cdf, quantile, fit_mle
import Random: rand
import Statistics: mean, std
import QuadGK, Interpolations

struct SkewT{T<:Real} <: Distributions.ContinuousUnivariateDistribution
  μ::T; σ::T; ν::T; λ::T
  a::T; b::T; c::T
end


function SkewT(μ::T, σ::T, ν::T, λ::T) where {T<:Real}
  σ > 0       || throw(DomainError(σ, "σ must be > 0"))
  ν > 2.05    || throw(DomainError(ν, "ν must be > 2.05"))
  abs(λ) < 1  || throw(DomainError(λ, "|λ| must be < 1"))

  c = SpecialFunctions.loggamma((ν + 1)/2) - SpecialFunctions.loggamma(ν/2) - 0.5*log(pi*(ν - 2))
  a = 4*λ*exp(c)*(ν - 2)/(ν - 1)
  b = sqrt(1 + 3*λ^2 - a^2)

  SkewT(μ, σ, ν, λ, a, b, c)
end

@inline logpdf(d::SkewT, x::Real) = begin
  (; μ, σ, λ, ν, a, b, c) = d
  z = (x - μ) / σ
  s = sign(z + a/b)
  llf = ((b*z + a)/(1 + s*λ))^2
  zlogpdf = log(b) + c - ((ν + 1)/2) * log(1 + llf/(ν - 2))
  zlogpdf - log(σ)
end

@inline skewt_logpdf(μ::Real, σ::Real, ν::Real, λ::Real, x::Real) = begin
  σ > 0       || throw(DomainError(σ, "σ must be > 0"))
  ν > 2.05    || throw(DomainError(ν, "ν must be > 2.05"))
  abs(λ) < 1  || throw(DomainError(λ, "|λ| must be < 1"))

  c = SpecialFunctions.loggamma((ν + 1)/2) - SpecialFunctions.loggamma(ν/2) - 0.5*log(pi*(ν - 2))
  a = 4*λ*exp(c)*(ν - 2)/(ν - 1)
  b = sqrt(1 + 3*λ^2 - a^2)

  z = (x - μ) / σ
  s = sign(z + a/b)
  llf = ((b*z + a)/(1 + s*λ))^2
  zlogpdf = log(b) + c - ((ν + 1)/2) * log(1 + llf/(ν - 2))
  zlogpdf - log(σ)
end

pdf(d::SkewT, x::Real) = exp(logpdf(d, x))

cdf(d::SkewT, x::Real) = begin
  (; μ, σ, ν, λ, a, b) = d
  z = (x - μ) / σ

  var   = ν/(ν - 2)
  scale = sqrt(var)
  d     = (b*z + a) * scale
  y1    = d/(1 - λ)
  y2    = d/(1 + λ)

  z < -a/b ?
    (1 - λ) * StatsFuns.tdistcdf(ν, y1) :
    (1 - λ)/2 + (1 + λ) * (StatsFuns.tdistcdf(ν, y2) - 0.5)
end

quantile(d::SkewT, p::Real) = begin
  (; μ, σ, ν, λ, a, b) = d
  0.0 < p < 1 || throw(DomainError(p, "p must be in (0,1)"))

  thresh = (1 - λ)/2
  q = p < thresh ?
    StatsFuns.tdistinvcdf(ν, p/(1 - λ)) :
    StatsFuns.tdistinvcdf(ν, 0.5 + (p - thresh)/(1 + λ))

  signp = p < thresh ? -1 : 1
  factor = sqrt((ν - 2)/ν)
  z = (q * (1 + signp*λ) * factor - a) / b
  μ + σ*z
end

rand(rng::Random.AbstractRNG, d::SkewT) = quantile(d, rand(rng))

fit_mle_free(::Type{SkewT}, x::AbstractVector{<:Real}) = begin
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

fit_mle_fixed(::Type{SkewT}, x::AbstractVector{<:Real}; fix::NamedTuple) = begin
  min_ν = 2.051
  init_named = (μ=mean(x), σ=log(std(x)), ν=log(5-min_ν), λ=0)
  init = [init_named[k] for k in setdiff(keys(init_named), keys(fix))]
  init = Float64.(init) # To avoid mixing Dual with Float64

  @inline decode(Q) = begin
    i=0
    μ = haskey(fix,:μ) ? fix[:μ] : Q[i+=1]
    σ = haskey(fix,:σ) ? fix[:σ] : exp(Q[i+=1])
    ν = haskey(fix,:ν) ? fix[:ν] : exp(Q[i+=1]) + min_ν
    λ = haskey(fix,:λ) ? fix[:λ] : tanh(Q[i+=1])
    μ, σ, ν, λ = promote(μ, σ, ν, λ) # To avoid mixing Dual with Float64
    (μ,σ,ν,λ)
  end

  nll(Q) = begin
    μ, σ, ν, λ = decode(Q)
    if σ < 1e-12 || ν <= min_ν || abs(λ) >= 1 || ν > 100 return Inf end
    d = SkewT(μ, σ, ν, λ)
    -sum(logpdf.(Ref(d), x))
  end

  res = Optim.optimize(nll, init, Optim.BFGS(); autodiff = :forward)

  Q = Optim.converged(res) ? Optim.minimizer(res) : error("Can't estimate SkewT")
  SkewT(decode(Q)...)
end

fit_mle(::Type{SkewT}, x::AbstractVector{<:Real}, fix::Union{Nothing,NamedTuple}=nothing) = begin
  fix === nothing ? fit_mle_free(SkewT, x) : fit_mle_fixed(SkewT, x; fix)
end

mean_exp(d::SkewT; l::Real, h::Real) = begin
  f(x) = pdf(d, x) * exp(x)
  QuadGK.quadgk(f, l, h)[1]
end

# Fast approx for `adj = log E[e^x] - μ`, with truncated upper tail at 0.9999 quantile.
# Details skewt/fit_skewt_mean_exp.jl
@inline skewt_mean_exp_adj(σ::Real, ν::Real, λ::Real) = begin
  # Truncated at 0.9999 quantile
  # @assert 0.002 <= σ <= 0.3 "σ out of range" # ignored, it may go out during fitting
  @assert 2.5 <= ν <= 8.0 "ν out of range: $ν"
  @assert -0.1 <= λ <= 0.05 "λ out of range: $λ"

  lν = log(ν-2.4)
  m1 = exp(
    -0.6446035285361307 -0.00973506852001564*lν +0.948060065080043*λ
    -0.009137070041612714*lν*lν +0.008049584856130821*λ*λ +
    -0.27304426222650746*lν*λ
  )
  m2 = -0.007834351347080036 +0.003499118160540425*lν -0.04135562494536131*λ
  -4.2401628337359495e-5 + m1*σ*σ + m2*σ
end;

# Very simple approx for `adj = log E[e^x] - μ`, with truncated upper tail at 0.9999 quantile.
# Details skewt/fit_skewt_mean_exp.jl
skewt_mean_exp_adj_simple(σ::Real, ν::Real, λ::Real) = begin
  # Truncated at 0.9999 quantile
  # @assert 0.002 <= σ <= 0.3 "σ out of range" # ignored, it may go out during fitting
  @assert 2.5 <= ν <= 8.0 "ν out of range"
  @assert -0.1 <= λ <= 0.05 "λ out of range"

  (0.8741839146540278 + 0.07021915718742801*log(ν) + 0.5156547228197067*λ)*σ*σ/2
end;


# Keeping just as a comparison, bounded version is twice slower as reparametrised
# version `fit_mle_free`.
fit_mle_free_bounded(::Type{SkewT}, x::AbstractVector{<:Real}) = begin
  lower = [-Inf, 1e-12, 2.051, -1.0+1e-6]
  upper = [ Inf, Inf,   Inf,    1.0-1e-6]

  nll(Q) = begin
    d = SkewT(Q...)
    -sum(logpdf.(Ref(d), x))
  end

  # Sometimes it doens't converge, trying various inits
  inits = [
    [mean(x),   std(x) + 1e-6,                       4,  0],
    [median(x), median(abs.(x .- median(x))) + 1e-6, 10, 0]
  ]

  for init in inits
    res = Optim.optimize(nll, lower, upper, init, Optim.Fminbox(Optim.BFGS()); autodiff = :forward)
    !Optim.converged(res) && continue
    Q = Optim.minimizer(res)
    return SkewT(Q...)
  end

  error("Can't estimate SkewT")
end