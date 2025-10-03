import Statistics, SpecialFunctions, StatsFuns, Optim, Distributions, Random
import Distributions: logpdf, pdf, cdf, quantile, fit_mle
import Random: rand
import Statistics: mean, std
import QuadGK, Interpolations

# struct SkewT{T <: Real} <: Distributions.ContinuousUnivariateDistribution
#   μ::T; σ::T; ν::T; λ::T
#   a::T; b::T; c::T
# end
struct SkewT <: Distributions.ContinuousUnivariateDistribution
  μ::Real; σ::Real; ν::Real; λ::Real
  a::Real; b::Real; c::Real
end

function SkewT(μ::Real, σ::Real, ν::Real, λ::Real) #where {T <: Real}
  σ > 0       || throw(DomainError(σ, "σ must be > 0"))
  ν > 2.05    || throw(DomainError(ν, "ν must be > 2.05"))
  abs(λ) < 1  || throw(DomainError(λ, "|λ| must be < 1"))

  c = SpecialFunctions.loggamma((ν + 1)/2) - SpecialFunctions.loggamma(ν/2) - 0.5*log(pi*(ν - 2))
  a = 4*λ*exp(c)*(ν - 2)/(ν - 1)
  b = sqrt(1 + 3*λ^2 - a^2)

  SkewT(μ, σ, ν, λ, a, b, c)
end

logpdf(d::SkewT, x::Real) = begin
  (; μ, σ, λ, ν, a, b, c) = d
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

  @inline decode(θ) = begin
    i=0
    μ = haskey(fix,:μ) ? fix[:μ] : θ[i+=1]
    σ = haskey(fix,:σ) ? fix[:σ] : exp(θ[i+=1])
    ν = haskey(fix,:ν) ? fix[:ν] : exp(θ[i+=1]) + min_ν
    λ = haskey(fix,:λ) ? fix[:λ] : tanh(θ[i+=1])
    μ, σ, ν, λ = promote(μ, σ, ν, λ) # To avoid mixing Dual with Float64
    (μ,σ,ν,λ)
  end

  nll(θ) = begin
    μ, σ, ν, λ = decode(θ)
    if σ < 1e-12 || ν <= min_ν || abs(λ) >= 1 || ν > 100 return Inf end
    d = SkewT(μ, σ, ν, λ)
    -sum(logpdf.(Ref(d), x))
  end

  res = Optim.optimize(nll, init, Optim.BFGS(); autodiff = :forward)

  θ = Optim.converged(res) ? Optim.minimizer(res) : error("Can't estimate SkewT")
  SkewT(decode(θ)...)
end

fit_mle(::Type{SkewT}, x::AbstractVector{<:Real}, fix::Union{Nothing,NamedTuple}=nothing) = begin
  fix === nothing ? fit_mle_free(SkewT, x) : fit_mle_fixed(SkewT, x; fix)
end

mean_exp(d::SkewT; l::Real, h::Real) = begin
  f(x) = pdf(d, x) * exp(x)
  QuadGK.quadgk(f, l, h)[1]
end

# Fast approximate solution for E[e^x] with truncated upper tail and its inverse
# as `adj = log E[e^x] - μ`. Details skewt/fit_skewt_mean_exp.jl
make_mean_exp_approx_adj() = begin
  σ_range = (0.001,  0.5)
  ν_range = (2.7,    10.0)
  λ_range = (-0.1,   0.1)
  hp_range = 0.9999

  Q = [
    -9.784757129983294, 0.22962024441216067, -2.9396608958256905, 0.062033597122393314, 0.5260265105943711,
    0.4288006423257013, -7.091982222193769, -0.3451145220530314, 8.061066984268344, -0.003940296718428804,
    0.0010526088347396069, -0.023738010983512296, 0.14907271265816044, 0.14344803112579835, -0.7775383881678878
  ] # re = 1.001

  skewt_mean_exp_adj(σ, ν, λ, hp, Q, q085=nothing, q099=nothing) = begin
    @assert σ_range[1] <= σ <= σ_range[2] "σ out of range"
    @assert ν_range[1] <= ν <= ν_range[2] "ν out of range"
    @assert λ_range[1] <= λ <= λ_range[2] "λ out of range"
    @assert hp ≈ hp_range "hp out of range"

    if q085 === nothing || q099 === nothing
      d0 = SkewT(0.0, σ, ν, λ)
      q085, q099 = quantile(d0, 0.85), quantile(d0, 0.99)
    end

    lν = log(ν-2.5)
    a, b = q085, q099; la, lb = log(a), log(b)

    m1 = exp(Q[1] + Q[2]lν + Q[3]λ + Q[4]lν^2 + Q[5]λ^2 + Q[6]*lν*λ + Q[7]*la + Q[8]*la^2 + Q[9]*lb)
    m2 = Q[10] + Q[11]lν + Q[12]λ + Q[13]*a + Q[14]*b
    (σ*m1)^2 + σ*m2
  end;

  (σ, ν, λ, hp) -> skewt_mean_exp_adj(σ, ν, λ, hp, Q)
end