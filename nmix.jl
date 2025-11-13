import Statistics, SpecialFunctions, StatsFuns, Optim, Distributions, Random
import Distributions: logpdf, pdf, cdf, quantile, fit_mle
import Random: rand
import Statistics: mean, std
import QuadGK, Interpolations
import Roots
using StatsFuns: normcdf, normlogpdf, normlogcdf, logaddexp

struct NMix{T<:Real} <: Distributions.ContinuousUnivariateDistribution
  w::NTuple{5, T}; μ::NTuple{5, T}; σ::NTuple{5, T}
end

@inline nmix_logpdf(
  w::NTuple{N, <:Real}, μ::Union{Real, NTuple{N, <:Real}}, σ::NTuple{N, <:Real}, x::Real
) where {N} = begin
  z  = (x .- μ) ./ σ
  aj = log.(w) .- log.(σ) .+ normlogpdf.(z)
  m  = maximum(aj)
  m + log(sum(exp.(aj .- m)))
end

@inline logpdf(d::NMix, x::Real) = nmix_logpdf(d.w, d.μ, d.σ, x)

@inline nmix_logpdf(
  w::NTuple{N, <:Real}, μ::Union{Real, NTuple{N, <:Real}}, σ::NTuple{N, <:Real}, α::Real, x::Real
) where {N} = begin
  @assert -10.0 <= α <= 10.0 α
  z  = (x .- μ) ./ σ
  aj = log.(w) .+ log(2) .- log.(σ) .+ normlogpdf.(z) .+ normlogcdf.(α .* z)
  m  = maximum(aj)
  m + log(sum(exp.(aj .- m)))
end

@inline logpdf(d::NMix, α::Real, x::Real) = nmix_logpdf(d.w, d.μ, d.σ, α, x)

pdf(d::NMix, x::Real) = exp(logpdf(d, x))

cdf(d::NMix, x::Real) = sum(d.w .* normcdf.((x .- d.μ) ./ d.σ))

quantile(d::NMix, p::Real) = begin
  @assert 1e-8 <= p <= 1 - 1e-8 "p must be in [1e-8, 1-1e-8]"
  l, h = Inf, -Inf
  for i in 1:length(d.w)
    li, hi = d.μ[i] - 6*d.σ[i], d.μ[i] + 6*d.σ[i]
    li < l && (l = li); hi > h && (h = hi)
  end
  Roots.find_zero(x -> cdf(d, x) - p, (l, h), Roots.Brent())
end

rand(rng::Random.AbstractRNG, d::NMix) = begin
  (; w, μ, σ) = d
  u = rand(rng)
  i = u < w[1] ? 1 : (u < w[1] + w[2] ? 2 : 3)
  μ[i] + σ[i] * randn(rng)
end

mean_exp(d::NMix) = sum(d.w .* exp.(d.μ .+ 0.5 .* d.σ.^2))

mean_exp(d::NMix, α::Real) = begin
  δ = α / sqrt(1 + α*α)
  sum(d.w .* exp.(d.μ .+ 0.5 .* d.σ.*d.σ) .* (2 .* normcdf.(δ .* d.σ)))
end

@inline nmix_le_mean_logpdf(
  le_mean::Real, ws::NTuple{N, <:Real}, σ::NTuple{N, <:Real}, x::Real; same_loc=true
) where {N} = begin
  # μ - means corresponding to le_mean, μ_mix - mixture mean E[x]
  if same_loc
    μ = le_mean - log(sum(ws .* exp.(0.5 .* σ .* σ)))
    μ_mix = μ
  else
    μ = le_mean .- 0.5 .* σ .* σ
    μ_mix = sum(ws .* μ)
  end
  nmix_logpdf(ws, μ, σ, x), μ_mix
end

@inline nmix_le_mean_logpdf(
  le_mean::Real, ws::NTuple{N, <:Real}, σ::NTuple{N, <:Real}, α::Real, x::Real; same_loc=true
) where {N} = begin
  @assert -10.0 <= α <= 10.0 α

  # μ - means corresponding to le_mean, μ_mix - mixture mean E[x]
  δ = α / sqrt(1 + α*α)
  κ = sqrt(2/π) * δ
  if same_loc
    s = sum(ws .* exp.(0.5 .* σ .* σ) .* (2 .* normcdf.(δ .* σ)))
    μ = le_mean - log(s)
    μ_mix = μ + κ * sum(ws .* σ)
  else
    μ = le_mean .- 0.5 .* σ .* σ .- (log(2) .+ normlogcdf.(δ .* σ))
    μ_mix = sum(ws .* (μ .+ κ .* σ))
  end

  nmix_logpdf(ws, μ, σ, α, x), μ_mix
end

# tmix ---------------------------------------------------------------------------------------------
@inline tmix_ws_ss(ν::Real) = begin
  # See fit_tmix.jl for fitting
  # Q=[
  #   -0.523857, -0.349351, -2.164389, -0.074823, 3.886313, 1.234103,
  #   -0.608911, -1.804228, 0.278592, 7.081085, -3.867854, -2.255962, 9.032569
  # ], loss=1.00156

  @assert 2.5 <= ν <= 15.0 ν

  lν = log(ν)
  v = exp(-0.608911 - 1.804228*lν + 0.278592*lν*lν)

  ws = exp.(-0.523857 .+ (-7.304031, 1.067509, 9.439049, 17.810589, 4.533427, -19.899341) .* v)
  ws = ws ./ sum(ws)

  ss = exp.(-0.074823 .+ ( 7.851771, 3.07844, -1.694891, -6.468222, 1.102224,  15.033435) .* v)

  ws, ss
end

@inline tmix_logpdf(loc::Real, s::Real, ν::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = tmix_ws_ss(ν)
  nmix_logpdf(ws, loc, ss .* s, x)
end

@inline tmix_logpdf(loc::Real, s::Real, ν::Real, α::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = tmix_ws_ss(ν)
  nmix_logpdf(ws, loc, ss .* s, α, x)
end

@inline tmix_le_mean_logpdf(le_mean::Real, s::Real, ν::Real, x::Real; same_loc=true) = begin
  @assert 0 < s s
  ws, ss = tmix_ws_ss(ν)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, x; same_loc)
end

@inline tmix_le_mean_logpdf(le_mean::Real, s::Real, ν::Real, α::Real, x::Real; same_loc=true) = begin
  @assert 0 < s s
  ws, ss = tmix_ws_ss(ν)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, α, x; same_loc)
end

# nigmix -------------------------------------------------------------------------------------------
@inline nigmix_ws_ss(α::Real) = begin
  # See fit_nigmix.jl for fitting
  # Q=[
  #   0.955056, 0.14849, -0.361283, -1.811039, 0.635754, 0.185508, 0.710302,
  #   -0.278778, -0.076414, 0.737572
  # ], loss=1.000659

  @assert 0.7 <= α <= 10.0 α

  lα = log(α)
  v = exp(0.710302 - 0.278778*lα - 0.076414*lα*lα)

  ws = exp.( 0.955056 .+ (0.14849, -0.212793, -0.453901, -0.663889, -0.855917, -1.035609) .* v)
  ws = ws ./ sum(ws)

  ss = exp.(-1.811039 .+ (0.635754, 0.821262,  0.945064,  1.052886,  1.151487,  1.243753) .* v)

  ws, ss
end

@inline nigmix_logpdf(loc::Real, scale::Real, α::Real, x::Real) = begin
  @assert 0 < scale scale
  ws, ss = nigmix_ws_ss(α)
  nmix_logpdf(ws, loc, ss .* scale, x)
end

@inline nigmix_logpdf(loc::Real, s::Real, α::Real, α2::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = nigmix_ws_ss(α)
  nmix_logpdf(ws, loc, ss .* s, α2, x)
end

@inline nigmix_le_mean_logpdf(le_mean::Real, s::Real, α::Real, x::Real; same_loc=true) = begin
  @assert 0 < s s
  ws, ss = nigmix_ws_ss(α)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, x; same_loc)
end

@inline nigmix_le_mean_logpdf(
  le_mean::Real, s::Real, α::Real, α2::Real, x::Real; same_loc=true
) = begin
  @assert 0 < s s
  ws, ss = nigmix_ws_ss(α)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, α2, x; same_loc)
end