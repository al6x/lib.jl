import Statistics, SpecialFunctions, StatsFuns, Optim, Distributions, Random
import Distributions: logpdf, pdf, cdf, quantile, fit_mle
import Random: rand
import Statistics: mean, std
import QuadGK, Interpolations
import Roots
using StatsFuns: normlogpdf, normlogcdf, logaddexp

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
  le_mean::Real, ws::NTuple{N, <:Real}, σ::NTuple{N, <:Real}, x::Real
) where {N} = begin
  μ = le_mean .- 0.5 .* σ .* σ # means corresponding to le_mean.
  μ_mix = sum(ws .* μ)         # mixture mean

  nmix_logpdf(ws, μ, σ, x), μ_mix
end

@inline nmix_le_mean_logpdf(
  le_mean::Real, ws::NTuple{N, <:Real}, σ::NTuple{N, <:Real}, α::Real, x::Real
) where {N} = begin
  @assert -10.0 <= α <= 10.0 α

  # means corresponding to le_mean.
  δ = α / sqrt(1 + α*α)
  μ = le_mean .- 0.5 .* σ .* σ .- (log(2) .+ normlogcdf.(δ .* σ))

  # mixture mean
  κ = sqrt(2/π) * δ
  μ_mix = sum(ws .* (μ .+ κ .* σ))

  nmix_logpdf(ws, μ, σ, α, x), μ_mix
end

# tmix ---------------------------------------------------------------------------------------------
@inline tmix_ws_ss(ν::Real) = begin
  # See fit_tmix.jl for fitting
  @assert 2.5 <= ν <= 15.0 ν

  a = -1.72159 + 3.223223*ν - 0.103005*ν*ν

  ws = exp.(
    (-51.060801, -54.117216, -59.211241, -66.342876, -75.512121) ./ a .+
    (  1.592159,   2.482642,   2.671449,   2.15858,    0.944035)
  )
  ws = ws ./ sum(ws)

  ss = exp.((-6.455494, -3.495181, 0.586545, 5.789699, 12.114266) ./ a)

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

@inline tmix_le_mean_logpdf(le_mean::Real, s::Real, ν::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = tmix_ws_ss(ν)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, x)
end

@inline tmix_le_mean_logpdf(le_mean::Real, s::Real, ν::Real, α::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = tmix_ws_ss(ν)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, α, x)
end

# nigmix -------------------------------------------------------------------------------------------
@inline nigmix_ws_ss(α::Real) = begin
  # See fit_nigmix.jl for fitting
  @assert 0.7 <= α <= 10.0 α

  lα = log(α)
  a = exp(0.331822 + 0.208549*lα + 0.012952*lα*lα)

  ws = exp.(
    (17.489715, 16.744012,  14.622189, 11.124246,  6.250183) .-
    (16.81154,  13.449232,  10.086924,  6.724616,  3.362308) ./ a
  )
  ws = ws ./ sum(ws)

  ss = exp.(
    (-3.26369264, -3.011738, -2.7257566, -2.3865299, -1.944441) .+
    (5.55397,      4.443176, 3.332382,    2.221588,   1.110794) ./a
  )

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

@inline nigmix_le_mean_logpdf(le_mean::Real, s::Real, α::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = nigmix_ws_ss(α)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, x)
end

@inline nigmix_le_mean_logpdf(le_mean::Real, s::Real, α::Real, α2::Real, x::Real) = begin
  @assert 0 < s s
  ws, ss = nigmix_ws_ss(α)
  nmix_le_mean_logpdf(le_mean, ws, ss .* s, α2, x)
end