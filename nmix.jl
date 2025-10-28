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

@inline nmix_le_mean_logpdf(le_mean::Real, s::Real, wr::Real, sr::Real, x::Real) = begin
  # ratios for weights and scales for 5 components, so that w1/w5 = wr, s5/s1 = sr.
  # The 0,4 indices used instead of 1,5 to simplify calculations.
  # w_pow, r_pow = 2.0, 2.0

  @assert 0 < s && 0 < wr <= 1000 && 0 < sr <= 100

  wm = -log(wr) / 16 # wm = -log(wr) / (4.0^w_pow)
  sm =  log(sr) / 16 # sm =  log(sr) / (4.0^r_pow)

  # weights, w = exp.(wm .* i .^ w_pow) / wsum
  w1, w2, w3, w4, w5 = 1.0, exp(wm), exp(4wm), exp(9wm), exp(16wm)
  wsum = w1 + w2 + w3 + w4 + w5
  w1, w2, w3, w4, w5 = w1 / wsum, w2 / wsum, w3 / wsum, w4 / wsum, w5 / wsum

  # scales, s_k = exp.(sm .* i .^ r_pow) .* s
  s1, s2, s3, s4, s5 = s, exp(sm)*s, exp(4sm)*s, exp(9sm)*s, exp(16sm)*s

  # local means (le_mean - 0.5 * s_k^2)
  l1 = le_mean - 0.5 * s1*s1
  l2 = le_mean - 0.5 * s2*s2
  l3 = le_mean - 0.5 * s3*s3
  l4 = le_mean - 0.5 * s4*s4
  l5 = le_mean - 0.5 * s5*s5

  # mixture mean
  μ = w1*l1 + w2*l2 + w3*l3 + w4*l4 + w5*l5

  # skew-normal mixture logpdf at x
  # log w + log 2 - log σ + log φ(z) + log Φ(α z)
  # log2 = 0.6931471805599453
  # z1 = (x - l1)/s1; a1 = log(w1) + log2 - log(s1) + normlogpdf(z1) + normlogcdf(α*z1)
  # z2 = (x - l2)/s2; a2 = log(w2) + log2 - log(s2) + normlogpdf(z2) + normlogcdf(α*z2)
  # z3 = (x - l3)/s3; a3 = log(w3) + log2 - log(s3) + normlogpdf(z3) + normlogcdf(α*z3)
  # z4 = (x - l4)/s4; a4 = log(w4) + log2 - log(s4) + normlogpdf(z4) + normlogcdf(α*z4)
  # z5 = (x - l5)/s5; a5 = log(w5) + log2 - log(s5) + normlogpdf(z5) + normlogcdf(α*z5)

  z1 = (x - l1)/s1; a1 = log(w1) - log(s1) + normlogpdf(z1)
  z2 = (x - l2)/s2; a2 = log(w2) - log(s2) + normlogpdf(z2)
  z3 = (x - l3)/s3; a3 = log(w3) - log(s3) + normlogpdf(z3)
  z4 = (x - l4)/s4; a4 = log(w4) - log(s4) + normlogpdf(z4)
  z5 = (x - l5)/s5; a5 = log(w5) - log(s5) + normlogpdf(z5)

  # Pairwise stable summation
  logpdf = logaddexp(logaddexp(logaddexp(a1, a2), logaddexp(a3, a4)), a5)
  logpdf, μ
end

@inline nmix_le_mean_logpdf(le_mean::Real, s::Real, wr::Real, sr::Real, α::Real, x::Real) = begin
  # ratios for weights and scales for 5 components, so that w1/w5 = wr, s5/s1 = sr.
  # The 0,4 indices used instead of 1,5 to simplify calculations.
  # w_pow, r_pow = 2.0, 2.0

  @assert 0 < s && 0 < wr <= 1000 && 0 < sr <= 100 && -0.5 <= α <= 0.5

  wm = -log(wr) / 16 # wm = -log(wr) / (4.0^w_pow)
  sm =  log(sr) / 16 # sm =  log(sr) / (4.0^r_pow)

  # weights, w = exp.(wm .* i .^ w_pow) / wsum
  w1, w2, w3, w4, w5 = 1.0, exp(wm), exp(4wm), exp(9wm), exp(16wm)
  wsum = w1 + w2 + w3 + w4 + w5
  w1, w2, w3, w4, w5 = w1 / wsum, w2 / wsum, w3 / wsum, w4 / wsum, w5 / wsum

  # scales, s_k = exp.(sm .* i .^ r_pow) .* s
  s1, s2, s3, s4, s5 = s, exp(sm)*s, exp(4sm)*s, exp(9sm)*s, exp(16sm)*s

  # local means (le_mean - 0.5 * s_k^2)
  l1 = le_mean - 0.5 * s1*s1
  l2 = le_mean - 0.5 * s2*s2
  l3 = le_mean - 0.5 * s3*s3
  l4 = le_mean - 0.5 * s4*s4
  l5 = le_mean - 0.5 * s5*s5

  # mixture mean
  μ = w1*l1 + w2*l2 + w3*l3 + w4*l4 + w5*l5

  # skew-normal mixture logpdf at x
  log2 = 0.6931471805599453
  z1 = (x - l1)/s1; a1 = log(w1) + log2 - log(s1) + normlogpdf(z1) + normlogcdf(α*z1)
  z2 = (x - l2)/s2; a2 = log(w2) + log2 - log(s2) + normlogpdf(z2) + normlogcdf(α*z2)
  z3 = (x - l3)/s3; a3 = log(w3) + log2 - log(s3) + normlogpdf(z3) + normlogcdf(α*z3)
  z4 = (x - l4)/s4; a4 = log(w4) + log2 - log(s4) + normlogpdf(z4) + normlogcdf(α*z4)
  z5 = (x - l5)/s5; a5 = log(w5) + log2 - log(s5) + normlogpdf(z5) + normlogcdf(α*z5)

  # Pairwise stable summation
  logpdf = logaddexp(logaddexp(logaddexp(a1, a2), logaddexp(a3, a4)), a5)
  logpdf, μ
end

@inline logpdf(d::NMix, x::Real) = begin
  s = -Inf; @inbounds for j in 1:5
    z  = (x - d.μ[j]) / d.σ[j]
    aj = log(d.w[j]) - log(d.σ[j]) + normlogpdf(z)
    s  = logaddexp(s, aj)
  end; s
end

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

# Skewed -------------------------------------------------------------------------------------------
@inline logpdf(d::NMix, α::Real, x::Real) = begin
  s = -Inf; @inbounds for j in 1:5
    z  = (x - d.μ[j]) / d.σ[j]
    aj = log(d.w[j]) + log(2) - log(d.σ[j]) + normlogpdf(z) + normlogcdf(α*z)
    s  = logaddexp(s, aj)
  end; s
end

mean_exp(d::NMix, α::Real) = begin
  δ = α / sqrt(1 + α^2)
  sum(d.w .* exp.(d.μ .+ 0.5 .* d.σ.^2) .* (2 .* normcdf.(δ .* d.σ)))
end


# @inline nmix_logpdf(w::NTuple{5, T}, μ::NTuple{5, T}, σ::NTuple{5, T}, x::Real) where {T<:Real} = begin
#   s = -Inf; @inbounds for j in 1:length(w)
#     z  = (x - μ[j]) / σ[j]
#     aj = log(w[j]) - log(σ[j]) + normlogpdf(z)
#     s  = logaddexp(s, aj)
#   end; s
# end

# @inline nmix_logpdf(w::NTuple{5,Real}, μ::NTuple{5,Real}, σ::NTuple{5,Real}, α::Real, x::Real) = begin
#   s = -Inf
#   @inbounds for j in 1:5
#     z  = (x - μ[j]) / σ[j]
#     aj = log(w[j]) + log(2) - log(σ[j]) + normlogpdf(z) + normlogcdf(α*z)
#     s  = logaddexp(s, aj)
#   end
#   s
# end