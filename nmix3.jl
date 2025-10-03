import Statistics, SpecialFunctions, StatsFuns, Optim, Distributions, Random
import Distributions: logpdf, pdf, cdf, quantile, fit_mle
import Random: rand
import Statistics: mean, std
import QuadGK, Interpolations
import Roots: find_zero, Brent

struct NMix3{T <: Real} <: Distributions.ContinuousUnivariateDistribution
  w::NTuple{3, T}, μ::NTuple{3, T}, σ::NTuple{3, T}
end

NMix3(w::NTuple{3, T}, μ::NTuple{3, T}, σ::NTuple{3, T}) where {T <: Real} = begin
  # @assert w[3] ≥ 0      "w3 must be ≥ 0"
  # @assert w[2] ≥ 2*w[3] "w2 must be ≥ 2*w3"
  # @assert w[1] ≥ 2*w[2] "w1 must be ≥ 2*w2"
  # @assert 1-1e-8 < w[1] + w[2] + w[3] ≤ 1.0  "weights must sum to 1"

  # @assert σ[1] ≥ 1e-6     "σ1 must be ≥ 1e-6"
  # @assert σ[2] ≥ 1.2σ[1] "σ2 must be ≥ 1.2*σ1"
  # @assert σ[3] ≥ 1.2σ[2] "σ3 must be ≥ 1.2*σ2"

  NMix3{T}(w, μ, σ)
end

logpdf(d::NMix3, x::Real) = begin
  (; w, μ, σ) = d

  z1 = (x - μ[1]) / σ[1]
  z2 = (x - μ[2]) / σ[2]
  z3 = (x - μ[3]) / σ[3]

  a1 = log(w[1]) - log(σ[1]) + normlogpdf(z1)
  a2 = log(w[2]) - log(σ[2]) + normlogpdf(z2)
  a3 = log(w[3]) - log(σ[3]) + normlogpdf(z3)

  m = max(a1, a2, a3)
  m + log(exp(a1 - m) + exp(a2 - m) + exp(a3 - m))
end

pdf(d::NMix3, x::Real) = exp(logpdf(d, x))

cdf(d::NMix3, x::Real) = begin
  w, μ, σ = d.ws, d.μs, d.σs
  w[1]*StatsFuns.normcdf((x - μ[1])/σ[1]) +
  w[2]*StatsFuns.normcdf((x - μ[2])/σ[2]) +
  w[3]*StatsFuns.normcdf((x - μ[3])/σ[3])
end

quantile(d::NMix3, p::Real) = begin
  @assert 0.0 < p < 1.0 "p must be in (0,1)"
  find_zero(x -> cdf(d, x) - p, (-Inf, Inf), Brent())
end

rand(rng::Random.AbstractRNG, d::NMix3) = begin
  (; w, μ, σ) = d
  u = rand(rng)
  i = u < w[1] ? 1 : (u < w[1] + w[2] ? 2 : 3)
  μ[i] + σ[i] * randn(rng)
end

mean_exp(d::NMix3) = begin
  (; w, μ, σ) = d
  w[1]*exp(μ[1] + 0.5*σ[1]^2) + w[2]*exp(μ[2] + 0.5*σ[2]^2) + w[3]*exp(μ[3] + 0.5*σ[3]^2)
end

fit_mle_free(::Type{NMix3}, x::AbstractVector{<:Real}) = begin
  decode(Q) = begin
    Qw = (tanh.(Q[1:3]) .+ 1) ./ 2
    w1 = 0.5 + 0.5*Qw[1]; w2 = 0.5*w1*Qw[2]; w3 = 0.5*w2*Qw[3]
    wsum = w1 + w2 + w3
    w1 /= wsum; w2 /= wsum; w3 /= wsum

    σ1 = 1e-6 + exp(Q[7]); σ2 = 1.2*σ1 + exp(Q[8]); σ3 = 1.2*σ2 + exp(Q[9])

    (w1, w2, w3), (Q[4], Q[5], Q[6]), (σ1, σ2, σ3)
  end

  nll(Q) = begin
    d = NMix3(decode(Q)...)
    -sum(logpdf.(Ref(d), x))
  end

  m, s = mean(x), std(x)
  @assert s > 0 "data std must be > 0"
  inits = [
    [0.0, 0.0, 0.0, m, m, m, log(s), log(s), log(s)],
  ]

  for init in inits
    res = Optim.optimize(nll, init, Optim.BFGS(); autodiff = :forward)
    !Optim.converged(res) && continue
    Q = Optim.minimizer(res)
    return NMix3(decode(Q)...)
  end

  error("Can't estimate NMix3")
end