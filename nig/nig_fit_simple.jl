# ===== MLE for NIG (univariate) =====
# Reparam:
# θ = (β, ℓδ, μ, a) with δ = exp(ℓδ), α = √(β² + exp(a))
# Ensures α>|β| and δ>0 automatically.

using Statistics
using Optim
using ForwardDiff

# Map θ -> (α,β,δ,μ)
_unpack(θ) = (sqrt(θ[1]^2 + exp(θ[4])), θ[1], exp(θ[2]), θ[3])

# Negative log-likelihood (uses your stabilized logpdf)
function _nll(θ::AbstractVector, x::AbstractVector{<:Real})
  α, β, δ, μ = _unpack(θ)
  d = NIG(α, β, δ, μ)
  s = 0.0
  @inbounds @simd for xi in x
    s -= logpdf(d, xi)
  end
  return s
end

# Simple, robust starting values
function _init_θ(x::AbstractVector{<:Real})
  μ0 = mean(x)
  s  = std(x)
  s = s > 0 ? s : 1.0
  β0 = 0.0                      # neutral start; skewness-based start optional
  δ0 = s
  α0 = sqrt(β0^2 + 1/s^2)       # keep α0>|β0|
  return [β0, log(δ0), μ0, log(max(α0^2 - β0^2, 1e-6))]
end

# Fit function: returns a NIG instance and the Optim result
function fit_nig(x::AbstractVector{<:Real}; θ0::AbstractVector = _init_θ(x),
                 autodiff::Bool = true, maxiters::Int = 10_000)
  ad = autodiff ? Optim.AutoForwardDiff() : Optim.Disabled()
  obj = θ -> _nll(θ, x)
  res = optimize(obj, θ0, LBFGS(); autodiff = ad, maxiters = maxiters)
  θ̂  = Optim.minimizer(res)
  α, β, δ, μ = _unpack(θ̂)
  return NIG(α, β, δ, μ), res
end