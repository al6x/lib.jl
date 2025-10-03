# ===== Robust MLE for NIG (univariate) =====
# Reparam: θ = (β, ℓδ, μ, a) with δ = exp(ℓδ), α = √(β² + exp(a)) ⇒ α>|β|, δ>0 always.

using Optim, ForwardDiff, Statistics

_unpack(θ) = (sqrt(θ[1]^2 + exp(θ[4])), θ[1], exp(θ[2]), θ[3])

# Stable logpdf already defined earlier; using that here.
function _nll(θ::AbstractVector, x::AbstractVector{<:Real})
  α, β, δ, μ = _unpack(θ)
  d = NIG(α, β, δ, μ)
  s = 0.0
  @inbounds @simd for xi in x
    li = logpdf(d, xi)
    if !isfinite(li); return Inf end
    s -= li
  end
  return s
end

# Neutral & quick starts
function _init_θ_basic(x::AbstractVector{<:Real})
  μ0 = mean(x); s = std(x); s = s>0 ? s : 1.0
  β0, δ0 = 0.0, s
  α0 = sqrt(β0^2 + 1/s^2)                # keep α0>|β0|
  return [β0, log(δ0), μ0, log(max(α0^2 - β0^2, 1e-6))]
end

# Slightly perturbed starts for multi-start
function _init_θ_perturb(x; k=4)
  θ0 = _init_θ_basic(x)
  outs = Vector{typeof(θ0)}()
  push!(outs, θ0)
  for _ in 2:k
    β = θ0[1] + 0.5*randn()
    ℓδ = θ0[2] + 0.25*randn()
    μ = θ0[3] + 0.25*std(x)*randn()
    a = θ0[4] + 0.5*randn()
    push!(outs, [β, ℓδ, μ, a])
  end
  return outs
end

# Core optimizer call
function _optim_run(x, θ0; maxiters=10_000, autodiff::Bool=true)
  ad = autodiff ? Optim.AutoForwardDiff() : Optim.Disabled()
  res = optimize(θ -> _nll(θ, x), θ0, LBFGS(); autodiff=ad, maxiters=maxiters)
  return res
end

"""
  fit_nig(x; multistart=1, maxiters=10_000, autodiff=true, θ0=nothing)

Return (d::NIG, res::Optim.MultivariateOptimizationResults).
- Uses unconstrained reparam to enforce α>|β|, δ>0.
- If `multistart>1`, tries several perturbed starts and keeps the best.
"""
function fit_nig(x::AbstractVector{<:Real};
                 multistart::Int=1, maxiters::Int=10_000,
                 autodiff::Bool=true, θ0=nothing)
  starts = θ0 === nothing ? (multistart>1 ? _init_θ_perturb(x; k=multistart) : [_init_θ_basic(x)]) : [θ0]
  best_res, best_val, best_θ = nothing, Inf, nothing
  for s in starts
    res = _optim_run(x, s; maxiters=maxiters, autodiff=autodiff)
    val = Optim.minimum(res)
    if isfinite(val) && val < best_val
      best_res, best_val, best_θ = res, val, Optim.minimizer(res)
    end
  end
  @assert best_res !== nothing "fit_nig failed to converge from provided starts."
  α, β, δ, μ = _unpack(best_θ)
  return NIG(α, β, δ, μ), best_res
end