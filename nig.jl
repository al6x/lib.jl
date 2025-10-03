# ========== Normal Inverse Gaussian (NIG) ==========
# Parametrization: α > |β|, δ > 0, μ ∈ ℝ
# pdf(x) = (αδ/π) * exp(δγ + β(x-μ)) * K₁(α*sqrt(δ²+(x-μ)²)) / sqrt(δ²+(x-μ)²)
# where γ = sqrt(α² - β²), K₁ is modified Bessel K of order 1.

using Distributions
using SpecialFunctions: besselk
const _SQRT2 = sqrt(2.0); const _INV_SQRT2PI = 1.0/sqrt(2π)

struct NIG{T<:Real}
  α::T; β::T; δ::T; μ::T; γ::T

  function NIG{T}(α::T, β::T, δ::T, μ::T) where {T<:Real}
    @assert α > 0 "α must be > 0"
    @assert δ > 0 "δ must be > 0"
    @assert α > abs(β) "α must be > |β|"
    γ = sqrt(α^2 - β^2)
    new{T}(α, β, δ, μ, γ)
  end
end
NIG(α::Real, β::Real, δ::Real, μ::Real) = NIG{float(promote_type(typeof.( (α,β,δ,μ) )...))}(α,β,δ,μ)

# --- basic moments (used for numerics)
mean(d::NIG) = d.μ + d.δ*d.β/d.γ
var(d::NIG)  = d.δ * d.α^2 / (d.γ^3)

# --- CDF via Lugannani–Rice saddlepoint approximation (no quadrature)
# K(t)   = μ t + δ(γ - sqrt(α² - (β+t)²))
# K'(t)  = μ + δ(β+t)/sqrt(α² - (β+t)²)
# K''(t) = δ α² / (sqrt(α² - (β+t)²))^3
_k(d::NIG, t) = d.μ*t + d.δ*(d.γ - sqrt(d.α^2 - (d.β + t)^2))
_dk(d::NIG, t) = d.μ + d.δ*(d.β + t)/sqrt(d.α^2 - (d.β + t)^2)
_ddk(d::NIG, t) = d.δ*(d.α^2) / (sqrt(d.α^2 - (d.β + t)^2))^3

# Normal helpers
_normcdf(z) = 0.5 * erfc(-z/_SQRT2)
_normpdf(z) = _INV_SQRT2PI * exp(-0.5*z*z)

# Safe domain endpoints for t (|β+t|<α)
_domain(d::NIG) = ((-d.α - d.β) + 1e-12, (d.α - d.β) - 1e-12)

# Newton-with-bracketing for K'(t) = x
function _solve_saddlepoint_fast(d::NIG, x::Real; tol=1e-12, maxit=40)
  a, b = _domain(d)
  fa, fb = _dk(d, a) - x, _dk(d, b) - x
  @assert fa < 0 && fb > 0 "Monotone bracket failed; check params"

  # sensible initial guess from quadratic CGF approximation at t=0
  m, v = mean(d), var(d)
  t = (x - m) / v
  t = clamp(t, a, b)

  for _ in 1:maxit
    f  = _dk(d, t) - x
    if abs(f) ≤ tol; return t end
    fp = _ddk(d, t)              # >0 (convex)
    # Newton step
    step = f / fp
    tnew = t - step
    # safeguard: keep inside (a,b); if out, bisect
    if !(a < tnew < b)
      tnew = 0.5*(a + b)
    end
    # maintain bracket
    if (_dk(d, tnew) - x) > 0
      b = tnew
    else
      a = tnew
    end
    # convergence on position as well
    if abs(tnew - t) ≤ tol*max(1.0, abs(t)); t = tnew; break; end
    t = tnew
  end
  return t
end

# Replace cdf() to use the fast solver (LR formula unchanged)
function cdf(d::NIG, x::Real)
  m, s = mean(d), sqrt(var(d))
  if abs(x - m) ≤ 1e-8*max(1.0, s)
    # avoid LR singularity near mean
    return 0.5 * erfc(-(x - m) / (sqrt(2.0)*s))
  end
  t = _solve_saddlepoint_fast(d, x)
  K  = _k(d, t)
  K2 = _ddk(d, t)
  w  = sign(t) * sqrt(2*(t*x - K))
  u  = t * sqrt(K2)
  Φw = 0.5 * erfc(-w / sqrt(2.0))
  φw = (1/sqrt(2π)) * exp(-0.5*w*w)
  return Φw + φw * (1/w - 1/u)
end



# ---------- log-CDF (simple & robust enough) ----------
# For now: use log(cdf(...)). (LR-native log form is longer; can add later if needed.)
logcdf(d::NIG, x::Real) = log(cdf(d, x))

# ---------- Quantile via safeguarded bisection ----------
function quantile(d::NIG, p::Real; tol=1e-12, maxit=200)
  @assert 0.0 < p < 1.0 "p ∈ (0,1)"
  m, s = mean(d), sqrt(var(d))

  # initial bracket using normal-ish start; expand until bracketed
  lo, hi = m - 8s, m + 8s
  flo, fhi = cdf(d, lo) - p, cdf(d, hi) - p
  k = 0
  while !(flo < 0 && fhi > 0) && k < 30
    # expand asymmetrically toward the side that doesn't bracket
    if flo ≥ 0; lo -= 2s; flo = cdf(d, lo) - p; end
    if fhi ≤ 0; hi += 2s; fhi = cdf(d, hi) - p; end
    k += 1
  end
  @assert flo < 0 && fhi > 0 "Failed to bracket quantile"

  a, b = lo, hi
  fa, fb = flo, fhi
  for _ in 1:maxit
    mid = 0.5*(a + b)
    fm = cdf(d, mid) - p
    if abs(fm) ≤ tol || (b - a) ≤ tol*max(1.0, abs(mid))
      return mid
    end
    if fm > 0; b = mid; fb = fm else a = mid; fa = fm end
  end
  return 0.5*(a + b)
end

# ---------- Cumulant generating function K(t) and MGF ----------
# Domain: |β + t| < α
function logmgf(d::NIG, t::Real)
  @assert abs(d.β + t) < d.α "logmgf: t outside domain |β+t|<α"
  return d.μ*t + d.δ*(d.γ - sqrt(d.α^2 - (d.β + t)^2))
end
mgf(d::NIG, t::Real) = exp(logmgf(d, t))

# ---------- Jensen adjustment ----------
# c = log E[e^X] = logmgf(d, 1), requires |β+1|<α
jensen_shift(d::NIG) = logmgf(d, 1.0)

# Solve for t such that M_X(t) = m (or equivalently logmgf(t) = log m)
function solve_t_for_mgf(d::NIG, m::Real; tol=1e-12, maxit=100)
  @assert m > 0 "m must be > 0"
  y = log(m)
  τ = 1e-10
  tmin = (-d.α - d.β) + τ
  tmax = ( d.α - d.β) - τ
  # logmgf is convex on (tmin,tmax); use bisection on monotone derivative K'(t)
  # We instead directly bisect on f(t)=logmgf(t)-y since K is convex and continuous.
  a, b = tmin, tmax
  fa, fb = logmgf(d, a) - y, logmgf(d, b) - y
  @assert fa ≤ 0 && fb ≥ 0 "Target outside mgf range on domain"
  t = 0.0
  for _ in 1:maxit
    t = 0.5*(a + b)
    ft = logmgf(d, t) - y
    if abs(ft) ≤ tol || (b - a) ≤ tol*max(1.0, abs(t)); break; end
    (ft > 0) ? (b = t; fb = ft) : (a = t; fa = ft)
  end
  return t
end

# Stable hypot and logs
_hypot(a::Real, b::Real) = hypot(a, b)  # handles large/small nicely

# log K1(z) with asymptotics in the tails
# - small z:  K1(z) ~ 1/z + z*log(z)/2 + ...
# - large z:  Kν(z) ~ sqrt(pi/(2z)) * e^{-z} * (1 + (4ν^2-1)/(8z) + (4ν^2-1)(4ν^2-9)/(2!(8z)^2) + ...)
# We blend with direct log(besselk(1,z)) in the middle.
function _logK1_stable(z::Real)
  @assert z > 0 "logK1 requires z>0"
  if z < 1e-6
    # K1(z) ≈ 1/z + O(z log z); log ≈ -log(z) is sufficient here
    return -log(z)
  elseif z > 50
    # 2-term large-z expansion for higher accuracy
    ν = 1.0
    a1 = (4ν^2 - 1) / (8z)                 # = 3/(8z)
    a2 = ((4ν^2 - 1)*(4ν^2 - 9)) / (2*(8z)^2)  # = (3*-5)/(2*(8z)^2) = -15/(128 z^2)
    logpref = 0.5*log(pi/(2z))
    return logpref - z + log1p(a1 + a2)
  else
    # moderate range: rely on library
    return log(besselk(1.0, z))
  end
end

# New logpdf using stabilized pieces
function logpdf(d::NIG, x::Real)
  y = x - d.μ
  r = _hypot(d.δ, y)              # sqrt(δ^2 + (x-μ)^2) safely
  z = d.α * r
  # log f = log(αδ/π) + δγ + βy + log K1(z) - log r
  return (log(d.α) + log(d.δ) - log(pi)) + d.δ*d.γ + d.β*y + _logK1_stable(z) - log(r)
end

# --- logpdf / pdf
function logpdf_unstable(d::NIG, x::Real)
  # Direct but uses bessel and is less stable
  y = x - d.μ
  r = sqrt(d.δ^2 + y^2)
  z = d.α * r
  # log((αδ/π) * exp(δγ + βy) * K1(z) / r)
  return log(d.α) + log(d.δ) - log(pi) + d.δ*d.γ + d.β*y + log(besselk(1.0, z)) - log(r)
end

pdf(d::NIG, x::Real) = exp(logpdf(d, x))

# -------- Inverse-Gaussian RNG (μ>0, λ>0), Michael–Schucany–Haas method --------
# Returns one IG(μ, λ) variate.
function _rand_IG(μ::Real, λ::Real)
  # Step 1: sample from χ ~ N(0,1)
  χ = randn()
  y = χ^2
  μ2 = μ*μ
  x = μ + (μ2*y)/(2λ) - (μ/(2λ)) * sqrt(4μ*λ*y + μ2*y^2)
  # Step 2: accept/reject transform
  u = rand()
  return (u ≤ μ/(μ + x)) ? x : (μ2/x)
end

# -------- NIG RNG using the mixture representation --------
# W ~ IG(δ/γ, δ^2), X | W ~ Normal(μ + βW, W)
function rand(d::NIG)
  μW, λW = d.δ/d.γ, d.δ^2
  W = _rand_IG(μW, λW)
  return d.μ + d.β*W + sqrt(W)*randn()
end

function rand(d::NIG, n::Integer)
  μW, λW = d.δ/d.γ, d.δ^2
  X = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    W = _rand_IG(μW, λW)
    X[i] = d.μ + d.β*W + sqrt(W)*randn()
  end
  return X
end