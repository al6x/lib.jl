module Fin

import StatsBase, LinearAlgebra, SpecialFunctions, Roots
using Distributions, Lib, Dates
using StatsFuns: normcdf

# Includes from and to
calendar_to_trading(from::Date, to::Date) = begin
  n = Dates.value(to - from) + 1
  ct_to_tt = Vector{Union{Int, Missing}}(undef, n)
  c = 0
  for (i, d) in enumerate(from:Day(1):to)
    if dayofweek(d) in 1:5
      c += 1; ct_to_tt[i] = c
    else
      ct_to_tt[i] = missing
    end
  end
  tt_to_ct = [i for (i, v) in enumerate(ct_to_tt) if v !== missing]
  ct_to_tt, tt_to_ct
end

calendar_to_trading(from::Date, length::Int) = begin
  to = from + Day(length - 1)
  calendar_to_trading(from, to)
end

smooth_vol(vs::AbstractVector{T}) where {T<:Union{Missing,Real}} = begin
  vsr_min = median(vs |> skipmissing |> collect) * 0.1
  vs_smooth = ewa(vs, 1/10)
  max.(vs_smooth, vsr_min)
  # max.(vs, vs_smooth .* 0.2, vsr_min)
  # max.(vs_smooth .* 0.2, vsr_min)
end

vol(rs::AbstractVector{T}) where {T<:Union{Missing,Real}} = begin
  center = mean(rs |> skipmissing |> collect); centered = rs .- center
  abs.(centered) |> smooth_vol
end

# max(|r_{t}|, ..., |r_{t+n-1}|)
vol_max_n(rs::Vector{Float64}; n = 3) = begin
  T = length(rs)
  center = mean(rs); centered = rs .- center
  vs = map(1:T) do t
    maximum(abs.(@view centered[t:min(t + n - 1, T)]))
  end |> smooth_vol
  vs
end

calc_diff(op, x::AbstractVector{T}) where {T<:Union{Missing,Real}} = begin
  y, prev = Vector{Union{Missing,Float64}}(undef, length(x)), missing
  for i in 1:length(x)
    y[i] = if ismissing(x[i]) missing else
      v = ismissing(prev) ? missing : op(prev, x[i])
      prev = x[i]
      v
    end
  end
  y
end

ewa(x::AbstractVector{T}, α::Real; fill_missing=false) where {T<:Union{Missing,Real}} = begin
  @assert 0 < α < 1 "α must be in (0,1)"
  y, n, i = similar(x), length(x), 1
  @inbounds while i <= n && x[i] === missing
    y[i] = missing; i += 1
  end
  if i <= n
    y[i] = s = x[i]; i += 1
    @inbounds for j in i:n
      if x[j] === missing
        y[j] = fill_missing ? s : missing
      else
        y[j] = s = α * x[j] + (1 - α) * s
      end
    end
  end
  y
end

roll_slow(op, vs::Vector{Union{Missing, Float64}}; window, min) = begin
  window == 1 && return vs
  r = Vector{Union{Missing, Float64}}(missing, length(vs))
  for t in 1:length(vs)
    vs[t] === missing && continue
    present = skipmissing(vs[max((t - window + 1), 1):t])
    count(_ -> true, present) ≥ min || continue
    r[t] = op(present)
  end
  r
end

rperiod(rs::Vector{<:Union{Missing,Float64}}; period::Int, overlap=false) = begin
  @assert period > 0
  n = length(rs)
  rsp, psp = Vector{Union{Missing,Float64}}(undef, n), Vector{Union{Missing,Int}}(undef, n)
  for t in 1:n
    t2 = t + period
    rsp[t], psp[t] = if ismissing(rs[t]) || t2 > n
      missing, missing
    else
      count, sum = 0, 0.0; for i in t+1:t2-1
        r = rs[i]; !ismissing(r) && (sum += r; count += 1)
      end

      rlast, tlast = rs[t2], t2
      # If last return at t2 is non trading day using next trading day,
      # but with the gap no more than 2 days.
      max_last_gap = 2
      if ismissing(rlast)
        while ismissing(rlast) && tlast < min(n, t2 + max_last_gap)
          tlast += 1; rlast = rs[tlast]
        end
      end
      # Return missing if insufficient data or last return is missing
      ismissing(rlast) || (count + 1 < period/2) ?
        (missing, missing) :
        (sum + rlast, tlast - t)
    end
  end

  if overlap == false
    i = 1; while i <= n
      ismissing(rsp[i]) && (i += 1; continue)
      del = i+1:min(i+period-1, n)
      rsp[del] .= missing
      psp[del] .= missing
      i += period
    end
  end

  rsp, psp
end

garman_klass_rv(o::Float64, h::Float64, l::Float64, c::Float64) = begin
  rv = 0.5 * log(h / l)^2 - (2 * log(2) - 1) * log(c / o)^2
  @assert rv ≥ 0 "rv negative: $o, $h, $l, $c"
  rv
end

garman_klass_rv(input::NamedTuple) = begin
  (; o, h, l, c) = input; n = length(c)
  [ismissing(o[t]) ? missing : garman_klass_rv(o[t], h[t], l[t], c[t]) for t in 1:n]
end

# Garman-Klass with Overnight
overnight_rs_var(input::NamedTuple;
  adj_gap=0.15,         # vol over weekends don't follow sqrt(T)
  overnight_weight=0.01 # optimal weight found from garch fitting
) = begin
  (; o, h, l, c) = input
  n = length(c)
  @assert length(o) == length(h) == length(l) == n

  rv = Vector{Union{Missing,Float64}}(undef, n)
  fill!(rv, missing)

  last = nothing  # last index with full OHLC present

  @inbounds for t in 1:n
    if ismissing(o[t]) || ismissing(h[t]) || ismissing(l[t]) || ismissing(c[t])
      continue
    end

    # Garman–Klass (day t)
    r_hl = log(h[t] / l[t])
    r_co = log(c[t] / o[t])
    gk   = 0.5 * r_hl^2 - (2log(2) - 1) * r_co^2

    if last === nothing || ismissing(c[last])
      rv[t] = missing
    else
      gap = t - last  # calendar index gap between observed days
      u_full = log(o[t] / c[last])
      u = u_full / sqrt(gap^adj_gap)
      rv[t] = overnight_weight* u^2 + gk
    end

    last = t
  end

  rv
end

@inline tdist_to_normal_approx_prepare(ν::Real) = begin
  lnu = log(ν)
  a = 0.10418921580343157 + 0.8192336024018511 * (ν - 1.888374523210565)^(-0.4450288202386115)
  b = 0.012844378593201458 + 0.9568788088064243 * lnu + 0.01765372193044918 * lnu * lnu
  a, b
end

@inline tdist_to_normal_approx(a::Real, b::Real, x::Real) = begin
  # Details in fit_t_to_normal_joint.py
  b * asinh(a * x)
end

@inline tdist_to_normal_approx(ν::Real, x::Real) = begin
  # Details in fit_t_to_normal_joint.py
  # Var = ν/(ν-2)
  tdist_to_normal_approx(tdist_to_normal_approx_prepare(ν)..., x)
end


long_mem(H::Real) = begin
  Hl, Hh = 0.005, 0.25

  periods = [0.73, 8.51, 251.76]
  R       = [-0.490416, 0.329208, -0.137544, 0.147115, -0.34957, 0.093252]

  eps = 1e-12
  @assert Hl - eps ≤ H ≤ Hh + eps

  s  = [1.0, 2.0, 3.0]
  s2 = s .* s

  h  = 2.0*(H - Hl)/(Hh - Hl) - 1.0
  h2 = h*h

  wr = exp.(
    R[1] .* s .+ R[2] .* s2 .+ R[3] .* s .* h .+ R[4] .* s2 .* h .+ R[5] .* s .* h2 .+
    R[6] .* s2 .* h2
  )

  ϕ = exp.(-1.0 ./ periods)
  lag0 = sum(wr .* sqrt.(1.0 .- ϕ.^2))

  periods, wr ./ lag0
end


ewas_cov(ϕ) = begin
  λ = sqrt.(1.0 .- ϕ.^2)
  (λ * λ') ./ (1.0 .- (ϕ * ϕ'))
end

ewas_var(ϕ, w) = LinearAlgebra.dot(w, ewas_cov(ϕ) * w)


# TDist fit ----------------------------------------------------------------------------------------
fit_mle_tdist(::Type{TDist}, x::AbstractVector{<:Real}) = begin
  min_ν = 1.1

  @inline decode(θ) = θ[1], exp(θ[2]), exp(θ[3]) + min_ν

  nll(θ) = begin
    μ, σ, ν = decode(θ)
    if σ < 1e-12 || ν <= min_ν || ν > 100 return Inf end
    d = LocationScale(μ, σ, TDist(ν))
    -sum(logpdf.(Ref(d), x))
  end

  inits = [
    [mean(x), log(std(x) + 1e-6), log(5 - min_ν)],
    [median(x), log(median(abs.(x .- median(x))) + 1e-6), log(15 - min_ν)]
  ]

  for init in inits
    res = Optim.optimize(nll, init, Optim.BFGS(); autodiff = :forward)
    !Optim.converged(res) && continue
    θ = Optim.minimizer(res)
    μ, σ, ν = decode(θ)
    return LocationScale(μ, σ, TDist(ν))
  end

  error("Can't estimate TDist")
end


# InvGamma trunc moments ---------------------------------------------------------------------------
inv_gamma_trunc_mean(ν) = begin
  # E[X]    X ~ InvGamma(ν/2, ν/2), X ≤ Q(0.999)
  @assert 2.7 - 1e-12 <= ν <= 100.0 + 1e-12
  lν = log(ν)
  1.07314375 * exp(-0.0129366  * lν) +
  5.00634417 * exp(-1.39532474 * lν) +
  74.2459949 * exp(-4.43238737 * lν)
end
inv_gamma_trunc_sd(ν) = begin
  # Sd[X]    X ~ InvGamma(ν/2, ν/2), X ≤ Q(0.999)
  @assert 2.7 - 1e-12 <= ν <= 100.0 + 1e-12
  lν = log(ν)
  1.72639242 * exp(-0.53830701 * lν) +
  28.77606354 * exp(-2.18319098 * lν) +
  1452.13398057 * exp(-5.64297095 * lν)
end
inv_gamma_trunc_mad(ν) = begin
  # Meanad[X]    X ~ InvGamma(ν/2, ν/2), X ≤ Q(0.999)
  @assert 2.7 - 1e-12 <= ν <= 100.0 + 1e-12
  lν = log(ν)
  1.25054177 * exp(-0.51923066 * lν) +
  9.04120428 * exp(-1.94214674 * lν) +
  157.51493911 * exp(-4.97340282 * lν)
end


# InvGamma to InvGamma Quantile Transform ----------------------------------------------------------
inv_gamma_qt_interp(grid, T, nu1, nu2) = begin
  n = length(grid)
  (grid[1] ≤ nu1 ≤ grid[n] && grid[1] ≤ nu2 ≤ grid[n]) || error("nu outside grid")

  i = 1
  for k in 1:n-1
    if grid[k] ≤ nu1 ≤ grid[k+1]
      i = k; break
    end
  end

  j = 1
  for k in 1:n-1
    if grid[k] ≤ nu2 ≤ grid[k+1]
      j = k; break
    end
  end

  tx = (nu1 - grid[i]) / (grid[i+1] - grid[i])
  ty = (nu2 - grid[j]) / (grid[j+1] - grid[j])

  T[i,   j]   * (1 - tx) * (1 - ty) +
  T[i+1, j]   * tx       * (1 - ty) +
  T[i,   j+1] * (1 - tx) * ty +
  T[i+1, j+1] * tx       * ty
end

# Quantile transform
# X2 = Qnu2 Fnu1 X1 where X1 ~ InvGamma(nu1/2, nu1/2) and X2 ~ InvGamma(nu2/2, nu2/2)
inv_gamma_qt_abc(nu1, nu2) = begin
  g = [2.7, 3.8, 5.3, 7.3, 10.3, 14.3, 20.0]

  A = [
     0.000000  -0.056904  -0.086932  -0.101222  -0.106516  -0.105407  -0.100464;
     0.061680   0.000000  -0.035123  -0.054497  -0.065188  -0.068963  -0.068637;
     0.097964   0.036537   0.000000  -0.021670  -0.035348  -0.042137  -0.044852;
     0.117364   0.058425   0.022354   0.000000  -0.015115  -0.023611  -0.028192;
     0.129236   0.073306   0.038292   0.015876   0.000000  -0.009583  -0.015442;
     0.137108   0.083289   0.049042   0.026628   0.010283   0.000000  -0.006697;
     0.144403   0.091658   0.057665   0.035062   0.018243   0.007367   0.000000
  ]

  B = [
     1.000000   0.821218   0.685292   0.579630   0.486373   0.412651   0.349415;
     1.217305   1.000000   0.834627   0.706065   0.592619   0.502946   0.426020;
     1.462643   1.199721   1.000000   0.845126   0.708803   0.601266   0.509150;
     1.743766   1.425994   1.185585   1.000000   0.837342   0.709493   0.600277;
     2.110213   1.718204   1.423346   1.197126   1.000000   0.845825   0.714630;
     2.534646   2.054511   1.695488   1.421727   1.184558   1.000000   0.843583;
     3.058036   2.467656   2.028697   1.695956   1.409310   1.187355   1.000000
  ]

  C = [
     0.000000  -0.011242  -0.017130  -0.019629  -0.020155  -0.019445  -0.018074;
     0.025210   0.000000  -0.014252  -0.021519  -0.024858  -0.025435  -0.024531;
     0.079011   0.029318   0.000000  -0.016265  -0.025365  -0.029028  -0.029768;
     0.171861   0.084050   0.030885   0.000000  -0.018855  -0.028108  -0.032192;
     0.335520   0.184661   0.091615   0.035867   0.000000  -0.019375  -0.029860;
     0.577175   0.337000   0.187028   0.095389   0.034566   0.000000  -0.020401;
     0.945071   0.572698   0.338019   0.192557   0.093901   0.035960   0.000000
  ]

  [
    inv_gamma_qt_interp(g, A, nu1, nu2),
    inv_gamma_qt_interp(g, B, nu1, nu2),
    inv_gamma_qt_interp(g, C, nu1, nu2)
  ]
end

# Quantile transform
# X2 = Qnu2 Fnu1 X1 where X1 ~ InvGamma(nu1/2, nu1/2) and X2 ~ InvGamma(nu2/2, nu2/2)
inv_gamma_qt_approx(abc, x) = begin
  lx = log(x)
  exp(abc[1] + abc[2]*lx + abc[3]*lx*lx)
end

# InvGamma moments ---------------------------------------------------------------------------------
inv_gamma_mean(ν) = ν / (ν - 2.0)

# Meanad[X]    X ~ InvGamma(ν/2, ν/2)
inv_gamma_meanad_approx(nu) = begin
  @assert 2.7 - 1e-12 <= nu <= 12.0 + 1e-12  "nu out of range"
  lnu = log(nu)
  return (
      1.33939142e+00 * exp(-0.53295322 * lnu) +
      1.44156559e+01 * exp(-2.17926837 * lnu) +
      1.40225631e+03 * exp(-6.73443418 * lnu)
  )
end


# log InvGamma moments -----------------------------------------------------------------------------
# E[log X]    X ~ InvGamma(ν/2, ν/2)
log_inv_gamma_mean(nu) = log(nu / 2) - SpecialFunctions.digamma(nu / 2)

# Sd[log X]    X ~ InvGamma(ν/2, ν/2)
log_inv_gamma_sd(nu) = sqrt(SpecialFunctions.trigamma(nu / 2))

# E[log X]    X ~ InvGamma(ν/2, ν/2)
log_inv_gamma_mean_approx(nu) = begin
  @assert 2.7 - 1e-12 <= nu <= 12.0 + 1e-12  "nu out of range"
  0.00164321 * nu^-0.96571545 +
  0.96944422 * nu^-0.99792029 +
  0.30963144 * nu^-1.82446200
end

# Sd[log X]    X ~ InvGamma(ν/2, ν/2)
log_inv_gamma_sd_approx(nu) = begin
  @assert 2.7 - 1e-12 <= nu <= 12.0 + 1e-12  "nu out of range"
  0.01269530 * nu^-0.48710743 +
  1.39887374 * nu^-0.50002316 +
  0.81386454 * nu^-1.57763595
end

# Meanad[log X]    X ~ InvGamma(ν/2, ν/2)
log_inv_gamma_meanad_approx(nu) = begin
  @assert 2.7 - 1e-12 <= nu <= 12.0 + 1e-12  "nu out of range"
  0.01041277 * nu^-0.48716736 +
  1.12230849 * nu^-0.50124504 +
  0.53765492 * nu^-1.58717139
end

# log trunc InvGamma moments -----------------------------------------------------------------------
# E[log X_trunc], X_trunc = m tanh(√(X/m))^2, X ~ InvGamma(ν/2, ν/2), m = Q(p_trunc)
log_inv_gamma_trunc_mean(nu, p_trunc=0.9998) = begin
  @assert 2.7 <= nu <= 100.0 "nu out of range"
  @assert p_trunc == 0.9998 "p_trunc must be 0.9998"
  snu = sqrt(nu)
  -8.385624358479423e-01 +
    4.9783461e-01   * exp(-1.174725e-02 * snu)   +
    5.7340695e-01   * exp(-2.2462881e-01 * snu)  +
    6.9264932e-01   * exp(-6.3617036e-01 * snu)  +
    7.7644988e-01   * exp(-1.62791129e+00 * snu) +
    1.941587793e+01 * exp(-3.44820119e+00 * snu)
end

# Sd[log X_trunc], X_trunc = m tanh(√(X/m))^2, X ~ InvGamma(ν/2, ν/2), m = Q(p_trunc)
log_inv_gamma_trunc_sd(nu, p_trunc=0.9998) = begin
  @assert 2.7 <= nu <= 100.0 "nu out of range"
  @assert p_trunc == 0.9998 "p_trunc must be 0.9998"
  snu = sqrt(nu)
  3.0114684157074526e-02 +
    2.5054106e-01   * exp(-1.4004671e-01 * snu)  +
    7.3624265e-01   * exp(-4.6981411e-01 * snu)  +
    1.67561183e+00  * exp(-1.02679808e+00 * snu) +
    2.97687948e+00  * exp(-1.96778207e+00 * snu) +
    3.123797783e+01 * exp(-4.16020499e+00 * snu)
end


# Log InvGamma mix moments -------------------------------------------------------------------------
#
#   X[t], Y[t] ~ InvGamma(nu/2, nu/2)
#   cond[t] = sigmoid(sharpness (X[t] - Q(X, threshold_p)))
#   c[t] = cond[t] ct + (1-cond) cc
#   Z[t] = c[t] X[t] + (1-c[t]) Y[t]
#
#   threshold_p, sharpness = 0.9, 2.0
#
#   E[log Z]  = E[log InvGamma(nu/2, nu/2)]*mean_adj(nu, cc, ctm)
#   Sd[log Z] = Sd[log InvGamma(nu/2, nu/2)]*sd_adj(nu, cc, ctm)
#
log_inv_gamma_mix_model(q, cc, ct) = begin
  @assert 0.05 - 1e-12 <= cc <= 0.95 + 1e-12 "cc outside grid"
  @assert cc - 1e-12 <= ct <= 1.0 + 1e-12 "ct outside grid"

  x = cc; y = ct

  exp(q[1] + q[2]*x + q[3]*y + q[4]*x^2 + q[5]*y^2 + q[6]*x*y + q[7]*x^3 + q[8]*y^3)
end

log_inv_gamma_mix_interp(nu, nu_grid, q_mat) = begin
  @assert nu_grid[1] - 1e-12 <= nu <= nu_grid[end] + 1e-12 "nu outside grid"

  lnu = log(nu); lnu_grid = log.(nu_grid)

  i = 1
  for k in 1:(length(lnu_grid) - 1)
    if lnu_grid[k] <= lnu && lnu <= lnu_grid[k + 1]
      i = k; break
    end
  end

  t = (lnu - lnu_grid[i]) / (lnu_grid[i + 1] - lnu_grid[i])
  q = Vector{Float64}(undef, size(q_mat, 2))
  for k in 1:size(q_mat, 2)
    q[k] = (1 - t) * q_mat[i, k] + t * q_mat[i + 1, k]
  end
  q
end

log_inv_gamma_mix_mean_adj(nu, cc, ct, threshold_p, sharpness) = begin
  @assert threshold_p == 0.9
  @assert sharpness == 2.0

  nu_grid = [
    2.700000, 3.462050, 4.439182, 5.692100, 7.298642, 9.358617, 12.000000
  ]

  q_mean_mat = [
     0.126969   0.358653   0.847231  -0.256419  -0.884172  -0.041355  -0.395688   0.375025
     0.103605   0.298864   0.958771  -0.186279  -0.957255  -0.058128  -0.456125   0.401428
     0.084307   0.225986   1.077475  -0.121996  -1.025214  -0.074319  -0.504819   0.423637
     0.074707   0.154409   1.179803  -0.076570  -1.066759  -0.089227  -0.534463   0.432563
     0.068209   0.086365   1.263763  -0.046099  -1.087767  -0.101212  -0.550344   0.431369
     0.060024   0.029656   1.340764  -0.032193  -1.104836  -0.113602  -0.554787   0.427929
     0.050533  -0.012133   1.414419  -0.033551  -1.124803  -0.127090  -0.544850   0.425906
  ]

  q = log_inv_gamma_mix_interp(nu, nu_grid, q_mean_mat)
  log_inv_gamma_mix_model(q, cc, ct)
end

log_inv_gamma_mix_sd_adj(nu, cc, ct, threshold_p, sharpness) = begin
  @assert threshold_p == 0.9
  @assert sharpness == 2.0

  nu_grid = [
    2.700000, 3.462050, 4.439182, 5.692100, 7.298642, 9.358617, 12.000000
  ]

  q_sd_mat = [
    -0.037508  -0.640793   0.224539   0.321627  -0.030201   0.124021   0.027534  -0.031554
    -0.036327  -0.674132   0.133315   0.231287   0.102366   0.162786   0.132787  -0.091505
    -0.032101  -0.701174   0.034247   0.159634   0.227634   0.204797   0.221174  -0.144596
    -0.023479  -0.726893  -0.062621   0.119972   0.323330   0.246131   0.281676  -0.179048
    -0.015185  -0.748924  -0.149759   0.122263   0.384572   0.283074   0.306117  -0.193534
    -0.006780  -0.776778  -0.227465   0.163275   0.415072   0.321329   0.298466  -0.190580
    -0.000198  -0.809870  -0.290755   0.237758   0.419920   0.363233   0.259931  -0.175085
  ]

  q = log_inv_gamma_mix_interp(nu, nu_grid, q_sd_mat)
  log_inv_gamma_mix_model(q, cc, ct)
end

log_inv_gamma_mix_moments_test() = begin
  tests = [
    (2.7, 0.1, 0.8, 0.414033, 1.037879, 0.650492, 1.096483),
    (6.0, 0.6, 0.9, 0.175828, 0.628438, 0.289291, 0.541970),
  ]

  for (nu, cc, ct, mean_x, sd_x, mean_mix, sd_mix) in tests
    mean_fit = mean_x * log_inv_gamma_mix_mean_adj(nu, cc, ct, 0.9, 2.0)
    sd_fit   = sd_x   * log_inv_gamma_mix_sd_adj(nu, cc, ct, 0.9, 2.0)

    @assert abs(mean_fit - mean_mix) / abs(mean_mix) < 0.01 "mean test failed"
    @assert abs(sd_fit   - sd_mix)   / abs(sd_mix) < 0.01 "sd test failed"
  end
end

# E[√InvGamma(ν/2, ν/2)]
sqrt_inv_gamma_mean(ν::Real) = begin
  ν > 1 || error("ν must be > 1")
  α = ν / 2
  return sqrt(ν / 2) * exp(SpecialFunctions.loggamma(α - 0.5) - SpecialFunctions.loggamma(α))
end

tdist_meanad(ν) =
  (2*sqrt(ν)*SpecialFunctions.gamma((ν+1)/2)) /
  (sqrt(pi)*(ν-1)*SpecialFunctions.gamma(ν/2))


# Black-Scholes price and implied volatility -------------------------------------------------------
# T in days, r1y in float percentages like 0.05
bs_price(; K, T, r1y, q=0.0, σ, right) = begin
  τ, S = T/365, 1.0
  sτ = σ*sqrt(τ)
  d1 = (log(S/K) + (r1y - q + σ^2/2)*τ)/sτ
  d2 = d1 - sτ
  right == :call ?
    S*exp(-q*τ)*normcdf(d1) - K*exp(-r1y*τ)*normcdf(d2) :
    K*exp(-r1y*τ)*normcdf(-d2) - S*exp(-q*τ)*normcdf(-d1)
end

implied_vol(; price, K, T, r1y, right, q=0.0, atol=1e-9) = begin
  τ, S = T/365, 1.0
  lo, hi = right == :call ?
    (max(S*exp(-q*τ) - K*exp(-r1y*τ), 0.0), S*exp(-q*τ)) :
    (max(K*exp(-r1y*τ) - S*exp(-q*τ), 0.0), K*exp(-r1y*τ))
  # @assert lo <= price <= hi "no arbitrage violated: $((; price, K, T, r1y, q, right))"
  lo <= price <= hi || return nothing

  f(σ) = bs_price(; K, T, r1y, q, σ, right) - price
  Roots.find_zero(f, (1e-6, 5.0), Roots.Brent(); atol)
end

implied_vol_df(; price, K, T, df, right, q=0.0, atol=1e-9) = begin
  r1y = -log(df) / (T/365)
  implied_vol(; price, K, T, r1y, right, q, atol)
end


# implied_vol(; price, K, T, r1y, right, q=0.0, rtol=0.001, atol=1e-6) = begin
#   @show price, K, T, r1y, q, right
#   @assert 1e-8 < price <= 1.0 "price out of range"
#   Roots.find_zero(σ ->
#     bs_price(; K, T, r1y, q, σ, right) - price, (1e-8, 1.0),
#   Roots.Brent(); rtol, atol)
# end

end
