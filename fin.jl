module Fin

import StatsBase, LinearAlgebra, SpecialFunctions, SkewTs
using Distributions, Lib, Dates

# Skew-t survival-tail exponent ν, shared by both tails.
estimate_tail(rs::AbstractVector{<:Real}) = begin
  @assert length(rs) >= 2000
  SkewTs.fit_mle(SkewTs.SkewT, rs).ν
end

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
# Q_nu2(F_nu1(x)), nu in [2.5,20]. Fit: SV/misc/fit_ig_qt.py.
# Tables for a, b, log(s), on the 7-point geometric nu grid, source index first.
inv_gamma_qt_prepare(nu1, nu2) = begin
  @assert 2.5 <= nu1 <= 20.0 && 2.5 <= nu2 <= 20.0
  C = reshape([
    -1.15746970815374484e+00, -9.88534685086160270e-01, -8.46274570450719743e-01,
    -7.21008281835988418e-01, -6.06871798896302828e-01, -5.02368235608950497e-01,
    -4.08246182087874332e-01, -1.42167219269422085e+00, -1.21969825729895209e+00,
    -1.04694593468300967e+00, -8.93172201284652134e-01, -7.52536671119529599e-01,
    -6.23797833755865616e-01, -5.08630702891837760e-01, -1.73545049141680541e+00,
    -1.49726012508533457e+00, -1.29079572250372010e+00, -1.10521569493111671e+00,
    -9.34631844279158575e-01, -7.78021589104070155e-01, -6.36699479011767044e-01,
    -2.09013602536123777e+00, -1.81454669349913167e+00, -1.57329900177956405e+00,
    -1.35473372261863534e+00, -1.15264621842905712e+00, -9.66039828403021938e-01,
    -7.96843729067406081e-01, -2.47154157474954239e+00, -2.16016618480892086e+00,
    -1.88589004973708141e+00, -1.63585028350848649e+00, -1.40312522499654335e+00,
    -1.18646362497722002e+00, -9.87965118136817977e-01, -2.86509443661998686e+00,
    -2.52178684851844626e+00, -2.21874255979417701e+00, -1.94116299888311361e+00,
    -1.68095698123347104e+00, -1.43630680684085110e+00, -1.21142552134548498e+00,
    -3.25469114922753233e+00, -2.88850062090411486e+00, -2.56235268277601635e+00,
    -2.26286949470502874e+00, -1.98010970176244472e+00, -1.71157680521239208e+00,
    -1.46040520578401711e+00, -3.20437575958875898e-01, -8.02258580469302346e-02,
    1.25005726082058372e-01, 2.97606444830014860e-01, 4.37297659946808803e-01,
    5.42902139598532285e-01, 6.14188555220279153e-01, -5.06705351212620947e-01,
    -2.35135580444734898e-01, -2.26366425964957406e-04, 2.00711721844918894e-01,
    3.67146771367104496e-01, 4.97316055879130969e-01, 5.90646380635925450e-01,
    -7.53026204855309356e-01, -4.44788384002008930e-01, -1.74637562807850144e-01,
    6.06005427527052248e-02, 2.60194931315690825e-01, 4.21704248011244787e-01,
    5.43661365006557151e-01, -1.05942143574038261e+00, -7.11695732185969709e-01,
    -4.03169603411659305e-01, -1.29828155185916450e-01, 1.07659412777394936e-01,
    3.06213223681693125e-01, 4.63342972441987411e-01, -1.41788047493970581e+00,
    -1.03132669186011205e+00, -6.84728307083109056e-01, -3.72695294903205854e-01,
    -9.53958311286169786e-02, 1.43651652865546264e-01, 3.40764590339938389e-01,
    -1.81539995974136747e+00, -1.39377957741147496e+00, -1.01281173738824259e+00,
    -6.64893395387122110e-01, -3.49125599205100501e-01, -6.91170713742396275e-02,
    1.70364337305186131e-01, -2.23459963155800612e+00, -1.78652171458636566e+00,
    -1.37731463385098696e+00, -9.99182951610010806e-01, -6.49332283361297335e-01,
    -3.30989906453713179e-01, -4.98501855201304370e-02, 7.82434821268139014e-01,
    7.23499314965055818e-01, 6.65356986352967916e-01, 5.99556679526598302e-01,
    5.18919857438272736e-01, 4.19692372392092228e-01, 3.00611053470778766e-01,
    9.10352753698274619e-01, 8.52870167917935906e-01, 7.95144589622659548e-01,
    7.29464681781465840e-01, 6.49427544064046236e-01, 5.51648364976908501e-01,
    4.36155059164830661e-01, 1.03205759346991321e+00, 9.77545669691285202e-01,
    9.22136192999847948e-01, 8.58807655038892892e-01, 7.81794386861699331e-01,
    6.87970473598335741e-01, 5.76351217304525432e-01, 1.13916520797452803e+00,
    1.08887022550426082e+00, 1.03756204341770997e+00, 9.78780155771212246e-01,
    9.07197257230000664e-01, 8.19780751913786632e-01, 7.15799671879556065e-01,
    1.22648807633648316e+00, 1.18146568482022030e+00, 1.13582879240055923e+00,
    1.08351210881175897e+00, 1.01943699571981461e+00, 9.40531753374830748e-01,
    8.45777075758320285e-01, 1.29262077399009589e+00, 1.25359039996969179e+00,
    1.21482674383612688e+00, 1.17042819627786954e+00, 1.11542055648692728e+00,
    1.04661328933326714e+00, 9.64339194584232118e-01, 1.33722941181765109e+00,
    1.30589741028614781e+00, 1.27470245386240144e+00, 1.23915261270919341e+00,
    1.19422156958695980e+00, 1.13671272170445392e+00, 1.06582417821867970e+00
  ], 7, 7, 3)
  u, v = log(nu1/2.5)*6/log(8.0), log(nu2/2.5)*6/log(8.0)
  i, j = min(floor(Int, u), 5)+1, min(floor(Int, v), 5)+1
  tx, ty = u-(i-1), v-(j-1)
  p = [(1-tx)*(1-ty)*C[i,j,k] + tx*(1-ty)*C[i+1,j,k] +
    (1-tx)*ty*C[i,j+1,k] + tx*ty*C[i+1,j+1,k] for k in 1:3]
  (p[1], p[2], exp(2*p[3]), nu1/nu2 - 1)
end

# Globally smooth square-root model. All table interpolation is done during preparation.
inv_gamma_qt_approx(qt, x) = begin
  @assert isfinite(x) && x > 0
  a, b, s2, k = qt
  k == 0 && return x
  z = log(x); v = z-b
  exp(z + k*(a + (v + sqrt(v*v + s2))/2))
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

# log softcap InvGamma moments ---------------------------------------------------------------------
# E[log X_softcap], X_softcap = m tanh(√(X/m))^2, X ~ InvGamma(ν/2, ν/2), m = Q(p_trunc)
log_inv_gamma_softcap_mean(nu, p_trunc=0.9999) = begin
  @assert 2.5 <= nu <= 20.0 "nu out of range"
  @assert p_trunc == 0.9999 "p_trunc must be 0.9999"
  snu = sqrt(nu)
  -2.71694328372849814e+00 +
    2.34351183649962991e+00 * exp(-2.89512421158110193e-04 * snu) +
    8.40053332072438930e-01 * exp(-2.59570714161257210e-01 * snu) +
    6.87812906468564855e-01 * exp(-9.25133100404034070e-01 * snu) +
    1.71173162658185483e+01 * exp(-3.21037696813277007e+00 * snu) +
    9.48856004301426001e+06 * exp(-1.66032840135198789e+01 * snu)
end

# Sd[log X_softcap], X_softcap = m tanh(√(X/m))^2, X ~ InvGamma(ν/2, ν/2), m = Q(p_trunc)
log_inv_gamma_softcap_sd(nu, p_trunc=0.9999) = begin
  @assert 2.5 <= nu <= 20.0 "nu out of range"
  @assert p_trunc == 0.9999 "p_trunc must be 0.9999"
  snu = sqrt(nu)
  -1.19317874768620023e-02 +
    3.44390705729325619e-01 * exp(-1.19284494996698159e-01 * snu) +
    7.77586316456215454e-01 * exp(-6.31986246613399194e-01 * snu) +
    6.18380799606367604e-01 * exp(-6.32228715710016709e-01 * snu) +
    2.87968605907560216e+00 * exp(-1.56128017515236039e+00 * snu) +
    2.77135955981734483e+01 * exp(-3.88923289089931856e+00 * snu)
end


# Log InvGamma mix moments -------------------------------------------------------------------------
# LEGACY: uncapped mixture with an unnormalized gate, unused by current SVJ/MSVJ normalization.
# Retained with its coefficients and self-test for future use.
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

end
