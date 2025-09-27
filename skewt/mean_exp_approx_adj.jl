includet("../skewt.jl")
using DataFrames, Random, Statistics, StatsBase, Optim, Distributions

Random.seed!(0)

σ_range = (0.001,  0.5)
ν_range = (2.7,    4.0)
λ_range = (-0.1,   0.1)
hp_range = 0.9999

Q_skewt_mean_exp_adj = [
  -2.5293139635108637, -0.6573930578555529, 1.9699792080439948, -0.12806815272246, -2.9795712225102453,
  -0.2067505291778299, -1.0755334965951282, -0.37237840421331136, 2.045224620758677, -0.005350161835873002,
  0.0007996593124777508, -0.022282241134172886, 0.1247552899561173, 0.15371297791271646, -0.7035997759098427
] # re = 1.0008

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
  a, b = q085, q099
  la, lb = log(a), log(b)

  m1 = exp(Q[1] + Q[2]lν + Q[3]λ + Q[4]lν^2 + Q[5]λ^2 + Q[6]*lν*λ + Q[7]*la + Q[8]*la^2 + Q[9]*lb)
  m2 = Q[10] + Q[11]lν + Q[12]λ + Q[13]*a + Q[14]*b
  (σ*m1)^2 + σ*m2
end;

fit_skewt_mean_exp_adj(ds) = begin
  residuals(Q) = log.(ds.emean) .- (
    ds.μ + skewt_mean_exp_adj.(ds.σ, ds.ν, ds.λ, ds.hp, Ref(Q), ds.q085, ds.q099)
  )
  goal(Q)      = 100000*mean(residuals(Q).^2)
  penalty(Q)   = (1/1000000)mean(Q[2:end].^2)
  loss(Q)      = goal(Q) #+ penalty(Q)

  inits = [randn(15) for _ in 1:10]
  rs = [optimize(loss, init, LBFGS(); autodiff=:forward) for init in inits]
  rs = filter(Optim.converged, rs)
  isempty(rs) && error("can't fit")

  mins = map(Optim.minimum, rs)
  Q = Optim.minimizer(rs[argmin(mins)])

  # re = abs.(residuals(Q))
  # re_i = argmax(re)
  # println((; ds[re_i, :]..., model=emean_model(ds.d[re_i], ds.hp[re_i], Q, ds.q085[re_i], ds.q099[re_i]), re=re[re_i]))

  re = round(exp(maximum(abs.(residuals(Q)))); digits=4)
  println("Fit re=$re, goal=$(round(goal(Q); digits=4)), penalty=$(round(penalty(Q); digits=4))")

  return Q, re
end;

Q, re = fit_skewt_mean_exp_adj(ds)

ds = let
  e = 1e-6
  μ_grid  = [0.0, 0.5]; # Could be any value, not restricted
  σ_grid  = collect(exp.(range(log(σ_range[1]+e), log(σ_range[2]-e); length=20)));
  ν_grid  = collect(exp.(range(log(ν_range[1]+e), log(ν_range[2]-e); length=10)));
  λ_grid  = collect(range(λ_range...; length=10));
  hp_grid = [hp_range]

  ds = DataFrame([(; μ, σ, ν, λ, hp) for μ in μ_grid, σ in σ_grid, ν in ν_grid, λ in λ_grid, hp in hp_grid])

  ds.d = SkewT.(ds.μ, ds.σ, ds.ν, ds.λ)
  ds.h = quantile.(ds.d, ds.hp)

  d0 = SkewT.(0.0, ds.σ, ds.ν, ds.λ)
  ds.q085 = quantile.(d0, 0.85)
  ds.q099 = quantile.(d0, 0.99)

  ds.emean = [mean_exp(d; l=log(1e-6), h=ds.h[i]) for (i, d) in enumerate(ds.d)];
  ds
end

Q, re = fit_lmean_model(ds)