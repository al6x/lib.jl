include("../skewt.jl")
using DataFrames, Random, Statistics, StatsBase, Optim, Distributions, Plots

Random.seed!(0)

σ_range = (0.002,  0.3);
ν_range = (2.5,    20.0);
hp_range = 0.9999;

ds = let
  e = 1e-6
  μ_grid  = [0.0, 0.5]; # Could be any value, not restricted
  σ_grid  = collect(exp.(range(log(σ_range[1]+e), log(σ_range[2]-e); length=20)));
  ν_grid  = collect(exp.(range(log(ν_range[1]+e), log(ν_range[2]-e); length=10)));
  hp_grid = [hp_range]

  ds = DataFrame([(; μ, σ, ν, hp) for μ in μ_grid, σ in σ_grid, ν in ν_grid, hp in hp_grid])

  ds.d = SkewT.(ds.μ, ds.σ, ds.ν, 0.0)
  ds.h = quantile.(ds.d, ds.hp)

  ds.emean = [mean_exp(d; l=log(1e-6), h=ds.h[i]) for (i, d) in enumerate(ds.d)];
  ds
end;

t_mean_exp_adj(Q, σ, ν, hp) = begin
  @assert σ_range[1] <= σ <= σ_range[2] "σ out of range"
  @assert ν_range[1] <= ν <= ν_range[2] "ν out of range"
  @assert hp == hp_range "hp out of range"

  lν = log(ν-2.4)
  m1 = exp(Q[1] + Q[2]lν + Q[3]lν^2)
  m2 = Q[4] + Q[5]lν + Q[6]lν^2
  Q[7] + m1*σ^2 + m2*σ
end;

fit_t_mean_exp_adj(ds, N, model) = begin
  residuals(Q) = log.(ds.emean) .- (
    ds.μ + model.(Ref(Q), ds.σ, ds.ν, ds.hp)
  )
  goal(Q)      = 100000*mean(residuals(Q).^2)
  penalty(Q)   = (1/1000000)mean(Q[2:end].^2)
  loss(Q)      = goal(Q) #+ penalty(Q)

  inits = [randn(N) for _ in 1:10]
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

Q, re = fit_t_mean_exp_adj(ds, 7, t_mean_exp_adj)

# [
#   -0.6583260051026948, -0.006687011379218902, -0.0021854282879885264,
#   -0.006712209526179466, 0.0035009249251920452, -0.0005517594722773062,
#   -5.1144033055916596e-5
# ], 1.0011

# Final --------------------------------------------------------------------------------------------
@inline t_mean_exp_adj(σ::Real, ν::Real) = begin
  # Truncated at 0.9999 quantile
  # @assert 0.002 <= σ <= 0.3 "σ out of range" # ignored, it may go out during fitting
  @assert 2.5 <= ν <= 20.0 "ν out of range: $ν"
  lν = log(ν-2.4)
  m1 = exp(-0.6583260051026948 -0.006687011379218902*lν -0.0021854282879885264*lν^2)
  m2 = -0.006712209526179466 +0.0035009249251920452*lν -0.0005517594722773062*lν^2
  -5.1144033055916596e-5 + m1*σ^2 + m2*σ
end;
@assert t_mean_exp_adj(0.15, 3.0) ≈ t_mean_exp_adj(Q, 0.15, 3.0, 0.9999)