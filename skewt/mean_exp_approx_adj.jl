includet("../skewt.jl")
using DataFrames, Random, Statistics, StatsBase, Optim, Distributions

Random.seed!(0)

σ_range = (0.002,  0.3);
ν_range = (2.7,    8.0);
λ_range = (-0.1,   0.05);
hp_range = 0.9999;

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

  ds.emean = [mean_exp(d; l=log(1e-6), h=ds.h[i]) for (i, d) in enumerate(ds.d)];
  ds
end

skewt_mean_exp_adj(σ, ν, λ, hp, Q) = begin
  @assert σ_range[1] <= σ <= σ_range[2] "σ out of range"
  @assert ν_range[1] <= ν <= ν_range[2] "ν out of range"
  @assert λ_range[1] <= λ <= λ_range[2] "λ out of range"
  @assert hp == hp_range "hp out of range"

  lν = log(ν-2.5)
  m1 = exp(Q[1] + Q[2]lν + Q[3]λ + Q[4]lν^2 + Q[5]λ^2 + Q[6]*lν*λ)
  m2 = Q[7] + Q[8]lν + Q[9]λ
  Q[10] + m1*σ^2 + m2*σ
end;

skewt_mean_exp_adj_simple(σ, ν, λ, hp, Q) = begin
  @assert σ_range[1] <= σ <= σ_range[2] "σ out of range"
  @assert ν_range[1] <= ν <= ν_range[2] "ν out of range"
  @assert λ_range[1] <= λ <= λ_range[2] "λ out of range"
  @assert hp == hp_range "hp out of range"

  m1 = Q[1] + Q[2]log(ν) + Q[3]λ
  m1*σ^2/2
end;

fit_skewt_mean_exp_adj(ds, N, model) = begin
  residuals(Q) = log.(ds.emean) .- (
    ds.μ + model.(ds.σ, ds.ν, ds.λ, ds.hp, Ref(Q))
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

Q, re = fit_skewt_mean_exp_adj(ds, 10, skewt_mean_exp_adj)


# [
#   -0.648218866392912, -0.012407729414709195, 0.8718567522522687, -0.006870895633471439,
#   0.0054997790794226705, -0.25093014015464354, -0.00707318980863869, 0.0032400538810423035,
#   -0.03602921073340562, -5.373827757175762e-5
# ], 1.0014


Q, re = fit_skewt_mean_exp_adj(ds, 3, skewt_mean_exp_adj_simple)

# [0.4741825626694302, 0.0038481375624699853, 0.23232365015451695], 1.0027