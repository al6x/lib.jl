includet("../skewt.jl")
using DataFrames, Random, Statistics, StatsBase, Optim, Interpolations, Distributions

Random.seed!(0)

ds = let
  μ_grid  = collect(range(log(1.0001), log(1.8); length=5));
  σ_grid  = collect(exp.(range(log(0.001), log(1); length=5)));
  ν_grid  = collect(exp.(range(log(2.7), log(6.0); length=3)));
  λ_grid  = collect((range(-0.2, 0.1; length=5)));
  hp_grid = collect(exp.(range(log(0.99), log(1-2e-4); length=3)));
  hp_grid = [0.9997]

  ds = DataFrame([(; μ, σ, ν, λ, hp) for μ in μ_grid, σ in σ_grid, ν in ν_grid, λ in λ_grid, hp in hp_grid])

  distrs = SkewT.(ds.μ, ds.σ, ds.ν, ds.λ)
  ds.h = quantile.(distrs, ds.hp)
  ds.median = quantile.(distrs, 0.5)

  distrs_std = SkewT.(0.0, 1.0, ds.ν, ds.λ)
  ds.h_std = quantile.(distrs_std, ds.hp)
  ds.median_std = quantile.(distrs_std, 0.5)

  ds.mean_exp = [mean_exp(d; l=log(1e-6), h=ds.h[i]) for (i, d) in enumerate(distrs)];
  ds
end

fit_mean_log_model(ds; inits=[randn(Qn) for _ in 1:10]) = begin
  residuals(Q) = ds.μ .- mean_log_model.(ds.mean_exp, ds.σ, ds.ν, ds.λ, ds.hp, Ref(Q))
  goal(Q)      = 100*mean(residuals(Q).^2)
  penalty(Q)   = (1/10)mean(Q[7:end].^2)
  loss(Q)      = goal(Q) #+ penalty(Q)

  # inits = [randn(Qn) for _ in 1:100]
  rs = [optimize(loss, init, LBFGS(); autodiff=:forward) for init in inits]
  rs = filter(Optim.converged, rs)
  isempty(rs) && error("can't fit")

  mins = map(Optim.minimum, rs)
  Q = Optim.minimizer(rs[argmin(mins)])

  rel_errors = exp.(abs.(residuals(Q)))
  max_error_i = argmax(rel_errors)


  println(ds[max_error_i, :])
  max_error_row = ds[max_error_i, :]
  max_error = mean_log_model(max_error_row.mean_exp, max_error_row.σ, max_error_row.ν, max_error_row.λ, max_error_row.hp, Q)
  println((exp(abs(ds[max_error_i, :μ]-max_error)), round(ds[max_error_i, :μ]; digits=5), round(max_error; digits=5)))



  re = round(rel_errors[max_error_i]; digits=4)
  println("Fit re=$re, goal=$(round(goal(Q); digits=4)), penalty=$(round(penalty(Q); digits=4))")

  return Q, re
end

# Qn = 15;
mean_log_model(mean_exp, σ, ν, λ, hp, Q) = begin
  lν, lhp = log(ν-2.5), log(hp)
  mx = exp(Q[1] + Q[2]lν + Q[3]λ + Q[7]lν^2 + Q[8]λ^2 + Q[9]lν*λ)
  sx = exp(Q[4] + Q[5]lν + Q[6]λ + Q[10]lν^2 + Q[11]λ^2 + Q[12]*lν*λ + Q[16]λ^3)
  sx2 = Q[13] + Q[14]lν + Q[15]λ
  (log(mean_exp) - ((σ*sx)^2 + σ*sx2))/ mx
end;


# Q_last = Q
Q, re = fit_mean_log_model(ds; inits=[[Q_last..., 0]])
a=1
# loss


# λ_model(period, vol_q) = λ_model(period, vol_q, Q_λ);

# # mean_exp_model(μ, σ, ν, λ, hp, Q) = begin
# #   exp(Q[1]μ + ???
# # end;


# mean_log(mean_exp, σνλ, approx) = begin
#   (σ, ν, λ) = σνλ
#   ...
# end

# approx = calc_mean_log_approx(-0.1:0.1:0.1, 0.1:0.1:2.0, 2.5:0.1:4.0, -0.5:0.1:0.5; lp=1e-8, 1-1e-4)

# mean_log(0.5, (0.1, 3.0, -0.1), approx)