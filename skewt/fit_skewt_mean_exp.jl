includet("../skewt.jl")
using DataFrames, Random, Statistics, StatsBase, Optim, Interpolations, Distributions

Random.seed!(0)

ds = let
  μ_grid  = collect(range(log(1.0001), log(1.8); length=2));
  σ_grid  = collect(exp.(range(log(0.001), log(1); length=10)));
  ν_grid  = collect(exp.(range(log(2.7), log(6.0); length=5)));
  λ_grid  = collect((range(-0.2, 0.1; length=10)));
  # hp_grid = collect(exp.(range(log(0.99), log(0.9999); length=3)));
  hp_grid = [0.9997]

  ds = DataFrame([(; μ, σ, ν, λ, hp) for μ in μ_grid, σ in σ_grid, ν in ν_grid, λ in λ_grid, hp in hp_grid])

  ds.d = SkewT.(0.0, ds.σ, ds.ν, ds.λ)
  ds.h = quantile.(ds.d, ds.hp)
  ds.q085 = quantile.(ds.d, 0.85)
  ds.q099 = quantile.(ds.d, 0.99)

  ds.mean_exp = [mean_exp(d; l=log(1e-6), h=ds.h[i]) for (i, d) in enumerate(ds.d)];
  ds
end

fit_exp_model(ds) = begin
  residuals(Q) = log.(ds.mean_exp) .- log.(mean_exp_model.(ds.d, ds.hp, Ref(Q), ds.q085, ds.q099))
  goal(Q)      = 100000*mean(residuals(Q).^2)
  penalty(Q)   = (1/1000000)mean(Q[2:end].^2)
  loss(Q)      = goal(Q) #+ penalty(Q)

  inits = [randn(Qn) for _ in 1:10]
  rs = [optimize(loss, init, LBFGS(); autodiff=:forward) for init in inits]
  rs = filter(Optim.converged, rs)
  isempty(rs) && error("can't fit")

  mins = map(Optim.minimum, rs)
  Q = Optim.minimizer(rs[argmin(mins)])

  re = abs.(residuals(Q))
  re_i = argmax(re)
  println((; ds[re_i, :]..., model=mean_exp_model(ds.d[re_i], ds.hp[re_i], Q, ds.q085[re_i], ds.q099[re_i]), re=re[re_i]))

  re = round(exp(maximum(abs.(residuals(Q)))); digits=4)
  println("Fit re=$re, goal=$(round(goal(Q); digits=4)), penalty=$(round(penalty(Q); digits=4))")

  return Q, re
end

Qn = 15;
mean_exp_model(d, hp, Q, q085=quantile(d, 0.85), q099=quantile(d, 0.99)) = begin
  (; μ, σ, ν, λ) = d

  lν = log(ν-2.5)
  a, b = (q085 - μ), (q099 - μ)
  la, lb = log(a), log(b)

  m1 = exp(Q[1] + Q[2]lν + Q[3]λ + Q[4]lν^2 + Q[5]λ^2 + Q[6]*lν*λ + Q[7]*la + Q[8]*la^2 + Q[9]*lb)
  m2 = Q[10] + Q[11]lν + Q[12]λ + Q[13]*a + Q[14]*b
  exp(μ + (σ*m1)^2 + σ*m2)
end;

Q, re = fit_exp_model(ds)

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