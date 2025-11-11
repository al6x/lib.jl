include.(["./nmix.jl", "./Lib.jl", "./plots.jl", "./skewt.jl", "./helpers.jl"])
using PyCall, Statistics, DataFrames, Plots, CSV
using .Lib

# Typical parameters for annual log-returns
# ALPHA, BETA, LOC, SCALE = 7.0, 0.0, 0.0, 0.015

py"""
import numpy as np
from scipy.stats import norminvgauss

def nig_pdf_xy(l, s, α, β, tq):
  # Hybrid probability, quantile grid
  qs = np.linspace(tq, 1 - tq, 100)
  x_prob = norminvgauss.ppf(qs, α, β, loc=l, scale=s)
  x_quan = np.linspace(x_prob[0], x_prob[-1], 100)
  x = np.sort(np.unique(np.concatenate((x_prob, x_quan))))
  y = norminvgauss.pdf(x, α, β, loc=l, scale=s)

  return x, y, f"NIG({l}, {s}, {α}, {β})"
"""

nig_pdf_xy(l, s, α, β, tq) = py"nig_pdf_xy"(l, s, α, β, tq);

t_pdf_xy(μ, σ, ν, λ, tq) = begin
  d = SkewT(μ, σ, ν, λ)
  qs = collect(range(tq, 1 - tq; length=50))
  x_prob = quantile.(Ref(d), qs)
  x_quan = collect(range(x_prob[1], x_prob[end]; length=50))
  x = vcat(x_prob, x_quan) |> unique |> sort
  y = pdf.(Ref(d), x)
  x, y, "SkewT($μ, $σ, $ν, $λ)"
end;

@inline nmorph_logpdf(l, s, wr, sr, w_pow, r_pow, x) = begin
  # w_pow, r_pow = 2.0, 2.0
  ci = (0.0, 1.0, 2.0, 3.0, 4.0)

  wm = log(wr) / ci[end]^w_pow
  ws = exp.(-wm.* ci .^ w_pow)
  ws = ws ./ sum(ws)
  # @assert ws[1]/ws[5] ≈ wr

  sm = log(sr) / ci[end]^r_pow
  ss = exp.(sm .* ci .^ r_pow) .* s
  # @assert ss[5]/ss[1] ≈ sr

  logpdf(NMix(ws, (l, l, l, l, l), ss), x)
end;

targets = [
  t_pdf_xy(0.0, 1.0, 2.7, 0.0, 1e-4),
  # t_pdf_xy(0.0, 1.0, 3.0, 0.0, 1e-4),
  # t_pdf_xy(0.0, 1.0, 7.0, 0.0, 1e-4),
  # t_pdf_xy(0.0, 1.0, 10.0, 0.0, 1e-4),

  # nig_pdf_xy(0.0, 1.0, 1.0, 0.0, 1e-4),
  # nig_pdf_xy(0.0, 1.0, 2.0, 0.0, 1e-4),
  # nig_pdf_xy(0.0, 1.0, 4.0, 0.0, 1e-4),
  # nig_pdf_xy(0.0, 1.0, 8.0, 0.0, 1e-4),
];

@inline nmorph_logpdf(Q, x) = begin
  T = typeof(Q[1])
  nmorph_logpdf(T(0.0), Q[1], Q[2], Q[3], Q[4], Q[5], x);
end

fit_to_target_pdf(x, y) = begin
  lower, upper = unzip([
    (1e-6, Inf), (1.0, Inf), (1.0, Inf), (0.1, 5), (0.1, 5),
  ])

  Q0 = [std(x), 10.0, 2.0, 2.0, 2.0]

  loss(R) = begin
    sr = R[4]
    # penalty = 5e-4*sqrt(sr^2 + 1)
    penalty = 0.0
    mean((nmorph_logpdf.(Ref(R), x) .- log.(y)) .^ 2) + penalty, penalty
  end

  R = Optim.optimize(
    R -> loss(R)[1], lower, upper, Q0, Optim.Fminbox(Optim.LBFGS()); autodiff = :forward
  ) |> ensure_converged

  R, loss(R)
end;

ds = vcat([begin
  Q, loss = fit_to_target_pdf(x, yt)
  println("$name: Q=$(round.(Q; digits=4)), loss=$(round.(loss; digits=4))")
  y = exp.(nmorph_logpdf.(Ref(Q), x))
  DataFrame(; x, y, yt, name)
end for (x, yt, name) in targets]...);

plot_xyc_by("NMix", ds; x=(:x, :symlog), y=:y, y2=:yt, by=:name, mark=:line, mark2=:circle);
plot_xyc_by("NMix", ds; x=(:x, :symlog), y=(:y, :log), y2=:yt, by=:name, mark=:line, mark2=:circle);

# SkewT(0.0, 1.0, 2.7, 0.0): Q=[-0.0, 0.5726, 63.1202, 9.428], loss=0.0319
# SkewT(0.0, 1.0, 3.0, 0.0): Q=[0.0, 0.6348, 50.5796, 7.4271], loss=0.0258
# SkewT(0.0, 1.0, 7.0, 0.0): Q=[-0.0, 0.8033, 7.9059, 2.4561], loss=0.003
# SkewT(0.0, 1.0, 10.0, 0.0): Q=[0.0, 0.8246, 4.2794, 1.9824], loss=0.0009
# NIG(0.0, 1.0, 1.0, 0.0): Q=[-0.0, 0.6727, 3.1177, 2.888], loss=0.0068
# NIG(0.0, 1.0, 2.0, 0.0): Q=[-0.0, 0.5224, 2.1781, 2.2147], loss=0.002
# NIG(0.0, 1.0, 4.0, 0.0): Q=[0.0, 0.395, 1.4889, 1.7971], loss=0.0004
# NIG(0.0, 1.0, 8.0, 0.0): Q=[0.0, 0.2939, 1.0231, 1.5325], loss=0.0001

# Fitting TDist ν ----------------------------------------------------------------------------------
wr_sr(R, ν) = begin
  wr = exp(R[1] .+ R[2].*log(ν) .+ R[3].*log(ν).^2)
  sr = exp(R[4] .+ R[5].*log(ν) .+ R[6].*log(ν).^2)
  wr, sr
end

# let
  νs = exp.(range(log(2.5), log(15.0); length=100));
  tdistrs = map(νs) do ν t_pdf_xy(0.0, 1.0, ν, 0.0, 1e-4) end;

  wr_init = [5.9630, -0.9800, -0.285]
  sr_init = [5.4329, -3.4374,  0.6125]
  R0 = [wr_init..., sr_init...]

  loss(R) = begin
    T = typeof(R[1])
    l, s = T(0.0), T(1.0)
    [begin
      ν, (xt, yt) = νs[i], tdistrs[i]
      wr, sr = wr_sr(R, ν)
      llhs = map(xt) do x nmorph_logpdf(l, s, wr, sr, x) end
      mean((llhs .- log.(yt)) .^ 2)
    end for i in 1:length(νs)] |> mean
  end

  R = Optim.optimize(
    loss, R0, Optim.LBFGS(); autodiff = :forward
    # Optim.NelderMead()
  ) |> ensure_converged

  println("found: R=$(round.(R; digits=4))")
# end

# R = [5.9630, -0.9800, -0.285, 5.4329, -3.4374,  0.6125]
R = [12.9373, -8.1129, 3.9393, 4.5774, -1.9635, 0.5365]
wr_sr(ν) = wr_sr(R, ν)

wr_sr(ν) = begin
  # log log sr = 0.8024 − 1.1451 log log ν − 0.3245 (log log ν)^2
  # log wr = −4.33153714 -1.99113114 log sr + 8.64600104 log sr ^ 0.5
  @assert 2.5 <= ν <= 15.0 ν
  llν = log(log(ν))
  sr = exp(exp(0.8024 - 1.1451*llν - 0.3245*llν^2))
  lsr = log(sr)
  wr = exp(-4.33153714 - 1.99113114*lsr + 8.64600104*sqrt(lsr))
  wr, sr
end

wr_sr(3.0)

# Plotting
# νs = [2.7, 3.0, 5.0, 7.0, 10.0, 15.0]
νs = [2.995]
tdistrs = map(νs) do ν t_pdf_xy(0.0, 1.0, ν, 0.0, 1e-4) end;

ν_ds = vcat([begin
  wr, sr = wr_sr(ν)
  wr, sr = 50.5796, 7.4271 # 50.4175, 7.4225
  y = map(x) do xi exp(nmorph_logpdf(0.0, 1.0, wr, sr, xi)) end
  DataFrame(; x, y, yt, name)
end for (ν, (x, yt, name)) in zip(νs, tdistrs)]...);

plot_xyc_by("NMix TDist", ν_ds; x=:x, y=:y, y2=:yt, by=:name, mark=:line, mark2=:circle);
plot_xyc_by("NMix TDist", ν_ds; x=(:x, :symlog), y=(:y, :log), y2=:yt, by=:name, mark=:line);













# Fitting NMix
ν_fit = DataFrame([begin
  Q, loss = fit_to_target_pdf(x, yt)
  l, s, wr, sr = Q
  (; ν, wr, sr)
end for (ν, (x, yt, name)) in zip(νs, tdistrs)]);
vscodedisplay(ν_fit)
let
  (; ν, wr, sr) = ν_fit
  plot(
    scatter(sr, wr; xlabel="sr", ylabel="wr",  title="sr vs wr", xscale=:log10, yscale=:log10),
    scatter(ν, sr; xlabel="ν",  ylabel="sr",  title="ν vs sr", xscale=:log10, yscale=:log10),
    scatter(ν, wr; xlabel="ν",  ylabel="wr",  title="ν vs wr", xscale=:log10, yscale=:log10),
    layout=(3,1), size=(1100,2000)
  )
end

# Predicting sr from ν


fit_ν_sr(νs, srs) = begin
  loss(Q) = mean((log.(predict_sr.(Ref(Q), νs)) .- log.(srs)).^2)
  Q0 = [0.0, 0.0, 0.0]
  Optim.optimize(
    loss, Q0, Optim.LBFGS(); autodiff=:forward
  ) |> ensure_converged
end

Qsr = fit_ν_sr(ν_fit.ν, ν_fit.sr)
println(round.(Qsr; digits=4))

predict_sr(ν) = predict_sr(Qsr, ν)

# plot_xyc_by("TDist sr vs wr", ν_fit; x=:sr, y=:wr);
# plot_xyc_by("TDist ν vs sr", ν_fit; x=:ν, y=:sr);
# plot_xyc_by("TDist ν vs wr", ν_fit; x=:ν, y=:wr);



to_tsv(ν_fit, "./tmp/tdist_df.tsv")

# log log sr = 0.8024 − 1.1451 log log ν − 0.3245 (log log ν)^2

# log wr = −4.33153714 -1.99113114 log sr + 8.64600104 log sr ^ 0.5