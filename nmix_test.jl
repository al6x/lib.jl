include.(["./nmix.jl", "./Lib.jl", "./plots.jl", "./skewt.jl"])
using PyCall, Statistics, DataFrames
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

@inline nmorph_logpdf(Q, x) = nmorph_logpdf(Q[1], Q[2], Q[3], Q[4], 2.5, 1.5, x);

fit_to_target_pdf(x, y) = begin
  lower, upper = unzip([
    (-Inf, Inf), (1e-6, Inf), (1.0, 100), (1.0, 10), (0.1, 10), (0.1, 10),
  ])

  Q0 = [mean(x), std(x), 5.0, 1.5, 1.0, 1.0]

  loss(Q) = mean((nmorph_logpdf.(Ref(Q), x) .- log.(y)) .^ 2)

  Q = Optim.optimize(
    loss, lower, upper, Q0, Optim.Fminbox(Optim.LBFGS()); autodiff = :forward
  ) |> ensure_converged

  Q, loss(Q)
end;

targets = [
  t_pdf_xy(0.0, 1.0, 2.7, 0.0, 1e-4),
  t_pdf_xy(0.0, 1.0, 3.0, 0.0, 1e-4),
  t_pdf_xy(0.0, 1.0, 7.0, 0.0, 1e-4),
  t_pdf_xy(0.0, 1.0, 10.0, 0.0, 1e-4),

  nig_pdf_xy(0.0, 1.0, 1.0, 0.0, 1e-4),
  nig_pdf_xy(0.0, 1.0, 2.0, 0.0, 1e-4),
  nig_pdf_xy(0.0, 1.0, 4.0, 0.0, 1e-4),
  nig_pdf_xy(0.0, 1.0, 8.0, 0.0, 1e-4),
];

ds = vcat([begin
  Q, loss = fit_to_target_pdf(x, yt)
  println("$name: Q=$(round.(Q; digits=4)), loss=$(round(loss; digits=4))")
  y = exp.(nmorph_logpdf.(Ref(Q), x))
  DataFrame(; x, y, yt, name)
end for (x, yt, name) in targets]...);

plot_xyc_by("NMix", ds; x=:x, y=:y, y2=:yt, by=:name);
plot_xyc_by("NMix", ds; x=(:x, :symlog), y=(:y, :log), y2=:yt, by=:name, mark=:line);

# SkewT(0.0, 1.0, 2.7, 0.0): Q=[-0.0, 0.5152, 58.9768, 10.0], loss=0.0386
# SkewT(0.0, 1.0, 3.0, 0.0): Q=[-0.0, 0.509, 70.209, 10.0], loss=0.004
# SkewT(0.0, 1.0, 7.0, 0.0): Q=[0.0, 0.7575, 17.867, 2.8938], loss=0.0011
# SkewT(0.0, 1.0, 10.0, 0.0): Q=[0.0, 0.7957, 8.1363, 2.1857], loss=0.0005
# NIG(0.0, 1.0, 1.0, 0.0): Q=[0.0, 0.6236, 4.2274, 3.2361], loss=0.0036
# NIG(0.0, 1.0, 2.0, 0.0): Q=[-0.0, 0.4991, 3.0572, 2.3939], loss=0.0012
# NIG(0.0, 1.0, 4.0, 0.0): Q=[-0.0, 0.3849, 2.1321, 1.8895], loss=0.0003
# NIG(0.0, 1.0, 8.0, 0.0): Q=[-0.0, 0.2897, 1.4721, 1.5807], loss=0.0