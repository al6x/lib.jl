include.(["../nmix.jl", "../Lib.jl", "../plots.jl", "../helpers.jl"])
using PyCall, Statistics, DataFrames, Plots, CSV, Optim
using StatsFuns: normlogpdf
using .Lib

# Typical parameters for annual log-returns
# ALPHA, BETA, LOC, SCALE = 7.0, 0.0, 0.0, 0.015

py"""
import numpy as np
from scipy.stats import norminvgauss

def nig_pdf_grid(l, s, α, β, tq):
  # Hybrid probability, quantile grid
  qs = np.linspace(tq, 1 - tq, 100)
  x_prob = norminvgauss.ppf(qs, α, β, loc=l, scale=s)
  x_quan = np.linspace(x_prob[0], x_prob[-1], 100)
  x = np.sort(np.unique(np.concatenate((x_prob, x_quan))))
  y = norminvgauss.pdf(x, α, β, loc=l, scale=s)
  p = norminvgauss.cdf(x, α, β, loc=l, scale=s)
  return x, y, p

def nig_pdf(l, s, α, β, x):
  return norminvgauss.pdf(x, α, β, loc=l, scale=s)
"""

nig_pdf_grid(l, s, α, β, tq) = py"nig_pdf_grid"(l, s, α, β, tq);

# Fitting ------------------------------------------------------------------------------------------
@inline nigmix_logpdf(Q, μ, s, α, x; debug=false) = begin
  lα = log(α)
  v = exp(Q[7] + Q[8]*lα + Q[9]*lα*lα)

  ci = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0).^Q[10]
  # ci = (
  #   (Q[10] .+ Q[11].*(1.0, 2.0, 3.0, 4.0))...,
  #   (Q[12], Q[13])...,
  # ) # loss=1.0015

  ws = exp.(Q[1] .+ (Q[2] .+ Q[3].*ci).*v)
  ws = ws./sum(ws)
  ss = exp.(Q[4] .+ (Q[5] .+ Q[6].*ci).*v)

  if debug
    r(v) = round(v; digits=6)
    v1, v2, v3 = r(Q[7]), r(Q[8]), r(Q[9])
    aw, bw = r(Q[1]), r.(Q[2] .+ Q[3].*ci)
    as, bs = r(Q[4]), r.(Q[5] .+ Q[6].*ci)
    println("  $((; v1, v2, v3, aw, bw, as, bs))")
  end

  nmix_logpdf(ws, μ, ss .* s, x)
end;

αs = exp.(range(log(0.7), log(10.0); length=10));
nig_distrs = map(αs) do α nig_pdf_grid(0.0, 1.0, α, 0.0, 1e-4) end;
fit(model, Q0) = begin
  weights = [begin
    _, (_, yt) = αs[i], nig_distrs[i]
    sqrt.(yt)
  end for i in 1:length(αs)]
  mweight = median(vcat(weights...))
  weights = [w ./ mweight for w in weights]

  loss(Q) = begin
    sum = 0.0
    for i in 1:length(αs)
      α, (xt, yt) = αs[i], nig_distrs[i]
      llhs = model.(Ref(Q), 0.0, 1.0, Ref(α), xt)
      sum += mean(weights[i].*(llhs .- log.(yt)) .^ 2)
    end
    sum / length(αs)
  end

  Q = Optim.optimize(
    # loss, lower, upper, Q0, Optim.Fminbox(Optim.LBFGS()); autodiff = :forward
    # loss, lower, upper, Q0, Optim.Fminbox(Optim.NelderMead())
    loss, Q0, Optim.LBFGS(); autodiff = :forward
  ) |> ensure_converged

  println("found: Q=$(round.(Q; digits=6)) loss=$(round(exp(loss(Q)); digits=6))")

  Q, exp(loss(Q))
end;

Q = let
  results = []
  for i in 1:10
    println(i)
    try
      Q0 = randn(10)
      # Q0[10] = 0.5 + randn();
      # Q0 = randn(13)
      # Q0[10] = 0.7 + 0.1randn(); Q0[11] = 0.3 + 0.1randn()
      # Q0[12] = 3.5 + 0.5randn(); Q0[13] = 5.43 + 0.5randn()
      Q, loss = fit(nigmix_logpdf, Q0);
      push!(results, (Q, loss))
    catch e
      println("failed")
    end
  end
  _, bi = findmin(getindex.(results, 2))
  Q, loss = results[bi]
  println("best Q=$(round.(Q; digits=6)), loss=$(loss)")
  nigmix_logpdf(Q, 0.0, 1.0, 3.0, 0.0; debug=true);
  Q
end;

# Plotting -----------------------------------------------------------------------------------------
plot_fit(model, type="error") = begin
  αs = [0.7, 1.5, 3.0, 6.0, 10]

  dfs = map(αs) do α
    x, y, p = nig_pdf_grid(0.0, 1.0, α, 0.0, 1e-4)
    y2 = exp.(model.(0.0, 1.0, Ref(α), x))
    e = y ./ y2
    DataFrame(; x, e, y, y2, α, p)
  end
  df = vcat(dfs...)
  df = df[df.p .< 0.5, :]

  if type == "pdf"
    df = df[df.α .== 3.0, :]
    p1 = plot(df.p, df.y, legend=:bottomright, xlabel="Probability", ylabel="Density")
    plot!(p1, df.p, df.y2)

    p2 = plot(df.p, df.y, xscale=:log10, yscale=:log10, legend=:bottomright,
      xlabel="Probability", ylabel="Density")
    plot!(p2, df.p, df.y2)

    plot(p1, p2, layout=(2,1), plot_title="α=3")
  else
    df |> @vlplot(
      :line,
      x={:p, type="quantitative", scale={type="log", domain=[1e-4, 0.5], clamp=true}},
      y={:e, type="quantitative", scale={type="log", domain=[0.7, 1.6], clamp=true}},
      color={:α, type="nominal"},
      width=600, height=400,
    )
  end
end;

fit_nigmix_logpdf(μ, s, α, x) = nigmix_logpdf(Q, μ, s, α, x)
plot_fit(fit_nigmix_logpdf, "error")
plot_fit(fit_nigmix_logpdf, "pdf")

# Final --------------------------------------------------------------------------------------------
@inline nigmix_ws_ss(α::Real) = begin
  # See fit_nigmix.jl for fitting
  # Q=[
  #   0.955056, 0.14849, -0.361283, -1.811039, 0.635754, 0.185508, 0.710302,
  #   -0.278778, -0.076414, 0.737572
  # ], loss=1.000659

  @assert 0.7 <= α <= 10.0 α

  lα = log(α)
  v = exp(0.710302 - 0.278778*lα - 0.076414*lα*lα)

  ws = exp.( 0.955056 .+ (0.14849, -0.212793, -0.453901, -0.663889, -0.855917, -1.035609) .* v)
  ws = ws ./ sum(ws)

  ss = exp.(-1.811039 .+ (0.635754, 0.821262,  0.945064,  1.052886,  1.151487,  1.243753) .* v)

  ws, ss
end

@inline nigmix_logpdf_final(μ, s, ν, x) = begin
  ws, ss = nigmix_ws_ss(ν)
  nmix_logpdf(ws, μ, ss .* s, x)
end;

plot_fit(nigmix_logpdf_final, "error")
plot_fit(nigmix_logpdf_final, "pdf")