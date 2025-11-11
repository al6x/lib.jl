include.(["../nmix.jl", "../Lib.jl", "../plots.jl", "../skewt.jl", "../helpers.jl"]);
using Statistics, DataFrames, Plots, CSV
using StatsFuns: normlogpdf
using .Lib

t_pdf_grid(μ, σ, ν, λ, tq) = begin
  d = SkewT(μ, σ, ν, λ)
  qs = collect(range(tq, 1 - tq; length=100))
  x_prob = quantile.(Ref(d), qs)
  x_quan = collect(range(x_prob[1], x_prob[end]; length=100))
  x = vcat(x_prob, x_quan) |> sort |> unique
  y = pdf.(Ref(d), x)
  ps = cdf.(Ref(d), x)
  x, y, ps
end;

# Raw ----------------------------------------------------------------------------------------------
@inline tmix_logpdf_raw(Q, μ, s, x; debug=false) = begin
  ws = (Q[1], Q[2], Q[3], Q[4], Q[5], Q[6])
  ws = ws ./ sum(ws)
  ss = (Q[7], Q[8], Q[9], Q[10], Q[11], Q[12])

  debug && begin
    order = sortperm([ws...]; rev=true)
    println("ws=$(round.(ws[order]; digits=6)) ss=$(round.(ss[order]; digits=6))")
  end
  nmix_logpdf(ws, μ, ss .* s, x)
end;

@inline tmix_logpdf_raw(Q, μ, s, x; debug=false) = begin
  w1 = Q[1]; w2=w1*Q[2]; w3=w2*Q[3]; w4=w3*Q[4]; w5=w4*Q[5]; w6=w5*Q[6];
  ws = (w1, w2, w3, w4, w5, w6)
  ws = ws ./ sum(ws)
  s1 = Q[7]; s2=s1*5*Q[8]; s3=s2*5*Q[9]; s4=s3*5*Q[10]; s5=s4*5*Q[11]; s6=s5*5*Q[12];
  ss = (s1, s2, s3, s4, s5, s6)

  debug && begin
    order = sortperm([ws...]; rev=true)
    println("ws=$(round.(ws[order]; digits=6)) ss=$(round.(ss[order]; digits=6))")
  end
  nmix_logpdf(ws, μ, ss .* s, x)
end;

ν_raw = 3.0
raw_grid = t_pdf_grid(0.0, 1.0, ν_raw, 0.0, 1e-4);
fit_raw(model, Q0) = begin
  xt, yt = raw_grid

  lower, upper = unzip([
    (1e-6, 1), (1e-6, 10.0), (1e-6, 1.5), (1e-6, 1), (1e-6, 1), (1e-6, 1),
    (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1),
  ])

  weights = sqrt.(yt) # tails less important
  weights = weights ./ median(weights)

  loss(Q) = begin
    llhs = model.(Ref(Q), 0.0, 1.0, xt) #./abs.(xt)
    mean(weights.*(llhs .- log.(yt)) .^ 2)
  end

  Q = Optim.optimize(
    loss, lower, upper, Q0, Optim.Fminbox(Optim.LBFGS()); autodiff = :forward
    # loss, lower, upper, Q0, Optim.Fminbox(Optim.NelderMead())
  ) |> ensure_converged

  println("found: Q=$(round.(Q; digits=6)) loss=$(round(exp(loss(Q)); digits=6))")

  Q, exp(loss(Q))
end;

Q0_raw = 0.1.*rand(12);
Q_raw, _ = fit_raw(tmix_logpdf_raw, Q0_raw); # loss=1.000017
tmix_logpdf_raw(Q, 0.0, 1.0, 0.0; debug=true)

# Fit for ν=3.0
# Q=[
#   0.500345, 2.462465, 0.589565, 0.30334, 0.228527, 0.181583,
#   0.365046, 0.321954, 0.33054, 0.340025, 0.353739, 0.410356
# ] loss=1.000002
#
# ws = (0.279752, 0.279752, 0.279752, 0.129822, 0.026776, 0.004147)
# ss = (0.400234, 0.615806, 0.833984, 1.414057, 2.634474, 5.62312)

# TMix ---------------------------------------------------------------------------------------------
@inline tmix_logpdf_best(Q, μ, s, ν, x) = begin
  # loss=1.001 but error not consistent across ν
  lν = log(ν)
  v = exp(Q[7] + Q[8]*lν + Q[9]*lν*lν)

  c2 = log.((0.279752, 0.279752, 0.279752, 0.129822, 0.026776, 0.004147))
  ci = log.((0.400234, 0.615806, 0.833984, 1.414057, 2.634474, 5.62312))

  ws = exp.(Q[1] .+ (Q[2] .+ Q[3].*c2).*v) .+ 1e-9
  ws = ws./sum(ws)
  ss = exp.(Q[4] .+ (Q[5] .+ Q[6].*ci).*v) .+ 1e-9

  nmix_logpdf(ws, μ, ss .* s, x)
end;

@inline tmix_logpdf(Q, μ, s, ν, x; debug=false) = begin
  # loss=1.005, error consistent across ν
  lν = log(ν)
  v = exp(Q[7] + Q[8]*lν + Q[9]*lν*lν)

  # ci = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0).^0.5 # loss=1.0049
  # ci = (1.0, 2.0, 3.0, 4.0, 3.5, 5.43).^0.5 # loss=1.001609
  ci = (
    (Q[10] .+ Q[11].*(1.0, 2.0, 3.0, 4.0))...,
    (Q[12], Q[13])...,
  ) # loss=1.0015

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

fit(model, Q0) = begin
  # νs = exp.(range(log(2.5), log(15.0); length=10));
  νs = [2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 7.0, 15.0];
  tdistrs = map(νs) do ν t_pdf_grid(0.0, 1.0, ν, 0.0, 1e-4) end;

  weights = [begin
    ν, (_, yt) = νs[i], tdistrs[i]
    sqrt.(yt)./ν
  end for i in 1:length(νs)]
  mweight = median(vcat(weights...))
  weights = [w ./ mweight for w in weights]

  loss(Q) = begin
    sum = 0.0
    for i in 1:length(νs)
      ν, (xt, yt) = νs[i], tdistrs[i]
      llhs = model.(Ref(Q), 0.0, 1.0, Ref(ν), xt)
      sum += mean(weights[i].*(llhs .- log.(yt)) .^ 2)
    end
    sum / length(νs)
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
  for i in 1:20
    println(i)
    try
      Q0 = randn(13)
      Q0[10] = 0.7 + 0.1randn(); Q0[11] = 0.3 + 0.1randn()
      Q0[12] = 3.5 + 0.5randn(); Q0[13] = 5.43 + 0.5randn()

      Q, loss = fit(tmix_logpdf, Q0);
      # Q, loss = fit(tmix_logpdf, randn(9));
      push!(results, (Q, loss))
    catch e
      println("failed")
    end
  end
  _, bi = findmin(getindex.(results, 2))
  Q, loss = results[bi]
  println("best Q=$(round.(Q; digits=6)), loss=$(loss)")
  tmix_logpdf(Q, 0.0, 1.0, 3.0, 0.0; debug=true);
  Q
end;

# Plotting -----------------------------------------------------------------------------------------
plot_fit(model, type="error") = begin
  νs = [2.7, 3.0, 4.0, 5.0, 8.0, 15]

  dfs = map(νs) do ν
    x, y, p = t_pdf_grid(0.0, 1.0, ν, 0.0, 1e-4)
    y2 = exp.(model.(0.0, 1.0, Ref(ν), x))
    e = y ./ y2
    DataFrame(; x, e, y, y2, ν, p)
  end
  df = vcat(dfs...)
  df = df[df.p .< 0.5, :]

  if type == "pdf"
    df = df[df.ν .== 3.0, :]
    p1 = plot(df.p, df.y, legend=:bottomright, xlabel="Probability", ylabel="Density")
    plot!(p1, df.p, df.y2)

    p2 = plot(df.p, df.y, xscale=:log10, yscale=:log10, legend=:bottomright,
      xlabel="Probability", ylabel="Density")
    plot!(p2, df.p, df.y2)

    plot(p1, p2, layout=(2,1), plot_title="ν=3")
  else
    df |> @vlplot(
      :line,
      x={:p, type="quantitative", scale={type="log", domain=[1e-4, 0.5], clamp=true}},
      y={:e, type="quantitative", scale={type="log", domain=[0.7, 1.6], clamp=true}},
      color={:ν, type="nominal"},
      width=600, height=400,
    )
  end
end;

fit_tmix_logpdf(μ, s, ν, x) = tmix_logpdf(Q, μ, s, ν, x)
plot_fit(fit_tmix_logpdf, "error")
plot_fit(fit_tmix_logpdf, "pdf")

# Final --------------------------------------------------------------------------------------------
@inline tmix_ws_ss(ν::Real) = begin
  # See fit_tmix.jl for fitting
  # Q=[
  #   -0.523857, -0.349351, -2.164389, -0.074823, 3.886313, 1.234103,
  #   -0.608911, -1.804228, 0.278592, 7.081085, -3.867854, -2.255962, 9.032569
  # ], loss=1.00156

  @assert 2.5 <= ν <= 15.0 ν

  lν = log(ν)
  v = exp(-0.608911 - 1.804228*lν + 0.278592*lν*lν)

  ws = exp.(-0.523857 .+ (-7.304031, 1.067509, 9.439049, 17.810589, 4.533427, -19.899341) .* v)
  ws = ws ./ sum(ws)

  ss = exp.(-0.074823 .+ ( 7.851771, 3.07844, -1.694891, -6.468222, 1.102224,  15.033435) .* v)

  ws, ss
end

@inline tmix_logpdf_final(μ, s, ν, x) = begin
  ws, ss = tmix_ws_ss(ν)
  nmix_logpdf(ws, μ, ss .* s, x)
end;

plot_fit(tmix_logpdf_final, "error")
plot_fit(tmix_logpdf_final, "pdf")