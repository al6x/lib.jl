import StatsFuns
import Distributions: logpdf, SkewNormal

@inline skewnormal_logpdf(ξ, ω, α, x) = begin
  z = (x - ξ) / ω
  lp = -log(ω) - 0.5*z^2 - 0.5*log(2π) # log φ(z)
  Φ = 0.5 * (1 + StatsFuns.erf(α*z / sqrt(2))) # Φ(α z)
  lp + log(2 * Φ)
end

skewnormal_mean_exp(ξ, ω, α) = begin
  f(x) = pdf(SkewNormal(ξ, ω, α), x) * exp(x)
  QuadGK.quadgk(f, -Inf, Inf)[1]
end

skewnormal_mean_exp(ξ, ω, α) = begin
  δ = α / sqrt(1 + α^2)
  Φ = 0.5 * (1 + StatsFuns.erf(δ * ω / sqrt(2)))
  2 * exp(ξ + 0.5 * ω^2) * Φ
end

skewnormal_fit_mle(x::AbstractVector{<:Real}) = begin
  nll(θ) = begin
    ξ, ω, α = θ[1], exp(θ[2]), θ[3]
    ω < 1e-12 && return Inf
    -sum(logpdf.(Ref(SkewNormal(ξ, ω, α)), x))
  end

  ξ0 = mean(x); ω0 = log(std(x) + 1e-6)
  inits = [[ξ0, ω0, -1.0], [ξ0, ω0,  1.0]]

  best = nothing; bestval = Inf
  for θ0 in inits
    res = Optim.optimize(nll, θ0, Optim.BFGS(); autodiff=:forward)
    !Optim.converged(res) && continue
    v = Optim.minimum(res)
    if v < bestval
      bestval = v; best = res
    end
  end

  best === nothing && error("Can't estimate SkewNormal")

  θ = Optim.minimizer(best)
  SkewNormal(θ[1], exp(θ[2]), θ[3])
end