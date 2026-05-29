using Distributions

tail_declustered_with_runs(x, u, runs) = begin
  @assert u > 0
  n, kept = length(x), Float64[]
  in_cluster, best, quiet = false, -Inf, 0

  for t in 1:n
    xt = x[t]
    if xt > u
      if !in_cluster
        in_cluster = true; best = xt
      elseif xt > best
        best = xt
      end
      quiet = 0
    elseif in_cluster
      quiet += 1
      if quiet >= runs
        push!(kept, best)
        in_cluster = false; best = -Inf; quiet = 0
      end
    end
  end
  in_cluster && push!(kept, best)
  kept
end

fit_hill(x, u; runs::Union{Int, Nothing}=nothing) = begin
  @assert u > 0
  ex = isnothing(runs) ? x[x .> u] : tail_declustered_with_runs(x, u, runs)
  z = (ex .- u) ./ u
  @assert !isempty(z) && all(z .>= 0)
  α = 1/mean(log1p.(z))
  @assert isfinite(α) "Hill non-finite"
  α
end

