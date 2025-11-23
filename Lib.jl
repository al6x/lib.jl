module Lib

import Printf, Optim, CSV
using Statistics, Random
using SpecialFunctions: loggamma

export dedent, configure!, unzip, flatten, spread,
  smape, sigmoid, abs_soft, ensure_converged

smape(y, ŷ, ϵ = 1e-6) = abs(y - ŷ) / ((abs(y) + abs(ŷ) + ϵ) / 2)

@inline sigmoid(x) = 1.0 / (1.0 + exp(-x))

# abs_soft(x; δ=1e-3) = δ^2 * (sqrt(1 + (x/δ)^2) - 1)

ensure_converged(
  r::Union{Optim.UnivariateOptimizationResults, Optim.MultivariateOptimizationResults}
) = Optim.converged(r) ? r.minimizer : error("Can't fit")

spread(nt::NamedTuple) = begin
  n = nothing
  for v in values(nt)
    v isa AbstractArray || continue
    n === nothing && (n = length(v); continue)
    length(v) == n || error("Mismatched lengths among vector fields.")
  end
  n === nothing && return nt  # No composite fields, return as-is
  (; (k => (v isa AbstractArray ? v : fill(v, n)) for (k, v) in pairs(nt))...)
end

dedent(s::AbstractString) = begin
  lines = split(s, '\n')

  # drop leading/trailing blank lines (whitespace-only)
  while !isempty(lines) && isempty(strip(lines[1])); popfirst!(lines); end
  while !isempty(lines) && isempty(strip(lines[end])); pop!(lines); end
  isempty(lines) && return ""

  min_indent = minimum(length(l) - length(lstrip(l)) for l in lines if !isempty(strip(l)))
  join([isempty(strip(l)) ? "" : l[min_indent+1:end] for l in lines], "\n")
end

unzip(rows::AbstractVector{<:Union{Tuple,AbstractVector}}) = begin
  isempty(rows) && error("empty input")
  N = length(first(rows))
  return ntuple(j -> getindex.(rows, j), N)
end

unzip(rows::Base.Generator) = unzip(collect(rows))

flatten(v::AbstractVector{<:AbstractVector}) = collect(Iterators.flatten(v))

# mutable struct LibConfig show_round::Int end
# const config = Ref{Union{LibConfig, Nothing}}(nothing)

# function configure!(; show_round=4)
#   config[] = LibConfig(show_round)
# end
# configure!()

# function Base.show(io::IO, f::Float64)
#   Printf.@printf(io, "%.*f", config[].show_round, f)
# end

export tdist_logpdf, tdist_logpdf_std

@inline tdist_logpdf(ν, x) = begin
  @assert ν > 1 "ν must be > 1"
  c = loggamma((ν+1)/2) - loggamma(ν/2) - 0.5*log(ν*pi)
  c - (ν+1)/2 * log1p(x^2/ν)
end

@inline tdist_logpdf(μ, scale, ν, x) = begin
  @assert scale > 0 "σ must be > 0"
  tdist_logpdf(ν, (x-μ)/scale) - log(scale)
end

@inline tdist_logpdf_std(μ, σ, ν, x) = begin
  @assert σ > 0 "σ must be > 0"
  scale = σ * sqrt((ν - 2) / ν)
  tdist_logpdf(ν, (x-μ)/scale) - log(scale)
end

Base.round(x::NamedTuple; opts...) =
  (; (k => v isa AbstractFloat ? round(v; opts...) : v for (k,v) in pairs(x))...)

Random.seed!(op::Function, seed::Integer) = begin
  state = copy(Random.default_rng())
  try
    Random.seed!(seed)
    op()
  finally
    copy!(Random.default_rng(), state)
  end
end

export skipnan
skipnan(x::AbstractArray) = Iterators.filter(!isnan, x)

export say
say(msg::String) = run(`say $msg`)

export mscore
mscore(x::Vector{Union{Missing,<:Real}}; c=nothing) = begin
  present = skipmissing(x)
  c === nothing && (c = median(present))
  s = median(abs.(present .- c))
  (x .- c) ./ s
end


# Finance ------------------------------------------------------------------------------------------
calc_diff(op, x::Vector{Union{Missing,Float64}}) = begin
  y, prev = Vector{Union{Missing,Float64}}(undef, length(x)), missing
  for i in 1:length(x)
    y[i] = if ismissing(x[i]) missing else
      v = ismissing(prev) ? missing : op(prev, x[i])
      prev = x[i]
      v
    end
  end
  y
end

ewa(x::AbstractVector{<:Union{Missing,Real}}, α::Real; fill_missing=false) = begin
  @assert 0 < α < 1 "α must be in (0,1)"
  y, n, i = similar(x), length(x), 1
  @inbounds while i <= n && x[i] === missing
    y[i] = missing; i += 1
  end
  if i <= n
    y[i] = s = x[i]; i += 1
    @inbounds for j in i:n
      if x[j] === missing
        y[j] = fill_missing ? s : missing
      else
        y[j] = s = α * x[j] + (1 - α) * s
      end
    end
  end
  y
end

roll_slow(op, vs::Vector{Union{Missing, Float64}}; window, min) = begin
  r = Vector{Union{Missing, Float64}}(missing, length(vs))
  for t in 1:length(vs)
    vs[t] === missing && continue
    present = skipmissing(vs[max((t - window + 1), 1):t])
    count(_ -> true, present) ≥ min || continue
    r[t] = op(present)
  end
  r
end

calc_r2(rs::Vector; period::Int) = begin
  @assert period > 0
  n = length(rs)
  [begin
    t2 = t + period
    if ismissing(rs[t]) || t2 > n
      missing
    else
      sum = 0.0; for i in t+1:t2-1
        r = rs[i]; !ismissing(r) && (sum += r)
      end

      rlast = rs[t2]
      # If last return at t2 is non trading day using next trading day,
      # but no more than 2 days ahead
      if ismissing(rlast)
        tlast = t2
        while ismissing(rlast) && tlast < min(n, t + period + 2)
          tlast += 1; rlast = rs[tlast]
        end
      end
      ismissing(rlast) ? missing : sum + rlast
    end
  end for t in 1:n]
end

export nan_to_missing, missing_to_nan
nan_to_missing(x::AbstractArray{Float64}) = map(v -> isnan(v) ? missing : v, x)
missing_to_nan(x::AbstractArray{Union{Missing,Float64}}) = map(v -> ismissing(v) ? NaN : v, x)

end