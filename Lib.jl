module Lib

import Printf, Optim, CSV
using Statistics, Random, Distributions
using SpecialFunctions: loggamma
using StatsFuns: logaddexp, logit

include("Lib/vhash.jl")
include("Lib/data.jl")
include("Lib/rjson.jl")

export logit

export @p
macro p(ex)
  quote
    val = $(esc(ex))
    println($(string(ex)), " = ", val)
    val
  end
end

export dedent, configure!, unzip, spread, vhash, vhash!, to_data, from_data, to_data_struct,
  from_data_struct, get_cached,
  smape, sigmoid, softabs, ensure_converged,
  rjson, rjson_pretty, rjson_key

get_cached(make::Function, cache, key; max=nothing) = begin
  if max !== nothing && length(cache) >= max && !haskey(cache, key)
    key_drop = first(keys(cache))
    @warn "cache full, dropping entry" max key_drop
    delete!(cache, key_drop)
  end
  get!(make, cache, key)
end
smape(y, ŷ, ϵ = 1e-6) = abs(y - ŷ) / ((abs(y) + abs(ŷ) + ϵ) / 2)

@inline sigmoid(x) = 1.0 / (1.0 + exp(-x))
# @inline logit(x) = log(x / (1 - x))

export sigmoid_scaled, logit_scaled
@inline sigmoid_scaled(l::Real, c::Real, h::Real, z::Real; th::Real=2.0) = begin
  @assert l < c < h "c out of range"
  d = log((c - l) / (h - c))
  # Large asymmetry makes the probabilities of the larger intervals infinitesimally small,
  # can't be used in optimiser params transformation or bayesian inference.
  @assert abs(d) < th "asymmetry is too large: l=$(l), c=$(c), h=$(h)"
  l + (h - l) * sigmoid(d + z)
end
logit_scaled(l::Real, c::Real, h::Real, x::Real; th::Real=2.0) = begin
  @assert l < c < h "c out of range"
  d = log((c - l) / (h - c))
  # Large asymmetry makes the probabilities of the larger intervals infinitesimally small,
  # can't be used in optimiser params transformation or bayesian inference.
  @assert abs(d) < th "asymmetry is too large: l=$(l), c=$(c), h=$(h)"
  @assert l < x < h "x out of range"
  logit((x - l) / (h - l)) - d
end


softabs(x, ϵ) = sqrt(x*x + ϵ*ϵ) - ϵ

export softplus
# @inline softplus(x, α) = log1p(exp(α*x)) / α
@inline softplus(x, α) = begin
  y = α * x
  (max(y, 0) + log1p(exp(-abs(y)))) / α
end

export softmax
@inline softmax(a, b, α) = logaddexp(α*a, α*b)/α # log(exp(α a) + exp(α b))/α

export softmin
@inline softmin(a, b, α) = -logaddexp(-α*a, -α*b)/α # -log(exp(-α a) + exp(-α b))/α

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

# flatten(v::AbstractVector{<:AbstractVector}) = collect(Iterators.flatten(v))

export t_logpdf, t_logpdf_std

@inline t_logpdf(ν, x) = begin
  @assert ν > 1 "ν must be > 1"
  c = loggamma((ν+1)/2) - loggamma(ν/2) - 0.5*log(ν*pi)
  c - (ν+1)/2 * log1p(x^2/ν)
end

@inline t_logpdf(μ, scale, ν, x) = begin
  @assert scale > 0 "σ must be > 0"
  t_logpdf(ν, (x-μ)/scale) - log(scale)
end

@inline t_logpdf_std(μ, σ, ν, x) = begin
  @assert σ > 0 "σ must be > 0"
  @assert ν > 2 "ν must be > 2 for finite variance"
  scale = σ * sqrt((ν - 2) / ν)
  t_logpdf(ν, (x-μ)/scale) - log(scale)
end


round_any(x; opts...) = x
round_any(x::AbstractFloat; opts...) = round(x; opts...)
round_any(x::NamedTuple; opts...) = (; (k => round_any(v; opts...) for (k,v) in pairs(x))...)
round_any(x::NTuple{N, Any}; opts...) where {N} = ntuple(i -> round_any(x[i]; opts...), N)
round_any(x::AbstractArray; opts...) = round_any.(x; opts...)

Base.round(x::NamedTuple; opts...) = round_any(x; opts...)
Base.round(x::Tuple; opts...) = round_any(x; opts...)
Base.round(x::AbstractArray; opts...) = round_any(x; opts...)

export round4
round4(x) = round_any(x; digits=4)

export round6
round6(x) = round_any(x; digits=6)

Random.seed!(op::Function, seed::Integer) = begin
  state = copy(Random.default_rng())
  try
    Random.seed!(seed)
    op()
  finally
    copy!(Random.default_rng(), state)
  end
end

export skipnan, skipnothing
skipnan(x::AbstractArray) = Iterators.filter(!isnan, x)
skipnothing(x::AbstractArray) = Iterators.filter(!isnothing, x)

export say
say(msg::String) = begin run(`say $msg`); nothing end

export mean_meanad
mean_meanad(x::AbstractArray{T}) where {T<:Union{Missing,Real}} = begin
  present = skipmissing(x)
  c = mean(present); s = mean(abs.(present .- c))
  c, s
end

export to_mean_meanad
to_mean_meanad(x::AbstractArray{T}) where {T<:Union{Missing,Real}} = begin
  c, s = mean_meanad(x)
  (x .- c) ./ s, c, s
end

export nan_to_missing, missing_to_nan
nan_to_missing(x::AbstractArray{Float64}) = map(v -> isnan(v) ? missing : v, x)
missing_to_nan(x::AbstractArray{Union{Missing,Float64}}) = map(v -> ismissing(v) ? NaN : v, x)

export never
never(msg="Never") = throw(ErrorException(msg))

export fit_robust
fit_robust(::Type{Normal}, x; iqr=((0.1587, 0.5, 0.8413), 2)) = begin
  qs, m = iqr
  q = quantile(x, qs)
  μ, σ = q[2], (q[3] - q[1]) / m
  @assert σ > 0 "IQR-based σ must be > 0"
  Normal(μ, σ)
end

export symlink
symlink(src, dst; force::Bool=false) = begin
  force && (ispath(dst) || islink(dst)) && rm(dst; force=true)
  Base.Filesystem.symlink(src, dst)
end

run_cmd(
  cmd::Cmd, args::Union{Nothing,NamedTuple,String}=nothing; stdin::Union{Nothing,String}=nothing
) = begin
  if isa(args, String)
    cmd = `$cmd $args`
  elseif isa(args, NamedTuple)
    cmd = `$cmd $(["$(k)=$(v)" for (k,v) in pairs(args)]...)`
  end

  out, err = Pipe(), Pipe()
  proc = run(pipeline(
    cmd, stdin = stdin === nothing ? devnull : IOBuffer(stdin), stdout = out, stderr = err
  ), wait=false)
  close(out.in); close(err.in)
  stdout, stderr = read(out, String), read(err, String)
  wait(proc)

  proc.exitcode, stdout, stderr
end

export show_table
show_table(ds) = Main.vscodedisplay(ds)

export isgteapprox, islteapprox
isgteapprox(x, y; args...) = x > y || isapprox(x, y; args...)
islteapprox(x, y; args...) = x < y || isapprox(x, y; args...)

end
