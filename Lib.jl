module Lib

import Printf, Optim, CSV
using Statistics

export dedent, configure!, unzip, flatten, spread,
  smape, sigmoid, abs_soft

export ensure_converged

smape(y, ŷ, ϵ = 1e-6) = abs(y - ŷ) / ((abs(y) + abs(ŷ) + ϵ) / 2)

sigmoid(x) = 1.0 / (1.0 + exp(-x))

abs_soft(x; δ=1e-3) = δ^2 * (sqrt(1 + (x/δ)^2) - 1)

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

unzip(rows::AbstractVector{T}) where {T<:Union{Tuple,AbstractVector}} = begin
  isempty(rows) && error("empty input")
  N = length(first(rows))
  return ntuple(j -> getindex.(rows, j), N)
end

unzip(rows::Base.Generator) = unzip(collect(rows))

flatten(v::AbstractVector{<:AbstractVector{T}}) where {T} = collect(Iterators.flatten(v))

mutable struct LibConfig show_round::Int end
const config = Ref{Union{LibConfig, Nothing}}(nothing)

function configure!(; show_round=4)
  config[] = LibConfig(show_round)
end
configure!()

# function Base.show(io::IO, f::Float64)
#   Printf.@printf(io, "%.*f", config[].show_round, f)
# end

end