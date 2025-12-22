@inline isfinite_dual(x::ForwardDiff.Dual) =
  isfinite(ForwardDiff.value(x)) && all(isfinite, ForwardDiff.partials(x))
@inline isfinite_dual(x::Number) = isfinite(x)

dual_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
dual_value(x::Number) = x
dual_value(x::AbstractArray) = map(dual_value, x)
dual_value(x::Tuple) = map(dual_value, x)
dual_value(x::NamedTuple) = NamedTuple{keys(x)}(map(dual_value, values(x)))
dual_value(x::Dict) = Dict(dual_value(k)=>dual_value(v) for (k,v) in x)
dual_value(x) = x  # fallback