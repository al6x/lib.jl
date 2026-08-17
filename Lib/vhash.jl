import SHA, Dates

vhash(xs...; kwargs...) = begin
  acc = SHA.SHA256_CTX()
  if isempty(kwargs)
    vhash!(acc, length(xs) == 1 ? xs[1] : xs)
  else
    vhash!(acc, xs)
    vhash_update!(acc, "kw{")
    for (k, v) in sort!(collect(pairs(kwargs)); by=first)
      vhash!(acc, string(k))
      vhash!(acc, v)
    end
    vhash_update!(acc, "}")
  end
  bytes2hex(SHA.digest!(acc))[1:12]
end

vhash_update!(acc, x::AbstractString) = SHA.update!(acc, codeunits(x))
vhash_update!(acc, x::AbstractVector{UInt8}) = SHA.update!(acc, x)

vhash!(acc, x::Nothing) = vhash_update!(acc, "n;")
vhash!(acc, x::Missing) = vhash_update!(acc, "m;")
vhash!(acc, x::Bool) = vhash_update!(acc, x ? "b1;" : "b0;")
vhash!(acc, x::Integer) = (vhash_update!(acc, "i"); put_s!(acc, string(x)))
vhash!(acc, x::AbstractFloat) = (vhash_update!(acc, "f"); put_s!(acc, repr(x)))
vhash!(acc, x::Symbol) = (vhash_update!(acc, "y"); vhash!(acc, string(x)))
vhash!(acc, x::Dates.TimeType) = (vhash_update!(acc, "d"); vhash!(acc, string(x)))

vhash!(acc, x::AbstractString) = begin
  vhash_update!(acc, "s")
  put_s!(acc, string(sizeof(x)))
  vhash_update!(acc, x)
  vhash_update!(acc, ";")
end

vhash!(acc, x::AbstractVector{UInt8}) = begin
  vhash_update!(acc, "u")
  put_s!(acc, string(length(x)))
  vhash_update!(acc, x)
end

vhash!(acc, x::Tuple) = begin
  vhash_update!(acc, "t[")
  foreach(v -> vhash!(acc, v), x)
  vhash_update!(acc, "]")
end

vhash!(acc, x::AbstractVector) = begin
  vhash_update!(acc, "v[")
  foreach(v -> vhash!(acc, v), x)
  vhash_update!(acc, "]")
end

vhash!(acc, x::NamedTuple) = begin
  vhash_update!(acc, "nt{")
  for k in keys(x)
    vhash!(acc, string(k))
    vhash!(acc, getfield(x, k))
  end
  vhash_update!(acc, "}")
end

vhash!(acc, x::AbstractDict) = begin
  vhash_update!(acc, "dict{")
  for k in sort!(collect(keys(x)); by=repr)
    vhash!(acc, k)
    vhash!(acc, x[k])
  end
  vhash_update!(acc, "}")
end

vhash!(acc, x) = error("vhash unsupported type: $(typeof(x))")

put_s!(acc, s::AbstractString) = begin
  vhash_update!(acc, s)
  vhash_update!(acc, ";")
end
