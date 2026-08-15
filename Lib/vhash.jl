import SHA, Dates

vhash(xs...; kwargs...) = begin
  acc = SHA.sha256_ctx()
  if isempty(kwargs)
    vhash!(acc, length(xs) == 1 ? xs[1] : xs)
  else
    vhash!(acc, xs)
    SHA.update!(acc, "kw{")
    for (k, v) in sort!(collect(pairs(kwargs)); by=first)
      vhash!(acc, string(k))
      vhash!(acc, v)
    end
    SHA.update!(acc, "}")
  end
  bytes2hex(SHA.digest!(acc))[1:12]
end

vhash!(acc, x::Nothing) = SHA.update!(acc, "n;")
vhash!(acc, x::Missing) = SHA.update!(acc, "m;")
vhash!(acc, x::Bool) = SHA.update!(acc, x ? "b1;" : "b0;")
vhash!(acc, x::Integer) = (SHA.update!(acc, "i"); put_s!(acc, string(x)))
vhash!(acc, x::AbstractFloat) = (SHA.update!(acc, "f"); put_s!(acc, repr(x)))
vhash!(acc, x::Symbol) = (SHA.update!(acc, "y"); vhash!(acc, string(x)))
vhash!(acc, x::Dates.TimeType) = (SHA.update!(acc, "d"); vhash!(acc, string(x)))

vhash!(acc, x::AbstractString) = begin
  SHA.update!(acc, "s")
  put_s!(acc, string(sizeof(x)))
  SHA.update!(acc, x)
  SHA.update!(acc, ";")
end

vhash!(acc, x::AbstractVector{UInt8}) = begin
  SHA.update!(acc, "u")
  put_s!(acc, string(length(x)))
  SHA.update!(acc, x)
end

vhash!(acc, x::Tuple) = begin
  SHA.update!(acc, "t[")
  foreach(v -> vhash!(acc, v), x)
  SHA.update!(acc, "]")
end

vhash!(acc, x::AbstractVector) = begin
  SHA.update!(acc, "v[")
  foreach(v -> vhash!(acc, v), x)
  SHA.update!(acc, "]")
end

vhash!(acc, x::NamedTuple) = begin
  SHA.update!(acc, "nt{")
  for k in keys(x)
    vhash!(acc, string(k))
    vhash!(acc, getfield(x, k))
  end
  SHA.update!(acc, "}")
end

vhash!(acc, x::AbstractDict) = begin
  SHA.update!(acc, "dict{")
  for k in sort!(collect(keys(x)); by=repr)
    vhash!(acc, k)
    vhash!(acc, x[k])
  end
  SHA.update!(acc, "}")
end

vhash!(acc, x) = error("vhash unsupported type: $(typeof(x))")

put_s!(acc, s::AbstractString) = begin
  SHA.update!(acc, s)
  SHA.update!(acc, ";")
end
