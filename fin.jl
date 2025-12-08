StatsBase.meanad(x, center::Union{Nothing,Float64}=nothing) = begin
  present = skipmissing(x)
  center === nothing && (center = median(present))
  mean(abs.(present .- center))
end

calc_diff(op, x::Vector{<:Union{Missing,Float64}}) = begin
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
  window == 1 && return vs
  r = Vector{Union{Missing, Float64}}(missing, length(vs))
  for t in 1:length(vs)
    vs[t] === missing && continue
    present = skipmissing(vs[max((t - window + 1), 1):t])
    count(_ -> true, present) ≥ min || continue
    r[t] = op(present)
  end
  r
end

rperiod(rs::Vector; period::Int, overlap=false) = begin
  @assert period > 0
  n = length(rs)
  rsp::Vector{Union{Missing,Float64}} = [begin
    t2 = t + period - 1
    if ismissing(rs[t]) || t2 > n
      missing
    else
      count, sum = 0, 0.0; for i in t:t2-1
        r = rs[i]; !ismissing(r) && (sum += r; count += 1)
      end

      rlast = rs[t2]
      # If last return at t2 is non trading day using next trading day,
      # but with the gap no more than 2 days.
      max_last_gap = 2
      if ismissing(rlast)
        tlast = t2
        while ismissing(rlast) && tlast < min(n, t2 + max_last_gap)
          tlast += 1; rlast = rs[tlast]
        end
      end
      # Return missing if insufficient data or last return is missing
      ismissing(rlast) || (count + 1 < period/2) ? missing : sum + rlast
    end
  end for t in 1:n]

  if overlap == false
    i = 1; while i <= n
      if ismissing(rsp[i]) i += 1 else
        rsp[i+1:min(i+period-1, n)] .= missing
        i += period
      end
    end
  end

  rsp
end

# Output variance, not scale
yang_zhang_rv(input::NamedTuple) = begin
  (; o, h, l, c) = input
  @assert length(o) == length(h) == length(l) == length(c)

  idx = [i for i in eachindex(o) if !ismissing(o[i])]

  n = length(idx)
  n <= 2 && return missing  # need at least 2 days to get overnight + variance

  m = n - 1  # number of usable days (because overnight uses C_{t-1})
  oret = Vector{Float64}(undef, m)
  cret = Vector{Float64}(undef, m)
  rs   = Vector{Float64}(undef, m)

  for j in 2:n
    i   = idx[j]
    ip  = idx[j-1]

    Oi, Hi, Li, Ci = o[i], h[i], l[i], c[i]
    Cprev          = c[ip]

    oret[j-1] = log(Oi / Cprev)
    cret[j-1] = log(Ci / Oi)

    r_ho = log(Hi / Oi)
    r_lo = log(Li / Oi)
    r_co = log(Ci / Oi)
    rs[j-1] = r_ho * (r_ho - r_co) + r_lo * (r_lo - r_co)
  end

  N = m
  μo = mean(oret)
  μc = mean(cret)

  σo2  = sum((oret .- μo).^2) / (N - 1)
  σc2  = sum((cret .- μc).^2) / (N - 1)
  σrs2 = mean(rs)

  k = 0.34 / (1.34 + (N + 1) / (N - 1))

  σo2 + k * σc2 + (1 - k) * σrs2
end


garman_klass_rv(o::Float64, h::Float64, l::Float64, c::Float64) =
  0.5 * log(h / l)^2 - (2 * log(2) - 1) * log(c / o)^2

garman_klass_rv(input::NamedTuple) = begin
  (; o, h, l, c) = input; n = length(c)
  [ismissing(o[t]) ? missing : garman_klass_rv(o[t], h[t], l[t], c[t]) for t in 1:n]
end


yang_zhang_rv(c_prev::Float64, o::Float64, h::Float64, l::Float64, c::Float64) = begin
  u  = log(o / c_prev)        # overnight
  co = log(c / o)             # intraday

  r_ho = log(h / o)
  r_lo = log(l / o)
  r_co = log(c / o)
  rs   = r_ho * (r_ho - r_co) + r_lo * (r_lo - r_co)

  (u - co)^2 + 0.5 * rs
end

yang_zhang_rv(input::NamedTuple) = begin
  (; o, h, l, c) = input; n = length(c)
  rv = Vector{Union{Missing,Float64}}(missing, n)
  last = nothing  # index of last non-missing OHLC

  for t in 1:n
    ismissing(o[t]) && continue
    # c_prev_eff::Float64 # adjusted by gap as /sqrt(gap)

    c_prev_eff = if last === nothing
      o[t] # for first day use current open as previous close
    else
      gap = t - last
      if gap == 1
        c[last]
      else
        # gap > 1: scale overnight return by 1/sqrt(gap)
        c_prev = c[last]
        u_full = log(o[t] / c_prev)
        u_per  = u_full / sqrt(gap)
        o[t] / exp(u_per)
      end
    end

    rv[t] = yang_zhang_rv(c_prev_eff, o[t], h[t], l[t], c[t])
    last = t
  end
  rv
end