includet("./duals.jl")
includet("./lib.jl")
includet("./fin.jl")

module Garch

using Random, Statistics, Optim, ForwardDiff, SpecialFunctions
using ..Lib
# import StatsBase

error_message(e) = begin
  if e isa CompositeException
    e2 = first(e.exceptions)
    e2 isa TaskFailedException ? string(e2.task.exception) : string(e2)
  else
    string(e)
  end
end

rand_bounded(bounds, n) = begin
  lower, upper = unzip(bounds)
  [lower .+ rand(length(lower)) .* (upper .- lower) for _ in 1:n]
end

llh(
  M::NamedTuple, H::NamedTuple, R::Vector{RT},
  inputs::Vector{NTuple{N, Float64}}, r2s::Vector{NTuple{2, Float64}};
  weights::Union{Nothing,Vector{Float64}}=nothing
) where {N, RT<:Real} = begin
  Q = M.decode(H, R)
  n = length(inputs)
  llh, count = zero(RT), 0
  il = findfirst(v -> !isnan(v[1]), inputs); lh = il + H.warmup_d - 1
  state = M.init_state(Q, inputs[il])
  predict = M.predict
  for t in il:(n - H.period)
    it = inputs[t]; isnan(it[1]) && continue
    r2 = r2s[t]
    state, llh_t2 = predict(Q, state, it, r2)
    # @assert all(isfinite_dual.(values(state))) "state is not finite: $(dual_value(state))"

    if t > lh && !isnan(r2[1]) # ignoring during warmup or missing
      # @assert isfinite_dual(llh_t2) "llh is not finite"
      w = (weights === nothing ? 1.0 : weights[t])
      llh += w*llh_t2; count += 1
    end
  end
  llh, count
end;

predict(
  M::NamedTuple, inputs::Vector{NTuple{N, Float64}}; include_warmup=true
) where {N} = begin
  (; H, R) = M
  Q = M.decode(H, R)
  n = length(inputs)
  il = findfirst(v -> !isnan(v[1]), inputs); lh = il + H.warmup_d - 1
  state = M.init_state(Q, inputs[il])
  states = fill!(Vector{typeof(state)}(undef, n), ntuple(_ -> NaN, length(state)))
  llhs = fill!(Vector{Union{Missing, Float64}}(undef, n), missing)
  r2s = Main.rperiod_as_tuples(getindex.(inputs, 1) |> nan_to_missing; H.period, overlap=false)
  predict = M.predict
  for t in il:n
    it = inputs[t]; isnan(it[1]) && continue
    r2 = r2s[t]
    state, llh_t2 = predict(Q, state, it, r2)
    if include_warmup || t > lh
      states[t] = state
      !isnan(r2[1]) && (llhs[t] = llh_t2)
    end
  end
  fn = fieldnames(typeof(state))
  r = NamedTuple{fn}(ntuple(i -> getfield.(states, fn[i]) |> nan_to_missing, length(fn)))
  (; r..., llh=llhs)
end;

llh(
  M::NamedTuple, H::NamedTuple, R::Vector{RT},
  inputs::Vector{Vector{NTuple{N, Float64}}}, r2s::Vector{Vector{NTuple{2, Float64}}};
  weights::Union{Nothing,Vector{Vector{Float64}}}=nothing, parallel=false
) where {N, RT<:Real} = begin
  n = length(inputs)
  llhs, counts = Vector{RT}(undef, n), zeros(Int, n)

  if parallel
    Threads.@threads for i in eachindex(inputs)
      ws = weights === nothing ? nothing : weights[i]
      llhs[i], counts[i] = Garch.llh(M, H, R, inputs[i], r2s[i]; weights=ws)
    end
  else
    for i in eachindex(inputs)
      ws = weights === nothing ? nothing : weights[i]
      llhs[i], counts[i] = Garch.llh(M, H, R, inputs[i], r2s[i]; weights=ws)
    end
  end

  sum(llhs), sum(counts)
end;

fit(
  M::NamedTuple,
  H::Union{NamedTuple, Vector{<:NamedTuple}},
  R::Union{Vector{Float64},Vector{Vector{Float64}}},
  inputs::Union{Vector{NTuple{N, Float64}}, Vector{Vector{NTuple{N, Float64}}}};

  weights::Union{Nothing,Vector{Float64},Vector{Vector{Float64}}}=nothing,
  moments::Union{Nothing, NamedTuple}=nothing, tail_q=0.99,
  penalty=((H, R) -> 0.0),
  f_atol::Union{Nothing, Float64}=nothing, patience = 10, interrupted=()->false,
  debug=false, parallel=false, scale::Union{Nothing, Float64}=nothing,
) where {N} = begin
  inputs = isa(inputs, Vector{NTuple{N, Float64}}) ? [inputs] : inputs
  weights = isa(weights, Vector{Float64}) ? [weights] : weights
  Rs = isa(R, Vector{Float64}) ? [R] : R
  Hs = isa(H, NamedTuple) ? [H] : H

  scale = if scale !== nothing; scale else
    all_rs = vcat([getindex.(it, 1) |> skipnan |> collect for it in inputs]...)
    median(abs.(all_rs .- median(all_rs)))
  end
  @assert scale > 0 "scale must be positive"

  period, warmup_d = Hs[1].period, Hs[1].warmup_d
  @assert all(getproperty.(Hs, :period) .== period) "period must be same for all Hs"
  @assert all(getproperty.(Hs, :warmup_d) .== warmup_d) "warmup_d must be same for all Hs"

  # Target returns at t2
  r2s = [begin
    r2s = Main.rperiod_as_tuples(getindex.(it, 1) |> nan_to_missing; period, overlap=false)
    [r2s[2:end-1]; (NaN, NaN)]::Vector{NTuple{2, Float64}}
  end for it in inputs]

  # Inits, bounds, zs
  HRs = [(h, r) for h in Hs for r in Rs]
  lower, upper = unzip(M.bounds)

  zs = moments !== nothing ? moments.make_zs(H)::Vector{Float64} : nothing

  Loss(H::NamedTuple) = @inline loss(R::AbstractVector{RT}) where {RT<:Real} = begin
    llh, count = Garch.llh(M, H, R, inputs, r2s; weights, parallel)
    nllh = llh/count + log(scale)

    mloss = moments !== nothing ? moments.loss(H, Q, inputs, zs) : 0.0

    debug && let
      nllh_s = round(Main.dual_value(nllh); digits=6)
      R_s = round.(Main.dual_value(R); digits=6)
      Q_s = round(M.decode(H, Main.dual_value(R)); digits=6)
      m_s = moments !== nothing ? " M=$(round(Main.dual_value(mloss); digits=6))" : ""
      println("nllh=$nllh_s$m_s H=$H R=$R_s Q=$Q_s")
    end

    loss = -nllh + mloss
    loss + penalty(H, R)
  end

  # Stopping stagnant optimisation
  best, best_R, stagnant = Inf, nothing, 0
  callback(tr) = begin
    interrupted() && return true
    state = last(tr)
    if f_atol !== nothing
      l = state.value
      if best - l > f_atol
        best, best_R = l, copy(state.metadata["x"])
        stagnant = 0
      else
        stagnant += 1
      end
      stagnant > patience
    else
      false
    end
  end

  optimise(H, R) = begin
    options = Optim.Options(; callback, extended_trace=true, store_trace=true)
    r = optimize(Loss(H), lower, upper, R, Fminbox(LBFGS()), options; autodiff=:forward)
    # optimize(loss, lower, upper, R_init, Fminbox(LBFGS()))
    # optimize(loss, lower, upper, R_init, Fminbox(NelderMead()))
    R = if stagnant > patience
      @warn "optimization stopped due to stagnation"
      best_R
    else
      Optim.converged(r) || throw("not converged")
      r.minimizer
    end
    debug && @debug "optim done"
    (; H, R, loss=r.minimum)
  end

  results = if length(HRs) == 1
    H, R = HRs[1]
    [optimise(H, R)]
  else
    [try optimise(H, R) catch e
      @warn "optim failed: $(error_message(e))";
      # showerror(stdout, e, catch_backtrace());
      # bt = sprint(showerror, e, catch_backtrace())
      # !occursin("ForwardDiff.Dual", bt) && println(bt)
      missing
    end for (H, R) in HRs] |> skipmissing |> collect
  end

  # Result
  isempty(results) && error("can't fit")
  (; H, R) = results[argmin(getproperty.(results, :loss))]
  Q = M.decode(H, R)

  llh, count = Garch.llh(M, H, R, inputs, r2s; parallel)
  nllh = llh/count + log(scale)
  mloss = moments !== nothing ? moments.loss(H, Q, inputs, zs) : 0.0
  pnlt = penalty(H, R)

  # Tail llh
  tllh = [begin
    states = Garch.predict((; M..., R), it; include_warmup=false);
    rs = getindex.(it, 1)
    th = quantile(abs.(skipnan(rs)), tail_q)
    idx = [i for i in eachindex(rs) if !ismissing(rs[i]) && abs(rs[i]) > th]
    idx = idx[idx .> (H.warmup_d + findfirst(!isnan, rs))]
    mean(states.llh[idx])
  end for it in inputs] |> mean

  let
    nllh_s, llh_s = round(nllh; digits=4), round(llh; digits=4)
    R_s = round.(R; digits=6)
    m_s = moments !== nothing ? " M=$(round(mloss; digits=4))" : ""
    pnlt_s = pnlt > 0 ? " penalty=$(round(pnlt; digits=4))" : ""
    Q_s = round(Q; digits=6)
    tllh_s = round(tllh; digits=4)
    info = "nllh=$nllh_s$m_s$pnlt_s tllh=$tllh_s llh=$llh_s count=$count"
    @info("found $info\nH=$H\nR=$R_s\nQ=$Q_s")
  end

  (; H, R, Q, nllh, llh, tllh, count)
end;

@inline t_mean_abs(ν) = begin
  @assert ν > 1
  2*sqrt(ν-2)/(ν-1) *
    exp(SpecialFunctions.loggamma((ν+1)/2) - 0.5*log(pi) - SpecialFunctions.loggamma(ν/2))
end

# to_arrays(vs::AbstractVector{<:NamedTuple}) = begin
#   n = length(vs); @assert n > 0 "Input vector is empty"
#   proto = vs[1]; m = length(proto)
#   r = [Vector{Union{Missing,Float64}}(undef, n) for _ in 1:m]
#   for j in 1:m, i in 1:n; r[j][i] = vs[i][j] end
#   Tuple(r)
# end

end