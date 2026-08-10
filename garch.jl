module Garch

include("duals.jl")

using Random, Statistics, Optim, ForwardDiff, SpecialFunctions, Lib, ADTypes
import Fin

# Protocol
function predict end
function transform end
function init_params end
function init_state end
function bounds end
function make_inputs end

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

predict(
  M, H::NamedTuple, R::Vector{RT}, inputs::Vector{NTuple{N, Float64}}; include_warmup=true
) where {N, RT<:Real} = begin
  # (; H, R) = M
  Q, D = transform(M, H, R)
  n = length(inputs)
  il = findfirst(v -> !isnan(v[1]), inputs); lh = il + H.warmup_d - 1
  state = init_state(M, Q, D, inputs[il])
  states = fill!(Vector{typeof(state)}(undef, n), ntuple(_ -> NaN, length(state)))
  llhs = fill!(Vector{Union{Missing, Float64}}(undef, n), missing)
  # Target returns at t2
  r2s = rperiod_as_tuples(getindex.(inputs, 1) |> nan_to_missing; H.period, overlap=false)

  for t in il:n
    it = inputs[t]; isnan(it[1]) && continue
    r2 = r2s[t]
    state, llh_t2 = predict(M, Q, D, state, it, r2)
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
  M, H::NamedTuple, R::Vector{RT},
  inputs::Vector{NTuple{N, Float64}}, r2s::Vector{NTuple{2, Float64}};
  weights::Union{Nothing,Vector{Float64}}=nothing
) where {N, RT<:Real} = begin
  Q, D = transform(M, H, R)
  n = length(inputs)

  llhw, countw, llh, count = zero(RT), 0.0, zero(RT), 0
  il = findfirst(v -> !isnan(v[1]), inputs); lh = il + H.warmup_d - 1
  state = init_state(M, Q, D, inputs[il])
  # predict = M.predict
  # for t in il:(n - H.period)
  for t in il:n
    it = inputs[t]; isnan(it[1]) && continue
    r2 = r2s[t]
    state, llh_t2 = predict(M, Q, D, state, it, r2)
    # @assert all(isfinite_dual.(values(state))) "state is not finite: $(dual_value(state))"

    if t > lh && !isnan(r2[1]) # ignoring during warmup or missing
      # @assert isfinite_dual(llh_t2) "llh is not finite"
      w = (weights === nothing ? 1.0 : weights[t])
      llhw += w*llh_t2; countw += w
      llh += llh_t2; count += 1
    end
  end
  llhw, countw, llh, count
end;

fit(
  M,
  H::Union{NamedTuple, Vector{<:NamedTuple}},
  R::Union{Vector{Float64},Vector{Vector{Float64}}},
  inputs::Union{Vector{NTuple{N, Float64}}, Vector{Vector{NTuple{N, Float64}}}};

  weights::Union{Nothing,Vector{Float64},Vector{Vector{Float64}}}=nothing,
  # moments::Union{Nothing, NamedTuple}=nothing,
  tail_q=0.99,
  penalty=((H, R) -> 0.0),
  f_atol::Union{Nothing, Float64}=nothing, patience = 10, interrupted=()->false,
  debug=false, info=false, parallel=false
) where {N} = begin
  inputs = isa(inputs, Vector{NTuple{N, Float64}}) ? [inputs] : inputs
  weights = isa(weights, Vector{Float64}) ? [weights] : weights
  Rs = isa(R, Vector{Float64}) ? [R] : R
  Hs = isa(H, NamedTuple) ? [H] : H

  scalef(x) = mean(abs.(x .- mean(x)))

  # Scales could be different across inputs.
  scales = [scalef(getindex.(it, 1) |> skipnan |> collect) for it in inputs]

  period, warmup_d = Hs[1].period, Hs[1].warmup_d
  @assert all(getproperty.(Hs, :period) .== period) "period must be same for all Hs"
  @assert all(getproperty.(Hs, :warmup_d) .== warmup_d) "warmup_d must be same for all Hs"

  # Target returns at t2 put at index t `r2s[t] = sum(r[t+1 : t+period])`
  r2s = [rperiod_as_tuples(getindex.(it, 1) |> nan_to_missing; period, overlap=false)
    for it in inputs]

  # Inits, bounds, zs
  HRs = [(h, r) for h in Hs for r in Rs]
  lower, upper = unzip(init_params(M).bounds)

  # zs = moments !== nothing ? moments.make_zs(H)::Vector{Float64} : nothing

  fit_counts = nothing
  Loss(H::NamedTuple) = @inline loss(R::AbstractVector{RT}) where {RT<:Real} = begin
    n = length(inputs)
    # llhw, countw, llh, count
    llhsw, countsw = Vector{RT}(undef, n), Vector{Float64}(undef, n)
    llhs, counts = Vector{RT}(undef, n), Vector{Int}(undef, n)

    if parallel
      Threads.@threads for i in eachindex(inputs)
        ws = weights === nothing ? nothing : weights[i]
        llhsw[i], countsw[i], llhs[i], counts[i] = Garch.llh(M, H, R, inputs[i], r2s[i]; weights=ws)
      end
    else
      for i in eachindex(inputs)
        ws = weights === nothing ? nothing : weights[i]
        llhsw[i], countsw[i], llhs[i], counts[i] = Garch.llh(M, H, R, inputs[i], r2s[i]; weights=ws)
      end
    end
    fit_counts = counts

    # Series may have different scales and length
    nllhw = sum(llhsw .+ countsw .* log.(scales)) / sum(countsw)


    # mloss = moments !== nothing ? moments.loss(H, Q, inputs, zs) : 0.0
    mloss = 0.0

    debug && let
      # Series may have different scales and length
      nllh = sum(llhs .+ counts .* log.(scales)) / sum(counts)

      nllh_s = round(dual_value(nllh); digits=6)
      R_s = round.(dual_value(R); digits=6)
      Q_s = round(M.transform(H, dual_value(R))[1]; digits=6)
      # m_s = moments !== nothing ? " M=$(round(dual_value(mloss); digits=6))" : ""
      # println("nllh=$nllh_s$m_s H=$H R=$R_s Q=$Q_s")
      println("nllh=$nllh_s H=$H R=$R_s Q=$Q_s")
    end

    loss = -nllhw + mloss
    loss + penalty(H, R)
  end

  optimise(H, R) = begin
    # Stopping stagnant optimisation
    best_loss, best_R, loss_progress, stagnant = Inf, nothing, Inf, 0
    callback(tr) = begin
      interrupted() && return true
      f_atol === nothing && return false

      loss = tr.f_x

      if loss < best_loss
        best_loss, best_R = loss, copy(tr.x)
      end

      if loss < loss_progress - f_atol
        loss_progress = loss
        stagnant = 0
      else
        stagnant += 1
      end
      stagnant > patience
    end

    options = Optim.Options(; callback, extended_trace=true, store_trace=true)
    r = optimize(Loss(H), lower, upper, R, Fminbox(LBFGS()), options;
      autodiff=ADTypes.AutoForwardDiff())
    # optimize(loss, lower, upper, R_init, Fminbox(LBFGS()))
    # optimize(loss, lower, upper, R_init, Fminbox(NelderMead()))
    R, loss = if !isnothing(f_atol) && (stagnant > patience)
      info && @info "optimization stopped due to stagnation"
      best_R, best_loss
    else
      Optim.converged(r) || throw("not converged")
      r.minimizer, r.minimum
    end
    debug && @debug "optim done"
    (; H, R, loss)
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
  Q = transform(M, H, R)[1]

  # Info
  let
    nllh, ntllh, llh, count = let
      llhs, counts, tllhs, tcounts = [begin
        states = Garch.predict(M, H, R, it; include_warmup=false);
        llhs = states.llh
        llhsp = llhs |> skipmissing |> collect

        # Tail
        rs = getindex.(it, 1)
        th = quantile(abs.(skipnan(rs)), tail_q)
        tidx = [i for i in eachindex(rs) if !isnan(rs[i]) && abs(rs[i]) > th]
        tllhs = llhs[tidx]
        # tscale = scalef(rs[tidx] |> skipnan |> collect)
        tllhsp = tllhs |> skipmissing |> collect

        sum(llhsp), length(llhsp), sum(tllhsp), length(tllhsp)
      end for it in inputs] |> unzip

      # Counts depend only on inputs and warmup and should be constant during fit and predict.
      @assert fit_counts == counts "Counts during fit and predict must match"

      # Series may have different scales and length
      nllh = sum(llhs .+ counts .* log.(scales)) / sum(counts)
      ntllh = sum(tllhs .+ tcounts .* log.(scales)) / sum(tcounts)

      nllh, ntllh, sum(llhs), sum(counts)
    end

    # mloss = moments !== nothing ? moments.loss(H, Q, inputs, zs) : 0.0
    pnlt = penalty(H, R)

    let
      nllh_s, llh_s = round(nllh; digits=4), round(llh; digits=4)
      R_s = round.(R; digits=6)
      # m_s = moments !== nothing ? " M=$(round(mloss; digits=4))" : ""
      pnlt_s = pnlt > 0 ? " penalty=$(round(pnlt; digits=4))" : ""
      Q_s = round(Q; digits=6)
      ntllh_s = round(ntllh; digits=4)
      # info = "nllh=$nllh_s$m_s$pnlt_s tllh=$ntllh_s llh=$llh_s count=$count"
      msg = "nllh=$nllh_s$pnlt_s tllh=$ntllh_s llh=$llh_s count=$count"
      info && @info("found $msg\nH=$H\nR=$R_s\nQ=$Q_s")
    end

    (; H, R, Q, nllh, ntllh, llh, count)
  end
end;

@inline t_mean_abs(ν) = begin
  @assert ν > 2.0
  2*sqrt(ν-2)/(ν-1) *
    exp(SpecialFunctions.loggamma((ν+1)/2) - 0.5*log(pi) - SpecialFunctions.loggamma(ν/2))
end

rperiod_as_tuples(rs::Vector{<:Union{Missing,Float64}}; period::Int, overlap=false) = begin
  r2s, p2s = Fin.rperiod(rs; period, overlap)
  [(r2, p2) for (r2, p2) in zip(r2s |> missing_to_nan, float.(p2s) |> missing_to_nan)]
end

end
