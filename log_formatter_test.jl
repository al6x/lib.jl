include("./log_formatter.jl")

using Logging, Test

log_output(op) = begin
  io = IOBuffer()
  with_logger(TerminalLogger(; io, colors=false)) do
    op()
  end
  String(take!(io))
end

@testset "log_formatter" begin
  @test log_output(() -> begin @info "getting prices" log_id="MSFT" price=100 end) ==
    "  main | MSFT    getting prices {price: 100}\n"
  @test log_output(() -> begin @warn "no response" _log_id="MSFT" end) ==
    "W main | MSFT    no response\n"
  @test log_output(() -> @warn "config" (; now=1)) == "W main |         config {now: 1}\n"
  out = log_output() do
    try
      error("some error")
    catch e
      @error "cant get price" log_id="MSFT" exception=(e, catch_backtrace())
    end
  end
  @test startswith(out, "E main | MSFT    cant get price\n                 some error\n")
  @test occursin("Stacktrace:", out)

  empty!(disabled_logs)
  disable_logs!(:Main)
  @test log_output(() -> @info "hidden") == ""
  empty!(disabled_logs)
end
