include.(["./Lib.jl", "./fin.jl"])
using Test, .Lib

@testset "rperiod all present" begin
  msn = missing
  rs   = [ 1.0,  0.0, -1.0,  2.0,  3.0,  -1.0]
  rs2d = [ 1.0, -1.0,  1.0,  5.0,  2.0,   msn]

  @test isequal(Lib.rperiod(rs; period=1, overlap=true), rs)
  @test isequal(Lib.rperiod(rs; period=2, overlap=true), rs2d)

  # Also testing with a simple cumsum based implementation, just in case
  r2cumsum_with_overlap(rs; period) = begin
    cs = cumsum(rs)
    cs[period:end] .- [0.0; cs[1:end-period]]
  end

  @test isequal(r2cumsum_with_overlap(rs; period=1), rs)
  @test isequal(r2cumsum_with_overlap(rs; period=2), rs2d |> skipmissing |> collect)
end;

@testset "rperiod with missing" begin
  msn = missing
  rs   = [ 1.0,  0.0, -1.0,  msn,  2.0,  3.0,  msn,  msn, -1.0,  msn,  msn,  msn,  1.0]
  rs2d = [ 1.0, -1.0,  1.0,  msn,  5.0,  2.0,  msn,  msn,  msn,  msn,  msn,  msn,  msn]

  @test isequal(Lib.rperiod(rs; period=1, overlap=true), rs)
  @test isequal(Lib.rperiod(rs; period=2, overlap=true), rs2d)
end;

@testset "rperiod all present no overlap" begin
  msn = missing
  rs   = [ 1.0,  0.0, -1.0,  2.0,  3.0,  -1.0]
  rs2d = [ 1.0,  msn,  1.0,  msn,  2.0,   msn]

  @test isequal(Lib.rperiod(rs; period=1, overlap=false), rs)
  @test isequal(Lib.rperiod(rs; period=2, overlap=false), rs2d)

  r2cumsum_no_overlap(rs; period) = begin
    cs = cumsum(rs)
    out = cs[period:period:end]
    out .- [0.0; out[1:end-1]]
  end
  @test isequal(r2cumsum_no_overlap(rs; period=1), rs)
  @test isequal(r2cumsum_no_overlap(rs; period=2), rs2d |> skipmissing |> collect)
end;