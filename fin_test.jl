# includet.(["./Lib.jl", "./fin.jl"])
using Test, Lib
import Fin

@testset "rperiod all present" begin
  msn = missing
  rs   = [ 1.0,  0.0, -1.0,  2.0,  3.0,  -1.0]

  rs1d = [ 0.0, -1.0,  2.0,  3.0, -1.0,   msn]
  ps1d = [ 1.0,  1.0,  1.0,  1.0,  1.0,   msn]

  rs2d = [-1.0,  1.0,  5.0,  2.0,  msn,   msn]
  ps2d = [ 2.0,  2.0,  2.0,  2.0,  msn,   msn]

  @test isequal(Fin.rperiod(rs; period=1, overlap=true), (rs1d, ps1d))
  @test isequal(Fin.rperiod(rs; period=2, overlap=true), (rs2d, ps2d))
end;

@testset "rperiod with missing" begin
  msn = missing
  rs   = [ 1.0,  0.0, -1.0,  msn,  2.0,  3.0,  msn,  msn, -1.0,  msn,  msn,  msn,  1.0]

  rs1d = [ 0.0, -1.0,  2.0,  msn,  3.0, -1.0,  msn,  msn,  msn,  msn,  msn,  msn,  msn]
  ps1d = [ 1.0,  1.0,  2.0,  msn,  1.0,  3.0,  msn,  msn,  msn,  msn,  msn,  msn,  msn]

  rs2d = [-1.0,  1.0,  2.0,  msn,  2.0, -1.0,  msn,  msn,  1.0,  msn,  msn,  msn,  msn]
  ps2d = [ 2.0,  3.0,  2.0,  msn,  4.0,  3.0,  msn,  msn,  4.0,  msn,  msn,  msn,  msn]

  @test isequal(Fin.rperiod(rs; period=1, overlap=true), (rs1d, ps1d))
  @test isequal(Fin.rperiod(rs; period=2, overlap=true), (rs2d, ps2d))
end;

@testset "rperiod all present no overlap" begin
  msn = missing
  rs   = [ 1.0,  0.0, -1.0,  2.0,  3.0, -1.0]

  rs1d = [ 0.0, -1.0,  2.0,  3.0, -1.0,  msn]
  ps1d = [ 1.0,  1.0,  1.0,  1.0,  1.0,  msn]

  rs2d = [-1.0,  msn,  5.0,  msn,  msn,   msn]
  ps2d = [ 2.0,  msn,  2.0,  msn,  msn,   msn]

  @test isequal(Fin.rperiod(rs; period=1, overlap=false), (rs1d, ps1d))
  @test isequal(Fin.rperiod(rs; period=2, overlap=false), (rs2d, ps2d))
end;