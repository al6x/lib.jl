using Logging

const disabled_logs = Set{String}()

disable_logs!(args...) = begin
  for arg in args; push!(disabled_logs, string(arg)) end
  nothing
end

struct TerminalLogger <: AbstractLogger
  io
  min_level::LogLevel
  colors::Bool
  disabled::Set{String}
end

TerminalLogger(;
  io=stderr, min_level::LogLevel=Info, colors::Bool=!haskey(ENV, "NO_COLOR"),
  disabled::Set{String}=disabled_from_env()
) = TerminalLogger(io, min_level, colors, disabled)

install_logger!(; args...) = global_logger(TerminalLogger(; args...))

Logging.min_enabled_level(log::TerminalLogger) = log.min_level
Logging.catch_exceptions(::TerminalLogger) = false

Logging.shouldlog(
  log::TerminalLogger, level, _module, group, id
) = level >= log.min_level && !disabled(log, level, _module)

Logging.handle_message(
  log::TerminalLogger, level, message, _module, group, id, file, line; kwargs...
) = begin
  pairs = log_pairs(kwargs)
  id_i = findfirst(p -> p[1] in (:log_id, :_log_id), pairs)
  log_id = id_i === nothing ? "" : string(pairs[id_i][2])
  id_i !== nothing && deleteat!(pairs, id_i)

  main = "$(level_prefix(level)) $(format_module(_module)) | $(rpad(log_id, 7)[1:7]) $(message)"
  print(log.io, color(log, main, level_color(level)))
  if is_exception_log(pairs)
    println(log.io)
    print_exception(log, only(pairs)[2])
  else
    isempty(pairs) || print(log.io, color(log, " $(format_data(pairs))", :grey))
    println(log.io)
  end
  nothing
end

is_exception_log(pairs) = length(pairs) == 1 && only(pairs)[1] == :exception
print_exception(log::TerminalLogger, e::Exception) = begin
  println(log.io, color(log, indent(sprint(showerror, e), 17), :red))
end
print_exception(log::TerminalLogger, ex) = begin
  e, bt = ex
  println(log.io, color(log, indent(sprint(showerror, e), 17), :red))
  println(log.io, color(log, indent(sprint(Base.show_backtrace, bt), 17), :grey))
end

log_pairs(kwargs) = begin
  ps = collect(kwargs)
  length(ps) == 1 || return ps
  k, v = only(ps)
  (v isa NamedTuple && startswith(string(k), "(;")) ? collect(Base.pairs(v)) : ps
end

disabled(log::TerminalLogger, level, _module) = begin
  mod = string(_module)
  short = module_name(_module)
  lvl = lowercase(string(level))
  any(x -> x in log.disabled || x in disabled_logs, (mod, short, lvl, "$mod.$lvl", "$short.$lvl"))
end

disabled_from_env() = begin
  s = get(ENV, "disable_logs", "")
  Set(string(x) for x in strip.(split(s, ",")) if !isempty(x))
end

module_name(_module) = last(split(string(_module), "."))
format_module(_module) = lowercase(rpad(module_name(_module), 4)[1:4])

level_prefix(level) = level == Debug || level == Info ? " " :
  level == Warn ? "W" : level == Error ? "E" : error("unknown log level: level=$level")
level_color(level) = level == Debug ? :grey : level == Info ? :black :
  level == Warn ? :yellow : level == Error ? :red : error("unknown log level: level=$level")

format_data(pairs) = "{" * join(("$(k): $(format_value(v))" for (k, v) in pairs), ", ") * "}"
format_value(v::AbstractString) = repr(v)
format_value(v::Symbol) = string(v)
format_value(v) = repr(v)
indent(s, n) = join((" "^n * line for line in split(s, '\n')), "\n")

const ansi_codes = Dict(:grey => "90", :black => "30", :yellow => "33", :red => "31")
color(log::TerminalLogger, s, c::Symbol) = log.colors ? "\e[$(ansi_codes[c])m$s\e[0m" : s
