module NetEval

# Codex: user runs `import NetEval; NetEval.start()`. Read tmp/neteval.json, then send Julia
# code to the TCP port with `nc`, ending with a line containing __NETEVAL_END__.

using Sockets

export start, stop

const server = Ref{Union{Nothing,Sockets.TCPServer}}(nothing)
const task = Ref{Union{Nothing,Task}}(nothing)
const port = Ref(0)
const end_marker = "__NETEVAL_END__"

json(x) = "\"" * replace(string(x),
  "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t") * "\""
json(d::NamedTuple) = "{" * join(["\"$k\":" * json(v) for (k, v) in pairs(d)], ",") * "}"
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)

eval_code(code) = begin
  out_path, out = mktemp(); err_path, errout = mktemp()
  try
    isdefined(Main, :Revise) && Core.eval(Main, :(Revise.revise()))
    result = redirect_stdout(out) do
      redirect_stderr(errout) do
        Core.eval(Main, Meta.parse("begin\n$code\nend"))
      end
    end
    close(out); close(errout)
    (; ok=true, result=sprint(show, MIME"text/plain"(), result),
      stdout=read(out_path, String), stderr=read(err_path, String))
  catch e
    try close(out) catch _ end
    try close(errout) catch _ end
    (; ok=false, error=sprint(showerror, e, catch_backtrace()),
      stdout=read(out_path, String), stderr=read(err_path, String))
  finally
    rm(out_path; force=true); rm(err_path; force=true)
  end
end

handle(sock) = try
  lines = String[]
  while true
    line = readline(sock; keep=true)
    chomp(line) == end_marker && break
    push!(lines, line)
  end
  write(sock, json(eval_code(join(lines))))
catch e
  write(sock, json((; ok=false, error=sprint(showerror, e, catch_backtrace()))))
finally
  close(sock)
end

start(; host=ip"127.0.0.1", port_=parse(Int, get(ENV, "NETEVAL_PORT", "8765"))) = begin
  !isnothing(server[]) && return (; host=string(host), port=port[])
  port[] = port_
  server[] = listen(host, port_)
  isdir("tmp") && write("tmp/neteval.json", json((; host=string(host), port=port_)))
  task[] = @async while true
    try
      sock = accept(server[])
      @async handle(sock)
    catch e
      isnothing(server[]) && break
      rethrow(e)
    end
  end
  (; host=string(host), port=port_)
end

stop() = begin
  s = server[]; server[] = nothing
  isnothing(s) || close(s)
  server[] = nothing; task[] = nothing; port[] = 0
  nothing
end

function __init__()
  get(ENV, "NETEVAL", "") == "1" && start()
end

end
