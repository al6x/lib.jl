import JSON

rjson(v::Nothing) = "null"
rjson(v::Bool) = v ? "true" : "false"
rjson(v::Integer) = string(v)
rjson(v::AbstractFloat) = isfinite(v) ? string(v) : error("unsupported rjson number: v=$v")
rjson(v::Symbol) = rjson_string(String(v))
rjson(v::AbstractString) = rjson_string(v)
rjson(v::AbstractVector) = "[" * join(rjson.(v), ", ") * "]"
rjson(v::Tuple) = "[" * join(rjson.(v), ", ") * "]"
rjson(v::NamedTuple) = rjson_object(pairs(v))
rjson(v::AbstractDict) = rjson_object(sort(collect(pairs(v)); by=p -> string(p[1])))
rjson(v) = error("unsupported rjson value: type=$(typeof(v))")

rjson_object(pairs) = begin
  props = ["$(rjson_key(k)): $(rjson(v))" for (k, v) in pairs if v !== nothing]
  "{" * join(props, ", ") * "}"
end

rjson_pretty(v; indent=0, width=100) = rjson(v)
rjson_pretty(v::AbstractVector; indent=0, width=100) = begin
  isempty(v) && return "[]"
  flat = rjson(v); indent + length(flat) <= width && return flat
  body = join([spaces(indent + 2) * rjson_pretty(x; indent=indent + 2, width) for x in v],
    ",\n")
  "[\n$body\n$(spaces(indent))]"
end
rjson_pretty(v::Tuple; indent=0, width=100) = rjson_pretty(collect(v); indent, width)
rjson_pretty(v::NamedTuple; indent=0, width=100) = rjson_pretty_object(pairs(v); indent, width)
rjson_pretty(v::AbstractDict; indent=0, width=100) =
  rjson_pretty_object(sort(collect(pairs(v)); by=p -> string(p[1])); indent, width)

rjson_pretty_object(pairs; indent=0, width=100) = begin
  props = [(k, v) for (k, v) in pairs if v !== nothing]
  isempty(props) && return "{}"
  flat = rjson_object(props); indent + length(flat) <= width && return flat
  body = join([
    "$(spaces(indent + 2))$(rjson_key(k)): $(rjson_pretty(v; indent=indent + 2, width))"
    for (k, v) in props
  ], ",\n")
  "{\n$body\n$(spaces(indent))}"
end

spaces(n) = " "^n

rjson_key(k) = begin
  s = k isa Symbol ? String(k) : string(k)
  occursin(r"^[a-zA-Z0-9_]+$", s) ? s : JSON.json(s)
end

rjson_string(s::AbstractString) =
  rjson_bare_string(s) ? String(s) : JSON.json(s)

rjson_bare_string(s::AbstractString) =
  !isempty(s) &&
  !occursin(r"^(true|false|null|undefined)$", s) &&
  !occursin(r"^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$", s) &&
  occursin(r"^[a-zA-Z_][a-zA-Z0-9_./-]*$", s)
