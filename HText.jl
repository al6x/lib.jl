module HText

using ..Lib
import JSON

# Helpers ------------------------------------------------------------------------------------------
space_indent2(str) = join("  " .* split(str, '\n'), "\n")

const greek_to_latin2 = [
  "α"=>"a","β"=>"b","γ"=>"g","δ"=>"d","ε"=>"e","ζ"=>"z","η"=>"e","θ"=>"th","ι"=>"i",
  "κ"=>"k","λ"=>"l", "μ"=>"u","ν"=>"n","ξ"=>"x","ο"=>"o","π"=>"p","ρ"=>"r",
  "σ"=>"s","ς"=>"s","τ"=>"t","υ"=>"y", "φ"=>"f","ϕ"=>"f","χ"=>"ch","ψ"=>"ps","ω"=>"o",
  "ϑ"=>"th","ϰ"=>"k","ϱ"=>"r","ϲ"=>"s","ϵ"=>"e"
];

safe_name(s::AbstractString) =
  s |> lowercase |>
  x -> replace(x, greek_to_latin2...) |>
  x -> replace(x, r"[^a-z0-9]" => "-") |>
  x -> replace(x, r"-+" => "-") |>
  x -> strip(x, ['-']);

# Doc ----------------------------------------------------------------------------------------------
export Doc, write, text, code, clear, asset_path, save_asset, text_block, @bind_doc

struct Doc path end

Doc(; path::AbstractString) = Doc(path)

clear(d::Doc) = isfile(d.path) && rm(d.path)

Base.write(d::Doc, text::AbstractString) = begin
  mkpath(dirname(d.path))
  open(d.path, "a") do io; write(io, text) end
  nothing
end

text_block(d; props...) = begin
  props_s = join(["$k: $(JSON.json(v))" for (k, v) in props if v !== nothing], ", ")
  isempty(props_s) && return
  write(d, "-- {$props_s}\n\n")
  nothing
end

text(d::Doc, text::AbstractString; id=nothing, tags=nothing, tab=nothing) = begin
  text_block(d; id, tags, tab)

  text = text |> dedent |> rstrip
  write(d, text * "\n\n")
end

code(d::Doc, code::AbstractString; lang=nothing, id=nothing, tags=nothing, tab=nothing) = begin
  text_block(d; id, tags, tab)

  lang_s = lang === nothing ? "" : " $lang"
  write(d, "```$lang_s\n$(dedent(code))\n```\n\n")
end

asset_path(d::Doc, fname) = joinpath(splitext(d.path)[1], fname);

save_asset(d::Doc, name::AbstractString, obj; id=nothing, tags=nothing, tab=nothing) = begin
  fname = "$(safe_name(name)).png"
  path = asset_path(d, fname)
  mkpath(dirname(path))

  mod = nameof(parentmodule(typeof(obj)))
  if mod == :Plots
    getfield(parentmodule(typeof(obj)), :savefig)(obj, path)
  elseif mod == :VegaLite
    getfield(parentmodule(typeof(obj)), :save)(path, obj)
  elseif mod in (:Makie, :GLMakie, :CairoMakie, :WGLMakie)
    getfield(parentmodule(typeof(obj)), :save)(path, obj)
  elseif obj isa AbstractString
    open(path, "w") do io; write(io, obj) end
  else
    error("Unsupported asset type: $(typeof(obj))")
  end

  text(d, name; id, tags, tab)
  text(d, "![$name](this/$fname)")
  path
end

save_asset(d::Doc, title_obj::Tuple{AbstractString,Any}; props...) = save_asset(
  d, title_obj[1], title_obj[2]; props...)

macro bind_doc(d)
  quote
    htext_doc = $d

    HText.text(text::AbstractString; props...) = HText.text(htext_doc, text; props...)
    HText.code(code::AbstractString; props...) = HText.code(htext_doc, code; props...)
    HText.text_block(; props...) = HText.text_block(htext_doc; props...)
    HText.save_asset(name::AbstractString, obj; props...) = HText.save_asset(
      htext_doc, name, obj; props...)
    HText.save_asset(title_obj::Tuple{AbstractString,Any}; props...) = HText.save_asset(
      htext_doc, title_obj; props...)

    # String literals
    macro text_str(s) :(text($(Meta.parse("\"$s\"")))) end
    macro code_str(s) :(code($(Meta.parse("\"$s\"")))) end

    nothing
  end |> esc
end

end