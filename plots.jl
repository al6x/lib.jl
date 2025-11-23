using JSON, VegaLite

# redblue, coolwarm, icefire
plot_xyc_by(
  title, ds; x, y, y2=nothing, color=nothing, by=nothing, detail=nothing,
  xdomain=nothing, ydomain=nothing, palette="viridis",
  xscale=nothing, yscale=nothing, pointsize=30, mark=:line_with_points, mark2=:line_with_points,
  columns = 3,
  width=1024, ratio=1.5
) = begin
  x isa Tuple{Any} && ((x,) = x)
  x isa Tuple{Any, Any} && ((x, xscale) = x)
  x isa Tuple{Any, Any, Any} && ((x, xscale, xdomain) = x)

  y isa Tuple{Any} && ((y,) = y)
  y isa Tuple{Any, Any} && ((y, yscale) = y)
  y isa Tuple{Any, Any, Any} && ((y, yscale, ydomain) = y)

  xscale === nothing && (xscale = "linear")
  yscale === nothing && (yscale = "linear")

  tparts = ["$title x=$x, y=$y"]
  color  !== nothing && push!(tparts, ", color=$color")
  detail !== nothing && push!(tparts, "($detail)")
  y2     !== nothing && push!(tparts, ", dashed=$y2")
  by     !== nothing && push!(tparts, " by=$by")
  ftitle = join(tparts, "")

  xscale_props = xdomain === nothing ? (type=xscale,) : (type=xscale, domain=xdomain)
  yscale_props = ydomain === nothing ? (type=yscale,) : (type=yscale, domain=ydomain)

  color_props = color === nothing ? (;) :
    (; color=(field=color, type=:ordinal, scale=(scheme=palette, reverse=true)))
  detail_props = detail === nothing ? (;) :
    (; detail=(field=detail, type=:ordinal))

  # reducing point size for facets
  npointsize = by === nothing ? pointsize : round(Int, pointsize/columns)
  poinsize_props = mark in (:point, :circle) ? (size=(value=npointsize,),) : (;)

  # y1
  layers = []
  mark_props1 =
    mark == :line             ? (type=:line, clip=true) :
    mark == :line_with_points ? (type=:line, clip=true, point=true) :
    mark == :circle           ? (type=:circle, clip=true) :
    (type=mark, clip=true,)
  encoding1 = (
    x = (field=x, type=:quantitative, scale=xscale_props),
    y = (field=y, type=:quantitative, scale=yscale_props),
    color_props...,
    detail_props...,
    poinsize_props...
  )
  push!(layers, (mark=mark_props1, encoding=encoding1,))

  if y2 !== nothing
    mark_props2 =
      mark2 == :line ?             (type=:line, clip=true, strokeDash=[4,4]) :
      mark2 == :line_with_points ? (type=:line, clip=true, strokeDash=[4,4], point=true) :
      mark2 == :diamond          ? (type=:point, clip=true, shape=:diamond) :
      mark2 == :circle           ? (type=:circle, clip=true) :
      (type=mark2, clip=true,)
    encoding2 = (;
      encoding1...,
      y = (field=y2, type=:quantitative, scale=yscale_props),
    )
    push!(layers, (mark=mark_props2, encoding=encoding2,))
  end

  # Spec
  height = ceil(Int, width/ratio)
  width_prop  = by === nothing ? width  : ceil(Int, width/columns)
  height_prop = by === nothing ? height : ceil(Int, height/columns)
  vspec = by === nothing ?
    (title=ftitle, layer=layers, width=width_prop, height=height_prop) :
    (
      title=ftitle,
      facet=(field=String(by), type=:ordinal),
      columns,
      spec=(layer=layers, width=width_prop, height=height_prop)
    )

  spec = VegaLite.VLSpec(JSON.parse(JSON.json(vspec)))
  fig = spec(ds)
  display(fig)
  ftitle, fig
end

plot_xyc_by(ds; args...) = plot_xyc_by("", ds; args...)