module PlotsExt

using JSON, VegaLite

# redblue, bluered, redyellowblue, blueyellowred, coolwarm, icefire
plot_xyc_by(
  title, ds; x, color=nothing, by=nothing, detail=nothing,
  xdomain=nothing, ydomain=nothing, palette=:viridis, reverse_palette=false,
  colortype=:ordinal, colorscale=nothing,
  xlabel=nothing, ylabel=nothing, colorlabel=nothing,
  xscale=nothing, yscale=nothing, pointsize=30, mark=:line_with_points,

  y=nothing, opacity=1.0,
  y2=nothing, mark2=:line_with_points, opacity2=1.0,
  y3=nothing, mark3=:line_with_points, opacity3=1.0,
  bound::Union{Nothing, Tuple{Any, Any}}=nothing, opacity_bound=0.2,

  xrulers=[], yrulers=[],

  xtype=:quantitative, ytype=:quantitative, resolve_y=nothing,
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

  tparts = ["$title x=$x"]
  y      !== nothing && push!(tparts, ", y=$y")
  color  !== nothing && push!(tparts, ", color=$color")
  detail !== nothing && push!(tparts, "($detail)")
  by     !== nothing && push!(tparts, " by=$by")
  y2     !== nothing && push!(tparts, ", dashed=$y2")
  y3     !== nothing && push!(tparts, ", dashed=$y3")
  ftitle = join(tparts, "")

  xscale_props = xdomain === nothing ? (type=xscale,) : (type=xscale, domain=xdomain)
  yscale_props = ydomain === nothing ? (type=yscale,) : (type=yscale, domain=ydomain)

  x_label_props = xlabel === nothing ? (;) : (; title=xlabel)
  y_label_props = ylabel === nothing ? (;) : (; title=ylabel)
  color_label_props = colorlabel === nothing ? (;) : (; title=colorlabel)

  # Palette
  palette = string(palette)
  if palette == "bluered"
    palette = "redblue"
    reverse_palette = !reverse_palette
  elseif palette == "redyellowblue"
    palette = "spectral"
  elseif palette == "blueyellowred"
    palette = "spectral"
    reverse_palette = !reverse_palette
  end

  color_scale_props = colorscale === nothing ? (;) : (; type=colorscale)
  color_props = if color === nothing; (;) else
    # Showing colors only if there is multiple values, otherwise
    # for some palettes the chart line is too pale.
    count = length(Set(skipmissing(getproperty(ds, Symbol(color)))))
    count <= 1 ? (;) :
      (; color=(field=color, type=colortype, color_label_props...,
        scale=(scheme=palette, reverse=reverse_palette, color_scale_props...)))
  end
  detail_props = detail === nothing ? (;) :
    (; detail=(field=detail, type=:ordinal))

  # reducing point size for facets
  npointsize = by === nothing ? pointsize : round(Int, pointsize/columns)
  poinsize_props = mark in (:point, :circle) ? (size=(value=npointsize,),) : (;)

  layers = []
  # xaxis_props = temporal_x ? (; axis=(format="%Y-%m-%d", labelAngle=-45,
  #   labelOverlap=:greedy, tickCount=max(2, round(Int, width/(by === nothing ? 1 : columns)/55)),
  #   grid=true)) : (;)
  encoding_base = (
    x = (field=x, type=xtype, scale=xscale_props, x_label_props...),
    color_props...,
    detail_props...,
    poinsize_props...
  )
  if y !== nothing
    mark_props1 =
      mark == :line             ? (; type=:line, clip=true, opacity) :
      mark == :line_with_points ? (; type=:line, clip=true, point=true, opacity) :
      mark == :circle           ? (; type=:circle, clip=true, opacity) :
      (; type=mark, clip=true, opacity,)
    encoding1 = (; encoding_base..., y = (field=y, type=ytype, scale=yscale_props, y_label_props...))
    push!(layers, (mark=mark_props1, encoding=encoding1,))
  end

  add_yn(yn, markn, opacity) = begin
    mark_props =
      markn == :line ?             (type=:line, clip=true, strokeDash=[4,4], opacity=opacity) :
      markn == :line_with_points ? (type=:line, clip=true, strokeDash=[4,4], point=true, opacity=opacity) :
      markn == :diamond          ? (type=:point, clip=true, shape=:diamond, opacity=opacity) :
      markn == :circle           ? (type=:circle, clip=true, opacity=opacity) :
      (type=markn, clip=true, opacity=opacity,)

    encoding = (;
      encoding_base...,
      y = (field=yn, type=ytype, scale=yscale_props, y_label_props...),
    )

    push!(layers, (mark=mark_props, encoding,))
  end
  y2 !== nothing && add_yn(y2, mark2, opacity2)
  y3 !== nothing && add_yn(y3, mark3, opacity3)

  add_ybound(ylo, yhi, opacity) = begin
    mark_props = (; type=:area, clip=true, opacity)

    encoding = (;
      encoding_base...,
      y = (field=ylo, type=ytype, scale=yscale_props, y_label_props...),
      y2 = (field=yhi,),
    )

    push!(layers, (mark=mark_props, encoding,))
  end
  bound !== nothing && add_ybound(bound..., opacity_bound)

  # Rulers
  for x in xrulers
    push!(layers, (;
      mark=(; type=:rule, color=:grey, clip=true),
      encoding=(; x=(; datum=x, type=:quantitative, scale=xscale_props, x_label_props...)),
    ))
  end
  for y in yrulers
    push!(layers, (;
      mark=(; type=:rule, color=:grey, clip=true),
      encoding=(; y=(; datum=y, type=:quantitative, scale=yscale_props, y_label_props...)),
    ))
  end

  # Spec
  height = ceil(Int, width/ratio)
  width_prop  = by === nothing ? width  : ceil(Int, width/columns)
  height_prop = by === nothing ? height : ceil(Int, height/columns)
  resolve_props = resolve_y === nothing ? (;) : (; resolve=(; scale=(; y=resolve_y)))
  vspec = by === nothing ?
    (title=ftitle, layer=layers, width=width_prop, height=height_prop) :
    (
      title=ftitle,
      facet=(field=String(by), type=:ordinal),
      columns,
      spec=(layer=layers, width=width_prop, height=height_prop),
      resolve_props...
    )

  spec_json = JSON.json(vspec)

  # println(spec_json)
  # println(JSON.json(ds))

  spec = VegaLite.VLSpec(JSON.parse(spec_json))
  fig = spec(ds)
  display(fig)
  ftitle, fig
end
export plot_xyc_by

plot_xyc_by(ds; args...) = plot_xyc_by("", ds; args...)

end
