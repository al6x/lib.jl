import Dates, DataFrames

to_data(::Nothing) = nothing
to_data(::Missing) = nothing
to_data(x::Union{String,Real}) = x
to_data(x::Symbol) = string(x)
to_data(x::Dates.TimeType) = string(x)
to_data(xs::AbstractVector) = to_data.(xs)
to_data(ds::DataFrames.DataFrame) =
  [Dict(string(c) => to_data(row[c]) for c in names(ds)) for row in DataFrames.eachrow(ds)]
to_data_struct(x) = Dict(string(f) => to_data(getfield(x, f)) for f in fieldnames(typeof(x)))

from_data(::Type{String}, x::String) = x
from_data(::Type{T}, x::Real) where {T<:Real} = T(x)
from_data(::Type{Dates.Date}, x::Dates.Date) = x
from_data(::Type{Dates.Date}, x::String) = Dates.Date(x, Dates.DateFormat("yyyy-mm-dd"))
from_data(::Type{Dates.DateTime}, x::Dates.DateTime) = x
from_data(::Type{Dates.DateTime}, x::String) =
  Dates.DateTime(x, Dates.DateFormat("yyyy-mm-dd HH:MM:SS"))
from_data(::Type{Vector{T}}, xs) where {T} = T[from_data(T, x) for x in xs]
from_data_struct(::Type{T}, data) where {T} = T((from_data(fieldtype(T, i),
  data_get(data, fieldnames(T)[i])) for i in 1:fieldcount(T))...)

data_get(data, f::Symbol) = haskey(data, string(f)) ? data[string(f)] : data[f]
