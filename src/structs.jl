"""
"""
Base.@kwdef mutable struct Agent
    action_model::Function
    substruct::Any
    params::Dict{String,Any} = Dict()
    states::Dict{String,Any} = Dict("action" => missing)
    settings::Dict{String,Any} = Dict()
    history::Dict{String,Vector{Any}} = Dict("action" => [missing])
end


"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end
