"""
"""
Base.@kwdef mutable struct Agent
    action_model::Function
    substruct::Any
    parameters::Dict = Dict()
    initial_state_parameters::Dict{String,Any} = Dict()
    states::Dict{String,Any} = Dict("action" => missing)
    settings::Dict{String,Any} = Dict()
    shared_parameters::Dict = Dict()
    history::Dict{String,Vector{Any}} = Dict("action" => [missing])
end


"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end

"""
Shared parameters
"""
Base.@kwdef mutable struct SharedParameter
    value::Real
    derived_parameters::Vector
end
