"""
"""
Base.@kwdef mutable struct Agent
    action_model::Function
    substruct::Any
    parameters::Dict = Dict()
    initial_state_parameters::Dict{String,Any} = Dict()
    shared_parameters::Dict = Dict()
    states::Dict{String,Any} = Dict("action" => missing)
    history::Dict{String,Vector{Any}} = Dict("action" => [missing])
    settings::Dict{String,Any} = Dict()
    save_history::Bool = true
end


"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end

"""
Type for shared parameters containing both the parameter value and a vector of parameter names that will share that value
"""
Base.@kwdef mutable struct SharedParameter
    value::Real
    derived_parameters::Vector
end


"""
"""
Base.@kwdef struct Multilevel
    group::Union{Symbol,String}
    distribution = Normal
    parameters::Vector{String} = []
end

"""
"""
Base.@kwdef mutable struct ParameterInfo
    name::Union{String,Tuple}
    group_dependencies::Vector
    group_levels::Union{Array,Matrix} = []
    multilevel_dependent::Bool
    distribution::Any
    parameters::Vector
end


"""
Type to use for specifying a paramter that sets a state's initial value
"""
Base.@kwdef mutable struct InitialStateParameter
    state
end