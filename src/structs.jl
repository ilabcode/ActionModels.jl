"""
"""
Base.@kwdef mutable struct Agent
    action_model::Function
    substruct::Any
    parameters::Dict = Dict()
    initial_state_parameters::Dict{String,Any} = Dict()
    parameter_groups::Dict = Dict()
    states::Dict{String,Any} = Dict("action" => missing)
    history::Dict{String,Vector{Any}} = Dict("action" => [missing])
    settings::Dict{String,Any} = Dict()
    save_history::Bool = true
end

#TYPE FOR RETURNING THINGS FROM STATISTICAL MDOEL ###
struct StatisticalModelReturn
    agent_parameters::Vector{Dict}
    statistical_values::Any
end
#Add default value for statistical values
StatisticalModelReturn(agent_parameters::Vector{D}) where {D<:Dict} =
    StatisticalModelReturn(agent_parameters, nothing)

#TYPE FOR RETURNING GENERATED QUANTITIES
struct GeneratedQuantitites
    agents_parameters::Vector{Dict}
    agents_states::Vector{Dict}
    statistical_values::Union{Some{Any},Nothing}
end

### FOR THE GREATER FITMODEL FUNCTION
mutable struct FitModelResults
    model::DynamicPPL.Model
    tracked_model::Union{Nothing,DynamicPPL.Model}
    chains::Chains
end

"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end





"""
Type to use for specifying a paramter that sets a state's initial value
"""
Base.@kwdef mutable struct InitialStateParameter
    state::Any
end

"""
Type for specifying a group of parameters
"""
Base.@kwdef mutable struct ParameterGroup
    name::String
    parameters::Vector
    value::Real
end

"""
Type for shared parameters containing both the parameter value and a vector of parameter names that will share that value
"""
Base.@kwdef mutable struct GroupedParameters
    value::Real
    grouped_parameters::Vector
end
