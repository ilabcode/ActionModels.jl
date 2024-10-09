##################
## AGENT STRUCT ##
##################
"""
Agent struct
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


######################################
## STRUCTS FOR CREATE AND FIT MODEL ##
######################################
struct PopulationModelReturn
    agent_parameters::Vector{Dict}
    other_values::Union{Nothing,Any}
end
PopulationModelReturn(agent_parameters::Vector{D}) where {D<:Dict} =
    PopulationModelReturn(agent_parameters, nothing)

struct CheckRejections end
struct MissingActions end
mutable struct FitModelResults
    chains::Chains
    model::DynamicPPL.Model
end

"""
Custom error type which will result in rejection of a sample
"""
struct RejectParameters <: Exception
    errortext::Any
end


####################################
## STRUCTS FOR SETTING PARAMETERS ##
####################################
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

"""
Internal type for prepared regression priors
"""
Base.@kwdef mutable struct RegPrior
    β::Distribution
    σ::Union{Nothing,Vector{Distribution}}
end

"""
Input struct for setting regression priors
"""
Base.@kwdef struct RegressionPrior{D1<:Distribution, D2<:Distribution}
    β::Union{D1, Vector{D1}} = TDist(3)*2.5
    σ::Union{D2, Vector{Vector{D2}}} = truncated(TDist(3)*2.5, lower = 0)
end

"""
Input struct for specifying a regression
"""
struct Regression
    formula::MixedModels.FormulaTerm
    prior::RegPrior
    link::Function

    Regression(formula::MixedModels.FormulaTerm, prior::RegPrior = RegPrior(), link::Function = identity) = begin
        new(formula, prior, link)
    end
    Regression(formula::MixedModels.FormulaTerm, link::Function, prior::RegPrior = RegPrior()) = begin
        new(formula, prior, link)
    end
end