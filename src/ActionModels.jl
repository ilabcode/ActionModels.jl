module ActionModels

#Load packages
using Reexport
using Turing, ReverseDiff, DataFrames, AxisArrays, RecipesBase, Logging
using ProgressMeter, Distributed #TODO: get rid of this (only needed for parameter recovery)
using JLD2
@reexport using Distributions
using Turing: DynamicPPL, ForwardDiff, AutoReverseDiff, AbstractMCMC
#Export functions
export Agent, RejectParameters, InitialStateParameter, ParameterGroup
export init_agent, premade_agent, warn_premade_defaults, multiple_actions, check_agent
export independent_agents_population_model,
    create_model, fit_model, parameter_recovery, single_recovery
export ChainSaveResume
export plot_parameters, plot_trajectories, plot_trajectory, plot_trajectory!
export get_history,
    get_states,
    get_parameters,
    set_parameters!,
    reset!,
    give_inputs!,
    single_input!,
    set_save_history!
export extract_quantities, rename_chains, update_states!, get_estimates, get_trajectories

#Load premade agents
function __init__()
    # Only if not precompiling
    if ccall(:jl_generating_output, Cint, ()) == 0
        premade_agents["binary_rescorla_wagner_softmax"] =
            premade_binary_rescorla_wagner_softmax
        premade_agents["continuous_rescorla_wagner_gaussian"] =
            premade_continuous_rescorla_wagner_gaussian
    end
end

#Types for agents and errors
include("structs.jl")

const id_separator = "."
const id_column_separator = ":"
const tuple_separator = "__"

#Functions for creating agents
include("create_agent/init_agent.jl")
include("create_agent/create_premade_agent.jl")
include("create_agent/multiple_actions.jl")
include("create_agent/check_agent.jl")
#Functions for fitting agents to data
include("fitting/create_model.jl")
include("fitting/agent_model.jl")
include("fitting/fit_model.jl")
include("fitting/parameter_recovery.jl")
include("fitting/population_models/independent_agents_population_model.jl")
include("fitting/population_models/single_agent_population_model.jl")
include("fitting/helper_functions/check_model.jl")
include("fitting/helper_functions/extract_quantities.jl")
include("fitting/helper_functions/get_estimates.jl")
include("fitting/helper_functions/get_trajectories.jl")
include("fitting/helper_functions/helper_functions.jl")

#Plotting functions for agents
include("plots/plot_trajectories.jl")
include("plots/plot_parameters.jl")
include("plots/plot_trajectory.jl")

#Utility functions for agents
include("utils/get_history.jl")
include("utils/get_parameters.jl")
include("utils/get_states.jl")
include("utils/give_inputs.jl")
include("utils/reset.jl")
include("utils/set_parameters.jl")
include("utils/warn_premade_defaults.jl")
include("utils/pretty_printing.jl")
include("utils/update_states.jl")
include("utils/set_save_history.jl")

#Premade agents
include("premade_models/binary_rescorla_wagner_softmax.jl")
include("premade_models/continuous_rescorla_wagner_gaussian.jl")
end
