module ActionModels

#Load packages
using Turing, TuringGLM, Distributions, DataFrames, RecipesBase, Logging, Distributed, LinearAlgebra

#Export functions
export Agent, RejectParameters, SharedParameter, Multilevel
export init_agent, premade_agent, warn_premade_defaults, multiple_actions, check_agent
export create_agent_model, fit_model
export plot_parameter_distribution,
    plot_predictive_simulation, plot_trajectory, plot_trajectory!
export get_history, get_states, get_parameters, set_parameters!, reset!, give_inputs!
export get_posteriors

function __init__()
    premade_agents["binary_rw_softmax"] = premade_binary_rw_softmax
    premade_agents["continuous_rescorla_wagner"] = premade_continuous_rescorla_wagner
end

#Types for agents and errors
include("structs.jl")

#Functions for creating agents
include("create_agent/init_agent.jl")
include("create_agent/create_premade_agent.jl")
include("create_agent/multiple_actions.jl")
include("create_agent/check_agent.jl")
#Functions for fitting agents to data
include("fitting/fitting_helper_functions.jl")
include("fitting/create_model.jl")
include("fitting/fit_model.jl")
include("fitting/prefit_checks.jl")
include("fitting/create_statistical_model.jl")

#Plotting functions for agents
include("plots/plot_predictive_simulation.jl")
include("plots/plot_parameter_distribution.jl")
include("plots/plot_trajectory.jl")

#Functions for making premade agent
include("premade_models/premade_agents.jl")
include("premade_models/premade_action_models.jl")

#Utility functions for agents
include("utils/get_history.jl")
include("utils/get_parameters.jl")
include("utils/get_states.jl")
include("utils/give_inputs.jl")
include("utils/reset.jl")
include("utils/set_parameters.jl")
include("utils/warn_premade_defaults.jl")
include("utils/get_posteriors.jl")
include("utils/pretty_printing.jl")
end
