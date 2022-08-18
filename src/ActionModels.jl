module ActionModels

#Load packages
using Turing, Distributions, RecipesBase, Logging

#Types for agents and errors
include("structs.jl")

#Functions for creating agents
include("create_agent/init_agent.jl")
include("create_agent/create_premade_agent.jl")

#Functions for fitting agents to data
include("fitting/fit_model.jl")

#Plotting functions for agents
include("plots/predictive_simulation_plot.jl")
include("plots/parameter_distribution_plot.jl")
include("plots/trajectory_plot.jl")

#Functions for making premade agent
include("premade_models/premade_agents.jl")
include("premade_models/premade_action_models.jl")

#Utility functions for agents
include("utils/get_history.jl")
include("utils/get_params.jl")
include("utils/get_states.jl")
include("utils/give_inputs.jl")
include("utils/reset.jl")
include("utils/set_params.jl")
include("utils/warn_premade_defaults.jl")
include("utils/get_posteriors.jl")
end