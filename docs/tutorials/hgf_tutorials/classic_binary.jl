# This tutorial is a copy of the 3 level binary hgf tutorial in MATLAB
#

# First load packages
using ActionModels
using HierarchicalGaussianFiltering
using Turing
using CSV
using DataFrames
using Plots
using StatsPlots

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/tutorials/data/"

# Load the data 
inputs = CSV.read(data_path * "classic_binary_inputs.csv", DataFrame)[!, 1];

# Create an HGF
hgf_params = Dict(
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("x2", "evolution_rate") => -2.5,
    ("x2", "initial_mean") => 0,
    ("x2", "initial_precision") => 1,
    ("x3", "evolution_rate") => -6.0,
    ("x3", "initial_mean") => 1,
    ("x3", "initial_precision") => 1,
    ("u", "x1", "value_coupling") => 1.0,
    ("x1", "x2", "value_coupling") => 1.0,
    ("x2", "x3", "volatility_coupling") => 1.0,
);
hgf = premade_hgf("binary_3level", hgf_params, verbose = false);

# Create an agent
agent_params = Dict("sigmoid_action_precision" => 5);
agent = premade_agent("hgf_unit_square_sigmoid_action", hgf, agent_params, verbose = false);

# Evolve agent and save actions
actions = give_inputs!(agent, inputs);

# Plot the trajectory of the agent
plot_trajectory(agent, ("u", "input_value"))
plot_trajectory!(agent, ("x1", "prediction"))

plot_trajectory(agent, ("x2", "posterior"))
plot_trajectory(agent, ("x3", "posterior"))

# Set fixed parameters
fixed_params = Dict(
    "sigmoid_action_precision" => 5,
    ("u", "category_means") => Real[0.0, 1.0],
    ("u", "input_precision") => Inf,
    ("x2", "initial_mean") => 0,
    ("x2", "initial_precision") => 1,
    ("x3", "initial_mean") => 1,
    ("x3", "initial_precision") => 1,
    ("u", "x1", "value_coupling") => 1.0,
    ("x1", "x2", "value_coupling") => 1.0,
    ("x2", "x3", "volatility_coupling") => 1.0,
    ("x2", "evolution_rate") => -3.0,
    ("x3", "evolution_rate") => -6.0,
);

# Set priors for parameter recovery
param_priors = Dict(("x2", "evolution_rate") => Normal(-3.0, 0.5));

# Prior predictive plot
plot_predictive_simulation(param_priors, agent, inputs, ("x1", "prediction_mean"), n_simulations = 3)

# Get the actions from the MATLAB tutorial
actions = CSV.read(data_path * "classic_binary_actions.csv", DataFrame)[!, 1];

# Fit the actions
chain = fit_model(
    agent,
    inputs,
    actions,
    param_priors,
    fixed_params,
    verbose = true,
    n_iterations = 10,
)

#Plot the chains
plot(chain)

# Plot the posterior
plot_parameter_distribution(chain, param_priors)

# Posterior predictive plot
plot_predictive_simulation(chain, agent, inputs, ("x1", "prediction_mean"), n_simulations = 3)
