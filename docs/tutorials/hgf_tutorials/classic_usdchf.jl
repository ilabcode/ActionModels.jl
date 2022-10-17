# This is a replication of the tutorial from the MATLAB toolbox, using an HGF to filter the exchange rates between USD and CHF

# First load packages
using ActionModels
using HierarchicalGaussianFiltering
using Turing
using Plots
using StatsPlots

# Get the path for the HGF superfolder
hgf_path = dirname(dirname(pathof(HierarchicalGaussianFiltering)))
# Add the path to the data files
data_path = hgf_path * "/docs/tutorials/data/"

# Load the data
inputs = Float64[]
open(data_path * "classic_usdchf_inputs.dat") do f
    for ln in eachline(f)
        push!(inputs, parse(Float64, ln))
    end
end

#Create HGF
hgf = premade_hgf("continuous_2level", verbose = false);
agent = premade_agent("hgf_gaussian_action", hgf, verbose = false);

# Set parameters for parameter recovyer
parameters = Dict(
    ("u", "x1", "value_coupling") => 1.0,
    ("x1", "x2", "volatility_coupling") => 1.0,
    ("u", "evolution_rate") => -log(1e4),
    ("x1", "evolution_rate") => -13,
    ("x2", "evolution_rate") => -2,
    ("x1", "initial_mean") => 1.04,
    ("x1", "initial_precision") => 1 / (0.0001),
    ("x2", "initial_mean") => 1.0,
    ("x2", "initial_precision") => 1 / 0.1,
    "gaussian_action_precision" => 100,
);

set_params!(agent, parameters)
reset!(agent)

# Evolve agent
actions = give_inputs!(agent, inputs);

# Plot trajectories
trajectory_plot(
    agent,
    "u",
    size = (1300, 500),
    xlims = (0, 615),
    markersize = 3,
    markercolor = "green2",
    title = "HGF trajectory",
    ylabel = "CHF-USD exchange rate",
    xlabel = "Trading days since 1 January 2010",
)

trajectory_plot!(agent, ("x1", "posterior"), color = "red")
trajectory_plot!(
    agent,
    "action",
    size = (1300, 500),
    xlims = (0, 614),
    markersize = 3,
    markercolor = "orange",
)

trajectory_plot(
    agent,
    "x2",
    color = "blue",
    size = (1300, 500),
    xlims = (0, 615),
    xlabel = "Trading days since 1 January 2010",
    title = "Volatility parent trajectory"
)

# Set priors for turing fitting
fixed_params = Dict(
    ("u", "x1", "value_coupling") => 1.0,
    ("x1", "x2", "volatility_coupling") => 1.0,
    ("x1", "initial_mean") => 0,
    ("x1", "initial_precision") => 2000,
    ("x2", "initial_mean") => 1.0,
    ("x2", "initial_precision") => 600.0,
    "gaussian_action_precision" => 100,
);

param_priors = Dict(
    ("u", "evolution_rate") => Normal(-10, 2),
    ("x1", "evolution_rate") => Normal(-10, 4),
    ("x2", "evolution_rate") => Normal(-4, 4),
);

# Prior predictive simulation plot
predictive_simulation_plot(param_priors, agent, inputs, ("x1", "posterior_mean"); n_simulations = 3)

# Do parameter recovery
chain = fit_model(
    agent,
    inputs,
    actions,
    param_priors,
    fixed_params,
    verbose = true,
    n_iterations = 10,
)

# Plot the chains
plot(chain)

# Plot prior posterior distributions
parameter_distribution_plot(chain, param_priors)

# Posterior predictive plot
predictive_simulation_plot(
    chain,
    agent,
    inputs,
    ("x1", "posterior_mean");
    n_simulations = 3,
)
