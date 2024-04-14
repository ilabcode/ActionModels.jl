using ActionModels
using HierarchicalGaussianFiltering
using Distributions
using Plots
using StatsPlots

agent = premade_agent(
    "hgf_gaussian",
    premade_hgf("continuous_2level", verbose = false),
    verbose = false,
)

priors = Dict(("x1", "volatility") => Normal(-5, 1))

inputs = [1, 1.2, 1.4]

actions = [1, 1.1, 1.5]

results = fit_model(agent, priors, inputs, actions, n_iterations = 100, n_chains = 1)

plot_parameter_distribution(results, priors)
