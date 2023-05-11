using ActionModels
using HierarchicalGaussianFiltering
using Distributions

agent = premade_agent("hgf_gaussian_action")

priors = Dict(("x1", "evolution_rate") => Normal(-5, 1))

inputs = [1, 1.2, 1.4]

actions = [1, 1.1, 1.5]

@test fit_model(agent, priors, inputs, actions, n_iterations = 100, n_chains = 1)
