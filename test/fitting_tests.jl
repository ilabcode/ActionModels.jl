
using ActionModels
using Distributions
using Plots
using StatsPlots

agent = premade_agent("premade_binary_rw_softmax")

param_priors = Dict("learning_rate" => Uniform(0, 1))

inputs = [1, 0, 1,1,1,1,1,0,0,0,0]

actions = give_inputs!(agent, inputs)

chains = fit_model(agent, param_priors, inputs, actions, n_chains = 1, n_iterations = 10)

plot(fitted_model)

plot_parameter_distribution(fitted_model, priors)

get_posteriors(fitted_model)
