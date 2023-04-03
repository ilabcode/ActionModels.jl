using ActionModels
using Distributions

agent = premade_agent("binary_rw_softmax")

param_priors = Dict("learning_rate" => Uniform(0, 1))

inputs = [1, 0, 1]

actions = give_inputs!(agent, inputs)

chains = fit_model(agent, param_priors, inputs, actions, n_chains = 1, n_iterations = 10)

get_posteriors(chains)
