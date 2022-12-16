using ActionModels
using Distributions

agent = premade_agent("premade_binary_rw_softmax")

priors = Dict("learning_rate" => Uniform(0, 1))

inputs = [1, 0, 1]

actions = give_inputs!(agent, inputs)

chains = fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 10)
