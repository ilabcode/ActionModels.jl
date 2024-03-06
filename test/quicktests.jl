using ActionModels
using Distributions

agent = premade_agent("binary_rw_softmax")

inputs = [1, 0, 1]

actions = give_inputs!(agent, inputs)



param_priors = Dict("learning_rate" => Uniform(0, 1))

chains = fit_model(
    agent,
    param_priors,
    inputs,
    actions,
    n_chains = 2,
    n_iterations = 10,
    n_cores = 2,
)

get_posteriors(chains)
