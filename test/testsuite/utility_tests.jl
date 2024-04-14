using ActionModels
using Distributions

#Agent
agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

#Variations of get_states
get_states(agent)

get_states(agent, "value_probability")

get_states(agent, ["value_probability", "action"])

#Variations of get_parameters
get_parameters(agent)

get_parameters(agent, ("initial", "value"))

get_parameters(agent, [("initial", "value"), "learning_rate"])

#Variations of set_parameters
set_parameters!(agent, ("initial", "value"), 1)

set_parameters!(agent, Dict("learning_rate" => 3, "action_precision" => 0.5))

#Variations of get_history
get_history(agent, "value")

get_history(agent)

reset!(agent)
