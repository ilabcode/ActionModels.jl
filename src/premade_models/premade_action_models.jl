"""
    binary_rw_softmax(agent::Agent, input::Bool)

Action model that learns from binary inputs with a classic Rescorla-Wagner model. Passes learnt probabilities through a softmax to get the action prpbability distribution.

Parameters: "learning_rate" and "softmax_action_precision".
States: "value", "value_probability", "action_probability".
"""
function binary_rw_softmax(agent::Agent, input::Union{Bool,Integer})

    #Read in parameters
    learning_rate = agent.parameters["learning_rate"]
    action_precision = agent.parameters["softmax_action_precision"]

    #Read in states
    old_value = agent.states["value"]

    #Sigmoid transform the value
    old_value_probability = 1 / (1 + exp(-old_value))

    #Get new value state
    new_value = old_value + learning_rate * (input - old_value_probability)

    #Pass through softmax to get action probability
    action_probability = 1 / (1 + exp(-action_precision * new_value))

    #Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
    action_distribution = Distributions.Bernoulli(action_probability)

    #Update states
    agent.states["value"] = new_value
    agent.states["value_probability"], 1 / (1 + exp(-new_value))
    agent.states["action_probability"], action_probability
    #Add to history
    push!(agent.history["value"], new_value)
    push!(agent.history["value_probability"], 1 / (1 + exp(-new_value)))
    push!(agent.history["action_probability"], action_probability)

    return action_distribution
end
