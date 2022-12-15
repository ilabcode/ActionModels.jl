"""
    binary_rw_softmax(agent::Agent, input)

returns action distribution according to a binary Rescorla-Wagner softmax action model.
Agent parameters needed in this action model are "learning_rate" and "softmax_action_precision"

"""

function binary_rw_softmax(agent::Agent, input)

    #Read in parameters
    learning_rate = agent.params["learning_rate"]
    action_precision = agent.params["softmax_action_precision"]

    #Read in states
    old_value = agent.states["value"]

    #Sigmoid transform the value
    transformed_old_value = 1 / (1 + exp(-old_value))

    #Get new value state
    new_value = old_value + learning_rate * (input - transformed_old_value)

    #Pass through softmax to get action probability
    action_probability = 1 / (1 + exp(-action_precision * new_value))

    #Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
    action_distribution = Distributions.Bernoulli(action_probability)

    #Update states
    agent.states["value"] = new_value
    agent.states["transformed_value"], 1 / (1 + exp(-new_value))
    agent.states["action_probability"], action_probability
    #Add to history
    push!(agent.history["value"], new_value)
    push!(agent.history["transformed_value"], 1 / (1 + exp(-new_value)))
    push!(agent.history["action_probability"], action_probability)

    return action_distribution
end
