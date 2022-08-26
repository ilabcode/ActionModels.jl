function binary_rw_softmax(agent::AgentStruct, input)

    #Read in parameters
    learning_rate = agent.params["learning_rate"]
    action_precision = agent.params["softmax_action_precision"]

    #Read in states
    old_value = agent.states["value"]

    #Sigmoid transform the value
    transformed_old_value = 1 / (1 + exp(-old_value))

    #Get new value state
    new_value = old_value + learning_rate * (input - transformed_old_value)

    #Update value
    agent.states["value"] = new_value
    #Add it to history
    push!(agent.history["value"], new_value)

    #Pass through softmax to get action probability
    action_probability = 1 / (1 + exp(-action_precision * new_value))

    #Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
    distribution = Distributions.Bernoulli(action_probability)

    return distribution
end