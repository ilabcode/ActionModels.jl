````@example creating_own_action_model
function custom_rescorla_wagner_softmax(agent::Agent, input)

    # Read in parameters from the agent
    learning_rate = agent.parameters["learning_rate"]
    action_precision = agent.parameters["softmax_action_precision"]

    #Read in states with an initial value
    old_value = agent.states["value"] 

    #We dont have any settings in this model. If we had, we would read them in as well.

    # ----- This is where the update step starts -------

    # Sigmoid transform the value
    old_value_probability = 1 / (1 + exp(-old_value))   

    #Get new value state
    new_value = old_value + learning_rate * (input - old_value_probability) 

    #Pass through softmax to get action probability
    action_probability = 1 / (1 + exp(-action_precision * new_value))

    #----- This is where the update step ends -------

    #Create Bernoulli normal distribution our action probability which we calculated in the update step
    action_distribution = Distributions.Bernoulli(action_probability) 

    #Update the states and save them to agent's history

    agent.states["value"] = new_value
    agent.states["transformed_value"] = 1 / (1 + exp(-new_value))
    agent.states["action_probability"] = action_probability

    push!(agent.history["value"], new_value)
    push!(agent.history["transformed_value"], 1 / (1 + exp(-new_value)))
    push!(agent.history["action_probability"], action_probability)

    # return the action distribution to sample actions from
    return action_distribution
end
````