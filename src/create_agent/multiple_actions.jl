"""
    multiple_actions(agent::Agent, input::Any)

Action model that combines multiple action models stored in the agent.
"""
function multiple_actions(agent::Agent, input::Any)

    #Extract vector of action models
    action_models = agent.settings["action_models"]

    #Initialize vector for action distributions
    action_distributions = []

    #Do each action model separately
    for action_model in action_models
        #And append them to the vector of action distributions
        push!(action_distributions, action_model(agent, input))
    end

    return action_distributions
end
