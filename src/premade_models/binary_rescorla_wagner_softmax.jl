"""
    binary_rescorla_wagner_softmax(agent::Agent, input::Bool)

Action model that learns from binary inputs with a classic Rescorla-Wagner model. Passes learnt probabilities through a softmax to get the action prpbability distribution.

Parameters: "learning_rate" and "action_precision".
States: "value", "value_probability", "action_probability".
"""
function binary_rescorla_wagner_softmax(agent::Agent, input::Union{Bool,Integer})

    #Read in parameters
    learning_rate = agent.parameters["learning_rate"]
    action_precision = agent.parameters["action_precision"]

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
    update_states!(agent, "value", new_value)
    update_states!(agent, "value_probability", 1 / (1 + exp(-new_value)))
    update_states!(agent, "action_probability", action_probability)
    update_states!(agent, "input", input)

    return action_distribution
end



"""
    premade_binary_rescorla_wagner_softmax(config::Dict)

Create premade agent that uses the binary_rescorla_wagner_softmax action model.

# Config defaults:
 - "learning_rate": 1
 - "action_precision": 1
 - ("initial", "value"): 0
"""

function premade_binary_rescorla_wagner_softmax(config::Dict)

    #Default parameters and settings
    default_config =
        Dict("learning_rate" => 0.1, "action_precision" => 1, ("initial", "value") => 0)
    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(default_config, config)

    #Merge to overwrite defaults
    config = merge(default_config, config)

    ## Create agent 
    action_model = binary_rescorla_wagner_softmax
    parameters = Dict(
        "learning_rate" => config["learning_rate"],
        "action_precision" => config["action_precision"],
        InitialStateParameter("value") => config[("initial", "value")],
    )
    states = Dict(
        "value" => missing,
        "value_probability" => missing,
        "action_probability" => missing,
        "input" => missing,
    )
    settings = Dict()

    return init_agent(
        action_model,
        parameters = parameters,
        states = states,
        settings = settings,
    )
end
