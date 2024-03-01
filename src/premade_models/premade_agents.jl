"""
    premade_binary_rw_softmax(config::Dict)

Create premade agent that uses the binary_rw_softmax action model.

# Config defaults:
 - "learning_rate": 1
 - "softmax_action_precision": 1
 - ("initial", "value"): 0
"""

function premade_binary_rw_softmax(config::Dict)

    #Default parameters and settings
    default_config = Dict(
        "learning_rate" => 1,
        "softmax_action_precision" => 1,
        ("initial", "value") => 0,
    )

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(default_config, config)

    #Merge to overwrite defaults
    config = merge(default_config, config)

    ## Create agent 
    action_model = binary_rw_softmax
    parameters = Dict(
        "learning_rate" => config["learning_rate"],
        "softmax_action_precision" => config["softmax_action_precision"],
        ("initial", "value") => config[("initial", "value")],
    )
    states = Dict(
        "value" => missing,
        "value_probability" => missing,
        "action_probability" => missing,
    )
    settings = Dict()

    return init_agent(
        action_model,
        parameters = parameters,
        states = states,
        settings = settings,
    )
end


function premade_continuous_rescorla_wagner(config::Dict)

    #Default parameters and settings
    default_config = Dict(
        "learning_rate" => 0.1,
        "action_noise" => 1,
        ("initial", "value") => 0,
    )

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(default_config, config)

    #Merge to overwrite defaults
    config = merge(default_config, config)

    ## Create agent
    action_model = continuous_rescorla_wagner
    parameters = Dict(
        "learning_rate" => config["learning_rate"],
        "action_noise" => config["action_noise"],
        ("initial", "value") => config[("initial", "value")],
    )
    states = Dict(
        "input" => missing,
        "value" => missing
    )
    settings = Dict()

    return init_agent(
        action_model,
        parameters = parameters,
        states = states,
        settings = settings,
    )
end
