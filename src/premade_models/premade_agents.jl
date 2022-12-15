"""
    premade_binary_rw_softmax(config::Dict)

Create premade agent according to binary Rescorla-Wagner softmax model.
Parameters in this agent are "learning_rate" and "softmax_action_precision".

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
    params = Dict(
        "learning_rate" => config["learning_rate"],
        "softmax_action_precision" => config["softmax_action_precision"],
        ("initial", "value") => config[("initial", "value")],
    )
    states = Dict(
        "value" => missing,
        "transformed_value" => missing,
        "action_probability" => missing,
    )
    settings = Dict()

    return init_agent(action_model, params = params, states = states, settings = settings)
end
