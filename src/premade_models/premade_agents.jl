function premade_rw_softmax(specs::Dict)

    #Default parameters and settings
    defaults = Dict("learning_rate" => 1, "softmax_action_precision" => 1, "start_value" => 1)

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(defaults, specs)

    #Merge to overwrite defaults
    specs = merge(defaults, specs)

    ## Create agent 
    action_model = binary_rw_softmax
    params = Dict(
        "learning_rate" => specs["learning_rate"],
        "sigmoid_action_precision" => specs["sigmoid_action_precision"],
    )
    states = Dict("value" => "start_value")
    settings = Dict()

    return init_agent(action_model, params, states, settings)
end
