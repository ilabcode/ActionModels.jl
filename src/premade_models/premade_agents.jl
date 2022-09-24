function premade_rw_softmax(specs::Dict)

    #Default parameters and settings
    default_specs =
        Dict("learning_rate" => 1, "softmax_action_precision" => 1, "initial_value" => 1)

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(default_specs, specs)

    #Merge to overwrite defaults
    specs = merge(default_specs, specs)

    ## Create agent 
    action_model = binary_rw_softmax
    params = Dict(
        "learning_rate" => specs["learning_rate"],
        "softmax_action_precision" => specs["softmax_action_precision"],
    )
    states = Dict("value" => specs["initial_value"], "action_probability" => missing)
    settings = Dict()

    return init_agent(action_model, params = params, states = states, settings = settings)
end
