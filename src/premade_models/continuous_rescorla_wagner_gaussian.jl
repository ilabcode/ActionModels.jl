function continuous_rescorla_wagner_gaussian(agent::Agent, input::Real)

    ## Read in parameters from the agent
    learning_rate = agent.parameters["learning_rate"]
    action_noise = agent.parameters["action_noise"]

    ## Read in states with an initial value
    old_value = agent.states["value"]

    ##We dont have any settings in this model. If we had, we would read them in as well.
    ##-----This is where the update step starts -------

    ##Get new value state
    new_value = old_value + learning_rate * (input - old_value)


    ##-----This is where the update step ends -------
    ##Create Bernoulli normal distribution our action probability which we calculated in the update step
    action_distribution = Distributions.Normal(new_value, action_noise)

    ##Update the states and save them to agent's history
    update_states!(agent, "value", new_value)
    update_states!(agent, "input", input)

    ## return the action distribution to sample actions from
    return action_distribution
end



function premade_continuous_rescorla_wagner_gaussian(config::Dict)

    #Default parameters and settings
    default_config =
        Dict("learning_rate" => 0.1, "action_noise" => 1, ("initial", "value") => 0)

    #Warn the user about used defaults and misspecified keys
    warn_premade_defaults(default_config, config)

    #Merge to overwrite defaults
    config = merge(default_config, config)

    ## Create agent
    action_model = continuous_rescorla_wagner_gaussian
    parameters = Dict(
        "learning_rate" => config["learning_rate"],
        "action_noise" => config["action_noise"],
        InitialStateParameter("value") => config[("initial", "value")],
    )
    states = Dict("input" => missing, "value" => missing)
    settings = Dict()

    return init_agent(
        action_model,
        parameters = parameters,
        states = states,
        settings = settings,
    )
end
