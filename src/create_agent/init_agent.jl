"""
"""
function init_agent(
    action_model::Function,
    substruct::Any = nothing,
    params::Dict = Dict(),
    states::Dict = Dict(),
    settings::Dict = Dict(),
)

    #Create action model struct
    agent = HGF.AgentStruct(
        action_model = action_model,
        substruct = substruct,
        params = params,
        states = states,
        settings = settings,
    )

    #If an action state was not specified
    if !("action" in keys(agent.states))
        #Add an empty action state
        agent.states["action"] = missing
    end

    #For each specified state
    for (state_key, state_value) in states
        #Add it to the history
        agent.history[state_key] = [state_value]
    end

    return agent
end


"""
"""
function init_agent(
    action_model::Vector{Function},
    substruct::Any = nothing,
    params::Dict = Dict(),
    states::Dict = Dict(),
    settings::Dict = Dict(),
)

    #Create action model struct
    agent = HGF.AgentStruct(
        action_model = multiple_actions,
        substruct = substruct,
        params = params,
        states = states,
        settings = settings,
    )

    #If an action state was not specified
    if !("action" in keys(agent.states))
        #Add an empty action state
        agent.states["action"] = missing
    end

    #For each specified state
    for (state_key, state_value) in states
        #Add it to the history
        agent.history[state_key] = [state_value]
    end

    #If a setting called action_models has been specified manually
    if "action_models" in keys(settings)
        #Throw an error
        throw(
            ArgumentError(
                "Using a setting called 'action_models' with multiple action models is not supported",
            ),
        )
    else
        #Add vector of action models to settings
        agent.settings["action_models"] = action_model
    end

    return agent
end