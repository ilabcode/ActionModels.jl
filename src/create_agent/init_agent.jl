
"""
    init_agent(action_model::Function; substruct::Any = nothing, params::Dict = Dict(), states::Union{Dict, Vector} = Dict(),
    settings::Dict = Dict(),)

Function to initialize agent. 

Note that action_model can also be specified as a vector of action models: action_model::Vector{Function}.
In this case the action models will be stored in the agent's settings. In that case use the function 'multiple_actions'

# Arguments
 - 'action_model::Function': input an action model either from premade action models or a custom actionmodel
 - 'substruct::Any = nothing': struct with own parameters and states which utility functions also get called on. Check advanced usage guide.
 - 'params::Dict = Dict()': Parameters of the agent. These are both parameters specifying initial states of the agent as well as the agents regular parameters.
 - 'states::Union{Dict, Vector} = Dict()': Write the states of the agent
 - 'settings::Dict = Dict()': used for variables that are not parameters or states that you want to use. e.g. if multiple action models are specified, they are stored in the settings.

"""

function init_agent() end



function init_agent(
    action_model::Function;
    substruct::Any = nothing,
    params::Dict = Dict(),
    states::Union{Dict,Vector} = Dict(),
    settings::Dict = Dict(),
)

    ##Create action model struct
    agent = Agent(
        action_model = action_model,
        substruct = substruct,
        params = Dict(),
        initial_state_params = Dict(),
        states = Dict(),
        settings = settings,
    )


    ##Add params to either initial state params or params
    for (param_key, param_value) in params
        #If the param is an initial state parameter
        if param_key isa Tuple && param_key[1] == "initial"

            #Add the parameter to the initial state parameters
            agent.initial_state_params[param_key[2]] = param_value

        else
            #For other parameters, add to params
            agent.params[param_key] = param_value
        end
    end


    ##Add states
    #If states is a dictionary
    if states isa Dict
        #Insert as states
        agent.states = states
        #If states is a vector
    elseif states isa Vector
        #Go through each state
        for state in states
            #And set to missing
            agent.states[state] = missing
        end
    end

    #If an action state was not specified
    if !("action" in keys(agent.states))
        #Add an empty action state
        agent.states["action"] = missing
    end

    #Initialize states
    for (state_key, initial_value) in agent.initial_state_params

        #If the state exists
        if state_key in keys(agent.states)
            #Set initial state
            agent.states[state_key] = initial_value

        else
            #Throw error
            throw(
                ArgumentError(
                    "The state $(state_key) has an initial state parameter, but does not exist in the agent.",
                ),
            )
        end
    end

    #For each specified state
    for (state_key, state_value) in agent.states
        #Add it to the history
        agent.history[state_key] = [state_value]
    end

    return agent
end


"""
"""
function init_agent(
    action_model::Vector{Function};
    substruct::Any = nothing,
    params::Dict = Dict(),
    states::Dict = Dict(),
    settings::Dict = Dict(),
)

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
        settings["action_models"] = action_model
    end

    #Create agent with the multiple actions action model
    agent = init_agent(
        multiple_actions,
        substruct = substruct,
        params = params,
        states = states,
        settings = settings,
    )

    return agent
end
