
"""
    init_agent(action_model::Function; substruct::Any = nothing, parameters::Dict = Dict(), states::Union{Dict, Vector} = Dict(),
    settings::Dict = Dict())
    
Initialize and agent. 

Note that action_model can also be specified as a vector of action models: action_model::Vector{Function}.
In this case the action models will be stored in the agent's settings. In that case use the function 'multiple_actions'

# Arguments
 - 'action_model::Function': a function specifying the agent's action model. Can be any function that takes an agent and a single input as arguments, and returns a probability distribution from which actions are sampled.
 - 'substruct::Any = nothing': struct containing additional parameters and states. This structure also get passed to utility functions. Check advanced usage guide.
 - 'parameters::Dict = Dict()': dictionary containing parameters of the agent. Keys are parameter names (strings, or tuples of strings), values are parameter values.
 - 'states::Union{Dict, Vector} = Dict()': dictionary containing states of the agent. Keys are state names (strings, or tuples of strings), values are initial state values. Can also be a vector of state name strings.
 - 'settings::Dict = Dict()': dictionary containing additional settings for the agent. Keys are setting names, values are setting values.

"""
function init_agent() end


function init_agent(
    action_model::Function;
    substruct::Any = nothing,
    parameters::Dict = Dict(),
    states::Union{Dict,Vector} = Dict(),
    settings::Dict = Dict(),
)

    ##Create action model struct
    agent = Agent(
        action_model = action_model,
        substruct = substruct,
        parameters = Dict(),
        initial_state_parameters = Dict(),
        states = Dict(),
        settings = settings,
    )


    ##Add parameters to either initial state parameters or parameters
    for (param_key, param_value) in parameters
        #If the param is an initial state parameter
        if param_key isa Tuple && param_key[1] == "initial"

            #Add the parameter to the initial state parameters
            agent.initial_state_parameters[param_key[2]] = param_value

        else
            #For other parameters, add to parameters
            agent.parameters[param_key] = param_value
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
    for (state_key, initial_value) in agent.initial_state_parameters

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
    parameters::Dict = Dict(),
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
        parameters = parameters,
        states = states,
        settings = settings,
    )

    return agent
end
