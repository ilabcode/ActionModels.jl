### Functions for getting a single param ###
"""
    get_params(agent::Agent, target_param::Union{String,Tuple})

Get out target parameter from agent 
"""
function Turing.get_params(agent::Agent, target_param::Union{String,Tuple})
    #If the target parameter is in the agent's parameters
    if target_param in keys(agent.params)
        #Extract it
        param = agent.params[target_param]

        #If the target parameter is in the agent's initial state parameters
    elseif target_param isa Tuple &&
           target_param[1] == "initial" &&
           target_param[2] in keys(agent.initial_state_params)
        #Extract it
        param = agent.initial_state_params[target_param[2]]
    else
        #Otherwise look in the substruct
        param = get_params(agent.substruct, target_param)
    end

    return param
end

"""
"""
function Turing.get_params(substruct::Nothing, target_param::Union{String,Tuple})
    throw(
        ArgumentError("The specified parameter $target_param does not exist in the agent"),
    )
    return nothing
end


### Functions for getting multiple parameters ###
"""
    get_params(agent::Agent, target_params::Vector)

Returns a vector of the target parameters specefied in target_params
    
    get_params(agent::Agent)

Returns all parameters from agent

"""
function Turing.get_params(agent::Agent, target_params::Vector)
    #Initialize dict
    params = Dict()

    #Go through each state
    for param_name in target_params
        #Get them with get_history, and add to the tuple
        params[param_name] = get_params(agent, param_name)
    end

    return params
end


### Function for getting all parameters ###
"""
"""
function Turing.get_params(agent::Agent)

    #Collect names of all agent parameters
    target_params = collect(keys(agent.params))

    #Go through each key in the initial state params
    for initial_state_key in keys(agent.initial_state_params)
        #Add it to the target params
        push!(target_params, ("initial", initial_state_key))
    end

    #Get the agent's parameters
    params = get_params(agent, target_params)

    #Get parameters from the substruct
    substruct_params = ActionModels.get_params(agent.substruct)

    #Merge substruct parameters and agent parameters
    params = merge(params, substruct_params)

    return params
end


function Turing.get_params(substruct::Nothing)
    #If the substruct is empty, return an empty list
    return Dict()
end
