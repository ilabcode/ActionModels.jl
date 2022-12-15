### Functions for getting a single param ###
"""
    get_parameters(agent::Agent, target_param::Union{String,Tuple})

Get out target parameter from agent 
"""
function get_parameters(agent::Agent, target_param::Union{String,Tuple})
    #If the target parameter is in the agent's parameters
    if target_param in keys(agent.parameters)
        #Extract it
        param = agent.parameters[target_param]

        #If the target parameter is in the agent's initial state parameters
    elseif target_param isa Tuple &&
           target_param[1] == "initial" &&
           target_param[2] in keys(agent.initial_state_parameters)
        #Extract it
        param = agent.initial_state_parameters[target_param[2]]
    else
        #Otherwise look in the substruct
        param = get_parameters(agent.substruct, target_param)
    end

    return param
end

"""
"""
function get_parameters(substruct::Nothing, target_param::Union{String,Tuple})
    throw(
        ArgumentError("The specified parameter $target_param does not exist in the agent"),
    )
    return nothing
end


### Functions for getting multiple parameters ###
"""
    get_parameters(agent::Agent, target_parameters::Vector)

Returns a vector of the target parameters specefied in target_parameters
    
    get_parameters(agent::Agent)

Returns all parameters from agent

"""
function get_parameters(agent::Agent, target_parameters::Vector)
    #Initialize dict
    parameters = Dict()

    #Go through each state
    for param_name in target_parameters
        #Get them with get_history, and add to the tuple
        parameters[param_name] = get_parameters(agent, param_name)
    end

    return parameters
end


### Function for getting all parameters ###
"""
"""
function get_parameters(agent::Agent)

    #Collect names of all agent parameters
    target_parameters = collect(keys(agent.parameters))

    #Go through each key in the initial state parameters
    for initial_state_key in keys(agent.initial_state_parameters)
        #Add it to the target parameters
        push!(target_parameters, ("initial", initial_state_key))
    end

    #Get the agent's parameters
    parameters = get_parameters(agent, target_parameters)

    #Get parameters from the substruct
    substruct_parameters = ActionModels.get_parameters(agent.substruct)

    #Merge substruct parameters and agent parameters
    parameters = merge(parameters, substruct_parameters)

    return parameters
end


function get_parameters(substruct::Nothing)
    #If the substruct is empty, return an empty list
    return Dict()
end
