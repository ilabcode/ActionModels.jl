### Functions for getting a single param ###
"""
"""
function Turing.get_params(agent::Agent, target_param::Union{String,Tuple})
    #If the state is in the agent's parameters
    if target_param in keys(agent.params)
        #Extract it
        param = agent.params[target_param]
    else
        #Otherwise look in the substruct
        param = ActionModels.get_params(agent.substruct, target_param)
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
"""
function Turing.get_params(agent::Agent, target_params::Vector)
    #Initialize dict
    params = Dict()

    #Go through each state
    for param_name in target_params
        #Get them with get_history, and add to the tuple
        params[param_name] = ActionModels.get_params(agent, param_name)
    end

    return params
end


### Function for getting all parameters ###
"""
"""
function Turing.get_params(agent::Agent)

    #Collect names of all agent parameters
    target_params = collect(keys(agent.params))

    #Get the agent's parameters
    params = ActionModels.get_params(agent, target_params)

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