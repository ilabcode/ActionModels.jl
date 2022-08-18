###Function for setting a single parameter ###
"""
"""
function set_params!(agent::AgentStruct, target_param::Union{String,Tuple}, param_value::Any)

    #If the parameter exists in the agent
    if target_param in keys(agent.params)
        #Set it
        agent.params[target_param] = param_value
    else
        #Otherwise, look in the substruct
        set_params!(agent.substruct, target_param, param_value)
    end
end

"""
"""
function set_params!(substruct::Nothing, target_param::Union{String,Tuple}, param_value::Any)
    throw(ArgumentError("The specified parameter $target_param does not exist in the agent"))
end



### Function for setting multiple parameters
"""
"""
function set_params!(agent::AgentStruct, params::Dict)

    #For each parameter to set
    for (param_key, param_value) in params
        #Set that parameter
        set_params!(agent, param_key, param_value)
    end
end
