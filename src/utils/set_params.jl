###Function for setting a single parameter ###
"""
set_params!(agent::Agent,  target_param::Union{String,Tuple}, param_value::Any)

Function for setting a single parameter in an agent. Input to the function is an agent, the parameter name and the parameter value.
"""
function set_params!(agent::Agent, target_param::Union{String,Tuple}, param_value::Any)

    #If the parameter exists in the agent's params
    if target_param in keys(agent.params)
        #Set it
        agent.params[target_param] = param_value

        #If the parameter exists in the agent's initial state params
    elseif target_param isa Tuple &&
           target_param[1] == "initial" &&
           target_param[2] in keys(agent.initial_state_params)
        #Set it
        agent.initial_state_params[target_param[2]] = param_value

    else
        #Otherwise, look in the substruct
        set_params!(agent.substruct, target_param, param_value)
    end
end

"""
"""
function set_params!(
    substruct::Nothing,
    target_param::Union{String,Tuple},
    param_value::Any,
)
    throw(
        ArgumentError("The specified parameter $target_param does not exist in the agent"),
    )
end



### Function for setting multiple parameters
"""
    set_params!(agent::Agent, params::Dict)

Setting multiple parameters in dictionary where parameter name is specified followed by parameter value. 
"""
function set_params!(agent::Agent, params::Dict)

    #For each parameter to set
    for (param_key, param_value) in params
        #Set that parameter
        set_params!(agent, param_key, param_value)
    end
end
