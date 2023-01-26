###Function for setting a single parameter ###
"""
    set_parameters!(agent::Agent, target_param::Union{String,Tuple}, param_value::Any)

Setting a single parameter value for an agent.

    set_parameters!(agent::Agent, parameter_values::Dict)

Set mutliple parameters values for an agent. Takes a dictionary of parameter names and values.
"""
function set_parameters! end

### Function for setting a single parameter ###
function set_parameters!(agent::Agent, target_param::Union{String,Tuple}, param_value::Any)

    #If the parameter exists in the agent's parameters
    if target_param in keys(agent.parameters)
        #Set it
        agent.parameters[target_param] = param_value

        #If the parameter exists in the agent's initial state parameters
    elseif target_param isa Tuple &&
           target_param[1] == "initial" &&
           target_param[2] in keys(agent.initial_state_parameters)
        #Set it
        agent.initial_state_parameters[target_param[2]] = param_value

        #If the target param is a shared parameter
    elseif target_param in keys(agent.shared_parameters)

        #Extract shared parameter
        shared_parameter = agent.shared_parameters[target_param]

        #Set the shared parameter value
        setfield!(shared_parameter, :value, param_value)

        #For each derived parameter
        for derived_param in shared_parameter.derived_parameters
            #Set that parameter
            set_parameters!(agent, derived_param, param_value)
        end
    else
        #Otherwise, look in the substruct
        set_parameters!(agent.substruct, target_param, param_value)
    end
end

function set_parameters!(
    substruct::Nothing,
    target_param::Union{String,Tuple},
    param_value::Any,
)
    throw(
        ArgumentError("The specified parameter $target_param does not exist in the agent"),
    )
end


### Function for setting multiple parameters ###
function set_parameters!(agent::Agent, parameter_values::Dict)

    #For each parameter to set
    for (param_key, param_value) in parameter_values
        #Set that parameter
        set_parameters!(agent, param_key, param_value)
    end
end
