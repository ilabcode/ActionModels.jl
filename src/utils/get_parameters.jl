"""
    get_parameters(agent::Agent, target_param::Union{String,Tuple})

Get a single parameter from an agent. Returns a single value.

    get_parameters(agent::Agent, target_param::Vector)

Get a set of parameter values from an agent. Returns a dictionary of parameters and their values.

    get_parameters(agent::Agent)

Get all parameters from an agent. Returns a dictionary of parameters and their values.
"""
function get_parameters end


### Functions for getting a single param ###
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

        #If the target parameter is in the agents's shared parameters
    elseif target_param in keys(agent.parameter_groups)
        #Extract it, take only the value
        param = agent.parameter_groups[target_param].value

    else
        #Otherwise look in the substruct
        param = get_parameters(agent.substruct, target_param)
    end

    return param
end

function get_parameters(substruct::Nothing, target_param::Union{String,Tuple})
    throw(
        ArgumentError("The specified parameter $target_param does not exist in the agent"),
    )
    return nothing
end


### Functions for getting multiple parameters ###
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
function get_parameters(agent::Agent)

    #Get all parameters from the substruct
    substruct_parameters = get_parameters(agent.substruct)

    #Collect keys for parameters
    parameter_keys = collect(keys(agent.parameters))

    #Collect keys for initial state parameters
    initial_state_parameter_keys =
        map(x -> ("initial", x), collect(keys(agent.initial_state_parameters)))

    #Collect keys for shared parameters
    parameter_group_keys = collect(keys(agent.parameter_groups))

    #Combine all parameter keys into one
    target_parameters =
        vcat(parameter_keys, initial_state_parameter_keys, parameter_group_keys)

    #If there are shared parameters
    if length(parameter_group_keys) > 0
        #Go through each shared parameter
        for parameter_group in values(agent.parameter_groups)
            #Remove derived parameters from the list
            filter!(x -> x ∉ parameter_group.grouped_parameters, target_parameters)

            #Filter the substruct parameter dictionary to remove parameters with keys that are derived parameters
            filter!(x -> x[1] ∉ parameter_group.grouped_parameters, substruct_parameters)
        end
    end

    #Get the agent's parameter values
    agent_parameters = get_parameters(agent, target_parameters)

    #Merge agent parameters and substruct parameters
    parameters = merge(agent_parameters, substruct_parameters)

    return parameters
end

function get_parameters(substruct::Nothing)
    #If the substruct is empty, return an empty list
    return Dict()
end
