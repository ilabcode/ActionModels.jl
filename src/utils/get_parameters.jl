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
    elseif target_param in keys(agent.shared_parameters)
        #Extract it, take only the value
        param = agent.shared_parameters[target_param].value

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

    #Collect keys for parameters
    parameter_keys = collect(keys(agent.parameters))

    #Collect keys for initial state parameters
    initial_state_parameter_keys =
        map(x -> ("initial", x), collect(keys(agent.initial_state_parameters)))

    #Collect keys for shared parameters
    shared_parameter_keys = collect(keys(agent.shared_parameters))

    #Combine all parameter keys into one
    target_parameters =
        vcat(parameter_keys, initial_state_parameter_keys, shared_parameter_keys)

    #If there are shared parameters
    if length(shared_parameter_keys) > 0

        #Go through each shared parameter
        for shared_parameter in values(agent.shared_parameters)
            #Remove derived parameters from the list
            filter!(x -> x ∉ shared_parameter.derived_parameters, target_parameters)
        end
    end

    #Get the agent's parameter values
    parameters = get_parameters(agent, target_parameters)

    #Get parameters from the substruct
    substruct_parameters = get_parameters(agent.substruct)

    #Merge substruct parameters and agent parameters
    parameters = merge(parameters, substruct_parameters)
    return parameters
end

function get_parameters(substruct::Nothing)
    #If the substruct is empty, return an empty list
    return Dict()
end
