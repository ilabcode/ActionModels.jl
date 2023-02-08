"""

Function for checking the structure of the agent
"""
function check_agent(agent::Agent)

   if length(agent.shared_parameters)> 0

    ## Check for the same derived parameter in multiple shared parameters 
    #Get out the derived parameters of all shared parameters
    vector_of_derived_parameters = [agent.shared_parameters[i].derived_parameters for i in unique(keys(agent.shared_parameters))]
    #combine them to one list
    all_derived_parameters = [y for v in vector_of_derived_parameters for y in v]
    #check for duplicate names
    if length(all_derived_parameters) > length(unique(all_derived_parameters ))
        #Throw an error
        throw(
            ArgumentError(
                "The same derived parameter has two shared parameters",
            ),
        )
    end

    ## Check if the shared parameter is part of own derived parameters
    #Go through each specified shared parameter
    for (shared_parameter_key, dict_value) in agent.shared_parameters
        #check if the name of the shared parameter is part of its own derived parameters
        if shared_parameter_key in dict_value.derived_parameters
            throw(ArgumentError("The shared parameter is part of the list of derived parameters"))
        end
    end

    end

end