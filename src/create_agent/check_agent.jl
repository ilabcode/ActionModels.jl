"""

Function for checking the structure of the agent
"""
function check_agent(agent::Agent)

    if length(agent.shared_parameters) > 0

        ## Check for the same derived parameter in multiple shared parameters 
        #Get out the derived parameters of all shared parameters 
        derived_parameters = [
            parameter for list_of_derived_parameters in [
                agent.shared_parameters[i].derived_parameters for
                i in keys(agent.shared_parameters)
            ] for parameter in list_of_derived_parameters
        ]

        #check for duplicate names
        if length(derived_parameters) > length(unique(derived_parameters))
            #get a tuple of the occurances of the parameters and their names to take out the repeated parameters
            repeated_parameters = [
                derived_parameter[2] for derived_parameter in [
                    (count(==(parameter), derived_parameters), parameter) for
                    parameter in unique(derived_parameters)
                ] if derived_parameter[1] > 1
            ]
            #Throw an error
            throw(
                ArgumentError(
                    "The parameter(s) $repeated_parameters has two shared parameters",
                ),
            )
        end

        ## Check if the shared parameter is part of own derived parameters
        #Go through each specified shared parameter
        for (shared_parameter_key, dict_value) in agent.shared_parameters
            #check if the name of the shared parameter is part of its own derived parameters
            if shared_parameter_key in derived_parameters
                throw(
                    ArgumentError(
                        "The shared parameter $shared_parameter_key is among the parameters it is defined to set",
                    ),
                )
            end
        end

    end

end
