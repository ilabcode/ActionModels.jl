"""

Function for checking the structure of the agent
"""
function check_agent(agent::Agent)

    if length(agent.shared_parameters) > 0

        ## Check for the same derived parameter in multiple shared parameters 
        #Get out the derived parameters of all shared parameters 
        derived_parameters = [
            parameter for list_of_derived_parameters in [
                agent.shared_parameters[parameter_key].derived_parameters for
                parameter_key in keys(agent.shared_parameters)
            ] for parameter in list_of_derived_parameters
        ]

        #check for duplicate names
        if length(derived_parameters) > length(unique(derived_parameters))
            #Throw an error
            throw(
                ArgumentError(
                    "At least one parameter is set by multiple shared parameters. This is not supported.",
                ),
            )
        end
    end


end
