"""

Function for checking the structure of the agent
"""
function check_agent(agent::Agent)

    if length(agent.parameter_groups) > 0

        ## Check for the same derived parameter in multiple shared parameters 
        #Get out the derived parameters of all shared parameters 
        grouped_parameters = [
            parameter for list_of_grouped_parameters in [
                agent.parameter_groups[parameter_key].grouped_parameters for
                parameter_key in keys(agent.parameter_groups)
            ] for parameter in list_of_grouped_parameters
        ]

        #check for duplicate names
        if length(grouped_parameters) > length(unique(grouped_parameters))
            #Throw an error
            throw(
                ArgumentError(
                    "At least one parameter is set by multiple shared parameters. This is not supported.",
                ),
            )
        end
    end


end
