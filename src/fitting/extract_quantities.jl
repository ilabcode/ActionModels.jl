
#########################################################
####### FUNCTION FOR EXTRACTING GENERATED QUANTITIES ####
#########################################################

function extract_quantities(fitted_model::Chains, model::DynamicPPL.Model)

    #Extract the generated quantities from the fitted model
    quantities = generated_quantities(model, fitted_model)

    #Extract information for later
    _quantities = first(quantities)
    n_agents = length(_quantities.agents_parameters)
    parameter_keys = keys(first(_quantities.agents_parameters))

    #Create containers for the restructured values
    agent_parameters = [
        Dict(parameter_key => Vector{Real}() for parameter_key in parameter_keys) for
        _ = 1:n_agents
    ]
    statistical_values = Vector()

    #For each sample
    for (sample_idx, sample) in enumerate(quantities)

        #Unpack the sample
        sample_agent_parameters = sample.agents_parameters
        sample_statistical_values = sample.statistical_values

        #For each agent
        for agent_idx = 1:n_agents

            #For each parameter
            for parameter_key in parameter_keys
                #save the sampled parameter value
                push!(
                    agent_parameters[agent_idx][parameter_key],
                    sample_agent_parameters[agent_idx][parameter_key],
                )
            end
        end

        #Store the statistical value for the sample
        push!(statistical_values, sample_statistical_values)
    end

    return (agent_parameters, statistical_values)
end
