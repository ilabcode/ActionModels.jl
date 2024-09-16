
#########################################################
####### FUNCTION FOR EXTRACTING GENERATED QUANTITIES ####
#########################################################

function extract_quantities(fitted_model::Chains, model::DynamicPPL.Model)

    #Check whether track_states = true
    model.args.track_states || error(
        "The passed model does not have track_changes = true. This is required for extracting agent states. Recreate the model with track_changes = true and repeat.",
    )

    #Extract the generated quantities from the fitted model
    quantities = generated_quantities(model, fitted_model)

    #Extract information for later
    _quantities = first(quantities)
    n_agents = length(_quantities.agents_parameters)
    parameter_keys = keys(first(_quantities.agents_parameters))
    state_keys = keys(first(_quantities.agents_states))

    #Create containers for the restructured values
    agent_parameters = [
        Dict(parameter_key => Vector{Real}() for parameter_key in parameter_keys) for
        _ = 1:n_agents
    ]
    agent_states = [
        Dict{Any,Array}(state_key => Vector() for state_key in state_keys) for
        _ = 1:n_agents
    ]
    statistical_values = Vector()

    #For each sample
    for (sample_idx, sample) in enumerate(quantities)

        #Unpack the sample
        sample_agent_parameters = sample.agents_parameters
        sample_agent_states = sample.agents_states
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

            #For each state
            for state_key in state_keys
                #save the sampled state value
                push!(
                    agent_states[agent_idx][state_key],
                    sample_agent_states[agent_idx][state_key],
                )
            end
        end

        #Store the statistical value for the sample
        push!(statistical_values, sample_statistical_values)
    end

    #For each agent
    for agent_idx = 1:n_agents
        #For each state
        for state_key in state_keys
            #Make the vector of vectors into a matrix
            agent_states[agent_idx][state_key] =
                transpose(reduce(hcat, agent_states[agent_idx][state_key]))
        end
    end

    #Give option for returning whole chains, CI, etc

    return (agent_parameters, agent_states, statistical_values)
end
