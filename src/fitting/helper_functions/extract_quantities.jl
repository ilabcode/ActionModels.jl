
#########################################################
####### FUNCTION FOR EXTRACTING GENERATED QUANTITIES ####
#########################################################

function extract_quantities(model::DynamicPPL.Model, fitted_model::Chains)

    # Extract the generated quantities from the fitted model
    quantities = generated_quantities(model, fitted_model)

    # Extract agent ids
    agent_ids = model.args.agent_ids
    n_agents = length(agent_ids)

    # Extract parameter keys
    parameter_keys = collect(keys(first(first(quantities).agent_parameters)))
    parameter_keys_symbols = Symbol.(parameter_keys)

    # Get the dimensionality of the array
    n_samples = length(quantities)
    n_agents = length(agent_ids)
    n_parameters = length(parameter_keys)

    # Create an empty 3-dimensional AxisArray
    empty_array = Array{Float64}(undef, n_samples, n_agents, n_parameters)
    parameter_values = AxisArray(empty_array, Axis{:sample}(1:n_samples), Axis{:agent}(agent_ids), Axis{:parameter}(parameter_keys_symbols))

    # Populate the AxisArray
    for (sample_idx, sample) in enumerate(quantities)
        sample_agent_parameters = sample.agent_parameters

        for (agent_idx, agent_id) in enumerate(agent_ids)
            agent_parameters = sample_agent_parameters[agent_idx]

            for (parameter_key, parameter_key_symbol) in zip(parameter_keys, parameter_keys_symbols)
                parameter_values[sample_idx, agent_id, parameter_key_symbol] = agent_parameters[parameter_key]
            end
        end
    end

    #Extract the other values from the population model
    other_values = [quantity.other_values for quantity in quantities]
    #If they are all nothing, shorten them to one nothing
    if all(isnothing, other_values)
        other_values = nothing
    end

    #Only return other values if there are any
    if isnothing(other_values)
        return  parameter_values
    else
        return (agent_parameters = parameter_values, other_values = other_values)
    end
end
