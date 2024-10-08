#######################################################
### FUNCTION FOR GETTING SAMPLED STATE TRAJECTORIES ###
#######################################################
function get_trajectories(
    model::DynamicPPL.Model,
    chains::Chains,
    target_states::Vector{T};
    agent_parameters::AxisArray = extract_quantities(model, chains),
    inputs_per_agent::Vector = model.args.inputs_per_agent,
) where {T<:Union{String,Tuple,Any}}

    #Extract agent and make it save its history
    agent = model.args.agent
    set_save_history!(agent, true)

    #Extract dimensions
    agent_ids, parameters, samples, chains = agent_parameters.axes
    n_timesteps = length(first(inputs_per_agent)) + 1

    # Extract parameter keys
    state_key_symbols = [
        begin
            if state_key isa Tuple
                Symbol(join(state_key, tuple_separator))
            else
                Symbol(state_key)
            end
        end for state_key in target_states
    ]

    # Create an empty AxisArray
    empty_array = Array{Union{Missing,Float64}}(
        undef,
        length(agent_ids),
        length(target_states),
        n_timesteps,
        length(samples),
        length(chains),
    )
    state_trajectories = AxisArray(
        empty_array,
        Axis{:agent}(collect(agent_ids)),
        Axis{:state}(state_key_symbols),
        Axis{:timestep}(0:n_timesteps-1),
        Axis{:sample}(1:samples[end]),
        Axis{:chain}(1:chains[end]),
    )

    #For each chain, each sample, each agent
    for chain in chains
        for sample_idx in samples
            for (agent_id, inputs) in zip(agent_ids, inputs_per_agent)

                ## Set parameters in agent and give inputs ##
                # Extract parameter values for the current agent and sample
                parameter_values = Dict{Union{String,Tuple},Real}()
                for parameter in parameters

                    ## Put parameter keys in the right format
                    parameter_string = string(parameter)
                    parameter_string = split(parameter_string, tuple_separator)

                    #if the parameter is a composite parameter
                    if length(parameter_string) > 1
                        #Put it in a tuple
                        parameter_string = Tuple(string.(parameter_string))
                    else
                        #Otherwise, just take the first element
                        parameter_string = string(parameter)
                    end

                    #Store the parameter value
                    parameter_values[parameter_string] =
                        agent_parameters[agent_id, parameter, sample_idx, chain]
                end

                # Set the parameters of the agent
                set_parameters!(agent, parameter_values)
                reset!(agent)

                #Simulate forward with the agent
                give_inputs!(agent, inputs)

                #For each target state
                for state in target_states

                    # Join tuples
                    if state isa Tuple
                        state_key_symbol = Symbol(join(state, tuple_separator))
                    else
                        state_key_symbol = Symbol(state)
                    end
                    #Extract the state's history
                    state_history = get_history(agent, state)

                    #For each timestep
                    for (timestep, state_value) in enumerate(state_history)
                        #Store the value
                        state_trajectories[
                            agent_id,
                            state_key_symbol,
                            timestep,
                            sample_idx,
                            chain,
                        ] = state_value
                    end
                end
            end
        end
    end

    return state_trajectories
end
