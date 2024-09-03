### FUNCTION FOR DOING FULL AGENT AND STATISTICAL MODEL ###
@model function full_model(
    agent::Agent,
    statistical_model::DynamicPPL.Model,
    inputs::Array{IA},
    actions::Array{AA},
    track_states::Bool = false,
    multiple_inputs::Bool = size(inputs, 2) > 1,
    multiple_actions::Bool = size(actions, 2) > 1,
) where {IAR<:Real,AAR<:Real,IA<:Array{IAR},AA<:Array{AAR}}

    #Check whether errors occur
    try

        #Generate the agent parameters from the statistical model
        @submodel (agents_parameters, statistical_values) = statistical_model

        #If states are tracked
        if track_states
            #Initialize a vector for storing the states of the agents
            agents_states = Vector{Dict}(undef, length(agents_parameters))
            parameters_per_agent = Vector{Dict}(undef, length(agents_parameters))
        else
            agents_states = nothing
            parameters_per_agent = nothing
        end

        ## For each agent ##
        for (agent_idx, agent_parameters) in enumerate(agents_parameters)

            #Set the agent parameters
            set_parameters!(agent, agent_parameters)
            reset!(agent)

            ## Construct input iterator ##
            #If there is only one input
            if !multiple_inputs
                #Iterate over inputs one at a time
                input_iterator = enumerate(inputs[agent_idx])
            else
                #Iterate over rows of inputs
                input_iterator = enumerate(Vector.(eachrow(inputs[agent_idx])))
            end

            #Go through each timestep 
            for (timestep, input) in input_iterator

                ## Sample actions ##

                #Get the action probability distributions from the action model
                action_distribution = agent.action_model(agent, input)

                #If there is only one action
                if !multiple_actions

                    #Sample the action from the probability distribution
                    @inbounds actions[agent_idx][timestep] ~ action_distribution

                    #Save the action to the agent in case it needs it in the future
                    @inbounds update_states!(
                        agent,
                        "action",
                        ad_val.(actions[agent_idx][timestep]),
                    )

                    #If there are multiple actions
                else
                    #Go through each separate action
                    for (action_idx, single_distribution) in enumerate(action_distribution)

                        #Sample the action from the probability distribution
                        @inbounds actions[agent_idx][timestep, action_idx] ~
                            single_distribution
                    end

                    #Add the actions to the agent in case it needs it in the future
                    @inbounds update_states!(
                        agent,
                        "action",
                        ad_val.(actions[agent_idx][timestep, :]),
                    )
                end
            end

            #If states are tracked
            if track_states
                #Save the parameters of the agent
                parameters_per_agent[agent_idx] = get_parameters(agent)
                #Save the history of tracked states for the agent
                agents_states[agent_idx] = get_history(agent)
            end
        end

        #if states are tracked
        if track_states
            #Return agents' parameters and tracked states
            return (
                agent_parameters = parameters_per_agent,
                agent_states = agents_states,
                statistical_values = statistical_values,
            )
        else
            #Otherwise, return nothing
            return nothing
        end

        #If an error occurs
    catch error
        #If it is of the custom errortype RejectParameters
        if error isa RejectParameters
            #Make Turing reject the sample
            Turing.@addlogprob!(-Inf)
        else
            #Otherwise, just throw the error
            rethrow(error)
        end
    end
end



#######################################################################################################
### SIMPLE STATISTICAL MODEL WHERE AGENTS ARE INDEPENDENT AND THEIR PARAMETERS HAVE THE SAME PRIORS ###
#######################################################################################################
@model function simple_statistical_model(
    prior::Dict{T,D},
    n_agents::I,
    agent_parameters::Vector{Dict{Any,Real}} = [Dict{Any,Real}() for _ = 1:n_agents],
) where {T<:Union{String,Tuple,Any},D<:Distribution,I<:Int}

    #Create container for sampled parameters
    parameters = Dict{Any,Vector{Real}}()

    #Go through each of the parameters in the prior
    for (parameter, distribution) in prior
        #And sample a value for each agent
        parameters[parameter] ~ filldist(distribution, n_agents)
    end

    #Go through each parameter
    for (parameter, values) in parameters
        #Go through each agent
        for (agent_idx, value) in enumerate(values)
            #Store the value in the right way
            agent_parameters[agent_idx][parameter] = value
        end
    end

    return agent_parameters, nothing
end
