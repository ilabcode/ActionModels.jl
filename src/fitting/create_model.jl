"""
    create_agent_model(agent,param_priors,inputs,actions,impute_missing_actions)

Create a Turing model object used for fitting an ActionModels agent.
"""
@model function create_agent_model(
    agent::Agent,
    hierarchical_parameters_information::Dict,
    agent_parameters_information::Dict,
    inputs::Dict,
    actions::Dict,
    groups::Vector,
    group_dependency_levels::Vector,
    multiple_inputs,
    multiple_actions,
    impute_missing_actions::Bool,
)
    #Initialize dictionaries for storing sampled parameters
    hierarchical_parameters = Dict()
    agent_parameters = Dict()

    #Go from few to many group dependencies
    for n_dependencies in group_dependency_levels

        #Go through each parameter with amount of group dependencies
        for (parameter_key, parameter_info) in
            hierarchical_parameters_information[n_dependencies]

            #Set up a sub-dictionary for the value of this parameter in each group
            hierarchical_parameters[parameter_key] = Dict()

            #Go through each group that the parameter belongs to
            for group in parameter_info.group_levels

                #If the parameter does not depend on a higher level
                if !parameter_info.multilevel_dependent

                    #Sample the parameter from the given prior distribution
                    hierarchical_parameters[parameter_key][group] ~
                        parameter_info.distribution

                    #Otherwise if it depends on a higher-level parameter
                else

                    #Create empty vector for storing distribution parameter values
                    distribution_parameters = []

                    #Go through each parameter in the higher-level distribution
                    for distribution_parameter_key in parameter_info.parameters

                        #Get the dictionary of parameter values in the parent
                        parent_parameters =
                            hierarchical_parameters[distribution_parameter_key]

                        #Get out the different groups that the parent can belong to
                        parent_parameters_groups = collect(keys(parent_parameters))

                        #Find the group of the parent which the current parameter is derived from
                        higher_group = parent_parameters_groups[findall(
                            x -> all(x .∈ [group]),
                            parent_parameters_groups,
                        )][1]

                        #Store the parameter value from that group, which should already be sampled
                        push!(distribution_parameters, parent_parameters[higher_group])
                    end

                    #Give the distribution parameters to the specified distribution, and sample this parameter
                    hierarchical_parameters[parameter_key][group] ~
                        parameter_info.distribution(distribution_parameters...)
                end
            end
        end
    end

    #If no errors occur
    try

        #Go through each group
        for group in groups

            ### Sample parameters ###
            #Initialize a within-group parameter dictionary
            agent_parameters[group] = Dict()

            #Sample within-group parameters from the priors
            for (param_key, parameter_info) in agent_parameters_information

                #If the parameter does not depend on a higher level
                if !parameter_info.multilevel_dependent

                    #Sample the parameter from the given prior
                    agent_parameters[group][param_key] ~ parameter_info.distribution

                    #Otherwise if it depends on a higher-level parameter
                else

                    #Create empty vector for storing distribution parameter values
                    distribution_parameters = []

                    #Go through each parameter in the higher-level distribution
                    for distribution_parameter_key in parameter_info.parameters

                        #Get the dictionary of parameter values in the parent
                        parent_parameters =
                            hierarchical_parameters[distribution_parameter_key]

                        #Get out the different groups that the parent can belong to
                        parent_parameters_groups = collect(keys(parent_parameters))

                        #Find the group of the parent which the current parameter is derived from
                        higher_group = parent_parameters_groups[findall(
                            x -> all(x .∈ [group]),
                            parent_parameters_groups,
                        )][1]

                        #Store the parameter value from that group, which should already be sampled
                        push!(distribution_parameters, parent_parameters[higher_group])
                    end

                    #And sample the agents' parameters from the distribution
                    agent_parameters[group][param_key] ~
                        parameter_info.distribution(distribution_parameters...)
                end
            end

            #Set agent parameters to the sampled values
            set_parameters!(agent, agent_parameters[group])
            reset!(agent)


            ### Give inputs ###
            #If there is only one input
            if !multiple_inputs
                #Iterate over inputs one at a time
                iterator = enumerate(inputs[group])
            else
                #Iterate over rows of inputs
                iterator = enumerate(eachrow(inputs[group]))
            end

            #Go through each timestep
            for (timestep, input) in iterator

                #Get the action probability distribution from the action model
                action_distribution = agent.action_model(agent, input)

                ### Sample actions ###
                #If there is only a single action
                if !multiple_actions

                    #If the action isn't missing, or if missing actions are to be imputed
                    if !ismissing(actions[group][timestep]) || impute_missing_actions
                        #Sample the action from the probability distribution
                        actions[group][timestep] ~ action_distribution
                    end

                    #If there are multiple actions
                else
                    #Go through each separate action
                    for (action_idx, single_distribution) in enumerate(action_distribution)

                        #If the action isn't missing, or if missing actions are to be imputed
                        if !ismissing(actions[group][timestep, action_idx]) ||
                           impute_missing_actions

                            #Sample the action from the probability distribution
                            actions[group][timestep, action_idx] ~ single_distribution
                        end
                    end
                end
            end
        end

        #If an error occurs
    catch e
        #If it is of the custom errortype RejectParameters
        if e isa RejectParameters
            #Make Turing reject the sample
            Turing.@addlogprob!(-Inf)
        else
            #Otherwise, just throw the error
            throw(e)
        end
    end
end
