"""
    create_agent_model(agent,param_priors,inputs,actions,impute_missing_actions)

Create a Turing model object used for fitting an ActionModels agent.
"""
@model function create_agent_model(
    agent::Agent,
    multilevel_parameters_info::Vector,
    agent_parameters_info::Vector,
    inputs::Dict,
    actions::Dict,
    multilevel_groups::Vector,
    multiple_inputs::Bool,
    multiple_actions::Bool,
    impute_missing_actions::Bool,
)
    #Initialize dictionaries for storing sampled parameters
    multilevel_parameters = Dict()
    agent_parameters = Dict()

    #Go through each multilevel parameter
    for parameter_info in multilevel_parameters_info

        #Set up a sub-dictionary for the value of this parameter in each group
        multilevel_parameters[parameter_info.name] = Dict()

        #Go through each group that the parameter belongs to
        for group in parameter_info.group_levels

            #If the parameter does not depend on a higher level
            if !parameter_info.multilevel_dependent

                #Sample the parameter from the given prior distribution
                multilevel_parameters[parameter_info.name][group] ~
                    parameter_info.distribution

                #Otherwise if it depends on a higher-level parameter
            else

                #Create empty vector for storing distribution parameter values
                distribution_parameters = []

                #Go through each parameter in the higher-level distribution
                for distribution_parameter_key in parameter_info.parameters

                    #Get the dictionary of parameter values in the parent
                    parent_parameters = multilevel_parameters[distribution_parameter_key]

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
                multilevel_parameters[parameter_info.name][group] ~
                    parameter_info.distribution(distribution_parameters...)
            end
        end
    end

    #If no errors occur
    try

        #Go through each group
        for group in multilevel_groups

            ### Sample parameters ###
            #Initialize a within-group parameter dictionary
            agent_parameters[group] = Dict()

            #Sample within-group parameters from the priors
            for parameter_info in agent_parameters_info

                #If the parameter does not depend on a higher level
                if !parameter_info.multilevel_dependent

                    #Sample the parameter from the given prior
                    agent_parameters[group][parameter_info.name] ~
                        parameter_info.distribution

                    #Otherwise if it depends on a higher-level parameter
                else

                    #Create empty vector for storing distribution parameter values
                    distribution_parameters = []

                    #Go through each parameter in the higher-level distribution
                    for distribution_parameter_key in parameter_info.parameters

                        #Get the dictionary of parameter values in the parent
                        parent_parameters =
                            multilevel_parameters[distribution_parameter_key]

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
                    agent_parameters[group][parameter_info.name] ~
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
