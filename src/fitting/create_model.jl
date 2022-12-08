"""
"""
@model function create_agent_model(agent, param_priors, inputs, actions, impute_missing_actions)

    #Initialize dictionary for storing sampled parameters
    fitted_params = Dict()

    #Give Turing prior distributions for each fitted parameter
    for (param_key, param_prior) in param_priors
        fitted_params[param_key] ~ param_prior
    end

    #Set agent parameters to the sampled values
    set_params!(agent, fitted_params)
    reset!(agent)

    #If the input is a single vector
    if inputs isa Vector
        #Prepare to through one value at a time
        iterator = enumerate(inputs)
    else
        #For an array, go through each row
        iterator = enumerate(eachrow(inputs))
    end

    #For each timestep and input
    for (timestep, input) in iterator
        #If no errors occur
        try

            #Get the action probability distribution from the action model
            action_probability_distribution = agent.action_model(agent, input)

            #If only a single action is made at each timestep
            if actions isa Vector

                #If the action isn't missing, or if missing actions are to be imputed
                if !ismissing(actions[timestep]) || impute_missing_actions
                    #Pass it to Turing
                    actions[timestep] ~ action_probability_distribution
                end

                #If multiple actions are made at each timestep
            elseif actions isa Array

                #Go throgh each action distribution
                for (action_indx, distribution) in
                    enumerate(action_probability_distribution)

                    #If the action isn't missing, or if missing actions are to be imputed
                    if !ismissing(actions[timestep, action_indx]) ||
                    impute_missing_actions
                        #Pass it to Turing
                        actions[timestep, action_indx] ~ distribution
                    end
                end
            end
        catch e
            #If the custom errortype RejectParameters occurs
            if e isa RejectParameters
                #Make Turing reject the sample
                Turing.@addlogprob!(-Inf)
            else
                #Otherwise, just throw the error
                throw(e)
            end
        end
    end
end
