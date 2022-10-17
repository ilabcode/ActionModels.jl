"""""
function fit_model(
    agent::AgentStruct,
    inputs::Array,
    actions::Vector,
    param_priors::Dict,
    fixed_params::Dict = Dict();
    sampler = NUTS(),
    n_iterations = 1000,
    n_chains = 1,
    verbose = true,
)
Function to fit an agent's parameters.
"""
function fit_model(
    agent::AgentStruct,
    inputs::Array,
    actions::Array,
    param_priors::Dict,
    fixed_params::Dict = Dict();
    impute_missing_actions = false,
    sampler = NUTS(),
    n_iterations = 1000,
    n_chains = 1,
    verbose = true,
)
    #If there are different amounts of inputs and actions
    if size(inputs, 1) != size(actions, 1)
        throw(
            ArgumentError(
                "inputs and actions differs in their first dimension. This is not supported",
            ),
        )
    end

    #Store old parameters 
    old_params = get_params(agent)

    #Unless warnings are hidden
    if verbose
        #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
        if any(
            key -> !(key in keys(param_priors)) && !(key in keys(fixed_params)),
            keys(old_params),
        )
            #Make a warning
            @warn "the agent has parameters which are not specified in the fixed or sampled parameters. The agent's current parameter values are used as fixed parameters"
        end
    end


    ### Run forward once as testrun ###
    #Initialize dictionary for populating with median parameter values
    sampled_params = Dict()
    #Go through each of the agent's parameters
    for (param_key, param_prior) in param_priors
        #Add the median value to the tuple
        sampled_params[param_key] = median(param_prior)
    end
    #Set parameters in agent
    set_params!(agent, sampled_params)
    #Set fixed parameters
    set_params!(agent, fixed_params)
    #Reset the agent
    reset!(agent)
    #Run it forwards
    test_actions = give_inputs!(agent, inputs)

    #If the model returns a different amount of actions from what was inputted
    if size(test_actions) != size(actions)
        throw(
            ArgumentError(
                "The passed actions is a different shape from what the model returns",
            ),
        )
    end


    ### Fit model ###
    #Initialize dictionary for storing sampled parameters
    fitted_params = Dict()

    #Create turing model macro for parameter estimation
    @model function fit_agent(actions)

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
                        if !ismissing(actions[timestep, action_indx]) || impute_missing_actions
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

    #If warnings are to be ignored
    if !verbose
        #Create a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
        #Use that logger
        chains = Logging.with_logger(sampling_logger) do

            #Fit model to inputs and actions, as many separate chains as specified
            map(i -> sample(fit_agent(actions), sampler, n_iterations), 1:n_chains)

        end
    else
        #Fit model to inputs and actions, as many separate chains as specified
        chains = map(i -> sample(fit_agent(actions), sampler, n_iterations), 1:n_chains)
    end

    #Concatenate chains together
    chains = chainscat(chains...)

    #Reset the agent to its original parameters
    set_params!(agent, old_params)
    reset!(agent)


    ## Set pretty parameter names ###
    #Since Turing includes the dictionary name 'fitted_params', we remove it
    #Initialize dict for replacement names
    replacement_param_names = Dict()
    #For each parameter
    for param_key in keys(param_priors)
        #Set to replace the fitted_params[] version with just the parameter name
        replacement_param_names["fitted_params[$param_key]"] = param_key
    end
    #Input the dictionary to replace the names
    chains = replacenames(chains, replacement_param_names)

    return chains
end