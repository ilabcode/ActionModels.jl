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
Function to fit an agent parameters.
"""
function fit_model(
    agent::AgentStruct,
    inputs::Vector,
    actions::Vector,
    param_priors::Dict,
    fixed_params::Dict = Dict();
    skip_missing_actions = true,
    sampler = NUTS(),
    n_iterations = 1000,
    n_chains = 1,
    verbose = true,
)
    #If there are different amounts of inputs and actions
    if size(inputs,1) != size(actions,1)
        throw(ArgumentError("inputs and actions differs in their first dimension. This is not supported"))
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

        #For each input
        for (input_indx, input) in enumerate(inputs)

            #If no errors occur
            try
                #Get the action probability distribution from the action model
                action_probability_distribution = agent.action_model(agent, input)

                #If only a single action probability distribution was returned
                if action_probability_distribution isa Distribution

                    #If the action isn't missing, or if missing actions arent' skipped
                    if !ismissing(actions[input_indx]) || !skip_missing_actions
                        #Pass it to Turing
                        actions[input_indx] ~ action_probability_distribution
                    end

                    #If a list of action probabilities were returned
                elseif action_probability_distribution isa Vector{<:Distribution}
                    #Go throgh each returned distribution
                    for (response_indx, distribution) in
                        enumerate(action_probability_distribution)
                        #Add it one at a time
                        actions[input_indx, response_indx] ~ distribution
                    end
                else
                    throw(
                        ArgumentError(
                            "The action model does not return a Distribution, nor a Vector{<:Distributions}. This is not supported",
                        ),
                    )
                end
            catch e
                #If the custom errortype ParamError occurs
                if e isa ParamError
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