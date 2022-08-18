"""""
function fit_model(
    agent::AgentStruct,
    inputs::Vector{Float64},
    responses::Union{Vector{Float64},Missing},
    params_priors_list = (;)::NamedTuple{Distribution},
    fixed_params_list = (;)::NamedTuple{String,Real},
    sampler = NUTS(),
    iterations = 1000,
)
Function to fit an agent parameters.
"""
function fit_model(
    agent::AgentStruct,
    inputs::Array,
    responses::Vector,
    param_priors::Dict,
    fixed_params::Dict = Dict();
    sampler = NUTS(),
    n_iterations = 1000,
    n_chains = 1,
    hide_warnings = false,
)

    #Store old parameters 
    old_params = get_params(agent)

    ### Run forward once as testrun ###
    #Set fixed parameters
    set_params!(agent, fixed_params)

    #Initialize dictionary for populating with median parameter values
    sampled_params = Dict()

    #Go through each of the agent's parameters
    for (param_key, param_prior) in param_priors
        #Add the median value to the tuple
        sampled_params[param_key] = median(param_prior)
    end

    #Set parameters in agent
    set_params!(agent, sampled_params)

    #Reset the agent
    reset!(agent)

    #Run it forwards
    give_inputs!(agent, inputs)

    ### Fit model ###
    #Initialize dictionary for storing sampled parameters
    fitted_params = Dict()

    #Create turing model macro for parameter estimation
    @model function fit_agent(responses)

        #Give Turing prior distributions for each fitted parameter
        for (param_key, param_prior) in param_priors
            fitted_params[param_key] ~ param_prior
        end

        #Set agent parameters to the sampled values
        set_params!(agent, fitted_params)
        reset!(agent)

        #For each input
        for input_indx in range(1, length(inputs))
            #If no errors occur
            try
                #Get the action probability distribution from the action model
                action_probability_distribution =
                    agent.action_model(agent, inputs[input_indx])

                #If only a single action probability distribution was returned
                if length(action_probability_distribution) == 1
                    #Pass it to Turing
                    responses[input_indx] ~ action_probability_distribution
                else
                    #Go throgh each returned distribution
                    for response_indx = 1:length(action_probability_distribution)
                        #Add it one at a time
                        responses[input_indx, response_indx] ~
                            action_probability_distribution[response_indx]
                    end
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
    if hide_warnings
        #Create a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
        #Use that logger
        chains = Logging.with_logger(sampling_logger) do

            #Fit model to inputs and responses, as many separate chains as specified
            map(i -> sample(fit_agent(responses), sampler, n_iterations), 1:n_chains)

        end
    else
        #Fit model to inputs and responses, as many separate chains as specified
        chains = map(i -> sample(fit_agent(responses), sampler, n_iterations), 1:n_chains)
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