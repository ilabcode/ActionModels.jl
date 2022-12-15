"""""
    fit_model(agent::Agent,inputs::Array,actions::Vector,param_priors::Dict,fixed_parameters::Dict = Dict();
    sampler = NUTS(),n_iterations = 1000, n_chains = 1,verbose = true,)

Returns a summary of the fitted parameters (parameters specified with param_prios). 

# Arguments
 - 'agent::Agent': a specified agent created with either premade agent or init_agent.
 - 'inputs:Array': array of inputs.
 - 'actions::Array': array of actions.
 - 'param_priors::Dict': priors (written as distributions) for the parameters you wish to fit.
 - 'fixed_parameters::Dict = Dict()': fixed parameters.
 - 'impute_missing_actions = false': if true, include missing actions in the fitting process.
 - 'sampler = NUTS()': specify the type of sampler.
 - 'n_iterations = 1000': iterations pr. chain.
 - 'n_chains = 1': amount of chains.
 - 'verbose = true': set to false to hide warnings
"""
function fit_model(
    agent::Agent,
    param_priors::Dict,
    inputs::Array,
    actions::Array;
    fixed_parameters::Dict = Dict(),
    sampler = NUTS(),
    n_cores::Integer = 1,
    n_iterations::Integer = 1000,
    n_chains = 2,
    verbose = true,
    show_sample_rejections = false,
    impute_missing_actions::Bool = false,
)
    #Store old parameters for resetting the agent later
    old_parameters = get_parameters(agent)

    ### CHECKS ###
    #Unless warnings are hidden
    if verbose
        #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
        if any(
            key -> !(key in keys(param_priors)) && !(key in keys(fixed_parameters)),
            keys(old_parameters),
        )
            @warn "the agent has parameters which are not specified in the fixed or sampled parameters. The agent's current parameter values are used as fixed parameters"
        end

        #If a parameter has been specified both in the fixed and sampled parameters
        if any(key -> key in keys(fixed_parameters), keys(param_priors))
            @warn "one or more parameters have been specified both in the fixed and sampled parameters. The fixed parameter value is used"

            #Remove the parameter from the sampled parameters
            for key in keys(fixed_parameters)
                if key in keys(param_priors)
                    delete!(param_priors, key)
                end
            end
        end
    end

    #If there are no parameters to sample
    if length(param_priors) == 0
        #Throw an error
        throw(
            ArgumentError(
                "no parameters to sample. Either an empty dictionary of parameter priors was passed, or all parameters with priors were also specified as fixed parameters",
            ),
        )
    end

    #If there are different amounts of inputs and actions
    if size(inputs, 1) != size(actions, 1)
        throw(
            ArgumentError(
                "there are different amounts of inputs and actions (they differ in their first dimension). This is not supported",
            ),
        )
    end

    ### Set fixed parameters to agent ###
    set_parameters!(agent, fixed_parameters)

    ### Run forward once ###
    #Initialize dictionary for populating with median parameter values
    sampled_parameters = Dict()
    #Go through each of the agent's parameters
    for (param_key, param_prior) in param_priors
        #Add the median value to the tuple
        sampled_parameters[param_key] = median(param_prior)
    end
    #Set sampled parameters
    set_parameters!(agent, sampled_parameters)
    #Reset the agent
    reset!(agent)

    try
        #Run it forwards
        test_actions = give_inputs!(agent, inputs)

        #If the model returns a different amount of actions from what was inputted
        if size(test_actions) != size(actions)
            throw(
                ArgumentError(
                    "the passed actions is a different shape from what the model returns",
                ),
            )
        end

    catch e
        #If a RejectParameters error occurs
        if e isa RejectParameters
            #Warn the user that prior median parameter values gives a sample rejection
            if verbose
                @warn "simulating with median parameter values from the prior results in a rejected sample."
            end
        else
            #Otherwise throw the actual error
            throw(e)
        end
    end

    ### FIT MODEL ###
    #If only one core is specified, use sequentiel sampling
    if n_cores == 1

        #Initialize Turing model
        model =
            create_agent_model(agent, param_priors, actions, inputs, impute_missing_actions)

        #If sample rejection warnings are to be shown
        if show_sample_rejections
            #Fit model to inputs and actions, as many separate chains as specified
            chains = map(i -> sample(model, sampler, n_iterations), 1:n_chains)

            #If sample rejection warnings are not to be shown
        else
            #Create a logger which ignores messages below error level
            sampling_logger = Logging.SimpleLogger(Logging.Error)
            #Use that logger
            chains = Logging.with_logger(sampling_logger) do

                #Fit model to inputs and actions, as many separate chains as specified
                map(i -> sample(model, sampler, n_iterations), 1:n_chains)
            end
        end

        #Otherwise, use parallel sampling
    elseif n_cores > 1

        #If the user has already created processsses
        if length(procs()) == 1
            #Add worker processes
            addprocs(n_cores, exeflags = "--project")

            #Set flag to remove the workers later
            remove_workers_at_end = true
        else
            #Error
            @warn """
            n_cores was set to > 1, but workers have already been created. No new workers were created, and the existing ones are used for parallelization.
            Note that the following variable names are broadcast to the workers: sampler agent param_priors inputs actions impute_missing_actions
            """
            #Set flag to not remove the workers later
            remove_workers_at_end = false
        end

        #Load packages on worker processes
        @everywhere @eval using ActionModels
        @everywhere @eval using Turing

        #Broadcast necessary information to workers
        @everywhere sampler = $sampler
        @everywhere agent = $agent
        @everywhere param_priors = $param_priors
        @everywhere inputs = $inputs
        @everywhere actions = $actions
        @everywhere impute_missing_actions = $impute_missing_actions

        #If sample rejection warnings are to be shown
        if show_sample_rejections
            #Fit model to inputs and actions, as many separate chains as specified
            chains = pmap(
                i -> sample(
                    create_agent_model(
                        agent,
                        param_priors,
                        inputs,
                        actions,
                        impute_missing_actions,
                    ),
                    sampler,
                    n_iterations,
                    save_state = false,
                ),
                1:n_chains,
            )

            #If sample rejection warnings are not to be shown
        else
            #Create a logger which ignores messages below error level
            sampling_logger = Logging.SimpleLogger(Logging.Error)
            #Use that logger
            chains = Logging.with_logger(sampling_logger) do

                #Fit model to inputs and actions, as many separate chains as specified
                pmap(
                    i -> sample(
                        create_agent_model(
                            agent,
                            param_priors,
                            inputs,
                            actions,
                            impute_missing_actions,
                        ),
                        sampler,
                        n_iterations,
                        save_state = false,
                    ),
                    1:n_chains,
                )
            end
        end

        #If workers are to be removed
        if remove_workers_at_end
            #Remove workers
            rmprocs(workers())
        end

    else
        throw(
            ArgumentError(
                "n_cores was set to a non-positive integer. This is not supported.",
            ),
        )
    end

    #Concatenate chains together
    chains = chainscat(chains...)

    ### CLEANUP ###
    #Reset the agent to its original parameters
    set_parameters!(agent, old_parameters)
    reset!(agent)

    #Turing includes the dictionary name 'fitted_parameters' in the parameter names, so it must be removed
    #Initialize dict for replacement names
    replacement_param_names = Dict()
    #For each parameter
    for param_key in keys(param_priors)
        #Set to replace the fitted_parameters[] version with just the parameter name
        replacement_param_names["fitted_parameters[$param_key]"] = param_key
    end
    #Input the dictionary to replace the names
    chains = replacenames(chains, replacement_param_names)

    return chains
end
