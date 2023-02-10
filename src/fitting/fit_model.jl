"""""
    fit_model(agent::Agent, inputs::Array, actions::Vector, param_priors::Dict, kwargs...)

Use Turing to fit the parameters of an agent to a set of inputs and corresponding actions.

# Arguments
 - 'agent::Agent': an ActionModels agent object created with either premade_agent or init_agent.
 - 'param_priors::Dict': dictionary containing priors (as Distribution objects) for fitted parameters. Keys are parameter names, values are priors.
 - 'inputs:Array': array of inputs. Each row is a timestep, and each column is a single input value.
 - 'actions::Array': array of actions. Each row is a timestep, and each column is a single action.
 - 'fixed_parameters::Dict = Dict()': dictionary containing parameter values for parameters that are not fitted. Keys are parameter names, values are priors. For parameters not specified here and without priors, the parameter values of the agent are used instead.
 - 'sampler = NUTS()': specify the type of Turing sampler.
 - 'n_cores = 1': set number of cores to use for parallelization. If set to 1, no parallelization is used.
 - 'n_iterations = 1000': set number of iterations per chain.
 - 'n_chains = 2': set number of amount of chains.
 - 'verbose = true': set to false to hide warnings.
 - 'show_sample_rejections = false': set whether to show warnings whenever samples are rejected.
 - 'impute_missing_actions = false': set whether the values of missing actions should also be estimated by Turing.

 # Examples
```julia
#Create a premade agent: binary Rescorla-Wagner
agent = premade_agent("premade_binary_rw_softmax")

#Set priors for the learning rate
param_priors = Dict("learning_rate" => Uniform(0, 1))

#Set inputs and actions
inputs = [1, 0, 1]
actions = [1, 1, 0]

#Fit the model
fit_model(agent, param_priors, inputs, actions, n_chains = 1, n_iterations = 10)
```
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
            create_agent_model(agent, param_priors, inputs, actions, impute_missing_actions)

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
