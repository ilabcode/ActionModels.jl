```@meta
EditURL = "<unknown>/src/Advanced_use/custom_fit_model.jl"
```

# Creating a custom fit_model() function

If you wish to alter the fit_model() function to you own specific use case we have provided a compact version of the function with the basic structure for you to work on.

## The create_agent_model() function

````@example custom_fit_model
@model function create_agent_model(
    agent,
    param_priors,
    inputs,
    actions,
    impute_missing_actions,
)

    #Initialize dictionary for storing sampled parameters
    fitted_parameters = Dict()

    #Give Turing prior distributions for each fitted parameter
    for (param_key, param_prior) in param_priors
        fitted_parameters[param_key] ~ param_prior
    end

    #Set agent parameters to the sampled values
    set_parameters!(agent, fitted_parameters)
    reset!(agent)

    #For each timestep and input ASSUMING INPUTS AND ACTIONS ARE VECTORS
    for (timestep, input) in inputs
        #If no errors occur
        try

            #Get the action probability distribution from the action model
            action_probability_distribution = agent.action_model(agent, input)

            #If the action isn't missing, or if missing actions are to be imputed CAN REMOVE THIS; BUT GOOD FOR
            if !ismissing(actions[timestep]) || impute_missing_actions
                #Pass it to Turing
                actions[timestep] ~ action_probability_distribution
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
````

## The fit_model() function

````@example custom_fit_model
function fit_model(
    agent::Agent,
    param_priors::Dict,
    inputs::Vector,  # in this structure it is assumed that inputs are vectors
    actions::Vector;  # in this structure it is assumed that actions are vectors as well
    fixed_parameters::Dict = Dict(),
    sampler = NUTS(),
    n_cores::Integer = 1,
    n_iterations::Integer = 1000,
    n_chains = 2,
    impute_missing_actions::Bool = false,
)
    #Store old parameters for resetting the agent later
    old_parameters = get_parameters(agent)

    ### Set fixed parameters to agent ###
    set_parameters!(agent, fixed_parameters)

    #we set multiple cores as default. This can be reomved if desired and altered to own parallellization structure
   
    #Add worker processes
    addprocs(n_cores, exeflags = "--project")

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

    #Remove workers
    rmprocs(workers())

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
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

