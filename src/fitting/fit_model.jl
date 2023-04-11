###########################
### FITTING A DATAFRAME ###
###########################
"""""
    fit_model(agent::Agent, inputs::Array, actions::Vector, priors::Dict, kwargs...)

Use Turing to fit the parameters of an agent to a set of inputs and corresponding actions.

# Arguments
 - 'agent::Agent': an ActionModels agent object created with either premade_agent or init_agent.
 - 'priors::Dict': dictionary containing priors (as Distribution objects) for fitted parameters. Keys are parameter names, values are priors.
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
priors = Dict("learning_rate" => Uniform(0, 1))

#Set inputs and actions
inputs = [1, 0, 1]
actions = [1, 1, 0]

#Fit the model
fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 10)
```
"""
function fit_model(
    agent::Agent,
    priors::Dict,
    data::DataFrame;
    group_cols::Vector = [],
    input_cols::Vector = [:input],
    action_cols::Vector = [:action],
    fixed_parameters::Dict = Dict(),
    sampler = NUTS(),
    n_cores::Integer = 1,
    n_iterations::Integer = 1000,
    n_chains::Integer = 2,
    verbose::Bool = true,
    show_sample_rejections::Bool = false,
    impute_missing_actions::Bool = false,
    sampler_kwargs...,
)
    ### SETUP ###

    #Convert column names to symbols
    group_cols = Symbol.(group_cols)
    input_cols = Symbol.(input_cols)
    action_cols = Symbol.(action_cols)

    ## Store old parameters for resetting the agent later ##
    old_parameters = get_parameters(agent)

    ## Set fixed parameters to agent ##
    set_parameters!(agent, fixed_parameters)

    ## Run checks ##
    prefit_checks(
        agent = agent,
        data = data,
        priors = priors,
        group_cols = group_cols,
        input_cols = input_cols,
        action_cols = action_cols,
        fixed_parameters = fixed_parameters,
        old_parameters = old_parameters,
        n_cores = n_cores,
        verbose = verbose,
    )

    ## Set logger ##
    #If sample rejection warnings are to be shown
    if show_sample_rejections
        #Use a standard logger
        sampling_logger = Logging.SimpleLogger()
    else
        #Use a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
    end

    ## Store whether there are multiple inputs and actions ##
    multiple_inputs = length(input_cols) > 1
    multiple_actions = length(action_cols) > 1

    ## Extract information from dataframe ##
    #Create empty containers for grouped actions and inputs
    inputs = Dict()
    actions = Dict()
    group_combinations = []

    #Go through each group
    for (group_key, group_data) in pairs(groupby(data, group_cols))
        #Store inputs and actions in matreces
        inputs[Tuple(group_key)] = Array(group_data[:, input_cols])
        actions[Tuple(group_key)] = Array(group_data[:, action_cols])

        #Add the group_key
        push!(group_combinations, Tuple(group_key))
    end

    ## Create list of amount of groups to be dependent on ##
    group_dependency_levels = collect(0:length(group_cols))

    ## Create dictionary with group levels per group ##
    #Initialize empty dict
    group_levels = Dict()

    #For each column in the group columns
    for group_col in group_cols
        #Save the unique levels from the column
        group_levels[group_col] = unique(data[:, group_col])
    end

    ## Create dictionary with the groups that each parameter depends on ##
    #Create dictionary where each parameter has all levels
    group_dependencies = Dict(zip(keys(priors), repeat([copy(group_cols)], length(priors))))

    #Go through each specified prior
    for (parameter_key, info) in priors

        #For hierarchically dependent parameters 
        if info isa Multilevel

            #If the specified group is not in the group columns
            if info.group ∉ group_cols
                throw(
                    ArgumentError(
                        "the parameter $parameter_key depends on the group $(info.group), but this group is not in the specified group columns",
                    ),
                )
            end

            #Recursively remove its group from all higher parameters 
            remove_higher_dependencies!(info, group_dependencies, priors, [])
        end
    end

    ## Create dictionary with all necessary information about parameters ##
    #Initialize empty dict
    hierarchical_parameters_information = Dict()

    #Create subdicts for each 
    for n in group_dependency_levels
        hierarchical_parameters_information[n] = Dict()
    end

    #Go through each prior
    for (parameter_key, info) in priors

        #Get out the group dependencies for the parameter
        parameter_group_dependencies = group_dependencies[parameter_key]

        #If there are no group dependencies
        if isempty(parameter_group_dependencies)
            #There are no group levels
            parameter_group_levels = [()]

            #Otherwise
        else
            #Get out all group combinations for the parameter
            parameter_group_levels = collect(
                Iterators.product(
                    map(key -> group_levels[key], parameter_group_dependencies)...,
                ),
            )
        end

        #For multilevel dependent parameters
        if info isa Multilevel
            #Set the multilevel dependency to true
            multilevel_dependent = true

            #Extract the distribution and parameters
            distribution = info.distribution
            parameters = info.parameters

            #For normal parameters
        else
            #Set the multilevel dependency to false
            multilevel_dependent = false

            #Extract the distribution and parameters
            distribution = info
            parameters = []
        end

        #Save the information for using when fitting
        hierarchical_parameters_information[length(parameter_group_dependencies)][parameter_key] =
            (
                group_dependencies = parameter_group_dependencies,
                group_levels = parameter_group_levels,
                multilevel_dependent = multilevel_dependent,
                distribution = distribution,
                parameters = parameters,
            )
    end

    #Extract the information for the parameters at the agent level
    agent_parameters_information =
        hierarchical_parameters_information[pop!(group_dependency_levels)]


    ### FIT MODEL ###
    #If only one core has been specified, use sequential sampling
    if n_cores == 1

        ### FIT MODEL ###
        #Initialize Turing model
        model = create_agent_model(
            agent,
            hierarchical_parameters_information,
            agent_parameters_information,
            inputs,
            actions,
            group_combinations,
            group_dependency_levels,
            multiple_inputs,
            multiple_actions,
            impute_missing_actions,
        )

        #Use the logger specified earlier
        chains = Logging.with_logger(sampling_logger) do

            #Fit model to data, as many chains as specified
            map(i -> sample(model, sampler, n_iterations, sampler_kwargs...), 1:n_chains)
        end

        #Otherwise, use parallel sampling
    elseif n_cores > 1

        #If the user has not already created processses
        if length(procs()) == 1
            #Add worker processes
            addprocs(n_cores, exeflags = "--project")

            #Set flag to remove the workers later
            remove_workers_at_end = true

            #Otherwise
        else
            #Warn them that ActionModels is using the workers already created
            @warn """
            n_cores was set to > 1, but workers have already been created. No new workers were created, and the existing ones are used for parallelization.
            Note that the following variable names are broadcast to the workers:
            sampler agent hierarchical_parameters_information agent_parameters_information inputs actions
            group_combinations group_dependency_levels multiple_inputs multiple_actions impute_missing_actions
            """
            #Set flag to not remove the workers later
            remove_workers_at_end = false
        end

        #Load packages on worker processes
        @everywhere @eval using ActionModels, Turing
        #Broadcast necessary information to workers
        @everywhere sampler = $sampler
        @everywhere agent = $agent
        @everywhere hierarchical_parameters_information =
            $hierarchical_parameters_information
        @everywhere agent_parameters_information = $agent_parameters_information
        @everywhere inputs = $inputs
        @everywhere actions = $actions
        @everywhere group_combinations = $group_combinations
        @everywhere group_dependency_levels = $group_dependency_levels
        @everywhere multiple_inputs = $multiple_inputs
        @everywhere multiple_actions = $multiple_actions
        @everywhere impute_missing_actions = $impute_missing_actions

        #Use the specified logger
        chains = Logging.with_logger(sampling_logger) do

            #Fit model to inputs and actions, as many separate chains as specified
            pmap(
                i -> sample(
                    create_agent_model(
                        agent,
                        hierarchical_parameters_information,
                        agent_parameters_information,
                        inputs,
                        actions,
                        group_combinations,
                        group_dependency_levels,
                        multiple_inputs,
                        multiple_actions,
                        impute_missing_actions,
                    ),
                    sampler,
                    n_iterations,
                    save_state = false,
                    sampler_kwargs...,
                ),
                1:n_chains,
            )
        end

        #If workers are to be removed
        if remove_workers_at_end
            #Remove workers
            rmprocs(workers())
        end
    end

    #Concatenate chains together
    chains = chainscat(chains...)

    ### CLEANUP ###
    #Initialize dict for replacement names
    replacement_names = Dict()

    ## Set replacement names for hierarchical parameters ##
    #Go through each amount of group dependencies
    for n_dependencies in group_dependency_levels

        #Go through each hierarchical parameter
        for (parameter_key, parameter_info) in
            hierarchical_parameters_information[n_dependencies]

            #Go through each group with that parameter
            for group in parameter_info.group_levels

                #If there are no group dependencies
                if isempty(group)
                    #Don't print anything
                    group_string = ""

                    #If there is only one group
                elseif length(group) == 1
                    #Extract the group from the tuple it is in
                    group_string = group[1]

                    #If there are multiple group dependencies
                else
                    #Print the whole group dependency tuple
                    group_string = group
                end

                #Set a replacement name
                replacement_names["hierarchical_parameters[$parameter_key][$group]"] = "$group_string $parameter_key"

            end
        end
    end

    ## Set replacement names for agent parameters ##
    #Go through each agent parameter
    for (parameter_key, parameter_info) in agent_parameters_information

        #Go through each group with that parameter
        for group in parameter_info.group_levels

            #If there are no group dependencies
            if isempty(group)
                #Don't print anything
                group_string = ""

                #If there is only one group
            elseif length(group) == 1
                #Extract the group from the tuple it is in
                group_string = group[1]

                #If there are multiple group dependencies
            else
                #Print the whole group dependency tuple
                group_string = group
            end

            #Set a replacement name
            replacement_names["agent_parameters[$group][$parameter_key]"] = "$group_string $parameter_key"

        end
    end

    #Input the dictionary to replace the names
    chains = replacenames(chains, replacement_names)

    return chains
end


###########################
### FITTING TWO VECTORS ###
###########################
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
    priors::Dict,
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

    #Create column names
    input_cols = map(x -> "input$x", 1:size(inputs, 2))
    action_cols = map(x -> "action$x", 1:size(actions, 2))

    #Create dataframe of the inputs and actions
    data = DataFrame(hcat(inputs, actions), vcat(input_cols, action_cols))

    #Run the main fit model function
    chains = fit_model(
        agent,
        priors,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        fixed_parameters = fixed_parameters,
        sampler = sampler,
        n_cores = n_cores,
        n_iterations = n_iterations,
        n_chains = n_chains,
        verbose = verbose,
        show_sample_rejections = show_sample_rejections,
        impute_missing_actions = impute_missing_actions,
    )

    return chains
end
