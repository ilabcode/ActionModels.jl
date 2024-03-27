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
    independent_group_cols::Vector = [],
    multilevel_group_cols::Vector = [],
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
    independent_group_cols = Symbol.(independent_group_cols)
    multilevel_group_cols = Symbol.(multilevel_group_cols)
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
        independent_group_cols = independent_group_cols,
        multilevel_group_cols = multilevel_group_cols,
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
    multilevel = length(multilevel_group_cols) > 0

    ## Structure multilevel parameter information ##
    general_parameters_info = extract_structured_parameter_info(;
        priors = priors,
        multilevel_group_cols = multilevel_group_cols,
    )

    ## Structure data ##
    #Group data into independent groups
    independence_grouped_dataframe = groupby(data, independent_group_cols)

    #Initialize vectors of independent datasets and their keys
    independent_groups_keys = []
    independent_groups_info = []

    #Go through data for each independent group
    for (independent_group_key, independent_group_data) in
        pairs(independence_grouped_dataframe)

        ## Get the key for the independent group ##
        #If there is only independent group distinction
        if length(independent_group_cols) == 1

            #Get out that group level as key
            independent_group_key = independent_group_data[1, first(independent_group_cols)]

            #If there are multiple
        else
            #Save the key for the independent group as a tuple
            independent_group_key = Tuple(independent_group_data[1, independent_group_cols])
        end

        #Add it as a key
        push!(independent_groups_keys, independent_group_key)

        #Extract and save data as dicts of multilevel grouped arrays
        push!(
            independent_groups_info,
            extract_structured_data(
                data = independent_group_data,
                multilevel_group_cols = multilevel_group_cols,
                input_cols = input_cols,
                action_cols = action_cols,
                general_parameters_info = general_parameters_info,
            ),
        )
    end

    ## Copy the fitting info for each chain that is to be sampled ##
    fit_info_all = repeat(independent_groups_info, n_chains)

    ### FIT MODEL ###
    #If only one core has been specified, use sequential sampling
    if n_cores == 1

        #Use the logger specified earlier
        chains = Logging.with_logger(sampling_logger) do

            #Fit model to data, as many sets of fititng info as specified
            map(
                fit_info -> sample(
                    create_agent_model(
                        agent,
                        fit_info.multilevel_parameters_info,
                        fit_info.agent_parameters_info,
                        fit_info.inputs,
                        fit_info.actions,
                        fit_info.multilevel_groups,
                        multiple_inputs,
                        multiple_actions,
                        impute_missing_actions,
                    ),
                    sampler,
                    n_iterations;
                    sampler_kwargs...,
                ),
                fit_info_all,
            )
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
            agent fit_info_all multiple_inputs multiple_actions impute_missing_actions sampler
            """
            #Set flag to not remove the workers later
            remove_workers_at_end = false
        end

        #Load ActionModels, Turing and sister packages on worker processes
        @everywhere @eval using ActionModels, Turing
        #Broadcast necessary information to workers
        @everywhere agent = $agent
        @everywhere fit_info_all = $fit_info_all
        @everywhere multiple_inputs = $multiple_inputs
        @everywhere multiple_actions = $multiple_actions
        @everywhere impute_missing_actions = $impute_missing_actions
        @everywhere sampler = $sampler

        #Use the specified logger
        chains = Logging.with_logger(sampling_logger) do

            #Fit model to inputs and actions, as many separate chains as specified
            pmap(
                fit_info -> sample(
                    create_agent_model(
                        agent,
                        fit_info.multilevel_parameters_info,
                        fit_info.agent_parameters_info,
                        fit_info.inputs,
                        fit_info.actions,
                        fit_info.multilevel_groups,
                        multiple_inputs,
                        multiple_actions,
                        impute_missing_actions,
                    ),
                    sampler,
                    n_iterations;
                    save_state = false,
                    sampler_kwargs...,
                ),
                fit_info_all,
            )
        end

        #If workers are to be removed
        if remove_workers_at_end
            #Remove workers
            rmprocs(workers())
        end
    end

    ### CLEANUP ###

    ## Combine chains correctly ##
    #If there was only one dataset to be fit
    if length(independent_group_cols) == 0

        #Concatenate all the chains together
        results = chainscat(chains...)

        #Rename parameter names
        results = rename_chains(results, first(independent_groups_info))

        #If there were multiple independent groups
    else

        #Initialize dictionary for results for each group
        results = Dict()

        #Go through each independent group
        for (group_indx, (independent_group_key, independent_group_info)) in
            enumerate(zip(independent_groups_keys, independent_groups_info))

            #Get the chains belonging to that group and concatenate them
            group_chains =
                chainscat(chains[group_indx:length(independent_groups_info):end]...)

            #Rename parameter names
            group_chains = rename_chains(group_chains, independent_group_info)

            #Concatenate them, and store them
            results[independent_group_key] = group_chains
        end
    end

    return results
end


############################
### FITTING TWO MATRECES ###
############################
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


#########################################
### FITTING A DATAFRAME AND TuringGLM ###
#########################################

function fit_model(
    agent_model::Agent,
    statistical_model::Union{TuringGLM.FormulaTerm, Vector{TuringGLM.FormulaTerm}},
    data,
    priors::TuringGLM.Prior = TuringGLM.DefaultPrior();
    input_cols::Union{Vector,String,Symbol},
    action_cols::Union{Vector,String,Symbol},
    grouping_cols::Union{Vector,String,Symbol},
    sampler = NUTS(),
    n_cores::Integer = 1,
    n_iterations::Integer = 1000,
    n_chains::Integer = 2,
    verbose::Bool = true,
    show_sample_rejections::Bool = false,
    impute_missing_actions::Bool = false,
    sampler_kwargs...,
)

    input_cols = Symbol.(input_cols)
    action_cols = Symbol.(action_cols)


    # TODO
    # prefit_checks(
    #     agent_model = agent_model,
    #     statistical_model = statistical_model,
    #     data = data,
    #     priors = priors,
    #     input_cols = input_cols,
    #     action_cols = action_cols,
    #     n_cores = n_cores,
    #     verbose = verbose,
    # )

    ## Set logger ##
    #If sample rejection warnings are to be shown
    if show_sample_rejections
        #Use a standard logger
        sampling_logger = Logging.SimpleLogger()
    else
        #Use a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
    end



    inputs = []
    actions = []
    for (run_key, run_df) in pairs(groupby(data, :id))
        push!(inputs, Array(run_df[:, input_cols]))
        push!(actions, Array(run_df[:, action_cols]))
    end

    @show inputs
    @show actions



    # TODO: check if statistical models differ within time series (we assume they don't)
    statistical_data = unique(data, grouping_cols)

    (statmodel, X) = ActionModels.statistical_model_turingglm(statistical_model, statistical_data)

    @model function do_full_model(
        agent_model, statmodel, statistical_data, inputs, actions, X
        )
        #learning_rate ~ filldist(Uniform(0,1), 3)
        @submodel learning_rate = statmodel(X)
        for agent_idx in 1:length(inputs)
            # i = input_series
            # a = actions[agent_idx]
            set_parameters!(agent_model, Dict("learning_rate" => learning_rate[agent_idx]) )
            reset!(agent_model)

            for (timestep, input) in enumerate(inputs[agent_idx])
                action_distribution = agent_model.action_model(agent_model, input)
                actions[agent_idx][timestep] ~ action_distribution
            end

        end

    end

    full_model = do_full_model(agent_model, statmodel, statistical_data, inputs, actions, X)
    chains = sample(full_model, sampler, n_iterations)
    # rename parameters
    #replacement_names = Dict("agent_param[$id]" => "learning_rate[$id]") #
    replacement_names = Dict("agent_param[$idx]" => "learning_rate[$(Tuple(id))]" for (idx, id) in enumerate(eachrow(statistical_data[!,grouping_cols])))
    return replacenames(chains, replacement_names)

end
