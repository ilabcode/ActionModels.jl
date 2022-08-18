"""
"""
function predictive_simulation_plot(
    param_distributions::Union{Chains,Dict},
    agent::AgentStruct,
    inputs::Vector,
    target_state::Union{String,Tuple};
    fixed_params::Dict = Dict(),
    n_simulations::Int = 1000,
    hide_warnings::Bool = false,
    median_color::Union{String,Symbol} = :red,
    title::String = "Sampled trajectories",
    label::Union{String,Tuple} = target_state,
    alpha::Real = 0.1,
    linewidth::Real = 2,
)

    ### Setup ###
    #Save old params for resetting the agent later
    old_params = get_params(agent)

    #Set the fixed parameters to the agent
    set_params!(agent, fixed_params)

    #If a Turing Chains of posteriors has been inputted
    if param_distributions isa Chains
        #Extract the postrior distributions as a dictionary
        param_distributions = get_posteriors(param_distributions, type = "distribution")
    end

    #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
    if any(
        key -> !(key in keys(param_distributions)) && !(key in keys(fixed_params)),
        keys(get_params(agent)),
    )
        #Unless warnings are hidden
        if !hide_warnings
            #Make a warning
            @warn "the agent has parameters which are not specified in the fixed or sampled parameters. For these parameters, the agent's current parameter values are used"
        end
    end

    ### Plot single simulations with sampled parameters ###
    #Initialize counter for number of simulations
    simulation_number = 1

    while simulation_number <= n_simulations

        #Try to run the simulation and plot it
        try
            #Create empty tuple for populating with sampled parameter values
            sampled_params = Dict()

            #For each specified parameter 
            for (param_key, param_distribution) in param_distributions
                #Add a sampled parameter value to the dict
                sampled_params[param_key] = rand(param_distribution)
            end

            #Set parameters
            set_params!(agent, sampled_params)
            reset!(agent)

            #Evolve agent
            give_inputs!(agent, inputs)

            #For the first simulation
            if simulation_number == 1
                #Initialize the trajectory plot
                trajectory_plot(
                    agent,
                    target_state;
                    color = :gray,
                    alpha = alpha,
                    label = "",
                    title = title,
                )
                #For other simulations
            else
                #Add trajectories to the same plot
                trajectory_plot!(
                    agent,
                    target_state;
                    color = :gray,
                    alpha = alpha,
                    label = "",
                    title = title,
                )
            end

            #Advance the simulation counter
            simulation_number += 1

            #If there is an error
        catch e
            #If the error is a user-specified Parameter Error
            if e isa ParamError
                if !hide_warnings
                    #Warn the user
                    @warn "A set of sampled parameters was rejected. If this occurs too often, try different parameter distributions"
                end
                #Skip the iteration
                continue
            else
                #Otherwise, throw the error
                throw(e)
            end
        end
    end

    ### Plot simulation with parameter medians ###
    #Create empty list for parameter medians
    param_medians = Dict()

    #For each specified parameter 
    for (param_key, param_distribution) in param_distributions
        #Add a sampled parameter value to the dict
        param_medians[param_key] = median(param_distribution)
    end

    #Set parameters
    set_params!(agent, param_medians)
    reset!(agent)

    #Look for errors
    try
        #Evolve agent
        give_inputs!(agent, inputs)
        #If there is an error
    catch e
        #If it is a PaeramError
        if e isa ParamError
            throw(
                ParamError(
                    "Evolving the agent with the medians of the parameter distributions resulted in numerical errors. Try different parameter distributions",
                ),
            )
        else
            throw(e)
        end
    end

    #If the label is a composite state
    if label isa Tuple
        #Join it into a string
        label = join(label, " ")
    end

    #Plot the median
    plot = trajectory_plot!(
        agent,
        target_state;
        color = median_color,
        label = label,
        title = title,
        linewidth = linewidth,
    )

    #Reset agent to old settings
    set_params!(agent, old_params)
    reset!(agent)

    return plot
end