"""
plot_predictive_simulation(param_distributions::Union{Chains,Dict}, agent::Agent, inputs::Array, target_state::Union{String,Tuple};
    fixed_parameters::Dict = Dict(), n_simulations::Int = 100, verbose::Bool = true, median_color::Union{String,Symbol} = :red, title::String = "Sampled trajectories",
    label::Union{String,Tuple} = target_state, alpha::Real = 0.1, linewidth::Real = 2,
)

Simulate distributions of states and actions for an agent, with parameters sampled from a specified distirbutions, and given a series of inputs.

# Arguments
- 'param_distributions::Union{Chains,Dict}': Distributions to sample parameters from. Can be a dictionary containing keys and distributions, or a Turing Chain object containing the posterior distributions after fitting.
- 'agent::Agent': an ActionModels agent object created with either premade_agent or init_agent.
- 'inputs:Array': array of inputs. Each row is a timestep, and each column is a single input value.
- 'target_state::Union{String,Tuple}': the state for which to plot the simulated distribution. If set to 'action', plot the action distribution. Note that the target state must be in the agent's history. 
- 'fixed_parameters::Dict = Dict()': dictionary containing parameter values for parameters that are not fitted. Keys are parameter names, values are priors. For parameters not specified here and without priors, the parameter values of the agent are used instead.
- 'n_simulations::Int = 100': set number of simulations you want to run.
- 'verbose = true': set to false to hide warnings.
- 'median_color::Union{String,Symbol} = :red': specify color of median value in the plot.
- 'title::String = "Sampled trajectories"': title on graph.
- 'label::Union{String,Tuple} = target_state': label on graph.
- 'alpha::Real = 0.1': the transparency of each simulated trajectory line.
- 'linewidth::Real = 2': specify linewidth on your plot.
"""
function plot_predictive_simulation(
    param_distributions::Union{Chains,Dict},
    agent::Agent,
    inputs::Array,
    target_state::Union{String,Tuple};
    fixed_parameters::Dict = Dict(),
    n_simulations::Int = 100,
    verbose::Bool = true,
    median_color::Union{String,Symbol} = :red,
    title::String = "Sampled trajectories",
    label::Union{String,Tuple} = target_state,
    alpha::Real = 0.1,
    linewidth::Real = 2,
)

    ### Setup ###
    #Save old parameters for resetting the agent later
    old_parameters = ActionModels.get_parameters(agent)

    #Set the fixed parameters to the agent
    set_parameters!(agent, fixed_parameters)

    #If a Turing Chains of posteriors has been inputted
    if param_distributions isa Chains
        #Extract the postrior distributions as a dictionary
        param_distributions = get_posteriors(param_distributions, type = "distribution")
    end

    #Unless warnings are hidden
    if verbose
        #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
        if any(
            key -> !(key in keys(param_distributions)) && !(key in keys(fixed_parameters)),
            keys(old_parameters),
        )
            #Make a warning
            @warn "the agent has parameters which are not specified in the fixed or sampled parameters. The agent's current parameter values are used instead"
        end
    end

    ### Plot single simulations with sampled parameters ###
    #Initialize counter for number of simulations
    simulation_number = 1
    #Initialize counter for number of rejected samples
    n_rejected_samples = 0

    while simulation_number <= n_simulations

        #Try to run the simulation and plot it
        try
            #Create empty tuple for populating with sampled parameter values
            sampled_parameters = Dict()

            #For each specified parameter 
            for (param_key, param_distribution) in param_distributions
                #Add a sampled parameter value to the dict
                sampled_parameters[param_key] = rand(param_distribution)
            end

            #Set parameters
            set_parameters!(agent, sampled_parameters)
            reset!(agent)

            #Evolve agent
            give_inputs!(agent, inputs)

            #For the first simulation
            if simulation_number == 1
                #Initialize the trajectory plot
                plot_trajectory(
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
                plot_trajectory!(
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
            if e isa RejectParameters

                #Count the sample as rejected
                n_rejected_samples += 1

                #Advance the simulation counter
                simulation_number += 1

                #Skip the iteration
                continue
            else
                #Otherwise, throw the error
                throw(e)
            end
        end
    end

    #If all samples were rejected
    if n_rejected_samples == n_simulations
        #Warn
        @warn "all $n_simulations sampled parameters were rejected. No plot is produced"

        return nothing
    end

    #If some samples were rejected
    if n_rejected_samples > 0
        #Warn
        @warn "$n_rejected_samples out of $n_simulations sampled parameters were rejected"
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
    set_parameters!(agent, param_medians)
    reset!(agent)

    #Look for errors
    try
        #Evolve agent
        give_inputs!(agent, inputs)
        #If there is an error
    catch e
        #If it is a PaeramError
        if e isa RejectParameters
            throw(
                RejectParameters(
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
    plot = plot_trajectory!(
        agent,
        target_state;
        color = median_color,
        label = label,
        title = title,
        linewidth = linewidth,
    )

    #Reset agent to old settings
    set_parameters!(agent, old_parameters)
    reset!(agent)

    return plot
end
