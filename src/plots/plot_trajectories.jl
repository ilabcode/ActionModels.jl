@userplot struct Plot_Trajectories{
    T<:Tuple{
        AxisArrays.AxisArray{
            Union{Missing,Float64},
            5,
            Array{Union{Missing,Float64},5},
            Tuple{
                AxisArrays.Axis{:agent,Vector{Symbol}},
                AxisArrays.Axis{:state,Vector{Symbol}},
                AxisArrays.Axis{:timestep,UnitRange{Int64}},
                AxisArrays.Axis{:sample,UnitRange{Int64}},
                AxisArrays.Axis{:chain,UnitRange{Int64}},
            },
        },
    },
}
    args::T
end

"""
"""
plot_trajectories
@recipe function f(
    plt::Plot_Trajectories,
    sample_color::Union{String,Symbol} = :gray,
    sample_alpha::Real = 0.1,
    sample_linewidth::Real = 0.5,
    summary_function::Function = median,
    summary_alpha::Real = 1,
    summary_color::Union{String,Symbol} = :red,
    summary_linewidth::Real = 1,
    plot_width::Int = 800,
    plot_height::Int = 600,
    subplot_titles = Dict(),
)

    #Extract trajectories
    trajectories = plt.args[1]

    #Extract dimensions
    agent_ids, state_keys, timesteps, samples, chains = trajectories.axes

    #Get number of subplots
    n_subplots = length(state_keys)

    #Specify how to arrange subplots, and their size
    layout := (n_subplots, 1)
    size := (plot_width, plot_height * n_subplots)

    #Initialize counter for plot number
    plot_number = 0

    #For each state to be plotted
    for state_key in state_keys

        ## Setup ##
        #Advance the plot number track one step
        plot_number = plot_number += 1

        #If the user has specified a subplot title
        if state_key in keys(subplot_titles)
            #Use user-specified title
            title := subplot_titles[state_key]
        else
            #Otherwise use the parameter name as the subplot title
            title := string(state_key)
        end

        #Set the xticks to be the timesteps
        xticks := (1:length(timesteps), 0:timesteps[end])
        xlabel := "Timestep"

        #Set the font size
        legendfontsize --> 15

        #Plot samples for each agent
        for agent_id in agent_ids
            #For each chain and sample
            for chain in chains
                for sample in samples

                    #Extract the values for the current agent and parameter across samples and chains
                    values = trajectories[agent_id, state_key, :, sample, chain]

                    #Create a plot of this sample
                    @series begin
                        seriestype := :line
                        subplot := plot_number
                        color := sample_color
                        alpha := sample_alpha
                        linewidth := sample_linewidth
                        label := nothing
                        return values
                    end
                end
            end
        end

        #Plot summarized value for each agent
        for agent_id in agent_ids

            #Get vector of point estimates
            summary_values = [
                summary_function(trajectories[agent_id, state_key, timestep+1, :, :])
                for timestep in timesteps
            ]

            #Plot the summary value
            @series begin
                seriestype := :line
                subplot := plot_number
                color := summary_color
                linewidth := summary_linewidth
                alpha := summary_alpha
                label := nothing
                return summary_values
            end
        end
    end
end
