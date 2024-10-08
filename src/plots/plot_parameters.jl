@userplot struct Plot_Parameters{T<:Tuple{Chains,Chains}}
    args::T
end
"""
    plot_parameter_distribution(fitted_model, param_priors;
        subplot_titles = Dict(),
        show_distributions = true,
        show_intervals = true,
        prior_color = :green,
        posterior_color = :orange,
        prior_interval_offset = 0,
        posterior_interval_offset = 0.01,
        inner_interval = 0.5,
        outer_interval = 0.8,
        plot_width = 900,
        plot_height = 300)

Plot the prior and posterior distributions of the parameters of a fitted model.

# Arguments
 - 'subplot_titles': A dictionary of parameter names and their corresponding plot titles.
 - 'show_distributions': Whether to show full distributions.
 - 'show_intervals': Whether to show uncertainty intervals.
 - 'prior_color': Color of the prior distribution.
 - 'posterior_color': Color of the posterior distribution.
 - 'prior_interval_offset': Offset of the prior interval bars.
 - 'posterior_interval_offset': Offset of the posterior interval bars.
 - 'inner_interval': Size of the inner uncertainty interval.
 - 'outer_interval': Size of the outer uncertainty interval.
 - 'plot_width': Width of the plot.
 - 'plot_height': Height of the plot.
"""
plot_parameters
@recipe function f(
    plt::Plot_Parameters;
    summary_function = median,
    show_distributions = true,
    show_intervals = true,
    inner_interval = 0.5,
    outer_interval = 0.8,
    prior_color = :green,
    posterior_color = :orange,
    plot_width = 900,
    plot_height = 300,
    prior_interval_offset = 0,
    posterior_interval_offset = 0.01,
    subplot_titles = Dict(),
)

    #Extract prior and posterior chains
    prior_chains, posterior_chains = plt.args

    #Extract parmaeter keys
    parameter_keys = describe(prior_chains)[1].nt.parameters

    #Get number of subplots
    n_subplots = length(parameter_keys)

    #Specify how to arrange subplots, and their size
    layout := (n_subplots, 1)
    size := (plot_width, plot_height * n_subplots)

    #Initialize counter for plot number
    plot_number = 0

    #Set quantiles that corresponds to specified uncertainty intervals
    interval_quantiles = [
        0.5 - outer_interval * 0.5,
        0.5 - inner_interval * 0.5,
        0.5,
        0.5 + inner_interval * 0.5,
        0.5 + outer_interval * 0.5,
    ]

    #For each parameter
    for parameter_key in parameter_keys

        ## Setup ##
        #Advance the plot number track one step
        plot_number = plot_number += 1

        #If the user has specified a subplot title
        if parameter_key in keys(subplot_titles)
            #Use user-specified title
            title := subplot_titles[parameter_key]
        else
            #Otherwise use the parameter name as the subplot title
            title := string(parameter_key)
        end

        #Remove the y axis
        yshowaxis := false
        yticks := false

        #Set the font size
        legendfontsize --> 15

        ## Get the prior and posterior samples ##
        #Get the prior and posterior samples
        prior = Array(prior_chains[:, string(parameter_key), :])[:]
        posterior = Array(posterior_chains[:, string(parameter_key), :])[:]

        ### Plot prior and posterior distribution ###
        #if show distributions
        if show_distributions

            ## Prior distribution
            @series begin
                #Set to be a density plot
                seriestype := :density

                #Set color
                color := prior_color
                #Set transparency
                fill := (0, 0.5)
                #Set subplot nr
                subplot := plot_number
                #Only put legend on first plot
                if plot_number != 1
                    legend := nothing
                end
                #Remove labels
                label := nothing
                #Set to trim the distribution
                trim := true

                #Plot the distribution
                prior
            end

            ## Posterior distribution
            @series begin
                #This is a density (not a functional form like the prior)
                seriestype := :density

                #Set color
                color := posterior_color
                #Set transparency
                fill := (0, 0.5)
                #Set subplot number
                subplot := plot_number
                #Only show legend on the first plot
                if plot_number != 1
                    legend := nothing
                end
                #Remove label
                label := nothing
                #Set to trim the distribution
                trim := true

                #Plot the posterior
                posterior
            end
        end


        ### Plot uncertainty intervals ###
        if show_intervals

            ### Get uncertainty interval bar sizes ###
            #Get quantiles 
            prior_quantiles = Turing.Statistics.quantile(prior, interval_quantiles)
            posterior_quantiles = Turing.Statistics.quantile(posterior, interval_quantiles)

            #Get prior median and interval bounds
            prior_median = prior_quantiles[3]
            prior_inner_interval_lower = (prior_quantiles[3] - prior_quantiles[2])
            prior_inner_interval_upper = (prior_quantiles[4] - prior_quantiles[3])
            prior_outer_interval_lower = (prior_quantiles[3] - prior_quantiles[1])
            prior_outer_interval_upper = (prior_quantiles[5] - prior_quantiles[3])

            #Get posterior median and interval bounds
            posterior_median = posterior_quantiles[3]
            posterior_inner_interval_lower =
                (posterior_quantiles[3] - posterior_quantiles[2])
            posterior_inner_interval_upper =
                (posterior_quantiles[4] - posterior_quantiles[3])
            posterior_outer_interval_lower =
                (posterior_quantiles[3] - posterior_quantiles[1])
            posterior_outer_interval_upper =
                (posterior_quantiles[5] - posterior_quantiles[3])

            ## Prior outer interval
            @series begin
                #A scatterplot errorbar will be used to show the interval
                seriestype := :scatter
                #Set color
                color := prior_color
                #Subplot number
                subplot := plot_number
                #Only show legend on first plot
                if plot_number != 1
                    legend := nothing
                end
                #Remove labels
                label := nothing

                #Set a low line thickness
                markerstrokewidth := 1
                #Set color of the bar
                markerstrokecolor := prior_color

                #Set the size of the errorbar
                xerror := ([prior_outer_interval_lower], [prior_outer_interval_upper])

                #Plot the point in order to get the errorbar
                [(prior_median, prior_interval_offset)]
            end

            ## Prior inner interval
            @series begin
                #A scatterplot errorbar will be used to show the interval
                seriestype := :scatter
                #Set color
                color := prior_color
                #Subplot number
                subplot := plot_number
                #Only show legend on first plot
                if plot_number != 1
                    legend := nothing
                end
                #Remove labels
                label := nothing

                #Set a higher line thickness
                markerstrokewidth := 3
                #Set color of the bar
                markerstrokecolor := prior_color

                #Set the size of the errorbar
                xerror := ([prior_inner_interval_lower], [prior_inner_interval_upper])

                #Plot the point in order to get the errorbar
                [(prior_median, prior_interval_offset)]
            end


            ## Posterior outer interval
            @series begin
                #A scatterplot errorbar will be used to show the interval
                seriestype := :scatter
                #Set color
                color := posterior_color
                #Subplot number
                subplot := plot_number
                #Only show legend on first plot
                if plot_number != 1
                    legend := nothing
                end
                #Remove labels
                label := nothing

                #Set a low line thickness
                markerstrokewidth := 1
                #Set color of the bar
                markerstrokecolor := posterior_color

                #Set the size of the errorbar
                xerror :=
                    ([posterior_outer_interval_lower], [posterior_outer_interval_upper])

                #Plot the point in order to get the errorbar
                [(posterior_median, posterior_interval_offset)]
            end

            ## Posterior inner interval
            @series begin
                #A scatterplot errorbar will be used to show the interval
                seriestype := :scatter
                #Set color
                color := posterior_color
                #Subplot number
                subplot := plot_number
                #Only show legend on first plot
                if plot_number != 1
                    legend := nothing
                end
                #Remove labels
                label := nothing

                #Set a low line thickness
                markerstrokewidth := 3
                #Set color of the bar
                markerstrokecolor := posterior_color

                #Set the size of the errorbar
                xerror :=
                    ([posterior_inner_interval_lower], [posterior_inner_interval_upper])

                #Plot the point in order to get the errorbar
                [(posterior_median, posterior_interval_offset)]
            end
        end

        ### Plot point estimates ###
        #Get point estimates
        prior_point_estimate = summary_function(prior)
        posterior_point_estimate = summary_function(posterior)

        #Prior
        @series begin
            #Scatterplot or a single point
            seriestype := :scatter
            #Set color
            color := prior_color
            #Subplot number
            subplot := plot_number
            #Only show legend for first plot
            if plot_number != 1
                legend := nothing
            end

            #Point size
            markerstrokewidth := 1
            markersize := 5

            #Set label to prior
            label := "Prior"

            #Plot the prior median
            [(prior_point_estimate, prior_interval_offset)]
        end


        @series begin
            #Scatterplot or a single point
            seriestype := :scatter
            #Set color
            color := posterior_color
            #Subplot number
            subplot := plot_number
            #Only show legend for first plot
            if plot_number != 1
                legend := nothing
            end

            #Point size
            markerstrokewidth := 1
            markersize := 5

            #Set label to prior
            label := "Posterior"

            #Plot the prior median
            [(posterior_point_estimate, posterior_interval_offset)]
        end
    end
end
