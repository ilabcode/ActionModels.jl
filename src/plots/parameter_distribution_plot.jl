@userplot Parameter_Distribution_Plot

@recipe function f(
    pl::Parameter_Distribution_Plot;
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
    plot_height = 300,
)

    ### Setup ###
    #Get arguments
    chain = pl.args[1]
    param_priors = pl.args[2]

    #Get number of subplots
    n_subplots = length(param_priors)

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
    for (param_key, param_prior) in param_priors

        #Get posterior from the chain
        param_posterior = Array(chain[:, string(param_key), :])[:]

        ### Get uncertainty interval bar sizes ###
        #Get the quantiles for the prior and posterior
        prior_quantiles = Turing.Statistics.quantile(param_prior, interval_quantiles)
        posterior_quantiles =
            Turing.Statistics.quantile(param_posterior, interval_quantiles)

        #Get prior median and interval bounds
        prior_median = prior_quantiles[3]
        prior_inner_interval_lower = (prior_quantiles[3] - prior_quantiles[2])
        prior_inner_interval_upper = (prior_quantiles[4] - prior_quantiles[3])
        prior_outer_interval_lower = (prior_quantiles[3] - prior_quantiles[1])
        prior_outer_interval_upper = (prior_quantiles[5] - prior_quantiles[3])

        #Get posterior median and interval bounds
        posterior_median = posterior_quantiles[3]
        posterior_inner_interval_lower = (posterior_quantiles[3] - posterior_quantiles[2])
        posterior_inner_interval_upper = (posterior_quantiles[4] - posterior_quantiles[3])
        posterior_outer_interval_lower = (posterior_quantiles[3] - posterior_quantiles[1])
        posterior_outer_interval_upper = (posterior_quantiles[5] - posterior_quantiles[3])


        ### General plotting settings ###
        #Advance the plot number track one step
        plot_number = plot_number += 1

        #If the user has specified a subplot title
        if param_key in keys(subplot_titles)
            #Use user-specified title
            title := subplot_titles[param_key]
        else
            #Otherwise use the parameter name as the subplot title
            title := join(param_key, " ")
        end

        #Remove the y axis
        yshowaxis := false
        yticks := false

        #Set the font size
        legendfontsize --> 15


        ### Plot prior and posterior distribution ###
        #if show distributions
        if show_distributions

            ## Prior distribution
            @series begin
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

                #Plot the distribution
                param_prior
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
                #Plot the posterior
                param_posterior
            end
        end

        ### Plot uncertainty intervals ###
        if show_intervals

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

        ### Plot medians ###
        ## Prior median
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
            [(prior_median, prior_interval_offset)]
        end

        ## Posterior
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
            [(posterior_median, posterior_interval_offset)]
        end
    end
end
