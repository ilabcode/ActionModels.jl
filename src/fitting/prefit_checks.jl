


function prefit_checks(;
    agent,
    data,
    priors,
    independent_group_cols,
    multilevel_group_cols,
    input_cols,
    action_cols,
    fixed_parameters,
    old_parameters,
    n_cores,
    verbose = true,
)

    #Unless warnings are hidden
    if verbose
        #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
        if any(
            key -> !(key in keys(priors)) && !(key in keys(fixed_parameters)),
            keys(old_parameters),
        )
            @warn "the agent has parameters which are not specified in the fixed or sampled parameters. The agent's current parameter values are used as fixed parameters"
        end

        #If a parameter has been specified both in the fixed and sampled parameters
        if any(key -> key in keys(fixed_parameters), keys(priors))
            @warn "one or more parameters have been specified both in the fixed and sampled parameters. The fixed parameter value is used"

            #Remove the parameter from the sampled parameters
            for key in keys(fixed_parameters)
                if key in keys(priors)
                    delete!(priors, key)
                end
            end
        end
    end


    #If there are no parameters to sample
    if length(priors) == 0
        #Throw an error
        throw(
            ArgumentError(
                "no parameters to sample. Either an empty dictionary of parameter priors was passed, or all parameters with priors were also specified as fixed parameters",
            ),
        )
    end


    #Check that user-specified columns exist in the dataset
    if any(multilevel_group_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified group columns that do not exist in the dataframe",
            ),
        )
    elseif any(input_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified input columns that do not exist in the dataframe",
            ),
        )
    elseif any(action_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified action columns that do not exist in the dataframe",
            ),
        )
    end


    #if a non-positive-integer amount of cores has been specified
    if n_cores < 1
        #Error
        throw(
            ArgumentError(
                "n_cores was set to a non-positive integer. This is not supported.",
            ),
        )
    end

    # ### Run forward once as test ###
    # #Initialize dictionary for populating with median parameter values
    # sampled_parameters = Dict()
    # #Go through each of the agent's parameters
    # for (param_key, param_prior) in priors
    #     #Add the median value to the tuple
    #     sampled_parameters[param_key] = median(param_prior)
    # end
    # #Set sampled parameters
    # set_parameters!(agent, sampled_parameters)
    # #Reset the agent
    # reset!(agent)

    # try
    #     #Run it forwards
    #     test_actions = give_inputs!(agent, inputs)

    #     #If the model returns a different amount of actions from what was inputted
    #     if size(test_actions) != size(actions)
    #         throw(
    #             ArgumentError(
    #                 "the passed actions is a different shape from what the model returns",
    #             ),
    #         )
    #     end

    # catch e
    #     #If a RejectParameters error occurs
    #     if e isa RejectParameters
    #         #Warn the user that prior median parameter values gives a sample rejection
    #         if verbose
    #             @warn "simulating with median parameter values from the prior results in a rejected sample."
    #         end
    #     else
    #         #Otherwise throw the actual error
    #         throw(e)
    #     end
    # end
end
