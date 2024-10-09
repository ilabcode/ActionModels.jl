##########################################
### STATISTICAL MODEL FOR SINGLE AGENT ###
##########################################
@model function single_agent_population_model(
    prior::Dict{T,D},
) where {T<:Union{String,Tuple,Any},D<:Distribution}

    #Create container for sampled parameters
    parameters = Dict{T,Float64}()

    #Go through each of the parameters in the prior
    for (parameter, distribution) in prior
        #And sample a value for each agent
        parameters[parameter] ~ distribution
    end

    return PopulationModelReturn([parameters])
end

#######################################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A INPUT/OUTPUT SEQUENCE, AND SINGLE-AGENT PRIOR ###
#######################################################################################################################
function create_model(
    agent::Agent,
    prior::Dict{T,D},
    inputs::Array{T1},
    actions::Array{T2};
    kwargs...,
) where {
    T<:Union{String,Tuple,Any},
    D<:Distribution,
    T1<:Union{Real,Missing},
    T2<:Union{Real,Missing},
}
    #Create column names
    input_cols = map(x -> "input$x", 1:size(inputs, 2))
    action_cols = map(x -> "action$x", 1:size(actions, 2))

    #Create dataframe of the inputs and actions
    data = DataFrame(hcat(inputs, actions), vcat(input_cols, action_cols))

    #Add grouping column
    grouping_cols = "agent"
    data[!, grouping_cols] .= 1

    #Create the single-agent statistical model
    population_model = single_agent_population_model(prior)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        agent,
        population_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        kwargs...,
    )
end

################################################################################
####### FUNCTION FOR RENAMING CHAINS FOR A SINGLE-AGENT STATISTICAL MODEL ######
################################################################################
function rename_chains(
    chains::Chains,
    model::DynamicPPL.Model,
    #Arguments from statistical model
    prior::Dict{T,D},
) where {T<:Union{String,Tuple,Any},D<:Distribution}

    ## Make dict with replacement names ##
    replacement_names = Dict{String,String}()

    #Go through each parameter in the prior
    for (parameter_key, _) in prior

        #If the parameter name is a string
        if parameter_key isa String
            #Include quation marks in the name to be replaced
            parameter_key_left = "\"$(parameter_key)\""
        else
            #Otherwise, keep it as it is
            parameter_key_left = parameter_key
        end

        #If the parameter key is a tuple
        if parameter_key isa Tuple
            #Join the tuple with double underscores
            parameter_key_right = join(parameter_key, tuple_separator)
        else
            #Otherwise, keep it as it is
            parameter_key_right = parameter_key
        end

        #Set a replacement name
        replacement_names["parameters[$parameter_key_left]"] = "$parameter_key_right"
    end

    #Replace names in the fitted model and return it
    replacenames(chains, replacement_names)
end

############################################################
### CHECKS TO MAKE FOR THE SINGLE-AGENT STAISTICAL MODEL ###
############################################################
function check_population_model(
    #Arguments from statistical model
    prior::Dict{T,D};
    #Arguments from the agent
    verbose::Bool,
    agent::Agent,
) where {T<:Union{String,Tuple,Any},D<:Distribution}
    #Unless warnings are hidden
    if verbose
        #If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
        if any(key -> !(key in keys(prior)), keys(agent.parameters))
            @warn "the agent has parameters which are not estimated. The agent's current parameter values are used as fixed parameters"
        end
    end

    #If there are no parameters to sample
    if length(prior) == 0
        #Throw an error
        throw(ArgumentError("No parameters where specified in the prior."))
    end
end
