
#######################################################################################################
### SIMPLE STATISTICAL MODEL WHERE AGENTS ARE INDEPENDENT AND THEIR PARAMETERS HAVE THE SAME PRIORS ###
#######################################################################################################
@model function simple_statistical_model(
    prior::Dict{T,D},
    n_agents::I,
    agent_parameters::Vector{Dict{Any,Real}} = [Dict{Any,Real}() for _ = 1:n_agents],
) where {T<:Union{String,Tuple,Any},D<:Distribution,I<:Int}

    #Create container for sampled parameters
    parameters = Dict{Any,Vector{Real}}()

    #Go through each of the parameters in the prior
    for (parameter, distribution) in prior
        #And sample a value for each agent
        parameters[parameter] ~ filldist(distribution, n_agents)
    end

    #Go through each parameter
    for (parameter, values) in parameters
        #Go through each agent
        for (agent_idx, value) in enumerate(values)
            #Store the value in the right way
            agent_parameters[agent_idx][parameter] = value
        end
    end

    return StatisticalModelReturn(agent_parameters)
end



############################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A SINGLE-AGENT PRIOR ###
############################################################################################################
function create_model(
    agent::Agent,
    prior::Dict{T,D},
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T2},
    grouping_cols::Union{Vector{T3},T3} = Vector{String}(),
    track_states::Bool = false,
) where {
    T<:Union{String,Tuple,Any},
    D<:Distribution,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
}

    #Get number of agents in the dataframe
    n_agents = length(groupby(data, grouping_cols))

    #Create a statistical model where the agents are independent and sampled from the same prior
    statistical_model = simple_statistical_model(prior, n_agents)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        agent,
        statistical_model,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        track_states = track_states,
    )
end

#######################################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A INPUT/OUTPUT SEQUENCE, AND SINGLE-AGENT PRIOR ###
#######################################################################################################################
function create_model(
    agent::Agent,
    prior::Dict{T,D},
    inputs::Array{T1},
    actions::Array{T2},
    track_states::Bool = false,
) where {T<:Union{String,Tuple,Any},D<:Distribution,T1<:Union{Real, Missing},T2<:Union{Real, Missing}}

    #Create column names
    input_cols = map(x -> "input$x", 1:size(inputs, 2))
    action_cols = map(x -> "action$x", 1:size(actions, 2))
    grouping_cols = Vector{Nothing}()

    #Create dataframe of the inputs and actions
    data = DataFrame(hcat(inputs, actions), vcat(input_cols, action_cols))

    #Create a full model combining the agent model and the statistical model
    return create_model(
        agent,
        prior,
        data;
        input_cols = input_cols,
        action_cols = action_cols,
        grouping_cols = grouping_cols,
        track_states = track_states,
    )
end



##########################################################################
####### FUNCTION FOR RENAMING CHAINS FOR A SIMPLE STATISTICAL MODEL ######
##########################################################################
function rename_chains(
    fitted_model::Chains,
    prior::Dict{T,D},
    data::DataFrame;
    grouping_cols::Union{Vector{C},C},
) where {T<:Union{String,Tuple,Any},D<:Distribution,C<:Union{String,Symbol}}

    ## Make dict with index to agent mapping ##
    idx_to_agent = Dict{Int,Any}()

    #Go through each unique agent ID
    for (agent_idx, agent_key) in
        enumerate(eachrow(unique(data, grouping_cols)[:, grouping_cols]))

        #If there is only one grouping column
        if length(grouping_cols) == 1
            #Set the agent index to the agent ID
            idx_to_agent[agent_idx] = agent_key[1]
        else
            #Set the agent index to the tuple of agent IDs
            idx_to_agent[agent_idx] = Tuple(agent_key)
        end
    end

    ## Make dict with replacement names ##
    replacement_names = Dict{String,String}()

    for (agent_idx, agent_key) in idx_to_agent

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
                parameter_key_right = join(parameter_key, "__")
            else
                #Otherwise, keep it as it is
                parameter_key_right = parameter_key
            end

            #Set a replacement name
            replacement_names["parameters[$parameter_key_left][$agent_idx]"] = "$agent_key $parameter_key_right"
        end
    end

    #Replace names in the fitted model and return it
    fitted_model = replacenames(fitted_model, replacement_names)
end
