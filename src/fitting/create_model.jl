###########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A STATISTICAL MODEL ###
###########################################################################################################
function create_model(
    agent::Agent,
    statistical_model::DynamicPPL.Model,
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    tracked_states::Union{Vector{PRM},Nothing} = nothing,
) where {
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
    PRM<:Union{String,Tuple},
}

    #Create a copy of the agent to avoid changing the original 
    agent_model = deepcopy(agent)

    # Convert column names to symbols
    input_cols = Symbol.(input_cols)
    action_cols = Symbol.(action_cols)
    grouping_cols = Symbol.(grouping_cols)

    ## Extract data ##
    #One matrix per agent, for inputs and actions separately
    inputs = Vector{Array{Real}}()
    actions = Vector{Array{Real}}()
    for agent_data in groupby(data, grouping_cols)
        push!(inputs, Array{Real}(agent_data[:, input_cols]))
        push!(actions, Array{Real}(agent_data[:, action_cols]))
    end

    #Create a full model combining the agent model and the statistical model
    return full_model(agent_model, statistical_model, inputs, actions, tracked_states)
end

############################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A SINGLE-AGENT PRIOR ###
############################################################################################################
function create_model(
    agent::Agent,
    prior::Dict{T,D},
    data::DataFrame;
    input_cols::Union{Vector{T1},T1},
    action_cols::Union{Vector{T2},T3},
    grouping_cols::Union{Vector{T3},T3},
    tracked_states::Union{Vector{PRM},Nothing} = nothing,
) where {
    T<:Union{String,Tuple,Any},
    D<:Distribution,
    T1<:Union{String,Symbol},
    T2<:Union{String,Symbol},
    T3<:Union{String,Symbol},
    PRM<:Union{String,Tuple},
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
        tracked_states = tracked_states,
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
    tracked_states::Union{Vector{PRM},Nothing} = nothing,
) where {
    T<:Union{String,Tuple,Any},
    D<:Distribution,
    T1<:Real,
    T2<:Real,
    PRM<:Union{String,Tuple},
}

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
        tracked_states = tracked_states,
    )
end
