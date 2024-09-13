##########################################
### STATISTICAL MODEL FOR SINGLE AGENT ###
##########################################
@model function single_statistical_model(
    prior::Dict{T,D},
) where {T<:Union{String,Tuple,Any},D<:Distribution}

    #Create container for sampled parameters
    parameters = Dict{T,Float64}()

    #Go through each of the parameters in the prior
    for (parameter, distribution) in prior
        #And sample a value for each agent
        parameters[parameter] ~ distribution
    end

    return StatisticalModelReturn([parameters])
end

#######################################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A INPUT/OUTPUT SEQUENCE, AND SINGLE-AGENT PRIOR ###
#######################################################################################################################
function create_model(
    agent::Agent,
    prior::Dict{T,D},
    inputs::Array{T1},
    actions::Array{T2};
    track_states::Bool = false,
) where {
    T<:Union{String,Tuple,Any},
    D<:Distribution,
    T1<:Union{Real,Missing},
    T2<:Union{Real,Missing},
}

    #Create column names
    input_cols = map(x -> "input$x", 1:size(inputs, 2))
    action_cols = map(x -> "action$x", 1:size(actions, 2))
    grouping_cols = Vector{String}()

    #Create dataframe of the inputs and actions
    data = DataFrame(hcat(inputs, actions), vcat(input_cols, action_cols))

    #Create the single-agent statistical model
    statistical_model = single_statistical_model(prior)

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